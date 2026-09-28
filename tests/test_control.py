"""Check one run's control contract, one command at a time.

These tests use the real control state machine, command decisions, pause
accounting, worker thread, guard, and episodes. They also use a real ``Budget``
built from the example profile and a real ``MemoryJournal``. A
``ScriptedSeat`` replaces the live session, scripted moves replace the
operator, and a ``FakeClock`` advances only when a test moves it. These tests
do not cover browser handoffs.
"""

from __future__ import annotations

import dataclasses
import io
import threading
import time
from collections.abc import Callable

import pytest
from fakes import (
    EVERYTHING,
    FakeClock,
    Move,
    Screen,
    ScriptedSeat,
    ScriptedSurface,
    doing,
    when,
)
from import_contract import assert_module_import_contract

import computeruse.control
import computeruse.manual
from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    ObservationMode,
    ObservationRequest,
    Outcome,
    PageInfo,
    PageState,
)
from computeruse.budget import Budget
from computeruse.control import (
    OVERTAKEN,
    Button,
    Control,
    ControlChanged,
    Dispatch,
    Order,
    PageProbe,
    Receipt,
    Recording,
    SessionProbe,
    Status,
    explain,
    owner_of,
    settled,
)
from computeruse.escalation import (
    Ask,
    CheckFailure,
    Command,
    HandoffOutcome,
    Interruption,
    InterventionRequest,
    Mode,
    Owner,
    State,
    Trigger,
    Verdict,
)
from computeruse.journal import (
    Commanded,
    HandedBack,
    JsonlJournal,
    ManualAction,
    MemoryJournal,
    Paused,
)
from computeruse.manual import Detail, GapKind, ManualKind
from computeruse.profile import ActionKind, Profile
from computeruse.surface import SurfaceError

MEMBER = "https://sandbox.example.test/members/12345"
SEND = Action(ActionKind.CLICK, AxLocator("button", "Send"), effect="submit_payment")
READ = Action(ActionKind.READ, AxLocator("cell", "Balance"))
CHECK = Action(ActionKind.ASSERT, AxLocator("cell", "Balance"), "4200.00")
AWAIT = Action(ActionKind.WAIT_FOR, AxLocator("cell", "Balance"))
TYPE = Action(ActionKind.TYPE, AxLocator("textbox", "Amount"), "12")
LOOK = ObservationRequest(ObservationMode.STRUCTURED)

PERSON_STOPPED = (
    "a person used the session while automation held it; take control or resume"
)
UNPROTECTED = (
    "An edited or secret field left the page or could not be protected. "
    "The run cannot establish where its value went. Reload or leave the "
    "page before resuming. Resume alone cannot restore protection."
)

PRIMARY = {
    State.READY: Button("Start", Command.START, enabled=True),
    State.RUNNING: Button("Stop", Command.STOP, enabled=True),
    State.STOPPING: Button("Stopping...", None, enabled=False),
    State.PAUSED: Button("Resume", Command.RESUME, enabled=True),
    State.HUMAN_CONTROL: Button("Resume", Command.RESUME, enabled=True),
    State.AWAITING_APPROVAL: Button("Resume", Command.RESUME, enabled=True),
    State.CHECKING: Button("Checking...", None, enabled=False),
    State.COMPLETED: Button("Completed", None, enabled=False),
    State.FAILED: Button("Failed", None, enabled=False),
    State.TERMINATED: Button("Terminated", None, enabled=False),
}
"""The one button a channel draws, for each state."""

TAKE = frozenset({Command.TAKE_CONTROL})

OFFERED = {
    State.READY: frozenset({Command.START, Command.TERMINATE}),
    State.RUNNING: frozenset({Command.STOP, Command.TERMINATE}) | TAKE,
    State.STOPPING: frozenset({Command.TERMINATE}) | TAKE,
    State.PAUSED: frozenset({Command.RESUME, Command.TERMINATE}) | TAKE,
    State.AWAITING_APPROVAL: (
        frozenset({Command.APPROVE, Command.RESUME, Command.REJECT, Command.TERMINATE})
        | TAKE
    ),
    State.HUMAN_CONTROL: frozenset({Command.RESUME, Command.TERMINATE}),
    State.CHECKING: frozenset({Command.TERMINATE}),
    State.COMPLETED: frozenset(),
    State.FAILED: frozenset(),
    State.TERMINATED: frozenset(),
}
"""The commands each state offers when a person can see the session.

A stop that is still settling offers Take control only when it will pause.
"""

OWNERS = {
    State.READY: Owner.OPERATOR,
    State.RUNNING: Owner.AUTOMATION,
    State.STOPPING: Owner.AUTOMATION,
    State.PAUSED: Owner.OPERATOR,
    State.AWAITING_APPROVAL: Owner.OPERATOR,
    State.HUMAN_CONTROL: Owner.HUMAN,
    State.CHECKING: Owner.AUTOMATION,
    State.COMPLETED: Owner.NONE,
    State.FAILED: Owner.NONE,
    State.TERMINATED: Owner.NONE,
}


class Window:
    """A control window. It keeps every status it is drawn with."""

    def __init__(self, *, listening: bool = True) -> None:
        self.shown: list[Status] = []
        self.control: Control | None = None
        self._listening = listening

    def attach(self, control: Control) -> None:
        self.control = control

    def show(self, status: Status) -> None:
        self.shown.append(status)

    def listening(self) -> bool:
        return self._listening


@dataclasses.dataclass
class Rig:
    """One control with its seat, clock, budget, journal, and window."""

    profile: Profile
    control: Control
    seat: ScriptedSeat
    clock: FakeClock
    budget: Budget
    journal: MemoryJournal
    window: Window

    def begin(self) -> bool:
        return self.control.begin(
            profile=self.profile,
            budget=self.budget,
            journal=self.journal,
            context="pay member 10001",
        )

    def events[E](self, kind: type[E]) -> list[E]:
        return [event for event in self.journal.events if isinstance(event, kind)]

    def states(self) -> list[State]:
        return [status.state for status in self.window.shown]


def rig(
    profile: Profile,
    *moves: Move,
    visible: bool = True,
    worker: bool = False,
    wait_for_start: bool = False,
    listening: bool | None = True,
    start: bool = True,
) -> Rig:
    """Build a control over a scripted seat, and start it unless told not to.

    ``listening`` None means the control has no channel at all.
    """
    clock = FakeClock()
    seat = ScriptedSeat(clock, moves=list(moves), visible=visible, location=MEMBER)
    window = Window(listening=bool(listening))
    control = Control(
        mode=Mode.DISCOVERY,
        clock=clock,
        seat=seat,
        channels=[window] if listening is not None else [],
        worker=worker,
        wait_for_start=wait_for_start,
    )
    seat.control = control
    built = Rig(
        profile,
        control,
        seat,
        clock,
        Budget(profile.budgets, clock),
        MemoryJournal(),
        window,
    )
    if start:
        assert built.begin()
    return built


def order(
    control: Control,
    command: Command,
    *,
    status: Status | None = None,
    run: str | None = None,
    revision: int | None = None,
    intervention: str | None = None,
    note: str | None = None,
) -> Receipt:
    """Send ``command`` as an operator looking at ``status`` would."""
    seen = status or control.status()
    named = seen.offer.intervention if seen.offer is not None else ""
    return control.submit(
        Order(
            command,
            seen.run if run is None else run,
            seen.revision if revision is None else revision,
            named if intervention is None else intervention,
            note=note,
        )
    )


def sending(
    state: State, receipts: list[Receipt], *orders: tuple[Command, dict]
) -> Move:
    """Send each order once the run reaches ``state``, all from one status."""

    def move(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not state:
            return False
        assert seat.control is not None
        for command, changes in orders:
            receipts.append(order(seat.control, command, status=status, **changes))
        return True

    return move


def noting(state: State, notes: list[Status]) -> Move:
    """Read the status once the run reaches ``state``, as a person would."""

    def move(_seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not state:
            return False
        notes.append(status)
        return True

    return move


def asking(
    ask: Ask = Ask.APPROVAL,
    trigger: Trigger = Trigger.RISKY_ACTION,
    timeout_s: float = 900.0,
) -> InterventionRequest:
    """Build a request like the loop's, for a click or for a person's help."""
    return InterventionRequest(
        trigger=trigger,
        goal="pay member 10001",
        profile_id="acme-cu/vendor-core-9.2/sandbox",
        step=1,
        route="/members/:id",
        reason="click is declared risky",
        timeout_s=timeout_s,
        action=SEND if ask is Ask.APPROVAL else None,
        ask=ask,
    )


def stopped(built: Rig) -> None:
    """Press Stop and let the owner thread reach its boundary."""
    assert order(built.control, Command.STOP).verdict is Verdict.ACCEPTED
    assert built.control.halted()


def refused(attempt: Callable[[], object]) -> bool:
    try:
        attempt()
    except ControlChanged:
        return True
    return False


def attempts(guarded) -> dict[str, bool]:
    """Try each automation kind through the guard and report refusals."""
    return {
        "observe": refused(lambda: guarded.observe(LOOK)),
        "read": refused(lambda: guarded.act(READ)),
        "click": refused(lambda: guarded.act(SEND)),
    }


@pytest.fixture
def brief(edited_profile) -> Profile:
    """The example profile with a one-minute execution budget."""
    return edited_profile(
        budgets={
            "max_steps": 40,
            "max_wall_clock_s": 60,
            "max_retries_per_step": 2,
            "max_navigations": 15,
        }
    )


# What a channel draws.


def test_the_primary_button_offers_and_owner_follow_the_state(profile) -> None:
    built = rig(
        profile,
        when(State.READY, Command.START),
        when(State.RUNNING, Command.STOP),
        when(State.PAUSED, Command.TAKE_CONTROL),
        when(State.HUMAN_CONTROL, Command.RESUME),
        when(State.AWAITING_APPROVAL, Command.REJECT),
        wait_for_start=True,
        start=False,
    )
    assert built.begin()
    assert built.control.halted()
    assert built.control.hold().outcome is HandoffOutcome.RESUMED
    assert built.control.intervene(asking()).outcome is HandoffOutcome.REJECTED
    built.control.finish("completed")

    assert set(built.states()) == set(State) - {State.FAILED, State.TERMINATED}
    for status in built.window.shown:
        assert status.primary == PRIMARY[status.state], status.state
        assert status.offered == OFFERED[status.state], status.state
        assert status.owner is OWNERS[status.state], status.state
        assert status.run == built.control.run


@pytest.mark.parametrize(
    ("ending", "state"),
    [
        ("completed", State.COMPLETED),
        ("terminated", State.TERMINATED),
        ("failed", State.FAILED),
        ("exhausted", State.FAILED),
        ("handed_off", State.FAILED),
        ("blocked", State.FAILED),
    ],
)
def test_an_ended_run_shows_its_state_on_a_disabled_button(
    profile, ending: str, state: State
) -> None:
    built = rig(profile)
    built.control.finish(ending)
    status = built.control.status()

    assert status.state is state
    assert status.primary == PRIMARY[state]
    assert status.offered == frozenset()
    assert status.owner is Owner.NONE
    assert status.ending == ending


def test_a_takeover_that_is_still_settling_offers_only_terminate(profile) -> None:
    built = rig(profile)
    order(built.control, Command.STOP)
    assert built.control.status().offered == OFFERED[State.STOPPING]

    order(built.control, Command.TAKE_CONTROL)

    assert built.control.status().state is State.STOPPING
    assert built.control.status().offered == frozenset({Command.TERMINATE})


def test_take_control_is_never_offered_when_nobody_can_see_the_session(
    profile,
) -> None:
    built = rig(
        profile,
        when(State.READY, Command.START),
        when(State.RUNNING, Command.STOP),
        when(State.PAUSED, Command.RESUME),
        when(State.AWAITING_APPROVAL, Command.REJECT),
        visible=False,
        wait_for_start=True,
        start=False,
    )
    assert built.begin()
    assert built.control.halted()
    assert built.control.hold().outcome is HandoffOutcome.RESUMED
    assert built.control.intervene(asking()).outcome is HandoffOutcome.REJECTED

    assert {State.RUNNING, State.PAUSED, State.AWAITING_APPROVAL} <= set(built.states())
    for status in built.window.shown:
        assert status.offered == OFFERED[status.state] - TAKE, status.state
    refusal = order(built.control, Command.TAKE_CONTROL)
    assert refusal.verdict is Verdict.INVALID
    assert refusal.reason == "nobody can see this session, so it cannot be handed over"
    assert Owner.HUMAN not in built.seat.transfers


@pytest.mark.parametrize("state", list(State))
def test_each_state_names_who_may_operate_the_session(state: State) -> None:
    assert owner_of(state) is OWNERS[state]


# Binding a command to the run and the screen it was sent from.


def test_an_order_for_another_run_is_stale(profile) -> None:
    built = rig(profile)

    receipt = order(built.control, Command.STOP, run="run-elsewhere")

    assert receipt.verdict is Verdict.STALE
    assert receipt.reason == "that command was for another run"
    assert built.control.status().state is State.RUNNING


def test_a_double_click_is_applied_once(profile) -> None:
    built = rig(profile)
    seen = built.control.status()

    first = order(built.control, Command.STOP, status=seen)
    second = order(built.control, Command.STOP, status=seen)

    assert first.verdict is Verdict.ACCEPTED
    assert second.verdict is Verdict.DUPLICATE
    assert second.reason == "that command was already applied"
    assert built.control.status().revision == seen.revision + 1


@pytest.mark.rule(14)
def test_an_order_sent_from_an_older_status_is_stale(profile) -> None:
    built = rig(profile)
    old = built.control.status()
    order(built.control, Command.STOP)

    late = order(built.control, Command.TERMINATE, status=old)
    ahead = order(
        built.control,
        Command.TERMINATE,
        revision=built.control.status().revision + 3,
    )

    assert late.verdict is Verdict.STALE
    assert late.reason == "the run changed since that command was sent"
    assert ahead.verdict is Verdict.INVALID
    assert ahead.reason == "that command names a state the run never had"
    assert built.control.status().state is State.STOPPING


@pytest.mark.parametrize("command", [Command.APPROVE, Command.REJECT, Command.RESUME])
@pytest.mark.parametrize("named", ["", "iv-9"])
def test_an_answer_must_name_the_open_intervention(
    profile, command: Command, named: str
) -> None:
    receipts: list[Receipt] = []
    built = rig(
        profile,
        sending(State.AWAITING_APPROVAL, receipts, (command, {"intervention": named})),
        when(State.AWAITING_APPROVAL, Command.REJECT),
    )

    handoff = built.control.intervene(asking())

    assert receipts[0].verdict is Verdict.STALE
    assert receipts[0].reason == (
        f"that answer was for {named or 'no request'}; the open request is iv-1"
    )
    assert handoff.outcome is HandoffOutcome.REJECTED


def test_an_answer_when_no_request_is_open_is_stale(profile) -> None:
    built = rig(profile)

    receipt = order(built.control, Command.APPROVE, intervention="iv-1")

    assert receipt.verdict is Verdict.STALE
    assert receipt.reason == "no request is open"


def test_an_answer_to_an_earlier_request_is_stale_while_a_newer_one_is_open(
    profile,
) -> None:
    receipts: list[Receipt] = []
    built = rig(
        profile,
        when(State.AWAITING_APPROVAL, Command.REJECT),
        sending(
            State.AWAITING_APPROVAL,
            receipts,
            (Command.APPROVE, {"intervention": "iv-1"}),
        ),
        when(State.AWAITING_APPROVAL, Command.REJECT),
    )

    first = built.control.intervene(asking())
    second = built.control.intervene(asking())

    assert (first.intervention, second.intervention) == ("iv-1", "iv-2")
    assert receipts[0].verdict is Verdict.STALE
    assert receipts[0].reason == "that answer was for iv-1; the open request is iv-2"
    assert second.outcome is HandoffOutcome.REJECTED


@pytest.mark.rule(14)
def test_resume_while_an_approval_waits_declines_it_and_never_approves(
    profile,
) -> None:
    receipts: list[Receipt] = []
    built = rig(
        profile,
        sending(
            State.AWAITING_APPROVAL, receipts, (Command.RESUME, {"note": "not yet"})
        ),
    )

    handoff = built.control.intervene(asking())

    assert receipts[0].verdict is Verdict.ACCEPTED
    # The run continues with the note. The proposal was not approved.
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert handoff.outcome is not HandoffOutcome.APPROVED
    assert handoff.operator_note == "not yet"


def test_a_request_for_a_person_cannot_be_approved(profile) -> None:
    receipts: list[Receipt] = []
    built = rig(
        profile,
        sending(State.PAUSED, receipts, (Command.APPROVE, {})),
        when(State.PAUSED, Command.RESUME),
    )

    handoff = built.control.intervene(asking(Ask.PERSON, Trigger.MISSING_USER_INPUT))

    assert receipts[0].verdict is Verdict.INVALID
    assert receipts[0].reason == (
        "this request cannot be approved; take control and do the step, "
        "then resume, or terminate"
    )
    assert handoff.outcome is HandoffOutcome.RESUMED


def test_a_second_terminate_is_a_duplicate(profile) -> None:
    built = rig(profile)

    first = order(built.control, Command.TERMINATE)
    second = order(built.control, Command.TERMINATE)

    assert first.verdict is Verdict.ACCEPTED
    assert second.verdict is Verdict.DUPLICATE
    assert second.reason == "the run is already terminated"


def test_every_order_is_journaled_whatever_became_of_it_and_no_note_is(
    profile,
) -> None:
    note = "member 10001 asked for 4,200.00"
    built = rig(profile)
    seen = built.control.status()
    sent = [
        (Command.STOP, order(built.control, Command.STOP, run="x", note=note)),
        (Command.START, order(built.control, Command.START, note=note)),
        (Command.STOP, order(built.control, Command.STOP, status=seen, note=note)),
        (Command.STOP, order(built.control, Command.STOP, status=seen, note=note)),
        (Command.RESUME, order(built.control, Command.RESUME, status=seen, note=note)),
        (Command.TERMINATE, order(built.control, Command.TERMINATE, note=note)),
        (Command.TERMINATE, order(built.control, Command.TERMINATE, note=note)),
    ]
    built.control.finish("terminated")
    stream = io.StringIO()
    written = JsonlJournal(stream)
    for event in built.journal.events:
        written.record(event)

    assert [(event.command, event.verdict) for event in built.events(Commanded)] == [
        (command, receipt.verdict) for command, receipt in sent
    ]
    assert {receipt.verdict for _, receipt in sent} == set(Verdict)
    assert "note" not in {field.name for field in dataclasses.fields(Commanded)}
    assert "10001" not in stream.getvalue()
    assert "4,200.00" not in stream.getvalue()


def test_a_request_id_is_journaled_only_if_the_control_issued_it(profile) -> None:
    """An operator can type a member number where the request id belongs."""
    built = rig(profile)
    refused = [
        order(built.control, Command.APPROVE, intervention="10001"),
        order(built.control, Command.RESUME, intervention="Jane"),
        order(built.control, Command.REJECT, intervention="iv-99"),
    ]
    built.control.finish("failed")

    assert {receipt.verdict for receipt in refused} == {Verdict.STALE}
    assert [event.intervention for event in built.events(Commanded)] == ["", "", ""]


def test_a_second_resume_after_a_failed_check_is_accepted(profile) -> None:
    receipts: list[Receipt] = []
    notes: list[Status] = []
    results = [(CheckFailure.STEP_CONDITION, "the balance is not shown yet"), None]
    built = rig(
        profile,
        when(State.PAUSED, Command.TAKE_CONTROL),
        sending(State.HUMAN_CONTROL, receipts, (Command.RESUME, {})),
        noting(State.HUMAN_CONTROL, notes),
        sending(State.HUMAN_CONTROL, receipts, (Command.RESUME, {})),
    )
    stopped(built)

    handoff = built.control.hold(lambda: results.pop(0))

    assert [receipt.verdict for receipt in receipts] == [Verdict.ACCEPTED] * 2
    assert receipts[1].status.revision > receipts[0].status.revision
    assert [event.intervention for event in built.events(Commanded)][-2:] == [
        "iv-1",
        "iv-1",
    ]
    assert [(event.passed, event.failure) for event in built.events(HandedBack)] == [
        (False, CheckFailure.STEP_CONDITION),
        (True, None),
    ]
    assert notes[0].notice == "the balance is not shown yet"
    assert notes[0].owner is Owner.HUMAN
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert handoff.intervention == "iv-1"


# The execution clock stops while a person holds the run.


def test_the_remaining_time_holds_still_while_paused(profile) -> None:
    remaining: list[float] = []
    paused: list[float] = []

    def waiting(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.PAUSED:
            return False
        assert seat.control is not None
        seat.clock.advance(50)
        now = seat.control.status()
        remaining.append(now.remaining_s)
        paused.append(now.paused_s)
        if len(remaining) < 4:
            return False
        seat.send(Command.RESUME, now)
        return True

    built = rig(profile, waiting)
    built.clock.advance(20)
    before = built.control.status().remaining_s
    stopped(built)

    assert built.control.hold().outcome is HandoffOutcome.RESUMED

    assert before == profile.budgets.max_wall_clock_s - 20
    assert remaining == [before] * 4
    assert paused == [50, 100, 150, 200]
    assert built.budget.remaining_seconds() == before


def test_a_pause_longer_than_the_whole_budget_leaves_it_unexhausted(brief) -> None:
    limit = brief.budgets.max_wall_clock_s
    built = rig(
        brief,
        doing(State.PAUSED, lambda seat: seat.clock.advance(3 * limit)),
        when(State.PAUSED, Command.RESUME),
    )
    built.clock.advance(10)
    stopped(built)

    assert built.control.hold().outcome is HandoffOutcome.RESUMED

    assert built.budget.exhausted() is None
    assert built.budget.remaining_seconds() == limit - 10
    assert built.budget.paused_seconds() == 3 * limit


def test_nested_pauses_count_once(profile) -> None:
    built = rig(profile)

    def inner(seat: ScriptedSeat) -> None:
        with built.budget.waiting_for_a_human():
            seat.clock.advance(40)

    built.seat.moves = [doing(State.PAUSED, inner), when(State.PAUSED, Command.RESUME)]
    stopped(built)

    with built.budget.waiting_for_a_human():
        built.clock.advance(25)
        assert built.control.hold().outcome is HandoffOutcome.RESUMED
        built.clock.advance(5)

    assert built.budget.paused_seconds() == 70
    assert built.budget.remaining_seconds() == profile.budgets.max_wall_clock_s


def test_repeated_pauses_add_up(profile) -> None:
    built = rig(
        profile,
        doing(State.PAUSED, lambda seat: seat.clock.advance(10)),
        when(State.PAUSED, Command.RESUME),
        doing(State.PAUSED, lambda seat: seat.clock.advance(20)),
        when(State.PAUSED, Command.RESUME),
    )
    for working in (7, 3):
        built.clock.advance(working)
        stopped(built)
        assert built.control.hold().outcome is HandoffOutcome.RESUMED

    assert built.budget.paused_seconds() == 30
    assert built.budget.remaining_seconds() == profile.budgets.max_wall_clock_s - 10


# The handoff timeout.


def test_the_handoff_timeout_runs_from_the_start_of_the_pause(profile) -> None:
    timeout = profile.escalation.handoff_timeout_s
    third = timeout / 3
    left: list[float] = []
    results = [(CheckFailure.STEP_CONDITION, "not yet")]

    def waiting(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.HUMAN_CONTROL:
            return False
        assert status.offer is not None
        left.append(status.offer.left_s)
        seat.clock.advance(third)
        return True

    built = rig(
        profile,
        doing(State.PAUSED, lambda seat: seat.clock.advance(third)),
        when(State.PAUSED, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, lambda seat: seat.clock.advance(third)),
        when(State.HUMAN_CONTROL, Command.RESUME),
        waiting,
    )
    built.clock.advance(15)
    stopped(built)
    began = built.clock()

    handoff = built.control.hold(lambda: results.pop(0) if results else None)

    assert handoff.outcome is HandoffOutcome.TIMED_OUT
    assert built.clock() - began == timeout
    assert left == [timeout - 2 * third]
    assert [(event.passed, event.failure) for event in built.events(HandedBack)] == [
        (False, CheckFailure.STEP_CONDITION)
    ]


@pytest.mark.parametrize("listening", [False, None])
def test_an_intervention_nobody_can_answer_times_out_at_once(
    profile, listening: bool | None
) -> None:
    built = rig(profile, listening=listening)
    built.seat.wait_s = 1.0

    handoff = built.control.intervene(asking())

    assert handoff.outcome is HandoffOutcome.TIMED_OUT
    assert built.clock() < 5
    assert [event.ask for event in built.events(Paused)] == [Ask.APPROVAL]


# Model calls.


def eventually(condition: Callable[[], bool]) -> bool:
    """Wait up to five seconds of real time for ``condition``."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


def test_a_stop_during_a_slow_model_call_is_applied_at_once(profile) -> None:
    built = rig(profile, worker=True)
    started, release = threading.Event(), threading.Event()
    returned: list[str] = []
    results: list[object] = []

    def decide() -> str:
        started.set()
        release.wait(10)
        returned.append("click Send")
        return "click Send"

    owner = threading.Thread(target=lambda: results.append(built.control.call(decide)))
    owner.start()
    try:
        assert started.wait(5)
        ticked = built.seat.slices
        assert eventually(lambda: built.seat.slices > ticked + 3)

        receipt = order(built.control, Command.STOP)

        assert receipt.verdict is Verdict.ACCEPTED
        assert receipt.status.state in {State.STOPPING, State.PAUSED}
        owner.join(5)
        assert not owner.is_alive()
        assert results == [OVERTAKEN]
        assert returned == []
    finally:
        release.set()
    assert eventually(lambda: returned == ["click Send"])
    assert built.control.status().state is State.STOPPING
    assert results == [OVERTAKEN]


def test_an_error_from_a_call_that_was_not_discarded_is_raised(profile) -> None:
    built = rig(profile, worker=True)

    def decide() -> str:
        raise ValueError("the provider refused the request")

    with pytest.raises(ValueError, match="the provider refused"):
        built.control.call(decide)


def test_without_a_worker_the_call_runs_inline(profile) -> None:
    built = rig(profile, worker=False)
    ticked = built.seat.slices

    result = built.control.call(threading.current_thread)

    assert result is threading.current_thread()
    assert built.seat.slices == ticked


# The guard.


def test_the_guard_refuses_automation_before_the_start_during_a_stop_and_after(
    profile,
) -> None:
    surface = ScriptedSurface([Screen(MEMBER)])
    built = rig(profile, start=False)
    guarded = built.control.guard(surface)
    seen = {State.READY: attempts(guarded)}
    assert built.begin()
    seen[State.RUNNING] = attempts(guarded)
    order(built.control, Command.STOP)
    seen[State.STOPPING] = attempts(guarded)
    order(built.control, Command.TERMINATE)
    seen[State.TERMINATED] = attempts(guarded)
    done = rig(profile)
    done.control.finish("completed")
    seen[State.COMPLETED] = attempts(done.control.guard(surface))

    everything = {"observe": True, "read": True, "click": True}
    assert seen == {
        State.READY: everything,
        State.RUNNING: {"observe": False, "read": False, "click": False},
        State.STOPPING: everything,
        State.TERMINATED: everything,
        State.COMPLETED: everything,
    }
    assert len(surface.looks) == 1
    assert surface.acted == [READ, SEND]


def test_the_guard_refuses_automation_while_the_run_is_held(profile) -> None:
    surface = ScriptedSurface([Screen(MEMBER)])
    built = rig(profile)
    guarded = built.control.guard(surface)
    tried: dict[State, dict[str, bool]] = {}

    def trying(state: State) -> Move:
        def move(_seat: ScriptedSeat, status: Status) -> bool:
            if status.state is not state:
                return False
            tried[state] = attempts(guarded)
            return True

        return move

    built.seat.moves = [
        trying(State.PAUSED),
        when(State.PAUSED, Command.TAKE_CONTROL),
        trying(State.HUMAN_CONTROL),
        when(State.HUMAN_CONTROL, Command.RESUME),
        trying(State.AWAITING_APPROVAL),
        when(State.AWAITING_APPROVAL, Command.REJECT),
    ]
    stopped(built)
    built.control.hold()
    built.control.intervene(asking())

    everything = {"observe": True, "read": True, "click": True}
    assert tried == {
        State.PAUSED: everything,
        State.HUMAN_CONTROL: everything,
        State.AWAITING_APPROVAL: everything,
    }
    assert surface.looks == []
    assert surface.acted == []


def test_the_guard_allows_only_reads_while_checking(profile) -> None:
    surface = ScriptedSurface([Screen(MEMBER)])
    built = rig(profile, when(State.PAUSED, Command.RESUME))
    guarded = built.control.guard(surface)
    tried: dict[str, bool] = {}

    def checks() -> None:
        assert built.control.status().state is State.CHECKING
        tried.update(attempts(guarded))
        tried["assert"] = refused(lambda: guarded.act(CHECK))
        tried["wait_for"] = refused(lambda: guarded.act(AWAIT))
        tried["type"] = refused(lambda: guarded.act(TYPE))

    stopped(built)
    assert built.control.hold(checks).outcome is HandoffOutcome.RESUMED

    assert tried == {
        "observe": False,
        "read": False,
        "assert": False,
        "wait_for": False,
        "click": True,
        "type": True,
    }
    assert surface.acted == [READ, CHECK, AWAIT]


@dataclasses.dataclass
class Pressing:
    """A surface an operator sends ``command`` to while it is acting."""

    control: Control
    command: Command
    outcome: Outcome | None
    sent: list[Action] = dataclasses.field(default_factory=list)
    settling: list[str] = dataclasses.field(default_factory=list)

    def location(self) -> str:
        return MEMBER

    def pages(self) -> tuple[PageInfo, ...]:
        return ()

    def capabilities(self):
        return EVERYTHING

    def observe(self, request):
        raise AssertionError(f"nothing observes here: {request}")

    def act(self, action: Action, *, expect=None) -> ActionResult:
        del expect
        self.sent.append(action)
        assert order(self.control, self.command).verdict is Verdict.ACCEPTED
        self.settling.append(self.control.status().settling)
        if self.outcome is None:
            raise SurfaceError("the page went away during the click")
        return ActionResult(self.outcome, PageState(MEMBER))


@pytest.mark.parametrize(
    ("outcome", "interrupted"),
    [
        (Outcome.OK, Interruption.PERFORMED),
        (Outcome.STALE, Interruption.PENDING),
        (Outcome.BLOCKED, Interruption.PENDING),
        (Outcome.NOT_FOUND, Interruption.PENDING),
        (Outcome.UNCERTAIN, Interruption.UNCERTAIN),
        (Outcome.HANDOFF, Interruption.UNCERTAIN),
        (Outcome.SURFACE_ERROR, Interruption.UNCERTAIN),
        (None, Interruption.UNCERTAIN),
    ],
)
def test_a_stop_during_an_operation_reports_what_became_of_it(
    profile, outcome: Outcome | None, interrupted: Interruption
) -> None:
    built = rig(profile, when(State.PAUSED, Command.RESUME))
    for _ in range(3):
        built.budget.charge_step()
    surface = Pressing(built.control, Command.STOP, outcome)
    guarded = built.control.guard(surface)

    if outcome is None:
        with pytest.raises(SurfaceError):
            guarded.act(SEND)
    else:
        assert guarded.act(SEND).outcome is outcome
    assert built.control.halted()
    assert built.control.hold().outcome is HandoffOutcome.RESUMED

    back = built.control.hand_back
    assert back is not None
    dispatched = Dispatch(3, ActionKind.CLICK, outcome or Outcome.UNCERTAIN)
    assert surface.settling == ["click dispatched at step 3 is settling"]
    assert settled(outcome) is interrupted
    assert back.interrupted is interrupted
    assert back.dispatched == dispatched
    assert built.events(HandedBack)[-1].interrupted is interrupted
    paused = next(s for s in built.window.shown if s.state is State.PAUSED)
    assert paused.interrupted == explain(interrupted, dispatched)
    assert surface.sent == [SEND]


@pytest.mark.parametrize("command", [Command.STOP, Command.TERMINATE])
def test_permits_turns_false_as_soon_as_a_stop_or_terminate_is_accepted(
    profile, command: Command
) -> None:
    built = rig(profile)
    permitted = built.seat.permitted
    assert permitted is not None
    assert permitted()
    receipts: list[Receipt] = []

    operator = threading.Thread(
        target=lambda: receipts.append(order(built.control, command))
    )
    operator.start()
    operator.join(5)

    assert receipts[0].verdict is Verdict.ACCEPTED
    assert permitted() is False
    assert built.control.permits() is False


# A person using the session.


def test_person_input_while_running_stops_automation_at_the_next_boundary(
    profile,
) -> None:
    built = rig(profile, when(State.PAUSED, Command.RESUME))
    built.seat.person(ManualKind.CLICK, role="button", name="Send", owned=False)

    assert built.control.halted()

    status = built.control.status()
    assert status.state is State.STOPPING
    assert status.notice == PERSON_STOPPED
    assert PERSON_STOPPED in [shown.notice for shown in built.window.shown]
    assert built.control.hold().outcome is HandoffOutcome.RESUMED
    assert [event.owned for event in built.events(ManualAction)] == [False]


@pytest.mark.parametrize(
    ("kind", "role", "name", "detail"),
    [
        (ManualKind.CLICK, "heading", "Account summary", Detail.NONE),
        (ManualKind.CLICK, "row", "Payment 10", Detail.NONE),
        (ManualKind.CLICK, "cell", "10.00", Detail.NONE),
        (ManualKind.CLICK, "img", "", Detail.NONE),
        (ManualKind.CLICK, "", "", Detail.NONE),
        (ManualKind.KEY, "", "", Detail.COMMAND),
        (ManualKind.KEY, "textbox", "Amount", Detail.UNCERTAIN),
        (ManualKind.EDIT, "textbox", "Amount", Detail.UNCERTAIN),
        (ManualKind.SCROLL, "", "", Detail.DOWN),
        (ManualKind.KEY, "", "", Detail.MOVE),
    ],
    ids=[
        "heading",
        "row",
        "cell",
        "image",
        "generic",
        "command-key",
        "uncertain-key",
        "uncertain-edit",
        "scroll",
        "movement-key",
    ],
)
def test_input_on_any_control_or_of_uncertain_origin_stops_automation(
    profile, kind: ManualKind, role: str, name: str, detail: Detail
) -> None:
    """No role or tag makes a click harmless, and no key is assumed inert.

    A page can handle a click on any element, from a listener on it or on an
    ancestor. A page can act on any key and any scroll without cancelling
    it. Input that the automation cannot account for may come from a person.
    """
    built = rig(profile)
    built.seat.person(kind, role=role, name=name, owned=False, detail=detail)

    assert built.control.halted()
    assert built.control.status().notice == PERSON_STOPPED


@pytest.mark.parametrize(
    ("kind", "role", "name", "detail"),
    [
        (ManualKind.NAVIGATION_REFUSED, "", "", Detail.NONE),
        (ManualKind.EDIT, "textbox", "Note", Detail.UNPROMPTED),
    ],
    ids=["refused-navigation", "page-script-edit"],
)
def test_input_that_cannot_change_the_screen_does_not_stop_automation(
    profile, kind: ManualKind, role: str, name: str, detail: Detail
) -> None:
    built = rig(profile)
    built.seat.person(kind, role=role, name=name, owned=False, detail=detail)

    assert not built.control.halted()
    assert built.control.status().state is State.RUNNING
    assert [event.kind for event in built.events(ManualAction)] == [kind]


def test_a_pause_a_person_caused_tells_the_operator_why(profile) -> None:
    built = rig(profile, when(State.PAUSED, Command.RESUME))
    built.seat.person(ManualKind.CLICK, role="button", name="Send", owned=False)
    assert built.control.halted()

    built.control.hold()

    paused = next(s for s in built.window.shown if s.state is State.PAUSED)
    assert paused.offer is not None
    assert "a person used the session" in f"{paused.offer.reason} {paused.notice}"


@pytest.mark.parametrize("command", [Command.APPROVE, Command.REJECT])
def test_an_answer_keeps_the_person_in_control_while_a_typed_field_is_unprotected(
    profile, command: Command
) -> None:
    """Resuming again is another check, never a confirmation that it is safe.

    The field stays unprotected through a second Resume, so that Resume fails
    the same way. Only once the seat can protect the field again does a
    Resume pass.
    """
    notes: list[Status] = []

    def reload(seat: ScriptedSeat) -> None:
        seat.unprotected = 0

    built = rig(
        profile,
        when(State.AWAITING_APPROVAL, command),
        noting(State.HUMAN_CONTROL, notes),
        when(State.HUMAN_CONTROL, Command.RESUME),
        noting(State.HUMAN_CONTROL, notes),
        doing(State.HUMAN_CONTROL, reload),
        when(State.HUMAN_CONTROL, Command.RESUME),
    )
    built.seat.unprotected = 1

    handoff = built.control.intervene(asking())

    states = built.states()
    answered = states[states.index(State.AWAITING_APPROVAL) :]
    assert answered[:3] == [
        State.AWAITING_APPROVAL,
        State.CHECKING,
        State.HUMAN_CONTROL,
    ]
    assert notes[0].owner is Owner.HUMAN
    assert notes[0].notice == UNPROTECTED
    assert notes[0].offered == OFFERED[State.HUMAN_CONTROL]
    assert notes[1].owner is Owner.HUMAN
    assert notes[1].notice == UNPROTECTED
    assert Owner.HUMAN in built.seat.transfers
    assert [(event.passed, event.failure) for event in built.events(HandedBack)] == [
        (False, CheckFailure.PROTECTION_LOST),
        (False, CheckFailure.PROTECTION_LOST),
        (True, None),
    ]
    assert handoff.outcome is HandoffOutcome.RESUMED


@pytest.mark.parametrize(
    ("kind", "role", "name", "detail", "changed"),
    [
        (ManualKind.NAVIGATION_REFUSED, "", "", Detail.NONE, False),
        (ManualKind.EDIT, "textbox", "Note", Detail.UNPROMPTED, False),
        (ManualKind.SCROLL, "", "", Detail.DOWN, True),
        (ManualKind.KEY, "", "", Detail.MOVE, True),
        (ManualKind.CLICK, "heading", "Account summary", Detail.NONE, True),
        (ManualKind.CLICK, "", "", Detail.NONE, True),
        (ManualKind.KEY, "", "", Detail.COMMAND, True),
        (ManualKind.CLICK, "button", "Send", Detail.NONE, True),
        (ManualKind.EDIT, "textbox", "Amount", Detail.NONE, True),
        (ManualKind.KEY, "", "", Detail.CONFIRM, True),
    ],
)
def test_only_input_that_can_change_the_screen_marks_an_approval_changed(
    profile,
    kind: ManualKind,
    role: str,
    name: str,
    detail: Detail,
    changed: bool,
) -> None:
    built = rig(profile)

    def looking(seat: ScriptedSeat) -> None:
        seat.person(kind, role=role, name=name, owned=False, detail=detail)

    built.seat.moves = [
        doing(State.AWAITING_APPROVAL, looking),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
    ]

    handoff = built.control.intervene(asking())

    assert handoff.outcome is HandoffOutcome.APPROVED
    assert handoff.changed is changed
    assert [step.event.kind for step in built.control.segments[-1].steps] == [kind]


@pytest.mark.rule(14, 16)
def test_input_at_the_last_transfer_tick_invalidates_and_records_the_approval(
    profile,
) -> None:
    slices = 0

    def late_input(seat: ScriptedSeat, status: Status) -> bool:
        nonlocal slices
        if status.state is not State.RUNNING:
            return False
        slices += 1
        if slices < 2:
            return False
        seat.person(ManualKind.CLICK, role="button", name="Send", owned=False)
        return True

    built = rig(profile, when(State.AWAITING_APPROVAL, Command.APPROVE), late_input)

    handoff = built.control.intervene(asking())

    assert handoff.outcome is HandoffOutcome.APPROVED
    assert handoff.changed
    assert built.control.halted()
    assert [step.event.kind for step in built.control.segments[-1].steps] == [
        ManualKind.CLICK
    ]


# Ending the run.


def test_finish_settles_the_control_once(profile) -> None:
    built = rig(profile)
    built.control.finish("completed")
    final = built.control.status()
    events = len(built.journal.events)
    shown = len(built.window.shown)

    built.control.finish("failed")

    assert built.control.status() == final
    assert final.state is State.COMPLETED
    assert len(built.journal.events) == events
    assert len(built.window.shown) == shown


def test_finish_drains_a_session_a_person_still_holds(profile) -> None:
    timeout = profile.escalation.handoff_timeout_s
    built = rig(
        profile,
        doing(
            State.HUMAN_CONTROL,
            lambda seat: seat.person(ManualKind.CLICK, role="button", name="Send"),
        ),
        doing(State.HUMAN_CONTROL, lambda seat: seat.clock.advance(timeout)),
    )
    order(built.control, Command.TAKE_CONTROL)
    assert built.control.halted()
    assert built.control.hold().outcome is HandoffOutcome.TIMED_OUT
    assert built.control.status().state is State.HUMAN_CONTROL
    built.seat.person(ManualKind.EDIT, role="textbox", name="Amount")
    handed = built.seat.handed_back

    built.control.finish("handed_off")

    assert built.seat.handed_back == handed + 1
    assert built.control.status().state is State.FAILED
    segment = built.control.segments[-1]
    assert segment.taken
    assert [step.event.kind for step in segment.steps] == [
        ManualKind.CLICK,
        ManualKind.EDIT,
    ]
    assert built.seat.transfers[-1] is Owner.NONE
    held = built.budget.paused_seconds()
    built.clock.advance(10)
    assert built.budget.paused_seconds() == held


def test_finish_still_ends_the_run_when_the_session_closed(profile) -> None:
    class Closed(ScriptedSeat):
        def hand_back(self, *, revoke: bool = True) -> Recording:
            del revoke
            raise SurfaceError("the browser closed")

    clock = FakeClock()
    seat = Closed(clock, location=MEMBER)
    window = Window()
    control = Control(
        mode=Mode.DISCOVERY, clock=clock, seat=seat, channels=[window], worker=False
    )
    seat.control = control
    seat.moves = [when(State.PAUSED, Command.TERMINATE)]
    budget = Budget(profile.budgets, clock)
    assert control.begin(
        profile=profile, budget=budget, journal=MemoryJournal(), context="goal"
    )
    order(control, Command.STOP)
    assert control.halted()
    assert control.hold().outcome is HandoffOutcome.TERMINATED

    control.finish("terminated")

    assert control.status().state is State.TERMINATED
    assert [gap.kind for gap in control.segments[-1].gaps] == [GapKind.NOT_DRAINED]
    assert seat.transfers[-1] is Owner.NONE


# What the control may import.


@pytest.mark.parametrize("module", [computeruse.control, computeruse.manual])
def test_the_control_imports_only_the_standard_library_and_computeruse(
    module,
) -> None:
    assert_module_import_contract(module)


# Regressions from the branch review. Each test fails without its fix.


def seated(
    profile: Profile, seat: ScriptedSeat, *, wait_for_start: bool = False
) -> Rig:
    """Build a control over a seat a test gave some unusual behaviour."""
    window = Window()
    control = Control(
        mode=Mode.DISCOVERY,
        clock=seat.clock,
        seat=seat,
        channels=[window],
        worker=False,
        wait_for_start=wait_for_start,
    )
    seat.control = control
    return Rig(
        profile,
        control,
        seat,
        seat.clock,
        Budget(profile.budgets, seat.clock),
        MemoryJournal(),
        window,
    )


class Terminating(ScriptedSeat):
    """A seat whose operator presses Terminate while the Start check flushes it."""

    def hand_back(self, *, revoke: bool = True) -> Recording:
        assert self.control is not None
        order(self.control, Command.TERMINATE)
        return super().hand_back(revoke=revoke)


@pytest.mark.parametrize("unprotected", [0, 1])
def test_a_terminate_accepted_while_the_start_check_runs_stands(
    profile, unprotected: int
) -> None:
    seat = Terminating(
        FakeClock(),
        moves=[when(State.READY, Command.START)],
        location=MEMBER,
        limit=100,
    )
    seat.unprotected = unprotected
    built = seated(profile, seat, wait_for_start=True)

    began = built.begin()

    assert began is False
    assert built.control.status().state is State.TERMINATED
    assert built.control.permits() is False
    assert Owner.AUTOMATION not in seat.transfers
    assert [(event.command, event.verdict) for event in built.events(Commanded)] == [
        (Command.START, Verdict.ACCEPTED),
        (Command.TERMINATE, Verdict.ACCEPTED),
    ]


class Striking(FakeClock):
    """A clock that applies one operator command the next time it is read.

    A pause reads the clock under the control's lock just before it settles
    its state. A command applied there is one that another thread sent after
    the caller read the state and before the pause opened.
    """

    __slots__ = ("strike",)

    def __init__(self) -> None:
        super().__init__()
        self.strike: Callable[[], object] | None = None

    def __call__(self) -> float:
        strike, self.strike = self.strike, None
        if strike is not None:
            strike()
        return super().__call__()


def resuming(opened: list[State]) -> Move:
    """Record the state of the open pause, then resume it."""

    def move(seat: ScriptedSeat, status: Status) -> bool:
        if status.offer is None or status.state not in {
            State.PAUSED,
            State.HUMAN_CONTROL,
        }:
            return False
        opened.append(status.state)
        seat.send(Command.RESUME, status)
        return True

    return move


def striking(
    profile: Profile, command: Command
) -> tuple[Rig, list[Receipt], list[State]]:
    """Stop a run, and arrange for ``command`` to land as the pause opens.

    Returns the rig, the receipt of ``command`` once it is sent, and the
    state each pause the operator resumed was in.
    """
    opened: list[State] = []
    clock = Striking()
    seat = ScriptedSeat(clock, moves=[resuming(opened)], location=MEMBER, limit=1000)
    built = seated(profile, seat)
    assert built.begin()
    stopped(built)
    receipts: list[Receipt] = []
    clock.strike = lambda: receipts.append(order(built.control, command))
    return built, receipts, opened


def test_a_takeover_sent_after_the_caller_read_the_state_still_wins(profile) -> None:
    built, receipts, opened = striking(profile, Command.TAKE_CONTROL)

    handoff = built.control.hold()

    assert [receipt.verdict for receipt in receipts] == [Verdict.ACCEPTED]
    assert receipts[0].status.state is State.STOPPING
    assert opened == [State.HUMAN_CONTROL]
    assert Owner.HUMAN in built.seat.transfers
    assert built.control.segments[-1].taken
    assert handoff.outcome is HandoffOutcome.RESUMED


def test_a_terminate_sent_after_the_caller_read_the_state_still_ends_the_run(
    profile,
) -> None:
    built, receipts, opened = striking(profile, Command.TERMINATE)

    handoff = built.control.hold()

    assert [receipt.verdict for receipt in receipts] == [Verdict.ACCEPTED]
    assert opened == []
    assert handoff.outcome is HandoffOutcome.TERMINATED
    assert built.control.status().state is State.TERMINATED
    assert built.control.permits() is False


@pytest.mark.parametrize(
    ("ask", "trigger", "command"),
    [
        (Ask.APPROVAL, Trigger.RISKY_ACTION, Command.APPROVE),
        (Ask.APPROVAL, Trigger.RISKY_ACTION, Command.REJECT),
        (Ask.APPROVAL, Trigger.RISKY_ACTION, Command.TAKE_CONTROL),
        (Ask.PERSON, Trigger.MISSING_USER_INPUT, Command.RESUME),
    ],
)
def test_an_answer_sent_after_the_deadline_is_refused_before_the_owner_notices(
    profile, ask: Ask, trigger: Trigger, command: Command
) -> None:
    timeout = 60.0
    receipts: list[Receipt] = []

    def late(seat: ScriptedSeat, status: Status) -> bool:
        if status.offer is None:
            return False
        assert seat.control is not None
        seat.clock.advance(timeout)
        receipts.append(seat.send(command, seat.control.status()))
        return True

    built = rig(profile, late)

    handoff = built.control.intervene(asking(ask, trigger, timeout_s=timeout))

    [refusal] = receipts
    assert refusal.verdict is Verdict.INVALID
    assert refusal.reason == "that request timed out and the run is ending"
    assert refusal.status.state in {State.AWAITING_APPROVAL, State.PAUSED}
    assert handoff.outcome is HandoffOutcome.TIMED_OUT
    assert built.control.status().offered == frozenset({Command.TERMINATE})
    again = order(built.control, command)
    assert again.verdict is Verdict.INVALID
    assert "timed out" in again.reason
    assert order(built.control, Command.TERMINATE).verdict is Verdict.ACCEPTED


def test_a_person_who_keeps_using_the_session_after_pressing_resume_keeps_control(
    profile,
) -> None:
    receipts: list[Receipt] = []
    notes: list[Status] = []

    def still_working(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.CLICK, role="button", name="Send")

    built = rig(
        profile,
        when(State.PAUSED, Command.TAKE_CONTROL),
        sending(State.HUMAN_CONTROL, receipts, (Command.RESUME, {})),
        doing(State.CHECKING, still_working),
        noting(State.HUMAN_CONTROL, notes),
        sending(State.HUMAN_CONTROL, receipts, (Command.RESUME, {})),
    )
    stopped(built)

    handoff = built.control.hold()

    assert [(event.passed, event.failure) for event in built.events(HandedBack)] == [
        (False, CheckFailure.IN_USE),
        (True, None),
    ]
    assert notes[0].owner is Owner.HUMAN
    assert notes[0].notice == (
        "The session is still in use. Resume when you have finished."
    )
    assert [receipt.verdict for receipt in receipts] == [Verdict.ACCEPTED] * 2
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert [step.event.kind for step in built.control.segments[-1].steps] == [
        ManualKind.CLICK
    ]


def test_channels_are_redrawn_once_a_second_even_without_a_new_revision(
    profile,
) -> None:
    built = rig(profile)
    revision = built.control.status().revision
    drawn = len(built.window.shown)
    built.budget.charge_step()

    built.clock.advance(0.5)
    built.control.halted()
    within = len(built.window.shown)
    built.clock.advance(0.5)
    built.control.halted()

    assert within == drawn
    assert len(built.window.shown) == drawn + 1
    latest = built.window.shown[-1]
    assert latest.revision == revision
    assert latest.step == 1
    assert latest.remaining_s == profile.budgets.max_wall_clock_s - 1


def test_a_request_id_shaped_like_one_but_never_issued_is_journaled_empty(
    profile,
) -> None:
    receipts: list[Receipt] = []
    built = rig(
        profile,
        sending(
            State.AWAITING_APPROVAL,
            receipts,
            (Command.REJECT, {"intervention": "iv-99"}),
            (Command.APPROVE, {"intervention": "iv-2"}),
            (Command.REJECT, {}),
        ),
    )

    handoff = built.control.intervene(asking())
    built.control.finish("failed")

    assert handoff.outcome is HandoffOutcome.REJECTED
    assert [receipt.verdict for receipt in receipts] == [
        Verdict.STALE,
        Verdict.STALE,
        Verdict.ACCEPTED,
    ]
    assert receipts[0].reason == "that answer was for iv-99; the open request is iv-1"
    assert [
        (event.command, event.verdict, event.intervention)
        for event in built.events(Commanded)
    ] == [
        (Command.REJECT, Verdict.STALE, ""),
        (Command.APPROVE, Verdict.STALE, ""),
        (Command.REJECT, Verdict.ACCEPTED, "iv-1"),
    ]


def test_a_sign_in_prompt_that_appears_while_nobody_used_the_session_holds_it(
    profile,
) -> None:
    """A credential prompt that grew with no person input is the application asking.

    During the pause the session navigates and the page's own script fills a
    field. Neither action is a person's input, so Resume leaves the person in
    control with the sign-in notice. After the person clicks and types, they
    can judge the same prompt count, and the next Resume succeeds.
    """
    receipts: list[Receipt] = []
    notes: list[Status] = []
    prompting = SessionProbe(
        (PageProbe("page-1", MEMBER, dialog=False, ready=True),),
        credential_prompts=1,
    )

    def expired(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.NAVIGATION, role="", name="")
        seat.person(ManualKind.EDIT, role="textbox", name="", detail=Detail.UNPROMPTED)
        seat.probes = [prompting]

    def signs_in(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.CLICK, role="textbox", name="Password")
        seat.person(ManualKind.EDIT, role="textbox", name="", secret=True)

    built = rig(
        profile,
        doing(State.PAUSED, expired),
        sending(State.PAUSED, receipts, (Command.RESUME, {})),
        noting(State.HUMAN_CONTROL, notes),
        doing(State.HUMAN_CONTROL, signs_in),
        sending(State.HUMAN_CONTROL, receipts, (Command.RESUME, {})),
    )
    stopped(built)

    handoff = built.control.hold()

    assert [(event.passed, event.failure) for event in built.events(HandedBack)] == [
        (False, CheckFailure.SIGN_IN_REQUIRED),
        (True, None),
    ]
    assert notes[0].owner is Owner.HUMAN
    assert notes[0].notice == (
        "The application requires another sign-in. Sign in, then resume."
    )
    assert [receipt.verdict for receipt in receipts] == [Verdict.ACCEPTED] * 2
    assert handoff.outcome is HandoffOutcome.RESUMED
