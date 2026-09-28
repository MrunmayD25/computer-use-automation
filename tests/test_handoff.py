"""Check stop, handoff, and resume behavior through a live control.

These tests use the real loop, control, policy gate, budget, and journal. A
``ScriptedSurface`` replaces the browser, a ``ScriptedDecider`` replaces the
model, and a ``ScriptedSeat`` plays operator moves between the control's wait
slices. A ``FakeClock`` advances only when a test or move advances it. These
tests cover how the run handles a person's actions. They do not cover browser
handoffs.
"""

from __future__ import annotations

import dataclasses
import io
import threading
from collections.abc import Callable

import pytest
from fakes import (
    DONE,
    DONE_TARGET,
    EVERYTHING,
    FakeClock,
    Move,
    Screen,
    ScriptedDecider,
    ScriptedSeat,
    ScriptedSurface,
    doing,
    grants,
    when,
)

from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    Capabilities,
    Expectation,
    Observation,
    ObservationMode,
    ObservationProvenance,
    ObservationRequest,
    PageInfo,
    Point,
    ScreenTarget,
)
from computeruse.budget import Budget
from computeruse.control import Control, Order, Receipt, Status
from computeruse.decider import (
    AskHuman,
    Decider,
    Decision,
    Observe,
    Propose,
    Remember,
    Task,
    Transcript,
)
from computeruse.escalation import (
    Ask,
    CheckFailure,
    Command,
    HandoffOutcome,
    Interruption,
    Mode,
    Owner,
    State,
    Trigger,
    Verdict,
)
from computeruse.journal import (
    Acted,
    Commanded,
    Escalated,
    HandedBack,
    Journal,
    JsonlJournal,
    ManualAction,
    MemoryJournal,
    Observed,
    Paused,
    Superseded,
)
from computeruse.loop import Ending, RunResult, Verification, discover
from computeruse.manual import Detail, ManualKind, Requirement
from computeruse.profile import ActionKind, Profile
from computeruse.surface import Surface

MEMBER = "https://sandbox.example.test/members/12345"
SAVINGS = "https://sandbox.example.test/members/12345/savings/history"
WIRE = "https://sandbox.example.test/members/12345/savings/wire"
ELSEWHERE = "https://elsewhere.example.test/login"

GOAL = "pay member 10001 the amount shown"
CONTROLS = (("button", "Search"), ("button", "Send"), ("cell", "4200.00"))

SEARCH = AxLocator("button", "Search")
SEND = AxLocator("button", "Send")
BALANCE = AxLocator("cell", "4200.00")

LOOK = Observe(ObservationRequest(ObservationMode.STRUCTURED), "read the controls")
PICTURE = Observe(ObservationRequest(ObservationMode.VISUAL), "look at the screen")
FIND = Propose(Action(ActionKind.CLICK, SEARCH, effect="search"), "search")
PAY = Propose(Action(ActionKind.CLICK, SEND, effect="submit_payment"), "pay")
STUCK = AskHuman(Trigger.MISSING_USER_INPUT, "need the amount to send")
PERSON_STOPPED = (
    "a person used the session while automation held it; take control or resume"
)

EVERY_KIND = {
    "observe": "safe",
    "read": "safe",
    "wait_for": "safe",
    "assert": "safe",
    "navigate": "safe",
    "scroll": "safe",
    "click": "safe",
    "type": "safe",
    "select": "safe",
    "press_key": "safe",
    "dismiss_dialog": "safe",
    "accept_dialog": "risky",
}
"""Every action type the example profile grants, each at its declared risk."""


class Window:
    """A control window that is always open. It keeps every status it draws."""

    def __init__(self) -> None:
        self.shown: list[Status] = []

    def attach(self, control: Control) -> None:
        del control

    def show(self, status: Status) -> None:
        self.shown.append(status)

    def listening(self) -> bool:
        return True


class Watched(Control):
    """The real control, keeping the budget the run gave it for a test to read."""

    budget: Budget | None = None

    def begin(
        self,
        *,
        profile: Profile,
        budget: Budget,
        journal: Journal,
        context: str,
        restrictions=tuple,
    ) -> bool:
        self.budget = budget
        return super().begin(
            profile=profile,
            budget=budget,
            journal=journal,
            context=context,
            restrictions=restrictions,
        )


@dataclasses.dataclass
class Deciding:
    """A scripted model that takes ``seconds`` per decision.

    ``during`` maps a call number to something the operator does while the
    model is deciding that call.
    """

    script: ScriptedDecider
    clock: FakeClock
    seconds: float = 0.0
    during: dict[int, Callable[[], object]] = dataclasses.field(default_factory=dict)
    calls: int = 0

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        """Supply the task for this scripted control test."""
        del goal, notices
        return Task()

    def decide(self, transcript: Transcript) -> Decision:
        self.calls += 1
        self.clock.advance(self.seconds)
        decision = self.script.decide(transcript)
        press = self.during.get(self.calls)
        if press is not None:
            press()
        return decision


class Blocking:
    """A scripted model whose call ``block_at`` waits until a move releases it."""

    def __init__(self, script: ScriptedDecider, block_at: int) -> None:
        self.script = script
        self.block_at = block_at
        self.calls = 0
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        """Supply the task for this scripted control test."""
        del goal, notices
        return Task()

    def decide(self, transcript: Transcript) -> Decision:
        self.calls += 1
        decision = self.script.decide(transcript)
        if self.calls == self.block_at:
            self.blocked.set()
            self.release.wait(10)
            self.returned.set()
        return decision


@dataclasses.dataclass
class Pressed:
    """A scripted surface the operator sends ``command`` to during its first act.

    ``advance`` is how long that act takes on the fake clock.
    """

    inner: ScriptedSurface
    seat: ScriptedSeat
    command: Command
    advance: float = 0.0
    pressed: bool = False

    def location(self) -> str:
        return self.inner.location()

    def pages(self) -> tuple[PageInfo, ...]:
        return self.inner.pages()

    def capabilities(self) -> Capabilities:
        return self.inner.capabilities()

    def observe(self, request: ObservationRequest) -> Observation:
        return self.inner.observe(request)

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        if not self.pressed and action.target != DONE_TARGET:
            self.pressed = True
            control = self.seat.control
            assert control is not None
            assert self.seat.send(self.command, control.status()).verdict is (
                Verdict.ACCEPTED
            )
            self.seat.clock.advance(self.advance)
        return self.inner.act(action, expect=expect)


@dataclasses.dataclass
class Session:
    """One discovery run's collaborators, built before the run starts."""

    profile: Profile
    clock: FakeClock
    seat: ScriptedSeat
    window: Window
    control: Watched
    scripted: ScriptedSurface
    script: ScriptedDecider
    deciding: Deciding
    journal: Journal
    surface: Surface
    decider: Decider

    def discover(self, goal: str = GOAL) -> RunResult:
        return discover(
            goal,
            self.profile,
            surface=self.surface,
            decider=self.decider,
            journal=self.journal,
            clock=self.clock,
            control=self.control,
        )

    def events[E](self, kind: type[E]) -> list[E]:
        assert isinstance(self.journal, MemoryJournal)
        return [event for event in self.journal.events if isinstance(event, kind)]

    def notices(self, call: int) -> tuple[str, ...]:
        """Return the notices the model was given on its ``call``-th decision."""
        return self.script.seen[call - 1].notices

    def press(self, command: Command, **changes: str) -> Callable[[], Receipt]:
        """Return the operator sending ``command`` from the status now drawn."""
        return lambda: self.seat.send(command, self.control.status(), **changes)


def session(
    profile: Profile,
    decisions: list[Decision],
    *moves: Move,
    screens: list[Screen] | None = None,
    location: str = MEMBER,
    worker: bool = False,
    journal: Journal | None = None,
) -> Session:
    clock = FakeClock()
    seat = ScriptedSeat(clock, moves=list(moves), location=location)
    window = Window()
    control = Watched(
        mode=Mode.DISCOVERY,
        clock=clock,
        seat=seat,
        channels=[window],
        worker=worker,
    )
    seat.control = control
    scripted = ScriptedSurface(screens or [Screen(location, controls=CONTROLS)])
    script = ScriptedDecider(list(decisions))
    deciding = Deciding(script, clock)
    return Session(
        profile,
        clock,
        seat,
        window,
        control,
        scripted,
        script,
        deciding,
        journal if journal is not None else MemoryJournal(),
        scripted,
        deciding,
    )


def noting(state: State, notes: list[Status]) -> Move:
    """Read the status once the run reaches ``state``, as a person would."""

    def move(_seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not state:
            return False
        notes.append(status)
        return True

    return move


def advancing(seconds: float) -> Callable[[ScriptedSeat], None]:
    def wait(seat: ScriptedSeat) -> None:
        seat.clock.advance(seconds)

    return wait


def order_from(status: Status, command: Command) -> Order:
    """Build the order an operator looking at ``status`` would send."""
    offer = status.offer
    return Order(
        command,
        status.run,
        status.revision,
        offer.intervention if offer is not None else "",
    )


def clicking(name: str) -> Callable[[ScriptedSeat], None]:
    def click(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.CLICK, role="button", name=name)

    return click


# Stopping and resuming keeps the run what it was.


def test_stop_then_resume_keeps_the_run_its_steps_its_session_and_its_budget(
    profile,
) -> None:
    s = session(
        profile,
        [LOOK, FIND, DONE],
        doing(State.PAUSED, advancing(120)),
        when(State.PAUSED, Command.RESUME),
    )
    s.deciding.during[2] = s.press(Command.STOP)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert result.steps == 2
    assert {status.run for status in s.window.shown} == {s.control.run}
    assert [event.step for event in s.events(Superseded)] == [2]
    assert s.scripted.acted == []
    assert [seen.steps_remaining for seen in s.script.seen] == [39, 38, 38]
    assert [seen.seconds_remaining for seen in s.script.seen] == [300.0] * 3
    assert [look.provenance for look in s.scripted.looks] == [
        ObservationProvenance.REQUESTED,
        ObservationProvenance.POST_INTERVENTION,
    ]
    assert s.scripted.checked == [Action(ActionKind.READ, DONE_TARGET)]
    assert s.seat.transfers == [
        Owner.AUTOMATION,
        Owner.OPERATOR,
        Owner.AUTOMATION,
        Owner.NONE,
    ]


def test_a_person_waiting_longer_than_the_budget_does_not_consume_it(
    edited_profile,
) -> None:
    profile = edited_profile(
        escalation={"handoff_timeout_s": 3600, "on_timeout": "abort"}
    )
    limit = profile.budgets.max_wall_clock_s
    s = session(
        profile,
        [LOOK, STUCK, DONE],
        doing(State.PAUSED, advancing(3 * limit)),
        when(State.PAUSED, Command.RESUME),
    )

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.clock() == 3 * limit
    assert s.script.seen[-1].seconds_remaining == limit
    assert [event.outcome for event in s.events(Escalated)] == [HandoffOutcome.RESUMED]


def test_nested_pauses_are_counted_once(profile) -> None:
    s = session(profile, [LOOK, STUCK, DONE])
    s.deciding.seconds = 10

    def inner(seat: ScriptedSeat) -> None:
        assert s.control.budget is not None
        with s.control.budget.waiting_for_a_human():
            seat.clock.advance(100)

    s.seat.moves = [
        doing(State.PAUSED, inner),
        doing(State.PAUSED, advancing(50)),
        when(State.PAUSED, Command.RESUME),
    ]

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.control.budget is not None
    assert s.control.budget.paused_seconds() == 150
    assert [seen.seconds_remaining for seen in s.script.seen] == [300, 290, 280]


# Nothing the automation does reaches a session a person holds.


def test_automation_cannot_act_while_the_person_holds_the_session(profile) -> None:
    s = session(profile, [LOOK, FIND, DONE])
    counts: list[tuple[int, int, int]] = []

    def working(seat: ScriptedSeat) -> None:
        counts.append((len(s.scripted.acted), len(s.scripted.looks), s.deciding.calls))
        seat.person(ManualKind.CLICK, role="button", name="Search")

    s.seat.moves = [
        doing(State.HUMAN_CONTROL, working),
        doing(State.HUMAN_CONTROL, working),
        doing(State.HUMAN_CONTROL, working),
        when(State.HUMAN_CONTROL, Command.RESUME),
    ]
    s.deciding.during[2] = s.press(Command.TAKE_CONTROL)

    result = s.discover()

    assert counts == [(0, 1, 2)] * 3
    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == []
    assert Owner.HUMAN in s.seat.transfers
    assert len(result.segments[0].steps) == 3


def test_a_late_model_response_cannot_act_after_a_takeover(profile) -> None:
    s = session(profile, [LOOK, FIND, DONE], worker=True)
    blocking = Blocking(s.script, block_at=2)
    s.decider = blocking
    acted_when_released: list[int] = []

    def take_while_blocked(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.RUNNING or not blocking.blocked.is_set():
            return False
        return seat.send(Command.TAKE_CONTROL, status).verdict is Verdict.ACCEPTED

    def release(_seat: ScriptedSeat) -> None:
        blocking.release.set()
        assert blocking.returned.wait(5)
        acted_when_released.append(len(s.scripted.acted))

    s.seat.moves = [
        take_while_blocked,
        doing(State.HUMAN_CONTROL, release),
        when(State.HUMAN_CONTROL, Command.RESUME),
    ]

    try:
        result = s.discover()
    finally:
        blocking.release.set()
    for thread in threading.enumerate():
        if thread.name == "computeruse-decision":
            thread.join(5)

    assert result.ending is Ending.COMPLETED
    assert acted_when_released == [0]
    assert s.scripted.acted == []
    assert [event.step for event in s.events(Superseded)] == [2]
    assert blocking.calls == 3


# Commands bound to what the operator saw.


def test_duplicate_and_stale_resume_and_approve_are_refused(profile) -> None:
    receipts: list[Receipt] = []
    approving: list[Status] = []

    def approve_twice(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.AWAITING_APPROVAL:
            return False
        approving.append(status)
        receipts.append(seat.send(Command.APPROVE, status))
        receipts.append(seat.send(Command.APPROVE, status))
        return True

    def resume_twice(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.PAUSED:
            return False
        assert seat.control is not None
        earlier = dataclasses.replace(
            order_from(status, Command.APPROVE), intervention="iv-1"
        )
        receipts.append(seat.send(Command.RESUME, approving[0]))
        receipts.append(seat.control.submit(earlier))
        receipts.append(seat.send(Command.RESUME, status))
        receipts.append(seat.send(Command.RESUME, status))
        return True

    def resume_late(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.RUNNING or status.offer is not None:
            return False
        assert seat.control is not None
        receipts.append(
            seat.control.submit(
                dataclasses.replace(
                    order_from(status, Command.RESUME), intervention="iv-2"
                )
            )
        )
        return True

    s = session(profile, [LOOK, PAY, STUCK, DONE], approve_twice, resume_twice)
    s.seat.moves.append(resume_late)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == [PAY.action]
    assert [
        (event.command, event.verdict, event.intervention)
        for event in s.events(Commanded)
    ] == [
        (Command.APPROVE, Verdict.ACCEPTED, "iv-1"),
        (Command.APPROVE, Verdict.DUPLICATE, "iv-1"),
        (Command.RESUME, Verdict.STALE, "iv-1"),
        (Command.APPROVE, Verdict.STALE, "iv-1"),
        (Command.RESUME, Verdict.ACCEPTED, "iv-2"),
        (Command.RESUME, Verdict.DUPLICATE, "iv-2"),
        (Command.RESUME, Verdict.STALE, "iv-2"),
    ]
    assert receipts[3].reason == "that answer was for iv-1; the open request is iv-2"
    assert receipts[-1].reason == "no request is open"


def test_resume_does_not_count_as_approval(profile) -> None:
    s = session(
        profile,
        [LOOK, PAY, PAY, DONE],
        when(State.AWAITING_APPROVAL, Command.TAKE_CONTROL),
        when(State.HUMAN_CONTROL, Command.RESUME),
        when(State.AWAITING_APPROVAL, Command.REJECT),
    )

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == []
    assert [
        (event.trigger, event.outcome, event.intervention)
        for event in s.events(Escalated)
    ] == [
        (Trigger.RISKY_ACTION, HandoffOutcome.RESUMED, "iv-1"),
        (Trigger.RISKY_ACTION, HandoffOutcome.REJECTED, "iv-2"),
    ]
    assert [event.ask for event in s.events(Paused)] == [Ask.APPROVAL, Ask.APPROVAL]


def test_a_step_that_must_name_its_record_is_left_to_a_person(profile) -> None:
    receipts: list[Receipt] = []

    def approve(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.PAUSED:
            return False
        receipts.append(seat.send(Command.APPROVE, status))
        return True

    target = ScreenTarget("obs-1", Point(10, 10))
    s = session(
        profile,
        [
            PICTURE,
            Propose(Action(ActionKind.CLICK, target, effect="approve_member")),
            DONE,
        ],
        approve,
        when(State.PAUSED, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, clicking("Approve")),
        when(State.HUMAN_CONTROL, Command.RESUME),
        location=SAVINGS,
    )

    result = s.discover()

    offer = next(st.offer for st in s.window.shown if st.state is State.PAUSED)
    assert offer is not None
    assert offer.ask is Ask.PERSON
    assert offer.trigger is Trigger.RECORD_EVIDENCE_REQUIRED
    assert receipts[0].verdict is Verdict.INVALID
    assert receipts[0].reason.startswith("this request cannot be approved")
    assert s.scripted.acted == []
    assert [(e.trigger, e.outcome) for e in s.events(Escalated)] == [
        (Trigger.RECORD_EVIDENCE_REQUIRED, HandoffOutcome.RESUMED)
    ]
    assert (
        "a person had the session for a step that must name its record; look "
        "again to see what they did"
    ) in s.notices(3)
    assert [step.requirement for step in result.segments[0].steps] == [
        Requirement.PERSON_EACH_RUN
    ]
    assert result.ending is Ending.COMPLETED


def test_an_approval_in_one_run_is_asked_for_again_in_the_next(profile) -> None:
    runs = []
    for _ in range(2):
        s = session(
            profile,
            [LOOK, PAY, DONE],
            when(State.AWAITING_APPROVAL, Command.APPROVE),
        )
        runs.append((s, s.discover()))

    for s, result in runs:
        assert result.ending is Ending.COMPLETED
        assert s.scripted.acted == [PAY.action]
        assert [(e.trigger, e.outcome) for e in s.events(Escalated)] == [
            (Trigger.RISKY_ACTION, HandoffOutcome.APPROVED)
        ]
        assert [event.ask for event in s.events(Paused)] == [Ask.APPROVAL]
        assert s.script.seen[0].restrictions == ()
    assert runs[0][0].control.run != runs[1][0].control.run


def test_an_approval_given_after_a_person_changed_the_screen_is_void(profile) -> None:
    kept = Remember("balance", "4200.00", may_change=False, source=BALANCE)
    s = session(
        profile,
        [LOOK, kept, PAY, DONE],
        doing(State.AWAITING_APPROVAL, clicking("Search")),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
    )
    s.scripted.extracts = ["4200.00"]

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == [Action(ActionKind.READ, BALANCE)]
    assert [e.outcome for e in s.events(Escalated)] == [HandoffOutcome.APPROVED]
    before, after = s.script.seen[2].memory, s.script.seen[3].memory
    assert [(fact.key, fact.may_change) for fact in before] == [("balance", False)]
    assert [(fact.key, fact.may_change) for fact in after] == [("balance", True)]
    assert (
        "the session changed while the approval was pending; look again" in s.notices(4)
    )


# Every ending says what happened.


@pytest.mark.parametrize("waiting", [State.AWAITING_APPROVAL, State.PAUSED])
def test_a_handoff_that_times_out_ends_the_run_handed_off(
    profile, waiting: State
) -> None:
    timeout = profile.escalation.handoff_timeout_s
    s = session(profile, [LOOK, PAY, DONE], doing(waiting, advancing(timeout)))
    if waiting is State.PAUSED:
        s.deciding.during[2] = s.press(Command.STOP)

    result = s.discover()

    assert result.ending is Ending.HANDED_OFF
    assert result.detail == "handoff timed out"
    assert s.scripted.acted == []


def test_a_rejection_lets_the_run_continue(profile) -> None:
    s = session(
        profile,
        [LOOK, PAY, DONE],
        when(State.AWAITING_APPROVAL, Command.REJECT),
    )

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == []
    assert "the operator declined click" in s.notices(3)


def test_a_termination_reports_the_operation_the_pause_interrupted(profile) -> None:
    s = session(profile, [LOOK, FIND, DONE], when(State.PAUSED, Command.TERMINATE))
    s.surface = Pressed(s.scripted, s.seat, Command.STOP)

    result = s.discover()

    assert result.ending is Ending.TERMINATED
    assert result.detail == (
        "the operator terminated the run; the click sent at step 2 came back "
        "ok and nothing was undone"
    )
    assert s.scripted.acted == [FIND.action]
    assert s.control.status().state is State.TERMINATED


@pytest.mark.parametrize("returned", [True, False])
def test_a_failed_resume_check_keeps_the_person_in_control(
    profile, returned: bool
) -> None:
    notes: list[Status] = []

    def wander(seat: ScriptedSeat) -> None:
        seat.location = WIRE

    def come_back(seat: ScriptedSeat) -> None:
        seat.location = MEMBER

    then = (
        [
            doing(State.HUMAN_CONTROL, come_back),
            when(State.HUMAN_CONTROL, Command.RESUME),
        ]
        if returned
        else [
            doing(State.HUMAN_CONTROL, advancing(profile.escalation.handoff_timeout_s))
        ]
    )
    s = session(
        profile,
        [LOOK, STUCK, DONE],
        when(State.PAUSED, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, wander),
        when(State.HUMAN_CONTROL, Command.RESUME),
        noting(State.HUMAN_CONTROL, notes),
        *then,
    )

    result = s.discover()

    assert notes[0].owner is Owner.HUMAN
    assert notes[0].notice == (
        "page-1 is outside the profile's routes; return it to a permitted page, "
        "then resume"
    )
    checked = [(event.passed, event.failure) for event in s.events(HandedBack)]
    assert checked[0] == (False, CheckFailure.OFF_ROUTE)
    if returned:
        assert checked[1:] == [(True, None)]
        assert result.ending is Ending.COMPLETED
    else:
        assert checked[1:] == []
        assert result.ending is Ending.HANDED_OFF
        assert result.detail == "a person was asked for: missing_user_input"


@pytest.mark.parametrize("lands", ["off_route", "past_the_budget"])
def test_terminate_during_an_action_ends_terminated_wherever_it_lands(
    profile, lands: str
) -> None:
    limit = profile.budgets.max_wall_clock_s
    after = ELSEWHERE if lands == "off_route" else MEMBER
    s = session(
        profile,
        [LOOK, FIND, DONE],
        screens=[Screen(MEMBER, controls=CONTROLS), Screen(after, controls=CONTROLS)],
    )
    s.surface = Pressed(
        s.scripted,
        s.seat,
        Command.TERMINATE,
        advance=2 * limit if lands == "past_the_budget" else 0.0,
    )

    result = s.discover()

    assert result.ending is Ending.TERMINATED
    assert result.detail == (
        "the operator terminated the run; the click sent at step 2 came back "
        "ok and nothing was undone"
    )
    assert s.scripted.acted == [FIND.action]
    assert len(s.scripted.looks) == 1


def test_resume_does_not_repeat_an_operation_that_was_already_performed(
    profile,
) -> None:
    s = session(profile, [LOOK, FIND, DONE], when(State.PAUSED, Command.RESUME))
    s.surface = Pressed(s.scripted, s.seat, Command.STOP)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == [FIND.action]
    assert len(s.events(Acted)) == 1
    assert (
        "the click sent at step 2 was performed; its effect is not confirmed; "
        "decide again from the screen as it is now"
    ) in s.notices(3)
    assert [event.interrupted for event in s.events(HandedBack)] == [
        Interruption.PERFORMED
    ]
    assert result.segments[0].interrupted is Interruption.PERFORMED


@dataclasses.dataclass
class StoppedAfter:
    """A scripted surface whose operator presses Stop just after its first act.

    The act has settled when Stop arrives, and the loop has not yet looked at
    what the act left behind.
    """

    inner: ScriptedSurface
    seat: ScriptedSeat
    acted: bool = False
    pressed: bool = False

    def location(self) -> str:
        if self.acted and not self.pressed:
            self.pressed = True
            control = self.seat.control
            assert control is not None
            self.seat.send(Command.STOP, control.status())
        return self.inner.location()

    def pages(self) -> tuple[PageInfo, ...]:
        return self.inner.pages()

    def capabilities(self) -> Capabilities:
        return self.inner.capabilities()

    def observe(self, request: ObservationRequest) -> Observation:
        return self.inner.observe(request)

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        result = self.inner.act(action, expect=expect)
        self.acted = self.acted or action.target != DONE_TARGET
        return result


def test_a_stop_after_an_action_settled_does_not_say_it_was_not_performed(
    profile,
) -> None:
    s = session(profile, [LOOK, FIND, DONE], when(State.PAUSED, Command.RESUME))
    s.surface = StoppedAfter(s.scripted, s.seat)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == [FIND.action]
    assert [event.step for event in s.events(Acted)] == [2]
    assert not any("was not performed" in notice for notice in s.notices(3))
    assert [event.interrupted for event in s.events(HandedBack)] != [
        Interruption.PENDING
    ]
    assert [event.step for event in s.events(Superseded)] == []


# What the model is told after a person had the session.


def test_the_next_decision_hears_what_the_person_did_and_looks_again(
    profile,
) -> None:
    def work(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.EDIT, role="textbox", name="Amount")
        seat.person(ManualKind.CLICK, role="button", name="Save")

    s = session(
        profile,
        [LOOK, FIND, DONE],
        doing(State.HUMAN_CONTROL, work),
        when(State.HUMAN_CONTROL, Command.RESUME),
    )
    s.deciding.during[2] = s.press(Command.TAKE_CONTROL)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.notices(3) == (
        (
            "while iv-1 was open a person: edited textbox 'Amount' on "
            "/members/:id; its value is withheld; clicked button 'Save' on "
            "/members/:id"
        ),
        (
            "the step that was waiting was not performed by the run; decide "
            "again from the screen as it is now"
        ),
        (
            "Fields edited by a person are protected as secrets. "
            "Their values are withheld."
        ),
    )
    assert s.scripted.looks[-1].provenance is ObservationProvenance.POST_INTERVENTION
    assert [event.provenance for event in s.events(Observed)][-1] is (
        ObservationProvenance.POST_INTERVENTION
    )
    assert s.script.seen[2].observations[0].observation_id == "obs-2"


def test_a_request_for_a_person_is_honoured_after_one_tool(profile) -> None:
    looked: list[list[ObservationMode]] = []

    def count(_seat: ScriptedSeat) -> None:
        looked.append([look.mode for look in s.scripted.looks])

    s = session(
        profile,
        [
            LOOK,
            AskHuman(Trigger.INSUFFICIENT_OBSERVATION, "the table is cut off"),
            DONE,
        ],
        doing(State.PAUSED, count),
        when(State.PAUSED, Command.RESUME),
    )

    result = s.discover()

    assert looked == [[ObservationMode.STRUCTURED]]
    assert [(event.ask, event.trigger) for event in s.events(Paused)] == [
        (Ask.PERSON, Trigger.INSUFFICIENT_OBSERVATION)
    ]
    assert result.ending is Ending.COMPLETED


def uninvited(s: Session) -> Callable[[], None]:
    """Return a person clicking Send while automation still holds the session."""
    return lambda: s.seat.person(
        ManualKind.CLICK, role="button", name="Send", owned=False
    )


def test_a_person_using_the_session_stops_automation_at_the_next_boundary(
    profile,
) -> None:
    s = session(profile, [LOOK, FIND, DONE], when(State.PAUSED, Command.RESUME))
    s.deciding.during[2] = uninvited(s)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == []
    assert [event.step for event in s.events(Superseded)] == [2]
    assert PERSON_STOPPED in [status.notice for status in s.window.shown]
    assert [event.owned for event in s.events(ManualAction)] == [False]


def test_input_that_stopped_automation_is_told_to_the_model_and_kept(
    profile,
) -> None:
    s = session(profile, [LOOK, FIND, DONE], when(State.PAUSED, Command.RESUME))
    s.deciding.during[2] = uninvited(s)

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert [step.event.kind for step in result.segments[0].steps] == [ManualKind.CLICK]
    assert any("clicked button 'Send'" in notice for notice in s.notices(3))


def test_a_voided_approval_reports_what_the_person_did_while_it_waited(
    profile,
) -> None:
    s = session(
        profile,
        [LOOK, STUCK, DONE, DONE],
        when(State.PAUSED, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, clicking("Save")),
        when(State.HUMAN_CONTROL, Command.RESUME),
        doing(State.AWAITING_APPROVAL, clicking("Refresh")),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
    )
    s.scripted.supports = {
        kind: forms for kind, forms in EVERYTHING.items() if kind is not ActionKind.READ
    }

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert result.verification is Verification.PERSON
    assert s.notices(3)[0] == (
        "while iv-1 was open a person: clicked button 'Save' on /members/:id"
    )
    assert not any("iv-1" in notice for notice in s.notices(4))
    assert (
        "while iv-2 was open a person: clicked button 'Refresh' on /members/:id"
        in s.notices(4)
    )


# What is written down, and what is kept for a recorder.


def test_the_journal_of_a_run_with_a_takeover_holds_no_text(profile) -> None:
    goal = "Pay member 10001 the sum of 4,200.00 from account S-7788"
    reason = "need the one-time code sent for S-7788"
    note = "typed code 553201 for member 10001"
    stream = io.StringIO()

    def work(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.EDIT, role="textbox", name="Code for S-7788")
        seat.person(ManualKind.CLICK, role="button", name="Confirm 4,200.00")

    s = session(
        profile,
        [LOOK, AskHuman(Trigger.MISSING_USER_INPUT, reason), DONE],
        when(State.PAUSED, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, work),
        when(State.HUMAN_CONTROL, Command.RESUME, note=note),
        journal=JsonlJournal(stream),
        screens=[Screen(MEMBER, controls=(("button", "Search"), ("cell", "10001")))],
    )

    result = s.discover(goal)

    written = stream.getvalue()
    assert result.ending is Ending.COMPLETED
    assert f"operator: {note}" in s.notices(3)
    for kind in ("Commanded", "Paused", "Transferred", "ManualAction", "HandedBack"):
        assert f'"event": "{kind}"' in written
    for text in (goal, reason, note, "10001", "4,200.00", "S-7788", "553201"):
        assert text not in written
    for name in ("Code for", "Confirm", "Amount", "Search"):
        assert name not in written


@pytest.mark.parametrize(
    ("click", "requirement"),
    [
        ({"any": "risky"}, Requirement.APPROVE_EACH_RUN),
        ({"any": "safe"}, Requirement.VALIDATE_AND_BIND),
        # One denied effect does not forbid every click. The step carries
        # no effect label, so a recorder labels it and the gate judges it on
        # every replay, where the denied effect is refused.
        (
            {"any": "safe", "effects": {"close_account": "deny"}},
            Requirement.APPROVE_EACH_RUN,
        ),
    ],
)
def test_the_run_result_carries_the_typed_manual_segment(
    edited_profile, click: dict, requirement: Requirement
) -> None:
    actions = grants(EVERY_KIND)
    actions["click"] = click
    profile = edited_profile(actions=actions)

    def work(seat: ScriptedSeat) -> None:
        seat.person(ManualKind.CLICK, role="button", name="Send")
        seat.person(ManualKind.EDIT, role="textbox", name="Password", secret=True)
        seat.person(ManualKind.SCROLL, name="", detail=Detail.DOWN)

    s = session(
        profile,
        [LOOK, STUCK, DONE],
        when(State.PAUSED, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, work),
        when(State.HUMAN_CONTROL, Command.RESUME),
    )

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    [segment] = result.segments
    assert segment.run == s.control.run
    assert segment.intervention == "iv-1"
    assert segment.mode is Mode.DISCOVERY
    assert segment.ask is Ask.PERSON
    assert segment.trigger is Trigger.MISSING_USER_INPUT
    assert segment.taken
    assert segment.interrupted is Interruption.NONE
    assert segment.complete
    assert [(step.event.kind, step.requirement) for step in segment.steps] == [
        (ManualKind.CLICK, requirement),
        (ManualKind.EDIT, Requirement.PERSON_EACH_RUN),
        (ManualKind.SCROLL, Requirement.CONTEXT),
    ]


# Regressions from the branch review. Each test fails without its fix.


def watching(edited_profile) -> Profile:
    """The example profile with looking at the screen declared risky."""
    return edited_profile(actions=grants({**EVERY_KIND, "observe": "risky"}))


def test_a_takeover_on_a_risky_look_tells_the_model_what_the_person_did(
    edited_profile,
) -> None:
    kept = Remember("balance", "4200.00", may_change=False, source=BALANCE)
    later = Remember("balance_later", "4200.00", may_change=False, source=BALANCE)
    s = session(
        watching(edited_profile),
        [LOOK, kept, LOOK, LOOK, later, DONE],
        when(State.AWAITING_APPROVAL, Command.APPROVE),
        when(State.AWAITING_APPROVAL, Command.TAKE_CONTROL),
        doing(State.HUMAN_CONTROL, clicking("Search")),
        when(State.HUMAN_CONTROL, Command.RESUME),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
    )
    s.scripted.extracts = ["4200.00", "4200.00"]

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert [event.outcome for event in s.events(Escalated)] == [
        HandoffOutcome.APPROVED,
        HandoffOutcome.RESUMED,
        HandoffOutcome.APPROVED,
    ]
    assert (
        "while iv-2 was open a person: clicked button 'Search' on /members/:id"
        in s.notices(4)
    )
    before, after = s.script.seen[2].memory, s.script.seen[3].memory
    assert [(fact.key, fact.may_change) for fact in before] == [("balance", False)]
    assert [(fact.key, fact.may_change) for fact in after] == [("balance", True)]
    epochs = {fact.key: fact.epoch for fact in s.script.seen[5].memory}
    assert epochs == {"balance": 0, "balance_later": 1}


@pytest.mark.parametrize("voided", ["click", "look"])
def test_an_approval_voided_by_a_persons_click_tells_the_model_what_they_did(
    edited_profile, voided: str
) -> None:
    looking = voided == "look"
    s = session(
        watching(edited_profile) if looking else edited_profile(),
        [LOOK, LOOK, DONE] if looking else [LOOK, PAY, DONE],
        doing(State.AWAITING_APPROVAL, clicking("Search")),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
    )

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == []
    notices = s.notices(2 if looking else 3)
    assert (
        "while iv-1 was open a person: clicked button 'Search' on /members/:id"
        in notices
    )
    assert "the session changed while the approval was pending; look again" in notices


def test_a_rejection_after_a_persons_input_tells_the_model_and_forgets_the_screen(
    profile,
) -> None:
    s = session(
        profile,
        [LOOK, PAY, DONE],
        doing(State.AWAITING_APPROVAL, clicking("Search")),
        when(State.AWAITING_APPROVAL, Command.REJECT),
    )

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.scripted.acted == []
    notices = s.notices(3)
    assert "the operator declined click" in notices
    assert (
        "while iv-1 was open a person: clicked button 'Search' on /members/:id"
        in notices
    )
    assert "the session changed while the approval was pending; look again" in notices
    assert s.script.seen[2].observations == ()


def test_a_stop_right_after_a_resume_supersedes_no_step_and_reports_none_unperformed(
    profile,
) -> None:
    s = session(profile, [LOOK, DONE])

    def stop_after_the_look(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not State.RUNNING or not s.scripted.looks:
            return False
        return seat.send(Command.STOP, status).verdict is Verdict.ACCEPTED

    s.seat.moves = [
        stop_after_the_look,
        when(State.PAUSED, Command.RESUME),
        when(State.RUNNING, Command.STOP),
        when(State.PAUSED, Command.RESUME),
    ]

    result = s.discover()

    assert result.ending is Ending.COMPLETED
    assert s.events(Superseded) == []
    assert [event.interrupted for event in s.events(HandedBack)] == [
        Interruption.NONE,
        Interruption.NONE,
    ]
    assert not any("was not performed" in notice for notice in s.notices(2))
    paused = [status for status in s.window.shown if status.state is State.PAUSED]
    assert {status.offer.intervention for status in paused if status.offer} == {
        "iv-1",
        "iv-2",
    }
    assert {status.interrupted for status in paused} == {""}
    assert [look.provenance for look in s.scripted.looks] == [
        ObservationProvenance.REQUESTED,
        ObservationProvenance.POST_INTERVENTION,
    ]
