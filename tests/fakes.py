"""Scripted collaborators for the discovery loop.

Protocols define every collaborator the loop needs. Tests can therefore drive
and assert its control flow without a browser, a model, or a real clock. Each
screen is one small description that both observation tools can render. One
script can exercise a run regardless of which tool it chooses first.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    Capabilities,
    Expectation,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    PageInfo,
    PageState,
    PendingDialog,
    TargetForm,
    VisualRegion,
    Window,
)
from computeruse.control import (
    Control,
    Order,
    PageProbe,
    Receipt,
    Recording,
    SessionProbe,
    Status,
)
from computeruse.decider import (
    AskHuman,
    CheckKind,
    Decision,
    Finish,
    ResultCheck,
    Task,
    Transcript,
)
from computeruse.escalation import (
    Command,
    Handoff,
    HandoffOutcome,
    InterventionRequest,
    Owner,
    State,
    Trigger,
    Via,
)
from computeruse.manual import Detail, Gap, ManualEvent, ManualKind, Widget
from computeruse.profile import ActionKind

DONE_TARGET = AxLocator("status", "Task done")
"""The control every scripted surface shows, for a scripted claim to check."""

DONE = Finish(
    {"status": "Task done"},
    "done",
    checks=(ResultCheck(CheckKind.RESULT, DONE_TARGET, "Task done", output="status"),),
)
"""A finish claim a scripted surface can verify: it reads DONE_TARGET."""

EVERYTHING: Capabilities = {kind: frozenset(TargetForm) for kind in ActionKind}
"""A scripted surface performs every action type with every target form."""


@dataclasses.dataclass
class Screen:
    """One page, described once and observable through either tool."""

    location: str
    controls: tuple[tuple[str, str], ...] = (("button", "Search"),)
    regions: tuple[str, ...] = ()
    structured: ObservationStatus = ObservationStatus.COMPLETE
    visual: ObservationStatus = ObservationStatus.COMPLETE
    dialog: PendingDialog | None = None


@dataclasses.dataclass
class ScriptedSurface:
    """A surface that shows a fixed sequence of screens.

    Each action advances to the next screen. The final screen repeats so a run
    can keep acting on it. Outcomes are consumed in order and default to ``OK``
    after the script runs out.

    ``painted`` contains the lines shown in a painted picture. Each read
    returns them in ``ActionResult.screen`` beside its extracted value.

    Every screen also shows ``DONE_TARGET``. The loop reads it to check a
    ``DONE`` claim. The fake keeps that read in ``checked`` instead of
    ``acted``. It consumes no outcome and does not advance the screen.
    """

    screens: list[Screen]
    outcomes: list[Outcome] = dataclasses.field(default_factory=list)
    extracts: list[str] = dataclasses.field(default_factory=list)
    painted: tuple[str, ...] = ()
    acted: list[Action] = dataclasses.field(default_factory=list)
    checked: list[Action] = dataclasses.field(default_factory=list)
    supports: Capabilities = dataclasses.field(default_factory=lambda: EVERYTHING)
    looks: list[ObservationRequest] = dataclasses.field(default_factory=list)
    expected: list[Expectation | None] = dataclasses.field(default_factory=list)
    windows: tuple[PageInfo, ...] = ()
    sequence: int = 0
    page_size: int | None = None

    def location(self) -> str:
        return self.screens[0].location

    def capabilities(self) -> Capabilities:
        return self.supports

    def pages(self) -> tuple[PageInfo, ...]:
        return self.windows

    def observe(self, request: ObservationRequest) -> Observation:
        self.looks.append(request)
        self.sequence += 1
        screen = self.screens[0]
        structured = request.mode is ObservationMode.STRUCTURED
        status = screen.structured if structured else screen.visual
        if structured and self.page_size is not None:
            # A bounded adapter shows one part of a long screen per look.
            controls = screen.controls
            part = controls[request.start : request.start + self.page_size]
            window = Window(request.start, len(part), len(controls))
            return Observation(
                observation_id=f"obs-{self.sequence}",
                mode=request.mode,
                status=ObservationStatus.PARTIAL if window.rest else status,
                page_state=PageState(screen.location),
                nodes=tuple(AxNode(role, name) for role, name in part),
                window=window,
            )
        return Observation(
            observation_id=f"obs-{self.sequence}",
            mode=request.mode,
            status=status,
            page_state=PageState(
                screen.location,
                tuple(f"{role}:{name}" for role, name in screen.controls)
                if structured
                else (),
            ),
            nodes=tuple(AxNode(role, name) for role, name in screen.controls)
            if structured
            else (),
            regions=tuple(VisualRegion(anchor) for anchor in screen.regions)
            if not structured
            else (),
            image=None if structured else b"pretend-png",
            dialog=screen.dialog,
        )

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        if action.kind is ActionKind.READ and action.target == DONE_TARGET:
            self.checked.append(action)
            return ActionResult(
                outcome=Outcome.OK,
                page_state=PageState(self.screens[0].location),
                extracted="Task done",
            )
        self.acted.append(action)
        self.expected.append(expect)
        outcome = self.outcomes.pop(0) if self.outcomes else Outcome.OK
        extracted = self.extracts.pop(0) if self.extracts else None
        if len(self.screens) > 1:
            self.screens.pop(0)
        return ActionResult(
            outcome=outcome,
            page_state=PageState(self.screens[0].location),
            extracted=extracted,
            screen=self.painted if action.kind is ActionKind.READ else (),
        )


def grants(declared: dict[str, str]) -> dict[str, dict[str, object]]:
    """Write each action type as the operator's explicit grant of every operation."""
    return {kind: {"any": risk} for kind, risk in declared.items()}


class ReadsNoRecords:
    """Reads every goal as a task about no particular record.

    A test decider that only scripts decisions uses this reading. The reading
    asks nothing of a finish claim beyond its own checks.
    """

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        del goal, notices
        return Task()


@dataclasses.dataclass
class ScriptedDecider:
    """A decider that replays a fixed list of decisions.

    Its reading of the goal is ``tasks``, taken in order with the last one
    repeated. The default task names no record and asks nothing of a finish
    claim beyond its own checks.
    """

    decisions: list[Decision]
    seen: list[Transcript] = dataclasses.field(default_factory=list)
    tasks: list[Task] = dataclasses.field(default_factory=lambda: [Task()])
    interpreted: list[tuple[str, ...]] = dataclasses.field(default_factory=list)

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        del goal
        self.interpreted.append(notices)
        return self.tasks.pop(0) if len(self.tasks) > 1 else self.tasks[0]

    def decide(self, transcript: Transcript) -> Decision:
        self.seen.append(transcript)
        if not self.decisions:
            return AskHuman(Trigger.NO_PROGRESS, "script exhausted")
        return self.decisions.pop(0)


@dataclasses.dataclass
class ScriptedEscalator:
    """An escalator that answers with a fixed list of handoffs."""

    handoffs: list[Handoff] = dataclasses.field(default_factory=list)
    requests: list[InterventionRequest] = dataclasses.field(default_factory=list)

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        if not self.handoffs:
            return Handoff(HandoffOutcome.TIMED_OUT)
        return self.handoffs.pop(0)


class FakeClock:
    """A monotonic clock that only moves when a test moves it."""

    __slots__ = ("_now",)

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


type Move = Callable[["ScriptedSeat", Status], bool]
"""One thing a scripted operator does when the status allows it.

It returns True once it has acted, and the seat moves on to the next one.
"""


@dataclasses.dataclass
class ScriptedSeat:
    """A live session a scripted operator sits at, without a browser.

    Each wait slice the control takes on the owner thread offers the next
    move the current status, the way a person glances at the control window
    between clicks. ``wait_s`` is how far the fake clock moves per slice, so a
    test can make a person slow without sleeping. ``probes`` are what the
    session looks like at each hand-back check, the last one repeating.
    """

    clock: FakeClock
    moves: list[Move] = dataclasses.field(default_factory=list)
    visible: bool = True
    wait_s: float = 0.0
    probes: list[SessionProbe] = dataclasses.field(default_factory=list)
    location: str = "https://sandbox.example.test/members/12345"
    limit: int = 200_000
    """Slices after which a wait that never ends fails the test instead."""
    control: Control | None = None
    transfers: list[Owner] = dataclasses.field(default_factory=list)
    pending: list[ManualEvent] = dataclasses.field(default_factory=list)
    gaps: list[Gap] = dataclasses.field(default_factory=list)
    unprotected: int = 0
    handed_back: int = 0
    slices: int = 0
    permitted: Callable[[], bool] | None = None

    def where(self) -> str:
        return "the scripted session" if self.visible else ""

    def idle(self, seconds: float) -> None:
        del seconds
        self.slices += 1
        if self.slices > self.limit:
            raise AssertionError("the scripted operator never ended the wait")
        self.clock.advance(self.wait_s)
        time.sleep(0.0005)
        if self.control is None or not self.moves:
            return
        if self.moves[0](self, self.control.status()):
            self.moves.pop(0)

    def transfer(self, owner: Owner, intervention: str) -> None:
        del intervention
        self.transfers.append(owner)

    def activity(self) -> tuple[ManualEvent, ...]:
        found = tuple(self.pending)
        self.pending.clear()
        return found

    def hand_back(self, *, revoke: bool = True) -> Recording:
        del revoke
        self.handed_back += 1
        return Recording(self.activity(), tuple(self.gaps), self.unprotected)

    def watch(self, permitted: Callable[[], bool]) -> None:
        self.permitted = permitted

    def probe(self) -> SessionProbe:
        if self.probes:
            return self.probes.pop(0) if len(self.probes) > 1 else self.probes[0]
        return SessionProbe(
            (PageProbe("page-1", self.location, dialog=False, ready=True),)
        )

    def send(
        self, command: Command, status: Status, *, note: str | None = None
    ) -> Receipt:
        """Send ``command`` as the operator looking at ``status`` would."""
        assert self.control is not None
        offer = status.offer
        return self.control.submit(
            Order(
                command,
                status.run,
                intervention=offer.intervention if offer is not None else "",
                revision=status.revision,
                via=Via.SCRIPT,
                note=note,
            )
        )

    def person(
        self,
        kind: ManualKind,
        *,
        role: str = "button",
        name: str = "Approve",
        owned: bool = True,
        secret: bool = False,
        detail: Detail = Detail.NONE,
    ) -> None:
        """Record one thing the scripted person did in the session."""
        self.pending.append(
            ManualEvent(
                sequence=len(self.pending) + 1,
                at=self.clock(),
                kind=kind,
                owned=owned,
                page="page-1",
                route="/members/:id",
                control=Widget.of(role),
                target=AxLocator(role, name) if name else None,
                location=self.location,
                secret=secret,
                detail=detail,
            )
        )


def when(state: State, command: Command, *, note: str | None = None) -> Move:
    """Send ``command`` once the run reaches ``state``."""

    def move(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not state:
            return False
        seat.send(command, status, note=note)
        return True

    return move


def doing(state: State, act: Callable[[ScriptedSeat], None]) -> Move:
    """Do something in the session once the run reaches ``state``."""

    def move(seat: ScriptedSeat, status: Status) -> bool:
        if status.state is not state:
            return False
        act(seat)
        return True

    return move
