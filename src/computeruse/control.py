"""Manage ownership of the live session during discovery and replay.

Each run has one control and one owner at a time. The owner is automation,
an operator deciding how to proceed, or a person using the session directly.
Discovery and replay share this control regardless of how they choose actions.

Commands may arrive from any thread through a window, terminal, or test.
Validate each against the run, current intervention, and displayed status
revision. Reject and record duplicate clicks, stale answers, and commands
for another run. Accepting a command changes only state, under a lock.

Only the thread running the discovery or replay loop can operate the browser
driver. It processes session events in short intervals while a worker calls
the model or a person owns the session. Discard decisions returned after
ownership changes.

Approval permits one proposal only after fresh policy, target, and record
checks. Returning the session grants no approval. Neither action changes the
profile or removes a learned restriction.
"""

from __future__ import annotations

import contextlib
import dataclasses
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

from computeruse import policy
from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    Capabilities,
    DomLocator,
    Expectation,
    Observation,
    ObservationRequest,
    Outcome,
    PageInfo,
    ScreenTarget,
    VisualAnchor,
)
from computeruse.budget import Budget, Clock
from computeruse.escalation import (
    Ask,
    CheckFailure,
    Command,
    Escalator,
    Handoff,
    HandoffOutcome,
    Interruption,
    InterventionRequest,
    Mode,
    Owner,
    State,
    Trigger,
    Verdict,
    Via,
)
from computeruse.journal import (
    Commanded,
    EvidenceError,
    HandedBack,
    Journal,
    ManualAction,
    Paused,
    RunEvent,
    Transferred,
)
from computeruse.manual import (
    Gap,
    GapKind,
    ManualEvent,
    ManualKind,
    ManualSegment,
    ManualStep,
    classify,
    gaps_in,
    merge,
)
from computeruse.operations import BoundOperation
from computeruse.policy import Restriction
from computeruse.profile import ActionKind, Profile
from computeruse.surface import Surface, control_at, operation_binding

FINAL = frozenset({State.COMPLETED, State.FAILED, State.TERMINATED})
AUTOMATION = frozenset({State.RUNNING, State.STOPPING, State.CHECKING})
HELD = frozenset({State.READY, State.PAUSED, State.AWAITING_APPROVAL})
ANSWERS = frozenset({Command.RESUME, Command.APPROVE, Command.REJECT})
"""Commands that must identify the intervention they answer."""

CHECK_READS = frozenset({ActionKind.READ, ActionKind.ASSERT, ActionKind.WAIT_FOR})
"""Read-only operations permitted during a caller's resume check."""

NOT_SENT = "the operator stopped the run before this input was sent"
"""Adapter detail for input refused after session ownership changed."""

SENT_ELSEWHERE = "the input was sent, and the page it led to is outside the profile"
"""Adapter detail for sent input whose navigation was subsequently reversed."""

MAX_EVENTS = 500
"""Maximum retained manual events per intervention. Excess events create a gap."""

READY_TRIES = 10
"""Maximum page-readiness checks during return of control, 0.2 seconds apart."""


def owner_of(state: State) -> Owner:
    """Return who may operate the session in ``state``.

    Examples
    --------
    >>> owner_of(State.STOPPING)
    <Owner.AUTOMATION: 'automation'>
    >>> owner_of(State.AWAITING_APPROVAL)
    <Owner.OPERATOR: 'operator'>
    """
    if state in AUTOMATION:
        return Owner.AUTOMATION
    if state in HELD:
        return Owner.OPERATOR
    if state is State.HUMAN_CONTROL:
        return Owner.HUMAN
    return Owner.NONE


@dataclasses.dataclass(frozen=True, slots=True)
class Order:
    """An operator command bound to the status that prompted it.

    Each channel sends the ``revision`` it last displayed or printed.
    This identifies duplicate clicks and commands based on stale status.
    Approve, reject, and resume must name their request in ``intervention``.
    """

    command: Command
    run: str
    revision: int
    intervention: str = ""
    via: Via = Via.SCRIPT
    note: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Button:
    """The current appearance of the shared Start, Stop, and Resume button."""

    label: str
    command: Command | None
    enabled: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Offer:
    """The current intervention for display, containing live data.

    ``context`` identifies the discovery goal or replay capability. Reasons
    and proposals may quote the screen. Channels display them without storage.
    """

    intervention: str
    ask: Ask
    trigger: Trigger | None
    reason: str
    context: str
    proposal: str
    route: str
    session: str
    timeout_s: float
    left_s: float


@dataclasses.dataclass(frozen=True, slots=True)
class Status:
    """The state a channel needs to display controls and describe the run."""

    run: str
    mode: Mode
    revision: int
    state: State
    owner: Owner
    step: int
    primary: Button
    offered: frozenset[Command]
    remaining_s: float
    paused_s: float
    notice: str = ""
    settling: str = ""
    offer: Offer | None = None
    ending: str = ""
    interrupted: str = ""
    """The interrupted operation's status for the operator."""
    purpose: str = ""
    """The control's purpose for its window title, such as a comparison.

    Empty for a discovery or replay started by the operator.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class Receipt:
    """The command decision and resulting control status."""

    verdict: Verdict
    reason: str
    status: Status


@dataclasses.dataclass(frozen=True, slots=True)
class Dispatch:
    """An automated operation and its eventual outcome."""

    step: int
    kind: ActionKind | None
    outcome: Outcome | None = None
    detail: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class PageProbe:
    """Live page state used to validate a returned session."""

    page: str
    location: str
    dialog: bool
    ready: bool


@dataclasses.dataclass(frozen=True, slots=True)
class SessionProbe:
    """Session metadata for checking return of control without reading values.

    ``credential_prompts`` counts visible fields the page marks as passwords
    or one-time codes. No value is read, not even whether one is empty.
    ``protection_lost`` says a field holding a secret left the page without
    the page navigating, so a capture could no longer mask it.
    """

    pages: tuple[PageProbe, ...]
    credential_prompts: int = 0
    protection_lost: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Recording:
    """Recorded state after a person returns the session.

    ``unprotected`` counts fields a person typed into that are still on the
    page and could not be marked as holding a secret. Any of them means a
    capture could show what the person typed, so the hand-back is refused.
    """

    events: tuple[ManualEvent, ...] = ()
    gaps: tuple[Gap, ...] = ()
    unprotected: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class HandBack:
    """A returned session that passed validation, containing live data."""

    intervention: str
    segment: ManualSegment
    interrupted: Interruption
    dispatched: Dispatch | None
    notices: tuple[str, ...] = ()


class Seat(Protocol):
    """The live-session operations required for human takeover.

    Every method runs on the owner thread, because only that thread may drive
    the session.
    """

    def where(self) -> str:
        """Return the session location, or "" if the session is not visible."""
        ...

    def idle(self, seconds: float) -> None:
        """Deliver pending session events for up to ``seconds``."""
        ...

    def transfer(self, owner: Owner, intervention: str) -> None:
        """Assign session ownership to ``owner``.

        Entering ``Owner.HUMAN`` revokes every capture and screen target the
        automation held and makes the surface refuse automation input and
        capture until the owner changes again.
        """
        ...

    def activity(self) -> tuple[ManualEvent, ...]:
        """Return manual input recorded since the last call, in order.

        Automated input must match an adapter input call and its target while
        automation owned the session. Other input is manual. If the adapter
        cannot determine its source, report the input as uncertain.
        """
        ...

    def hand_back(self, *, revoke: bool = True) -> Recording:
        """Flush the recording and protect every field a person typed into.

        Unprotected fields remain in ``unprotected`` until the adapter can
        protect them or their document is gone. Commands and notes cannot
        clear that tracking. With ``revoke`` false, retain existing captures
        and targets so approval can apply to the original proposal. Never
        run page scripts while a dialog is open.
        """
        ...

    def probe(self) -> SessionProbe:
        """Report pages, dialogs, readiness, and credential prompts.

        A page with a dialog open is reported from what the surface already
        knows, without running page script, which would wait for the dialog.
        """
        ...

    def watch(self, permitted: Callable[[], bool]) -> None:
        """Ask ``permitted`` immediately before sending any input.

        Target lookup can take seconds. A stop, termination, or manual input
        during that wait must prevent dispatch. Recheck permission immediately
        before sending input, and report a blocked operation if it was revoked.
        """
        ...


class Channel(Protocol):
    """A channel for displaying run status and receiving operator commands."""

    def attach(self, control: Control) -> None:
        """Connect operator commands to this control."""
        ...

    def show(self, status: Status) -> None:
        """Draw ``status``. Called on the owner thread when it changes."""
        ...

    def listening(self) -> bool:
        """Report whether an operator can send commands through this channel."""
        ...


class Overtaken:
    """The result used when ownership changes during a model call."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "OVERTAKEN"


OVERTAKEN = Overtaken()


class ControlChanged(Exception):  # noqa: N818  a change of hands, not a fault
    """Raised by a guarded surface when automation no longer owns the run.

    The loop and a replay catch it at their step boundary and pause there.
    It never counts as a failure of the application: nothing was sent.
    """


type Check = Callable[[], tuple[CheckFailure, str] | None]
"""An owner-thread check before automation resumes.

Return a failure with an explanation for the operator, or None.
"""


@dataclasses.dataclass(slots=True, eq=False)
class _Episode:
    """A pause from the stop request until automation regains ownership."""

    intervention: str
    ask: Ask
    trigger: Trigger | None
    reason: str
    request: InterventionRequest | None
    started: float
    deadline: float
    interrupted: Interruption
    dispatched: Dispatch | None
    baseline: SessionProbe | None = None
    events: list[ManualEvent] = dataclasses.field(default_factory=list)
    gaps: list[Gap] = dataclasses.field(default_factory=list)
    dropped: int = 0
    taken: bool = False
    released: bool = False
    answer: HandoffOutcome | None = None
    note: str | None = None
    expired: bool = False

    @property
    def changed(self) -> bool:
        """Report whether a person did anything that could change the screen."""
        return any(event.changes for event in self.events)


class Control:
    """The session authority for one run.

    Parameters
    ----------
    mode
        Discovery or replay.
    clock
        Monotonic seconds, the same clock the run's budget uses.
    seat
        The live session a person can be handed. None when the operator
        channel answers synchronously and cannot hand over input.
    channels
        Channels that display status and submit commands.
    escalator
        A synchronous operator channel. When given, each intervention is
        answered by calling it; commands from channels still apply.
    wait_for_start
        Hold the run until an operator sends Start.
    worker
        Run model calls on a worker thread so commands stay responsive
        during them. Only meaningful with a seat; scripted tests that need a
        fixed order turn it off.
    """

    @property
    def proposal(self) -> Action | None:
        """Return the pending proposal for the live operator display only."""
        with self._lock:
            episode = self._episode
            return episode.request.action if episode and episode.request else None

    def __init__(
        self,
        *,
        mode: Mode,
        clock: Clock,
        seat: Seat | None = None,
        channels: Sequence[Channel] = (),
        escalator: Escalator | None = None,
        wait_for_start: bool = False,
        worker: bool = True,
        interval_s: float = 0.02,
        purpose: str = "",
    ) -> None:
        self._mode = mode
        self._purpose = purpose
        self._clock = clock
        self._seat = seat
        self._channels = list(channels)
        self._escalator = escalator
        self._wait_for_start = wait_for_start
        self._worker = worker and seat is not None
        self._interval = interval_s
        self._run = f"run-{secrets.token_hex(4)}"
        self._where = seat.where() if seat is not None else ""
        self._lock = threading.RLock()
        self._state = State.READY
        self._revision = 0
        self._pending: State | None = None
        self._episode: _Episode | None = None
        self._count = 0
        self._accepted: set[tuple[Command, int]] = set()
        self._outbox: list[RunEvent] = []
        self._notice = ""
        self._in_flight: Dispatch | None = None
        self._interrupted_by: Dispatch | None = None
        self._discarded = False
        self._segments: list[ManualSegment] = []
        self._hand_back: HandBack | None = None
        self._early: list[ManualEvent] = []
        self._cause = ""
        self._retried = 0.0
        self._owner: threading.Thread | None = None
        self._budget: Budget | None = None
        self._journal: Journal | None = None
        self._profile: Profile | None = None
        self._context = ""
        self._restrictions: Callable[[], tuple[Restriction, ...]] = tuple
        self._ending = ""
        self._shown = -1
        self._finished = False
        for channel in self._channels:
            channel.attach(self)
        if seat is not None:
            seat.watch(self.permits)

    # Any thread.

    @property
    def run(self) -> str:
        """The run ID required on each command."""
        return self._run

    @property
    def mode(self) -> Mode:
        """Whether this control belongs to discovery or to replay."""
        return self._mode

    def status(self) -> Status:
        """Return the status an operator should see now."""
        with self._lock:
            return self._status()

    def submit(self, order: Order) -> Receipt:
        """Validate and apply an operator command.

        Safe to call from any thread. This changes no session objects. The
        owner thread handles the new state at its next boundary or wait interval.
        """
        with self._lock:
            verdict, reason = self._judge(order)
            if verdict is Verdict.ACCEPTED:
                self._apply(order)
            self._outbox.append(
                Commanded(
                    step=self._step(),
                    command=order.command,
                    verdict=verdict,
                    via=order.via,
                    intervention=(
                        order.intervention if self._issued(order.intervention) else ""
                    ),
                )
            )
            return Receipt(verdict, reason, self._status())

    def _issued(self, intervention: str) -> bool:
        """Report whether this control issued ``intervention``.

        Request IDs may contain accidental user data. Persist only IDs issued
        by this control.
        """
        number = intervention.removeprefix("iv-")
        return (
            intervention.startswith("iv-")
            and number.isdigit()
            and 1 <= int(number) <= self._count
        )

    # Owner thread: lifecycle.

    def validate_start(self, mode: Mode) -> None:
        """Require an unused control for the requested execution mode."""
        with self._lock:
            if self._mode is not mode:
                raise ValueError(f"execution requires a {mode.value} control")
            if self._finished:
                raise ValueError("the control already ended its invocation")
            if self._owner is not None:
                raise ValueError("the control already began its invocation")

    def begin(
        self,
        *,
        profile: Profile,
        budget: Budget,
        journal: Journal,
        context: str,
        restrictions: Callable[[], tuple[Restriction, ...]] = tuple,
    ) -> bool:
        """Begin execution, waiting for Start when configured.

        ``restrictions`` returns what the run has learned so far, so a manual
        step on a restricted operation is classified as strictly as the
        automation's own would be. Returns False when the operator terminated
        the run before it started.
        """
        self.validate_start(self._mode)
        self._profile = profile
        self._budget = budget
        self._journal = journal
        self._context = context
        self._restrictions = restrictions
        self._owner = threading.current_thread()
        if self._wait_for_start:
            if self._seat is not None:
                self._seat.transfer(Owner.OPERATOR, "")
            budget.pause()
            try:
                self._display()
                self._await_start()
            finally:
                budget.unpause()
        else:
            with self._lock:
                if self._state is State.READY:
                    self._state = State.RUNNING
                    self._revision += 1
        if self.status().state is State.TERMINATED:
            self._flush()
            return False
        if self._seat is not None:
            self._seat.transfer(Owner.AUTOMATION, "")
        self._tick(0)
        return True

    def _await_start(self) -> None:
        """Wait for Start, then protect anything a person typed before it."""
        while True:
            state = self.status().state
            if state is State.CHECKING:
                recording = (
                    self._seat.hand_back() if self._seat is not None else Recording()
                )
                self._take(recording.events)
                with self._lock:
                    # A Terminate accepted while the check ran stands.
                    if self._state is State.CHECKING and recording.unprotected:
                        self._state = State.READY
                        self._notice = (
                            "a field a person typed into could not be protected "
                            "from capture; reload the page, then start"
                        )
                        self._revision += 1
                    elif self._state is State.CHECKING:
                        self._state = State.RUNNING
                        self._notice = ""
                        self._revision += 1
                continue
            if state is not State.READY:
                return
            self._tick(self._interval)
            with self._lock:
                if self._state is State.READY and not self._attended():
                    # Every channel has gone, so nobody can press Start.
                    self._state = State.TERMINATED
                    self._notice = "nobody can start this run"
                    self._revision += 1

    def finish(self, ending: str) -> None:
        """Finalize control state once at the end of a run.

        A person still holding the session has their recording drained and
        kept. The final state is completed, terminated, or failed. A replay
        that stopped before its first change, as its caller asked, completes
        with the ending "stopped".
        """
        if self._finished:
            return
        self._finished = True
        episode = self._episode
        if episode is not None:
            if self._seat is not None:
                try:
                    recording = self._seat.hand_back()
                except Exception:  # a closed session still ends the run once
                    recording = Recording(gaps=(Gap(GapKind.NOT_DRAINED),))
                self._take(recording.events)
                episode.gaps.extend(recording.gaps)
            self._keep(episode)
            if self._budget is not None:
                self._budget.unpause()
        with self._lock:
            if ending == "terminated" or self._state is State.TERMINATED:
                self._state = State.TERMINATED
            elif ending in {"completed", "stopped"}:
                self._state = State.COMPLETED
            else:
                self._state = State.FAILED
            self._episode = None
            self._ending = ending
            self._revision += 1
        if self._seat is not None:
            with contextlib.suppress(Exception):
                self._seat.transfer(Owner.NONE, "")
        self._flush()
        self._display()

    # Owner thread: boundaries.

    def halted(self) -> bool:
        """Process pending commands and report whether automation must stop.

        True means no new operation may start: the operator stopped the run,
        took it, or ended it, or the run is waiting for a person.
        """
        self._tick(0)
        with self._lock:
            return self._state is not State.RUNNING

    def hold(self, checks: Check | None = None) -> Handoff:
        """Pause at an execution boundary until the intervention ends.

        Call it when ``halted`` is true. A stop becomes a pause, a takeover
        hands the person the session, and either waits until the operator
        resumes, terminates, or the profile's handoff timeout passes.
        """
        with self._lock:
            state = self._state
            target = self._pending or State.PAUSED
            cause, self._cause = self._cause, ""
        if state is State.TERMINATED:
            return Handoff(HandoffOutcome.TERMINATED)
        if state is not State.STOPPING:
            return Handoff(HandoffOutcome.RESUMED)
        taken = target is State.HUMAN_CONTROL
        self._open(
            Ask.PAUSE,
            None,
            None,
            cause
            or ("taken over by the operator" if taken else "stopped by the operator"),
            target,
        )
        if not self._attended() and self._escalator is None:
            return self._close(HandoffOutcome.TIMED_OUT)
        return self._serve(checks)

    def intervene(
        self, request: InterventionRequest, checks: Check | None = None
    ) -> Handoff:
        """Present ``request`` and return the operator's response.

        An approval request waits for approve once, reject, or a takeover.
        A request for help waits for a takeover or a resume. Either can be
        terminated, and either times out after the request's timeout.
        """
        self._tick(0)
        with self._lock:
            state = self._state
            pending = self._pending
        if state is State.TERMINATED:
            return Handoff(HandoffOutcome.TERMINATED)
        if state is State.STOPPING and request.ask is Ask.APPROVAL:
            # The operator stopped the run before it asked. The stop wins: the
            # proposal is not performed, and the model decides again after
            # the pause, which asks again if it still wants the action.
            self.discard()
            return self.hold(checks)
        target = (
            State.AWAITING_APPROVAL if request.ask is Ask.APPROVAL else State.PAUSED
        )
        if state is State.STOPPING and pending is State.HUMAN_CONTROL:
            target = State.HUMAN_CONTROL
        episode = self._open(
            request.ask, request.trigger, request, request.reason, target
        )
        request = dataclasses.replace(
            request, intervention=episode.intervention, session=self._where
        )
        episode.request = request
        if self._escalator is not None:
            return self._escalate(episode, request, checks)
        if not self._attended():
            # Nobody can answer, so waiting out the timeout would only hold a
            # run nobody is watching.
            return self._close(HandoffOutcome.TIMED_OUT)
        return self._serve(checks)

    def guard(self, surface: Surface) -> Surface:
        """Wrap ``surface`` so it refuses to act unless automation owns the run."""
        return _Guarded(surface, self)

    def call[T](self, work: Callable[[], T]) -> T | Overtaken:
        """Run a model call and discard its result if ownership changes.

        With a worker, the owner thread processes session events during the
        call. An abandoned call continues running, but its result is ignored.
        The decider must accept new calls while abandoned calls remain active.
        Without a worker, the call runs here. The caller must check ownership
        before using its result.
        """
        if not self._worker:
            return work()
        done = threading.Event()
        results: list[T] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                results.append(work())
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        worker = threading.Thread(target=run, name="computeruse-decision", daemon=True)
        worker.start()
        while True:
            self._tick(self._interval)
            if self.status().state is not State.RUNNING:
                return OVERTAKEN
            if done.is_set():
                break
        if errors:
            raise errors[0]
        return results[0]

    def permits(self) -> bool:
        """Report whether automation may send input now. Safe from any thread.

        On the thread that owns the session, the seat's recorded activity is
        taken first, so a person's input heard while an operation waited for
        its target stops that operation before it sends anything.
        """
        if self._seat is not None and threading.current_thread() is self._owner:
            self._take(self._seat.activity())
        with self._lock:
            return self._state is State.RUNNING

    def discard(self) -> None:
        """Record that an ownership change invalidated a model decision."""
        with self._lock:
            self._discarded = True

    @property
    def hand_back(self) -> HandBack | None:
        """The latest returned session that passed validation."""
        return self._hand_back

    @property
    def segments(self) -> tuple[ManualSegment, ...]:
        """Every manual segment of this run, in order. Live data."""
        return tuple(self._segments)

    def interrupted(self) -> Dispatch | None:
        """Return the operation a stop or termination arrived during, if any."""
        with self._lock:
            return self._interrupted_by

    def terminated(self) -> bool:
        """Report whether the operator terminated the run."""
        with self._lock:
            return self._state is State.TERMINATED

    def _attended(self) -> bool:
        return any(channel.listening() for channel in self._channels)

    # Judging and applying orders, under the lock.

    def _judge(self, order: Order) -> tuple[Verdict, str]:
        if order.run != self._run:
            return Verdict.STALE, "that command was for another run"
        if order.command is Command.TERMINATE and self._state is State.TERMINATED:
            return Verdict.DUPLICATE, "the run is already terminated"
        if (order.command, order.revision) in self._accepted:
            return Verdict.DUPLICATE, "that command was already applied"
        open_request = self._episode.intervention if self._episode else ""
        if (order.command in ANSWERS or order.intervention) and (
            order.intervention != open_request
        ):
            if not open_request:
                return Verdict.STALE, "no request is open"
            return Verdict.STALE, (
                f"that answer was for {order.intervention or 'no request'}; "
                f"the open request is {open_request}"
            )
        if order.revision < self._revision:
            return Verdict.STALE, "the run changed since that command was sent"
        if order.revision > self._revision:
            return Verdict.INVALID, "that command names a state the run never had"
        episode = self._episode
        if (
            episode is not None
            and order.command is not Command.TERMINATE
            and (episode.expired or self._clock() >= episode.deadline)
        ):
            return Verdict.INVALID, "that request timed out and the run is ending"
        if order.command not in self._offered():
            return Verdict.INVALID, self._why_not(order.command)
        return Verdict.ACCEPTED, ""

    def _apply(self, order: Order) -> None:
        command = order.command
        episode = self._episode
        self._accepted.add((command, order.revision))
        match command:
            case Command.START:
                self._state = State.CHECKING
            case Command.STOP | Command.TAKE_CONTROL if self._state is State.RUNNING:
                self._state = State.STOPPING
                self._pending = (
                    State.HUMAN_CONTROL
                    if command is Command.TAKE_CONTROL
                    else State.PAUSED
                )
                self._interrupted_by = self._in_flight
            case Command.TAKE_CONTROL if self._state is State.STOPPING:
                self._pending = State.HUMAN_CONTROL
            case Command.TAKE_CONTROL if episode is not None:
                self._state = State.HUMAN_CONTROL
                episode.taken = True
                episode.answer = None
            case Command.RESUME if episode is not None:
                self._state = State.CHECKING
                episode.note = order.note
            case Command.APPROVE | Command.REJECT if episode is not None:
                # Anything a person typed while the approval waited is
                # protected before automation looks again, so an answer goes
                # through the same hand-back as a resume.
                self._state = State.CHECKING
                episode.answer = (
                    HandoffOutcome.APPROVED
                    if command is Command.APPROVE
                    else HandoffOutcome.REJECTED
                )
                episode.note = order.note
            case Command.TERMINATE:
                if (
                    self._state in {State.RUNNING, State.STOPPING}
                    and self._in_flight is not None
                ):
                    self._interrupted_by = self._in_flight
                self._state = State.TERMINATED
                self._pending = None
            case _:
                return
        self._notice = ""
        self._revision += 1

    def _offered(self) -> frozenset[Command]:
        if self._episode is not None and self._episode.expired:
            return frozenset({Command.TERMINATE})
        take = {Command.TAKE_CONTROL} if self._where else set()
        match self._state:
            case State.READY:
                found = {Command.START, Command.TERMINATE}
            case State.RUNNING:
                found = {Command.STOP, Command.TERMINATE, *take}
            case State.STOPPING:
                found = {Command.TERMINATE}
                if self._pending is State.PAUSED:
                    found |= take
            case State.PAUSED:
                found = {Command.RESUME, Command.TERMINATE, *take}
            case State.AWAITING_APPROVAL:
                # Resume declines what was asked and lets the run continue;
                # it never approves.
                found = {
                    Command.APPROVE,
                    Command.RESUME,
                    Command.REJECT,
                    Command.TERMINATE,
                    *take,
                }
            case State.HUMAN_CONTROL:
                found = {Command.RESUME, Command.TERMINATE}
            case State.CHECKING:
                found = {Command.TERMINATE}
            case _:
                found = set()
        return frozenset(found)

    def _why_not(self, command: Command) -> str:
        state = self._state
        if state in FINAL:
            return "the run has ended"
        if self._episode is not None and self._episode.expired:
            return "that request timed out and the run is ending"
        if command is Command.APPROVE:
            if self._episode is not None and self._episode.ask is Ask.PERSON:
                return (
                    "this request cannot be approved; take control and do the "
                    "step, then resume, or terminate"
                )
            return "nothing is waiting for approval"
        if command is Command.TAKE_CONTROL and not self._where:
            return "nobody can see this session, so it cannot be handed over"
        return f"{command.value} is not available while the run is {state.value}"

    def _primary(self) -> Button:
        match self._state:
            case State.READY:
                return Button("Start", Command.START, enabled=True)
            case State.RUNNING:
                return Button("Stop", Command.STOP, enabled=True)
            case State.STOPPING:
                return Button("Stopping...", None, enabled=False)
            case State.PAUSED | State.HUMAN_CONTROL:
                return Button("Resume", Command.RESUME, enabled=True)
            case State.AWAITING_APPROVAL:
                return Button("Resume", Command.RESUME, enabled=True)
            case State.CHECKING:
                return Button("Checking...", None, enabled=False)
            case _:
                return Button(self._state.value.capitalize(), None, enabled=False)

    def _status(self) -> Status:
        budget = self._budget
        return Status(
            run=self._run,
            mode=self._mode,
            purpose=self._purpose,
            revision=self._revision,
            state=self._state,
            owner=owner_of(self._state),
            step=self._step(),
            primary=self._primary(),
            offered=self._offered(),
            remaining_s=budget.remaining_seconds() if budget is not None else 0.0,
            paused_s=budget.paused_seconds() if budget is not None else 0.0,
            notice=self._notice,
            settling=self._settling(),
            offer=self._offer(),
            ending=self._ending,
            interrupted=self._interruption(),
        )

    def _settling(self) -> str:
        flight = self._in_flight
        if flight is None or self._state not in {State.STOPPING, State.TERMINATED}:
            return ""
        what = flight.kind.value if flight.kind is not None else "an observation"
        return f"{what} dispatched at step {flight.step} is settling"

    def _interruption(self) -> str:
        episode = self._episode
        if episode is None:
            return ""
        return explain(episode.interrupted, episode.dispatched)

    def _offer(self) -> Offer | None:
        episode = self._episode
        if episode is None:
            return None
        request = episode.request
        return Offer(
            intervention=episode.intervention,
            ask=episode.ask,
            trigger=episode.trigger,
            reason=episode.reason,
            context=(request.capability or request.goal) if request else self._context,
            proposal=(describe(request.action) or confirming(request))
            if request
            else "",
            route=request.route if request else "",
            session=self._where,
            timeout_s=request.timeout_s if request else self._timeout(),
            left_s=max(0.0, episode.deadline - self._clock()),
        )

    def _step(self) -> int:
        return self._budget.steps if self._budget is not None else 0

    def _timeout(self) -> float:
        if self._profile is None:
            return 0.0
        return float(self._profile.escalation.handoff_timeout_s)

    # Owner thread: episodes.

    def _open(
        self,
        ask: Ask,
        trigger: Trigger | None,
        request: InterventionRequest | None,
        reason: str,
        target: State,
    ) -> _Episode:
        """Begin a pause: stop the clock, note what was interrupted, and wait."""
        timeout = request.timeout_s if request is not None else self._timeout()
        with self._lock:
            self._count += 1
            flight = self._interrupted_by
            if request is not None and request.action is not None:
                interrupted = Interruption.PENDING
            elif flight is not None:
                interrupted = settled(flight.outcome, flight.detail)
            elif self._discarded:
                interrupted = Interruption.PENDING
            else:
                interrupted = Interruption.NONE
            now = self._clock()
            episode = _Episode(
                intervention=f"iv-{self._count}",
                ask=ask,
                trigger=trigger,
                reason=reason,
                request=request,
                started=now,
                deadline=now + timeout,
                interrupted=interrupted,
                dispatched=flight,
                taken=target is State.HUMAN_CONTROL,
            )
            episode.events.extend(self._early)
            self._early.clear()
            if self._state is State.STOPPING and self._pending is State.HUMAN_CONTROL:
                # A takeover sent after the caller looked still wins.
                target = State.HUMAN_CONTROL
                episode.taken = True
            self._episode = episode
            self._hand_back = None
            if self._state is not State.TERMINATED:
                self._state = target
            self._pending = None
            self._notice = ""
            self._revision += 1
            self._outbox.append(
                Paused(
                    step=self._step(),
                    intervention=episode.intervention,
                    ask=ask,
                    trigger=trigger,
                )
            )
        if self._budget is not None:
            self._budget.pause()
        if self._seat is not None:
            self._seat.transfer(Owner.OPERATOR, episode.intervention)
            episode.baseline = self._seat.probe()
        self._flush()
        self._display()
        return episode

    def _serve(self, checks: Check | None) -> Handoff:
        """Wait for the operator while processing changes on the owner thread.

        The session is handed to the person before anything else is looked
        at, so a resume that arrives in the same slice as a takeover still
        finds the session recorded as theirs.
        """
        while True:
            with self._lock:
                state = self._state
                episode = self._episode
            if episode is None or state is State.TERMINATED:
                return self._close(HandoffOutcome.TERMINATED)
            if (
                episode.taken
                and not episode.released
                and state
                in {
                    State.HUMAN_CONTROL,
                    State.CHECKING,
                }
            ):
                self._release(episode)
            if state is State.STOPPING:
                self._restop()
                continue
            if state is State.CHECKING:
                self._check(episode, checks)
                continue
            if state is State.RUNNING:
                return self._close(episode.answer or HandoffOutcome.RESUMED)
            if self._clock() >= episode.deadline:
                with self._lock:
                    episode.expired = True
                return self._close(HandoffOutcome.TIMED_OUT)
            self._tick(self._interval)

    def _restop(self) -> None:
        """Apply a stop or takeover sent after an answer, before it was used."""
        with self._lock:
            episode = self._episode
            if episode is None or self._state is not State.STOPPING:
                return
            self._state = self._pending or State.PAUSED
            self._pending = None
            episode.answer = None
            episode.taken = episode.taken or self._state is State.HUMAN_CONTROL
            self._revision += 1

    def _release(self, episode: _Episode) -> None:
        """Hand the person the session."""
        episode.released = True
        episode.taken = True
        if self._seat is not None:
            self._seat.transfer(Owner.HUMAN, episode.intervention)
        with self._lock:
            self._outbox.append(
                Transferred(
                    step=self._step(),
                    intervention=episode.intervention,
                    owner=Owner.HUMAN,
                )
            )

    def _check(self, episode: _Episode, checks: Check | None) -> bool:
        """Validate a returned session before automation resumes.

        Check known dialog state first because page scripts wait for dialogs.
        Flush the recording, protect edited fields, and compare the session
        with its state before the pause. New manual input during these checks
        leaves ownership with the person until they return the session again.
        """
        dialog = self._open_dialog(episode)
        if dialog is not None:
            return self._decide(episode, dialog)
        # An answer leaves the screen to the proposal it was about, so the
        # automation keeps what it saw; if a person changed the screen, the
        # loop drops the approval and looks again anyway.
        revoke = episode.answer is None
        if self._seat is not None:
            recording = self._seat.hand_back(revoke=revoke)
        else:
            recording = Recording(gaps=(Gap(GapKind.NOT_RECORDED),))
        self._take(recording.events)
        episode.gaps.extend(recording.gaps)
        seen = len(episode.events)
        if episode.answer is not None:
            # An approval or a rejection: the person answered about the
            # proposal and handed nothing back, so only what they may have
            # typed is checked. Where the pages are is the loop's to judge.
            found = self._protection(recording)
        else:
            found = self._inspect(episode, recording)
            if found is None and checks is not None:
                try:
                    found = checks()
                except ControlChanged:
                    found = (
                        CheckFailure.STEP_CONDITION,
                        "control changed while the session was being checked",
                    )
        self._tick(0)
        late = [event for event in episode.events[seen:] if event.changes]
        if found is None and late and self._seat is not None:
            # Anything that arrived after the flush was not protected by it,
            # and the checks above ran on a session the person was still
            # changing. They keep it until they hand it back finished.
            recording = self._seat.hand_back(revoke=revoke)
            self._take(recording.events)
            found = _UNPROTECTED if recording.unprotected else _IN_USE
        return self._decide(episode, found)

    def _open_dialog(self, episode: _Episode) -> tuple[CheckFailure, str] | None:
        if self._seat is None:
            return None
        for page in self._seat.probe().pages:
            if page.dialog:
                if episode.answer is not None:
                    return None
                return (
                    CheckFailure.DIALOG_OPEN,
                    f"a dialog is still open on {page.page}; answer it, then resume",
                )
        return None

    def _decide(
        self, episode: _Episode, found: tuple[CheckFailure, str] | None
    ) -> bool:
        failure, notice = found if found is not None else (None, "")
        passed = failure is None
        with self._lock:
            if self._state is State.CHECKING:
                self._state = State.RUNNING if passed else State.HUMAN_CONTROL
            if not passed:
                episode.released = False
                episode.taken = True
                episode.answer = None
            self._notice = notice
            self._revision += 1
            self._outbox.append(
                HandedBack(
                    step=self._step(),
                    intervention=episode.intervention,
                    passed=passed,
                    failure=failure,
                    events=len(episode.events),
                    gaps=sum(gap.count for gap in episode.gaps),
                    interrupted=episode.interrupted,
                )
            )
        return passed

    def _inspect(
        self, episode: _Episode, recording: Recording
    ) -> tuple[CheckFailure, str] | None:
        """Compare the session now with the session when the pause began."""
        if self._seat is None:
            return None
        probe = self._settled_probe(self._seat)
        if not probe.pages:
            return CheckFailure.SESSION_CLOSED, "every page of the session is closed"
        for page in probe.pages:
            if page.dialog:
                return (
                    CheckFailure.DIALOG_OPEN,
                    f"a dialog is still open on {page.page}; answer it, then resume",
                )
            if (
                self._profile is not None
                and policy.route_for(self._profile, page.location) is None
            ):
                return (
                    CheckFailure.OFF_ROUTE,
                    (
                        f"{page.page} is outside the profile's routes; return it "
                        "to a permitted page, then resume"
                    ),
                )
        if not all(page.ready for page in probe.pages):
            return (
                CheckFailure.NOT_READY,
                "a page is still loading; resume when it has loaded",
            )
        if probe.protection_lost or recording.unprotected:
            return _UNPROTECTED
        return self._signed_in(episode, probe)

    def _signed_in(
        self, episode: _Episode, probe: SessionProbe
    ) -> tuple[CheckFailure, str] | None:
        """Check for new or unresolved sign-in requests.

        A credential prompt that appeared while nobody used the session is
        the application asking again, usually because the session expired.
        After a request to sign in, the prompts that were showing when the run
        asked must be gone. Anything else a person did is theirs to judge;
        they can see a prompt the run cannot tell from an ordinary form.
        """
        baseline = episode.baseline
        if baseline is None:
            return None
        grew = probe.credential_prompts > baseline.credential_prompts
        if grew and not any(event.input for event in episode.events):
            return (
                CheckFailure.SIGN_IN_REQUIRED,
                "The application requires another sign-in. Sign in, then resume.",
            )
        if (
            episode.trigger is Trigger.AUTHENTICATION_REQUIRED
            and baseline.credential_prompts
            and probe.credential_prompts >= baseline.credential_prompts
        ):
            return (
                CheckFailure.SIGN_IN_REQUIRED,
                ("The sign-in prompt is still visible. Sign in, then resume."),
            )
        return None

    def _protection(self, recording: Recording) -> tuple[CheckFailure, str] | None:
        if recording.unprotected:
            return _UNPROTECTED
        if self._seat is not None and self._seat.probe().protection_lost:
            return _UNPROTECTED
        return None

    def _settled_probe(self, seat: Seat) -> SessionProbe:
        probe = seat.probe()
        for _ in range(READY_TRIES):
            if all(page.ready for page in probe.pages):
                break
            seat.idle(0.2)
            probe = seat.probe()
        return probe

    def _escalate(
        self, episode: _Episode, request: InterventionRequest, checks: Check | None
    ) -> Handoff:
        """Answer an intervention through a synchronous operator channel.

        A person who hands the session back is checked like any other. When
        the check fails the channel is asked again, with the reason, because
        a synchronous channel cannot be told to keep waiting any other way.
        """
        escalator = self._escalator
        if escalator is None:
            return self._close(HandoffOutcome.TIMED_OUT)
        while True:
            handoff = escalator.request(request)
            self._record_answer(episode, handoff)
            with self._lock:
                terminated = self._state is State.TERMINATED
                if not terminated and handoff.outcome is HandoffOutcome.RESUMED:
                    episode.taken = True
                    self._state = State.CHECKING
                elif not terminated and handoff.outcome in {
                    HandoffOutcome.APPROVED,
                    HandoffOutcome.REJECTED,
                }:
                    self._state = State.RUNNING
                    episode.answer = handoff.outcome
            if terminated:
                return self._close(HandoffOutcome.TERMINATED)
            if handoff.outcome is not HandoffOutcome.RESUMED:
                return self._close(handoff.outcome)
            if self._check(episode, checks):
                return self._close(HandoffOutcome.RESUMED)
            request = dataclasses.replace(request, reason=self.status().notice)

    def _record_answer(self, episode: _Episode, handoff: Handoff) -> None:
        command = {
            HandoffOutcome.APPROVED: Command.APPROVE,
            HandoffOutcome.REJECTED: Command.REJECT,
            HandoffOutcome.RESUMED: Command.RESUME,
        }.get(handoff.outcome)
        with self._lock:
            episode.note = handoff.operator_note
            if command is None:
                return
            self._outbox.append(
                Commanded(
                    step=self._step(),
                    command=command,
                    verdict=Verdict.ACCEPTED,
                    via=Via.ESCALATOR,
                    intervention=episode.intervention,
                )
            )

    def _close(self, outcome: HandoffOutcome) -> Handoff:
        """End the wait and restore automation ownership if the run continues."""
        episode = self._episode
        if episode is None:
            return Handoff(outcome)
        self._tick(0)
        if outcome in {HandoffOutcome.TERMINATED, HandoffOutcome.TIMED_OUT}:
            if outcome is HandoffOutcome.TIMED_OUT:
                with self._lock:
                    episode.expired = True
                    self._notice = "the request timed out; the run is ending"
                    self._revision += 1
                self._display()
            return Handoff(outcome, episode.note, episode.intervention, episode.changed)
        if self._seat is not None:
            self._tick(0)
            self._seat.transfer(Owner.AUTOMATION, episode.intervention)
        handoff = Handoff(outcome, episode.note, episode.intervention, episode.changed)
        segment = self._keep(episode)
        if self._budget is not None:
            self._budget.unpause()
        with self._lock:
            if episode.released or episode.taken:
                self._outbox.append(
                    Transferred(
                        step=self._step(),
                        intervention=episode.intervention,
                        owner=Owner.AUTOMATION,
                    )
                )
            self._episode = None
            self._discarded = False
            self._interrupted_by = None
            self._revision += 1
            if outcome in {
                HandoffOutcome.RESUMED,
                HandoffOutcome.APPROVED,
                HandoffOutcome.REJECTED,
            }:
                self._hand_back = HandBack(
                    intervention=episode.intervention,
                    segment=segment,
                    interrupted=episode.interrupted,
                    dispatched=episode.dispatched,
                    notices=self._notices(segment),
                )
        self._tick(0)
        return handoff

    def _keep(self, episode: _Episode) -> ManualSegment:
        """Retain an episode's events as a typed manual segment."""
        events = tuple(episode.events)
        gaps = list(episode.gaps) + list(gaps_in(events))
        if episode.dropped:
            gaps.append(Gap(GapKind.EVENTS_DROPPED, episode.dropped))
        profile = self._profile
        held = self._restrictions()
        steps = tuple(
            ManualStep(
                event,
                classify(
                    event,
                    profile,
                    ask=episode.ask,
                    trigger=episode.trigger,
                    restrictions=held,
                ),
            )
            for event in events
            if profile is not None
        )
        request = episode.request
        segment = ManualSegment(
            run=self._run,
            intervention=episode.intervention,
            mode=self._mode,
            ask=episode.ask,
            trigger=episode.trigger,
            taken=episode.taken,
            steps=steps,
            gaps=merge(tuple(gaps)),
            interrupted=episode.interrupted,
            proposal=request.action if request is not None else None,
        )
        self._segments.append(segment)
        return segment

    def _notices(self, segment: ManualSegment) -> tuple[str, ...]:
        found = []
        if segment.taken and not segment.steps:
            found.append(
                "No manual input was recorded while a person held the session."
            )
        if any(step.event.kind is ManualKind.EDIT for step in segment.steps):
            found.append(
                "Fields edited by a person are protected as secrets. "
                "Their values are withheld."
            )
        if segment.gaps:
            kinds = ", ".join(gap.kind.value for gap in segment.gaps)
            found.append(f"the recording of what the person did has gaps: {kinds}")
        return tuple(found)

    # Owner thread: the slices every wait is made of.

    def _tick(self, seconds: float) -> None:
        """Deliver session events, record what a person did, and redraw."""
        if self._seat is not None:
            self._seat.idle(seconds)
            self._take(self._seat.activity())
        elif seconds:
            time.sleep(seconds)
        self._flush()
        self._display()

    def _take(self, events: tuple[ManualEvent, ...]) -> None:
        """Retain and journal manual input within the intervention's limit.

        A person who changes the session while automation owns it has made
        the ownership ambiguous. The run stops at its next boundary, as if
        the operator had pressed Stop, and waits for them to take control or
        resume: one owner at a time.
        """
        if not events:
            return
        with self._lock:
            if self._state is State.RUNNING and any(e.changes for e in events):
                self._state = State.STOPPING
                self._pending = State.PAUSED
                self._interrupted_by = self._in_flight
                self._cause = "a person used the session while automation held it"
                self._notice = f"{self._cause}; take control or resume"
                self._revision += 1
            episode = self._episode
            intervention = episode.intervention if episode is not None else ""
            for event in events:
                if episode is not None:
                    if len(episode.events) >= MAX_EVENTS:
                        episode.dropped += 1
                        continue
                    episode.events.append(event)
                elif self._state is State.STOPPING and len(self._early) < MAX_EVENTS:
                    # Kept for the pause this input is about to open.
                    self._early.append(event)
                self._outbox.append(
                    ManualAction(
                        step=self._step(),
                        intervention=intervention,
                        sequence=event.sequence,
                        kind=event.kind,
                        control=event.control,
                        route=event.route,
                        frame=event.position,
                        secret=event.secret,
                        owned=event.owned,
                        detail=event.detail,
                    )
                )

    def _flush(self) -> None:
        journal = self._journal
        if journal is None:
            return
        with self._lock:
            events, self._outbox = self._outbox, []
        for event in events:
            journal.record(event)

    def _display(self) -> None:
        """Refresh changed status immediately and otherwise once per second.

        The step and remaining time can change without a new revision.
        Refresh requests also reopen a closed or reloaded control window.
        Each channel redraws only the parts that changed.
        """
        with self._lock:
            changed = self._revision != self._shown
            now = self._clock()
            beat = now - self._retried >= 1.0
            if not changed and not beat:
                return
            self._shown = self._revision
            if beat:
                self._retried = now
            status = self._status()
        for channel in list(self._channels):
            try:
                channel.show(status)
            except EvidenceError:
                raise
            except Exception:  # a closed channel must not end the run
                self._channels.remove(channel)

    # Owner thread: operations the guard wraps.

    @contextlib.contextmanager
    def _operating(
        self, kind: ActionKind | None
    ) -> Iterator[list[tuple[Outcome, str]]]:
        settled: list[tuple[Outcome, str]] = []
        with self._lock:
            flight = Dispatch(self._step(), kind)
            self._in_flight = flight
        try:
            yield settled
        finally:
            with self._lock:
                outcome, detail = settled[0] if settled else (Outcome.UNCERTAIN, "")
                done = dataclasses.replace(flight, outcome=outcome, detail=detail)
                self._in_flight = None
                if self._interrupted_by is flight:
                    self._interrupted_by = done

    def _heed(self) -> None:
        """Process manual input since the last observation before an operation.

        A person who used the session while automation held it stops the run
        here, so no observation or action starts on a screen they changed.
        """
        if self._seat is not None:
            self._take(self._seat.activity())
            self._flush()

    def _may(self, kind: ActionKind | None) -> bool:
        with self._lock:
            if self._state is State.RUNNING:
                return True
            return self._state is State.CHECKING and (
                kind is None or kind in CHECK_READS
            )


class _Guarded:
    """A surface that refuses automation unless the run's control permits it.

    Every discovery and replay observation or action passes through this
    guard. None reaches the adapter while the run is stopped, waiting, owned
    by a person, or ended. Refusal raises an exception so callers cannot
    mistake it for an application response.
    """

    __slots__ = ("_control", "_surface")

    def __init__(self, surface: Surface, control: Control) -> None:
        self._surface = surface
        self._control = control

    def location(self) -> str:
        """Return the wrapped surface's location."""
        return self._surface.location()

    def capabilities(self) -> Capabilities:
        """Return the wrapped surface's capabilities."""
        return self._surface.capabilities()

    def pages(self) -> tuple[PageInfo, ...]:
        """Report session windows without taking an observation."""
        return self._surface.pages()

    def observe(self, request: ObservationRequest) -> Observation:
        """Observe only while automation owns the run."""
        self._control._heed()
        if not self._control._may(None):
            raise ControlChanged
        with self._control._operating(None) as settled:
            observation = self._surface.observe(request)
            settled.append((Outcome.OK, ""))
            return observation

    def control_at(self, target: ScreenTarget) -> str:
        """Name the control under a screenshot point, while automation owns the run."""
        self._control._heed()
        if not self._control._may(None):
            raise ControlChanged
        return control_at(self._surface, target)

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        """Act only while automation owns the run."""
        self._control._heed()
        if not self._control._may(action.kind):
            raise ControlChanged
        with self._control._operating(action.kind) as settled:
            result = self._surface.act(action, expect=expect)
            settled.append((result.outcome, result.detail or ""))
            return result

    def operation_binding(
        self,
        action: Action,
        observation: Observation | None,
        node: AxNode | None = None,
    ) -> BoundOperation | None:
        """Attest observed identity only while automation owns the session."""
        self._control._heed()
        if not self._control._may(None):
            raise ControlChanged
        return operation_binding(self._surface, action, observation, node)


def confirming(request: InterventionRequest) -> str:
    """Describe a result for human confirmation, including live values.

    Show every output value and each part the run could not verify.

    Examples
    --------
    >>> from computeruse.escalation import Trigger
    >>> request = InterventionRequest(
    ...     Trigger.UNVERIFIED_RESULT, "", "p", 4, "/", "", 60.0,
    ...     outputs={"order_total": "12.00"}, unverified=("total is not tied",))
    >>> confirming(request)
    'result order_total = 12.00; not verified: total is not tied'
    """
    parts = []
    if request.outputs:
        shown = ", ".join(
            f"{name} = {value}" for name, value in request.outputs.items()
        )
        parts.append(f"result {shown}")
    if request.unverified:
        parts.append("not verified: " + "; ".join(request.unverified))
    return "; ".join(parts)


def describe(action: Action | None) -> str:
    """Describe a proposal for the operator, including live values.

    Examples
    --------
    >>> describe(Action(ActionKind.CLICK, AxLocator("button", "Send"), effect="pay"))
    "click button 'Send' (effect: pay)"
    >>> describe(None)
    ''
    """
    if action is None:
        return ""
    match action.target:
        case AxLocator(role=role, name=name):
            target = f" {role} {name!r}"
        case DomLocator(tag=tag, attribute=attribute, value=value):
            target = f" {tag} with {attribute.value} {value!r}"
        case VisualAnchor():
            target = " a painted control"
        case ScreenTarget():
            target = " a point on the screenshot"
        case _:
            target = ""
    effect = f" (effect: {action.effect})" if action.effect else ""
    return f"{action.kind.value}{target}{effect}"


_IN_USE = (
    CheckFailure.IN_USE,
    ("The session is still in use. Resume when you have finished."),
)

_UNPROTECTED = (
    CheckFailure.PROTECTION_LOST,
    (
        "An edited or secret field left the page or could not be protected. "
        "The run cannot establish where its value went. Reload or leave the "
        "page before resuming. Resume alone cannot restore protection."
    ),
)


def settled(outcome: Outcome | None, detail: str = "") -> Interruption:
    """Classify an operation's delivery state from ``outcome``.

    A refusal means nothing was sent, with one exception: a surface that sent
    the input and then undid where it led reports ``SENT_ELSEWHERE``, and
    whatever the input did in the application may have happened.

    Examples
    --------
    >>> settled(Outcome.OK)
    <Interruption.PERFORMED: 'performed'>
    >>> settled(Outcome.STALE)
    <Interruption.PENDING: 'pending'>
    >>> settled(Outcome.BLOCKED, SENT_ELSEWHERE)
    <Interruption.UNCERTAIN: 'uncertain'>
    >>> settled(None)
    <Interruption.UNCERTAIN: 'uncertain'>
    """
    if outcome is Outcome.OK:
        return Interruption.PERFORMED
    if outcome in {None, Outcome.UNCERTAIN, Outcome.HANDOFF, Outcome.SURFACE_ERROR}:
        return Interruption.UNCERTAIN
    if detail == SENT_ELSEWHERE:
        return Interruption.UNCERTAIN
    return Interruption.PENDING


def explain(interrupted: Interruption, dispatched: Dispatch | None) -> str:
    """Describe an interruption for an operator or the model. Live text.

    Examples
    --------
    >>> explain(Interruption.PERFORMED, Dispatch(4, ActionKind.CLICK, Outcome.OK))
    'the click sent at step 4 was performed; its effect is not confirmed'
    """
    what = (
        f"the {dispatched.kind.value if dispatched.kind else 'observation'} "
        f"sent at step {dispatched.step}"
        if dispatched is not None
        else "the step that was waiting"
    )
    match interrupted:
        case Interruption.PENDING:
            return f"{what} was not performed by the run"
        case Interruption.PERFORMED:
            return f"{what} was performed; its effect is not confirmed"
        case Interruption.UNCERTAIN:
            return f"{what} may or may not have taken effect"
        case _:
            return ""
