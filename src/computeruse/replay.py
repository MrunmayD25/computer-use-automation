"""Deterministic replay of a validated capability without a model.

``replay`` walks a capability graph against a live surface. At each action,
it checks the screen, preconditions, and record, resolves the target
again, builds the action from this invocation's inputs, and passes it
through ``policy.evaluate``, the same gate discovery uses. Nothing here can
replace that gate. Saved restrictions and the node's own approval
requirement can only add a need for a person. Every look at the surface
passes ``policy.evaluate_observation`` first, and every internal read passes
``policy.evaluate``, each with the risk the profile declares for it.

A target belongs to the screen it was recorded on. The replay resolves it
only while the surface is on that route template, so a control with the same
name on another permitted screen is never used in its place. An observation
that did not cover the whole screen is unknown: it proves neither that a
control is absent nor that one match is the only one.

After every operation the replay reads the page and follows the one
transition whose conditions hold. No match and several matches are different
situations: no match waits a bounded number of looks and then asks a person;
several matches are reported as an ambiguous state and never resolved by
taking the first.

Every operation has a delivery state: not dispatched, dispatched with an
uncertain result, or verified complete. A timeout or an unacknowledged input
is uncertain, not failed. The replay never sends an uncertain mutating
operation again, and never moves on while one is uncertain. It looks for
positive evidence that the operation took effect: the node's verification,
its conditional transitions, or, for typing and selecting, the field showing
the value. When nothing can establish the result, it asks a person.

A person can approve one proposed action, perform a step themselves, or hand
the session back. Handing back is not approval. An approval is bound to one
request in one run and to the exact proposal and screen it was given for.
After it, the replay looks again and checks everything again, and it
dispatches only if the proposal it would send now is the one approved. An
approval is never remembered: the next attempt, and the next replay, ask
again. Nothing a person does during a replay changes the capability or the
profile.

The replay log records anonymous node and route references, action types,
delivery states, and reason codes. Inputs, outputs, page text,
and the proposal shown to a person stay in the live result and the operator
channel.

This module imports no model client and no decider, and a test enforces it.
"""

from __future__ import annotations

import dataclasses
import urllib.parse
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from computeruse import coverage, matching, operations, policy, reading
from computeruse.actions import (
    DIALOG_ACTIONS,
    Action,
    AxLocator,
    AxNode,
    DomLocator,
    Expectation,
    Observation,
    ObservationMode,
    ObservationProvenance,
    ObservationRequest,
    Operation,
    Outcome,
    PageState,
    RecordEvidence,
    ScreenTarget,
    SecretRef,
    Target,
    TargetForm,
    unbound,
)
from computeruse.budget import Budget, Clock
from computeruse.capability import (
    MUTATING,
    Absent,
    ActionNode,
    Approval,
    AtRoute,
    Bound,
    Capability,
    CapabilityError,
    CheckNode,
    Completed,
    Condition,
    DialogOpen,
    Edge,
    FocusTarget,
    HelpReason,
    HumanNode,
    Issue,
    Match,
    Node,
    Present,
    Purpose,
    RadioTarget,
    RecordSpec,
    Ref,
    RefKind,
    ResultConfirmationNode,
    ResultKind,
    ResultNode,
    Review,
    Shows,
    StructuralTarget,
    Template,
    TextTarget,
    ValueIs,
    VisualTarget,
    accepts_inputs,
    aims_by_pixels,
    branches,
    check_profile,
    effective_budgets,
    loads,
    preparation_review,
    record_label,
    strictest,
    text_matches,
    transitions_of,
    validate,
)
from computeruse.control import NOT_SENT, SENT_ELSEWHERE, Control, ControlChanged
from computeruse.diagnostics import (
    ROLES,
    CheckShape,
    FailureEvidence,
    Snapshot,
    snapshot,
)
from computeruse.escalation import HandoffOutcome, Mode
from computeruse.journal import Journal, MemoryJournal
from computeruse.manual import ManualKind
from computeruse.policy import Allowed, Denial, Denied
from computeruse.profile import ActionKind, Limit, Profile, Risk
from computeruse.reading import Line, LocalReader, Reader
from computeruse.retarget import (
    Found,
    Scene,
    Scenes,
    SurfaceScenes,
    locator_for,
    radio_spot,
    resolve,
    shown,
    spot,
    text_line,
    text_spot,
)
from computeruse.surface import Surface, SurfaceError, operation_binding


class Status(StrEnum):
    """The final status of a replay.

    ``OUTCOME`` is an expected business result the capability declares, such
    as a member not found. ``RECOVERY_EXHAUSTED`` means a bounded recovery
    ran out of traversals, or one node ran out of attempts. ``NEEDS_HELP``
    means the replay stopped where a person must decide what happens next,
    and ``HELP_TIMED_OUT`` means nobody answered in time. ``STOPPED`` means
    the caller asked the replay to stop before its first change, and it did;
    only the comparison that confirms website text asks for that.
    """

    SUCCEEDED = "succeeded"
    OUTCOME = "outcome"
    STOPPED = "stopped"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    NEEDS_HELP = "needs_help"
    HELP_TIMED_OUT = "help_timed_out"
    FAILED = "failed"
    TERMINATED = "terminated"


class Reason(StrEnum):
    """A log-safe reason for ending a replay or asking a person."""

    COMPLETED = "completed"
    BUSINESS_OUTCOME = "business_outcome"
    DECLARED_FAILURE = "declared_failure"
    INVALID_ARTIFACT = "invalid_artifact"
    UNREVIEWED = "unreviewed"
    PROFILE_CONFLICT = "profile_conflict"
    INVALID_INPUT = "invalid_input"
    INCOMPATIBLE_APPLICATION = "incompatible_application"
    SELECTION_UNCONFIRMED = "selection_unconfirmed"
    POLICY_DENIED = "policy_denied"
    RESTRICTION_DENIED = "restriction_denied"
    BLOCKED_BY_SURFACE = "blocked_by_surface"
    UNSUPPORTED_BY_SURFACE = "unsupported_by_surface"
    SURFACE_HANDOFF = "surface_handoff"
    NEW_WINDOW = "new_window"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_INVALIDATED = "approval_invalidated"
    RECORD_EVIDENCE_REQUIRED = "record_evidence_required"
    RECORD_MISMATCH = "record_mismatch"
    OFF_ROUTE = "off_route"
    OBSERVATION_INCOMPLETE = "observation_incomplete"
    PRECONDITION_UNMET = "precondition_unmet"
    PLANNED_STEP = "planned_step"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_AMBIGUOUS = "target_ambiguous"
    TARGET_UNAVAILABLE = "target_unavailable"
    NO_TRANSITION = "no_transition"
    AMBIGUOUS_STATE = "ambiguous_state"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    DELIVERY_UNCERTAIN = "delivery_uncertain"
    VERIFICATION_FAILED = "verification_failed"
    MISSING_VALUE = "missing_value"
    OUTPUT_INVALID = "output_invalid"
    READING_UNCONFIRMED = "reading_unconfirmed"
    COMPLETION_CHECK_FAILED = "completion_check_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SURFACE_FAILED = "surface_failed"
    APPROVAL_REJECTED = "approval_rejected"
    OPERATOR_TERMINATED = "operator_terminated"
    HELP_TIMED_OUT = "help_timed_out"
    HELP_UNRESOLVED = "help_unresolved"
    STALE_ANSWER = "stale_answer"
    STOPPED_BEFORE_CHANGE = "stopped_before_change"


REASON_TEXT = {
    Reason.COMPLETED: "every step ran and the success checks held",
    Reason.BUSINESS_OUTCOME: "the application gave an answer the capability knows",
    Reason.DECLARED_FAILURE: "the replay reached an ending declared as a failure",
    Reason.INVALID_ARTIFACT: "the capability file is not valid",
    Reason.UNREVIEWED: "the capability is a draft and needs review",
    Reason.PROFILE_CONFLICT: "the capability needs something the policy withholds",
    Reason.INVALID_INPUT: "an input is missing or has the wrong type",
    Reason.INCOMPATIBLE_APPLICATION: "the first screen is not the recorded one",
    Reason.SELECTION_UNCONFIRMED: "a required choice could not be confirmed",
    Reason.POLICY_DENIED: "the policy refused a step",
    Reason.RESTRICTION_DENIED: "a learned restriction refused a step",
    Reason.BLOCKED_BY_SURFACE: "the browser refused the input",
    Reason.UNSUPPORTED_BY_SURFACE: "the browser cannot perform this step",
    Reason.SURFACE_HANDOFF: "the browser handed the step to a person",
    Reason.NEW_WINDOW: "a new window opened",
    Reason.APPROVAL_REQUIRED: "a risky step needed approval",
    Reason.APPROVAL_INVALIDATED: "the page changed after the approval was given",
    Reason.RECORD_EVIDENCE_REQUIRED: "the page could not prove which record the "
    "step acts on",
    Reason.RECORD_MISMATCH: "the page showed a different record than intended",
    Reason.OFF_ROUTE: "the browser left the pages the capability works on",
    Reason.OBSERVATION_INCOMPLETE: "the page could not be read completely",
    Reason.PRECONDITION_UNMET: "the page was not in the state this step needs",
    Reason.PLANNED_STEP: "the capability leaves this step to a person",
    Reason.TARGET_NOT_FOUND: "the control this step acts on is not on the page",
    Reason.TARGET_AMBIGUOUS: "more than one control matches this step",
    Reason.TARGET_UNAVAILABLE: "the control is on the page but cannot be used",
    Reason.NO_TRANSITION: "no saved path matches the page after this step",
    Reason.AMBIGUOUS_STATE: "more than one saved path matches the page",
    Reason.RECOVERY_EXHAUSTED: "the bounded recovery ran out",
    Reason.DELIVERY_UNCERTAIN: "a change may have been sent, and the page "
    "cannot confirm it",
    Reason.VERIFICATION_FAILED: "the page did not show what this step expects",
    Reason.MISSING_VALUE: "a value the step needs was not available",
    Reason.OUTPUT_INVALID: "an output read from the page has the wrong type",
    Reason.READING_UNCONFIRMED: "two readings of a painted value disagreed",
    Reason.COMPLETION_CHECK_FAILED: "the final success checks did not hold",
    Reason.BUDGET_EXHAUSTED: "the time or step limit ran out",
    Reason.SURFACE_FAILED: "the browser stopped working",
    Reason.APPROVAL_REJECTED: "a person declined the risky step",
    Reason.OPERATOR_TERMINATED: "a person pressed Terminate",
    Reason.HELP_TIMED_OUT: "nobody answered the request for help in time",
    Reason.HELP_UNRESOLVED: "a person's help did not settle the step",
    Reason.STALE_ANSWER: "an answer came for a request that had moved on",
    Reason.STOPPED_BEFORE_CHANGE: "it stopped before the first change, as asked",
}
"""Plain-language reasons for replay endings."""


class Delivery(StrEnum):
    """The known delivery state of one operation attempt."""

    NOT_DISPATCHED = "not_dispatched"
    UNCERTAIN = "uncertain"
    COMPLETED = "completed"


class Actor(StrEnum):
    """Who performed a completed operation."""

    AUTOMATION = "automation"
    PERSON = "person"


class Ask(StrEnum):
    """The response type requested from a person.

    ``APPROVAL`` asks whether one proposed action, or one look at the
    surface, may run once. ``PERSON`` asks a person to act in the session,
    because approving would not supply what is missing.
    """

    APPROVAL = "approval"
    PERSON = "person"


class Answer(StrEnum):
    """A person's response to a request.

    ``APPROVED`` lets the proposed action run once, if every check still
    holds and the proposal is unchanged. ``RESUMED`` hands the session back.
    The replay checks fresh evidence after a hand-back, which is never an
    approval or proof that the person completed the step.
    """

    APPROVED = "approved"
    RESUMED = "resumed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    TERMINATED = "terminated"


@dataclasses.dataclass(frozen=True, slots=True)
class HelpRequest:
    """One request for a person. Live data for the operator channel.

    ``run`` is this replay's random id, and ``intervention`` is unique within
    it and starts with it, so no two requests in any two replays share one.
    ``proposal`` is the action an approval would run, with its values, so
    this request is shown and never persisted. ``planned`` names the reason a
    capability's human node gives, when the request comes from one.
    ``question`` says, in words, the operation, the record, and what the
    person must settle, with this invocation's values (rule 16). It is shown
    on the operator channel only and never written to a journal.
    """

    intervention: str
    run: str
    ask: Ask
    reason: Reason
    capability: str
    node: str
    route: str
    timeout_s: float
    proposal: Action | None = None
    planned: HelpReason | None = None
    step: int = 0
    expected: tuple[str, ...] = ()
    observed: Snapshot | None = None
    question: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class HelpAnswer:
    """A person's answer. ``evidence`` names the run history of what they did.

    ``intervention`` must be the id of the request being answered. An answer
    naming any other request, from this replay or another, is refused, and
    so is a second answer to a request already answered.
    """

    answer: Answer
    intervention: str
    evidence: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayStarted:
    """A replay began. Carries names from the artifact and the profile, no values."""

    run: str
    capability: str
    version: int
    profile_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class NodeVisited:
    """The replay entered a node."""

    step: int
    node: str
    kind: str


@dataclasses.dataclass(frozen=True, slots=True)
class Gated:
    """The gate judged an action, a read, or a look the replay was about to take."""

    node: str
    kind: ActionKind
    route: str
    denial: Denial | None
    approval: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Dispatched:
    """An action reached the surface, and what is known about its effect."""

    node: str
    kind: ActionKind
    outcome: Outcome | None
    delivery: Delivery


@dataclasses.dataclass(frozen=True, slots=True)
class Transitioned:
    """The replay followed the transition at ``edge`` of ``node``."""

    node: str
    edge: int
    to: str


@dataclasses.dataclass(frozen=True, slots=True)
class Helped:
    """A person was asked, and answered."""

    node: str
    ask: Ask
    reason: Reason
    answer: Answer


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayEnded:
    """The replay stopped."""

    status: Status
    reason: Reason
    node: str
    steps: int


type ReplayEvent = (
    ReplayStarted
    | NodeVisited
    | Gated
    | Dispatched
    | Transitioned
    | Helped
    | ReplayEnded
    | FailureEvidence
)


class ReplayLog(Protocol):
    """Receives replay events. Each one is safe to persist."""

    def record(self, event: ReplayEvent) -> None:
        """Append one event."""
        ...


class MemoryReplayLog:
    """Keeps replay events in a list."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[ReplayEvent] = []

    def record(self, event: ReplayEvent) -> None:
        """Append one event."""
        self.events.append(event)


@dataclasses.dataclass(frozen=True, slots=True)
class Attempt:
    """The result of one visit to an action node."""

    node: str
    visit: int
    delivery: Delivery
    actor: Actor | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class HistoryEntry:
    """A person's replay answer and the location of its evidence."""

    node: str
    intervention: str
    ask: Ask
    reason: Reason
    answer: Answer
    evidence: str


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayResult:
    """The caller-facing replay result, including live values.

    ``outputs`` are the typed outputs of a success or business outcome, and
    they are member data: they are returned here and never logged.
    ``issues`` explains a rejected artifact by code and position only.
    ``run`` is the replay's random id, which every help request carried.
    """

    status: Status
    reason: Reason
    node: str
    outcome: str = ""
    outputs: Mapping[str, str] = MappingProxyType({})
    attempts: tuple[Attempt, ...] = ()
    history: tuple[HistoryEntry, ...] = ()
    issues: tuple[Issue, ...] = ()
    steps: int = 0
    active_s: float = 0.0
    paused_s: float = 0.0
    run: str = ""
    diagnostic: FailureEvidence | None = None
    differed: frozenset[str] = frozenset()
    seen: frozenset[str] = frozenset()


def replay(
    capability: Capability | str,
    inputs: Mapping[str, str],
    *,
    profile: Profile,
    surface: Surface,
    control: Control,
    journal: Journal | None = None,
    log: ReplayLog,
    clock: Clock,
    scenes: Scenes | None = None,
    sleep: Callable[[float], None] | None = None,
    accept_draft: bool = False,
    probing: frozenset[str] = frozenset(),
    reader: Reader | None = None,
    stop_before_change: bool = False,
) -> ReplayResult:
    """Replay a capability against a live surface, deterministically.

    Parameters
    ----------
    capability
        A parsed capability, validated again here, or its JSON text.
    inputs
        This invocation's input values, by declared name.
    profile
        The operator's policy. It is the only source of permission.
    surface
        The live session, through the same protocol discovery uses.
    control
        The unused replay control that owns this session and its operator channels.
    log
        Receives persisted diagnostics. Never given a value.
    clock
        Monotonic seconds, shared with the budget.
    scenes
        Supplies fresh captures for visual targets, each taken only after the
        gate allows a visual look. Without it, a visual target is unavailable
        and the replay asks a person.
    sleep
        Waits between bounded looks for a state that has not appeared yet.
    accept_draft
        Replay a capability no person has reviewed. For tests and review.
    reader
        Reads the lines of text a capture shows, for a text target on a
        canvas. The default runs local recognition, loaded on first use.
    probing
        Unconfirmed texts a comparison replay is reading again for a second
        record. A step's check that uses one and fails does not stop the
        step, provided its other checks hold and at least one check did. Its
        texts are returned in ``ReplayResult.differed`` as read differently.

    Returns
    -------
    ReplayResult
        The ending, with outputs only when the completion checks held.
    """
    if not isinstance(control, Control):
        raise TypeError("replay requires a Control")
    control.validate_start(Mode.REPLAY)
    return _Replay(
        profile=profile,
        surface=surface,
        log=log,
        clock=clock,
        scenes=scenes,
        sleep=sleep or (lambda _: None),
        control=control,
        journal=journal or MemoryJournal(),
        probing=probing,
        reader=reader or LocalReader(),
        stop_before_change=stop_before_change,
    ).run(capability, inputs, accept_draft=accept_draft)


MAX_SHAPES = 32
"""The most check shapes one failure's evidence keeps."""

SETTLE_LOOKS = 3
"""How many looks, a pause apart, an uncertain step gets before a person is
asked, so a page still loading after the step is not mistaken for none."""


class _End(Exception):  # noqa: N818  an ending, not a fault
    def __init__(
        self, status: Status, reason: Reason, issues: tuple[Issue, ...] = ()
    ) -> None:
        super().__init__(reason.value)
        self.status = status
        self.reason = reason
        self.issues = issues


@dataclasses.dataclass(frozen=True, slots=True)
class _View:
    """One look at the surface: where it is, and what a gated observation saw.

    ``route`` is the allow-route template of ``location``, or None when the
    location is outside the profile. ``observation`` is None when the gate
    refused the look, a person declined it, or the page moved during it.
    """

    location: str
    route: str | None
    observation: Observation | None
    moved: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _Aim:
    found: Found
    target: Target | None = None
    node: AxNode | None = None
    dialog: str | None = None


type _Key = tuple[object, ...]


@dataclasses.dataclass(slots=True)
class _Tries:
    """One visit's counters, and the approval it holds, if any.

    ``helped`` says a person was just asked, so the next look first checks
    whether fresh evidence shows they completed the step. ``held`` counts
    the hand-backs the run had when
    the visit began, so only what a person did during this visit counts.
    """

    held: int
    misses: int = 0
    settles: int = 0
    rounds: int = 0
    helped: bool = False
    approved: _Key | None = None


class _Replay:
    def __init__(
        self,
        *,
        profile: Profile,
        surface: Surface,
        log: ReplayLog,
        clock: Clock,
        scenes: Scenes | None,
        sleep: Callable[[float], None],
        control: Control,
        journal: Journal,
        probing: frozenset[str] = frozenset(),
        reader: Reader | None = None,
        stop_before_change: bool = False,
    ) -> None:
        from computeruse.replay_control import ControlledOperator

        self.profile = profile
        self.stop_before_change = stop_before_change
        self.reader = reader or LocalReader()
        self.probing = probing
        self.differed: set[str] = set()
        # Probed texts this replay met on the page: a control it resolved
        # and acted on, or a check that held. Only these can be confirmed.
        self.seen: set[str] = set()
        self.read_approvals: set[tuple[str, int]] = set()
        self.surface = control.guard(surface)
        self.control = control
        self.journal = journal
        self.handbacks = 0
        self.confirmation_segment: int | None = None
        self.windows: set[str] = set()
        self.operator = ControlledOperator(control, profile)
        self.log = log
        self.clock = clock
        self.scenes = scenes or SurfaceScenes(self.surface)
        self.sleep = sleep
        self.started = clock()
        self.run_id = uuid.uuid4().hex
        self.budget: Budget | None = None
        self.capability: Capability | None = None
        self.current: Node | None = None
        self.inputs: dict[str, str] = {}
        self.values: dict[tuple[RefKind, str], str] = {}
        self.completed: set[str] = set()
        self.traversals: dict[tuple[str, int], int] = {}
        self.visits: dict[str, int] = {}
        self.attempts: list[Attempt] = []
        self.sent_operations: dict[str, tuple[ActionNode, Actor | None]] = {}
        self.history: list[HistoryEntry] = []
        self.answered: set[str] = set()
        self.asked = 0
        self.node = ""
        self.tries = 1
        self.observed: Snapshot | None = None
        self.expected: tuple[str, ...] = ()
        self.checked: tuple[CheckShape, ...] = ()

    @property
    def cap(self) -> Capability:
        assert self.capability is not None  # noqa: S101  set before any node runs
        return self.capability

    def run(
        self,
        document: Capability | str,
        inputs: Mapping[str, str],
        *,
        accept_draft: bool,
    ) -> ReplayResult:
        try:
            self._load(document, inputs, accept_draft=accept_draft)
            return self._walk()
        except _End as ending:
            return self._end(ending.status, ending.reason, issues=ending.issues)
        except SurfaceError:
            return self._end(Status.FAILED, Reason.SURFACE_FAILED)
        except ControlChanged:
            try:
                self._hold()
            except _End as ending:
                return self._end(ending.status, ending.reason, issues=ending.issues)
            return self._end(Status.NEEDS_HELP, Reason.HELP_UNRESOLVED)
        finally:
            self.control.finish("failed")

    # Load the artifact, profile, inputs, and live application in that order.

    def _load(
        self,
        document: Capability | str,
        inputs: Mapping[str, str],
        *,
        accept_draft: bool,
    ) -> None:
        try:
            capability = loads(document) if isinstance(document, str) else document
        except CapabilityError as error:
            raise _End(Status.FAILED, Reason.INVALID_ARTIFACT, error.issues) from None
        found = validate(capability)
        if found:
            raise _End(Status.FAILED, Reason.INVALID_ARTIFACT, found)
        self.capability = capability
        self.current = capability.node(capability.entry)
        self.node = capability.entry
        budgets = effective_budgets(capability, self.profile)
        self.budget = Budget(budgets, self.clock)
        self.tries = max(1, budgets.max_retries_per_step)
        self.log.record(
            ReplayStarted(
                self.run_id,
                capability.capability_id,
                capability.version,
                self.profile.profile_id,
            )
        )
        if capability.provenance.review is Review.DRAFT and not accept_draft:
            raise _End(Status.FAILED, Reason.UNREVIEWED)
        conflicts = check_profile(capability, self.profile)
        if conflicts:
            raise _End(Status.FAILED, Reason.PROFILE_CONFLICT, conflicts)
        self.inputs = self._accept(inputs)
        if not self.control.begin(
            profile=self.profile,
            budget=self._budget(),
            journal=self.journal,
            context=capability.label,
        ):
            raise _End(Status.TERMINATED, Reason.OPERATOR_TERMINATED)
        self.windows = {page.page_id for page in self.surface.pages()}
        for attempt in range(self.cap.limits.max_settle_observations):
            view = self._view()
            if view.route == capability.application.entry_route and self._holds(
                capability.application.markers, view
            ):
                return
            if (
                view.observation is None and not view.moved
            ) or view.route != capability.application.entry_route:
                break
            if attempt + 1 < self.cap.limits.max_settle_observations:
                self._pause()
        raise _End(Status.FAILED, Reason.INCOMPATIBLE_APPLICATION)

    def _accept(self, inputs: Mapping[str, str]) -> dict[str, str]:
        if not accepts_inputs(self.cap, inputs):
            raise _End(Status.FAILED, Reason.INVALID_INPUT)
        return dict(inputs)

    # Walk the capability graph.

    def _walk(self) -> ReplayResult:
        node_id = self.cap.entry
        budget = self._budget()
        while True:
            if budget.exhausted() is not None:
                raise _End(Status.FAILED, Reason.BUDGET_EXHAUSTED)
            budget.charge_step()
            node = self.cap.node(node_id)
            self.current = node
            self.node = node.node_id
            self.log.record(NodeVisited(budget.steps, node.node_id, type(node).TAG))
            # Evidence covers every set of checks this node tried.
            self.checked = ()
            match node:
                case ResultNode():
                    return self._result(node)
                case ActionNode():
                    node_id = self._action(node)
                case HumanNode():
                    node_id = self._human(node)
                case CheckNode():
                    if node.delay_ms:
                        self.sleep(node.delay_ms / 1000)
                    node_id = self._next(node)

    def _budget(self) -> Budget:
        assert self.budget is not None  # noqa: S101  set while loading
        return self.budget

    def _live(self) -> None:
        """Stop before anything else reaches the surface once the budget is spent.

        Called before every observation, capture, read, and dispatch, so a
        look that used the last of the time cannot be followed by input.
        Time a person spends answering is not counted.
        """
        if self.control.halted():
            self._hold()
        if self.confirmation_segment is not None and any(
            segment.taken
            or not segment.complete
            or any(step.event.changes for step in segment.steps)
            for segment in self.control.segments[self.confirmation_segment :]
        ):
            raise _End(Status.NEEDS_HELP, Reason.APPROVAL_INVALIDATED)
        if self._budget().exhausted(starting_step=False) is not None:
            raise _End(Status.FAILED, Reason.BUDGET_EXHAUSTED)

    def _hold(self) -> None:
        handoff = self.control.hold()
        if handoff.outcome is HandoffOutcome.TERMINATED:
            raise _End(Status.TERMINATED, Reason.OPERATOR_TERMINATED)
        if handoff.outcome is not HandoffOutcome.RESUMED:
            raise _End(Status.HELP_TIMED_OUT, Reason.HELP_TIMED_OUT)
        self.handbacks += 1

    def _rounds(self) -> int:
        """Bound the attempts one visit to one action node may make."""
        limits = self.cap.limits
        return (limits.max_settle_observations + self.tries + 2) * (
            limits.max_help_requests + 1
        )

    def _action(self, node: ActionNode) -> str:
        visit = self.visits.get(node.node_id, 0) + 1
        self.visits[node.node_id] = visit
        for sent, actor in self.sent_operations.values():
            same = bool(node.business and node.business == sent.business)
            unknown = (
                bool(node.business or sent.business)
                and not (node.business and sent.business)
                and node.route == sent.route
                and node.kind in MUTATING
            )
            if same or unknown:
                return self._already_sent(node, visit, actor)
        return self._action_visit(node, visit)

    def _action_visit(self, node: ActionNode, visit: int) -> str:
        """Resolve, authorize, and deliver one bounded visit to an action node."""
        tries = _Tries(held=self._held())
        while True:
            tries.rounds += 1
            if tries.rounds > self._rounds():
                raise _End(Status.RECOVERY_EXHAUSTED, Reason.RECOVERY_EXHAUSTED)
            view = self._view()
            if tries.helped and self._established(node, view) is True:
                self._attempt(node, visit, Delivery.COMPLETED, Actor.PERSON)
                return self._next(node)
            tries.helped = False
            problem = self._precondition(node, view)
            if problem is not None:
                limit = self.cap.limits.max_settle_observations
                tries.settles = self._retry(node, tries, tries.settles, limit, problem)
                continue
            aim = self._aim(node, view)
            if aim.found is not Found.FOUND:
                lost = _MISSED[aim.found]
                tries.misses = self._retry(node, tries, tries.misses, self.tries, lost)
                continue
            if self._touched(node, aim, tries.held):
                # A person may have performed this step while they held the
                # session. It is treated as sent: its result or a person
                # settles it, and the replay never sends it again.
                self._attempt(node, visit, Delivery.UNCERTAIN, Actor.PERSON)
                if self._established(node, self._view()) is not True:
                    self._settle_uncertain(node, visit)
                else:
                    self._attempt(node, visit, Delivery.COMPLETED, Actor.PERSON)
                return self._next(node)
            action = self._build(node, aim.target, view, aim.node)
            if not self._cleared(node, action, view, aim, tries):
                continue
            expect = dataclasses.replace(
                self._expect(view, aim), binding=node.binding, business=node.business
            )
            delivery = self._dispatch(node, visit, action, expect)
            if delivery is None:
                self._ask(node, tries, Reason.SURFACE_HANDOFF)
                continue
            if delivery is Delivery.NOT_DISPATCHED:
                self._attempt(node, visit, delivery, None)
                tries.misses = self._retry(
                    node,
                    tries,
                    tries.misses,
                    self.tries,
                    Reason.TARGET_NOT_FOUND,
                    wait=False,
                )
                continue
            self._attempt(node, visit, delivery, Actor.AUTOMATION)
            self._witness(node)
            if delivery is Delivery.UNCERTAIN:
                self._settle_uncertain(node, visit)
            return self._next(node)

    def _already_sent(self, node: ActionNode, visit: int, actor: Actor | None) -> str:
        """Settle an alternate route to a change without delivering it again."""
        if not self._verified(node):
            self._settle_uncertain(node, visit)
        else:
            self._attempt(node, visit, Delivery.COMPLETED, actor)
        return self._next(node)

    def _retry(
        self,
        node: ActionNode,
        tries: _Tries,
        count: int,
        limit: int,
        reason: Reason,
        *,
        wait: bool = True,
    ) -> int:
        """Count one more miss, and ask a person once the bound is reached.

        Any approval in hand is dropped, because whatever it was given for
        no longer holds. Returns the new count.
        """
        tries.approved = None
        count += 1
        if count < limit:
            if wait:
                self._pause()
            return count
        self._ask(node, tries, reason)
        return 0

    def _ask(self, node: ActionNode, tries: _Tries, reason: Reason) -> None:
        self._help(node, Ask.PERSON, reason)
        tries.helped = True

    def _cleared(
        self, node: ActionNode, action: Action, view: _View, aim: _Aim, tries: _Tries
    ) -> bool:
        """Report whether the action may be dispatched now.

        An approval clears exactly the proposal it was given for, on the
        screen it was given on, once. Anything that differs asks again.
        """
        try:
            needs = self._gate(node, action, view, aim.node)
        except ControlChanged:
            # Identity attestation is guarded too. No input has been sent
            # here, so a hand-back starts authorization again. Preserve
            # the earlier manual boundary to detect a person's delivery.
            self._hold()
            tries.approved, tries.helped = None, True
            return False
        if needs is None:
            tries.approved, tries.helped = None, True
            return False
        if not needs:
            return True
        if self.stop_before_change:
            # A step that needs approval is a change; this replay ends before it.
            raise _End(Status.STOPPED, Reason.STOPPED_BEFORE_CHANGE)
        key = _key(node, action, view, aim)
        if tries.approved == key:
            tries.approved = None
            return True
        reason = (
            Reason.APPROVAL_REQUIRED
            if tries.approved is None
            else Reason.APPROVAL_INVALIDATED
        )
        answer = self._help(node, Ask.APPROVAL, reason, action)
        tries.approved = key if answer is Answer.APPROVED else None
        tries.helped = True
        return False

    def _held(self) -> int:
        """Count the hand-backs so far, each of which kept what a person did."""
        return len(self.control.segments)

    def _touched(self, node: ActionNode, aim: _Aim, held: int) -> bool:
        """Report whether a person may have performed ``node`` since ``held``.

        A person's change counts when its kind could perform the node's
        action and it was not on a different named control. A control the
        replay names some way a person's input cannot be compared with
        counts too, because a step sent twice cannot be taken back.
        """
        kinds = _BY_HAND.get(node.kind, frozenset())
        target = aim.target
        for segment in self.control.segments[held:]:
            for step in segment.steps:
                event = step.event
                if not event.changes or event.kind not in kinds:
                    continue
                elsewhere = (
                    isinstance(target, AxLocator)
                    and event.target is not None
                    and (event.target.role, event.target.name)
                    != (target.role, target.name)
                )
                if not elsewhere:
                    return True
        return False

    def _established(self, node: ActionNode, view: _View) -> bool | None:
        """Report whether the page shows this node's operation took effect.

        Returns None when nothing can show it: a node with no verification,
        no conditional transitions, and no readable field to compare. For a
        node that names its record, evidence counts only together with the
        record it belongs to, so a confirmation for another member is not
        this member's confirmation. A record that cannot be seen is False,
        not None, so no claim stands in for it.
        """
        result = self._result_shown(node, view)
        if node.record is None or result is False:
            return result
        if not self._record_shown(node, view):
            # A step that changes nothing and left its page, such as opening a
            # record's page from a list, proved its record before the click;
            # the page it opened cannot show the row it came from.
            return (
                result is True
                and node.approval is Approval.NONE
                and view.route is not None
                and view.route != node.route
            )
        return result

    def _result_shown(self, node: ActionNode, view: _View) -> bool | None:
        if node.verify and self.probing:
            return self._probed(node.verify, view)
        if node.verify:
            return self._holds(node.verify, view)
        if branches(node):
            return any(self._holds(edge.when, view) for edge in node.transitions)
        if node.kind in {ActionKind.TYPE, ActionKind.SELECT}:
            return self._field_shows(node, view)
        return None

    def _probed(self, conditions: tuple[Condition, ...], view: _View) -> bool:
        """Judge a step's checks while a comparison reads unconfirmed texts again.

        A check that uses no probed text must hold. A check that uses one may
        fail, because its text can be the first record's data, such as a
        receipt number, and its texts are then kept as read differently. The
        step still needs at least one check that held.
        """
        soft = tuple(c for c in conditions if self._texts(c) & self.probing)
        firm = tuple(c for c in conditions if c not in soft)
        if firm and not self._holds(firm, view):
            return False
        missed = [condition for condition in soft if not self._one(condition, view)]
        if not firm and len(missed) == len(soft):
            return False
        for condition in missed:
            self.differed |= self._texts(condition) & self.probing
        return True

    def _witness(self, node: ActionNode) -> None:
        """Count the texts of a step this replay just sent as met on the page.

        Its target resolved on the live page for this record, and the input
        went. Its effect label is the run's own name for the operation, never
        page text, so a comparison can only show that the operation ran
        again under it.
        """
        if not self.probing:
            return
        texts: set[str] = set()
        if node.target is not None:
            texts |= self._texts(Present(node.target))
        if node.value is not None and node.value.kind is RefKind.CONSTANT:
            texts.add(node.value.value)
        if node.effect:
            texts.add(node.effect)
        if node.extraction is not None:
            texts |= {
                part.value
                for part in node.extraction.prefix + node.extraction.suffix
                if part.kind is RefKind.CONSTANT
            }
        self.seen |= texts & self.probing

    def _texts(self, condition: Condition) -> set[str]:
        """Return the constant texts a check compares or finds its control by."""
        refs: list[Ref | None] = []
        targets: list[str] = []
        match condition:
            case Present(target=target) | Absent(target=target):
                targets.append(target)
            case Shows(target=target, value=value):
                targets.append(target)
                refs.append(value)
            case ValueIs(value=value, expected=expected):
                refs.extend((value, expected))
            case DialogOpen(message=message):
                refs.append(message)
            case _:
                pass
        texts: set[str] = set()
        for target_id in targets:
            try:
                found = self.cap.target(target_id)
            except KeyError:
                continue
            if isinstance(found, StructuralTarget):
                refs.extend((found.name, found.value))
                if found.scope is not None:
                    refs.append(found.scope.name)
                texts.update(found.frame)
            elif isinstance(found, TextTarget):
                refs.append(found.text)
                texts.update(found.frame)
        texts.update(
            ref.value
            for ref in refs
            if ref is not None and ref.kind is RefKind.CONSTANT
        )
        return texts

    def _record_shown(self, node: ActionNode, view: _View) -> bool:
        """Report whether the result shows exactly the record this invocation named.

        ``result_record`` says where, when the artifact declares it.
        Otherwise the record's own source must still resolve, on its own
        route and in a complete observation, where the result is checked.
        """
        spec = node.result_record or node.record
        if spec is None:
            return True
        source = self.cap.target(spec.source)
        if not isinstance(source, StructuralTarget):
            return False
        try:
            resolved = resolve(source, view.observation, self._text, view.route)
            value = self._text(spec.value)
        except KeyError:
            return False
        if resolved.node is None or resolved.node.value is not None:
            return False
        text = shown(resolved.node)
        return text is not None and record_label(text, value, spec) is not None

    def _field_shows(self, node: ActionNode, view: _View) -> bool | None:
        """Compare a typed or selected field with the value this node entered."""
        if node.target is None or node.value is None:
            return None
        if node.value.kind is RefKind.SECRET:
            return None
        target = self.cap.target(node.target)
        if not isinstance(target, StructuralTarget):
            return None
        resolved = resolve(target, view.observation, self._value, view.route)
        if resolved.node is None:
            return False
        if node.kind is ActionKind.SELECT and resolved.node.options:
            options = resolved.node.options
            wanted = self._value(node.value)
            # A choice saved by ``contains`` holds its input among other words,
            # as ``_option_holding`` chose it; any other choice is the option's
            # stored value or its whole label.
            matches = [
                option
                for option in options
                if option.value == wanted
                or option.label == wanted
                or (
                    node.value_match is Match.CONTAINS
                    and matching.contains(option.label, wanted, once=True)
                )
            ]
            return (
                resolved.node.options_complete
                and not resolved.node.secret
                and len(matches) == 1
                and matches[0].selected
                and sum(option.selected for option in options) == 1
            )
        text = shown(resolved.node)
        if text is None:
            return None
        return text == self._value(node.value)

    def _precondition(self, node: ActionNode, view: _View) -> Reason | None:
        if view.route != node.route:
            return Reason.OFF_ROUTE
        if not self._holds(node.requires, view):
            return Reason.PRECONDITION_UNMET
        if node.record is None:
            return None
        source = self.cap.target(node.record.source)
        if not isinstance(source, StructuralTarget):
            return Reason.RECORD_EVIDENCE_REQUIRED
        resolved = resolve(source, view.observation, self._value, view.route)
        if resolved.found in {Found.INCOMPLETE, Found.UNAVAILABLE}:
            return Reason.OBSERVATION_INCOMPLETE
        text = None if resolved.node is None else shown(resolved.node)
        value = self._value(node.record.value)
        if text is None or record_label(text, value, node.record) is None:
            return Reason.RECORD_MISMATCH
        return None

    def _aim(self, node: ActionNode, view: _View) -> _Aim:
        if node.kind in DIALOG_ACTIONS:
            dialog = view.observation.dialog if view.observation else None
            if dialog is None:
                return _Aim(Found.NOT_FOUND)
            return _Aim(Found.FOUND, dialog=dialog.dialog_id)
        if node.target is None:
            return _Aim(Found.FOUND)
        target = self.cap.target(node.target)
        if view.route != target.route:
            return _Aim(Found.OFF_ROUTE)
        if isinstance(target, VisualTarget):
            spotted = spot(self._template(target), self._capture(target))
            return _Aim(spotted.found, spotted.target)
        if isinstance(target, RadioTarget):
            spotted = radio_spot(target, self._capture(target), self.reader, self._text)
            return _Aim(spotted.found, spotted.target)
        if isinstance(target, TextTarget):
            scene = self._capture(target)
            spotted = text_spot(target, scene, self._lines(scene), self._text)
            return _Aim(spotted.found, spotted.target)
        if isinstance(target, FocusTarget):
            scene = self._capture(target)
            if scene is None:
                return _Aim(Found.UNAVAILABLE)
            return _Aim(Found.FOUND, ScreenTarget(scene.capture_id))
        resolved = resolve(target, view.observation, self._value, view.route)
        return _Aim(resolved.found, resolved.locator, resolved.node)

    def _template(self, target: VisualTarget) -> Template:
        return next(
            item for item in self.cap.templates if item.template_id == target.template
        )

    def _build(
        self,
        node: ActionNode,
        target: Target | None,
        view: _View,
        aimed: AxNode | None = None,
    ) -> Action:
        value: str | SecretRef | None = None
        if node.value is not None:
            value = (
                SecretRef(node.value.name)
                if node.value.kind is RefKind.SECRET
                else self._value(node.value)
            )
        if node.value_match is Match.CONTAINS and isinstance(value, str):
            value = _option_holding(aimed, value)
        evidence = self._evidence(node.record, view)
        try:
            return Action(
                node.kind,
                target,
                value,
                self._destination(node),
                evidence,
                node.effect,
            )
        except ValueError:
            raise _End(Status.FAILED, Reason.INVALID_ARTIFACT) from None

    def _evidence(
        self, record: RecordSpec | None, view: _View
    ) -> RecordEvidence | None:
        """Build the record evidence an action carries to the gate and surface.

        A ``CONTAINS`` link takes its source's whole text from ``view`` and
        splits it around the value, so the surface checks the exact text it
        shows. The text lives only in this action.
        """
        if record is None:
            return None
        source = self.cap.target(record.source)
        if not isinstance(source, StructuralTarget):
            return None
        value = self._value(record.value)
        if record.match is Match.EQUALS and source.match is Match.EQUALS:
            locator = locator_for(source, self._value)
            return RecordEvidence(
                locator, value, record.relation, record.prefix, record.suffix
            )
        resolved = resolve(source, view.observation, self._value, view.route)
        text = None if resolved.node is None else shown(resolved.node)
        label = None if text is None else record_label(text, value, record)
        if resolved.locator is None or label is None:
            return None
        return RecordEvidence(resolved.locator, value, record.relation, *label)

    def _destination(self, node: ActionNode) -> str | None:
        destination = node.destination
        if destination is None:
            return None
        scope = self.profile.scope
        filled = {param.name: self._value(param.value) for param in destination.params}
        segments = []
        for segment in destination.route.split("/")[1:]:
            if not segment.startswith(":"):
                segments.append(segment)
                continue
            value = filled[segment[1:]]
            if value in {"", ".", ".."}:
                raise _End(Status.FAILED, Reason.MISSING_VALUE)
            segments.append(urllib.parse.quote(value, safe=""))
        query = urllib.parse.urlencode(
            [(param.name, self._value(param.value)) for param in destination.query]
        )
        fragment = (
            ""
            if destination.fragment is None
            else urllib.parse.quote(self._value(destination.fragment), safe="")
        )
        return urllib.parse.urlunsplit(
            (
                urllib.parse.urlsplit(destination.origin or scope.origin).scheme,
                urllib.parse.urlsplit(destination.origin or scope.origin).netloc,
                "/" + "/".join(segments),
                query,
                fragment,
            )
        )

    def _gate(
        self, node: ActionNode, action: Action, view: _View, aimed: AxNode | None
    ) -> bool | None:
        """Pass the action through the gate and the saved restrictions.

        Returns whether a person must approve it, or None when a person was
        asked to act instead, so the caller looks at the page again. Raises
        when the replay must stop. ``aimed`` is the control the target found,
        which says whether the action submits a form.
        """
        verdict = policy.evaluate(action, self.profile, view.location)
        binding = operation_binding(self.surface, action, view.observation, aimed)
        if (binding.name if binding else "") != node.binding or (
            binding.business if binding else ""
        ) != node.business:
            raise _End(Status.FAILED, Reason.POLICY_DENIED)
        physical = Operation.of(action, view.route or "", node=aimed)
        physical = dataclasses.replace(
            physical, binding=node.binding, business=node.business
        )
        binding_limit = operations.limit(self.profile, physical)
        if binding_limit is Limit.DENY:
            raise _End(Status.FAILED, Reason.POLICY_DENIED)
        limit = strictest(
            self.cap.restrictions, node, pixel=aims_by_pixels(self.cap, node)
        )
        submission = (
            policy.submission_limit(
                self.profile, Operation.of(action, verdict.route, node=aimed)
            )
            if isinstance(verdict, Allowed)
            else None
        )
        if submission is Limit.DENY:
            raise _End(Status.FAILED, Reason.POLICY_DENIED)
        needs = isinstance(verdict, Allowed) and (
            verdict.risk is Risk.RISKY
            or node.approval is Approval.EACH_RUN
            or limit is Limit.RISKY
            or submission is Limit.RISKY
            or binding_limit is Limit.RISKY
        )
        denial = verdict.reason if isinstance(verdict, Denied) else None
        self.log.record(Gated(node.node_id, node.kind, _route(verdict), denial, needs))
        if denial is Denial.RECORD_EVIDENCE_MISSING:
            self._help(node, Ask.PERSON, Reason.RECORD_EVIDENCE_REQUIRED)
            return None
        if denial is not None:
            raise _End(Status.FAILED, Reason.POLICY_DENIED)
        if limit is Limit.DENY:
            raise _End(Status.FAILED, Reason.RESTRICTION_DENIED)
        if not operations.selections_ready(self.profile, physical, aimed):
            self._help(node, Ask.PERSON, Reason.SELECTION_UNCONFIRMED)
            return None
        self._supported(action)
        if (
            action.kind is ActionKind.NAVIGATE
            and self._budget().exhausted(starting_step=False, navigating=True)
            is not None
        ):
            raise _End(Status.FAILED, Reason.BUDGET_EXHAUSTED)
        return needs

    def _supported(self, action: Action) -> None:
        """Refuse an action without a valid, explicit adapter declaration."""
        try:
            offered = self.surface.capabilities()
        except (AttributeError, TypeError):
            raise _End(Status.FAILED, Reason.UNSUPPORTED_BY_SURFACE) from None
        if not isinstance(offered, Mapping) or any(
            not isinstance(kind, ActionKind)
            or not isinstance(forms, frozenset)
            or any(not isinstance(form, TargetForm) for form in forms)
            for kind, forms in offered.items()
        ):
            raise _End(Status.FAILED, Reason.UNSUPPORTED_BY_SURFACE)
        if _form(action.target) not in offered.get(action.kind, frozenset()):
            raise _End(Status.FAILED, Reason.UNSUPPORTED_BY_SURFACE)

    def _expect(self, view: _View, aim: _Aim) -> Expectation:
        page = (
            view.observation.page_state
            if view.observation is not None
            else PageState(view.location)
        )
        if aim.dialog is not None:
            return Expectation(
                page, dialog=aim.dialog, windows=tuple(sorted(self.windows))
            )
        if aim.node is not None:
            return Expectation(
                page,
                aim.node.context,
                control=aim.node.control,
                strict=True,
                windows=tuple(sorted(self.windows)),
                submits_as=aim.node.submits_as,
                enter_as=aim.node.enter_as,
            )
        return Expectation(page, windows=tuple(sorted(self.windows)))

    def _dispatch(
        self, node: ActionNode, visit: int, action: Action, expect: Expectation
    ) -> Delivery | None:
        """Send one action, and return its delivery, or None if the surface handed off.

        A surface failure during a mutating action leaves the operation
        uncertain. The replay records that and stops, because the session is
        gone and nothing can be checked.
        """
        revision = self.handbacks
        self._live()
        if revision != self.handbacks:
            return Delivery.NOT_DISPATCHED
        try:
            result = self.surface.act(action, expect=expect)
        except ControlChanged:
            self._hold()
            return Delivery.NOT_DISPATCHED
        except SurfaceError:
            if node.kind in MUTATING:
                self._attempt(node, visit, Delivery.UNCERTAIN, Actor.AUTOMATION)
                self.log.record(
                    Dispatched(node.node_id, node.kind, None, Delivery.UNCERTAIN)
                )
            raise
        if result.outcome is Outcome.HANDOFF:
            self.log.record(
                Dispatched(
                    node.node_id, node.kind, result.outcome, Delivery.NOT_DISPATCHED
                )
            )
            return None
        if result.outcome is Outcome.BLOCKED and result.detail == NOT_SENT:
            # The surface took a Stop at the last moment and sent nothing, so
            # the replay waits for the operator like any other pause.
            self.log.record(
                Dispatched(
                    node.node_id, node.kind, result.outcome, Delivery.NOT_DISPATCHED
                )
            )
            self._hold()
            return Delivery.NOT_DISPATCHED
        outcome = result.outcome
        if (
            outcome is Outcome.BLOCKED
            and result.detail == SENT_ELSEWHERE
            and node.kind in MUTATING
        ):
            # The input left, but no permitted page returned. The result is
            # unknown, so the replay never sends the input again. A person or
            # later page evidence must settle it.
            outcome = Outcome.UNCERTAIN
        delivery = self._delivery(node, outcome, result.extracted)
        self.log.record(Dispatched(node.node_id, node.kind, result.outcome, delivery))
        return delivery

    def _delivery(
        self, node: ActionNode, outcome: Outcome, extracted: str | None
    ) -> Delivery:
        if outcome in _NOT_SENT:
            return Delivery.NOT_DISPATCHED
        if outcome is Outcome.BLOCKED:
            raise _End(Status.FAILED, Reason.BLOCKED_BY_SURFACE)
        if outcome is not Outcome.OK and node.kind not in MUTATING:
            return Delivery.NOT_DISPATCHED
        if outcome is Outcome.OK:
            if node.kind is ActionKind.NAVIGATE:
                self._budget().charge_navigation()
            if node.into is not None:
                self._store(node.into, self._taken(node, extracted))
            if not node.verify and not branches(node):
                # The surface acknowledged the input and the node asks for
                # nothing more. Typing and selecting are checked by the next
                # step's conditions. A painted choice can instead carry a
                # mandatory selection review into the later write approval.
                # Validation excludes paths that can bypass that review.
                return Delivery.COMPLETED
        # An acknowledged operation with a check, or a mutating operation
        # that may or may not have happened: only the page can say.
        return Delivery.COMPLETED if self._verified(node) else Delivery.UNCERTAIN

    def _taken(self, node: ActionNode, extracted: str | None) -> str | None:
        """Return what a read took: the whole reading, or a label's value."""
        if node.extraction is not None:
            if extracted is None:
                return None
            prefix = "".join(self._text(part) for part in node.extraction.prefix)
            suffix = "".join(self._text(part) for part in node.extraction.suffix)
            return matching.extract(extracted, prefix, suffix)
        target = self.cap.target(node.target) if node.target else None
        if isinstance(target, TextTarget) and target.label:
            return self._painted_value(target, extracted)
        return extracted

    def _store(self, into: Ref, extracted: str | None) -> None:
        declared = self.cap.field(into)
        if extracted is None or declared is None or not declared.accepts(extracted):
            raise _End(Status.FAILED, Reason.OUTPUT_INVALID)
        self.values[into.kind, into.name] = extracted

    def _verified(self, node: ActionNode) -> bool:
        """Look a bounded number of times for evidence the operation took effect.

        A node with nothing that could establish its result is never
        verified here, whatever its transitions say.
        """
        for _ in range(self.cap.limits.max_settle_observations):
            if self._established(node, self._view()) is True:
                return True
            self._pause()
        return False

    def _settle_uncertain(self, node: ActionNode, visit: int) -> None:
        """Settle an operation that may have happened. Never resend it.

        The page is read first, and a result it shows settles the step with
        nobody asked. Otherwise a person is asked whether it went through: an
        approval says it did, and the replay goes on. Any other answer reads
        the page again, since the person may have finished the step; a page
        that still cannot show it ends the replay, rather than asking again.
        """
        for look in range(SETTLE_LOOKS):
            if look:
                self._pause()
            if self._established(node, self._view()) is True:
                self._attempt(node, visit, Delivery.COMPLETED, Actor.AUTOMATION)
                return
        answer = self._help(node, Ask.APPROVAL, Reason.DELIVERY_UNCERTAIN)
        if answer is Answer.APPROVED:
            self._attempt(node, visit, Delivery.COMPLETED, Actor.PERSON)
            return
        if self._established(node, self._view()) is True:
            self._attempt(node, visit, Delivery.COMPLETED, Actor.PERSON)
            return
        raise _End(Status.NEEDS_HELP, Reason.DELIVERY_UNCERTAIN)

    def _human(self, node: HumanNode) -> str:
        if isinstance(node, ResultConfirmationNode):
            return self._confirm_result(node)
        reason = Reason.PLANNED_STEP
        while True:
            self._help(node, Ask.PERSON, reason, planned=node.reason)
            chosen = self._choose(node)
            if chosen is not None:
                self.completed.add(node.node_id)
                return self._follow(node, chosen)
            reason = Reason.NO_TRANSITION

    def _confirm_result(self, node: ResultConfirmationNode) -> str:
        chosen = self._choose(node)
        if chosen is None:
            raise _End(Status.NEEDS_HELP, Reason.NO_TRANSITION)
        answer = self._help(
            node, Ask.APPROVAL, Reason.PLANNED_STEP, planned=node.reason
        )
        if answer is not Answer.APPROVED:
            raise _End(Status.NEEDS_HELP, Reason.APPROVAL_REJECTED)
        if not self.control.segments[-1].complete:
            raise _End(Status.NEEDS_HELP, Reason.APPROVAL_INVALIDATED)
        self.confirmation_segment = len(self.control.segments)
        if self._choose(node) != chosen:
            raise _End(Status.NEEDS_HELP, Reason.NO_TRANSITION)
        self.completed.add(node.node_id)
        return self._follow(node, chosen)

    def _next(self, node: ActionNode | CheckNode) -> str:
        while True:
            chosen = self._choose(node)
            if chosen is not None:
                return self._follow(node, chosen)
            self._help(node, Ask.PERSON, Reason.NO_TRANSITION)

    def _choose(self, node: ActionNode | CheckNode | HumanNode) -> int | None:
        """Return the one transition whose conditions hold, looking a bounded time.

        Several holding at once is an ambiguous state and ends the replay.
        """
        for _ in range(self.cap.limits.max_settle_observations):
            view = self._view()
            matching = [
                index
                for index, edge in enumerate(node.transitions)
                if self._holds(edge.when, view)
            ]
            if len(matching) > 1:
                raise _End(Status.NEEDS_HELP, Reason.AMBIGUOUS_STATE)
            if matching:
                return matching[0]
            self._pause()
        return None

    def _follow(self, node: Node, index: int) -> str:
        edge: Edge = transitions_of(node)[index]
        key = (node.node_id, index)
        taken = self.traversals.get(key, 0)
        if edge.limit is not None and taken >= edge.limit:
            raise _End(Status.RECOVERY_EXHAUSTED, Reason.RECOVERY_EXHAUSTED)
        self.traversals[key] = taken + 1
        self.log.record(Transitioned(node.node_id, index, edge.to))
        return edge.to

    def _result(self, node: ResultNode) -> ReplayResult:
        for condition in node.checks:
            if not self._check(node, condition):
                raise _End(Status.FAILED, Reason.COMPLETION_CHECK_FAILED)
        if node.result is ResultKind.FAILURE:
            return self._end(Status.FAILED, Reason.DECLARED_FAILURE)
        outputs: dict[str, str] = {}
        for declared in self.cap.outputs:
            value = self.values.get((RefKind.OUTPUT, declared.name))
            if value is None:
                if declared.required and node.result is ResultKind.SUCCESS:
                    raise _End(Status.FAILED, Reason.OUTPUT_INVALID)
                continue
            if not declared.accepts(value):
                raise _End(Status.FAILED, Reason.OUTPUT_INVALID)
            outputs[declared.name] = value
        if node.result is ResultKind.OUTCOME:
            return self._end(
                Status.OUTCOME, Reason.BUSINESS_OUTCOME, node.outcome, outputs
            )
        self._live()
        return self._end(Status.SUCCEEDED, Reason.COMPLETED, outputs=outputs)

    def _check(self, node: ResultNode, condition: Condition) -> bool:
        """Evaluate a completion check, reading a shown value from the live page.

        A read that needs approval is bound to the read as built, the exact
        location, and the observed control and its context. After an
        approval the replay looks again and resolves the target again, and
        reads only if all of that is unchanged. Anything else asks again,
        within the bound on help requests.
        """
        if not isinstance(condition, Shows):
            return self._holds((condition,), self._view())
        target = self.cap.target(condition.target)
        if isinstance(target, TextTarget):
            # A painted line is read from a capture the gate allowed, as any
            # check of a painted screen is.
            return self._holds((condition,), self._view())
        if not isinstance(target, StructuralTarget):
            return False
        approved: _Key | None = None
        while True:
            view = self._view()
            try:
                resolved = resolve(target, view.observation, self._text, view.route)
                expected = self._text(condition.value)
            except KeyError:
                return False
            if resolved.found is not Found.FOUND or resolved.locator is None:
                return False
            read = Action(
                ActionKind.READ,
                resolved.locator,
                evidence=self._evidence(condition.record, view),
            )
            aim = _Aim(Found.FOUND, resolved.locator, resolved.node)
            if self._read_needs_approval(node, read, target, view):
                key = _read_key(read, view, aim)
                if approved != key:
                    reason = (
                        Reason.APPROVAL_REQUIRED
                        if approved is None
                        else Reason.APPROVAL_INVALIDATED
                    )
                    answer = self._help(node, Ask.APPROVAL, reason, read)
                    if answer is not Answer.APPROVED:
                        return False
                    approved = key
                    continue
            self._live()
            result = self.surface.act(read, expect=self._expect(view, aim))
            if result.outcome is not Outcome.OK or result.extracted is None:
                return False
            held = text_matches(
                result.extracted,
                expected,
                condition.match,
                purpose=condition.purpose,
                once=condition.once,
            )
            if held and self.probing:
                self.seen |= self._texts(condition) & self.probing
            return held

    def _read_needs_approval(
        self, node: ResultNode, read: Action, target: StructuralTarget, view: _View
    ) -> bool:
        """Gate a completion read as fully as an action node's read.

        The profile's verdict, its declared risk, record evidence, saved
        restrictions on the target, and the surface's declared support all
        apply. A refusal ends the replay. A safe permitted read needs nobody.
        """
        verdict = policy.evaluate(read, self.profile, view.location)
        probe = _probe(ActionKind.READ, target.route, target.target_id)
        limit = strictest(self.cap.restrictions, probe)
        needs = isinstance(verdict, Allowed) and (
            verdict.risk is Risk.RISKY or limit is Limit.RISKY
        )
        denial = verdict.reason if isinstance(verdict, Denied) else None
        self.log.record(
            Gated(node.node_id, ActionKind.READ, _route(verdict), denial, needs)
        )
        if denial is Denial.RECORD_EVIDENCE_MISSING:
            raise _End(Status.NEEDS_HELP, Reason.RECORD_EVIDENCE_REQUIRED)
        if denial is not None:
            raise _End(Status.FAILED, Reason.POLICY_DENIED)
        if limit is Limit.DENY:
            raise _End(Status.FAILED, Reason.RESTRICTION_DENIED)
        self._supported(read)
        return needs

    def _read_gate(self, target_id: str, view: _View) -> None:
        """Gate a check that compares a control's text as the read it is (rule 19).

        A presence, absence, or route check only looks, and needs nothing but
        the observation it uses. A check that compares text reads it, as
        discovery's checks do: the read grant and any deny apply, and a read
        the profile or a saved restriction makes risky needs a person's
        approval, once per visit to the step for all of its checks. A result
        step's own reads are gated by ``_read_needs_approval``, which asks per
        read as it always has. A refusal ends the replay,
        because a check that cannot be read is not a check that failed.
        """
        target = self.cap.target(target_id)
        node = self.current
        read = Action(ActionKind.READ, ScreenTarget(f"check-{target_id}"))
        verdict = policy.evaluate(read, self.profile, view.location)
        probe = _probe(ActionKind.READ, target.route, target_id)
        limit = strictest(
            self.cap.restrictions,
            probe,
            pixel=not isinstance(target, StructuralTarget),
        )
        if isinstance(verdict, Denied):
            raise _End(Status.FAILED, Reason.POLICY_DENIED)
        if limit is Limit.DENY:
            raise _End(Status.FAILED, Reason.RESTRICTION_DENIED)
        if verdict.risk is not Risk.RISKY and limit is not Limit.RISKY:
            return
        if node is None:
            raise _End(Status.NEEDS_HELP, Reason.APPROVAL_REQUIRED)
        if isinstance(node, ResultNode) and isinstance(target, StructuralTarget):
            # Gated, with its own approval, where the result reads it.
            return
        key = (node.node_id, self.visits.get(node.node_id, 0))
        if key in self.read_approvals:
            return
        answer = self._help(node, Ask.APPROVAL, Reason.APPROVAL_REQUIRED, read)
        if answer is not Answer.APPROVED:
            raise _End(Status.NEEDS_HELP, Reason.APPROVAL_REJECTED)
        self.read_approvals.add(key)

    # Evaluate transition and result conditions.

    def _holds(self, conditions: tuple[Condition, ...], view: _View) -> bool:
        self.expected = tuple(type(condition).TAG for condition in conditions)
        shapes: list[CheckShape] = []
        held = True
        for condition in conditions:
            held = self._one(condition, view)
            shapes.append(self._shape(condition, held))
            if not held:
                break
        # Every set this node tried is kept, bounded so evidence stays small.
        self.checked = (*self.checked, *shapes)[-MAX_SHAPES:]
        return held

    def _shape(self, condition: Condition, held: bool) -> CheckShape:
        """Describe a check for failure evidence, without any of its text."""
        tag = type(condition).TAG
        target_id = getattr(condition, "target", None)
        target = self.cap.target(target_id) if target_id is not None else None
        if not isinstance(target, StructuralTarget):
            return CheckShape(tag, held, form="visual" if target else "")
        return CheckShape(
            tag,
            held,
            form=target.form.value,
            role=target.role
            if target.role in ROLES
            else ("other" if target.role else ""),
            match=target.match.value,
            name=target.name.kind.value if target.name is not None else "",
            scope=target.scope.kind.value if target.scope is not None else "",
            frames=len(target.frame),
        )

    def _one(self, condition: Condition, view: _View) -> bool:
        held = self._condition(condition, view)
        if held and self.probing:
            self.seen |= self._texts(condition) & self.probing
        return held

    def _condition(self, condition: Condition, view: _View) -> bool:
        match condition:
            case AtRoute(route=route):
                return view.route == route
            case Present(target=target):
                return self._find(target, view) is Found.FOUND
            case Absent(target=target):
                return self._find(target, view) is Found.NOT_FOUND
            case Shows(
                target=target,
                value=value,
                match=match,
                record=record,
                purpose=purpose,
                once=once,
            ):
                self._read_gate(target, view)
                return self._shows(
                    target, value, match, view, record, purpose=purpose, once=once
                )
            case DialogOpen(kind=kind, message=message):
                dialog = view.observation.dialog if view.observation else None
                return (
                    dialog is not None
                    and dialog.kind == kind
                    and (message is None or dialog.message == self._text(message))
                )
            case Bound(value=value):
                held = self.values.get((value.kind, value.name))
                declared = self.cap.field(value)
                return (
                    held is not None
                    and declared is not None
                    and (declared.accepts(held))
                )
            case ValueIs(
                value=value, expected=expected, match=match, purpose=purpose, once=once
            ):
                return text_matches(
                    self._text(value),
                    self._text(expected),
                    match,
                    purpose=purpose,
                    once=once,
                )
            case Completed(node=node):
                return node in self.completed

    def _find(self, target_id: str, view: _View) -> Found:
        target = self.cap.target(target_id)
        if view.route != target.route:
            return Found.OFF_ROUTE
        try:
            if isinstance(target, VisualTarget):
                return spot(self._template(target), self._capture(target)).found
            if isinstance(target, RadioTarget):
                return radio_spot(
                    target, self._capture(target), self.reader, self._text
                ).found
            if isinstance(target, TextTarget):
                lines = self._lines(self._capture(target))
                return text_line(target, lines, self._text)[0]
            if isinstance(target, FocusTarget):
                captured = self._capture(target) is not None
                return Found.FOUND if captured else Found.UNAVAILABLE
            return resolve(target, view.observation, self._text, view.route).found
        except KeyError:
            return Found.UNAVAILABLE

    def _shows(
        self,
        target_id: str,
        value: Ref,
        match: Match,
        view: _View,
        record: RecordSpec | None = None,
        *,
        purpose: Purpose = Purpose.STATE,
        once: bool = False,
    ) -> bool:
        """Report whether a target shows a value, for the record it names.

        A check that carries a record holds only when the record's source
        displays this invocation's identifier and the page ties it to the
        target by the saved relation. A shared route is not that tie.
        """
        target = self.cap.target(target_id)
        if isinstance(target, TextTarget):
            return record is None and self._painted_shows(
                target, value, match, view, purpose=purpose, once=once
            )
        if not isinstance(target, StructuralTarget):
            return False
        try:
            resolved = resolve(target, view.observation, self._text, view.route)
            expected = self._text(value)
        except KeyError:
            return False
        if resolved.node is None:
            return False
        text = shown(resolved.node)
        if text is None or not text_matches(
            text, expected, match, purpose=purpose, once=once
        ):
            return False
        return record is None or self._tied(record, resolved.node, view)

    def _lines(self, scene: Scene | None) -> tuple[Line, ...]:
        """Read the lines one capture shows. Nothing read is kept."""
        return self.reader.lines(scene.image) if scene is not None else ()

    def _painted_shows(
        self,
        target: TextTarget,
        value: Ref,
        match: Match,
        view: _View,
        *,
        purpose: Purpose = Purpose.STATE,
        once: bool = False,
    ) -> bool:
        """Report whether a painted line shows a value, read from two fresh captures.

        The line is read from one capture and then another, and the two
        readings must agree, or the check is unresolved and a person decides
        (rule 10). Agreement detects a screen that changed between the two;
        it does not prove that recognition read the line correctly.
        """
        if view.route != target.route:
            return False
        try:
            expected = self._text(value)
            _, line = text_line(target, self._lines(self._capture(target)), self._text)
            _, again = text_line(target, self._lines(self._capture(target)), self._text)
        except KeyError:
            return False
        if line is None and again is None:
            return False
        if line is None or again is None or line.text != again.text:
            raise _End(Status.NEEDS_HELP, Reason.READING_UNCONFIRMED)
        text = (
            reading.after(line, self._text(target.text)) if target.label else line.text
        )
        # As in discovery, through ``matching``: painted words ignore case, a
        # colon breaks words, and only a value with spaces is also found with
        # its spaces dropped.
        return text_matches(
            text, expected, match, purpose=purpose, once=once, painted=True
        )

    def _painted_value(self, target: TextTarget, line: str | None) -> str:
        """Return the value after a label, only when a second capture agrees.

        The surface read one line at the point the target aimed at. Another
        capture is read the same way, and the two readings must agree
        exactly, or the value is not taken and a person decides.
        """
        anchor = self._text(target.text)
        first = reading.Line(line or "", 0, 0, 0, 0, 1.0)
        if not line or not reading.squash(line).startswith(reading.squash(anchor)):
            raise _End(Status.NEEDS_HELP, Reason.READING_UNCONFIRMED)
        _, again = text_line(target, self._lines(self._capture(target)), self._text)
        value = reading.after(first, anchor)
        if again is None or reading.after(again, anchor) != value or not value:
            raise _End(Status.NEEDS_HELP, Reason.READING_UNCONFIRMED)
        return value

    def _tied(self, record: RecordSpec, node: AxNode, view: _View) -> bool:
        """Report whether ``node`` belongs to the record this invocation named."""
        source = self.cap.target(record.source)
        if not isinstance(source, StructuralTarget) or view.observation is None:
            return False
        try:
            resolved = resolve(source, view.observation, self._text, view.route)
            value = self._text(record.value)
        except KeyError:
            return False
        evidence = resolved.node
        if evidence is None or evidence.value is not None:
            return False
        text = shown(evidence)
        if text is None or record_label(text, value, record) is None:
            return False
        return unbound(view.observation, record.relation, node, evidence) is None

    def _text(self, value: Ref) -> str:
        """Return a reference's text for this invocation, or raise ``KeyError``.

        A secret has no text here. Its name goes to the surface in a
        ``SecretRef``, and the surface resolves it at the moment of input.
        """
        match value.kind:
            case RefKind.CONSTANT:
                return value.value
            case RefKind.INPUT:
                return self.inputs[value.name]
            case RefKind.VARIABLE | RefKind.OUTPUT:
                return self.values[value.kind, value.name]
            case RefKind.SECRET:
                raise KeyError(value.name)

    def _value(self, value: Ref) -> str:
        """Return a reference's text for an action, or end the replay without it."""
        try:
            return self._text(value)
        except KeyError:
            raise _End(Status.FAILED, Reason.MISSING_VALUE) from None

    # Coordinate the session with the people who hold it.

    def _view(self) -> _View:
        """Look at the surface through the gate, with the risk the profile declares.

        A look the gate or a saved restriction refuses, or a risky one a
        person does not approve, leaves the view without an observation,
        which every condition reads as unknown. An observation of a page
        other than the one the look was allowed for is dropped the same way.
        """
        self._live()
        opened = {page.page_id for page in self.surface.pages()} - self.windows
        if opened:
            self._help(
                self.current or self.cap.node(self.cap.entry),
                Ask.PERSON,
                Reason.NEW_WINDOW,
            )
            self.windows = {page.page_id for page in self.surface.pages()}
            self._live()
        request = ObservationRequest(
            ObservationMode.STRUCTURED, ObservationProvenance.REFRESH
        )
        location, allowed = self._may_look(request)
        route = policy.route_for(self.profile, location)
        if not allowed:
            return _View(location, route, None)
        try:
            observation = self.surface.observe(request)
        except ControlChanged:
            self._hold()
            return _View(self.surface.location(), None, None)
        if observation.location != location:
            return _View(
                observation.location,
                policy.route_for(self.profile, observation.location),
                None,
                moved=True,
            )
        observation = coverage.gather(observation, request, self._coverage_page)
        self.observed = snapshot(observation)
        return _View(location, route, observation)

    def _coverage_page(self, request: ObservationRequest) -> Observation | None:
        self._live()
        location, allowed = self._may_look(request)
        if not allowed:
            return None
        try:
            observation = self.surface.observe(request)
        except ControlChanged:
            # A person took the session between pages. The replay holds, and
            # the pages read so far do not count as complete coverage.
            self._hold()
            return None
        return observation if observation.location == location else None

    def _capture(self, target: VisualTarget | TextTarget | FocusTarget) -> Scene | None:
        """Take a fresh capture for a visual match, only after the gate allows it.

        The capture is allowed for the location the gate judged, and only
        while that location is on the target's route. The provider is called
        after that and grants nothing itself.
        """
        if self.scenes is None:
            return None
        request = ObservationRequest(
            ObservationMode.VISUAL, ObservationProvenance.REFRESH
        )
        location, allowed = self._may_look(request)
        if not allowed or policy.route_for(self.profile, location) != target.route:
            return None
        scene = self.scenes.capture(target.frame)
        return scene if self.surface.location() == location else None

    def _may_look(self, request: ObservationRequest) -> tuple[str, bool]:
        """Decide whether the surface may be looked at where it is now.

        Returns the location the decision is about and whether the look may
        go ahead. The profile's verdict comes first, then saved restrictions
        on ``observe`` for the route: a deny ends the replay, and a risky
        verdict or restriction needs approval. An approval covers the mode
        and the exact location it was given for. After one, the location is
        read and judged again, and a page that moved needs its own decision.
        The budget is checked last, immediately before the caller looks.
        """
        approved: tuple[object, ...] | None = None
        while True:
            self._live()
            location = self.surface.location()
            verdict = policy.evaluate_observation(request, self.profile, location)
            if isinstance(verdict, Denied):
                return location, False
            limit = strictest(
                self.cap.restrictions, _probe(ActionKind.OBSERVE, verdict.route, None)
            )
            node = self.current
            assert node is not None  # noqa: S101  set before the first look
            if limit is Limit.DENY:
                self.log.record(
                    Gated(node.node_id, ActionKind.OBSERVE, verdict.route, None, False)
                )
                raise _End(Status.FAILED, Reason.RESTRICTION_DENIED)
            if verdict.risk is not Risk.RISKY and limit is not Limit.RISKY:
                self._live()
                return location, True
            context = (request.mode, location, verdict.route)
            if approved == context:
                self._live()
                return location, True
            self.log.record(
                Gated(node.node_id, ActionKind.OBSERVE, verdict.route, None, True)
            )
            reason = (
                Reason.APPROVAL_REQUIRED
                if approved is None
                else Reason.APPROVAL_INVALIDATED
            )
            look = Action(ActionKind.OBSERVE)
            if self._help(node, Ask.APPROVAL, reason, look) is not Answer.APPROVED:
                return location, False
            approved = context

    def _pause(self) -> None:
        self.sleep(self.cap.limits.settle_interval_ms / 1000)

    def _help(
        self,
        node: Node,
        ask: Ask,
        reason: Reason,
        proposal: Action | None = None,
        *,
        planned: HelpReason | None = None,
    ) -> Answer:
        """Ask a person, with the execution clock stopped while they answer."""
        if self.asked >= self.cap.limits.max_help_requests:
            raise _End(Status.NEEDS_HELP, _UNRESOLVED.get(reason, reason))
        self.asked += 1
        intervention = f"{self.run_id}-{self.asked}"
        timeout = float(self.profile.escalation.handoff_timeout_s)
        request = HelpRequest(
            intervention=intervention,
            run=self.run_id,
            ask=ask,
            reason=reason,
            capability=self.cap.label,
            node=node.node_id,
            route=_node_route(node),
            timeout_s=timeout,
            proposal=proposal,
            planned=planned,
            step=self._budget().steps,
            expected=self.expected,
            observed=self.observed,
            question=self._question(node, ask, reason, planned),
        )
        asked = self.clock()
        with self._budget().waiting_for_a_human():
            reply = self.operator.request(request)
        waited = self.clock() - asked
        if reply.intervention != intervention or reply.intervention in self.answered:
            raise _End(Status.NEEDS_HELP, Reason.STALE_ANSWER)
        self.answered.add(intervention)
        self.log.record(Helped(node.node_id, ask, reason, reply.answer))
        self.history.append(
            HistoryEntry(
                node.node_id, intervention, ask, reason, reply.answer, reply.evidence
            )
        )
        return _settle_answer(reply, ask, late=waited > timeout)

    def _question(
        self, node: Node, ask: Ask, reason: Reason, planned: HelpReason | None
    ) -> str:
        """Say what a person is asked to settle, for this invocation (rule 16).

        It names the step's operation, the request's values that say which
        record it is about, and the question itself. Live data only.
        """
        values = ", ".join(f"{name}={value}" for name, value in self.inputs.items())
        record = f" for {values}" if values else ""
        if isinstance(node, ActionNode):
            operation = f"{node.kind.value} {node.effect or ''}".strip()
        elif isinstance(node, HumanNode):
            operation = f"a person's step ({node.reason.value})"
        else:
            operation = f"the check at {node.node_id}"
        if reason is Reason.SELECTION_UNCONFIRMED:
            question = (
                "Choose each required picker item; its typed query is not a selection."
            )
        elif isinstance(node, ResultConfirmationNode):
            required = ", ".join(
                f"{value.name}={self._value(value)}" for value in node.confirm_inputs
            )
            question = (
                "Confirm the result belongs to this record and satisfies these "
                f"required values: {required}."
            )
        elif planned is HelpReason.RECORD_EVIDENCE:
            question = "Confirm the result on the screen belongs to this record."
        elif reason is Reason.DELIVERY_UNCERTAIN:
            question = (
                "Did this step go through? Check the page. It is never sent a "
                "second time."
            )
        elif ask is Ask.APPROVAL:
            question = "Approve this one operation, once, for this record."
            choices = self._review_choices(node)
            if choices:
                question = (
                    f"Verify the selected choices on the review screen ({choices}). "
                    "Approve only if every choice matches, otherwise reject."
                )
        elif planned is not None:
            question = (
                f"Do this step by hand ({planned.value.replace('_', ' ')}), then "
                "press Done, continue. The replay checks the page and goes on."
            )
        else:
            question = self._instruction(node, reason)
        # The question comes first, on its own line; the step follows it.
        return f"{question}\n{operation}{record}"

    def _instruction(self, node: Node, reason: Reason) -> str:
        """Tell a person what the replay needs them to do.

        It names the problem, the step the replay means to take, on which
        page and with which control, and what the page must show, with the
        live values the replay holds. The person may do the step or bring
        the page to that state; the replay then checks the page itself.
        """
        from computeruse.describe import describe_condition, describe_step

        live = {
            **self.inputs,
            **{name: text for (_, name), text in self.values.items()},
        }
        problem = REASON_TEXT.get(reason, reason.value.replace("_", " "))
        lines = [f"Help the replay: {problem}."]
        if isinstance(node, ActionNode):
            targets = {target.target_id: target for target in self.cap.targets}
            lines.append(
                f"Next step: {describe_step(node, targets, live)}, on {node.route}"
            )
            after = reason in {Reason.VERIFICATION_FAILED, Reason.NO_TRANSITION}
            checks = node.verify if after else node.requires
            if checks:
                said = "; ".join(
                    describe_condition(check, self.cap, live) for check in checks[:4]
                )
                more = "; and more" if len(checks) > 4 else ""
                label = "Afterwards it expects" if after else "It expects"
                lines.append(f"{label}: {said}{more}")
        lines.append(
            "Do the step yourself, or bring the page to that state, then press "
            "Done, continue. The replay checks the page and goes on."
        )
        return "\n".join(lines)

    def _review_choices(self, node: Node) -> str:
        """Name unresolved choices for this write using current invocation values."""
        choices = []
        for item in self.cap.nodes:
            if (
                isinstance(item, ActionNode)
                and item.node_id in self.completed
                and preparation_review(self.cap, item) == node.node_id
                and item.target is not None
            ):
                target = self.cap.target(item.target)
                if isinstance(target, TextTarget):
                    choices.append(f"{target.text.name}={self._text(target.text)}")
        return ", ".join(choices)

    def _attempt(
        self, node: ActionNode, visit: int, delivery: Delivery, actor: Actor | None
    ) -> None:
        self.attempts.append(Attempt(node.node_id, visit, delivery, actor))
        if (
            delivery is not Delivery.NOT_DISPATCHED
            and node.approval is Approval.EACH_RUN
        ):
            self.sent_operations.setdefault(node.node_id, (node, actor))
        if delivery is Delivery.COMPLETED:
            self.completed.add(node.node_id)

    def _end(
        self,
        status: Status,
        reason: Reason,
        outcome: str = "",
        outputs: Mapping[str, str] | None = None,
        *,
        issues: tuple[Issue, ...] = (),
    ) -> ReplayResult:
        ending = (
            "completed"
            if status in {Status.SUCCEEDED, Status.OUTCOME}
            else "stopped"
            if status is Status.STOPPED
            else "terminated"
            if status is Status.TERMINATED
            else "failed"
        )
        self.control.finish(ending)
        budget = self.budget
        steps = budget.steps if budget else 0
        active = budget.active_seconds() if budget else 0.0
        elapsed = self.clock() - self.started
        diagnostic = None
        if status not in {Status.SUCCEEDED, Status.OUTCOME, Status.STOPPED}:
            diagnostic = FailureEvidence(
                steps,
                self.node,
                reason.value,
                self.expected,
                self.observed,
                self.checked,
            )
            self.log.record(diagnostic)
        self.log.record(ReplayEnded(status, reason, self.node, steps))
        return ReplayResult(
            status=status,
            reason=reason,
            node=self.node,
            outcome=outcome,
            outputs=MappingProxyType(dict(outputs or {})),
            attempts=tuple(self.attempts),
            history=tuple(self.history),
            issues=issues,
            steps=steps,
            active_s=active,
            paused_s=max(0.0, elapsed - active) if budget else 0.0,
            run=self.run_id,
            diagnostic=diagnostic,
            differed=frozenset(self.differed),
            seen=frozenset(self.seen),
        )


def _option_holding(node: AxNode | None, value: str) -> str:
    """Return the one option label that holds ``value`` once as whole words.

    A choice saved by ``contains`` names its input, and the list's labels add
    other words, such as a product code. No match, or two, stops the replay
    rather than choosing the first.
    """
    labels = [
        option.label
        for option in (node.options if node is not None else ())
        if not option.disabled and matching.contains(option.label, value, once=True)
    ]
    if not labels:
        raise _End(Status.NEEDS_HELP, Reason.TARGET_NOT_FOUND)
    if len(labels) > 1:
        raise _End(Status.NEEDS_HELP, Reason.TARGET_AMBIGUOUS)
    return labels[0]


_NOT_SENT = frozenset(
    {Outcome.NOT_FOUND, Outcome.AMBIGUOUS, Outcome.STALE, Outcome.NOT_ACTIONABLE}
)

_BY_HAND = {
    ActionKind.CLICK: frozenset({ManualKind.CLICK, ManualKind.KEY, ManualKind.SUBMIT}),
    ActionKind.DOUBLE_CLICK: frozenset({ManualKind.CLICK}),
    ActionKind.TYPE: frozenset({ManualKind.EDIT}),
    ActionKind.SELECT: frozenset({ManualKind.SELECT, ManualKind.CLICK}),
    ActionKind.PRESS_KEY: frozenset({ManualKind.KEY, ManualKind.SUBMIT}),
}
"""How a person can perform a step of each action type."""
"""Outcomes where the surface refused before sending any input."""

_MISSED = {
    Found.NOT_FOUND: Reason.TARGET_NOT_FOUND,
    Found.AMBIGUOUS: Reason.TARGET_AMBIGUOUS,
    Found.UNAVAILABLE: Reason.TARGET_UNAVAILABLE,
    Found.OFF_ROUTE: Reason.OFF_ROUTE,
    Found.INCOMPLETE: Reason.OBSERVATION_INCOMPLETE,
}

_UNRESOLVED = {
    Reason.PLANNED_STEP: Reason.HELP_UNRESOLVED,
    Reason.APPROVAL_REQUIRED: Reason.HELP_UNRESOLVED,
    Reason.APPROVAL_INVALIDATED: Reason.HELP_UNRESOLVED,
}


def _key(node: ActionNode, action: Action, view: _View, aim: _Aim) -> _Key:
    """Return what an approval is bound to: the exact proposal and its screen.

    The key holds the action as built, with its values, record evidence, and
    effect; the route; the observed control and its context; and the dialog
    an answer names. It leaves out the observation id, so looking again at
    an unchanged screen keeps the approval valid. A visual target is keyed
    by its template, because every capture has a new id.
    """
    target: object = action.target
    if isinstance(target, ScreenTarget):
        target = ("template", node.target)
    control = aim.node.control if aim.node is not None else ""
    context = aim.node.context if aim.node is not None else ()
    return (
        node.node_id,
        action.kind,
        target,
        action.value,
        action.destination,
        action.evidence,
        action.effect,
        view.route,
        control,
        context,
        aim.dialog,
    )


def _read_key(read: Action, view: _View, aim: _Aim) -> _Key:
    """Return what a completion read's approval is bound to."""
    control = aim.node.control if aim.node is not None else ""
    context = aim.node.context if aim.node is not None else ()
    return (read, view.location, view.route, control, context)


def _probe(kind: ActionKind, route: str, target: str | None) -> ActionNode:
    """Build the operation a look or a completion read is, so restrictions apply."""
    return ActionNode(
        node_id="replay_probe",
        kind=kind,
        route=route,
        target=target,
        value=None,
        destination=None,
        effect=None,
        record=None,
        result_record=None,
        into=None,
        approval=Approval.NONE,
        mandatory=False,
        requires=(),
        verify=(),
        transitions=(),
    )


def _form(target: Target | None) -> str:
    match target:
        case AxLocator():
            return "accessibility"
        case DomLocator():
            return "dom"
        case ScreenTarget():
            return "screen"
        case None:
            return "none"
        case _:
            return "visual"


def _settle_answer(reply: HelpAnswer, ask: Ask, *, late: bool) -> Answer:
    if late or reply.answer is Answer.TIMED_OUT:
        raise _End(Status.HELP_TIMED_OUT, Reason.HELP_TIMED_OUT)
    if reply.answer is Answer.TERMINATED:
        raise _End(Status.TERMINATED, Reason.OPERATOR_TERMINATED)
    if reply.answer is Answer.REJECTED:
        reason = (
            Reason.APPROVAL_REJECTED
            if ask is Ask.APPROVAL
            else Reason.OPERATOR_TERMINATED
        )
        raise _End(Status.TERMINATED, reason)
    if reply.answer is Answer.APPROVED and ask is not Ask.APPROVAL:
        # Nothing was proposed, so an approval performs nothing. The replay
        # reads the page as it would after a hand-back.
        return Answer.RESUMED
    return reply.answer


def _route(verdict: Allowed | Denied) -> str:
    return verdict.route if isinstance(verdict, Allowed) else ""


def _node_route(node: Node) -> str:
    if isinstance(node, (ActionNode, HumanNode)):
        return node.route
    return ""
