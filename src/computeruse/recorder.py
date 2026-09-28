"""Build a capability from executed actions and verified results.

The recorder consumes typed execution events published by ``DiscoveryTrace``.
An event says which action ran and how it ended, how its target was described,
where each value came from, which checks held afterwards, which restrictions the run
learned, where a person took over, and how the result was verified. The
recorder counts but omits a proposal that the gate refused or an action that
definitely never reached the surface.

An operation that may have reached the surface is never dropped. A timeout,
an unacknowledged input, or a surface error says nothing certain about
whether the application acted, so the event's ``dispatch`` field, or its
outcome when that is absent, decides. A potentially dispatched mutating
operation is kept, with the checks the run verified after it as its
verification, when there are such checks. When a person's segment at the
same step took over, that segment stands for it. Otherwise the recording is
incomplete.

Values are parameterized by their declared source. The recorder is given the
invocation's input values in memory, and any recorded text equal to one
becomes a reference to that input. Text that merely contains one, such as a
row labelled with a member number and a name, cannot be written reusably, so
the step becomes a person's step or the recording is incomplete. The input
values themselves are never written.

A step the recorder cannot represent reliably is not dropped. When the run
verified what the screen showed afterwards, the step becomes an explicit
human node whose continuation is that check, and which names the operation
it stands in for. Saved restrictions apply to that human node as they would
to the action, so a denied operation makes the recording incomplete instead
of becoming a person's task. Otherwise the recording is reported
incomplete, with the step that stopped it.

A recording is a draft. A successful run does not prove the workflow is
reusable, so the capability is validated against the contract and the profile
before it is returned, and a person reviews it before a replay accepts it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum

from computeruse import matching, policy
from computeruse.actions import (
    ENTER,
    STRUCTURAL_ROLES,
    Operation,
    Outcome,
    Relation,
    Scope,
    ScopeKind,
    positional_slot,
    word_positions,
)
from computeruse.capability import (
    MUTATING,
    SCHEMA_VERSION,
    Absent,
    ActionNode,
    Application,
    Approval,
    AtRoute,
    Bound,
    Capability,
    Condition,
    Destination,
    DialogOpen,
    DomAttribute,
    Edge,
    EdgeOrigin,
    Extraction,
    Field,
    FocusTarget,
    HelpReason,
    HumanNode,
    Issue,
    Limits,
    LocatorForm,
    Match,
    Node,
    Param,
    Present,
    Provenance,
    ProvenanceKind,
    Purpose,
    RadioTarget,
    RecordSpec,
    Ref,
    RefKind,
    RestrictionScope,
    ResultConfirmationNode,
    ResultKind,
    ResultNode,
    Review,
    SavedRestriction,
    ScopeSpec,
    Shows,
    StoragePermit,
    StructuralTarget,
    SurfaceKind,
    Target,
    Template,
    TextTarget,
    ValueIs,
    VisualTarget,
    check_profile,
    constant,
    ref,
    strictest,
    template,
    validate,
)
from computeruse.matching import holds
from computeruse.operations import limit as operation_limit
from computeruse.policy import Restriction
from computeruse.profile import ActionKind, Limit, Profile, Risk

RECORDER = "computeruse_recorder_1"
"""The recorder version written into every capability's provenance."""


@dataclasses.dataclass(frozen=True, slots=True)
class TargetSample:
    """A structural target as discovery resolved it, before parameterization.

    The integration builds this from the locator the action used and the
    route template it ran on. It must not carry the surface's element id: an
    element id ends with its document and never becomes a replay identity.
    ``match`` preserves a contained reference when the integration proved
    that reference identified exactly one observed control.
    """

    route: str
    form: LocatorForm
    role: str = ""
    name: str = ""
    tag: str = ""
    attribute: DomAttribute | None = None
    value: str = ""
    frame: tuple[str, ...] = ()
    scope: Scope | None = None
    match: Match = Match.EQUALS


@dataclasses.dataclass(frozen=True, slots=True)
class VisualSample:
    """A painted control crop and the source of permission to store it.

    ``permit`` is None unless the operator reviewed the crop, or the crop is
    synthetic test material. Without a permit the crop is not stored and the
    step goes to a person.
    """

    route: str
    image: bytes
    frame: tuple[str, ...] = ()
    permit: StoragePermit | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ScreenSample:
    """A live screen coordinate. It identifies nothing a later run can find."""

    route: str


@dataclasses.dataclass(frozen=True, slots=True)
class TextSample:
    """A painted line of text a screenshot action aimed at, as it was read.

    ``text`` is the line, or the label before a value with ``label``. ``dx``
    and ``dy`` are the point's offset from the line's top left corner when
    the action aimed beside it, at a field its label names. The text is
    saved only as confirmed website words or as the input it holds.
    """

    route: str
    text: str
    label: bool = False
    dx: int = 0
    dy: int = 0
    frame: tuple[str, ...] = ()
    radio: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class FocusSample:
    """Keys a screenshot action sent to whatever held the keyboard focus."""

    route: str
    frame: tuple[str, ...] = ()


type TargetEvidence = (
    TargetSample | VisualSample | ScreenSample | TextSample | FocusSample
)


@dataclasses.dataclass(frozen=True, slots=True)
class ValueSample:
    """The value source declared by the integration.

    ``INPUT``, ``VARIABLE``, ``OUTPUT``, and ``SECRET`` name their source.
    ``CONSTANT`` carries its text, which the recorder still checks against
    the invocation's inputs.
    """

    source: RefKind
    name: str = ""
    text: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class RecordSample:
    """The record evidence an executed action carried."""

    source: TargetSample
    value: ValueSample
    relation: Relation
    prefix: str = ""
    suffix: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class DestinationSample:
    """A navigation's allow-route template and the source of each placeholder."""

    route: str
    params: tuple[tuple[str, ValueSample], ...] = ()
    query: tuple[tuple[str, ValueSample], ...] = ()
    fragment: ValueSample | None = None
    origin: str = ""


class CheckKind(StrEnum):
    """The condition type of a verified check."""

    AT_ROUTE = "at_route"
    PRESENT = "present"
    ABSENT = "absent"
    SHOWS = "shows"
    DIALOG_OPEN = "dialog_open"
    BOUND = "bound"
    VALUE_IS = "value_is"


@dataclasses.dataclass(frozen=True, slots=True)
class CheckSample:
    """One check the run verified against the live page."""

    kind: CheckKind
    target: TargetSample | VisualSample | TextSample | None = None
    value: ValueSample | None = None
    match: Match = Match.EQUALS
    route: str = ""
    dialog: str = ""
    record: RecordSample | None = None
    expected: ValueSample | None = None
    purpose: Purpose = Purpose.STATE
    once: bool = False


class Dispatch(StrEnum):
    """The integration's knowledge of whether an operation reached the surface.

    ``NOT_SENT`` means the adapter refused before any input, such as a target
    that was not found. ``SENT`` means input was delivered, whatever came of
    it. ``UNKNOWN`` means input may or may not have been delivered, such as a
    driver that failed mid-call.
    """

    NOT_SENT = "not_sent"
    SENT = "sent"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True, slots=True)
class Proposed:
    """The decider proposed an action. Counted, never recorded as a step."""

    step: int
    kind: ActionKind


@dataclasses.dataclass(frozen=True, slots=True)
class Executed:
    """An action reached the surface, with everything needed to reuse it.

    ``operation`` is the identity discovery bound restrictions to, used only
    to carry those restrictions onto reusable targets. ``risk`` is what the
    gate reported, ``approved`` says a person approved it, and ``flagged``
    says the model flagged it, any of which makes the step need a person on
    every replay. ``into`` names the variable or output a read filled.

    ``dispatch`` says whether input reached the surface, when the outcome
    alone cannot. None derives it from the outcome: ``ok`` was sent, a
    refusal such as ``not_found`` or ``blocked`` was not, and
    ``uncertain``, ``surface_error``, and ``handoff`` are unknown.
    """

    step: int
    kind: ActionKind
    route: str
    outcome: Outcome
    target: TargetEvidence | None = None
    value: ValueSample | None = None
    destination: DestinationSample | None = None
    effect: str | None = None
    record: RecordSample | None = None
    operation: Operation | None = None
    into: ValueSample | None = None
    risk: Risk = Risk.SAFE
    approved: bool = False
    flagged: bool = False
    dispatch: Dispatch | None = None
    requires: tuple[CheckSample, ...] = ()
    introduced: str = ""
    extraction: tuple[str, str] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Verified:
    """Checks that held on the live page after ``step``. Step 0 is the entry screen."""

    step: int
    checks: tuple[CheckSample, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Learned:
    """A restriction the run bound to an operation."""

    restriction: Restriction


@dataclasses.dataclass(frozen=True, slots=True)
class HumanSegment:
    """A person's intervention after ``step`` and its resulting state.

    ``continuation`` is what the run verified when it took the session back.
    ``gaps`` counts what the manual recording could not capture. Neither the
    steps a person took nor their values are carried: a replay asks a person
    to take them again.
    """

    step: int
    route: str
    reason: HelpReason
    continuation: tuple[CheckSample, ...]
    gaps: int = 0
    performs: ActionKind | None = None
    effect: str | None = None
    forbidden: bool = False
    permissions: tuple[ActionKind, ...] = ()
    replaces: int | None = None
    """The execution event this takeover replaces, when the caller knows it."""


class Verifier(StrEnum):
    """The source that established a run's final result."""

    EXECUTOR = "executor"
    PERSON = "person"


@dataclasses.dataclass(frozen=True, slots=True)
class Finished:
    """The run ended with a result, and these checks established it.

    ``confirmed`` is set when a person verified obligations the surface could
    not show, while every supplied check passed the executor's own reading.
    ``confirm_inputs`` names missing required values from declared inputs.
    The capability asks a person to confirm those obligations on every replay.
    """

    verifier: Verifier
    checks: tuple[CheckSample, ...]
    outcome: str = ""
    confirmed: bool = False
    confirm_inputs: tuple[Ref, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Unneeded:
    """The step at ``step`` changed nothing the run saw, and was not needed.

    The discovery bridge reports a click or key when the next step's control
    was already on the screen, unchanged, before the action, and a read that
    kept a fact the run already held with the same value. Such a step is left out of
    the capability if nothing proves it, it is safe, and no person approved
    or flagged it. A step that did matter after all fails a later check on
    replay, which then asks a person rather than going on.
    """

    step: int


type ExecutionEvent = (
    Proposed | Executed | Verified | Learned | HumanSegment | Finished | Unneeded
)


@dataclasses.dataclass(frozen=True, slots=True)
class DiscoveryRun:
    """The events came from a real discovery run with this id."""

    run: str


@dataclasses.dataclass(frozen=True, slots=True)
class Synthetic:
    """The events were written for a test. The capability says so."""


class Gap(StrEnum):
    """The reason a recording is incomplete."""

    NOT_FINISHED = "not_finished"
    NO_STEPS = "no_steps"
    UNVERIFIED_RESULT = "unverified_result"
    UNCERTAIN_DELIVERY = "uncertain_delivery"
    UNREPRESENTABLE_STEP = "unrepresentable_step"
    UNREPRESENTABLE_CHECK = "unrepresentable_check"
    MANUAL_WITHOUT_CONTINUATION = "manual_without_continuation"
    DENIED_STEP = "denied_step"
    AMBIGUOUS_TAKEOVER = "ambiguous_takeover"
    INVALID_CAPABILITY = "invalid_capability"
    UNBOUND_RECORD = "unbound_record"


@dataclasses.dataclass(frozen=True, slots=True)
class RecordingIssue:
    """One reason a recording is incomplete, and the run step it concerns.

    ``detail`` says which part could not be saved, in words, never a value.
    """

    gap: Gap
    step: int
    detail: str = ""


class Place(StrEnum):
    """Where saved text appears and who may confirm it.

    ``CONTROL`` is the label of an interface control outside any record's row,
    such as a button, a tab, or a column header, or a name the page's
    structure uses. ``RECORD`` is every other text: a cell, a value a check
    expects, text around a record's identifier, or a dialog's message. That is
    where customer data lives, so only a comparison or a person confirms it.
    """

    CONTROL = "control"
    RECORD = "record"


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    """A text the recording would save that nobody has confirmed yet."""

    text: str
    places: frozenset[Place]

    @property
    def interface(self) -> bool:
        """Report whether every use of the text is an interface control's."""
        return self.places == frozenset({Place.CONTROL})


INTERFACE_ROLES = frozenset(
    {
        "button",
        "checkbox",
        "columnheader",
        "combobox",
        "link",
        "menuitem",
        "radio",
        "searchbox",
        "slider",
        "spinbutton",
        "switch",
        "tab",
        "textbox",
    }
)
"""Roles whose accessible name labels the control itself, not a record's data."""


@dataclasses.dataclass(frozen=True, slots=True)
class Recording:
    """A draft capability or the reasons the recorder could not build one.

    ``candidates`` are texts the draft uses that the run's confirmed words did
    not cover. A draft with candidates is held in memory until each one is
    confirmed; nothing unconfirmed is saved.

    ``rejected`` is the assembled draft that failed validation, held only so
    a report can say which nodes the ``artifact_issues`` point at. It may
    hold unconfirmed text, so it is never saved.
    """

    capability: Capability | None
    issues: tuple[RecordingIssue, ...]
    artifact_issues: tuple[Issue, ...] = ()
    proposals: int = 0
    unperformed: int = 0
    candidates: tuple[Candidate, ...] = ()
    rejected: Capability | None = None

    @property
    def complete(self) -> bool:
        """Report whether a capability was produced."""
        return self.capability is not None


class _Unrepresentable(Exception):  # noqa: N818  a verdict about a step, not a fault
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class _Step:
    """One recorded step.

    ``stands_for`` lists the operations a person's step replaces, so
    restrictions learned on them still apply to it.
    """

    step: int
    node: Node
    operation: Operation | None = None
    stands_for: tuple[Operation, ...] = ()


class Recorder:
    """Accepts execution events for one run and builds a draft capability.

    Parameters
    ----------
    capability_id, version
        The identity to give the capability.
    profile
        The profile the run used. The recording is checked against it.
    origin
        ``DiscoveryRun`` for a real run, ``Synthetic`` for test events.
    inputs, outputs, variables
        The declared fields.
    outcomes
        Declared business outcomes a run may finish with.
    limits
        Bounds for replays of this capability.
    samples
        The run's input values, by name. Used only to find recorded text that
        came from an input, and never written.
    kept
        Values the run kept and saves as a variable or an output, by that
        reference. A control named by exactly such a value, such as a link
        named by an account number the run created, is saved as the
        reference. The values are never written.
    """

    def __init__(
        self,
        *,
        capability_id: str,
        version: int,
        profile: Profile,
        origin: DiscoveryRun | Synthetic,
        inputs: tuple[Field, ...],
        outputs: tuple[Field, ...],
        variables: tuple[Field, ...],
        outcomes: tuple[str, ...],
        limits: Limits,
        samples: Mapping[str, str],
        safe_text: frozenset[str] = frozenset(),
        collect: bool = False,
        excluded: tuple[str, ...] = (),
        kept: tuple[tuple[Ref, str], ...] = (),
    ) -> None:
        self._identity = (capability_id, version)
        self._profile = profile
        self._origin = origin
        self._fields = (inputs, outputs, variables)
        self._outcomes = outcomes
        self._limits = limits
        self._samples = {name: value for name, value in samples.items() if value}
        self._safe_text = safe_text
        # Collecting keeps an unconfirmed text as a candidate instead of
        # refusing the draft. A text holding any excluded value, such as an
        # output or a remembered fact, is still refused.
        self._collect = collect
        self._kept = tuple((reference, value) for reference, value in kept if value)
        self._excluded = tuple(
            dict.fromkeys(
                value
                for value in (*excluded, *self._samples.values(), *(v for _, v in kept))
                if value
            )
        )
        self._events: list[ExecutionEvent] = []

    def record(self, event: ExecutionEvent) -> None:
        """Accept one event, in the order the run produced them."""
        self._events.append(event)

    def proved(self, step: int) -> bool:
        """Report whether any checks were recorded for the action at ``step``."""
        return any(
            isinstance(event, Verified) and event.step == step and event.checks
            for event in self._events
        )

    def finish(self) -> Recording:
        """Build and validate the draft capability, or report why there is none."""
        return _Build(self).run()


class _Build:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
        self.profile = recorder._profile
        self.events = tuple(recorder._events)
        self.samples = recorder._samples
        inputs, outputs, variables = recorder._fields
        self.declared = {
            RefKind.INPUT: {item.name for item in inputs},
            RefKind.OUTPUT: {item.name for item in outputs},
            RefKind.VARIABLE: {item.name for item in variables},
        }
        self.issues: list[RecordingIssue] = []
        self.targets: list[Target] = []
        self.templates: list[Template] = []
        self.secrets: list[str] = []
        self.checks = _checks_by_step(self.events)
        self.unneeded = {e.step for e in self.events if isinstance(e, Unneeded)}
        self.candidates: dict[str, set[Place]] = {}
        self.assigned: set[tuple[RefKind, str]] = set()

    def gap(self, gap: Gap, step: int) -> None:
        self.issues.append(RecordingIssue(gap, step))

    def run(self) -> Recording:
        steps = self._steps()
        finished = [event for event in self.events if isinstance(event, Finished)]
        if not finished:
            self.gap(Gap.NOT_FINISHED, 0)
        if not steps:
            self.gap(Gap.NO_STEPS, 0)
        if self.issues or not finished:
            return self._incomplete()
        result = self._result(finished[-1])
        if finished[-1].confirmed and result.checks:
            steps = [*steps, self._confirmation(steps, result, finished[-1])]
        restrictions = self._restrictions(steps)
        nodes = self._link(steps, result, restrictions)
        capability = self._assemble(nodes, restrictions, steps)
        if self.issues:
            return self._incomplete()
        found = validate(capability) + check_profile(capability, self.profile)
        if found:
            self.gap(Gap.INVALID_CAPABILITY, 0)
            return dataclasses.replace(self._incomplete(found), rejected=capability)
        candidates = tuple(
            Candidate(text, frozenset(places))
            for text, places in sorted(self.candidates.items())
        )
        return Recording(capability, (), (), *self._counts(), candidates)

    def _incomplete(self, found: tuple[Issue, ...] = ()) -> Recording:
        return Recording(None, tuple(self.issues), found, *self._counts())

    def _counts(self) -> tuple[int, int]:
        proposals = sum(isinstance(event, Proposed) for event in self.events)
        unperformed = sum(
            isinstance(event, Executed) and _dispatch(event) is Dispatch.NOT_SENT
            for event in self.events
        )
        return proposals, unperformed

    def _steps(self) -> list[_Step]:
        steps: list[_Step] = []
        for event in self.events:
            if isinstance(event, HumanSegment):
                steps.append(self._segment(event, len(steps)))
            elif isinstance(event, Executed) and event.kind is not ActionKind.OBSERVE:
                built = self._executed(event, len(steps))
                if built is not None:
                    steps.append(built)
        return steps

    def _segment(self, event: HumanSegment, index: int) -> _Step:
        """Keep a person's segment, linked to the uncertain operation it replaced.

        A segment that begins right after an operation whose delivery is
        unknown took that operation over, unless it is an authentication
        step, which is a person's by nature. The human node then names the
        operation's action type and effect, and the step carries the
        operation's identity, so profile and learned restrictions still
        apply to it. Replaced operations that differ in type, effect, or
        route cannot be named by one node, and make the recording incomplete.
        """
        performs = event.performs
        effect = event.effect
        if event.forbidden:
            self.gap(Gap.DENIED_STEP, event.step)
        stands_for: tuple[Operation, ...] = ()
        replaced = (
            self._replaced(event.replaces if event.replaces is not None else event.step)
            if _takes_over(event)
            else ()
        )
        if replaced:
            shapes = {(item.kind, item.effect, item.route) for item in replaced}
            if len(shapes) != 1 or replaced[0].route != event.route:
                self.gap(Gap.AMBIGUOUS_TAKEOVER, event.step)
            performs, effect = replaced[0].kind, replaced[0].effect
            stands_for = tuple(
                item.operation for item in replaced if item.operation is not None
            )
        node = HumanNode(
            f"h{index + 1}",
            event.route,
            event.reason,
            performs,
            self._effect(performs, effect),
            True,
            (),
            permissions=event.permissions,
        )
        continuation = self._conditions(event.continuation, event.step)
        if not continuation:
            self.gap(Gap.MANUAL_WITHOUT_CONTINUATION, event.step)
        node = dataclasses.replace(node, transitions=(_edge("", continuation),))
        return _Step(event.step, node, stands_for=stands_for)

    def _replaced(self, step: int) -> tuple[Executed, ...]:
        """Return the mutating operations at ``step`` that may have been delivered."""
        return tuple(
            event
            for event in self.events
            if isinstance(event, Executed)
            and event.step == step
            and event.kind in MUTATING
            and event.outcome is not Outcome.OK
            and _dispatch(event) is not Dispatch.NOT_SENT
        )

    def _executed(self, event: Executed, index: int) -> _Step | None:
        dispatch = _dispatch(event)
        if dispatch is Dispatch.NOT_SENT:
            return None
        settled = event.outcome is Outcome.OK
        if not settled and event.kind not in MUTATING:
            # A read or a wait that failed changed nothing in the application.
            return None
        checks = self.checks.get(event.step, ())
        if settled and not checks and self._unneeded(event):
            return None
        if not settled:
            if self._taken_over(event.step):
                return None
            if not checks:
                self.gap(Gap.UNCERTAIN_DELIVERY, event.step)
                return None
        if settled and event.into is not None:
            # After this point, text equal to the value read by this step is saved as
            # the reference it filled. Earlier text is not.
            self.assigned.add((event.into.source, event.into.name))
        verify = self._conditions(checks, event.step)
        if not settled and not verify:
            self.gap(Gap.UNCERTAIN_DELIVERY, event.step)
            return None
        try:
            node = self._action(event, index, verify)
        except _Unrepresentable:
            return self._handed(event, index, verify, HelpReason.UNREPRESENTABLE_TARGET)
        if node.record is None and self.profile.requires_record(
            event.kind, event.route
        ):
            return self._handed(event, index, verify, HelpReason.RECORD_EVIDENCE)
        return _Step(event.step, node, event.operation)

    def _unneeded(self, event: Executed) -> bool:
        """Report whether a step may be left out; see ``Unneeded``."""
        return (
            event.step in self.unneeded
            and (
                event.kind in {ActionKind.CLICK, ActionKind.PRESS_KEY}
                or (event.kind is ActionKind.READ and event.into is None)
            )
            and event.risk is Risk.SAFE
            and not event.approved
            and not event.flagged
            and operation_limit(
                self.profile, event.operation or Operation(event.kind, event.route)
            )
            is None
        )

    def _taken_over(self, step: int) -> bool:
        """Report whether a person's segment took over the operation at ``step``."""
        return any(
            isinstance(event, HumanSegment)
            and (event.replaces if event.replaces is not None else event.step) == step
            and _takes_over(event)
            for event in self.events
        )

    def _handed(
        self,
        event: Executed,
        index: int,
        verify: tuple[Condition, ...],
        reason: HelpReason,
    ) -> _Step | None:
        """Keep a step a replay cannot perform as a person's, if it can be checked."""
        if not verify:
            self.gap(Gap.UNREPRESENTABLE_STEP, event.step)
            return None
        node = HumanNode(
            f"h{index + 1}",
            event.route,
            reason,
            event.kind,
            self._effect(event.kind, event.effect),
            True,
            (),
        )
        return _Step(
            event.step,
            dataclasses.replace(node, transitions=(_edge("", verify),)),
            event.operation,
            () if event.operation is None else (event.operation,),
        )

    def _choice(self, event: Executed) -> Ref | None:
        """Name the input a chosen option's label holds among other words.

        A list often labels an option with a code and a name, such as
        "GOLD-CHK: Gold checking", while the input is "Gold checking". Such a
        choice is saved as the input, matched as whole words, so the label's
        other words are never written.
        """
        sample = event.value
        if (
            event.kind is not ActionKind.SELECT
            or sample is None
            or sample.source is not RefKind.CONSTANT
            or sample.text in self.samples.values()
        ):
            return None
        held = self._word(sample.text)
        return None if held is None else ref(RefKind.INPUT, held)

    def _action(
        self, event: Executed, index: int, verify: tuple[Condition, ...]
    ) -> ActionNode:
        target = None if event.target is None else self._target(event.target)
        chosen = self._choice(event)
        value = (
            chosen
            if chosen is not None
            else None
            if event.value is None
            else self._value(event.value)
        )
        record = None if event.record is None else self._record(event.record)
        destination = (
            None if event.destination is None else self._destination(event.destination)
        )
        into = None if event.into is None else self._value(event.into)
        careful = event.risk is Risk.RISKY or event.approved or event.flagged
        careful = (
            careful
            or operation_limit(
                self.profile, event.operation or Operation(event.kind, event.route)
            )
            is Limit.RISKY
        )
        return ActionNode(
            node_id=f"s{index + 1}",
            kind=event.kind,
            route=event.route,
            target=target,
            value=value,
            destination=destination,
            effect=self._effect(event.kind, event.effect),
            record=record,
            result_record=_result_record(record, verify, self.targets),
            into=into,
            approval=Approval.EACH_RUN if careful else Approval.NONE,
            mandatory=careful,
            requires=self._conditions(event.requires, event.step),
            verify=verify,
            transitions=(),
            value_match=Match.EQUALS if chosen is None else Match.CONTAINS,
            introduced=event.introduced,
            extraction=None
            if event.extraction is None
            else Extraction(
                self._parts(event.extraction[0]), self._parts(event.extraction[1])
            ),
            binding=event.operation.binding if event.operation else "",
            business=event.operation.business if event.operation else "",
        )

    # Parameterize recorded values and target text.

    def _parts(self, text: str) -> tuple[Ref, ...]:
        """Parameterize a boundary without guessing among overlapping sources."""
        sources = [
            (ref(RefKind.INPUT, name), sample) for name, sample in self.samples.items()
        ] + [
            (reference, value)
            for reference, value in self.recorder._kept
            if (reference.kind, reference.name) in self.assigned
        ]
        found = sorted(
            (
                (start, end, reference)
                for reference, sample in sources
                for start, end in matching.positions(text, sample)
            ),
            key=lambda item: (item[0], item[1]),
        )
        parts: list[Ref] = []
        at = 0
        for start, end, reference in found:
            if start < at:
                raise _Unrepresentable
            if start > at:
                parts.append(self._text(text[at:start], Place.RECORD))
            parts.append(reference)
            at = end
        if at < len(text):
            parts.append(self._text(text[at:], Place.RECORD))
        return tuple(parts)

    def _effect(self, kind: ActionKind | None, effect: str | None) -> str | None:
        """Keep declared restrictions without persisting arbitrary model labels."""
        declared = self.profile.effects.get(kind, {}) if kind is not None else {}
        if effect is None or self._private(effect):
            return None
        if effect in declared:
            return effect
        if isinstance(self.recorder._origin, Synthetic):
            return effect
        if effect in self.recorder._safe_text:
            return effect
        if self._may_collect(effect):
            # An effect is the run's own label for an operation, never page
            # data, so it is confirmed as a control's text.
            self.candidates.setdefault(effect, set()).add(Place.CONTROL)
            return effect
        return None

    def _text(self, text: str, place: Place = Place.RECORD) -> Ref:
        """Turn recorded text into a reference, refusing an embedded input value.

        A text equal to an input, or to a value the run kept, is saved as
        that reference. A discovered text the run's confirmed words do not
        cover is refused, or kept as a candidate when collecting. A candidate
        is saved only once it is confirmed, so a text holding an output, a
        remembered fact, or any other excluded value is refused even then.
        """
        exact = [name for name, sample in self.samples.items() if text == sample]
        kept = {
            reference
            for reference, value in self.recorder._kept
            if value == text and (reference.kind, reference.name) in self.assigned
        }
        # Equal strings do not say where a value came from. A text that
        # matches more than one input, more than one kept fact, or both an
        # input and a fact is refused rather than given to one of them by
        # the order they are checked in (rule 9).
        if len(exact) + len(kept) > 1:
            raise _Unrepresentable("its text matches more than one input or kept value")
        if exact:
            return ref(RefKind.INPUT, exact[0])
        if kept:
            return kept.pop()
        if self._private(text):
            raise _Unrepresentable(
                "its literal text holds an invocation value, output, fact, or secret"
            )
        if (
            isinstance(self.recorder._origin, DiscoveryRun)
            and text not in self.recorder._safe_text
        ):
            if not self._may_collect(text):
                raise _Unrepresentable("its text is not confirmed website text")
            self.candidates.setdefault(text, set()).add(place)
        return constant(text)

    def _word(self, text: str) -> str | None:
        """Name the one input a longer text holds once as whole words.

        Such a text, like ``Sam Lee (U-7)``, is saved as a reference to
        that input matched by ``CONTAINS``, and its other words are never
        written. A text holding any other input value, or this one twice or
        inside a longer word, has no such reading.
        """
        held = [
            name for name, sample in self.samples.items() if sample and sample in text
        ]
        if len(held) != 1 or text == self.samples[held[0]]:
            return None
        if len(word_positions(text, self.samples[held[0]])) != 1:
            return None
        return held[0]

    def _painted_name(self, text: str) -> tuple[Ref, bool] | None:
        """Name the one available reference a painted line holds once.

        Recognition and a painted label can differ from the value in case
        and spacing, as the option "Paper" does from the input "paper", and
        replay compares painted text without either. A remembered value is
        available only after its read. Multiple sources or repeated values
        are refused rather than assigned a reference by precedence.
        """
        sources = {
            (ref(RefKind.INPUT, name), sample) for name, sample in self.samples.items()
        } | {
            (reference, value)
            for reference, value in self.recorder._kept
            if (reference.kind, reference.name) in self.assigned
        }
        squashed = "".join(text.split()).casefold()
        equal = {
            reference
            for reference, sample in sources
            if sample and "".join(sample.split()).casefold() == squashed
        }
        if len(equal) > 1:
            raise _Unrepresentable
        if equal:
            return equal.pop(), True
        held = [
            (reference, positions)
            for reference, sample in sources
            if (positions := matching.positions(text, sample, painted=True))
        ]
        if not held:
            return None
        if len(held) != 1 or len(held[0][1]) != 1:
            raise _Unrepresentable
        return held[0][0], True

    def _name(self, text: str, place: Place) -> tuple[Ref, bool]:
        """Turn a locator's text into a reference, and say if it holds an input."""
        held = self._word(text)
        if held is not None:
            return ref(RefKind.INPUT, held), True
        return self._text(text, place), False

    def _may_collect(self, text: str) -> bool:
        return self.recorder._collect and bool(text.strip()) and not self._private(text)

    def _private(self, text: str) -> bool:
        return any(holds(text, value) for value in self.recorder._excluded)

    def _value(self, sample: ValueSample) -> Ref:
        if sample.source is RefKind.CONSTANT:
            if not sample.text:
                raise _Unrepresentable
            return self._text(sample.text)
        if sample.source is RefKind.SECRET:
            if sample.name not in self.profile.secrets:
                raise _Unrepresentable
            if sample.name not in self.secrets:
                self.secrets.append(sample.name)
            return ref(RefKind.SECRET, sample.name)
        if sample.name not in self.declared.get(sample.source, set()):
            raise _Unrepresentable
        return ref(sample.source, sample.name)

    def _target(self, sample: TargetEvidence) -> str:
        match sample:
            case ScreenSample():
                raise _Unrepresentable
            case VisualSample():
                return self._visual(sample)
            case TextSample():
                return self._keep(self._painted(sample))
            case FocusSample():
                return self._keep(FocusTarget("", sample.route, sample.frame))
            case TargetSample():
                return self._keep(self._structural(sample, ""))

    def _structural(self, sample: TargetSample, target_id: str) -> StructuralTarget:
        for name in sample.frame:
            # Frame names cannot carry invocation references in this schema.
            if self._text(name, Place.CONTROL).kind is not RefKind.CONSTANT:
                raise _Unrepresentable
        # A row names its record, so text in or naming a row is a record's.
        in_row = sample.scope is not None and sample.scope.kind is ScopeKind.ROW
        accessible = sample.form is LocatorForm.ACCESSIBILITY
        scope = None
        contains = sample.match is Match.CONTAINS
        if sample.scope is not None:
            place = Place.RECORD if in_row else Place.CONTROL
            named, held = self._name(sample.scope.name, place)
            contains = contains or held
            # A column binds only a row found by a reference's value.
            column = sample.scope.column if named.kind is not RefKind.CONSTANT else 0
            scope = ScopeSpec(sample.scope.kind, named, column)
        if (
            accessible
            and sample.role not in STRUCTURAL_ROLES
            and self._text(sample.role, Place.CONTROL).kind is not RefKind.CONSTANT
        ):
            raise _Unrepresentable
        labels = (
            Place.CONTROL
            if not in_row and (not accessible or sample.role in INTERFACE_ROLES)
            else Place.RECORD
        )
        name = None
        if accessible and sample.name:
            name, held = self._name(sample.name, labels)
            contains = contains or held
        return StructuralTarget(
            target_id=target_id,
            route=sample.route,
            form=sample.form,
            role=sample.role if accessible else "",
            name=name,
            tag="" if accessible else sample.tag,
            attribute=None if accessible else sample.attribute,
            value=None if accessible else self._dom_value(sample, labels),
            frame=sample.frame,
            scope=scope,
            match=Match.CONTAINS if contains else Match.EQUALS,
        )

    def _painted(self, sample: TextSample) -> TextTarget:
        """Turn a painted line into a text target, saving no record's data.

        A label is the website's own words, confirmed like any control's
        text. A line that holds one available reference once is saved as
        that reference, found by whole words. The rest of the line is never
        written.
        """
        if sample.label:
            text = self._text(sample.text, Place.CONTROL)
            if text.kind is not RefKind.CONSTANT:
                raise _Unrepresentable
            return TextTarget("", sample.route, sample.frame, text, label=True)
        named = self._painted_name(sample.text)
        text, held = named or self._name(sample.text, Place.CONTROL)
        target_type = RadioTarget if sample.radio else TextTarget
        return target_type(
            "",
            sample.route,
            sample.frame,
            text,
            Match.CONTAINS if held else Match.EQUALS,
            dx=sample.dx,
            dy=sample.dy,
        )

    def _dom_value(self, sample: TargetSample, place: Place) -> Ref:
        """Turn a DOM target's attribute value into a reference.

        A slot the collector named by a column's position, such as ``td#2``,
        holds no page text, so it needs no confirmation.
        """
        if sample.attribute is DomAttribute.SLOT and positional_slot(sample.value):
            return constant(sample.value)
        return self._text(sample.value, place)

    def _visual(self, sample: VisualSample) -> str:
        permit = sample.permit
        if permit is None or (
            permit is StoragePermit.SYNTHETIC
            and not isinstance(self.recorder._origin, Synthetic)
        ):
            raise _Unrepresentable
        stored = template(f"v{len(self.templates) + 1}", sample.image, permit)
        for held in self.templates:
            if held.sha256 == stored.sha256:
                stored = held
                break
        else:
            self.templates.append(stored)
        return self._keep(
            VisualTarget("", sample.route, stored.template_id, sample.frame)
        )

    def _keep(self, target: Target) -> str:
        """Return the id of an equal target already kept, or keep this one."""
        for held in self.targets:
            if dataclasses.replace(held, target_id="") == target:
                return held.target_id
        kept = dataclasses.replace(target, target_id=f"t{len(self.targets) + 1}")
        self.targets.append(kept)
        return kept.target_id

    def _record(self, sample: RecordSample) -> RecordSpec:
        value = self._value(sample.value)
        if value.kind not in {RefKind.INPUT, RefKind.VARIABLE}:
            raise _Unrepresentable
        source = self._keep(self._structural(sample.source, ""))
        if sample.prefix or sample.suffix:
            # The text around an identifier is usually the record's own, such
            # as a name. It is read from the live page on each replay instead.
            return RecordSpec(source, value, sample.relation, match=Match.CONTAINS)
        return RecordSpec(source, value, sample.relation)

    def _destination(self, sample: DestinationSample) -> Destination:
        for name, _ in sample.query:
            if self._text(name, Place.CONTROL).kind is not RefKind.CONSTANT:
                raise _Unrepresentable
        return Destination(
            sample.route,
            tuple(Param(name, self._value(value)) for name, value in sample.params),
            tuple(Param(name, self._value(value)) for name, value in sample.query),
            None if sample.fragment is None else self._value(sample.fragment),
            sample.origin,
        )

    def _conditions(
        self, samples: tuple[CheckSample, ...], step: int
    ) -> tuple[Condition, ...]:
        conditions: list[Condition] = []
        for sample in samples:
            try:
                conditions.append(self._condition(sample))
            except _Unrepresentable as refused:
                kind = sample.kind.value.replace("_", " ")
                why = str(refused) or "cannot be saved"
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK, step, f"a {kind} check: {why}"
                    )
                )
                return ()
        return tuple(conditions)

    def _condition(self, sample: CheckSample) -> Condition:
        match sample.kind:
            case CheckKind.BOUND if sample.value is not None:
                return Bound(self._value(sample.value))
            case CheckKind.VALUE_IS if (
                sample.value is not None and sample.expected is not None
            ):
                expected, match, once = self._held(
                    sample.expected, sample.match, sample.once
                )
                return ValueIs(
                    self._value(sample.value),
                    self._value(expected),
                    match,
                    sample.purpose,
                    once,
                )
            case CheckKind.AT_ROUTE:
                return AtRoute(sample.route)
            case CheckKind.DIALOG_OPEN:
                return DialogOpen(
                    sample.dialog,
                    None if sample.value is None else self._value(sample.value),
                )
            case _:
                pass
        if sample.target is None:
            raise _Unrepresentable
        target = self._target(sample.target)
        if sample.kind is CheckKind.PRESENT:
            return Present(target)
        if sample.kind is CheckKind.ABSENT:
            return Absent(target)
        if sample.value is None:
            raise _Unrepresentable
        value, match, once = self._held(sample.value, sample.match, sample.once)
        return Shows(
            target,
            self._value(value),
            match,
            None if sample.record is None else self._record(sample.record),
            sample.purpose,
            once,
        )

    def _held(
        self, value: ValueSample, match: Match, once: bool
    ) -> tuple[ValueSample, Match, bool]:
        """Check the one input a longer text holds, instead of the whole text.

        A field or a line often shows more than the input a person gave, as
        a member box shows ``10001 Alex Morgan`` after ``10001`` was typed.
        Checking the whole text would save a record's name with it, and an
        exact check would hold for that member only. The input is checked
        instead, once and as whole words, which holds for every member and
        still proves the one the invocation named.
        """
        if value.source is not RefKind.CONSTANT:
            return value, match, once
        held = self._word(value.text)
        if held is None:
            return value, match, once
        return ValueSample(RefKind.INPUT, name=held), Match.CONTAINS, True

    # Assemble the validated capability graph.

    def _result(self, finished: Finished) -> ResultNode:
        if finished.verifier is not Verifier.EXECUTOR and not finished.confirmed:
            self.gap(Gap.UNVERIFIED_RESULT, 0)
        if finished.confirm_inputs and (
            not finished.confirmed or finished.verifier is not Verifier.PERSON
        ):
            self.gap(Gap.UNVERIFIED_RESULT, 0)
        checks = self._conditions(finished.checks, 0)
        if finished.outcome:
            return ResultNode("done", ResultKind.OUTCOME, finished.outcome, checks)
        return ResultNode("done", ResultKind.SUCCESS, "", checks)

    def _confirmation(
        self, steps: list[_Step], result: ResultNode, finished: Finished
    ) -> _Step:
        """Preserve a person's result obligations before the checked result.

        The replay reads every check itself, and the step goes on only once
        they hold. The person confirms the record tie and required values
        the surface could not show.
        """
        last = steps[-1].node
        assert isinstance(last, (ActionNode, HumanNode))  # noqa: S101  steps hold these two
        node = HumanNode(
            "confirm",
            last.route,
            HelpReason.RECORD_EVIDENCE,
            None,
            None,
            True,
            (_edge("", result.checks),),
        )
        if finished.confirm_inputs:
            node = ResultConfirmationNode(
                node.node_id,
                node.route,
                node.reason,
                node.performs,
                node.effect,
                node.mandatory,
                node.transitions,
                confirm_inputs=finished.confirm_inputs,
            )
        return _Step(steps[-1].step, node)

    def _restrictions(self, steps: list[_Step]) -> tuple[SavedRestriction, ...]:
        saved: list[SavedRestriction] = []
        for event in self.events:
            if not isinstance(event, Learned):
                continue
            for item in _translate(event.restriction, steps):
                if item not in saved:
                    saved.append(item)
        return tuple(saved)

    def _link(
        self,
        steps: list[_Step],
        result: ResultNode,
        restrictions: tuple[SavedRestriction, ...],
    ) -> tuple[Node, ...]:
        nodes: list[Node] = []
        for index, item in enumerate(steps):
            following = (
                steps[index + 1].node.node_id if index + 1 < len(steps) else "done"
            )
            node = item.node
            if isinstance(node, HumanNode):
                denied = strictest(restrictions, node) is Limit.DENY
                if denied or self._learned_deny(item.stands_for):
                    self.gap(Gap.DENIED_STEP, item.step)
                edge = dataclasses.replace(node.transitions[0], to=following)
                nodes.append(dataclasses.replace(node, transitions=(edge,)))
                continue
            assert isinstance(node, ActionNode)  # noqa: S101  steps hold these two
            aimed = next(
                (item for item in self.targets if item.target_id == node.target), None
            )
            limit = strictest(
                restrictions,
                node,
                pixel=isinstance(aimed, (VisualTarget, TextTarget, FocusTarget)),
            )
            if limit is Limit.DENY:
                self.gap(Gap.DENIED_STEP, item.step)
            careful = node.approval is Approval.EACH_RUN or limit is Limit.RISKY
            nodes.append(
                dataclasses.replace(
                    node,
                    approval=Approval.EACH_RUN if careful else Approval.NONE,
                    mandatory=careful,
                    transitions=(_edge(following, ()),),
                )
            )
        nodes.append(result)
        return tuple(nodes)

    def _learned_deny(self, operations: tuple[Operation, ...]) -> bool:
        """Report whether a learned deny covers any operation a person stands in for.

        This is checked against the operations themselves, before any
        translation onto reusable targets, so a deny is never lost because
        the person's step names no target.
        """
        return any(
            event.restriction.limit is Limit.DENY
            and policy.covers(event.restriction.operation, operation)
            for event in self.events
            if isinstance(event, Learned)
            for operation in operations
        )

    def _assemble(
        self,
        nodes: tuple[Node, ...],
        restrictions: tuple[SavedRestriction, ...],
        steps: list[_Step],
    ) -> Capability:
        recorder = self.recorder
        capability_id, version = recorder._identity
        inputs, outputs, variables = recorder._fields
        scope = self.profile.scope
        markers = self._conditions(self.checks.get(0, ()), 0)
        first = steps[0].node
        entry_route = first.route if isinstance(first, (ActionNode, HumanNode)) else ""
        origin = recorder._origin
        return Capability(
            schema_version=SCHEMA_VERSION,
            capability_id=capability_id,
            version=version,
            application=Application(
                profile_id=self.profile.profile_id,
                surface=SurfaceKind.BROWSER,
                origin=scope.origin,
                entry_route=entry_route,
                markers=markers,
            ),
            provenance=Provenance(
                kind=(
                    ProvenanceKind.DISCOVERED
                    if isinstance(origin, DiscoveryRun)
                    else ProvenanceKind.SYNTHETIC
                ),
                run=origin.run if isinstance(origin, DiscoveryRun) else "",
                recorder=RECORDER,
                review=Review.DRAFT,
            ),
            inputs=inputs,
            outputs=outputs,
            variables=variables,
            secrets=tuple(self.secrets),
            outcomes=recorder._outcomes,
            templates=tuple(self.templates),
            targets=tuple(self.targets),
            entry=nodes[0].node_id,
            nodes=nodes,
            restrictions=restrictions,
            limits=recorder._limits,
        )


def _takes_over(segment: HumanSegment) -> bool:
    return segment.reason is not HelpReason.AUTHENTICATION


def _result_record(
    record: RecordSpec | None, verify: tuple[Condition, ...], targets: list[Target]
) -> RecordSpec | None:
    """Return where a record-bound step's result showed its record, if one check did.

    The run must have verified, after the step, that a structural target
    showed the record's value, exactly or once as whole words. Two such
    checks are ambiguous and give none, so the replay falls back on the
    record's own source, which a step that leaves its page no longer shows.
    """
    if record is None:
        return None
    shown = [
        (condition.target, condition.match)
        for condition in verify
        if isinstance(condition, Shows)
        and condition.value == record.value
        and any(
            isinstance(target, StructuralTarget)
            and target.target_id == condition.target
            and target.tag not in {"input", "textarea", "select"}
            and target.role not in {"textbox", "searchbox", "combobox", "spinbutton"}
            for target in targets
        )
    ]
    if len(shown) != 1:
        return None
    target, match = shown[0]
    return RecordSpec(target, record.value, record.relation, match=match)


def _dispatch(event: Executed) -> Dispatch:
    """Return whether an executed event's input reached the surface.

    An ``ok`` outcome was delivered whatever the field says. Otherwise an
    explicit ``dispatch`` wins, and a sent operation that did not come back
    ``ok`` is unknown, because its effect is.
    """
    if event.outcome is Outcome.OK:
        return Dispatch.SENT
    if event.dispatch is Dispatch.NOT_SENT:
        return Dispatch.NOT_SENT
    if event.dispatch is not None or event.outcome in _UNKNOWN:
        return Dispatch.UNKNOWN
    return Dispatch.NOT_SENT


_UNKNOWN = frozenset({Outcome.UNCERTAIN, Outcome.SURFACE_ERROR, Outcome.HANDOFF})
"""Outcomes that leave open whether input reached the application."""


def _checks_by_step(
    events: tuple[ExecutionEvent, ...],
) -> dict[int, tuple[CheckSample, ...]]:
    found: dict[int, tuple[CheckSample, ...]] = {}
    for event in events:
        if isinstance(event, Verified):
            found[event.step] = found.get(event.step, ()) + event.checks
    return found


def _edge(to: str, when: tuple[Condition, ...]) -> Edge:
    return Edge(to, when, EdgeOrigin.OBSERVED, None)


def _translate(
    restriction: Restriction, steps: list[_Step]
) -> tuple[SavedRestriction, ...]:
    """Carry a learned restriction onto reusable targets, never narrowing it.

    Discovery binds a restriction to an ``Operation`` whose element ids end
    with their document, so none of them is kept. A restriction on a screen
    coordinate covers every action on its route, as it did during discovery.
    One on a form submission covers every click and every Enter on its route,
    because a replay cannot tell which submission a control performs. One on
    a single control covers each recorded target it matched, and its whole
    route when it matched none.
    """
    held = restriction.operation
    base = SavedRestriction(
        RestrictionScope.ROUTE,
        held.route,
        (),
        None,
        "",
        restriction.limit,
        restriction.source,
    )
    if held.business:
        return (
            dataclasses.replace(
                base,
                scope=RestrictionScope.OPERATION,
                target=held.business,
                kinds=tuple(kind for kind in ActionKind if kind not in policy.LOOKS),
            ),
        )
    if held.target == "screen" and held.kind not in policy.LOOKS:
        # Discovery covered every input on the screen, and no look or read
        # (rule 6), so the saved restriction names every input action type.
        inputs = tuple(kind for kind in ActionKind if kind not in policy.LOOKS)
        return (dataclasses.replace(base, kinds=inputs),)
    if held.target == "screen" or not held.resolved:
        return (base,)
    if (
        held.submission
        or held.submission_as
        or held.uncertain
        or (held.kind is ActionKind.PRESS_KEY and held.key == ENTER)
    ):
        return (
            dataclasses.replace(
                base, kinds=(ActionKind.CLICK, ActionKind.PRESS_KEY), key=ENTER
            ),
        )
    matched = sorted(
        {
            item.node.target
            for item in steps
            if isinstance(item.node, ActionNode)
            and item.node.target is not None
            and item.operation is not None
            and policy.covers(held, item.operation)
        }
    )
    narrowed = dataclasses.replace(base, kinds=(held.kind,), key=held.key)
    if not matched:
        return (narrowed,)
    return tuple(
        dataclasses.replace(narrowed, scope=RestrictionScope.TARGET, target=target)
        for target in matched
    )
