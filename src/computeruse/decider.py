"""The interface that chooses the next move and limits what it can see.

This protocol separates model decisions from execution. Replay executes a
saved capability without configuring or calling a model.

A decision is one of six things, and each is a typed value rather than prose.
The model may ask to look at the surface with a named tool, propose one
action, report that an action it already took needs approval in future, keep
a fact it will need later, ask for a human with a named reason, or report the
goal met with the checks that show it. The executor does not interpret free
text. ``rationale`` and ``detail`` exist only for the operator channel and
debugging; no control-flow branch reads them.

Before the first decision, the same decider states what the goal requires as
a ``Task``: the records it is about and the outputs it asks for. That is read
from the goal alone, before any page is seen, and the run holds it unchanged
until it ends, so a finish claim is measured against it and cannot redefine
it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from computeruse import matching
from computeruse.actions import (
    Action,
    AxLocator,
    Capabilities,
    DomLocator,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    PendingDialog,
    RecordEvidence,
    ScreenTarget,
)
from computeruse.escalation import Trigger
from computeruse.policy import Restriction
from computeruse.profile import ActionKind, Limit, Risk


class ModelError(RuntimeError):
    """The decider could not produce a decision this executor can act on.

    Raised for a transport failure, a malformed response, or a response that
    names a tool, mode, action type, or target the run never offered. The loop
    ends the run on this rather than retrying blindly, because a model that
    answered outside its contract has not been understood.
    """


class InvalidDecisionError(ModelError):
    """A provider response needs correction before any action can run.

    ``tool`` names the tool the model called, when it is one the run offered,
    so a journal can say which kind of answer was refused without its text.
    """

    def __init__(self, message: str, tool: str = "") -> None:
        super().__init__(message)
        self.tool = tool


@dataclasses.dataclass(frozen=True, slots=True)
class Turn:
    """One completed step in the history supplied to the next decision.

    A turn records either an action or an observation, because the model needs
    to know which tools it has already run against this screen. That history
    is also what an intervention request is checked against: a claim that
    observation was insufficient is only meaningful next to what was observed.
    """

    step: int
    action: Action | None = None
    observation: ObservationRequest | None = None
    outcome: Outcome | None = None
    status: ObservationStatus | None = None
    extracted: str | None = None
    detail: str | None = None
    side_effects: tuple[str, ...] = ()


MAX_FACT_VALUE = 300
"""The longest value working memory keeps. A longer reading is cut, not kept."""


class FactOrigin(StrEnum):
    """The source of a working-memory fact."""

    GOAL = "goal"
    CONTROL = "control"
    READ = "read"


@dataclasses.dataclass(frozen=True, slots=True)
class Fact:
    """A value kept for later decisions, with its source.

    Working memory is live context for the next decisions. It grants nothing,
    the gate never reads it, and no journal event carries a key or a value.
    ``origin`` says whether the value is in the goal, was kept by the model
    from a control an observation showed, or was returned by a ``read`` the
    executor performed. ``source`` is the control it was read from, when
    there is one, on the screen ``route`` names, at ``step``. ``location`` and
    ``epoch`` say which page it was read on and how many page-changing actions
    the run had taken by then, so the loop can tell whether it may have
    changed since. ``may_change`` marks a value the application can change,
    such as a balance: before an action uses it, the loop reads the source
    again, and refuses when it cannot. A secret is never a fact; it is only
    ever named.

    ``record`` is the record the source was tied to when the fact was kept,
    checked by the loop against the observation or by the surface for a
    read, such as the member number in the same row. It is the only thing
    that says which record a fact belongs to: its key and ``may_change`` are
    the model's words. ``page`` is the window it was read in and ``control``
    the element id the surface issued, and ``displayed`` says the page showed
    the value rather than a field holding what was typed. ``partial`` says
    the value is one part of its control's text, held there once as whole
    words, such as an identifier inside a sentence; a refresh then checks
    the value is still there, and never takes the whole text (rule 11).
    ``painted`` says the value is a line recognition read from a screenshot,
    so a check citing it compares as a painted line does.
    """

    key: str
    value: str
    step: int
    route: str
    source: AxLocator | DomLocator | None = None
    may_change: bool = True
    origin: FactOrigin = FactOrigin.CONTROL
    location: str = ""
    epoch: int = 0
    record: RecordEvidence | None = None
    page: str = ""
    control: str = ""
    displayed: bool = False
    partial: bool = False
    painted: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.value:
            raise ValueError("a fact needs a key and a value")
        if len(self.value) > MAX_FACT_VALUE:
            raise ValueError(f"a fact keeps at most {MAX_FACT_VALUE} characters")


@dataclasses.dataclass(frozen=True, slots=True)
class TaskRecord:
    """One record the goal is about, such as a member or an account.

    ``name`` is what the task calls it and ``value`` is its identifier as the
    goal writes it. ``within`` names another record of the task this one
    belongs to, such as an account within a member, and then the run must
    show that the two are tied, not only that both appear.
    """

    name: str
    value: str
    within: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class TaskOutput:
    """One value the goal asks for, and the record it belongs to, if any."""

    name: str
    of: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class TaskRequirement:
    """A required end state or operating context, settled from the goal.

    ``context`` marks who is signed on or which institution the work happens
    in. A context requirement is checked when the application shows it, and a
    check that shows another value fails the claim. An application that never
    shows it cannot prove it, so the run then takes it from the goal and says
    so in its result. Every other requirement is a state the task must reach,
    such as an approved status, and must be shown.
    """

    name: str
    expected: str
    of: str = ""
    context: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """The requirements a run must satisfy to complete the goal.

    These describe the task, not what the run may do: the profile alone says
    that. The loop checks that every identifier appears in the goal, and a
    completion must cover each record and output, tied to one another as
    declared. ``question`` is set when the goal can be read more than one
    way, and a person settles it before the run starts.

    A task with no records is a task about no particular record, such as
    reading a page title, and nothing about records is required of it.
    """

    records: tuple[TaskRecord, ...] = ()
    outputs: tuple[TaskOutput, ...] = ()
    question: str = ""
    requirements: tuple[TaskRequirement, ...] = ()
    changes: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class HumanReturn:
    """The latest hand-back, retained across looks without asserting success."""

    intervention: str
    took_control: bool
    actions: tuple[str, ...]
    verified: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Transcript:
    """The inputs supplied to the decider for one decision.

    ``allowed`` and ``permitted_modes`` come straight from the profile, so a
    tool or action type the operator never declared is never even offered as a
    choice. Both are still checked again at the gate: offering nothing is
    convenience, refusing is policy.

    ``observations`` may be empty, which is the initial state. The model
    chooses the first observation tool. The loop does not capture both modes
    or take an unrequested screenshot.

    ``attempted_modes`` is the other half of that, and the two differ on
    purpose. A tool that returned nothing usable is in ``attempted_modes`` and
    not in ``observations``, so the model can see that structured has already
    been tried and come back empty rather than asking for it again. The
    alternate allowance is counted against attempts for the same reason: a
    failed look is a look.

    ``dialog`` is the dialog the page is waiting on, kept apart from
    ``observations`` because an observation of a page behind a dialog is not
    usable and yet the dialog is exactly what the next decision is about.
    ``record_bound`` lists the action types the operator requires to carry
    record evidence on this screen. ``effect_rules`` are the operator's effect
    exceptions, and ``restrictions`` are the ones this run has bound to
    particular operations so far, so the model knows what already needs a
    person before it proposes anything.

    ``capabilities`` is what the surface can perform, so nothing it cannot
    perform is offered; None means the caller supplied no adapter, and the
    profile alone decides. ``history`` holds every turn of the run. A decider
    shows the latest in full and older ones in brief, and ``memory`` holds the
    facts the run chose to keep, so an early value is not lost with its turn.
    ``task`` is what the goal requires, as settled before the first decision.
    """

    goal: str
    location: str
    allowed: Mapping[ActionKind, Risk]
    permitted_modes: tuple[ObservationMode, ...]
    observations: tuple[Observation, ...] = ()
    attempted_modes: tuple[ObservationMode, ...] = ()
    history: tuple[Turn, ...] = ()
    notices: tuple[str, ...] = ()
    alternates_remaining: int = 0
    steps_remaining: int = 0
    seconds_remaining: float = 0.0
    dialog: PendingDialog | None = None
    record_bound: tuple[ActionKind, ...] = ()
    effect_rules: Mapping[ActionKind, Mapping[str, Limit]] = dataclasses.field(
        default_factory=dict
    )
    restrictions: tuple[Restriction, ...] = ()
    secret_names: tuple[str, ...] = ()
    capabilities: Capabilities | None = None
    memory: tuple[Fact, ...] = ()
    task: Task = Task()
    human_return: HumanReturn | None = None
    inputs: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def observed_modes(self) -> tuple[ObservationMode, ...]:
        """Return the tools whose results are still in hand."""
        return tuple(observation.mode for observation in self.observations)


@dataclasses.dataclass(frozen=True, slots=True)
class Observe:
    """Look at the surface with one named tool before deciding anything else."""

    request: ObservationRequest
    rationale: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Propose:
    """Take one action against the surface.

    ``fact`` names the working-memory fact the action's value came from, so
    the loop can check that fact again before the value is used.
    """

    action: Action
    rationale: str = ""
    fact: str = ""
    after: tuple[ResultCheck, ...] = ()
    input_name: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Remember:
    """Keep one value for later decisions, read from a control or the goal.

    The loop checks ``value`` against the control ``source`` names in the
    current structured observation, or against the goal when there is no
    source, and refuses a value it cannot find there. ``record`` ties the
    source to the record it belongs to, checked like an action's record
    evidence, and is what a later check needs to use the fact for a record.
    """

    key: str
    value: str
    may_change: bool = True
    source: AxLocator | DomLocator | None = None
    record: RecordEvidence | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class RiskFinding:
    """The model's report that an action it already took needs approval.

    ``step`` names the step of that action in this run. ``effect`` says what
    the model now believes the operation accomplished. ``reason`` is its own
    words and stays in live working context. A finding only adds a
    restriction; there is no finding that removes one.
    """

    step: int
    effect: str
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class FlagRisk:
    """Record a restriction on an operation already tried. Nothing is re-run."""

    finding: RiskFinding


@dataclasses.dataclass(frozen=True, slots=True)
class AskHuman:
    """Hand the live session to a person, for a reason the console can route.

    The model may raise a risk the profile did not declare, which escalates
    something that would otherwise have run. It cannot lower one: an action
    the operator declared risky escalates whatever the model asks for.
    """

    trigger: Trigger
    detail: str = ""
    after: tuple[ResultCheck, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class FactRef:
    """A check's target that is a working-memory fact rather than a control.

    Two kinds of fact support a result this way: one kept from a control and
    marked as one that does not change, and one the executor read on the page
    the run is still on, with no page-changing action since. That is how a
    value read on an earlier screen, or outside the current screenshot, can
    back a claim. Only the fact's ``record`` ties it to a record.
    """

    key: str


class CheckKind(StrEnum):
    """What a check supports: a result, a record, or a required state."""

    RESULT = "result"
    RECORD = "record"
    REQUIREMENT = "requirement"
    STATE = "state"


class Match(StrEnum):
    """How a check compares what the page shows with what it expects."""

    EQUALS = "equals"
    CONTAINS = "contains"


@dataclasses.dataclass(frozen=True, slots=True)
class ResultCheck:
    """One thing the page must show for a finished run's claim to stand.

    A ``RESULT`` check names the output it supports, and ``expected`` must be
    that output's value. A ``RECORD`` check names a record the goal is about,
    such as a member or account number, and ``expected`` must appear in the
    goal. The loop reads every target again from the live page before the run
    can complete, and compares what it shows with ``expected``.

    A check whose target is a ``FactRef`` cites a working-memory fact. When
    the fact was kept on the page the run is on, the loop reads its control
    again through the gate, with its record evidence. Otherwise it accepts a
    fact kept from a control and marked as one that does not change, or a
    value the executor read on this page with no page-changing action since.

    ``record`` ties a result to the record it belongs to, such as a balance to
    the account number in its row. It is record evidence like an action's,
    and the surface checks it the same way while reading the target, so a
    balance from the row of another account does not pass. Its value must
    appear in the goal.

    Examples
    --------
    >>> cell = AxLocator("cell", "$4.00")
    >>> ResultCheck(CheckKind.RESULT, cell, "$4.00", output="balance").match
    <Match.EQUALS: 'equals'>
    >>> ResultCheck(CheckKind.RESULT, cell, "$4.00")
    Traceback (most recent call last):
        ...
    ValueError: a result check names the output it supports
    """

    kind: CheckKind
    target: AxLocator | DomLocator | ScreenTarget | FactRef
    expected: str
    match: Match = Match.EQUALS
    output: str = ""
    record: RecordEvidence | None = None
    requirement: str = ""

    def __post_init__(self) -> None:
        if not self.expected.strip():
            raise ValueError("a check must expect a value")
        if self.record is not None:
            if isinstance(self.target, (ScreenTarget, FactRef)):
                raise ValueError("record evidence needs a structured target")
            if self.record.source.frame != self.target.frame:
                raise ValueError("record evidence must sit in the target's frame")
        if self.kind is CheckKind.RESULT and not self.output:
            raise ValueError("a result check names the output it supports")
        if self.kind is CheckKind.RECORD and self.output:
            raise ValueError("a record check supports no single output")
        if (self.kind is CheckKind.REQUIREMENT) != bool(self.requirement):
            raise ValueError("a requirement check names the requirement it supports")
        if self.kind is CheckKind.REQUIREMENT and self.output:
            raise ValueError("a requirement check supports no single output")


def uncovered_requirements(
    task: Task, passing: Iterable[tuple[ResultCheck, str]]
) -> tuple[TaskRequirement, ...]:
    """Return task requirements absent from passing checks and their record ties.

    Callers supply only checks that the executor verified, with each check's
    observed record identity. Discovery and recording use this same coverage
    decision so a saved human obligation matches the live request.
    """
    checked = tuple(passing)
    records = {record.name: record for record in task.records}
    return tuple(
        required
        for required in task.requirements
        if not any(
            check.kind is CheckKind.REQUIREMENT
            and check.requirement == required.name
            and check.expected == required.expected
            and (
                required.of not in records
                or matching.same_identifier(bound, records[required.of].value)
            )
            for check, bound in checked
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Finish:
    """A completion claim with its values and supporting checks.

    The loop completes a run only after every check passes against the live
    page. A claim with no checks, or with a check that fails, is refused and
    reported back, so the model can correct it or ask for a person. When all
    checks pass but leave task obligations uncovered, the loop immediately
    asks a person to confirm those obligations. Resume requests more work;
    only explicit approval can complete a result with missing evidence.

    ``outcome`` names a business outcome from ``OUTCOMES`` instead of outputs,
    such as a record the application shows does not exist. Such a claim has
    no outputs, and its checks must show the outcome for the goal's record.
    """

    outputs: Mapping[str, str]
    rationale: str = ""
    checks: tuple[ResultCheck, ...] = ()
    outcome: str = ""


type Decision = Observe | Propose | FlagRisk | Remember | AskHuman | Finish


@runtime_checkable
class Labeller(Protocol):
    """Assign an effect name to a click performed by a person.

    A decider that implements this lets a person's single click become an
    automated step. The label is judged by the gate exactly as the model's
    own labels are, so a deny rule still holds. A decider without it leaves
    the person's click a person's step.
    """

    def label_effect(self, role: str, name: str, route: str) -> str | None:
        """Return a short lowercase effect name for the click, or None."""
        ...


class Decider(Protocol):
    """Interpret the goal, then choose each decision."""

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        """Return what ``goal`` requires, from the goal alone.

        ``notices`` says why an earlier answer was refused, or what a person
        said about it. Must not look at or act on the surface.
        """
        ...

    def decide(self, transcript: Transcript) -> Decision:
        """Return the next decision. Must not act on the surface itself."""
        ...
