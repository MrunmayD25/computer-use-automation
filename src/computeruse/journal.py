"""Sanitized evidence of run decisions, refusals, and actions.

The event types exclude sensitive data instead of scrubbing it afterwards.
An event records the action type, anonymous route reference, and outcome. Goals, model
prose, and locator text stay in live working context because any of them may
contain field values or data read off the surface. A secret is recorded by
the name the profile gave it, never its value.

Perception follows the same rule. An event says which tool ran, how complete
its result was, and why it ran. It carries no screenshot, DOM text, or
accessible names. Those values may contain the record itself, while the
journal persists after the run.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any, Protocol

from computeruse.actions import (
    Action,
    ObservationMode,
    ObservationProvenance,
    ObservationStatus,
    Outcome,
    SecretRef,
)
from computeruse.diagnostics import FailureEvidence
from computeruse.escalation import (
    Ask,
    CheckFailure,
    Command,
    HandoffOutcome,
    Interruption,
    Owner,
    Trigger,
    Verdict,
    Via,
)
from computeruse.evidence import EvidenceError as EvidenceError
from computeruse.evidence import EvidenceWriter, emit, safe_record
from computeruse.manual import Detail, ManualKind, Widget
from computeruse.policy import Denial, Source
from computeruse.profile import ActionKind


class ValueSource(StrEnum):
    """The recorded source of an action value."""

    NONE = "none"
    LITERAL = "literal"
    SECRET = "secret"  # noqa: S105  the member names a source, not a value


@dataclasses.dataclass(frozen=True, slots=True)
class RunStarted:
    """The start of a run under a named policy, without its goal."""

    profile_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class Decided:
    """A decision, excluding the free-text reasoning kept in memory."""

    step: int
    kind: ActionKind | None


@dataclasses.dataclass(frozen=True, slots=True)
class Refused:
    """The gate rejected an action before it reached the surface."""

    step: int
    kind: ActionKind
    reason: Denial


@dataclasses.dataclass(frozen=True, slots=True)
class Observed:
    """The outcome and coverage of one observation.

    ``alternate`` marks a look that spent part of the profile's alternate
    allowance, which is what makes an over-observing run visible in evidence
    rather than only in a token bill.
    """

    step: int
    mode: ObservationMode
    status: ObservationStatus
    provenance: ObservationProvenance
    alternate: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Failed:
    """A collaborator failure that ended the run."""

    step: int
    stage: str


@dataclasses.dataclass(frozen=True, slots=True)
class Escalated:
    """The result of offering control to a person."""

    step: int
    trigger: Trigger
    outcome: HandoffOutcome
    intervention: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Commanded:
    """The result of an operator command sent to the run's control.

    Refused commands are recorded too, so a late approval that was turned
    away is in the evidence. The operator's note is not.
    """

    step: int
    command: Command
    verdict: Verdict
    via: Via
    intervention: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Paused:
    """Automation stopped and the run waited for a person.

    ``ask`` says whether the run waited for an approval, for help, or for the
    operator's own stop or takeover. The reason and the proposal stay in the
    live operator channel.
    """

    step: int
    intervention: str
    ask: Ask
    trigger: Trigger | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Transferred:
    """A transfer of the live session between automation and a person."""

    step: int
    intervention: str
    owner: Owner


@dataclasses.dataclass(frozen=True, slots=True)
class ManualAction:
    """One action a person took in the live session.

    The control is a category from a closed list and the frame is a position,
    so nothing here is text the page wrote. What the person typed, which key
    they pressed, and what the control was called are never recorded.
    """

    step: int
    intervention: str
    sequence: int
    kind: ManualKind
    control: Widget
    route: str
    frame: tuple[int, ...]
    secret: bool
    owned: bool
    detail: Detail = Detail.NONE


@dataclasses.dataclass(frozen=True, slots=True)
class HandedBack:
    """The checks performed when a person returned the session.

    ``passed`` false means the person kept control, for ``failure``.
    ``interrupted`` says what became of the operation the pause interrupted.
    """

    step: int
    intervention: str
    passed: bool
    failure: CheckFailure | None
    events: int
    gaps: int
    interrupted: Interruption


@dataclasses.dataclass(frozen=True, slots=True)
class Superseded:
    """A discarded decision that arrived after control changed hands."""

    step: int


@dataclasses.dataclass(frozen=True, slots=True)
class Acted:
    """An action reached the surface.

    ``target_kind`` names how the control was addressed, such as
    ``"AxLocator"``. The locator's own role, name, frame, and scope text stay
    out, because each of them can quote a member's record.
    """

    step: int
    kind: ActionKind
    route: str
    outcome: Outcome
    flagged: bool = False
    value_source: ValueSource = ValueSource.NONE
    secret_name: str | None = None
    target_kind: str | None = None
    side_effects: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Flagged:
    """The run bound a restriction to an operation.

    ``action_step`` is the step of the action the restriction is bound to.
    The effect label and the model's reason are left out: both are the
    model's own words about the screen.
    """

    step: int
    action_step: int
    kind: ActionKind
    route: str
    source: Source


@dataclasses.dataclass(frozen=True, slots=True)
class Corrected:
    """An unusable decision returned to the decider for correction.

    Why it was refused is told to the model and stays out of the journal,
    because the reason can quote what the model wrote. ``tool`` is the name of
    the tool the model called when the run offered it, ``unknown`` for any
    other name, and ``none`` when no single call could be read.
    """

    step: int
    tool: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Verified:
    """The loop checked a finish claim against the live page.

    Only counts are recorded. The outputs, the expected values, and what the
    page showed are member data and stay in the run's result. ``uncovered``
    counts the task's requirements that the passing checks did not cover.
    """

    step: int
    checks: int
    passed: int
    uncovered: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class Interpreted:
    """The requirements derived from the goal before the first decision.

    Only counts are recorded. The record names and identifiers are member
    data and stay in live working context. ``confirmed`` says a person
    settled a reading the decider called ambiguous.
    """

    records: int
    outputs: int
    confirmed: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Remembered:
    """The run kept a fact in working memory. Its key and value stay out."""

    step: int
    from_control: bool


@dataclasses.dataclass(frozen=True, slots=True)
class RunEnded:
    """The reason a run stopped."""

    ending: str
    steps: int
    detail: str


type RunEvent = (
    RunStarted
    | Decided
    | Observed
    | Refused
    | Escalated
    | Acted
    | Flagged
    | Corrected
    | Verified
    | Interpreted
    | Remembered
    | Failed
    | RunEnded
    | Commanded
    | Paused
    | Transferred
    | ManualAction
    | HandedBack
    | Superseded
    | FailureEvidence
)


class Journal(Protocol):
    """Receives run events in the order they happen."""

    def record(self, event: RunEvent) -> None:
        """Append an event; raise EvidenceError when required storage fails."""
        ...


class MemoryJournal:
    """Collect events for tests or later processing by the caller."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def record(self, event: RunEvent) -> None:
        """Append one event."""
        self.events.append(event)
        emit(event)


class JsonlJournal(EvidenceWriter):
    """Write and flush one JSON object per event.

    Flushing per event matters because the interesting runs are the ones that
    stop unexpectedly; a buffered log of a hung run is not evidence.
    """

    def record(self, event: object) -> None:
        """Append one event as a single JSON line."""
        super().record(event)
        emit(event)


def as_record(event: RunEvent) -> dict[str, Any]:
    """Return ``event`` as a JSON-serializable mapping tagged with its type.

    Examples
    --------
    >>> as_record(RunStarted(profile_id="demo"))
    {'event': 'RunStarted', 'profile_id': 'profile_1'}
    """
    return safe_record(event)


def describe_value(action: Action) -> tuple[ValueSource, str | None]:
    """Report where an action's value came from, without reporting the value.

    Examples
    --------
    >>> from computeruse.actions import AxLocator
    >>> field = AxLocator("textbox", "Password")
    >>> describe_value(Action(ActionKind.TYPE, field, SecretRef("login_password")))
    (<ValueSource.SECRET: 'secret'>, 'login_password')
    >>> describe_value(Action(ActionKind.TYPE, field, "12345"))
    (<ValueSource.LITERAL: 'literal'>, None)
    """
    match action.value:
        case None:
            return ValueSource.NONE, None
        case SecretRef(name=name):
            return ValueSource.SECRET, name
        case _:
            return ValueSource.LITERAL, None


def describe_target(action: Action) -> str | None:
    """Report how a control was addressed, without reporting which control.

    Examples
    --------
    >>> from computeruse.actions import AxLocator
    >>> describe_target(Action(ActionKind.CLICK, AxLocator("button", "Search")))
    'AxLocator'
    >>> describe_target(Action(ActionKind.OBSERVE)) is None
    True
    """
    if action.target is None:
        return None
    return type(action.target).__name__
