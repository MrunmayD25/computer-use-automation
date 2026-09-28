"""Define and validate versioned workflow graphs for replay.

A capability is untrusted data saved after discovery. Parse it into frozen
dataclasses, reject unknown content, and validate its graph before replay.
The format permits typed references, enumerated conditions, and exact values,
with no executable expressions, scripts, or arbitrary selectors.

Invocation values remain references. Inputs can identify members, secrets
name profile entries, and runtime reads populate variables or outputs.
Targets use reusable descriptions instead of live coordinates or model prose.
The only stored images are permitted templates of controls.

Only the replay profile grants permission. ``check_profile`` rejects
capabilities requiring undeclared actions, routes, secrets, or observation
modes. Saved restrictions can add approval requirements but cannot grant access.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import types
import typing
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from computeruse import matching
from computeruse.actions import (
    DIALOG_ACTIONS,
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    Operation,
    Point,
    Relation,
    ScopeKind,
    ScreenTarget,
    SecretRef,
    separates,
)
from computeruse.policy import Source
from computeruse.profile import (
    ActionKind,
    Budgets,
    Limit,
    ObservationMode,
    Profile,
    is_effect_name,
)
from computeruse.urls import http_origin, parse_http_url

SCHEMA_VERSION = 10
"""The only schema version this module reads.

Version 10 records operator-declared operation bindings and restrictions.

Version 9 adds structured extraction with parameterized prefix and suffix
boundaries. Earlier artifacts require recording under the current semantics.

Version 8 gives ``Shows`` and ``ValueIs`` the purpose discovery checked them
for and whether the value must appear exactly once, so replay compares text
as discovery did. A row scope may name the column its name came from. An
action node the recorder added, which discovery never performed, names in
``introduced`` the declared binding that allowed it, such as a contract's
binding of an input to a prefilled form field.

Version 7 adds ``value_match`` to action nodes, so a choice whose label holds
an input among other words, such as "GOLD-CHK: Gold checking" for the input
"Gold checking", is saved as that input.

Version 5 binds final checks to records, identifies expected dialog messages,
and separates executable destinations from permission routes. Versions 1 to 4
are refused rather than migrated. Version 4 added the action grants a human
segment requires.

Version 2 changed three meanings. An action's ``route`` is the screen it
starts on, also for a navigation, whose destination is only in
``destination``. A human node names the operation it stands in for, so saved
restrictions apply to it. A ``contains`` comparison matches only at word
boundaries.

Version 3 adds an action's ``result_record`` and changes what completes a
record-bound action. Its result must show the record this invocation
intended, through ``result_record`` or through the action's own record
source, before it counts as done. A saved restriction on ``observe`` now
also governs the replay's own looks and captures.

An artifact of any other version is refused with ``unknown_schema`` rather
than read under rules it was not written for. It has to be recorded again,
or edited and reviewed again, to be replayed.
"""
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_IDENTIFIER = 64
MAX_TEXT = 200
"""Maximum characters in a saved constant, such as an accessible name."""
MAX_NODES = 200
MAX_TARGETS = 200
MAX_EDGES = 8
"""Maximum outgoing transitions per node."""
MAX_TRAVERSALS = 5
"""Maximum traversals of one bounded recovery edge per replay."""
MAX_TEMPLATE_BYTES = 256 * 1024
MAX_TEMPLATE_SIDE = 512
MAX_LIMIT = 1000

PASSIVE = frozenset(
    {
        ActionKind.OBSERVE,
        ActionKind.READ,
        ActionKind.WAIT_FOR,
        ActionKind.ASSERT,
        ActionKind.WAIT,
        ActionKind.SCROLL,
        ActionKind.MOVE,
        ActionKind.NAVIGATE,
    }
)
"""Actions that read, wait, or move the view, and change no record."""

MUTATING = frozenset(ActionKind) - PASSIVE
"""Actions that may change data and cannot be assumed safe to repeat.

Every action outside ``PASSIVE`` belongs to this set. New action types
therefore count as mutating until explicitly classified as passive.
"""

EFFECTFUL = MUTATING - {ActionKind.TYPE, ActionKind.SELECT}
"""Mutating actions whose result a capability must verify before moving on.

Later checks can verify typed or selected field values. Clicks, keys, and
dialog responses can submit data, so each needs evidence of its own result.
Use the node's ``verify`` checks or conditional outgoing transitions, such
as a search that returns a record or a not-found message. Every outgoing
transition must be conditional if transitions supply the evidence.
"""

VISUAL_ACTIONS = frozenset({ActionKind.CLICK, ActionKind.DOUBLE_CLICK})
"""Actions supported by a stored visual template."""


class CapabilityError(ValueError):
    """A capability failed parsing or validation.

    ``issues`` lists problem codes and indexed paths, such as
    ``nodes[3].transitions[0]``. Diagnostics never quote artifact contents.
    """

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        codes = ", ".join(sorted({issue.code.value for issue in self.issues}))
        super().__init__(f"capability is not valid: {codes}")


class IssueCode(StrEnum):
    """The allowed capability validation error codes."""

    MALFORMED = "malformed"
    TOO_LARGE = "too_large"
    UNKNOWN_SCHEMA = "unknown_schema"
    UNKNOWN_FIELD = "unknown_field"
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    UNKNOWN_VALUE = "unknown_value"
    INVALID_IDENTIFIER = "invalid_identifier"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"
    MISSING_ENTRY = "missing_entry"
    UNKNOWN_NODE = "unknown_node"
    UNKNOWN_TARGET = "unknown_target"
    UNKNOWN_REFERENCE = "unknown_reference"
    INVALID_REFERENCE = "invalid_reference"
    INVALID_TARGET = "invalid_target"
    INVALID_ACTION = "invalid_action"
    INVALID_TEMPLATE = "invalid_template"
    INVALID_ROUTE = "invalid_route"
    INVALID_FIELD = "invalid_field"
    INVALID_LIMIT = "invalid_limit"
    INVALID_PROVENANCE = "invalid_provenance"
    INVALID_TRANSITION = "invalid_transition"
    UNREACHABLE_NODE = "unreachable_node"
    DEAD_END = "dead_end"
    UNBOUNDED_CYCLE = "unbounded_cycle"
    REPEATING_CHANGE = "repeating_change"
    UNAVAILABLE_VARIABLE = "unavailable_variable"
    MISSING_OUTPUT = "missing_output"
    MISSING_COMPLETION_CHECK = "missing_completion_check"
    UNVERIFIED_ACTION = "unverified_action"
    UNDECLARED_OUTCOME = "undeclared_outcome"
    BYPASSES_INTERVENTION = "bypasses_intervention"
    BYPASSES_RESTRICTION = "bypasses_restriction"
    DENIED_OPERATION = "denied_operation"
    UNSUPPORTED_TARGET = "unsupported_target"
    PROFILE_MISMATCH = "profile_mismatch"
    NOT_GRANTED = "not_granted"
    ROUTE_NOT_PERMITTED = "route_not_permitted"
    UNDECLARED_SECRET = "undeclared_secret"  # noqa: S105  a code, not a value
    MODE_NOT_PERMITTED = "mode_not_permitted"
    RECORD_EVIDENCE_UNAVAILABLE = "record_evidence_unavailable"
    EFFECT_MISSING = "effect_missing"
    EFFECT_DENIED = "effect_denied"


@dataclasses.dataclass(frozen=True, slots=True)
class Issue:
    """A validation error code and location, without artifact contents."""

    code: IssueCode
    where: str


class ValueType(StrEnum):
    """The type of an input, output, or variable. Every value is carried as text.

    ``DIGITS`` is a string of decimal digits, such as a member number, whose
    leading zeros matter. ``INTEGER`` and ``DECIMAL`` are numbers written
    plainly, with an optional leading minus and, for a decimal, one point.
    ``CHOICE`` must be one of the field's declared ``choices``.
    """

    TEXT = "text"
    DIGITS = "digits"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    CHOICE = "choice"


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """A declared input, output, or working variable.

    Examples
    --------
    >>> member = Field("member_id", ValueType.DIGITS, True, 5, 10, ())
    >>> member.accepts("10001"), member.accepts("10a01"), member.accepts("12")
    (True, False, False)
    >>> Field("balance", ValueType.DECIMAL, True, 1, 20, ()).accepts("-1204.50")
    True
    """

    name: str
    type: ValueType
    required: bool
    min_length: int
    max_length: int
    choices: tuple[str, ...]

    def accepts(self, value: str) -> bool:
        """Report whether ``value`` has this field's type and length."""
        if not self.min_length <= len(value) <= self.max_length:
            return False
        return _typed(self.type, value, self.choices)


def _typed(kind: ValueType, value: str, choices: tuple[str, ...]) -> bool:
    digits = "0123456789"
    unsigned = value.removeprefix("-")
    match kind:
        case ValueType.TEXT:
            return value == value.strip() and all(c.isprintable() for c in value)
        case ValueType.DIGITS:
            return bool(value) and all(c in digits for c in value)
        case ValueType.INTEGER:
            return bool(unsigned) and all(c in digits for c in unsigned)
        case ValueType.DECIMAL:
            whole, point, fraction = unsigned.partition(".")
            numeric = whole + fraction
            return (
                bool(whole)
                and all(c in digits for c in numeric)
                and (not point or bool(fraction))
            )
        case ValueType.BOOLEAN:
            return value in {"true", "false"}
        case ValueType.CHOICE:
            return value in choices


class RefKind(StrEnum):
    """The source of a value resolved during replay."""

    INPUT = "input"
    VARIABLE = "variable"
    OUTPUT = "output"
    SECRET = "secret"  # noqa: S105  a source, not a value
    CONSTANT = "constant"


@dataclasses.dataclass(frozen=True, slots=True)
class Ref:
    """A typed reference to a value, resolved only at run time.

    A ``CONSTANT`` carries its text, such as a button's accessible name. Every
    other kind carries only a name: an input or output the capability
    declares, a variable it assigns, or a secret the profile declares.

    Examples
    --------
    >>> ref(RefKind.INPUT, "member_id")
    Ref(kind=<RefKind.INPUT: 'input'>, name='member_id', value='')
    """

    kind: RefKind
    name: str
    value: str


def constant(text: str) -> Ref:
    """Return a constant reference to ``text``."""
    return Ref(RefKind.CONSTANT, "", text)


def ref(kind: RefKind, name: str) -> Ref:
    """Return a named reference of ``kind``."""
    return Ref(kind, name, "")


@dataclasses.dataclass(frozen=True, slots=True)
class ScopeSpec:
    """The row, region, form, or table containing a target.

    ``column`` is zero unless a row name is bound to a specific column.
    Otherwise, it is that column's position, counted from one. A value in
    another column cannot substitute for the named cell.
    """

    kind: ScopeKind
    name: Ref
    column: int = 0


Match = matching.Match
"""How a ``Shows`` condition compares text; ``matching`` defines both forms.

``EQUALS`` compares the whole text. ``CONTAINS`` finds the value as a run of
whole words, so member ``12345`` is contained in ``Member 12345, J. Doe`` and
not in ``Member 1123456`` or ``Member 12345-01``.

A structural target or a record link that holds a reference may also match
by ``CONTAINS``. Its control then shows the reference's value once as whole
words among other text, such as ``Sam Lee (U-7)``, and only the reference is
saved. The surrounding text is read from the live page on each replay and is
never stored.
"""

Purpose = matching.Purpose
"""The evidence a saved check must establish, controlling comparison rules."""


class LocatorForm(StrEnum):
    """How a structural target names its control."""

    ACCESSIBILITY = "accessibility"
    DOM = "dom"


@dataclasses.dataclass(frozen=True, slots=True)
class StructuralTarget:
    """A control found by the same role, name, frame, and scope rules as discovery.

    A target must resolve to exactly one control. Use ``scope`` to
    distinguish repeated labels, never an occurrence index. ``name``,
    ``value``, and the scope name may be references, allowing each invocation
    to locate its record without storing that record's identifier.

    ``route`` is the allow-route template of the screen the control lives on.

    ``match`` controls references in the name or scope name. ``EQUALS``
    compares the whole text. ``CONTAINS`` requires the referenced value
    exactly once as whole words. A menu item such as ``Sam Lee (U-7)`` can
    therefore use an ``operator_id`` reference without saving the person's
    name. Constant text always requires an exact match.
    """

    TAG: ClassVar[str] = "structural"

    target_id: str
    route: str
    form: LocatorForm
    role: str
    name: Ref | None
    tag: str
    attribute: DomAttribute | None
    value: Ref | None
    frame: tuple[str, ...]
    scope: ScopeSpec | None
    match: Match = Match.EQUALS


class StoragePermit(StrEnum):
    """The permission to store a visual template.

    ``OPERATOR_APPROVED`` means the operator reviewed the crop and found no
    record data in it. ``SYNTHETIC`` marks a crop drawn for a test. There is
    no permit for a crop of a live screen that nobody reviewed.
    """

    OPERATOR_APPROVED = "operator_approved"
    SYNTHETIC = "synthetic"


@dataclasses.dataclass(frozen=True, slots=True)
class Template:
    """A permitted, non-sensitive picture of a painted control.

    Templates are the only images a capability stores. They contain no
    position. Replay locates the control by finding a unique match on the
    current screen.
    """

    template_id: str
    media_type: str
    data: str
    sha256: str
    width: int
    height: int
    permit: StoragePermit

    def image(self) -> bytes:
        """Return the template's PNG bytes."""
        return base64.b64decode(self.data, validate=True)


def template(template_id: str, png: bytes, permit: StoragePermit) -> Template:
    """Build a template from PNG bytes, recording its digest and size."""
    with Image.open(io.BytesIO(png)) as picture:
        width, height = picture.size
    return Template(
        template_id=template_id,
        media_type="image/png",
        data=base64.b64encode(png).decode("ascii"),
        sha256=hashlib.sha256(png).hexdigest(),
        width=width,
        height=height,
        permit=permit,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class VisualTarget:
    """A painted control found by matching a stored template in a live capture.

    ``frame`` names the frame whose capture is searched. The match must be
    unique by the margin ``visual.locate`` applies, and the capture must be
    taken immediately before the action.
    """

    TAG: ClassVar[str] = "visual"

    target_id: str
    route: str
    template: str
    frame: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class TextTarget:
    """A control or value located through text in a fresh screenshot.

    Local text recognition reads a capture permitted by the policy gate.
    ``text`` matches the whole line with ``EQUALS``, or an input or variable
    appearing once as whole words with ``CONTAINS``. If ``label`` is set,
    the line begins with ``text`` and the remaining text supplies the value.

    ``dx`` and ``dy`` offset the target from the line's top-left corner,
    allowing a field beside a label. When both are zero, target the line's
    center. Matching ignores spacing and case to allow recognition errors,
    but the line must remain unique.
    """

    TAG: ClassVar[str] = "text"

    target_id: str
    route: str
    frame: tuple[str, ...]
    text: Ref
    match: Match = Match.EQUALS
    label: bool = False
    dx: int = 0
    dy: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class RadioTarget(TextTarget):
    """A painted radio marker beside independently recognized label text.

    Only clicks and presence checks may use this target. It identifies a
    control, never its selection state or a persisted value.
    """

    TAG: ClassVar[str] = "radio"


@dataclasses.dataclass(frozen=True, slots=True)
class FocusTarget:
    """The keyboard focus identified by a fresh screenshot.

    A canvas form or terminal can move focus without a coordinate action.
    Replay takes a permitted capture and types into its focus. The adapter
    checks that focus has not moved.
    """

    TAG: ClassVar[str] = "focus"

    target_id: str
    route: str
    frame: tuple[str, ...]


FOCUS_ACTIONS = frozenset({ActionKind.TYPE, ActionKind.PRESS_KEY})
"""The actions that may go to the keyboard focus alone."""


TEXT_ACTIONS = frozenset({ActionKind.CLICK, ActionKind.DOUBLE_CLICK, ActionKind.READ})
"""Actions supported by a text target at one screenshot point."""
MAX_OFFSET = 400
"""Maximum distance in capture pixels between a field and its label."""


type Target = StructuralTarget | VisualTarget | TextTarget | RadioTarget | FocusTarget


def text_matches(
    shown: str,
    expected: str,
    match: Match,
    *,
    purpose: Purpose = Purpose.STATE,
    once: bool = False,
    painted: bool = False,
) -> bool:
    """Compare text the way a ``Shows`` condition does, as discovery did.

    Examples
    --------
    >>> text_matches("Member 12345, J. Doe", "12345", Match.CONTAINS)
    True
    >>> text_matches("Member 1123456", "12345", Match.CONTAINS)
    False
    >>> text_matches("Acct 12345-01", "12345", Match.CONTAINS)
    False
    >>> text_matches("12345", "12345", Match.EQUALS)
    True
    """
    return matching.shows(
        shown, expected, match, purpose=purpose, once=once, painted=painted
    )


def record_label(shown: str, value: str, record: RecordSpec) -> tuple[str, str] | None:
    """Return the prefix and suffix around ``value`` if ``shown`` names the record.

    An exact link matches its saved prefix, value, and suffix. A ``CONTAINS``
    link requires the value once as whole words and returns the current text
    on either side.

    Examples
    --------
    >>> link = RecordSpec("t1", ref(RefKind.INPUT, "member_id"), Relation.ROW,
    ...     match=Match.CONTAINS)
    >>> record_label("NM1 · Acme Ltd", "NM1", link)
    ('', ' · Acme Ltd')
    >>> record_label("NM1 and NM1", "NM1", link) is None
    True
    >>> record_label("NM12", "NM1", link) is None
    True
    >>> record_label("NM1-01 · Pat Lee", "NM1", link) is None
    True
    """
    if record.match is Match.EQUALS:
        if shown != record.prefix + value + record.suffix:
            return None
        return record.prefix, record.suffix
    found = matching.positions(shown, value)
    if len(found) != 1:
        return None
    start, end = found[0]
    return shown[:start], shown[end:]


@dataclasses.dataclass(frozen=True, slots=True)
class AtRoute:
    """The current screen matches the given allowed route template."""

    TAG: ClassVar[str] = "at_route"

    route: str


@dataclasses.dataclass(frozen=True, slots=True)
class Present:
    """The target resolves to exactly one visible control."""

    TAG: ClassVar[str] = "present"

    target: str


@dataclasses.dataclass(frozen=True, slots=True)
class Absent:
    """The target matches no control."""

    TAG: ClassVar[str] = "absent"

    target: str


@dataclasses.dataclass(frozen=True, slots=True)
class Shows:
    """The target resolves to one control whose text matches a value.

    A secret field never satisfies this: its value is not read.
    """

    TAG: ClassVar[str] = "shows"

    target: str
    value: Ref
    match: Match
    record: RecordSpec | None = None
    purpose: Purpose = Purpose.STATE
    once: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class DialogOpen:
    """A dialog of the given kind is waiting, such as ``confirm``."""

    TAG: ClassVar[str] = "dialog_open"

    kind: str
    message: Ref | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Bound:
    """A variable or output holds a value that satisfies its declared type."""

    TAG: ClassVar[str] = "bound"

    value: Ref


@dataclasses.dataclass(frozen=True, slots=True)
class ValueIs:
    """A previously read variable matches a declared expected value."""

    TAG: ClassVar[str] = "value_is"

    value: Ref
    expected: Ref
    match: Match
    purpose: Purpose = Purpose.STATE
    once: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Completed:
    """The named node's operation was verified complete earlier in this replay."""

    TAG: ClassVar[str] = "completed"

    node: str


type Condition = (
    AtRoute | Present | Absent | Shows | DialogOpen | Bound | ValueIs | Completed
)


class EdgeOrigin(StrEnum):
    """Where a transition came from.

    ``OBSERVED`` was taken during the discovery run the capability was
    recorded from. ``AUTHORED`` was added afterwards, such as a recovery
    branch or an alternative continuation, and needs its own review and tests.
    """

    OBSERVED = "observed"
    AUTHORED = "authored"


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """A transition taken only when it uniquely satisfies its conditions.

    Every condition must hold. An edge with no conditions always holds, so it
    must be the only edge leaving its node. ``limit`` bounds how many times a
    replay may follow the edge, and every cycle must pass through a bounded
    edge.
    """

    to: str
    when: tuple[Condition, ...]
    origin: EdgeOrigin
    limit: int | None


class Approval(StrEnum):
    """Whether an action needs a person's approval on every replay."""

    NONE = "none"
    EACH_RUN = "each_run"


@dataclasses.dataclass(frozen=True, slots=True)
class RecordSpec:
    """Evidence identifying an action's record on the live page.

    ``source`` is a structural target showing the record's identifier, and
    ``value`` is the input or variable it must equal. This becomes the
    action's ``RecordEvidence``, which the gate requires where the profile
    says so and the surface checks again before acting.

    With ``match`` set to ``CONTAINS`` the source shows the value once as
    whole words among other text. The replay reads that text from the live
    page and splits it around the value, so ``prefix`` and ``suffix`` stay
    empty and no record's name is saved.
    """

    source: str
    value: Ref
    relation: Relation
    prefix: str = ""
    suffix: str = ""
    match: Match = Match.EQUALS


@dataclasses.dataclass(frozen=True, slots=True)
class Param:
    """One ``:name`` placeholder of a destination route, and its value."""

    name: str
    value: Ref


@dataclasses.dataclass(frozen=True, slots=True)
class Destination:
    """A navigation destination expressed as a route template and values."""

    route: str
    params: tuple[Param, ...]
    query: tuple[Param, ...] = ()
    fragment: Ref | None = None
    origin: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Extraction:
    """The exact surrounding text of a value inside a structured reading.

    Empty tuples mark the start or end of the reading. Every nonempty
    boundary must occur once. References keep invocation data out of the
    artifact, and the remaining text requires the usual storage permission.
    """

    prefix: tuple[Ref, ...]
    suffix: tuple[Ref, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ActionNode:
    """Perform one operation, verify it, then take the one matching transition.

    ``route`` is the allowed route template where the operation starts.
    Replay acts only on that route. Navigation uses ``destination`` for the
    target address.

    ``record`` identifies the intended record before execution.
    ``result_record`` identifies it in the result, such as a receipt on
    another screen. For a record-bound operation, completion requires the
    exact intended record through ``result_record`` when declared, or through
    ``record`` otherwise. The chosen source must resolve where the result is
    checked. Missing record
    evidence leaves the operation unresolved and requires human help.
    Preconditions apply only before the operation, which may change them.

    ``requires`` must hold before the operation is attempted. ``verify`` must
    hold afterwards for the operation to count as completed; until then a
    delivered operation is uncertain and is never sent again automatically.
    A read stores what it read in ``into``, a variable or an output.

    ``mandatory`` says every path to a successful result must pass through
    this node, which validation checks for every transition.
    """

    TAG: ClassVar[str] = "action"

    node_id: str
    kind: ActionKind
    route: str
    target: str | None
    value: Ref | None
    destination: Destination | None
    effect: str | None
    record: RecordSpec | None
    result_record: RecordSpec | None
    into: Ref | None
    approval: Approval
    mandatory: bool
    requires: tuple[Condition, ...]
    verify: tuple[Condition, ...]
    transitions: tuple[Edge, ...]
    value_match: Match = Match.EQUALS
    introduced: str = ""
    """How a select value identifies its option.

    With ``CONTAINS``, an input or variable must appear once as whole words
    in exactly one option's label, as with a ``contains`` target.
    """
    extraction: Extraction | None = None
    binding: str = ""
    business: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class CheckNode:
    """Evaluate declared conditions and take the one matching transition."""

    TAG: ClassVar[str] = "check"

    node_id: str
    transitions: tuple[Edge, ...]
    delay_ms: int = 0


class HelpReason(StrEnum):
    """Why a capability hands a step to a person on every replay."""

    AUTHENTICATION = "authentication"
    RECORD_EVIDENCE = "record_evidence"
    MANUAL_STEP = "manual_step"
    UNREPRESENTABLE_TARGET = "unrepresentable_target"


@dataclasses.dataclass(frozen=True, slots=True)
class HumanNode:
    """A manual step whose result replay verifies before continuing.

    The transitions are the continuation checks. After the person returns
    control, exactly one of them must hold, or the replay stays paused.

    ``performs`` and ``effect`` name the operation a person carries out in
    place of the automation, when the step stands in for one the recorder
    could not represent. Saved restrictions and the profile's effect
    exceptions apply to that operation as they would to an action node, so a
    denied operation cannot become a person's task. They are None for a step
    that is a person's by nature, such as entering a one-time code.
    """

    TAG: ClassVar[str] = "human"

    node_id: str
    route: str
    reason: HelpReason
    performs: ActionKind | None
    effect: str | None
    mandatory: bool
    transitions: tuple[Edge, ...]
    permissions: tuple[ActionKind, ...] = ()
    """Action types used during the manual segment, without granting permission."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultConfirmationNode(HumanNode):
    """A person confirms required result values the saved checks cannot show."""

    TAG: ClassVar[str] = "result_confirmation"

    confirm_inputs: tuple[Ref, ...]
    """Required result values a person confirms from this invocation's inputs."""


class ResultKind(StrEnum):
    """The possible endings reported by a result node."""

    SUCCESS = "success"
    OUTCOME = "outcome"
    FAILURE = "failure"


@dataclasses.dataclass(frozen=True, slots=True)
class ResultNode:
    """End the replay, after reading ``checks`` from the live page.

    A ``SUCCESS`` needs completion checks and every required output. An
    ``OUTCOME`` names a declared business outcome, such as a member not found,
    and needs checks that show it. A ``FAILURE`` ends the replay as failed.
    """

    TAG: ClassVar[str] = "result"

    node_id: str
    result: ResultKind
    outcome: str
    checks: tuple[Condition, ...]


type Node = ActionNode | CheckNode | HumanNode | ResultConfirmationNode | ResultNode


class SurfaceKind(StrEnum):
    """The kind of profile scope a capability was recorded under."""

    BROWSER = "browser"


@dataclasses.dataclass(frozen=True, slots=True)
class Application:
    """The application identity and entry checks for a capability.

    ``markers`` must hold on the entry screen before the first node runs, so
    a capability recorded against one application does not act on another
    that happens to share a route.
    """

    profile_id: str
    surface: SurfaceKind
    origin: str
    entry_route: str
    markers: tuple[Condition, ...]


class ProvenanceKind(StrEnum):
    """Where a capability came from.

    ``DISCOVERED`` is recorded from a real discovery run. ``SYNTHETIC`` is
    built from test events and must never be presented as discovered.
    ``AUTHORED`` is written by hand.
    """

    DISCOVERED = "discovered"
    SYNTHETIC = "synthetic"
    AUTHORED = "authored"


class Review(StrEnum):
    """Whether a person has reviewed this version. A recorder only writes drafts."""

    DRAFT = "draft"
    REVIEWED = "reviewed"


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """The version's origin, with an opaque discovery run ID in ``run``."""

    kind: ProvenanceKind
    run: str
    recorder: str
    review: Review


class RestrictionScope(StrEnum):
    """What a saved restriction covers.

    ``TARGET`` covers actions on one reusable target. ``ROUTE`` covers every
    action of its kinds on a route, and is what a restriction becomes when
    discovery could not tie it to one control a replay can find again.
    """

    TARGET = "target"
    ROUTE = "route"
    OPERATION = "operation"


@dataclasses.dataclass(frozen=True, slots=True)
class SavedRestriction:
    """A restriction learned during discovery, kept for every replay.

    ``kinds`` empty means every action type. ``key`` narrows a press_key
    restriction to one key, and empty means any key. A saved restriction is
    never lowered by an approval, a manual step, or a later replay.
    """

    scope: RestrictionScope
    route: str
    kinds: tuple[ActionKind, ...]
    target: str | None
    key: str
    limit: Limit
    source: Source


@dataclasses.dataclass(frozen=True, slots=True)
class Limits:
    """Replay limits, each capped by the profile's budgets.

    ``max_target_attempts`` bounds how often one node retries a target that
    was missing or stale. ``max_settle_observations`` bounds how often the
    replay looks again for a transition, a verification, or a target that
    has not appeared yet, ``settle_interval_ms`` apart.
    ``max_help_requests`` bounds how often one replay asks a person.
    """

    max_steps: int
    max_wall_clock_s: int
    max_target_attempts: int
    max_settle_observations: int
    settle_interval_ms: int
    max_help_requests: int


@dataclasses.dataclass(frozen=True, slots=True)
class Capability:
    """One versioned, reusable workflow."""

    schema_version: int
    capability_id: str
    version: int
    application: Application
    provenance: Provenance
    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]
    variables: tuple[Field, ...]
    secrets: tuple[str, ...]
    outcomes: tuple[str, ...]
    templates: tuple[Template, ...]
    targets: tuple[Target, ...]
    entry: str
    nodes: tuple[Node, ...]
    restrictions: tuple[SavedRestriction, ...]
    limits: Limits

    def node(self, node_id: str) -> Node:
        """Return the node with ``node_id``."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def target(self, target_id: str) -> Target:
        """Return the target with ``target_id``."""
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise KeyError(target_id)

    def field(self, value: Ref) -> Field | None:
        """Return the declaration a reference names, or None."""
        pool = {
            RefKind.INPUT: self.inputs,
            RefKind.OUTPUT: self.outputs,
            RefKind.VARIABLE: self.variables,
        }.get(value.kind, ())
        return next((item for item in pool if item.name == value.name), None)

    @property
    def label(self) -> str:
        """Return ``id@version``, which names this capability to an operator."""
        return f"{self.capability_id}@{self.version}"


# Serialization. The codec walks the dataclass definitions above, so the JSON
# form and the Python form cannot drift apart. Every key must be present,
# every unknown key is rejected, and a union member is chosen by its tag.


def to_document(capability: Capability) -> dict[str, Any]:
    """Return ``capability`` as a JSON-compatible mapping."""
    encoded = _encode(capability)
    assert isinstance(encoded, dict)  # noqa: S101  a dataclass encodes to a mapping
    return encoded


def dumps(capability: Capability) -> str:
    """Return ``capability`` as canonical JSON text."""
    return json.dumps(to_document(capability), sort_keys=True, indent=2) + "\n"


def from_document(document: object) -> Capability:
    """Parse and validate a capability document.

    Raises
    ------
    CapabilityError
        If the document is malformed or its graph is invalid.
    """
    if not isinstance(document, dict):
        raise CapabilityError([Issue(IssueCode.MALFORMED, "")])
    version = document.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        raise CapabilityError([Issue(IssueCode.UNKNOWN_SCHEMA, "schema_version")])
    issues: list[Issue] = []
    capability = _decode(Capability, document, "", issues)
    if issues or not isinstance(capability, Capability):
        raise CapabilityError(issues or [Issue(IssueCode.MALFORMED, "")])
    found = validate(capability)
    if found:
        raise CapabilityError(found)
    return capability


def loads(text: str) -> Capability:
    """Parse, then validate, capability JSON text."""
    if len(text.encode("utf-8")) > MAX_ARTIFACT_BYTES:
        raise CapabilityError([Issue(IssueCode.TOO_LARGE, "")])
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_keys,
            parse_constant=_no_constant,
        )
    except (ValueError, RecursionError):
        raise CapabilityError([Issue(IssueCode.MALFORMED, "")]) from None
    return from_document(document)


def load_capability(path: Path) -> Capability:
    """Read, parse, and validate a capability file."""
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_ARTIFACT_BYTES + 1)
        text = data.decode("utf-8")
    except (OSError, UnicodeError):
        raise CapabilityError([Issue(IssueCode.MALFORMED, "")]) from None
    if len(data) > MAX_ARTIFACT_BYTES:
        raise CapabilityError([Issue(IssueCode.TOO_LARGE, "")])
    return loads(text)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError("duplicate key")
        mapping[key] = value
    return mapping


def _no_constant(_: str) -> None:
    raise ValueError("non-finite number")


def _encode(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, object] = {}
        tag = getattr(type(value), "TAG", None)
        if tag is not None:
            encoded["type"] = tag
        for item in dataclasses.fields(value):
            encoded[item.name] = _encode(getattr(value, item.name))
        return encoded
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    if cls not in _HINTS:
        _HINTS[cls] = typing.get_type_hints(cls)
    return _HINTS[cls]


def _decode(kind: object, data: object, where: str, issues: list[Issue]) -> object:
    if isinstance(kind, typing.TypeAliasType):
        kind = kind.__value__
    origin = typing.get_origin(kind)
    if origin in {types.UnionType, typing.Union}:
        return _decode_union(typing.get_args(kind), data, where, issues)
    if origin is tuple:
        return _decode_tuple(typing.get_args(kind)[0], data, where, issues)
    if isinstance(kind, type) and dataclasses.is_dataclass(kind):
        return _decode_record(kind, data, where, issues)
    if isinstance(kind, type) and issubclass(kind, StrEnum):
        try:
            return kind(data)
        except ValueError:
            issues.append(Issue(IssueCode.UNKNOWN_VALUE, where))
            return None
    if kind is int and isinstance(data, int) and not isinstance(data, bool):
        return data
    if (kind is str and isinstance(data, str)) or (
        kind is bool and isinstance(data, bool)
    ):
        return data
    issues.append(Issue(IssueCode.WRONG_TYPE, where))
    return None


def _decode_union(
    members: tuple[object, ...], data: object, where: str, issues: list[Issue]
) -> object:
    if data is None and type(None) in members:
        return None
    options = [member for member in members if member is not type(None)]
    if len(options) == 1:
        return _decode(options[0], data, where, issues)
    tag = data.get("type") if isinstance(data, dict) else None
    for member in options:
        if getattr(member, "TAG", None) == tag:
            return _decode(member, data, where, issues)
    issues.append(Issue(IssueCode.UNKNOWN_VALUE, f"{where}.type"))
    return None


def _decode_tuple(
    item: object, data: object, where: str, issues: list[Issue]
) -> tuple[object, ...] | None:
    if not isinstance(data, list):
        issues.append(Issue(IssueCode.WRONG_TYPE, where))
        return None
    return tuple(
        _decode(item, element, f"{where}[{index}]", issues)
        for index, element in enumerate(data)
    )


def _decode_record(cls: type, data: object, where: str, issues: list[Issue]) -> object:
    if not isinstance(data, dict):
        issues.append(Issue(IssueCode.WRONG_TYPE, where))
        return None
    hints = _hints(cls)
    names = [item.name for item in dataclasses.fields(typing.cast("Any", cls))]
    tag = getattr(cls, "TAG", None)
    known = set(names) | ({"type"} if tag is not None else set())
    prefix = f"{where}." if where else ""
    for key in data:
        if key not in known:
            issues.append(Issue(IssueCode.UNKNOWN_FIELD, where))
            break
    if tag is not None and data.get("type") != tag:
        issues.append(Issue(IssueCode.UNKNOWN_VALUE, f"{prefix}type"))
        return None
    values: dict[str, object] = {}
    for name in names:
        if name not in data:
            issues.append(Issue(IssueCode.MISSING_FIELD, f"{prefix}{name}"))
            continue
        values[name] = _decode(hints[name], data[name], f"{prefix}{name}", issues)
    if len(values) != len(names):
        return None
    return cls(**values)


# Static validation. Everything here reads the artifact alone. What depends on
# the profile is in ``check_profile``, and what depends on the live page can
# only be checked during a replay.


def validate(capability: Capability) -> tuple[Issue, ...]:
    """Return validation problems detectable without a profile or live page."""
    graph = _Graph(capability)
    for check in (
        graph.check_header,
        graph.check_fields,
        graph.check_templates,
        graph.check_targets,
        graph.check_nodes,
        graph.check_structure,
        graph.check_availability,
        graph.check_obligations,
        graph.check_restrictions,
    ):
        check()
        if graph.fatal:
            break
    return tuple(graph.issues)


def identifier(text: str) -> bool:
    """Report whether ``text`` is a valid capability identifier.

    Examples
    --------
    >>> identifier("read_balance"), identifier("Read balance"), identifier("9x")
    (True, False, False)
    """
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    return (
        0 < len(text) <= MAX_IDENTIFIER
        and text[0].isalpha()
        and all(character in allowed for character in text)
    )


def route_template(route: str) -> bool:
    """Report whether ``route`` is a canonical browser route template.

    Examples
    --------
    >>> route_template("/members/:id"), route_template("/a/../b")
    (True, False)
    """
    if route.startswith(("http://", "https://")):
        try:
            parsed = parse_http_url(route)
            if parsed.query or parsed.fragment:
                return False
            route = parsed.path
        except ValueError:
            return False
    if not route.startswith("/") or "//" in route or len(route) > MAX_TEXT:
        return False
    if any(character in route for character in "%?#\\") or any(
        character.isspace() or not character.isprintable() for character in route
    ):
        return False
    segments = [segment for segment in route.split("/") if segment]
    return not any(segment in {".", ".."} for segment in segments)


def placeholders(route: str) -> tuple[str, ...]:
    """Return the ``:name`` placeholders of a route template, in order."""
    return tuple(segment[1:] for segment in route.split("/") if segment.startswith(":"))


def conditions_of(node: Node) -> Iterable[tuple[str, Condition]]:
    """Yield a node's conditions with their locations within the node."""
    match node:
        case ActionNode():
            for index, condition in enumerate(node.requires):
                yield f"requires[{index}]", condition
            for index, condition in enumerate(node.verify):
                yield f"verify[{index}]", condition
        case ResultNode():
            for index, condition in enumerate(node.checks):
                yield f"checks[{index}]", condition
        case _:
            pass
    for index, edge in enumerate(transitions_of(node)):
        for number, condition in enumerate(edge.when):
            yield f"transitions[{index}].when[{number}]", condition


def transitions_of(node: Node) -> tuple[Edge, ...]:
    """Return a node's transitions, or an empty tuple for a result node."""
    return () if isinstance(node, ResultNode) else node.transitions


def restriction_applies(saved: SavedRestriction, node: ActionNode) -> bool:
    """Report whether a saved restriction covers an action node.

    Examples
    --------
    >>> saved = SavedRestriction(RestrictionScope.ROUTE, "/pay", (), None, "",
    ...                          Limit.RISKY, Source.FINDING)
    >>> node = ActionNode("n1", ActionKind.CLICK, "/pay", "t1", None, None, None,
    ...                   None, None, None, Approval.EACH_RUN, True, (), (), ())
    >>> restriction_applies(saved, node)
    True
    """
    if saved.scope is RestrictionScope.OPERATION and node.business:
        return saved.target == node.business
    if saved.route != node.route:
        return False
    if saved.kinds and node.kind not in saved.kinds:
        return False
    if saved.scope is RestrictionScope.TARGET and saved.target != node.target:
        return False
    if not saved.key or node.kind is not ActionKind.PRESS_KEY:
        return True
    return (
        node.value is None
        or node.value.kind is not RefKind.CONSTANT
        or (node.value.value == saved.key)
    )


def strictest(
    restrictions: Iterable[SavedRestriction],
    node: ActionNode | HumanNode,
    *,
    pixel: bool = False,
) -> Limit | None:
    """Return the strictest saved restriction covering ``node``, if any.

    A human node is covered by a restriction on the operation it stands in
    for, and by any restriction on every action type on its route. A person
    doing the step does not lower either. ``pixel`` says the node aims by
    pixels or painted words, whose operation no id identifies; as in
    discovery, every restriction of its action type on its route covers it.
    """
    found: Limit | None = None
    for saved in restrictions:
        if isinstance(node, HumanNode):
            if not _covers_human(saved, node):
                continue
        elif not (
            restriction_applies(saved, node)
            or (
                pixel
                and saved.route == node.route
                and (not saved.kinds or node.kind in saved.kinds)
                and not (saved.scope is RestrictionScope.OPERATION and node.business)
            )
        ):
            continue
        if saved.limit is Limit.DENY:
            return Limit.DENY
        found = Limit.RISKY
    return found


def aims_by_pixels(capability: Capability, node: ActionNode | HumanNode) -> bool:
    """Return whether ``node`` targets screenshot coordinates or recognized text."""
    if not isinstance(node, ActionNode) or node.target is None:
        return False
    try:
        target = capability.target(node.target)
    except KeyError:
        return False
    return isinstance(target, (VisualTarget, TextTarget, FocusTarget))


def preparation_review(capability: Capability, node: ActionNode) -> str | None:
    """Find the mandatory input approval that must verify a painted choice.

    Acknowledging a safe choice need not paint readable evidence. Its selection
    can instead be checked by the person approving the later write, only along
    one unconditional preparation path. This does not prove the choice or
    remove any saved check. Replay must name the obligation in that approval.
    """
    targets = {target.target_id: target for target in capability.targets}
    target = targets.get(node.target or "")
    if (
        node.kind is not ActionKind.CLICK
        or not isinstance(target, TextTarget)
        or target.text.kind is not RefKind.INPUT
        or target.label
        or target.dx
        or target.dy
        or node.verify
        or Present(target.target_id) not in node.requires
    ):
        return None
    nodes = {item.node_id: item for item in capability.nodes}
    seen: set[str] = set()
    current = node
    while current.node_id not in seen:
        seen.add(current.node_id)
        if (
            not current.binding
            or not current.business
            or current.business != node.business
            or current.route != node.route
            or current.approval is not Approval.NONE
            or current.mandatory
            or strictest(capability.restrictions, current, pixel=True) is not None
            or current.target not in targets
            or targets[current.target].frame != target.frame
            or len(current.transitions) != 1
            or current.transitions[0].when
        ):
            return None
        after = nodes.get(current.transitions[0].to)
        if not isinstance(after, ActionNode) or after.route != node.route:
            return None
        aimed = targets.get(after.target or "")
        if aimed is None or aimed.frame != target.frame:
            return None
        if after.approval is Approval.EACH_RUN:
            return (
                after.node_id
                if after.kind in {ActionKind.CLICK, ActionKind.PRESS_KEY}
                and after.mandatory
                and after.binding
                and after.business
                and after.verify
                and Present(aimed.target_id) in after.requires
                else None
            )
        current = after
    return None


def _covers_human(saved: SavedRestriction, node: HumanNode) -> bool:
    if saved.route != node.route:
        return False
    if not saved.kinds and saved.scope is RestrictionScope.ROUTE:
        return True
    # A person's step names no reusable target, so a restriction on one
    # target could be about the control they use. It applies whenever the
    # action type matches.
    kinds = set(node.permissions)
    if node.performs is not None:
        kinds.add(node.performs)
    return bool(kinds) and (not saved.kinds or bool(kinds.intersection(saved.kinds)))


class _Graph:
    """The validator's working state for one capability."""

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        self.issues: list[Issue] = []
        self.fatal = False
        self.ids: dict[str, set[str]] = {}
        self.targets: dict[str, Target] = {}
        self.templates: dict[str, Template] = {}
        self.names: dict[RefKind, dict[str, Field]] = {}

    def add(self, code: IssueCode, where: str) -> None:
        self.issues.append(Issue(code, where))

    def check_header(self) -> None:
        capability = self.capability
        if capability.schema_version != SCHEMA_VERSION:
            self.add(IssueCode.UNKNOWN_SCHEMA, "schema_version")
        if not identifier(capability.capability_id):
            self.add(IssueCode.INVALID_IDENTIFIER, "capability_id")
        if not 1 <= capability.version <= 1_000_000:
            self.add(IssueCode.INVALID_FIELD, "version")
        self._application()
        self._provenance()
        self._limits()
        if len(capability.nodes) > MAX_NODES or len(capability.targets) > MAX_TARGETS:
            self.add(IssueCode.TOO_LARGE, "nodes")
            self.fatal = True

    def _application(self) -> None:
        application = self.capability.application
        if not identifier_like(application.profile_id):
            self.add(IssueCode.INVALID_FIELD, "application.profile_id")
        if not _origin(application.origin):
            self.add(IssueCode.INVALID_FIELD, "application.origin")
        if not route_template(application.entry_route):
            self.add(IssueCode.INVALID_ROUTE, "application.entry_route")

    def _provenance(self) -> None:
        provenance = self.capability.provenance
        needs_run = provenance.kind is ProvenanceKind.DISCOVERED
        if needs_run != bool(provenance.run):
            self.add(IssueCode.INVALID_PROVENANCE, "provenance.run")
        if provenance.run and not identifier_like(provenance.run):
            self.add(IssueCode.INVALID_PROVENANCE, "provenance.run")
        if not identifier_like(provenance.recorder):
            self.add(IssueCode.INVALID_PROVENANCE, "provenance.recorder")

    def _limits(self) -> None:
        ceilings = {"max_wall_clock_s": 86_400, "settle_interval_ms": 60_000}
        for item in dataclasses.fields(Limits):
            value = getattr(self.capability.limits, item.name)
            if not 1 <= value <= ceilings.get(item.name, MAX_LIMIT):
                self.add(IssueCode.INVALID_LIMIT, f"limits.{item.name}")

    def _claim(self, name: str, where: str, space: str) -> None:
        """Register ``name`` in one identifier group, such as nodes."""
        taken = self.ids.setdefault(space, set())
        if not identifier(name):
            self.add(IssueCode.INVALID_IDENTIFIER, where)
        elif name in taken:
            self.add(IssueCode.DUPLICATE_IDENTIFIER, where)
        else:
            taken.add(name)

    def check_fields(self) -> None:
        capability = self.capability
        pools = (
            (RefKind.INPUT, "inputs", capability.inputs),
            (RefKind.OUTPUT, "outputs", capability.outputs),
            (RefKind.VARIABLE, "variables", capability.variables),
        )
        for kind, section, pool in pools:
            self.names[kind] = {}
            for index, declared in enumerate(pool):
                where = f"{section}[{index}]"
                self._claim(declared.name, where, "fields")
                self.names[kind][declared.name] = declared
                if not _field_ok(declared):
                    self.add(IssueCode.INVALID_FIELD, where)
        for index, name in enumerate(capability.secrets):
            self._claim(name, f"secrets[{index}]", "secrets")
        for index, name in enumerate(capability.outcomes):
            self._claim(name, f"outcomes[{index}]", "outcomes")

    def check_templates(self) -> None:
        for index, item in enumerate(self.capability.templates):
            where = f"templates[{index}]"
            self._claim(item.template_id, where, "templates")
            self.templates[item.template_id] = item
            if not _template_ok(item):
                self.add(IssueCode.INVALID_TEMPLATE, where)

    def check_targets(self) -> None:
        for index, target in enumerate(self.capability.targets):
            where = f"targets[{index}]"
            self._claim(target.target_id, where, "targets")
            self.targets[target.target_id] = target
            if not route_template(target.route):
                self.add(IssueCode.INVALID_ROUTE, f"{where}.route")
            if any(not _frame_name(name) for name in target.frame):
                self.add(IssueCode.INVALID_TARGET, f"{where}.frame")
            if isinstance(target, VisualTarget):
                if target.template not in self.templates:
                    self.add(IssueCode.UNKNOWN_TARGET, f"{where}.template")
                continue
            if isinstance(target, TextTarget):
                self._text_target(target, where)
                continue
            if isinstance(target, FocusTarget):
                continue
            self._structural(target, where)

    def _text_target(self, target: TextTarget, where: str) -> None:
        self._text_ref(target.text, f"{where}.text")
        if isinstance(target, RadioTarget) and (target.label or target.dx or target.dy):
            self.add(IssueCode.INVALID_TARGET, where)
        words = target.text.kind in {RefKind.INPUT, RefKind.VARIABLE, RefKind.OUTPUT}
        if (target.match is Match.CONTAINS) != words or (target.label and words):
            # A label is the website's own words; a reference is found once
            # as whole words inside a line, never as the whole of one.
            self.add(IssueCode.INVALID_TARGET, f"{where}.match")
        if max(abs(target.dx), abs(target.dy)) > MAX_OFFSET or (
            target.label and (target.dx or target.dy)
        ):
            self.add(IssueCode.INVALID_TARGET, f"{where}.offset")

    def _structural(self, target: StructuralTarget, where: str) -> None:
        texts = [target.name, target.value]
        if target.scope is not None:
            texts.append(target.scope.name)
        for text in texts:
            if text is not None:
                self._text_ref(text, where)
        accessible = target.form is LocatorForm.ACCESSIBILITY
        shaped = (
            bool(target.role) and not target.tag and target.attribute is None
            if accessible
            else not target.role
            and target.attribute is not None
            and target.value is not None
            and target.name is None
        )
        if not shaped or len(target.role) > MAX_IDENTIFIER:
            self.add(IssueCode.INVALID_TARGET, where)
            return
        if target.match is Match.CONTAINS and not _contains_shaped(target):
            self.add(IssueCode.INVALID_TARGET, f"{where}.match")
        if target.scope is not None and (
            target.scope.column < 0
            or (target.scope.column and target.scope.kind is not ScopeKind.ROW)
            or (target.scope.column and target.scope.name.kind is RefKind.CONSTANT)
        ):
            # Only a row found by a reference's value is bound to a column.
            self.add(IssueCode.INVALID_TARGET, f"{where}.scope.column")
        try:
            _probe_locator(target)
        except ValueError:
            self.add(IssueCode.INVALID_TARGET, where)

    def _text_ref(self, value: Ref, where: str) -> None:
        """Check a reference a locator or a comparison may use as text."""
        if value.kind is RefKind.SECRET:
            self.add(IssueCode.INVALID_REFERENCE, where)
            return
        self.reference(value, where)

    def reference(self, value: Ref, where: str) -> None:
        if value.kind is RefKind.CONSTANT:
            text = value.value
            if value.name or not text or len(text) > MAX_TEXT or not text.isprintable():
                self.add(IssueCode.INVALID_REFERENCE, where)
            return
        if value.value or not value.name:
            self.add(IssueCode.INVALID_REFERENCE, where)
            return
        if value.kind is RefKind.SECRET:
            if value.name not in self.capability.secrets:
                self.add(IssueCode.UNKNOWN_REFERENCE, where)
            return
        if value.name not in self.names.get(value.kind, {}):
            self.add(IssueCode.UNKNOWN_REFERENCE, where)

    def check_nodes(self) -> None:
        capability = self.capability
        for index, node in enumerate(capability.nodes):
            where = f"nodes[{index}]"
            self._claim(node.node_id, where, "nodes")
        for index, node in enumerate(capability.nodes):
            where = f"nodes[{index}]"
            self._node(node, where)
            for part, condition in conditions_of(node):
                self.condition(condition, f"{where}.{part}")
            self._edges(node, where)
        for index, condition in enumerate(capability.application.markers):
            self.condition(condition, f"application.markers[{index}]")
        if capability.entry not in {node.node_id for node in capability.nodes}:
            self.add(IssueCode.MISSING_ENTRY, "entry")
            self.fatal = True

    def _node(self, node: Node, where: str) -> None:
        match node:
            case ActionNode():
                self._action(node, where)
            case HumanNode():
                if not route_template(node.route):
                    self.add(IssueCode.INVALID_ROUTE, f"{where}.route")
                if node.effect is not None and (
                    node.performs is None or not is_effect_name(node.effect)
                ):
                    self.add(IssueCode.INVALID_ACTION, f"{where}.effect")
                if not node.transitions or any(not e.when for e in node.transitions):
                    self.add(IssueCode.INVALID_TRANSITION, f"{where}.transitions")
                if isinstance(node, ResultConfirmationNode):
                    self._confirmation(node, where)
            case CheckNode():
                if not 0 <= node.delay_ms <= 10_000:
                    self.add(IssueCode.INVALID_FIELD, f"{where}.delay_ms")
                if not node.transitions or any(not e.when for e in node.transitions):
                    self.add(IssueCode.INVALID_TRANSITION, f"{where}.transitions")
            case ResultNode():
                self._result(node, where)

    def _confirmation(self, node: ResultConfirmationNode, where: str) -> None:
        results = {
            item.node_id
            for item in self.capability.nodes
            if isinstance(item, ResultNode) and item.result is ResultKind.SUCCESS
        }
        if any(edge.to not in results for edge in node.transitions):
            self.add(IssueCode.INVALID_TRANSITION, f"{where}.transitions")
        if (
            not node.confirm_inputs
            or node.reason is not HelpReason.RECORD_EVIDENCE
            or not node.mandatory
            or node.performs is not None
            or node.effect is not None
            or node.permissions
            or len(set(node.confirm_inputs)) != len(node.confirm_inputs)
        ):
            self.add(IssueCode.INVALID_FIELD, f"{where}.confirm_inputs")
        for value in node.confirm_inputs:
            field = self.capability.field(value)
            if value.kind is not RefKind.INPUT or field is None or not field.required:
                self.add(IssueCode.INVALID_REFERENCE, f"{where}.confirm_inputs")
            self.reference(value, f"{where}.confirm_inputs")

    def _result(self, node: ResultNode, where: str) -> None:
        if node.result is ResultKind.OUTCOME:
            if node.outcome not in self.capability.outcomes:
                self.add(IssueCode.UNDECLARED_OUTCOME, f"{where}.outcome")
        elif node.outcome:
            self.add(IssueCode.INVALID_FIELD, f"{where}.outcome")
        if node.result is not ResultKind.FAILURE and not node.checks:
            self.add(IssueCode.MISSING_COMPLETION_CHECK, f"{where}.checks")

    def _value_match(self, node: ActionNode, where: str) -> None:
        """Allow ``contains`` only for a select that names an input or variable."""
        if node.value_match is Match.CONTAINS and (
            node.kind is not ActionKind.SELECT
            or node.value is None
            or node.value.kind not in {RefKind.INPUT, RefKind.VARIABLE}
        ):
            self.add(IssueCode.INVALID_ACTION, f"{where}.value_match")

    def _binding(self, node: ActionNode, where: str) -> None:
        if bool(node.binding) != bool(node.business) or (
            node.binding
            and (not is_effect_name(node.binding) or not identifier_like(node.business))
        ):
            self.add(IssueCode.INVALID_ACTION, f"{where}.binding")

    def _action(self, node: ActionNode, where: str) -> None:
        self._binding(node, where)
        if not route_template(node.route):
            self.add(IssueCode.INVALID_ROUTE, f"{where}.route")
        target = self.targets.get(node.target) if node.target is not None else None
        if node.target is not None and target is None:
            self.add(IssueCode.UNKNOWN_TARGET, f"{where}.target")
            return
        if target is not None and target.route != node.route:
            self.add(IssueCode.INVALID_TARGET, f"{where}.target")
        if node.value is not None:
            self.reference(node.value, f"{where}.value")
            if node.value.kind is RefKind.SECRET and node.kind is not ActionKind.TYPE:
                self.add(IssueCode.INVALID_REFERENCE, f"{where}.value")
        if node.effect is not None and not is_effect_name(node.effect):
            self.add(IssueCode.INVALID_ACTION, f"{where}.effect")
        if node.kind in DIALOG_ACTIONS and not any(
            isinstance(check, DialogOpen) and check.message is not None
            for check in node.requires
        ):
            self.add(IssueCode.UNVERIFIED_ACTION, f"{where}.requires")
        self._destination(node, where)
        self._into(node, where)
        self._extraction(node, target, where)
        self._record(node, target, where)
        self._result_record(node, where)
        if not _fits(node, target):
            self.add(IssueCode.UNSUPPORTED_TARGET, f"{where}.target")
        self._value_match(node, where)
        if (
            node.kind in EFFECTFUL
            and not node.verify
            and not branches(node)
            and not self._focuses(node, target)
            and preparation_review(self.capability, node) is None
        ):
            self.add(IssueCode.UNVERIFIED_ACTION, f"{where}.verify")
        try:
            _probe_action(node, target)
        except ValueError:
            self.add(IssueCode.INVALID_ACTION, where)
        if not node.transitions:
            self.add(IssueCode.INVALID_TRANSITION, f"{where}.transitions")

    def _focuses(self, node: ActionNode, target: Target | None) -> bool:
        """Report whether ``node`` is a painted click the next typing proves.

        A click on a painted field moves the keyboard focus and paints
        nothing new, so no check can follow it alone. It is kept, because it
        chooses the field the next keys go to (rule 4). Its only step after
        is typing or a key sent to the focus, and whatever checks that typing,
        as the next step's conditions check any typing, checks the click too.
        """
        if (
            node.kind is not ActionKind.CLICK
            or not isinstance(target, TextTarget)
            or isinstance(target, RadioTarget)
            or target.text.kind is not RefKind.CONSTANT
            or node.approval is not Approval.NONE
            or node.mandatory
            or strictest(self.capability.restrictions, node, pixel=True) is not None
        ):
            return False
        if len(node.transitions) != 1 or node.transitions[0].when:
            return False
        after = next(
            (
                item
                for item in self.capability.nodes
                if item.node_id == node.transitions[0].to
            ),
            None,
        )
        return (
            isinstance(after, ActionNode)
            and after.kind in FOCUS_ACTIONS
            and after.target is not None
            and isinstance(self.targets.get(after.target), FocusTarget)
        )

    def _destination(self, node: ActionNode, where: str) -> None:
        destination = node.destination
        if destination is None:
            return
        if destination.origin and not _origin(destination.origin):
            self.add(IssueCode.INVALID_ROUTE, f"{where}.destination.origin")
        navigating = node.kind is ActionKind.NAVIGATE
        if (
            not route_template(destination.route)
            or "*" in destination.route
            or not navigating
        ):
            self.add(IssueCode.INVALID_ROUTE, f"{where}.destination")
        declared = placeholders(destination.route)
        names = tuple(param.name for param in destination.params)
        if sorted(declared) != sorted(names) or len(set(names)) != len(names):
            self.add(IssueCode.INVALID_ACTION, f"{where}.destination")
        for index, param in enumerate(destination.params + destination.query):
            self._text_ref(param.value, f"{where}.destination.params[{index}]")
        if destination.fragment is not None:
            self._text_ref(destination.fragment, f"{where}.destination.fragment")

    def _into(self, node: ActionNode, where: str) -> None:
        if node.into is None:
            return
        if node.kind is not ActionKind.READ or node.into.kind not in {
            RefKind.VARIABLE,
            RefKind.OUTPUT,
        }:
            self.add(IssueCode.INVALID_REFERENCE, f"{where}.into")
            return
        self.reference(node.into, f"{where}.into")

    def _extraction(self, node: ActionNode, target: Target | None, where: str) -> None:
        extraction = node.extraction
        if extraction is None:
            return
        if (
            node.kind is not ActionKind.READ
            or node.into is None
            or not isinstance(target, StructuralTarget)
            or not (extraction.prefix or extraction.suffix)
        ):
            self.add(IssueCode.INVALID_ACTION, f"{where}.extraction")
        for side in ("prefix", "suffix"):
            parts = getattr(extraction, side)
            if len(parts) > 32:
                self.add(IssueCode.INVALID_FIELD, f"{where}.extraction.{side}")
            for index, part in enumerate(parts):
                self._text_ref(part, f"{where}.extraction.{side}[{index}]")
                if part.kind is RefKind.SECRET or part == node.into:
                    self.add(IssueCode.INVALID_REFERENCE, f"{where}.extraction.{side}")

    def _record(self, node: ActionNode, target: Target | None, where: str) -> None:
        record = node.record
        if record is None:
            return
        source = self.targets.get(record.source)
        if not isinstance(source, StructuralTarget) or not isinstance(
            target, StructuralTarget
        ):
            self.add(IssueCode.INVALID_TARGET, f"{where}.record")
            return
        if source.frame != target.frame or source.route != node.route:
            self.add(IssueCode.INVALID_TARGET, f"{where}.record")
        if not _record_shaped(record):
            self.add(IssueCode.INVALID_TARGET, f"{where}.record")
        if record.value.kind not in {RefKind.INPUT, RefKind.VARIABLE}:
            self.add(IssueCode.INVALID_REFERENCE, f"{where}.record.value")
        else:
            self.reference(record.value, f"{where}.record.value")

    def _result_record(self, node: ActionNode, where: str) -> None:
        """Check where a record-bound action's result shows its record.

        It may sit on any screen, because the operation may navigate. It
        belongs only to an action that names its record, and must name the
        same value.
        """
        shown = node.result_record
        if shown is None:
            return
        source = self.targets.get(shown.source)
        if not isinstance(source, StructuralTarget):
            self.add(IssueCode.INVALID_TARGET, f"{where}.result_record")
            return
        if node.record is None or shown.value != node.record.value:
            self.add(IssueCode.INVALID_REFERENCE, f"{where}.result_record")
        if not _record_shaped(shown):
            self.add(IssueCode.INVALID_TARGET, f"{where}.result_record")

    def condition(self, condition: Condition, where: str) -> None:
        match condition:
            case AtRoute(route=route):
                if not route_template(route):
                    self.add(IssueCode.INVALID_ROUTE, where)
            case Present(target=target) | Absent(target=target):
                if target not in self.targets:
                    self.add(IssueCode.UNKNOWN_TARGET, where)
            case Shows(target=target, value=value, record=record):
                self._shows(target, value, where)
                self._check_record(target, record, where)
            case DialogOpen():
                self._dialog(condition, where)
            case Bound(value=value):
                if value.kind not in {RefKind.VARIABLE, RefKind.OUTPUT}:
                    self.add(IssueCode.INVALID_REFERENCE, where)
                else:
                    self.reference(value, where)
            case ValueIs(value=value, expected=expected):
                self._text_ref(value, where)
                self._text_ref(expected, where)
            case Completed(node=node):
                if not self._is_node(node):
                    self.add(IssueCode.UNKNOWN_NODE, where)

    def _dialog(self, condition: DialogOpen, where: str) -> None:
        if condition.kind not in {"alert", "confirm", "prompt", "beforeunload"}:
            self.add(IssueCode.UNKNOWN_VALUE, where)
        if condition.message is not None:
            self._text_ref(condition.message, where)

    def _shows(self, target: str, value: Ref, where: str) -> None:
        found = self.targets.get(target)
        if found is None:
            self.add(IssueCode.UNKNOWN_TARGET, where)
        elif isinstance(found, (VisualTarget, RadioTarget)):
            self.add(IssueCode.UNSUPPORTED_TARGET, where)
        self._text_ref(value, where)

    def _check_record(
        self, target_id: str, record: RecordSpec | None, where: str
    ) -> None:
        if record is None:
            return
        target = self.targets.get(target_id)
        source = self.targets.get(record.source)
        if not isinstance(target, StructuralTarget) or not isinstance(
            source, StructuralTarget
        ):
            self.add(IssueCode.INVALID_TARGET, where)
            return
        if source.frame != target.frame or source.route != target.route:
            self.add(IssueCode.INVALID_TARGET, where)
        if record.value.kind not in {RefKind.INPUT, RefKind.VARIABLE}:
            self.add(IssueCode.INVALID_REFERENCE, where)
        if not _record_shaped(record):
            self.add(IssueCode.INVALID_TARGET, where)
        self.reference(record.value, where)

    def _is_node(self, name: str) -> bool:
        return any(node.node_id == name for node in self.capability.nodes)

    def _edges(self, node: Node, where: str) -> None:
        edges = transitions_of(node)
        if len(edges) > MAX_EDGES:
            self.add(IssueCode.TOO_LARGE, f"{where}.transitions")
        unconditional = [edge for edge in edges if not edge.when]
        if unconditional and len(edges) > 1:
            self.add(IssueCode.INVALID_TRANSITION, f"{where}.transitions")
        for index, edge in enumerate(edges):
            if not self._is_node(edge.to):
                self.add(IssueCode.UNKNOWN_NODE, f"{where}.transitions[{index}].to")
            if edge.limit is not None and not 1 <= edge.limit <= MAX_TRAVERSALS:
                self.add(IssueCode.INVALID_LIMIT, f"{where}.transitions[{index}]")

    def check_structure(self) -> None:
        """Check reachability, dead ends, and that every cycle is bounded."""
        capability = self.capability
        index = {node.node_id: at for at, node in enumerate(capability.nodes)}
        reached = _reachable(capability, capability.entry, skip=None)
        for node in capability.nodes:
            where = f"nodes[{index[node.node_id]}]"
            if node.node_id not in reached:
                self.add(IssueCode.UNREACHABLE_NODE, where)
            elif not isinstance(node, ResultNode) and not _ends(capability, node):
                self.add(IssueCode.DEAD_END, where)
        if _has_unbounded_cycle(capability):
            self.add(IssueCode.UNBOUNDED_CYCLE, "nodes")
        for node in capability.nodes:
            if isinstance(node, ActionNode) and _repeats_change(capability, node):
                self.add(IssueCode.REPEATING_CHANGE, f"nodes[{index[node.node_id]}]")
        if self.issues:
            self.fatal = True

    def check_availability(self) -> None:
        """Check every variable and output is assigned on every path that reads it."""
        capability = self.capability
        entering = _available(capability)
        for position, node in enumerate(capability.nodes):
            where = f"nodes[{position}]"
            before = entering.get(node.node_id, frozenset())
            after = before | _assigns(node)
            for part, used in _uses(capability, node):
                held = after if part.startswith(("verify", "transitions")) else before
                if used not in held:
                    self.add(IssueCode.UNAVAILABLE_VARIABLE, f"{where}.{part}")
            if isinstance(node, ResultNode) and node.result is ResultKind.SUCCESS:
                for output in capability.outputs:
                    if output.required and (RefKind.OUTPUT, output.name) not in before:
                        self.add(IssueCode.MISSING_OUTPUT, f"{where}")

    def check_obligations(self) -> None:
        """Check no transition reaches a success while skipping a mandatory node."""
        capability = self.capability
        successes = {
            node.node_id
            for node in capability.nodes
            if isinstance(node, ResultNode) and node.result is ResultKind.SUCCESS
        }
        for position, node in enumerate(capability.nodes):
            if not isinstance(node, (ActionNode, HumanNode)) or not node.mandatory:
                continue
            if node.node_id == capability.entry:
                continue
            around = _reachable(capability, capability.entry, skip=node.node_id)
            if around & successes:
                self.add(IssueCode.BYPASSES_INTERVENTION, f"nodes[{position}]")

    def check_restrictions(self) -> None:
        capability = self.capability
        for index, saved in enumerate(capability.restrictions):
            where = f"restrictions[{index}]"
            if not route_template(saved.route):
                self.add(IssueCode.INVALID_ROUTE, where)
            scoped = saved.scope in {
                RestrictionScope.TARGET,
                RestrictionScope.OPERATION,
            }
            if scoped != (saved.target is not None):
                self.add(IssueCode.INVALID_FIELD, where)
            if (
                saved.scope is RestrictionScope.TARGET
                and saved.target not in self.targets
            ):
                self.add(IssueCode.UNKNOWN_TARGET, where)
            if len(saved.key) > MAX_IDENTIFIER:
                self.add(IssueCode.INVALID_FIELD, where)
        for position, node in enumerate(capability.nodes):
            if not isinstance(node, (ActionNode, HumanNode)):
                continue
            limit = strictest(
                capability.restrictions,
                node,
                pixel=aims_by_pixels(capability, node),
            )
            if limit is Limit.DENY:
                self.add(IssueCode.DENIED_OPERATION, f"nodes[{position}]")
            elif (
                limit is Limit.RISKY
                and isinstance(node, ActionNode)
                and node.approval is not Approval.EACH_RUN
            ):
                self.add(IssueCode.BYPASSES_RESTRICTION, f"nodes[{position}]")


def branches(node: ActionNode) -> bool:
    """Return whether every transition has conditions that verify the step."""
    return bool(node.transitions) and all(edge.when for edge in node.transitions)


def identifier_like(text: str) -> bool:
    """Report whether ``text`` is a short printable token such as a profile id."""
    return (
        0 < len(text) <= MAX_TEXT
        and all(character.isprintable() for character in text)
        and not any(character.isspace() for character in text)
    )


def _origin(text: str) -> bool:
    try:
        return http_origin(text) == text
    except ValueError:
        return False


def _frame_name(name: str) -> bool:
    return 0 < len(name) <= MAX_TEXT and name.isprintable()


def _field_ok(declared: Field) -> bool:
    if not 0 <= declared.min_length <= declared.max_length <= MAX_TEXT * 5:
        return False
    if (declared.type is ValueType.CHOICE) != bool(declared.choices):
        return False
    return all(
        0 < len(choice) <= MAX_TEXT and choice.isprintable()
        for choice in declared.choices
    ) and len(set(declared.choices)) == len(declared.choices)


def _template_ok(item: Template) -> bool:
    if item.media_type != "image/png" or len(item.data) > MAX_TEMPLATE_BYTES * 2:
        return False
    try:
        png = item.image()
    except ValueError:
        return False
    if len(png) > MAX_TEMPLATE_BYTES or hashlib.sha256(png).hexdigest() != item.sha256:
        return False
    try:
        with Image.open(io.BytesIO(png)) as picture:
            if picture.format != "PNG":
                return False
            size = picture.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return False
    return size == (item.width, item.height) and 0 < max(size) <= MAX_TEMPLATE_SIDE


def _contains_shaped(target: StructuralTarget) -> bool:
    """Report whether a ``CONTAINS`` target has a reference to look for.

    Only an accessibility name or a scope's name is compared by words, and
    one of them must be an input or a variable. A DOM target compares only
    its scope's name that way, such as the row that shows a member, and its
    attribute value stays exact. Otherwise the mode would loosen a constant
    label's exact match for nothing.
    """
    if target.form is not LocatorForm.ACCESSIBILITY:
        return target.scope is not None and target.scope.name.kind in {
            RefKind.INPUT,
            RefKind.VARIABLE,
        }
    names = [target.name]
    if target.scope is not None:
        names.append(target.scope.name)
    return any(
        name is not None and name.kind in {RefKind.INPUT, RefKind.VARIABLE}
        for name in names
    )


def _record_shaped(record: RecordSpec) -> bool:
    """Report whether a record link's saved text leaves the identifier whole.

    A ``CONTAINS`` link saves no text around the value at all.
    """
    if record.match is Match.CONTAINS:
        return not record.prefix and not record.suffix
    return separates(record.prefix, record.suffix)


def _probe_locator(target: StructuralTarget) -> AxLocator | DomLocator:
    """Build a locator from placeholder text, so the vocabulary's own rules apply."""
    if target.form is LocatorForm.ACCESSIBILITY:
        return AxLocator(target.role, "probe", target.frame)
    if target.attribute is None:
        raise ValueError("dom target needs an attribute")
    return DomLocator(target.tag, target.attribute, "probe", target.frame)


def _fits(node: ActionNode, target: Target | None) -> bool:
    """Return whether a screenshot target supports the node's action.

    A template is clicked. A text target's read takes the value after a
    label, and any other action aims at a line, or beside one.
    """
    if isinstance(target, VisualTarget):
        return node.kind in VISUAL_ACTIONS
    if isinstance(target, RadioTarget):
        return node.kind is ActionKind.CLICK
    if isinstance(target, TextTarget):
        return node.kind in TEXT_ACTIONS and target.label == (
            node.kind is ActionKind.READ
        )
    if isinstance(target, FocusTarget):
        return node.kind in FOCUS_ACTIONS
    return True


def _probe_action(node: ActionNode, target: Target | None) -> Action:
    """Build an action to validate the node's structure through ``Action``."""
    placeholder: AxLocator | DomLocator | ScreenTarget | None = None
    if isinstance(target, StructuralTarget):
        placeholder = _probe_locator(target)
    elif isinstance(target, (VisualTarget, TextTarget)):
        placeholder = ScreenTarget("probe", Point(0, 0))
    elif isinstance(target, FocusTarget):
        placeholder = ScreenTarget("probe")
    value: str | SecretRef | None = None
    if node.value is not None:
        value = (
            SecretRef(node.value.name) if node.value.kind is RefKind.SECRET else "probe"
        )
    destination = None if node.destination is None else "https://probe.test/"
    return Action(
        node.kind,
        placeholder,
        value,
        destination,
        effect=node.effect,
    )


def _repeats_change(capability: Capability, node: ActionNode) -> bool:
    """Report whether a cycle could send ``node``'s change a second time (rule 5).

    A visit's delivery record does not reach the next visit, so a cycle back
    to a step that changes data could send it again. A cycle a person wrote,
    such as a bounded retry of a search, closes only through authored edges
    and is allowed. A step that needs approval on every run is a business
    change, and no cycle may pass through it.
    """
    if node.kind not in EFFECTFUL:
        return False
    careful = node.approval is Approval.EACH_RUN or node.mandatory
    seen: set[str] = set()
    pending = [
        edge.to
        for edge in node.transitions
        if careful or edge.origin is not EdgeOrigin.AUTHORED
    ]
    while pending:
        current = pending.pop()
        if current == node.node_id:
            return True
        if current in seen or not _is_known(capability, current):
            continue
        seen.add(current)
        pending.extend(
            edge.to
            for edge in transitions_of(capability.node(current))
            if careful or edge.origin is not EdgeOrigin.AUTHORED
        )
    return False


def _successors(capability: Capability, node_id: str) -> tuple[str, ...]:
    try:
        node = capability.node(node_id)
    except KeyError:
        return ()
    return tuple(edge.to for edge in transitions_of(node))


def _reachable(capability: Capability, start: str, skip: str | None) -> set[str]:
    seen: set[str] = set()
    pending = [start] if start != skip else []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(
            following
            for following in _successors(capability, current)
            if following != skip
        )
    return seen


def _ends(capability: Capability, node: Node) -> bool:
    reached = _reachable(capability, node.node_id, skip=None)
    return any(
        isinstance(capability.node(name), ResultNode)
        for name in reached
        if _is_known(capability, name)
    )


def _is_known(capability: Capability, name: str) -> bool:
    return any(node.node_id == name for node in capability.nodes)


def _has_unbounded_cycle(capability: Capability) -> bool:
    """Report a cycle made only of edges with no traversal limit."""
    unbounded: dict[str, list[str]] = {
        node.node_id: [edge.to for edge in transitions_of(node) if edge.limit is None]
        for node in capability.nodes
    }
    state: dict[str, int] = {}

    def visit(name: str) -> bool:
        state[name] = 1
        for following in unbounded.get(name, ()):
            mark = state.get(following, 0)
            if mark == 1 or (mark == 0 and visit(following)):
                return True
        state[name] = 2
        return False

    return any(state.get(name, 0) == 0 and visit(name) for name in unbounded)


type _Held = frozenset[tuple[RefKind, str]]


def _assigns(node: Node) -> _Held:
    if isinstance(node, ActionNode) and node.into is not None:
        return frozenset({(node.into.kind, node.into.name)})
    return frozenset()


def _available(capability: Capability) -> dict[str, _Held]:
    """Compute definite assignments at each node until the sets stop changing."""
    everything: _Held = frozenset(
        (kind, declared.name)
        for kind, pool in (
            (RefKind.OUTPUT, capability.outputs),
            (RefKind.VARIABLE, capability.variables),
        )
        for declared in pool
    )
    entering: dict[str, _Held] = {node.node_id: everything for node in capability.nodes}
    entering[capability.entry] = frozenset()
    changed = True
    while changed:
        changed = False
        for node in capability.nodes:
            leaving = entering[node.node_id] | _assigns(node)
            for following in _successors(capability, node.node_id):
                if following not in entering or following == capability.entry:
                    continue
                narrowed = entering[following] & leaving
                if narrowed != entering[following]:
                    entering[following] = narrowed
                    changed = True
    return entering


def _uses(
    capability: Capability, node: Node
) -> Iterable[tuple[str, tuple[RefKind, str]]]:
    """Yield each variable or output a node reads, with the part that reads it."""
    for part, condition in conditions_of(node):
        if isinstance(condition, Bound):
            continue
        for value in _condition_refs(capability, condition):
            yield part, value
    if not isinstance(node, ActionNode):
        return
    values: list[Ref | None] = [node.value]
    if node.extraction is not None:
        values.extend(node.extraction.prefix + node.extraction.suffix)
    if node.record is not None:
        values.append(node.record.value)
        values.extend(_target_refs(capability, node.record.source))
    if node.result_record is not None:
        values.extend(_target_refs(capability, node.result_record.source))
    if node.target is not None:
        values.extend(_target_refs(capability, node.target))
    if node.destination is not None:
        values.extend(
            param.value for param in node.destination.params + node.destination.query
        )
        values.append(node.destination.fragment)
    for value in values:
        if value is not None and value.kind in {RefKind.VARIABLE, RefKind.OUTPUT}:
            yield "value", (value.kind, value.name)


def _condition_refs(
    capability: Capability, condition: Condition
) -> Iterable[tuple[RefKind, str]]:
    values: list[Ref | None] = []
    match condition:
        case Present(target=target) | Absent(target=target):
            values.extend(_target_refs(capability, target))
        case Shows(target=target, value=value, record=record):
            values.extend(_target_refs(capability, target))
            values.append(value)
            if record is not None:
                values.extend(_target_refs(capability, record.source))
                values.append(record.value)
        case ValueIs(value=value, expected=expected):
            values.extend((value, expected))
        case Bound(value=value):
            values.append(value)
        case DialogOpen(message=message):
            values.append(message)
        case _:
            pass
    for value in values:
        if value is not None and value.kind in {RefKind.VARIABLE, RefKind.OUTPUT}:
            yield value.kind, value.name


def _target_refs(capability: Capability, target_id: str) -> list[Ref | None]:
    try:
        target = capability.target(target_id)
    except KeyError:
        return []
    if isinstance(target, TextTarget):
        return [target.text]
    if not isinstance(target, StructuralTarget):
        return []
    found = [target.name, target.value]
    if target.scope is not None:
        found.append(target.scope.name)
    return found


# Profile compatibility. A capability can ask only for what the profile grants.


def accepts_inputs(capability: Capability, inputs: Mapping[str, str]) -> bool:
    """Check invocation values before a caller opens any application session."""
    declared = {item.name: item for item in capability.inputs}
    if set(inputs) - set(declared):
        return False
    return all(
        (
            name in inputs
            and isinstance(inputs[name], str)
            and item.accepts(inputs[name])
        )
        or (name not in inputs and not item.required)
        for name, item in declared.items()
    )


def check_profile(capability: Capability, profile: Profile) -> tuple[Issue, ...]:
    """Return capability requirements that exceed the profile's permissions.

    Replay rechecks the profile for every action. Passing this initial check
    grants no additional permission.
    """
    issues = _bindings_against(capability, profile)
    application = capability.application
    if application.profile_id != profile.profile_id:
        issues.append(Issue(IssueCode.PROFILE_MISMATCH, "application.profile_id"))
    scope = profile.scope
    if application.surface is not SurfaceKind.BROWSER:
        issues.append(Issue(IssueCode.PROFILE_MISMATCH, "application.surface"))
        return tuple(issues)
    if scope.origin != application.origin:
        issues.append(Issue(IssueCode.PROFILE_MISMATCH, "application.origin"))
    if not _permitted(profile, application.entry_route):
        issues.append(Issue(IssueCode.ROUTE_NOT_PERMITTED, "application.entry_route"))
    for index, name in enumerate(capability.secrets):
        if name not in profile.secrets:
            issues.append(Issue(IssueCode.UNDECLARED_SECRET, f"secrets[{index}]"))
    issues.extend(_modes(capability, profile))
    for position, node in enumerate(capability.nodes):
        if isinstance(node, ActionNode):
            where = f"nodes[{position}]"
            issues.extend(_node_against(capability, profile, node, where))
        elif isinstance(node, HumanNode):
            issues.extend(_human_against(profile, node, f"nodes[{position}]"))
        if isinstance(node, ResultNode):
            issues.extend(_checks_against(capability, profile, node, position))
    return tuple(issues)


def _bindings_against(capability: Capability, profile: Profile) -> list[Issue]:
    from computeruse.operations import declared, limit

    issues: list[Issue] = []
    business_ids = {
        binding.business
        for item in profile.operations
        if (binding := declared(profile, item.name)) is not None
    }
    for index, saved in enumerate(capability.restrictions):
        if (
            saved.scope is RestrictionScope.OPERATION
            and saved.target not in business_ids
        ):
            issues.append(Issue(IssueCode.PROFILE_MISMATCH, f"restrictions[{index}]"))
    for index, node in enumerate(capability.nodes):
        if not isinstance(node, ActionNode):
            continue
        where = f"nodes[{index}]"
        binding = declared(profile, node.binding)
        if node.binding and (binding is None or binding.business != node.business):
            issues.append(Issue(IssueCode.PROFILE_MISMATCH, f"{where}.binding"))
        definition = next(
            (item for item in profile.operations if item.name == node.binding), None
        )
        if definition is not None and (
            node.kind not in definition.kinds
            or node.route != definition.route
            or (definition.key and node.value != constant(definition.key))
        ):
            issues.append(Issue(IssueCode.PROFILE_MISMATCH, f"{where}.binding"))
        held = limit(
            profile,
            Operation(
                node.kind, node.route, binding=node.binding, business=node.business
            ),
        )
        if held is Limit.DENY:
            issues.append(Issue(IssueCode.DENIED_OPERATION, where))
        elif held is Limit.RISKY and node.approval is not Approval.EACH_RUN:
            issues.append(Issue(IssueCode.BYPASSES_RESTRICTION, where))
    return issues


def _checks_against(
    capability: Capability, profile: Profile, node: ResultNode, position: int
) -> list[Issue]:
    issues: list[Issue] = []
    for index, check in enumerate(node.checks):
        if isinstance(check, Shows):
            target = capability.target(check.target)
            if (
                profile.requires_record(ActionKind.READ, target.route)
                and check.record is None
            ):
                issues.append(
                    Issue(
                        IssueCode.RECORD_EVIDENCE_UNAVAILABLE,
                        f"nodes[{position}].checks[{index}]",
                    )
                )
    return issues


def _human_against(profile: Profile, node: HumanNode, where: str) -> list[Issue]:
    """Check a manual step's declared operation against the profile.

    Reject undeclared action types and denied effects just as for automated
    steps. Human intervention cannot override the profile.
    """
    issues: list[Issue] = []
    if not _permitted(profile, node.route):
        issues.append(Issue(IssueCode.ROUTE_NOT_PERMITTED, where))
    if any(kind not in profile.actions for kind in node.permissions):
        issues.append(Issue(IssueCode.NOT_GRANTED, where))
    if node.performs is None:
        return issues
    if node.performs not in profile.actions:
        issues.append(Issue(IssueCode.NOT_GRANTED, where))
    if profile.effect_limit(node.performs, node.effect) is Limit.DENY:
        issues.append(Issue(IssueCode.EFFECT_DENIED, where))
    return issues


def _permitted(profile: Profile, route: str) -> bool:
    scope = profile.scope
    if route.startswith(("http://", "https://")):
        return scope.route(route) is not None
    return scope.permits_route(route)


def _modes(capability: Capability, profile: Profile) -> list[Issue]:
    issues: list[Issue] = []
    if ActionKind.OBSERVE not in profile.actions:
        issues.append(Issue(IssueCode.NOT_GRANTED, "observe"))
    modes = profile.perception.allowed_modes
    kinds = {type(target) for target in capability.targets}
    if StructuralTarget in kinds and ObservationMode.STRUCTURED not in modes:
        issues.append(Issue(IssueCode.MODE_NOT_PERMITTED, "targets"))
    if kinds & {VisualTarget, TextTarget, RadioTarget, FocusTarget} and (
        ObservationMode.VISUAL not in modes
    ):
        issues.append(Issue(IssueCode.MODE_NOT_PERMITTED, "targets"))
    reads = any(
        isinstance(condition, Shows)
        for node in capability.nodes
        if isinstance(node, ResultNode)
        for condition in node.checks
    )
    if reads and ActionKind.READ not in profile.actions:
        issues.append(Issue(IssueCode.NOT_GRANTED, "read"))
    return issues


def _node_against(
    capability: Capability, profile: Profile, node: ActionNode, where: str
) -> list[Issue]:
    issues: list[Issue] = []
    if node.kind not in profile.actions:
        issues.append(Issue(IssueCode.NOT_GRANTED, where))
    if not _permitted(profile, node.route):
        issues.append(Issue(IssueCode.ROUTE_NOT_PERMITTED, where))
    if node.destination is not None and not _permitted(
        profile, node.destination.origin + node.destination.route
    ):
        issues.append(Issue(IssueCode.ROUTE_NOT_PERMITTED, f"{where}.destination"))
    if profile.effects.get(node.kind) and node.effect is None:
        issues.append(Issue(IssueCode.EFFECT_MISSING, where))
    if profile.effect_limit(node.kind, node.effect) is Limit.DENY:
        issues.append(Issue(IssueCode.EFFECT_DENIED, where))
    if profile.requires_record(node.kind, node.route):
        target = capability.target(node.target) if node.target else None
        if node.record is None or not isinstance(target, StructuralTarget):
            issues.append(Issue(IssueCode.RECORD_EVIDENCE_UNAVAILABLE, where))
    return issues


def effective_budgets(capability: Capability, profile: Profile) -> Budgets:
    """Return the tighter of the capability's limits and the profile's budgets.

    A capability can lower a limit but cannot exceed the operator's budget.
    """
    limits = capability.limits
    budgets = profile.budgets
    return Budgets(
        max_steps=min(limits.max_steps, budgets.max_steps),
        max_wall_clock_s=min(limits.max_wall_clock_s, budgets.max_wall_clock_s),
        max_retries_per_step=min(
            limits.max_target_attempts, budgets.max_retries_per_step
        ),
        max_navigations=budgets.max_navigations,
    )
