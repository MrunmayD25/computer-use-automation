"""Convert verified discovery events into a draft capability.

The trace stays in memory, separate from journals and model transcripts.
The draft uses executed actions, observed checks, and declared input bindings.
Manual work remains an explicit human node. A segment that cannot be
represented or is forbidden prevents export instead of being omitted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, unquote, urlsplit

from computeruse import matching, operations, policy, reading, visual
from computeruse.actions import (
    DIALOG_ACTIONS,
    DOM_TAGS,
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    DomAttribute,
    DomLocator,
    MouseInput,
    Observation,
    ObservationMode,
    Operation,
    Outcome,
    Scope,
    ScopeKind,
    ScreenTarget,
    SecretRef,
    Target,
    in_scope,
    names_node,
    positional_slot,
    word_positions,
)
from computeruse.capability import (
    FOCUS_ACTIONS,
    MAX_OFFSET,
    TEXT_ACTIONS,
    Field,
    HelpReason,
    Limits,
    LocatorForm,
    Match,
    Purpose,
    Ref,
    RefKind,
    StoragePermit,
    ValueType,
    ref,
)
from computeruse.contract import Contract
from computeruse.control import settled
from computeruse.decider import CheckKind as ResultKind
from computeruse.decider import Fact, FactRef, ResultCheck, uncovered_requirements
from computeruse.escalation import (
    Handoff,
    HandoffOutcome,
    Interruption,
    InterventionRequest,
    Trigger,
)
from computeruse.manual import (
    Detail,
    ManualKind,
    ManualSegment,
    Requirement,
    single_click,
)
from computeruse.profile import ActionKind, Profile, Risk
from computeruse.reading import Line, Reader
from computeruse.recorder import (
    INTERFACE_ROLES,
    CheckKind,
    CheckSample,
    DestinationSample,
    DiscoveryRun,
    Dispatch,
    Executed,
    Finished,
    FocusSample,
    Gap,
    HumanSegment,
    Learned,
    Recorder,
    Recording,
    RecordingIssue,
    RecordSample,
    ScreenSample,
    TargetEvidence,
    TargetSample,
    TextSample,
    Unneeded,
    ValueSample,
    Verified,
    Verifier,
    VisualSample,
    holds,
)
from computeruse.retarget import complete
from computeruse.urls import http_origin

if TYPE_CHECKING:
    from computeruse.loop import CheckResult, RunResult


@dataclasses.dataclass(frozen=True)
class ActionTaken:
    """One surface call and the checked observation preceding it."""

    step: int
    action: Action
    result: ActionResult
    location: str
    observation: Observation | None
    operation: Operation | None = None


@dataclasses.dataclass(frozen=True)
class Handed:
    """A completed intervention, including manual work when recorded."""

    request: InterventionRequest
    answer: Handoff
    segments: tuple[ManualSegment, ...]


@dataclasses.dataclass(frozen=True)
class CheckedAction:
    """Postconditions read from the live surface after an executed action."""

    step: int
    checks: tuple[ResultCheck, ...]
    observation: Observation | None


@dataclasses.dataclass(frozen=True)
class FactKept:
    """A verified reading retained only in the live trace."""

    fact: Fact
    reading: int | None = None


@dataclasses.dataclass(frozen=True)
class ValueUsed:
    """The named fact a proposal used, independent of its sample value."""

    step: int
    key: str
    input_value: bool = False


@dataclasses.dataclass(frozen=True)
class Labelled:
    """The effect and approval requirement of one manual click.

    The loop records this after control returns, the decider names the effect,
    and the policy gate permits it. ``careful`` means the gate, effect rules,
    or a learned restriction requires approval.
    """

    intervention: str
    effect: str
    careful: bool


type Entry = (
    ActionTaken | Observation | Handed | CheckedAction | FactKept | ValueUsed | Labelled
)


class DiscoveryTrace:
    """Collect execution evidence in memory, separate from the journal."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def action(
        self,
        step: int,
        action: Action,
        result: ActionResult,
        location: str,
        observation: Observation | None,
        operation: Operation | None = None,
    ) -> None:
        """Record the actual result of an adapter call."""
        self.entries.append(
            ActionTaken(step, action, result, location, observation, operation)
        )

    def look(self, observation: Observation) -> None:
        """Retain a permitted structured observation for later checks."""
        if observation.usable:
            self.entries.append(observation)

    def handoff(
        self,
        request: InterventionRequest,
        answer: Handoff,
        segments: tuple[ManualSegment, ...],
    ) -> None:
        """Record a control transfer in execution order."""
        if answer.outcome is HandoffOutcome.RESUMED or answer.changed:
            self.entries.append(Handed(request, answer, segments))

    def labelled(self, intervention: str, effect: str, *, careful: bool) -> None:
        """Retain the approved effect label for a manual click."""
        self.entries.append(Labelled(intervention, effect, careful))

    def verified(
        self,
        step: int,
        checks: tuple[ResultCheck, ...],
        observation: Observation | None,
    ) -> None:
        """Retain only postconditions verified by the discovery loop."""
        self.entries.append(CheckedAction(step, checks, observation))

    def kept(self, fact: Fact) -> None:
        """Retain verified source provenance without serializing its value."""
        reading = next(
            (
                index
                for index in range(len(self.entries) - 1, -1, -1)
                if isinstance(entry := self.entries[index], ActionTaken)
                and entry.step == fact.step
                and entry.action.kind is ActionKind.READ
                and entry.action.target == fact.source
                and entry.result.outcome is Outcome.OK
            ),
            None,
        )
        self.entries.append(FactKept(fact, reading))

    def used(self, step: int, key: str) -> None:
        """Bind an invocation value to a named fact."""
        self.entries.append(ValueUsed(step, key))

    def input_used(self, step: int, name: str) -> None:
        """Bind a value to its declared input name even when samples are equal."""
        self.entries.append(ValueUsed(step, name, input_value=True))

    def build(
        self,
        result: RunResult,
        *,
        profile: Profile,
        inputs: Mapping[str, str],
        capability_id: str,
        run: str,
        safe_text: frozenset[str] = frozenset(),
        contract: Contract | None = None,
        visual_templates: Mapping[int, bytes] | None = None,
        visual_results: Mapping[int, bytes] | None = None,
        collect: bool = False,
        reader: Reader | None = None,
        excluded: tuple[str, ...] = (),
    ) -> Recording:
        """Build a draft from a completed run or report why recording is incomplete.

        ``safe_text`` contains confirmed website text. With ``collect``, retain
        unconfirmed text as candidates unless it contains an output, remembered
        fact, or input. The caller must confirm every candidate before saving.

        ``reader`` extracts text from screenshots so canvas actions can use text
        targets. Without it, those actions require manual steps.

        ``excluded`` supplies other known private values, such as resolved
        secrets or comparison inputs. They override confirmed website text.
        """
        return _Export(
            self.entries,
            result,
            profile,
            inputs,
            capability_id,
            run,
            safe_text,
            contract,
            visual_templates,
            visual_results,
            collect,
            reader,
            excluded,
        ).build()


def field(name: str) -> Field:
    """Declare a bounded textual capability parameter or output."""
    return Field(name, ValueType.TEXT, True, 1, 1000, ())


def _value(value: str | SecretRef) -> ValueSample:
    if isinstance(value, SecretRef):
        return ValueSample(RefKind.SECRET, name=value.name)
    return ValueSample(RefKind.CONSTANT, text=value)


_PRESET = {
    "select": ActionKind.SELECT,
    "input": ActionKind.TYPE,
    "textarea": ActionKind.TYPE,
}
"""Form fields that may be prefilled and the action used to set each."""

APPEARING_ROLES = INTERFACE_ROLES | {"dialog", "menu", "tabpanel"}
"""Self-labeled controls whose appearance can verify a click."""


def _alone(lines: tuple[Line, ...], line: Line) -> bool:
    """Return whether the text identifies exactly one line in the capture."""
    same = reading.squash(line.text)
    return sum(reading.squash(other.text) == same for other in lines) == 1


def _label_above(lines: tuple[Line, ...], x: float, y: float) -> Line | None:
    """Return the nearest eligible label above or left of a field point.

    The label must end above the point or share its row to the left.
    Both distances must be within ``MAX_OFFSET``.

    Examples
    --------
    >>> label = Line("Nickname", 40, 300, 80, 20, 0.99)
    >>> other = Line("Review", 40, 420, 60, 20, 0.99)
    >>> _label_above((label, other), 60.0, 345.0) is label
    True
    """
    above = [
        line
        for line in lines
        if not line.holds(x, y)
        and (line.top + line.height <= y or line.left + line.width <= x)
        and abs(x - line.left) <= MAX_OFFSET
        and 0 <= y - line.top <= MAX_OFFSET
    ]
    if not above:
        return None
    return min(above, key=lambda line: (y - line.top) + abs(x - line.left))


def _label_before(text: str, value: str) -> str | None:
    """Return the label before a value at the end of a line, or None.

    Use ``matching.label_before``, also used by discovery to associate
    screenshot values with records.
    """
    return matching.label_before(text, value)


def _whole_line(text: str, expected: str) -> bool:
    """Report whether ``expected`` is the whole line, apart from spacing and case.

    Examples
    --------
    >>> _whole_line("No order with number O-2001.", "No order with number O-2001")
    True
    >>> _whole_line("No order with number O-2001.", "No order")
    False
    """
    line, wanted = reading.squash(text), reading.squash(expected)
    return bool(wanted) and line.rstrip(matching.EDGE) == wanted.rstrip(matching.EDGE)


def _held_input(node: AxNode, inputs: Iterable[str]) -> str | None:
    """Return the unique input appearing once as whole words in a control.

    Examples
    --------
    >>> _held_input(AxNode("text", "Sam Lee (U-7)"), ("U-7", "NM1"))
    'U-7'
    >>> _held_input(AxNode("text", "U-7 to NM1"), ("U-7", "NM1")) is None
    True
    """
    if node.secret or node.value is not None or not node.visible:
        return None
    held = [value for value in inputs if value and value in node.name]
    if len(held) != 1 or len(word_positions(node.name, held[0])) != 1:
        return None
    return held[0]


def _holding(seen: Observation, like: AxNode, value: str) -> list[AxNode]:
    """Return visible controls matching ``value`` and the given role and frame."""
    return [
        node
        for node in seen.nodes
        if node.visible
        and node.role == like.role
        and node.frame == like.frame
        and _held_input(node, (value,)) == value
    ]


def _node(target: Target | None, seen: Observation | None) -> AxNode | None:
    if seen is None:
        return None
    found = [node for node in seen.nodes if names_node(target, node)]
    return found[0] if len(found) == 1 else None


def _label(
    action: Action, value: str | SecretRef, seen: Observation | None
) -> str | SecretRef:
    """Resolve a selected option's stored value to its displayed label.

    Options can pair a label such as "Gold checking" with a code such as
    ``product-gold-checking``. Replay can match the request's text in that
    label instead of storing the code.
    """
    if action.kind is not ActionKind.SELECT or not isinstance(value, str):
        return value
    node = _node(action.target, seen)
    labels = [
        option.label
        for option in (node.options if node is not None else ())
        if option.value == value and option.label
    ]
    return labels[0] if len(labels) == 1 else value


def target_sample(
    target: Target | None,
    route: str,
    seen: Observation | None,
    *,
    inputs: tuple[str, ...] = (),
    safe_text: frozenset[str] = frozenset(),
    outputs: tuple[str, ...] = (),
) -> TargetSample | ScreenSample | None:
    """Prefer a unique semantic target, then a permitted reusable attribute.

    An output value cannot identify its own control because it varies by
    record. Use a slot or another attribute for those controls.
    """
    if target is None:
        return None
    node = _node(target, seen)
    if node is not None:
        candidates: list[AxLocator | DomLocator] = []
        for alias in _row_aliases(node, inputs, safe_text):
            # The row's own name may be its data, such as an event number.
            # Another cell of the same row that shows the request's value
            # finds the same row for any request.
            candidates.extend(_slot_candidates(alias, safe_text))
        if (
            node.row is not None
            and node.name not in inputs
            and not frozenset.__contains__(safe_text, node.name)
        ):
            # A name nobody confirmed, in a row, can be the record's data,
            # such as a receipt number. The cell's slot names where it is.
            candidates.extend(_slot_candidates(node, safe_text))
        if (node.name in safe_text or node.name in inputs) and node.name not in outputs:
            candidates.extend(
                (
                    AxLocator(node.role, node.name, frame=node.frame),
                    AxLocator(node.role, node.name, frame=node.frame, scope=node.scope),
                )
            )
        candidates.extend(_dom_candidates(node, inputs, safe_text))
        for candidate in candidates:
            if _node(candidate, seen) == node:
                target = candidate
                break
        else:
            if (
                node.row is not None
                and isinstance(target, DomLocator)
                and target.attribute is DomAttribute.ID
            ):
                # An id inside a repeated row names one record; see
                # ``_dom_candidates``. Without another way to find the cell
                # again, the step is not reusable.
                return ScreenSample(route)
    if isinstance(target, AxLocator) and _ambiguous_by_input(target, seen, inputs):
        # Saved as "the control holding this input", it would match more than
        # one control, such as a window's close and minimize buttons that both
        # carry the record in their names.
        return ScreenSample(route)
    if isinstance(target, AxLocator) and not target.occurrence:
        return TargetSample(
            route,
            LocatorForm.ACCESSIBILITY,
            role=target.role,
            name=target.name,
            frame=target.frame,
            scope=target.scope,
        )
    if isinstance(target, DomLocator) and not target.occurrence:
        return TargetSample(
            route,
            LocatorForm.DOM,
            tag=target.tag,
            attribute=target.attribute,
            value=target.value,
            frame=target.frame,
            scope=target.scope,
        )
    return ScreenSample(route)


def _ambiguous_by_input(
    target: AxLocator, seen: Observation | None, inputs: tuple[str, ...]
) -> bool:
    """Return whether an input reference would make the target ambiguous.

    Saving only the input removes surrounding text that may distinguish
    controls. The target is reusable only if exactly one control with its
    role, frame, and scope contains the input.
    """
    if seen is None:
        return False
    held = [value for value in inputs if value and value in target.name]
    if len(held) != 1 or target.name == held[0]:
        return False
    alike = [
        node
        for node in seen.nodes
        if node.visible
        and node.role == target.role
        and node.frame == target.frame
        and (target.scope is None or in_scope(target.scope, node))
        and len(word_positions(node.name, held[0])) == 1
    ]
    return len(alike) > 1


def _dom_candidates(
    node: AxNode, inputs: tuple[str, ...], safe_text: frozenset[str]
) -> list[DomLocator]:
    """Offer permitted attributes, classes, and semantic slots.

    IDs within repeated table rows identify individual records, even if they
    do not contain the record number. Exclude those IDs. Prefer the column
    slot within the requested record's row.
    """
    if node.tag not in DOM_TAGS:
        return []
    in_row = node.row is not None
    slots = _slot_candidates(node, safe_text)
    candidates: list[DomLocator] = list(slots) if in_row else []
    for attribute in (
        DomAttribute.NAME,
        DomAttribute.DATA_VALUE,
        DomAttribute.ID,
        DomAttribute.DATA_TESTID,
        DomAttribute.ARIA_LABEL,
        DomAttribute.TITLE,
        DomAttribute.PLACEHOLDER,
    ):
        if in_row and attribute is DomAttribute.ID:
            continue
        value = dict(node.attributes).get(attribute.value, "")
        if (
            value
            and (value in safe_text or value in inputs)
            and not any(sample in value and sample != value for sample in inputs)
        ):
            candidates.extend(
                DomLocator(node.tag, attribute, value, frame=node.frame, scope=scope)
                for scope in (None, node.scope)
            )
    for name in node.classes:
        if name in safe_text:
            candidates.extend(
                DomLocator(
                    node.tag,
                    DomAttribute.CSS_CLASS,
                    name,
                    frame=node.frame,
                    scope=scope,
                )
                for scope in (None, node.scope)
            )
    if not in_row:
        candidates.extend(slots)
    return candidates


def _field_labels(field: AxNode) -> set[str]:
    """Return a field's accessible name and visible label.

    The collector stores unlinked labels in the field's slot, such as
    ``select|Branch``.

    Examples
    --------
    >>> chosen = AxNode("combobox", "", tag="select", slot="select|Branch")
    >>> sorted(_field_labels(chosen))
    ['Branch']
    """
    labels = {matching.spaced(field.name)} if field.name.strip() else set()
    if "|" in field.slot:
        labels.add(matching.spaced(field.slot.split("|", 1)[1]))
    return labels


def _row_aliases(
    node: AxNode, inputs: tuple[str, ...], safe_text: frozenset[str]
) -> list[AxNode]:
    """Scope ``node`` by row aliases containing an input exactly once.

    The row's first unique cell may contain a generated sequence number that
    works only for one record. Another cell containing the requested input
    can identify the row across invocations.
    """
    if node.scope is None or node.scope.kind is not ScopeKind.ROW:
        return []
    first = node.scope.name
    if first in inputs or frozenset.__contains__(safe_text, first):
        return []
    aliases: list[AxNode] = []
    columns = node.row_columns or (0,) * len(node.row_names)
    for name, column in zip(node.row_names, columns, strict=False):
        if name == first:
            continue
        held = [value for value in inputs if value and word_positions(name, value)]
        if len(held) == 1 and len(word_positions(name, held[0])) == 1:
            # The alias keeps its column, so a replay never takes a cell in
            # another column that mentions the value for this one.
            scope = Scope(ScopeKind.ROW, name, column)
            aliases.append(dataclasses.replace(node, scope=scope))
    return aliases


def _slot_candidates(node: AxNode, safe_text: frozenset[str]) -> list[DomLocator]:
    """Offer a permitted slot both alone and within its scope.

    A slot identified by column position contains no page text and needs no
    text confirmation.
    """
    if not node.slot or not (node.slot in safe_text or positional_slot(node.slot)):
        return []
    return [
        DomLocator(
            node.tag, DomAttribute.SLOT, node.slot, frame=node.frame, scope=scope
        )
        for scope in (None, node.scope)
    ]


class _Collecting(frozenset[str]):
    """Confirmed website text and unconfirmed candidates eligible for review.

    During collection, target selection accepts text without excluded values.
    The recorder retains each unconfirmed text as a review candidate.
    """

    excluded: tuple[str, ...]

    def __new__(cls, known: frozenset[str], excluded: tuple[str, ...]) -> _Collecting:
        words = super().__new__(
            cls,
            (
                text
                for text in known
                if not any(holds(text, value) for value in excluded)
            ),
        )
        words.excluded = tuple(value for value in excluded if value)
        return words

    def __contains__(self, text: object) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        return not any(holds(text, value) for value in self.excluded)


def _names(target: object) -> set[str]:
    """Return the text a structural locator uses to identify its control."""
    names: set[str] = set()
    if isinstance(target, AxLocator):
        names.add(target.name)
    if isinstance(target, (AxLocator, DomLocator)) and target.scope is not None:
        names.add(target.scope.name)
    return names


def _naming(entries: list[Entry], checks: Iterable[CheckResult]) -> set[str]:
    """Return facts whose values later identify controls.

    A created record's identifier may name the link used to reopen it. Save
    that fact as a variable so replay locates the record it created instead
    of retaining the identifier from discovery.
    """
    read: dict[str, str] = {}
    naming: set[str] = set()
    for entry in entries:
        if isinstance(entry, FactKept) and entry.fact.source is not None:
            read[entry.fact.value] = entry.fact.key
        elif isinstance(entry, ActionTaken):
            naming.update(
                read[name] for name in _names(entry.action.target) if name in read
            )
    for checked in checks:
        if not checked.passed:
            continue
        naming.update(
            read[name] for name in _names(checked.check.target) if name in read
        )
    return naming


def _saved(check: ResultCheck) -> tuple[Match, Purpose]:
    """Preserve a discovery check's comparison rules during recording.

    Discovery matches record identifiers as whole words regardless of the
    check's match setting. Save those checks with ``CONTAINS`` to preserve
    that behavior under rule 3.

    Examples
    --------
    >>> from computeruse.actions import AxLocator
    >>> from computeruse.decider import CheckKind as Kind
    >>> _saved(ResultCheck(Kind.RECORD, AxLocator("cell", "M-1"), "M-1"))
    (<Match.CONTAINS: 'contains'>, <Purpose.RECORD: 'record'>)
    """
    purpose = Purpose(check.kind.value)
    if purpose is Purpose.RECORD:
        return Match.CONTAINS, purpose
    return Match(check.match.value), purpose


def _extraction_bounds(text: str, value: str) -> tuple[str, str] | None:
    text = matching.spaced(text)
    found = matching.positions(text, value)
    if len(found) != 1:
        return None
    start, end = found[0]
    prefix, suffix = text[:start], text[end:]
    return (prefix, suffix) if matching.extract(text, prefix, suffix) == value else None


class _Export:
    def __init__(
        self,
        entries: list[Entry],
        result: RunResult,
        profile: Profile,
        inputs: Mapping[str, str],
        identity: str,
        run: str,
        safe_text: frozenset[str],
        contract: Contract | None = None,
        visual_templates: Mapping[int, bytes] | None = None,
        visual_results: Mapping[int, bytes] | None = None,
        collect: bool = False,
        reader: Reader | None = None,
        excluded: tuple[str, ...] = (),
    ) -> None:
        self.entries = entries
        self.reader = reader
        self.visual_templates = visual_templates or {}
        self.visual_results = visual_results or {}
        self.captures = {
            entry.observation_id: entry
            for entry in entries
            if isinstance(entry, Observation) and entry.mode is ObservationMode.VISUAL
        }
        self.checked_steps = {
            entry.step for entry in entries if isinstance(entry, CheckedAction)
        }
        self.result = result
        self.profile = profile
        self.input_checked: set[int] = set()
        self.labels: dict[str, Labelled] = {}
        self.pending_seen: Observation | None = None
        self.appeared_checked: set[int] = set()
        self.painted_checked: set[int] = set()
        self.seen: Observation | None = None
        self.step = 0
        self.previous: ActionTaken | None = None
        self.transition: tuple[ActionTaken, int] | None = None
        self.action_steps: dict[int, int] = {}
        self.inputs = inputs
        self.contract = contract
        # Nothing that holds an output or a remembered value may be saved,
        # whoever would confirm it.
        excluded = (
            *excluded,
            *inputs.values(),
            *result.outputs.values(),
            *(entry.fact.value for entry in entries if isinstance(entry, FactKept)),
            *profile.confirmation.values(),
            *(value for case in profile.outcomes.values() for value in case.values()),
        )
        safe_text = frozenset(
            text
            for text in safe_text
            if not any(holds(text, value) for value in excluded)
        )
        self.safe_text: frozenset[str] = (
            _Collecting(safe_text, excluded) if collect else safe_text
        )
        declared_outputs = (
            {field.name for field in contract.outputs} if contract else set()
        )
        self.output_names = {
            name: name if name in declared_outputs else "output_" + str(index + 1)
            for index, name in enumerate(result.outputs)
        }
        self.issues: list[RecordingIssue] = []
        self.kept: tuple[tuple[Ref, str], ...] = ()
        self.read: set[str] = set()
        self.repeated: set[int] = set()
        self.partial: dict[int, list[tuple[ValueSample, Purpose]]] = {}
        self.partial_keys: set[str] = set()
        self.extractions: dict[int, tuple[str, str]] = {}
        self.set_controls: set[str] = set()
        self.preset_checks: list[CheckSample] = []
        self.fact_refs, self.fact_reads, self.used_values = self._facts(entries)
        self.pending: tuple[int, Handed] | None = None
        self.replaces: int | None = None
        limits = profile.budgets
        self.recorder = Recorder(
            capability_id=identity,
            version=1,
            profile=profile,
            origin=DiscoveryRun(run),
            inputs=contract.inputs
            if contract
            else tuple(field(name) for name in inputs),
            outputs=contract.outputs
            if contract
            else tuple(field(self.output_names[name]) for name in result.outputs),
            variables=tuple(
                field(value.name)
                for key, value in self.fact_refs.items()
                if value.source is RefKind.VARIABLE and key not in self.partial_keys
            ),
            outcomes=(result.outcome,) if result.outcome else (),
            samples=inputs,
            safe_text=frozenset(safe_text),
            collect=collect,
            excluded=excluded,
            kept=self.kept,
            limits=Limits(
                limits.max_steps,
                limits.max_wall_clock_s,
                max(1, limits.max_retries_per_step),
                3,
                100,
                5,
            ),
        )
        self.output_targets = {
            checked.check.output: checked.check.target
            for checked in result.checks
            if checked.passed
            and checked.check.kind is ResultKind.RESULT
            and checked.check.output in self.output_names
        }

    def _facts(
        self, entries: list[Entry]
    ) -> tuple[dict[str, ValueSample], dict[int, ValueSample], dict[int, ValueSample]]:
        outputs = {
            item.check.target.key: ValueSample(
                RefKind.OUTPUT, name=self.output_names[item.check.output]
            )
            for item in self.result.checks
            if isinstance(item.check.target, FactRef)
            and item.check.kind is ResultKind.RESULT
            and item.check.output in self.output_names
        }
        wanted = (
            {
                entry.key
                for entry in entries
                if isinstance(entry, ValueUsed) and not entry.input_value
            }
            | {
                item.check.target.key
                for item in self.result.checks
                if isinstance(item.check.target, FactRef)
            }
            | _naming(entries, self.result.checks)
        )
        refs = {
            key: outputs.get(
                key, ValueSample(RefKind.VARIABLE, name=f"fact_{index + 1}")
            )
            for index, key in enumerate(sorted(wanted))
        }
        reads: dict[int, ValueSample] = {}
        kept: dict[str, str] = {}
        for entry in entries:
            if (
                isinstance(entry, FactKept)
                and entry.reading is not None
                and entry.fact.source is not None
                and not entry.fact.may_change
                and kept.get(entry.fact.key) == entry.fact.value
                and entry.reading not in reads
            ):
                # A second read of an unchanging fact, such as an identifier,
                # adds nothing. A fact that may change is read again wherever
                # discovery read it again, whatever the sample showed (rule 11).
                self.repeated.add(entry.reading)
                continue
            if not isinstance(entry, FactKept) or entry.fact.key not in refs:
                continue
            if self._part_of_reading(entry) and not self._extract_at_reading(
                entries, entry
            ):
                self._check_at_reading(entries, entry)
                continue
            if entry.fact.source is None:
                refs[entry.fact.key] = _value(entry.fact.value)
            elif entry.reading is None or (
                entry.reading in reads and reads[entry.reading] != refs[entry.fact.key]
            ):
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        entry.fact.step,
                        "a kept fact differs from its saved reading",
                    )
                )
            else:
                reads[entry.reading] = refs[entry.fact.key]
                kept[entry.fact.key] = entry.fact.value
        samples: dict[Ref, set[str]] = {}
        for entry in entries:
            if isinstance(entry, FactKept) and entry.fact.key in kept:
                source = refs[entry.fact.key]
                reference = ref(source.source, source.name)
                samples.setdefault(reference, set()).add(entry.fact.value)
        self.kept = tuple(
            (reference, next(iter(values)))
            for reference, values in samples.items()
            if len(values) == 1
        )
        uses = {
            entry.step: ValueSample(RefKind.INPUT, name=entry.key)
            if entry.input_value
            else refs[entry.key]
            for entry in entries
            if isinstance(entry, ValueUsed) and (entry.input_value or entry.key in refs)
        }
        return refs, reads, uses

    def _extract_at_reading(self, entries: list[Entry], entry: FactKept) -> bool:
        """Save outputs and partial facts with checked extraction boundaries."""
        if entry.reading is None:
            return False
        key = entry.fact.key
        needed = (
            any(
                isinstance(item, ValueUsed) and item.key == key and not item.input_value
                for item in entries
            )
            or key in _naming(entries, self.result.checks)
            or any(
                isinstance(item.check.target, FactRef)
                and item.check.target.key == key
                and item.check.kind is ResultKind.RESULT
                for item in self.result.checks
            )
        )
        reading = entries[entry.reading]
        if not needed or not isinstance(reading, ActionTaken):
            return False
        bounds = _extraction_bounds(reading.result.extracted or "", entry.fact.value)
        if bounds is None:
            return False
        self.extractions[entry.reading] = bounds
        return True

    @staticmethod
    def _part_of_reading(entry: FactKept) -> bool:
        """Return whether a retained fact contains only part of the source text.

        Discovery can retain an identifier appearing once in longer text. Saving
        the whole read into that fact's variable would change the value used by
        later comparisons.
        """
        if entry.reading is None or entry.fact.source is None:
            return False
        # The loop marks the fact when it keeps it; comparing texts here
        # would mistake a long reading for a partial one.
        return entry.fact.partial

    def _check_at_reading(self, entries: list[Entry], entry: FactKept) -> None:
        """Save a partial fact as a check at its original read location.

        A final comparison against a requested value becomes a whole-word check
        on the original control. Other uses need an extracted value that this
        read cannot supply, so they leave a recording gap.
        """
        key = entry.fact.key
        compared = [
            checked.check
            for checked in self.result.checks
            if isinstance(checked.check.target, FactRef)
            and checked.check.target.key == key
            and checked.passed
        ]
        used = any(
            isinstance(item, ValueUsed) and item.key == key and not item.input_value
            for item in entries
        )
        if (
            used
            or not compared
            or key in _naming(entries, self.result.checks)
            or any(check.kind is ResultKind.RESULT for check in compared)
        ):
            self.issues.append(
                RecordingIssue(
                    Gap.UNREPRESENTABLE_CHECK,
                    entry.fact.step,
                    "a partial fact cannot be checked where it was read",
                )
            )
            return
        reading = entry.reading if entry.reading is not None else -1
        self.partial.setdefault(reading, []).extend(
            (_value(check.expected), _saved(check)[1]) for check in compared
        )
        self.partial_keys.add(key)

    def build(self) -> Recording:
        """Translate trace entries in order, retaining every manual boundary."""
        # A run that ended in a business outcome returns no outputs, so the
        # contract's outputs are compared only for a run that returned values.
        outputs = (
            set()
            if self.result.outcome
            else {field.name for field in self.contract.outputs}
            if self.contract
            else set()
        )
        if self.contract and (
            {field.name for field in self.contract.inputs} != set(self.inputs)
            or outputs != set(self.result.outputs)
            or any(
                not field.accepts(self.inputs[field.name])
                for field in self.contract.inputs
            )
        ):
            return Recording(None, (RecordingIssue(Gap.INVALID_CAPABILITY, 0),))
        if any(
            record.value not in self.inputs.values()
            for record in self.result.task.records
        ) or any(
            checked.check.kind is ResultKind.RECORD
            and checked.check.expected not in self.inputs.values()
            for checked in self.result.checks
        ):
            return Recording(None, (RecordingIssue(Gap.UNBOUND_RECORD, 0),))
        for index, entry in enumerate(self.entries):
            self._entry(index, entry)
        if self.pending:
            self.issues.append(
                RecordingIssue(Gap.MANUAL_WITHOUT_CONTINUATION, self.step)
            )
        for restriction in self.result.restrictions:
            self.recorder.record(Learned(restriction))
        if self.result.ending.value == "completed":
            self._finish()
        if self.issues:
            return Recording(None, tuple(self.issues))
        return self.recorder.finish()

    def _entry(self, index: int, entry: Entry) -> None:
        """Convert one trace entry in execution order."""
        if isinstance(entry, ActionTaken):
            self._action(entry, index)
        elif isinstance(entry, Observation):
            self._look(entry)
        elif isinstance(entry, CheckedAction):
            self._verified(entry)
        elif isinstance(entry, Handed):
            self._human(entry)
        elif isinstance(entry, Labelled):
            self.labels[entry.intervention] = entry
        elif isinstance(entry, FactKept) and any(
            value == entry.fact.value for _, value in self.kept
        ):
            self.read.add(entry.fact.value)

    def _sample(
        self, target: Target | None, route: str
    ) -> TargetSample | ScreenSample | None:
        sample = target_sample(
            target,
            route,
            self.seen,
            # A control named by a value the run read and kept is found by
            # that value, from the read on, never at the read itself.
            inputs=(*self.inputs.values(), *self.read),
            safe_text=self.safe_text,
            outputs=tuple(
                value
                for value in self.result.outputs.values()
                if value not in self.inputs.values()
            ),
        )
        if isinstance(sample, TargetSample):
            label = (
                sample.name
                if sample.form is LocatorForm.ACCESSIBILITY
                else sample.value
            )
            # An output names no control, unless the run read it first and
            # saves it as the reference the control is then found by.
            if (
                label
                and label in self.result.outputs.values()
                and label not in self.safe_text
                and all(label != value for _, value in self.kept)
            ):
                return ScreenSample(route)
        return sample

    def _record(self, action: Action, route: str) -> RecordSample | None:
        if action.evidence is None:
            return None
        source = self._read_source(action, route) or self._sample(
            action.evidence.source, route
        )
        if not isinstance(source, TargetSample):
            return None
        return RecordSample(
            source,
            _value(action.evidence.value),
            action.evidence.relation,
            action.evidence.prefix,
            action.evidence.suffix,
        )

    def _read_source(self, action: Action, route: str) -> TargetSample | None:
        if action.kind is not ActionKind.READ or action.evidence is None:
            return None
        seen = self.seen
        if not complete(seen) or seen is None:
            return None
        node = _node(action.target, seen)
        if (
            node is None
            or node != _node(action.evidence.source, seen)
            or not node.visible
            or node.secret
            or node.value is not None
            or node.scope is not None
            or any(
                kind in node.interactions
                for kind in (ActionKind.TYPE, ActionKind.SELECT)
            )
        ):
            return None
        existing = self._sample(action.target, route)
        if isinstance(existing, TargetSample) and existing.form is LocatorForm.DOM:
            return existing
        value = action.evidence.value
        if (
            sum(held == value for held in self.inputs.values()) != 1
            or len(matching.positions(node.name, value)) != 1
        ):
            return None
        alike = [
            candidate
            for candidate in seen.nodes
            if candidate.visible
            and candidate.role == node.role
            and candidate.frame == node.frame
            and len(matching.positions(candidate.name, value)) == 1
        ]
        if alike != [node]:
            return None
        return TargetSample(
            route,
            LocatorForm.ACCESSIBILITY,
            role=node.role,
            name=value,
            frame=node.frame,
            match=Match.CONTAINS,
        )

    def _output_extraction(
        self,
        entry: ActionTaken,
        index: int,
        into: ValueSample | None,
        target: TargetEvidence | None,
    ) -> tuple[str, str] | None:
        extraction = self.extractions.get(index)
        if (
            into is None
            or index in self.fact_reads
            or not isinstance(target, TargetSample)
            or entry.result.outcome is not Outcome.OK
        ):
            return extraction
        expected = next(
            self.result.outputs[name]
            for name, output in self.output_names.items()
            if output == into.name
        )
        shown = matching.spaced(entry.result.extracted or "")
        if shown == expected:
            return extraction
        extraction = _extraction_bounds(shown, expected)
        if extraction is None:
            self.issues.append(
                RecordingIssue(
                    Gap.UNREPRESENTABLE_CHECK,
                    self.step,
                    "an output cannot be extracted from its source text",
                )
            )
        return extraction

    def _action(self, entry: ActionTaken, index: int) -> None:
        if self.pending and not (
            entry.action.kind is ActionKind.READ
            and self.pending[1].request.step in self.checked_steps
        ):
            self.issues.append(
                RecordingIssue(Gap.MANUAL_WITHOUT_CONTINUATION, self.step)
            )
        self._next_target(entry)
        self._preset_fields(entry)
        self.step += 1
        self.action_steps.setdefault(entry.step, self.step)
        if entry.action.kind is not ActionKind.READ:
            self.action_steps[entry.step] = self.step
        self.previous = entry
        self.seen = entry.observation or self.seen
        action = entry.action
        if action.kind is not ActionKind.READ:
            self.transition = (entry, self.step)
        route = policy.route_for(self.profile, entry.location) or ""
        target = self._action_target(entry, route)
        into = next(
            (
                ValueSample(RefKind.OUTPUT, name=self.output_names[name])
                for name, control in self.output_targets.items()
                if control == action.target and action.kind is ActionKind.READ
            ),
            None,
        )
        acted_on = _node(action.target, self.seen)
        if acted_on is not None and action.kind in {ActionKind.TYPE, ActionKind.SELECT}:
            self.set_controls.add(acted_on.control)
        operation = entry.operation or Operation.of(action, route, node=acted_on)
        record = self._record(action, route)
        if action.evidence is not None and record is None:
            # The action was bound to one record; a replay must not act unbound.
            self.issues.append(RecordingIssue(Gap.UNREPRESENTABLE_STEP, self.step))
        risk = (
            Risk.RISKY
            if operations.limit(self.profile, operation) is not None
            else self.profile.actions.get(action.kind, Risk.RISKY)
        )
        delivery = settled(entry.result.outcome, entry.result.detail or "")
        dispatched = {
            Interruption.PENDING: Dispatch.NOT_SENT,
            Interruption.PERFORMED: Dispatch.SENT,
        }.get(delivery, Dispatch.UNKNOWN)
        event = Executed(
            self.step,
            action.kind,
            route,
            entry.result.outcome,
            target=target,
            value=self.used_values.get(
                entry.step, _value(_label(action, action.value, entry.observation))
            )
            if action.value is not None
            else None,
            destination=self._destination(action.destination)
            if action.destination
            else None,
            effect=action.effect,
            record=record,
            operation=operation,
            into=self.fact_reads.get(index, into)
            if action.kind is ActionKind.READ
            else into,
            risk=risk,
            flagged=action.flag_risky,
            dispatch=dispatched,
            extraction=self._output_extraction(entry, index, into, target),
            requires=self._dialog_checks()
            if action.kind in DIALOG_ACTIONS
            else (
                (CheckSample(CheckKind.PRESENT, target),)
                if isinstance(target, (TargetSample, VisualSample, TextSample))
                else ()
            )
            + tuple(self.preset_checks),
        )
        self.preset_checks = []
        self.recorder.record(event)
        if (
            action.kind is ActionKind.READ
            and isinstance(action.target, ScreenTarget)
            and event.into is None
        ):
            # The loop read the screen to check a claim; the claim's checks
            # carry that proof, so a replay need not read it as a step.
            self.recorder.record(Unneeded(self.step))
        if index in self.partial:
            if isinstance(target, TargetSample):
                self.recorder.record(
                    Verified(
                        self.step,
                        tuple(
                            # Discovery kept the value only where the text
                            # held it exactly once (rule 11).
                            CheckSample(
                                CheckKind.SHOWS,
                                target,
                                value,
                                Match.CONTAINS,
                                record=record,
                                purpose=purpose,
                                once=True,
                            )
                            for value, purpose in self.partial[index]
                        ),
                    )
                )
            else:
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "a partial value's check has no control to read",
                    )
                )
        if index in self.repeated:
            # The run read a fact it already held, with the same value. The
            # variable has it, so a replay does not need to read it again.
            self.recorder.record(Unneeded(self.step))
        if entry.result.dialog is not None:
            self.recorder.record(
                Verified(
                    self.step,
                    (
                        CheckSample(
                            CheckKind.DIALOG_OPEN,
                            value=_value(entry.result.dialog.message),
                            dialog=entry.result.dialog.kind,
                        ),
                    ),
                )
            )

    def _preset_fields(self, entry: ActionTaken) -> None:
        """Record prefilled input fields before form submission.

        Discovery may submit a default field unchanged, such as a selected
        operator. Replay needs a way to supply another invocation's value.
        Fields already set by discovery have their own action steps.

        Add a setter only if the field matches one input and the contract binds
        that input to the field. Mark the step as introduced because discovery
        did not execute it, as required by rule 4. Equal sample values alone do
        not establish a binding. For an unbound field, add a submission
        precondition requiring the correct input value.
        """
        action = entry.action
        seen = entry.observation
        if seen is None or action.kind not in {ActionKind.CLICK, ActionKind.PRESS_KEY}:
            return
        node = _node(action.target, seen)
        if node is None or not node.form or not (node.submits or node.enter):
            return
        route = policy.route_for(self.profile, entry.location) or ""
        for field in seen.nodes:
            kind = _PRESET.get(field.tag)
            if (
                kind is None
                or field is node
                or field.form != node.form
                or field.secret
                or field.value is None
                or not field.control
                or field.control in self.set_controls
            ):
                continue
            names = [
                name for name, value in self.inputs.items() if value == field.value
            ]
            if len(names) != 1:
                continue
            locator = AxLocator(field.role, field.name, field.frame, field.scope)
            if _node(locator, seen) is not field:
                continue
            sample = self._sample(locator, route)
            if not isinstance(sample, TargetSample):
                continue
            bound = self.contract is not None and any(
                binding.input == names[0]
                and matching.spaced(binding.field) in _field_labels(field)
                for binding in self.contract.bindings
            )
            if not bound:
                self.preset_checks.append(
                    CheckSample(
                        CheckKind.SHOWS,
                        sample,
                        ValueSample(RefKind.INPUT, name=names[0]),
                        Match.EQUALS,
                    )
                )
                continue
            self.step += 1
            self.set_controls.add(field.control)
            initialized = Action(kind, locator, field.value)
            operation = Operation.of(initialized, route, node=field)
            binding = operations.identify(self.profile, initialized, route, node=field)
            if binding is not None:
                operation = dataclasses.replace(
                    operation, binding=binding.name, business=binding.business
                )
            self.recorder.record(
                Executed(
                    self.step,
                    kind,
                    route,
                    Outcome.OK,
                    target=sample,
                    value=ValueSample(RefKind.INPUT, name=names[0]),
                    requires=(CheckSample(CheckKind.PRESENT, sample),),
                    introduced=f"binding:{names[0]}",
                    operation=operation,
                )
            )

    def _action_target(self, entry: ActionTaken, route: str) -> TargetEvidence | None:
        """Build a reusable target for an action."""
        action = entry.action
        if action.mouse != MouseInput():
            return ScreenSample(route)
        if not isinstance(action.target, ScreenTarget):
            return self._read_source(action, route) or self._sample(
                action.target, route
            )
        control = self._screen_control(entry, route)
        if control is not None:
            return control
        target = self._visual_target(entry, route)
        if isinstance(target, ScreenSample):
            target = self._painted(entry, route) or target
            if action.target.point is None and action.kind in FOCUS_ACTIONS:
                # Keys went to whatever held the keyboard focus. A painted
                # click just before them, which chose that field, stays in
                # the capability (rule 4); the typing's checks prove both.
                return FocusSample(route, ())
        return target

    def _screen_control(self, entry: ActionTaken, route: str) -> TargetSample | None:
        target = entry.action.target
        seen = entry.observation
        operation = entry.operation
        if (
            entry.action.kind is not ActionKind.CLICK
            or not isinstance(target, ScreenTarget)
            or target.point is None
            or operation is None
            or not operation.control
            or seen is None
            or seen.mode is not ObservationMode.STRUCTURED
            or not complete(seen)
        ):
            return None
        capture = self.captures.get(target.capture_id)
        if capture is None or capture.page_state.differs_from(seen.page_state):
            return None
        nodes = [
            node
            for node in seen.nodes
            if node.control == operation.control and node.visible and not node.secret
        ]
        if len(nodes) != 1:
            return None
        node = nodes[0]
        locator = AxLocator(node.role, node.name, node.frame, node.scope)
        if _node(locator, seen) != node:
            return None
        sample = self._sample(locator, route)
        return sample if isinstance(sample, TargetSample) else None

    def _painted(self, entry: ActionTaken, route: str) -> TextSample | None:
        """Record a screenshot action using text recognized in its capture.

        A click uses text under its point or a nearby label above it. Typing uses
        a label because field contents may be record data. A read uses the label
        before the extracted value. Each line must be unique. Live screen
        coordinates are never saved.
        """
        target = entry.action.target
        kind = entry.action.kind
        if (
            self.reader is None
            or not isinstance(target, ScreenTarget)
            or target.point is None
            or kind not in TEXT_ACTIONS
            or entry.action.mouse != MouseInput()
        ):
            return None
        observed = self.captures.get(target.capture_id)
        if observed is None or observed.image is None:
            return None
        lines = self.reader.lines(observed.image)
        x, y = target.point.x, target.point.y
        if kind is ActionKind.CLICK:
            try:
                choices = reading.radio_labels(observed.image, self.reader, lines)
            except reading.RecognitionError:
                return None
            chosen = [choice for choice in choices if choice.holds(x, y)]
            if len(chosen) == 1:
                label = chosen[0].label
                if _alone(tuple(choice.label for choice in choices), label):
                    return TextSample(route, label.text, radio=True)
                return None
        if kind is ActionKind.READ:
            line = reading.at(lines, x, y)
            value = self._read_value(entry)
            anchor = _label_before(line.text, value) if line and value else None
            if anchor is None or len(reading.anchored(lines, anchor)) != 1:
                return None
            return TextSample(route, anchor, label=True)
        line = reading.at(lines, x, y)
        if line is not None and kind is not ActionKind.TYPE and _alone(lines, line):
            return TextSample(route, line.text)
        label = _label_above(lines, x, y)
        if label is None or not _alone(lines, label):
            return None
        return TextSample(
            route, label.text, dx=round(x - label.left), dy=round(y - label.top)
        )

    def _read_value(self, entry: ActionTaken) -> str | None:
        """Return the output value produced by a screenshot read, if any."""
        for name, control in self.output_targets.items():
            if control == entry.action.target:
                return self.result.outputs.get(name)
        return None

    def _visual_target(
        self, entry: ActionTaken, route: str
    ) -> VisualSample | ScreenSample:
        target = entry.action.target
        image = self.visual_templates.get(entry.step)
        if (
            not isinstance(target, ScreenTarget)
            or target.point is None
            or image is None
            or entry.action.kind is not ActionKind.CLICK
            or entry.action.mouse != MouseInput()
        ):
            return ScreenSample(route)
        observed = self.captures.get(target.capture_id)
        if observed is None or observed.image is None:
            return ScreenSample(route)
        match = visual.locate(image, observed.image)
        box = match.box
        if match.result is not visual.MatchResult.FOUND or box is None:
            return ScreenSample(route)
        point = target.point
        if not (
            box.left <= point.x < box.left + box.width
            and box.top <= point.y < box.top + box.height
        ):
            return ScreenSample(route)
        sample = VisualSample(route, image, permit=StoragePermit.OPERATOR_APPROVED)
        if self.step == 1:
            self.recorder.record(Verified(0, (CheckSample(CheckKind.PRESENT, sample),)))
        return sample

    def _painted_checkpoint(self, seen: Observation) -> None:
        """Verify a screenshot click through newly visible text.

        The line must be unique, absent from the preceding capture, and free of
        input values. Prefer confirmed website labels over unconfirmed text.
        """
        previous = self.previous
        if (
            self.reader is None
            or previous is None
            or seen.image is None
            or previous.action.kind not in {ActionKind.CLICK, ActionKind.PRESS_KEY}
            or not isinstance(previous.action.target, ScreenTarget)
            or self.step in self.painted_checked
        ):
            return
        before = self.captures.get(previous.action.target.capture_id)
        if before is None or before.image is None:
            return
        self.painted_checked.add(self.step)
        old = {reading.squash(line.text) for line in self.reader.lines(before.image)}
        lines = self.reader.lines(seen.image)
        try:
            choices = reading.radio_labels(seen.image, self.reader, lines)
        except reading.RecognitionError:
            return
        known = frozenset(self.safe_text)
        for line in sorted(
            lines,
            key=lambda item: (item.text not in known, item.top, item.left),
        ):
            if (
                reading.squash(line.text) in old
                or not _alone(lines, line)
                or any(choice.holds(*line.centre) for choice in choices)
                or any(value and value in line.text for value in self.inputs.values())
            ):
                continue
            route = policy.route_for(self.profile, seen.location) or ""
            proof = CheckSample(CheckKind.PRESENT, TextSample(route, line.text))
            self.recorder.record(Verified(self.step, (proof,)))
            return

    def _visual_checkpoint(self, seen: Observation) -> None:
        if self.previous is None or seen.image is None:
            return
        image = self.visual_results.get(self.previous.step)
        if (
            image is None
            or visual.locate(image, seen.image).result is not visual.MatchResult.FOUND
        ):
            return
        route = policy.route_for(self.profile, seen.location) or ""
        sample = VisualSample(route, image, permit=StoragePermit.OPERATOR_APPROVED)
        self.recorder.record(
            Verified(self.step, (CheckSample(CheckKind.PRESENT, sample),))
        )

    def _verified(self, entry: CheckedAction) -> None:
        self.seen = entry.observation or self.seen
        checks = tuple(self._check_sample(check) for check in entry.checks)
        if any(check is None for check in checks):
            self.issues.append(
                RecordingIssue(
                    Gap.UNREPRESENTABLE_CHECK,
                    self.step,
                    "a step's check has no form that can be saved",
                )
            )
            return
        if self.pending and self.pending[1].request.step == entry.step:
            self._complete_human(tuple(check for check in checks if check is not None))
            return
        if entry.step not in self.action_steps:
            return
        self.recorder.record(
            Verified(
                self.action_steps[entry.step],
                tuple(check for check in checks if check is not None),
            )
        )

    def _check_sample(self, check: ResultCheck) -> CheckSample | None:
        if isinstance(check.target, ScreenTarget):
            return self._painted_check(check, _value(check.expected))
        if not isinstance(check.target, (AxLocator, DomLocator)) or self.seen is None:
            return None
        route = policy.route_for(self.profile, self.seen.location) or ""
        action = Action(ActionKind.READ, check.target, evidence=check.record)
        target = self._read_source(action, route) or self._sample(check.target, route)
        if not isinstance(target, TargetSample):
            return None
        record = self._record(action, route)
        if check.record is not None and record is None:
            return None
        match, purpose = _saved(check)
        return CheckSample(
            CheckKind.SHOWS,
            target,
            _value(check.expected),
            match,
            record=record,
            purpose=purpose,
        )

    def _painted_check(
        self, check: ResultCheck, value: ValueSample
    ) -> CheckSample | None:
        """Record a screenshot check using the value after its label."""
        target = check.target
        if self.reader is None or not isinstance(target, ScreenTarget):
            return None
        observed = self.captures.get(target.capture_id)
        if target.point is None or observed is None or observed.image is None:
            return None
        lines = self.reader.lines(observed.image)
        line = reading.at(lines, target.point.x, target.point.y)
        if line is None:
            return None
        route = policy.route_for(self.profile, observed.location) or ""
        anchor = _label_before(line.text, check.expected)
        if anchor is None and _whole_line(line.text, check.expected):
            # A check of a whole answer that ends with an input, such as a
            # message naming the searched record, is saved as the words
            # before the input and the input, so it holds for any record.
            anchor, value = self._before_input(line.text, value)
        match, purpose = _saved(check)
        if anchor is not None and len(reading.anchored(lines, anchor)) == 1:
            return CheckSample(
                CheckKind.SHOWS,
                TextSample(route, anchor, label=True),
                value,
                match,
                purpose=purpose,
            )
        if check.match.value == Match.CONTAINS.value and _alone(lines, line):
            # A value in the middle of a line, such as the operator in a
            # signed-on banner, is found by the line that holds it.
            return CheckSample(
                CheckKind.SHOWS,
                TextSample(route, line.text),
                value,
                Match.CONTAINS,
                purpose=purpose,
            )
        return None

    def _before_input(
        self, text: str, value: ValueSample
    ) -> tuple[str | None, ValueSample]:
        """Return the prefix and unique input value at the end of ``text``."""
        found = [
            (anchor, ValueSample(RefKind.INPUT, name=name))
            for name, held in self.inputs.items()
            if held and (anchor := _label_before(text, held)) is not None
        ]
        return found[0] if len(found) == 1 else (None, value)

    def _next_target(self, entry: ActionTaken) -> None:
        if entry.observation is None:
            return
        # A partial look can show that a control is there, not that it is the
        # only one or that it was absent before. It proves a step only when
        # the page's address changed, and never marks one unneeded.
        whole = complete(entry.observation)
        if not whole and self.step == 0:
            return
        current = _node(entry.action.target, entry.observation)
        if current is None:
            return
        route = policy.route_for(self.profile, entry.location) or ""
        seen = self.seen
        self.seen = entry.observation
        target = self._read_source(entry.action, route) or self._sample(
            entry.action.target, route
        )
        self.seen = seen
        if not isinstance(target, TargetSample):
            return
        if self.step == 0:
            self.recorder.record(
                Verified(
                    0,
                    (
                        CheckSample(CheckKind.AT_ROUTE, route=route),
                        CheckSample(CheckKind.PRESENT, target),
                    ),
                )
            )
            return
        if self.transition is None:
            return
        previous, transition_step = self.transition
        before = previous.observation
        moved = before is not None and before.location != entry.location
        if not whole or before is None or not complete(before):
            if not moved:
                return
            changed = True
        else:
            old = _node(entry.action.target, before)
            changed = (
                moved
                or old is None
                or (current is not None and current.enabled and not old.enabled)
            )
        if previous.action.kind in {
            ActionKind.CLICK,
            ActionKind.PRESS_KEY,
            ActionKind.NAVIGATE,
        } and (changed or self._safe_continuation(previous, route)):
            self.recorder.record(
                Verified(
                    transition_step,
                    (
                        CheckSample(CheckKind.AT_ROUTE, route=route),
                        CheckSample(CheckKind.PRESENT, target),
                    ),
                )
            )

    def _safe_continuation(self, entry: ActionTaken, route: str) -> bool:
        verdict = policy.evaluate(entry.action, self.profile, entry.location)
        operation = entry.operation or Operation.of(entry.action, route)
        return (
            isinstance(verdict, policy.Allowed)
            and verdict.risk is Risk.SAFE
            and not entry.action.flag_risky
            and entry.result.outcome is Outcome.OK
            and operations.limit(self.profile, operation) is None
            and policy.learned_limit(self.result.restrictions, operation) is None
        )

    def _dialog_checks(self) -> tuple[CheckSample, ...]:
        dialog = self.seen.dialog if self.seen else None
        if dialog is None:
            return ()
        return (
            CheckSample(
                CheckKind.DIALOG_OPEN, value=_value(dialog.message), dialog=dialog.kind
            ),
        )

    def _destination(self, location: str) -> DestinationSample:
        permitted_route = policy.route_for(self.profile, location) or ""
        permitted = urlsplit(permitted_route).path.split("/")
        url = urlsplit(location)
        segments: list[str] = []
        params: list[tuple[str, ValueSample]] = []
        for index, part in enumerate((url.path or "/").split("/")):
            declared = permitted[index] if index < len(permitted) else ""
            if part == declared and "*" not in part:
                segments.append(part)
            else:
                name = declared[1:] if declared.startswith(":") else f"path_{index}"
                segments.append(":" + name)
                params.append((name, _value(unquote(part))))
        query = tuple(
            (name, _value(value))
            for name, value in parse_qsl(url.query, keep_blank_values=True)
        )
        return DestinationSample(
            "/".join(segments),
            tuple(params),
            query,
            _value(unquote(url.fragment)) if url.fragment else None,
            http_origin(location),
        )

    def _checkpoint(self, seen: Observation) -> tuple[CheckSample, ...]:
        route = policy.route_for(self.profile, seen.location)
        if not route or not complete(seen):
            return ()
        if self.step and self.pending is None:
            return (
                self._field_checkpoint(seen, route)
                or self._choice_checkpoint(seen, route)
                or self._input_checkpoint(seen, route)
                or self._appeared_checkpoint(seen, route)
            )
        prior = self.previous.observation if self.previous else None
        moved = (
            prior is not None
            and policy.route_for(self.profile, prior.location) != route
        )
        # The first screen shows the same text whichever record a run is for,
        # such as the member an application opens with, so a comparison with
        # a second record cannot tell its data from the website's. Its check
        # uses only text already confirmed.
        known = self.safe_text if self.step else frozenset(self.safe_text)
        checks: list[CheckSample] = []
        for node in seen.nodes:
            if (
                node.secret
                or (node.name not in known and node.name not in self.inputs.values())
                or node.value is not None
                or (
                    node.role not in {"heading", "status", "alert"}
                    and node.name not in self.inputs.values()
                )
            ):
                continue
            same = prior is not None and any(
                old.identity == node.identity for old in prior.nodes
            )
            if self.pending is None and self.step and not moved and same:
                continue
            if any(
                value in node.name and value != node.name
                for value in self.inputs.values()
            ):
                continue
            target = self._sample(
                AxLocator(node.role, node.name, frame=node.frame), route
            )
            if isinstance(target, TargetSample):
                checks = [
                    CheckSample(CheckKind.AT_ROUTE, route=route),
                    CheckSample(CheckKind.SHOWS, target, _value(node.name)),
                ]
                break
        if self.pending is not None:
            for node in seen.nodes:
                if (
                    node.value is None
                    and not node.secret
                    and node.name in self.inputs.values()
                ):
                    target = self._sample(
                        AxLocator(node.role, node.name, frame=node.frame), route
                    )
                    if isinstance(target, TargetSample):
                        checks.append(
                            CheckSample(CheckKind.SHOWS, target, _value(node.name))
                        )
        return tuple(checks)

    def _field_checkpoint(
        self, seen: Observation, route: str
    ) -> tuple[CheckSample, ...]:
        if self.previous is None:
            return ()
        action = self.previous.action
        if action.kind not in {ActionKind.TYPE, ActionKind.SELECT} or not isinstance(
            action.value, str
        ):
            return ()
        node = _node(action.target, seen)
        target = self._sample(action.target, route)
        if (
            node is None
            or node.secret
            or node.value != action.value
            or not isinstance(target, TargetSample)
        ):
            return ()
        return (
            CheckSample(
                CheckKind.SHOWS,
                target,
                self.used_values.get(self.previous.step, _value(action.value)),
            ),
        )

    def _choice_checkpoint(
        self, seen: Observation, route: str
    ) -> tuple[CheckSample, ...]:
        previous = self.previous
        if (
            previous is None
            or previous.action.kind is not ActionKind.CLICK
            or not complete(previous.observation)
            or previous.observation is None
            or previous.observation.location != seen.location
        ):
            return ()
        option = _node(previous.action.target, previous.observation)
        if option is None or option.role != "option" or option.secret or option.scope:
            return ()
        held = [
            value for value in self.inputs.values() if value and value in option.name
        ]
        if len(held) > 1:
            return ()
        value = held[0] if held else option.name
        if held and len(matching.positions(option.name, value)) != 1:
            return ()

        role, frame = option.role, option.frame

        def matches(observation: Observation) -> int:
            return sum(
                node.visible
                and node.role == role
                and node.frame == frame
                and (
                    len(matching.positions(node.name, value)) == 1
                    if held
                    else node.name == value
                )
                for node in observation.nodes
            )

        if matches(previous.observation) != 1 or matches(seen) != 0:
            return ()
        target = TargetSample(
            route,
            LocatorForm.ACCESSIBILITY,
            role=option.role,
            name=option.name,
            frame=option.frame,
        )
        return (CheckSample(CheckKind.ABSENT, target),)

    def _input_checkpoint(
        self, seen: Observation, route: str
    ) -> tuple[CheckSample, ...]:
        """Verify a click through a newly displayed input value.

        After a click or keypress, a control may display an input as whole words
        that no control of its role displayed before. For example, an operator
        header may show ``Sam Lee (U-7)`` after selection. Match the input alone
        to preserve a check that changed from false to true. Use only the next
        observation, and consume that evidence once.
        """
        previous = self.previous
        if (
            previous is None
            or previous.action.kind not in {ActionKind.CLICK, ActionKind.PRESS_KEY}
            or previous.observation is None
            or not complete(previous.observation)
            or self.step in self.input_checked
        ):
            return ()
        self.input_checked.add(self.step)
        for node in seen.nodes:
            value = _held_input(node, self.inputs.values())
            if value is None or _holding(previous.observation, node, value):
                continue
            if len(_holding(seen, node, value)) != 1:
                continue
            target = self._sample(
                AxLocator(node.role, node.name, frame=node.frame), route
            )
            if isinstance(target, TargetSample):
                return (
                    CheckSample(CheckKind.SHOWS, target, _value(value), Match.CONTAINS),
                )
        return ()

    def _appeared_checkpoint(
        self, seen: Observation, route: str
    ) -> tuple[CheckSample, ...]:
        """Verify a click through a newly visible interface control.

        Use a self-labeled menu item, button, tab, or dialog outside record rows.
        Its appearance verifies the click without saving record data.
        """
        previous = self.previous
        if (
            previous is None
            or previous.action.kind not in {ActionKind.CLICK, ActionKind.PRESS_KEY}
            or previous.observation is None
            or not complete(previous.observation)
            or self.step in self.appeared_checked
        ):
            return ()
        self.appeared_checked.add(self.step)
        before = previous.observation
        for node in seen.nodes:
            if (
                not node.visible
                or node.secret
                or node.row is not None
                or node.role not in APPEARING_ROLES
                or not node.name
                # A name holding an input is the input proof's to judge.
                or any(value and value in node.name for value in self.inputs.values())
            ):
                continue
            locator = AxLocator(node.role, node.name, frame=node.frame)
            if any(names_node(locator, old) for old in before.nodes):
                continue
            if _node(locator, seen) is not node:
                continue
            target = self._sample(locator, route)
            if isinstance(target, TargetSample):
                return (CheckSample(CheckKind.PRESENT, target),)
        return ()

    def _look(self, seen: Observation) -> None:
        if seen.mode is ObservationMode.VISUAL:
            self._visual_checkpoint(seen)
            self._painted_checkpoint(seen)
            return
        self.seen = seen
        checks = self._checkpoint(seen)
        if self.pending:
            if self.pending[1].request.step not in self.checked_steps:
                self._complete_human(checks)
        elif checks:
            self.recorder.record(Verified(self.step, checks))

    def _complete_human(self, checks: tuple[CheckSample, ...]) -> None:
        if self.pending is None:
            return
        step, hand = self.pending
        if checks and self._automated(step, hand, checks):
            self.pending = None
            return
        if checks:
            action = hand.request.action
            reason = {
                Trigger.RECORD_EVIDENCE_REQUIRED: HelpReason.RECORD_EVIDENCE,
                Trigger.AUTHENTICATION_REQUIRED: HelpReason.AUTHENTICATION,
            }.get(hand.request.trigger, HelpReason.MANUAL_STEP)
            self.recorder.record(
                HumanSegment(
                    step,
                    hand.request.route,
                    reason,
                    checks,
                    sum(len(segment.gaps) for segment in hand.segments),
                    performs=action.kind if action else None,
                    effect=action.effect if action else None,
                    permissions=_manual_kinds(hand.segments),
                    replaces=self.replaces,
                    forbidden=any(
                        item.requirement is Requirement.FORBIDDEN
                        for segment in hand.segments
                        for item in segment.steps
                    ),
                )
            )
            self.pending = None

    def _automated(
        self, step: int, hand: Handed, checks: tuple[CheckSample, ...]
    ) -> bool:
        """Record a classified manual click as an automated step.

        The loop must have labeled the click. The intervention must not have
        interrupted uncertain delivery, and its control must match exactly once
        in the starting observation. Checks after control returns verify the
        click just as they would an automated action.
        """
        label = self.labels.get(hand.answer.intervention)
        event = single_click(hand.segments)
        if (
            label is None
            or event is None
            or event.target is None
            or self.replaces is not None
            or _node(event.target, self.pending_seen) is None
        ):
            return False
        route = policy.route_for(self.profile, event.location) or ""
        seen, self.seen = self.seen, self.pending_seen
        target = self._sample(event.target, route)
        self.seen = seen
        if not isinstance(target, TargetSample):
            return False
        self.recorder.record(
            Executed(
                step,
                ActionKind.CLICK,
                route,
                Outcome.OK,
                target=target,
                effect=label.effect,
                operation=event.operation,
                risk=Risk.RISKY if label.careful else Risk.SAFE,
                dispatch=Dispatch.SENT,
            )
        )
        self.recorder.record(Verified(step, checks))
        return True

    def _human(self, hand: Handed) -> None:
        if self.pending:
            self.issues.append(
                RecordingIssue(Gap.MANUAL_WITHOUT_CONTINUATION, self.step)
            )
        self.replaces = (
            self.step
            if self.previous is not None
            and self.previous.step == hand.request.step
            and self.previous.result.outcome
            in {Outcome.UNCERTAIN, Outcome.SURFACE_ERROR}
            and hand.request.trigger is Trigger.DELIVERY_UNCERTAIN
            else None
        )
        self.step += 1
        self.pending = (self.step, hand)
        self.pending_seen = self.seen
        self.previous = None
        self.transition = None

    def _finish(self) -> None:
        checks: list[CheckSample] = []
        for checked in self.result.checks:
            check = checked.check
            if isinstance(check.target, FactRef) and checked.passed:
                if check.target.key in self.partial_keys:
                    # Checked where the fact was read; see _check_at_reading.
                    continue
                reference = self.fact_refs.get(check.target.key)
                if reference is None:
                    self.issues.append(
                        RecordingIssue(
                            Gap.UNREPRESENTABLE_CHECK,
                            self.step,
                            "the finish cites a fact with no saved reference",
                        )
                    )
                elif check.kind is ResultKind.RESULT:
                    checks.append(CheckSample(CheckKind.BOUND, value=reference))
                else:
                    checks.append(
                        CheckSample(
                            CheckKind.VALUE_IS,
                            value=reference,
                            expected=_value(check.expected),
                            match=_saved(check)[0],
                            purpose=_saved(check)[1],
                        )
                    )
                continue
            if checked.passed and isinstance(check.target, ScreenTarget):
                painted = self._painted_check(
                    check,
                    ValueSample(RefKind.OUTPUT, name=self.output_names[check.output])
                    if check.kind is ResultKind.RESULT
                    and check.output in self.output_names
                    else _value(check.expected),
                )
                if painted is None:
                    self.issues.append(
                        RecordingIssue(
                            Gap.UNREPRESENTABLE_CHECK,
                            self.step,
                            "a painted finish check cannot be saved",
                        )
                    )
                else:
                    checks.append(painted)
                continue
            if not checked.passed or not isinstance(
                check.target, (AxLocator, DomLocator)
            ):
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "a finish check did not pass, or has no structural control",
                    )
                )
                continue
            route = (
                policy.route_for(self.profile, self.seen.location)
                if self.seen
                else None
            )
            action = Action(ActionKind.READ, check.target, evidence=check.record)
            target = self._read_source(action, route or "") or self._sample(
                check.target, route or ""
            )
            if not isinstance(target, TargetSample):
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "a finish check's control cannot be saved",
                    )
                )
                continue
            value = (
                ValueSample(RefKind.OUTPUT, name=self.output_names[check.output])
                if check.kind is ResultKind.RESULT
                else _value(check.expected)
            )
            record = self._record(action, route or "")
            if check.record is not None and record is None:
                # A check proved for one record must not replay for any record.
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "a record-bound finish check has no saved record",
                    )
                )
                continue
            match, purpose = _saved(check)
            checks.append(
                CheckSample(
                    CheckKind.SHOWS,
                    target,
                    value,
                    match,
                    record=record,
                    purpose=purpose,
                )
            )
        verifier = (
            Verifier(self.result.verification.value)
            if self.result.verification
            else Verifier.PERSON
        )
        last = self.previous
        if (
            checks
            and last is not None
            and last.action.kind in {ActionKind.CLICK, ActionKind.PRESS_KEY}
            and not self.recorder.proved(self.step)
        ):
            # The final checks were read on the live page right after the last
            # click, so they prove it when nothing else did, such as a search
            # that ends on a "nothing matched" message.
            self.recorder.record(Verified(self.step, tuple(checks)))
        confirmed = verifier is Verifier.PERSON and all(
            item.passed for item in self.result.checks
        )
        self.recorder.record(
            Finished(
                verifier,
                tuple(checks),
                self.result.outcome,
                confirmed,
                self._confirm_inputs() if confirmed else (),
            )
        )

    def _confirm_inputs(self) -> tuple[Ref, ...]:
        missing = uncovered_requirements(
            self.result.task,
            ((item.check, item.bound) for item in self.result.checks if item.passed),
        )
        references: list[Ref] = []
        for requirement in missing:
            if requirement.of and (
                len(self.result.task.records) != 1
                or self.result.task.records[0].name != requirement.of
            ):
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "a requirement belongs to another record",
                    )
                )
                continue
            names = _named_inputs(requirement.expected, self.inputs)
            if len(names) != 1:
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "a requirement matches no single input",
                    )
                )
                continue
            reference = ref(RefKind.INPUT, names[0])
            if reference in references:
                self.issues.append(
                    RecordingIssue(
                        Gap.UNREPRESENTABLE_CHECK,
                        self.step,
                        "two requirements name the same input",
                    )
                )
                continue
            references.append(reference)
        return tuple(references)


def _manual_kinds(segments: tuple[ManualSegment, ...]) -> tuple[ActionKind, ...]:
    """Carry the grants needed for manual work into each future invocation."""
    kinds: set[ActionKind] = set()
    mapping = {
        ManualKind.CLICK: ActionKind.CLICK,
        ManualKind.KEY: ActionKind.PRESS_KEY,
        ManualKind.EDIT: ActionKind.TYPE,
        ManualKind.SELECT: ActionKind.SELECT,
        ManualKind.SCROLL: ActionKind.SCROLL,
        ManualKind.NAVIGATION: ActionKind.NAVIGATE,
    }
    for segment in segments:
        for step in segment.steps:
            event = step.event
            if event.kind in mapping:
                kinds.add(mapping[event.kind])
            elif event.kind is ManualKind.DIALOG:
                if event.detail in {Detail.ACCEPTED, Detail.NONE}:
                    kinds.add(ActionKind.ACCEPT_DIALOG)
                if event.detail in {Detail.DISMISSED, Detail.NONE}:
                    kinds.add(ActionKind.DISMISS_DIALOG)
    return tuple(sorted(kinds))


def _named_inputs(expected: str, inputs: Mapping[str, str]) -> list[str]:
    """Return input names whose values appear in a requirement's expected text.

    The expected text may contain words around a value, such as "post
    statements" around ``post``. Prefer a value matching the whole text.
    Otherwise, find whole-word values without regard to case. Callers use
    the result only if exactly one input matches.

    Examples
    --------
    >>> inputs = {"delivery": "post", "nickname": "Blue jar"}
    >>> _named_inputs("post statements", inputs)
    ['delivery']
    >>> _named_inputs("Blue jar", inputs)
    ['nickname']
    >>> _named_inputs("electronic", inputs)
    []
    """
    exact = [name for name, value in inputs.items() if value == expected]
    if exact:
        return exact
    return [
        name
        for name, value in inputs.items()
        if value and matching.spelled(expected, value) is not None
    ]
