"""The live vocabulary of discovery actions and observations.

Accessibility and DOM locators identify controls structurally. ``VisualAnchor``
re-finds a canvas crop. ``ScreenTarget`` addresses pixels or keyboard focus in
one current capture, so ordinary pages and custom widgets use the same input
path. Every action still carries its type for the policy gate.

Screen coordinates and canvas crops stay in session memory. They are not part
of the replay contract. Recording converts live controls into reusable targets
and result checks. ``Observation`` is also working state, so adding a
perception channel does not change the sanitized journal.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from enum import StrEnum

from computeruse import matching
from computeruse.profile import (
    FOCUSABLE_ACTIONS,
    TARGETED_ACTIONS,
    ActionKind,
    ObservationMode,
    is_effect_name,
)

MAX_OCCURRENCE = 20
"""Larger occurrence values would guess among repeated labels."""


class ScopeKind(StrEnum):
    """The kind of container that makes a repeated label unique."""

    ROW = "row"
    REGION = "region"
    FORM = "form"
    TABLE = "table"


@dataclasses.dataclass(frozen=True, slots=True)
class Scope:
    """The container a target is looked for inside.

    A back-office table repeats "View" once per row, so the row is what makes
    one of them addressable. ``name`` is the accessible name of a region or
    form, or the text content of a row, and it must identify exactly one
    container or the target is refused as ambiguous. ``column`` is zero, or
    the position, counted from one, of the row cell that must show ``name``.
    """

    kind: ScopeKind
    name: str
    column: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scope must name the container it selects")


@dataclasses.dataclass(frozen=True, slots=True)
class AxLocator:
    """A structural control locator that a later run can resolve.

    Parameters
    ----------
    role
        The control's accessibility role, such as ``"button"``.
    name
        Its accessible name: the label a human operator reads. May be empty,
        because non-semantic legacy markup does not always supply one.
    frame
        Names of the frames containing the control, outermost first. Legacy
        back-office screens are frequently framesets, so frame identity is
        part of a control's address rather than ambient session state.
    scope
        The row, region, form, or table the control is looked for inside.
        This is how a repeated label is made unique without counting matches.
    occurrence
        Which match to take when role, name, and scope still do not identify
        one control within the frame. Zero-based.

    Examples
    --------
    >>> AxLocator("button", "Search").occurrence
    0
    >>> AxLocator("button", "View", scope=Scope(ScopeKind.ROW, "12345")).scope.kind
    <ScopeKind.ROW: 'row'>
    >>> AxLocator("", "Search")
    Traceback (most recent call last):
        ...
    ValueError: locator must name a role
    """

    role: str
    name: str
    frame: tuple[str, ...] = ()
    scope: Scope | None = None
    occurrence: int = 0

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("locator must name a role")
        _check_occurrence(self.occurrence)


class DomAttribute(StrEnum):
    """The closed set of element attributes available for exact DOM matches.

    The list is closed on purpose. An open attribute name, or a pattern, would
    let a model send a selector this executor cannot reason about; these are
    the attributes a legacy page actually carries and a reviewer can read.
    """

    ID = "id"
    NAME = "name"
    TYPE = "type"
    TITLE = "title"
    PLACEHOLDER = "placeholder"
    VALUE = "value"
    DATA_VALUE = "data-value"
    DATA_TESTID = "data-testid"
    HREF = "href"
    ARIA_LABEL = "aria-label"
    CSS_CLASS = "css-class"
    SLOT = "slot"
    TEXT = "text"


def positional_slot(value: str) -> bool:
    """Report whether a slot names a table column by position, not by text.

    The collector names a cell by its column's header, as ``td|Status``, and
    by its position, as ``td#2``, when the column has no header.

    Examples
    --------
    >>> positional_slot("td#2"), positional_slot("td|Status"), positional_slot("td#")
    (True, False, False)
    """
    tag, mark, index = value.partition("#")
    return bool(mark) and tag.isalpha() and index.isdigit()


STRUCTURAL_ROLES = frozenset(
    [
        "alert",
        "alertdialog",
        "application",
        "article",
        "banner",
        "blockquote",
        "button",
        "canvas",
        "caption",
        "cell",
        "checkbox",
        "code",
        "columnheader",
        "combobox",
        "complementary",
        "contentinfo",
        "definition",
        "deletion",
        "dialog",
        "directory",
        "document",
        "emphasis",
        "feed",
        "figure",
        "form",
        "generic",
        "grid",
        "gridcell",
        "group",
        "heading",
        "img",
        "insertion",
        "label",
        "legend",
        "link",
        "list",
        "listbox",
        "listitem",
        "log",
        "main",
        "marquee",
        "math",
        "menu",
        "menubar",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "meter",
        "navigation",
        "none",
        "note",
        "option",
        "paragraph",
        "presentation",
        "progressbar",
        "radio",
        "radiogroup",
        "region",
        "row",
        "rowgroup",
        "rowheader",
        "scrollbar",
        "search",
        "searchbox",
        "separator",
        "slider",
        "spinbutton",
        "status",
        "strong",
        "subscript",
        "superscript",
        "switch",
        "tab",
        "table",
        "tablist",
        "tabpanel",
        "term",
        "text",
        "textbox",
        "time",
        "timer",
        "toolbar",
        "tooltip",
        "tree",
        "treegrid",
        "treeitem",
    ]
)
"""Closed structural categories that carry no page-specific text."""


DOM_TAGS = frozenset(
    {
        "a",
        "button",
        "b",
        "canvas",
        "dd",
        "div",
        "dt",
        "input",
        "label",
        "li",
        "option",
        "output",
        "p",
        "select",
        "span",
        "strong",
        "small",
        "td",
        "textarea",
        "th",
    }
)
"""Tags a DOM locator may name. Anything else is not a control a run operates."""


@dataclasses.dataclass(frozen=True, slots=True)
class DomLocator:
    """One element named by tag plus one exact attribute value.

    This exists for the markup that has no accessible name at all: a ``span``
    carrying a ``data`` attribute, a table cell identified only by its text.
    Matching is equality, never a pattern, so what the operator reads in a
    recorded step is what the executor looks for. ``css-class`` names one
    class the element carries, among any others, because that is how a class
    list is read; ``text`` is the accessible name the observation reported.

    Examples
    --------
    >>> DomLocator("span", DomAttribute.CSS_CLASS, "posting-lock").tag
    'span'
    >>> DomLocator("iframe", DomAttribute.ID, "ledger")
    Traceback (most recent call last):
        ...
    ValueError: dom locator names an unsupported tag
    >>> DomLocator("span", DomAttribute.CSS_CLASS, "posting-lock locked")
    Traceback (most recent call last):
        ...
    ValueError: a css-class locator names exactly one class
    """

    tag: str
    attribute: DomAttribute
    value: str
    frame: tuple[str, ...] = ()
    scope: Scope | None = None
    occurrence: int = 0

    def __post_init__(self) -> None:
        if self.tag not in DOM_TAGS:
            raise ValueError("dom locator names an unsupported tag")
        if not self.value:
            raise ValueError("dom locator must carry a value to match")
        if self.attribute is DomAttribute.CSS_CLASS and len(self.value.split()) != 1:
            raise ValueError("a css-class locator names exactly one class")
        _check_occurrence(self.occurrence)


@dataclasses.dataclass(frozen=True, slots=True)
class VisualAnchor:
    """A pixel-only control identified by a crop held in memory.

    A canvas control has no role, no name, and no element to address. The
    visual observation that found it kept the crop in this process, and the
    resolver matches that crop against the live screenshot again immediately
    before acting, so the control may move between the observation and the
    click.

    ``anchor_id`` is a session token, not a recorded contract. The crop lives
    in memory and is never written to disk, so an action targeted this way
    cannot be replayed from an artifact yet. Persisting visual assets safely
    is a separate piece of work, and until it exists a visual anchor is a
    discovery-time affordance only.

    Examples
    --------
    >>> VisualAnchor("anchor-1").frame
    ()
    """

    anchor_id: str
    frame: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("visual anchor must name a captured region")


@dataclasses.dataclass(frozen=True, slots=True)
class Point:
    """A position in the pixels of one captured screen."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("screen coordinates must be finite")


@dataclasses.dataclass(frozen=True, slots=True)
class ScreenTarget:
    """A live input target, bound to a capture rather than a saved locator.

    A missing point addresses the keyboard focus captured with the screen.
    Coordinates stay in working memory and are not a replay contract.
    """

    capture_id: str
    point: Point | None = None

    def __post_init__(self) -> None:
        if not self.capture_id:
            raise ValueError("screen input must name a capture")


class MouseButton(StrEnum):
    """The three mouse buttons supported by the input driver."""

    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


@dataclasses.dataclass(frozen=True, slots=True)
class MouseInput:
    """Mouse button, held modifiers, drag path, and wheel displacement."""

    button: MouseButton = MouseButton.LEFT
    modifiers: tuple[str, ...] = ()
    path: tuple[Point, ...] = ()
    delta: Point = Point(0, 0)

    def __post_init__(self) -> None:
        if self.button not in {"left", "middle", "right"}:
            raise ValueError("unknown mouse button")
        if any(
            key not in {"Alt", "Control", "Meta", "Shift"} for key in self.modifiers
        ):
            raise ValueError("unknown mouse modifier")
        if len(self.path) > 200:
            raise ValueError("drag path exceeds 200 points")


type Target = AxLocator | DomLocator | VisualAnchor | ScreenTarget


class TargetForm(StrEnum):
    """The target form an adapter supports for an action.

    ``NONE`` is an action on the session itself, such as a navigation, a
    scroll of the page, or an answer to a dialog. The rest match the target
    types: an accessibility locator, a DOM locator, a painted canvas region,
    and a point or the keyboard focus of one screenshot.
    """

    NONE = "none"
    ACCESSIBILITY = "accessibility"
    DOM = "dom"
    VISUAL = "visual"
    SCREEN = "screen"


type Capabilities = Mapping[ActionKind, frozenset[TargetForm]]
"""The action types an adapter performs, each with the target forms it takes."""


def target_form(target: Target | None) -> TargetForm:
    """Return the form of ``target``, for comparison with an adapter's support.

    Examples
    --------
    >>> target_form(AxLocator("button", "Search"))
    <TargetForm.ACCESSIBILITY: 'accessibility'>
    >>> target_form(None)
    <TargetForm.NONE: 'none'>
    """
    match target:
        case AxLocator():
            return TargetForm.ACCESSIBILITY
        case DomLocator():
            return TargetForm.DOM
        case VisualAnchor():
            return TargetForm.VISUAL
        case ScreenTarget():
            return TargetForm.SCREEN
        case _:
            return TargetForm.NONE


def in_scope(scope: Scope, node: AxNode) -> bool:
    """Report whether ``node`` sits in the container ``scope`` names.

    A row is named by any of its ``row_names``; every other scope by its
    one name.

    Examples
    --------
    >>> row = Scope(ScopeKind.ROW, "600472")
    >>> cell = AxNode("cell", "Opened", scope=row, row_names=("600472", "M-1"))
    >>> in_scope(Scope(ScopeKind.ROW, "M-1"), cell)
    True
    >>> in_scope(Scope(ScopeKind.REGION, "M-1"), cell)
    False
    >>> placed = dataclasses.replace(cell, row_columns=(1, 3))
    >>> in_scope(Scope(ScopeKind.ROW, "M-1", column=3), placed)
    True
    >>> in_scope(Scope(ScopeKind.ROW, "M-1", column=2), placed)
    False
    """
    if node.scope is None or node.scope.kind is not scope.kind:
        return False
    if scope.column:
        return scope.name in row_names_at(node, scope.column)
    if scope.name == node.scope.name:
        return True
    return scope.kind is ScopeKind.ROW and scope.name in node.row_names


def row_names_at(node: AxNode, column: int) -> tuple[str, ...]:
    """Return the names of ``node``'s row that sit in the given column.

    Examples
    --------
    >>> cell = AxNode("cell", "Opened", row_names=("600472", "M-1"), row_columns=(1, 3))
    >>> row_names_at(cell, 3), row_names_at(cell, 2)
    (('M-1',), ())
    """
    return tuple(
        name
        for name, at in zip(node.row_names, node.row_columns, strict=False)
        if at == column
    )


def names_node(target: Target | None, node: AxNode) -> bool:
    """Report whether ``target`` names the control ``node`` describes.

    Used to carry an observed control's context into the expectation an
    action is checked against, and to withhold the value of a field a
    declared secret was typed into.

    Examples
    --------
    >>> node = AxNode("button", "Approve", frame=("ledger",))
    >>> names_node(AxLocator("button", "Approve", ("ledger",)), node)
    True
    >>> names_node(AxLocator("button", "Approve"), node)
    False

    A scoped target names only a node reported in that same scope, so a
    repeated label narrowed by its row is matched to that row's control.
    """
    if isinstance(target, (VisualAnchor, ScreenTarget)) or target is None:
        return False
    if target.scope is not None and not in_scope(target.scope, node):
        return False
    if isinstance(target, AxLocator):
        return (
            target.role == node.role
            and target.name == node.name
            and target.frame == node.frame
        )
    if isinstance(target, DomLocator):
        if target.tag != node.tag or target.frame != node.frame:
            return False
        if target.attribute is DomAttribute.SLOT:
            return node.slot == target.value
        if target.attribute is DomAttribute.TEXT:
            return node.name == target.value
        if target.attribute is DomAttribute.CSS_CLASS:
            return target.value in node.classes
        return any(
            name == target.attribute.value and value == target.value
            for name, value in node.attributes
        )
    return False


def _check_occurrence(occurrence: int) -> None:
    if occurrence < 0:
        raise ValueError("locator occurrence must not be negative")
    if occurrence > MAX_OCCURRENCE:
        raise ValueError("locator occurrence is past the disambiguation limit")


@dataclasses.dataclass(frozen=True, slots=True)
class SecretRef:
    """A value drawn from the environment, named but never carried.

    The name must appear in the profile's ``secrets`` table. Holding the
    reference instead of the value is why no redaction pass is needed later:
    the value is never in the run's data to begin with.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("secret reference must name a declared secret")


class Relation(StrEnum):
    """How record evidence relates to the operated control.

    ``ROW`` names the table row both sit in. ``CONTAINER`` names the smallest
    element holding both. For either, that boundary must hold exactly one
    control like the target, the same tag, role, and accessible name, and
    exactly one value like the evidence, the same tag in the same slot, such
    as the cell under the same column or the definition under the same term.
    A section that shows two members holds two Approve buttons, so it cannot
    be the boundary of either member, whatever anyone asserts about it.

    ``LABELLED`` is the page's own statement: the target names the evidence
    element in its ``aria-describedby`` or ``aria-labelledby``. There is no
    page-wide relation, because a page is not a record.
    """

    ROW = "row"
    CONTAINER = "container"
    LABELLED = "labelled"


@dataclasses.dataclass(frozen=True, slots=True)
class RecordEvidence:
    """The displayed value that identifies the record for an action.

    ``source`` names the control that showed ``value``, such as the cell
    holding a member number, by the same kind of locator a target uses.
    ``relation`` says how that control is tied to the target. The loop checks
    all three against the observation the decision was made from, and the
    surface reads them again immediately before acting. A value the model
    wrote that no observation showed is refused rather than trusted.

    ``value`` is member data. It lives in the run's working context and has
    no field in any journal event.

    Examples
    --------
    >>> source = DomLocator("dd", DomAttribute.ID, "member-number")
    >>> RecordEvidence(source, "10001", Relation.CONTAINER).relation
    <Relation.CONTAINER: 'container'>
    >>> RecordEvidence(source, "", Relation.CONTAINER)
    Traceback (most recent call last):
        ...
    ValueError: record evidence must carry the value it identifies
    """

    source: AxLocator | DomLocator
    value: str
    relation: Relation
    prefix: str = ""
    suffix: str = ""

    @property
    def displayed(self) -> str:
        """Return the exact composite label expected around this identifier."""
        return self.prefix + self.value + self.suffix

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("record evidence must carry the value it identifies")
        if not separates(self.prefix, self.suffix):
            raise ValueError(
                "a record label's prefix and suffix must not continue the identifier"
            )


def separates(prefix: str, suffix: str) -> bool:
    """Report whether static label text leaves the identifier whole.

    A prefix that ends, or a suffix that starts, with a letter or digit
    would let part of a longer identifier stand in for the whole one.

    Examples
    --------
    >>> separates("Member: ", "")
    True
    >>> separates("Member: ", "1")
    False
    >>> separates("NM", "")
    False
    >>> separates("(", ")")
    True
    """
    ends = bool(prefix) and prefix[-1].isalnum()
    starts = bool(suffix) and suffix[0].isalnum()
    return not ends and not starts


def word_positions(text: str, value: str) -> list[int]:
    """Return where ``value`` starts in ``text`` wherever it sits as whole words.

    Whole words follow ``matching``: spaces separate words, edge punctuation
    such as a bracket or a comma is not part of a word, and any other
    character joins a value to its neighbour.

    Examples
    --------
    >>> word_positions("OP2 and OP22, OP2", "OP2")
    [0, 14]
    >>> word_positions("Acct 12345-01", "12345")
    []
    >>> word_positions("anything", "")
    []
    """
    return [start for start, _ in matching.positions(text, value)]


_NEEDS_TARGET = TARGETED_ACTIONS
_NEEDS_VALUE = frozenset(
    {
        ActionKind.ASSERT,
        ActionKind.PRESS_KEY,
        ActionKind.SCROLL,
        ActionKind.SELECT,
        ActionKind.TYPE,
        ActionKind.WAIT,
    }
)
_MAY_TAKE_VALUE = frozenset({ActionKind.WAIT_FOR})
"""Actions whose value is optional: a wait_for may name its timeout."""
_NEEDS_DESTINATION = frozenset({ActionKind.NAVIGATE})


@dataclasses.dataclass(frozen=True, slots=True)
class Action:
    """One policy-checkable operation against a surface.

    ``kind`` is the unit the operator declares risk for, so it is always
    present. The remaining fields carry only what that kind needs, and an
    action that omits a required field or supplies an irrelevant one is
    rejected here rather than at the surface.

    A click is a click whether the control was found through the accessibility
    tree, through one DOM attribute, or by matching pixels. The targeting
    method lives in ``target`` so that ``kind`` stays the thing the operator
    declared risk for, and so no new action type appears as a way around that
    declaration.

    ``kind`` is what the browser does. ``effect`` is what that operation
    accomplishes in this application, such as ``submit_payment``, as the
    model understands it. A form submission is a click on its button or an
    Enter pressed into one of its fields, each carrying that effect; there is
    no separate submit action to route around. ``flag_risky`` adds a
    restriction: this operation needs a person. False adds nothing. It never
    means the operation was found safe, and it never clears a restriction.

    ``press_key`` may name a target. The key then goes to that control, and
    only after the surface has confirmed the control holds focus.

    Examples
    --------
    >>> Action(ActionKind.CLICK, AxLocator("button", "Search")).kind
    <ActionKind.CLICK: 'click'>
    >>> Action(ActionKind.CLICK, VisualAnchor("anchor-1")).kind
    <ActionKind.CLICK: 'click'>
    >>> Action(ActionKind.CLICK)
    Traceback (most recent call last):
        ...
    ValueError: click requires target
    >>> Action(ActionKind.NAVIGATE, destination="https://host.test/members")
    ... # doctest: +ELLIPSIS
    Action(kind=<ActionKind.NAVIGATE: 'navigate'>, ...)
    >>> Action(ActionKind.OBSERVE, destination="https://host.test/members")
    Traceback (most recent call last):
        ...
    ValueError: observe does not take destination
    """

    kind: ActionKind
    target: Target | None = None
    value: str | SecretRef | None = None
    destination: str | None = None
    evidence: RecordEvidence | None = None
    effect: str | None = None
    flag_risky: bool = False
    mouse: MouseInput = MouseInput()

    def __post_init__(self) -> None:
        if self.mouse != MouseInput() and not isinstance(self.target, ScreenTarget):
            raise ValueError("mouse options require a screen target")
        if self.kind not in FOCUSABLE_ACTIONS:
            _check(self.kind, "target", self.target, self.kind in _NEEDS_TARGET)
        if self.effect is not None and not is_effect_name(self.effect):
            raise ValueError("effect names use lowercase letters, digits, underscores")
        if self.kind not in _MAY_TAKE_VALUE:
            _check(self.kind, "value", self.value, self.kind in _NEEDS_VALUE)
        _check(
            self.kind,
            "destination",
            self.destination,
            self.kind in _NEEDS_DESTINATION,
        )
        if self.evidence is None:
            return
        if self.target is None:
            raise ValueError(f"{self.kind.value} does not take evidence")
        if (
            not isinstance(self.target, ScreenTarget)
            and self.evidence.source.frame != self.target.frame
        ):
            raise ValueError("record evidence must sit in the target's frame")


ENTER = "Enter"
"""The only key that can submit a form implicitly."""


@dataclasses.dataclass(frozen=True, slots=True)
class Operation:
    """The physical operation an action performs, independent of its label.

    A restriction the run learns is bound to this, so renaming the effect,
    leaving ``flag_risky`` off, or naming the same control another way does not
    escape it. The identity comes from the observed control the target
    resolved to, never from the locator the model wrote:

    - ``control`` is the session-local id the surface issued for that element.
      It is how two locators for one button are known to be one button. It
      means nothing outside this session and is not a locator for replay.
    - ``target`` describes the control by frame, tag, role, and accessible
      name. It is what carries a restriction to the same control after it is
      drawn again or the page is loaded again, when the element id is new.
      A target the observation could not resolve to one control is written
      as the locator, prefixed ``locator|``, and is treated as unresolved.
      A live pixel target uses ``screen``. Its restrictions cover every
      operation on the same route because pixels do not establish identity.
    - ``submission`` names the form submission the operation performs
      natively, as the form's id and the submitting button's id. A click on a
      submit button and an Enter that the browser turns into a click on that
      same button carry the same value, which is what makes them one
      operation for a restriction. An Enter that submits a form with no
      submit button carries the form alone.
    - ``form`` is the form the control belongs to. An Enter in a form whose
      submission the page does not make native cannot be proven equivalent to
      anything, so it is matched against every restriction on that form.
    - ``submission_as`` and ``form_as`` describe the same submission and form
      by what the page shows. They outlive the element ids, so a restriction
      still reaches the submission after the page is loaded again or the
      button is drawn again.

    Examples
    --------
    >>> button = AxNode("button", "Send payment", tag="button", control="d1:c4",
    ...                 form="d1:c2", submits="d1:c2>d1:c4")
    >>> Operation.of(
    ...     Action(ActionKind.CLICK, AxLocator("button", "Send payment")),
    ...     "/payments",
    ...     node=button,
    ... ).submission
    'd1:c2>d1:c4'
    """

    kind: ActionKind
    route: str
    target: str = ""
    control: str = ""
    key: str = ""
    submission: str = ""
    form: str = ""
    submission_as: str = ""
    form_as: str = ""

    binding: str = ""
    business: str = ""

    @classmethod
    def of(
        cls,
        action: Action,
        route: str,
        *,
        node: AxNode | None = None,
        painted: str = "",
    ) -> Operation:
        """Describe ``action`` taken on the screen ``route`` names.

        ``node`` is the one observed control the target resolved to, or None
        when it resolved to none or several. ``painted`` is a digest of the
        crop a painted target was cut from, which is what identifies it.
        """
        key = action.value if isinstance(action.value, str) else ""
        pressing = action.kind is ActionKind.PRESS_KEY
        if node is None:
            return cls(
                kind=action.kind,
                route=route,
                target=_unresolved(action.target, painted),
                key=key if pressing else "",
            )
        submission = submission_as = ""
        if action.kind is ActionKind.CLICK:
            submission, submission_as = node.submits, node.submits_as
        elif pressing and key == ENTER:
            submission, submission_as = node.enter, node.enter_as
        return cls(
            kind=action.kind,
            route=route,
            target=describe(node),
            control=node.control,
            key=key if pressing else "",
            submission=submission,
            form=node.form,
            submission_as=submission_as,
            form_as=node.form_as,
        )

    @property
    def resolved(self) -> bool:
        """Report whether the target was tied to one observed control."""
        return not self.target.startswith("locator|")

    @property
    def uncertain(self) -> bool:
        """Report whether this is an Enter in a form it may or may not submit."""
        return (
            self.kind is ActionKind.PRESS_KEY
            and self.key == ENTER
            and bool(self.form)
            and not self.submission
        )


def describe(node: AxNode) -> str:
    """Describe an observed control by frame, tag, role, and accessible name.

    Examples
    --------
    >>> describe(AxNode("button", "Send payment", tag="button"))
    '|button|button|Send payment'
    """
    return f"{'/'.join(node.frame)}|{node.tag}|{node.role}|{node.name}"


def _unresolved(target: Target | None, painted: str) -> str:
    if target is None:
        return ""
    if isinstance(target, ScreenTarget):
        return "screen"
    frame = "/".join(target.frame)
    if isinstance(target, VisualAnchor):
        return f"painted|{frame}|{painted}"
    if isinstance(target, AxLocator):
        return f"locator|ax|{target.role}|{target.name}|{frame}"
    return f"locator|dom|{target.tag}|{target.attribute.value}|{target.value}|{frame}"


def _check(kind: ActionKind, field: str, value: object, required: bool) -> None:
    if required and value is None:
        raise ValueError(f"{kind.value} requires {field}")
    if not required and value is not None:
        raise ValueError(f"{kind.value} does not take {field}")


class ObservationStatus(StrEnum):
    """The coverage of an observation.

    A truncated tree and a broken page are different situations for the next
    decision, so they are different statuses rather than one empty result.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ObservationProvenance(StrEnum):
    """Why an observation was taken, which is what the journal records."""

    INITIAL = "initial"
    REQUESTED = "requested"
    ALTERNATE = "alternate"
    POST_ACTION = "post_action"
    POST_INTERVENTION = "post_intervention"
    REFRESH = "refresh"


@dataclasses.dataclass(frozen=True, slots=True)
class ObservationRequest:
    """One request to look at the surface, before any page content is read.

    The gate sees this, not the result, because permission has to be settled
    before the adapter reads anything off the page.

    A structured request may ask for part of the screen: the controls after
    the first ``start``, only those in ``frame``, or only those in ``scope``.
    That is how a run reaches controls past the node limit without reading an
    unbounded document. Nothing here scrolls; scrolling is an action.

    Examples
    --------
    >>> ObservationRequest(ObservationMode.VISUAL).provenance
    <ObservationProvenance.REQUESTED: 'requested'>
    """

    mode: ObservationMode
    provenance: ObservationProvenance = ObservationProvenance.REQUESTED
    reason: str = ""
    start: int = 0
    frame: tuple[str, ...] | None = None
    scope: Scope | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("an observation window cannot start before the first")
        if self.narrowed and self.mode is not ObservationMode.STRUCTURED:
            raise ValueError("only the structured tool reads part of a screen")

    @property
    def narrowed(self) -> bool:
        """Report whether this asks for less than the whole screen from the top."""
        return bool(self.start) or self.frame is not None or self.scope is not None


@dataclasses.dataclass(frozen=True, slots=True)
class PageState:
    """The page identity used to detect a stale decision.

    ``location`` comes from session metadata and costs no page read.
    ``signature`` is the identity of the controls a structured observation
    saw, and is empty for a visual one. Comparing these two fields, and
    nothing else, is what keeps a fresh timestamp, a new observation id, or a
    blinking cursor in a screenshot from invalidating a decision. ``page`` is
    the surface's id for the window it was in, when it manages several, so two
    windows showing one URL are still two places.

    Examples
    --------
    >>> here = PageState("https://host.test/members/1", ("button:Search",))
    >>> here.differs_from(PageState("https://host.test/members/1"))
    False
    >>> here.differs_from(PageState("https://host.test/members/2"))
    True
    >>> here.differs_from(PageState("https://host.test/members/1", ("button:Wire",)))
    True
    """

    location: str
    signature: tuple[str, ...] = ()
    page: str = ""

    def differs_from(self, other: PageState) -> bool:
        """Report whether ``other`` describes a different page than this one."""
        if self.location != other.location:
            return True
        if self.page and other.page and self.page != other.page:
            return True
        if not self.signature or not other.signature:
            return False
        return self.signature != other.signature


@dataclasses.dataclass(frozen=True, slots=True)
class Expectation:
    """The conditions that must still hold when an action runs.

    Every decision is made about a screen that has already passed.
    ``page_state`` is where the surface was. ``context`` is the context path
    the observed control sat under: the page heading, its region, the heading
    above it, its row, and its form. The adapter re-derives that path from the
    control it is about to operate, and refuses the action when it differs, so
    a decision made about one record cannot land on the record that replaced
    it at the same URL.

    Moving a control under the same heading changes nothing here, and neither
    does unrelated content elsewhere on the page. This rejects a changed
    record, not a changed layout. A heading can be generic, or missing, so
    this path is never taken as proof of which record is shown; that is what
    ``RecordEvidence`` on the action is for.

    ``dialog`` names the dialog an answer was decided about. The surface
    answers only that dialog, so an approval given for one cannot be spent on
    another that opened in its place.

    ``displayed`` asks the surface to refuse a reading from a field, because a
    field shows what was typed into it and cannot confirm which record a page
    is about. ``displayed_evidence`` asks the same of the record evidence, for
    a reading that has to show which record a value belongs to.
    ``committed`` allows a field only when neither this run nor a person
    changed it in its current document, so its value came from the
    application. A required state or operating context is read this way.

    ``windows`` lists the page ids the run knows about. A surface that
    manages several windows refuses the action while any other one is open,
    because a new window goes to a person before automation continues. None
    means the caller knows of no windows and nothing is compared.

    ``control`` is the id of the observed control the decision, and any
    restriction check, was about. The surface refuses to act on a different
    control it already knows. ``strict`` also refuses a control drawn since
    the observation, which the loop asks for when a restriction on this
    screen could apply to it.

    ``observed_control`` holds the immutable metadata the loop judged for a
    screenshot point. The browser checks its reusable target metadata again
    before input, including row aliases and permitted DOM attributes. This
    check never compares editable values or unrelated page state.

    ``submits_as`` and ``enter_as`` describe the native submission the
    observed control performed when clicked, and when Enter was pressed in
    it, as the gate judged it; an empty text means none. The surface reads
    the live element's submission just before input and refuses the action
    when it differs, so a button a page turned into a submit button after
    the gate is never pressed under the old judgement (rule 6). None means
    nothing is compared.

    Examples
    --------
    >>> Expectation(PageState("https://host.test/queue")).context
    ()
    """

    page_state: PageState
    context: tuple[str, ...] = ()
    dialog: str | None = None
    control: str = ""
    strict: bool = False
    displayed: bool = False
    displayed_evidence: bool = False
    windows: tuple[str, ...] | None = None
    committed: bool = False
    submits_as: str | None = None
    enter_as: str | None = None
    binding: str | None = None
    business: str = ""
    observed_control: AxNode | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SelectOption:
    """A native option's visible label and exact application value, kept live."""

    label: str
    value: str
    disabled: bool = False
    selected: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class AxNode:
    """One control as a structured observation currently reports it.

    ``value`` is withheld whenever ``secret`` is set, which covers a password
    field and any field this run has typed a declared secret into. The value
    is not collected and then dropped; the adapter never reads it.

    ``row`` and ``ancestors`` name the table row and every element the
    control sits inside, nearest first, below the body. They are keys issued
    by one observation and mean nothing outside it. ``slot`` says which kind
    of value an element holds, such as the definition under the term "Member
    number", without the value itself. Record evidence is checked against
    all three.

    ``row_names`` lists every cell text that names the control's row, the
    scope's name first: each cell no other row repeats in its column. A row
    scope may name any of them, so a row whose first column is a sequence
    number can be found by the member it shows.

    ``classes`` is the element's class list, and ``in_view`` says whether it
    was inside the viewport when observed, so a run knows to scroll before
    it looks for the control in a screenshot.

    ``control`` is the id the surface issued for this element. It stays the
    same for the same element across observations of one document, whatever
    locator names it, and a new element gets a new id. ``form`` is the id of
    the form it belongs to. ``submits`` is the form and button a click on it
    submits natively, and ``enter`` is the form and button an Enter pressed
    into it submits natively, each written ``form>button``, or empty.

    The ``_as`` fields describe the same form and submissions by what the
    page shows instead of by element: the frame, the form's own attributes
    and field labels, and the button's tag, role, and name. Element ids end
    with their document. These descriptions are what a restriction is
    matched on after a reload or a redraw.
    """

    role: str
    name: str
    value: str | None = None
    frame: tuple[str, ...] = ()
    tag: str = ""
    attributes: tuple[tuple[str, str], ...] = ()
    interactions: tuple[ActionKind, ...] = ()
    scope: Scope | None = None
    context: tuple[str, ...] = ()
    row: str | None = None
    ancestors: tuple[str, ...] = ()
    slot: str = ""
    classes: tuple[str, ...] = ()
    in_view: bool = True
    control: str = ""
    form: str = ""
    submits: str = ""
    enter: str = ""
    form_as: str = ""
    submits_as: str = ""
    enter_as: str = ""
    enabled: bool = True
    visible: bool = True
    shadow: bool = False
    secret: bool = False
    options: tuple[SelectOption, ...] = ()
    options_complete: bool = True
    selections: tuple[tuple[str, bool], ...] = ()
    row_names: tuple[str, ...] = ()
    row_columns: tuple[int, ...] = ()

    @property
    def identity(self) -> str:
        """Return the node's part of a page signature, without any value."""
        frame = "/".join(self.frame)
        return f"{frame}|{self.role}:{self.name}|{'on' if self.enabled else 'off'}"


@dataclasses.dataclass(frozen=True, slots=True)
class VisualRegion:
    """A candidate control found in a screenshot by segmentation, not by a model.

    ``anchor_id`` is how a decision names this region, and it names the
    capture that produced it, so the same id is never handed out for two
    different controls. ``image`` is the crop itself, which is what makes the
    id mean something to a reader: an id beside a size describes nothing, and
    two canvases swapping places would produce the same description.

    ``width`` and ``height`` are reported so a decision can tell a button from
    a stray artifact. The region's position is not reported, because a
    position must never become the way a target is named.
    """

    anchor_id: str
    frame: tuple[str, ...] = ()
    width: int = 0
    height: int = 0
    hint: str = ""
    image: bytes | None = None
    image_media_type: str = "image/png"


DIALOG_ACTIONS = frozenset({ActionKind.ACCEPT_DIALOG, ActionKind.DISMISS_DIALOG})
"""The actions that answer a dialog, and the only ones a pending dialog allows."""


@dataclasses.dataclass(frozen=True, slots=True)
class PendingDialog:
    """A dialog the page opened and is waiting on.

    ``dialog_id`` is issued by the surface when the dialog opens and is never
    reused in the session, so an answer can name the one dialog it was
    decided about. ``kind`` is the browser's own word for it: alert, confirm,
    prompt, or beforeunload.

    Examples
    --------
    >>> PendingDialog("dialog-1", "confirm", "Post this entry?").dialog_id
    'dialog-1'
    """

    dialog_id: str
    kind: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class PageInfo:
    """One window the surface manages, as the next decision may choose it.

    ``route`` is the allowed route template for its current location, or
    empty when that location is outside the profile, in which case it cannot
    be selected and ``location`` is empty too. ``dialog`` is the id of a
    dialog waiting in it.
    """

    page_id: str
    active: bool
    route: str = ""
    dialog: str = ""
    location: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Window:
    """The portion of a structured screen covered by an observation.

    ``total`` counts the visible candidates the request covered, up to the
    adapter's counting limit; ``counted`` is false when there were more.
    ``collected`` is false when a frame in the window was refused, failed,
    or could not be read whole, so the window is partial for a reason other
    than the controls after it.
    """

    start: int
    shown: int
    total: int
    counted: bool = True
    frame: tuple[str, ...] | None = None
    scope: Scope | None = None
    collected: bool = True

    @property
    def rest(self) -> int:
        """Return how many counted candidates come after this window."""
        return max(0, self.total - self.start - self.shown)


@dataclasses.dataclass(frozen=True, slots=True)
class VisualMeta:
    """Metadata needed to interpret screenshot coordinates."""

    viewport_width: int
    viewport_height: int
    scroll_x: int
    scroll_y: int
    device_scale: float = 1.0
    masked: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """One tool result captured at a specific time.

    This is working state, not a recorded contract, so it is free to grow
    additional perception channels without disturbing anything that reads an
    artifact. ``observation_id`` identifies the capture and says nothing about
    whether the page changed; that question is ``page_state``'s.

    ``window`` says which part of a structured screen was described, and
    ``pages`` lists every window a surface manages, so a popup can be seen
    and chosen rather than silently operated or ignored.

    ``status`` describes the page body and ``dialog`` describes a dialog in
    front of it, and the two are separate on purpose. While a dialog waits the
    body cannot be read, so the status is ``UNAVAILABLE``, and yet the dialog
    itself is known and is enough to decide how to answer it.
    """

    observation_id: str
    mode: ObservationMode
    status: ObservationStatus
    page_state: PageState
    nodes: tuple[AxNode, ...] = ()
    regions: tuple[VisualRegion, ...] = ()
    image: bytes | None = None
    image_media_type: str = "image/png"
    visual: VisualMeta | None = None
    dialog: PendingDialog | None = None
    notes: tuple[str, ...] = ()
    window: Window | None = None
    pages: tuple[PageInfo, ...] = ()

    @property
    def location(self) -> str:
        """Return where the surface was when this observation was taken."""
        return self.page_state.location

    @property
    def usable(self) -> bool:
        """Report whether this observation can support a decision at all."""
        return self.status in {ObservationStatus.COMPLETE, ObservationStatus.PARTIAL}


class Outcome(StrEnum):
    """The outcome of resolving and attempting an action.

    The failures are kept apart because they need different answers. Nothing
    matched, several things matched, the thing matched but could not be
    operated, the page moved before the action ran, the adapter refused a side
    effect the profile denies, or the surface itself failed.
    """

    OK = "ok"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NOT_ACTIONABLE = "not_actionable"
    STALE = "stale"
    BLOCKED = "blocked"
    SURFACE_ERROR = "surface_error"
    UNCERTAIN = "uncertain"
    HANDOFF = "handoff"


@dataclasses.dataclass(frozen=True, slots=True)
class ActionResult:
    """The surface result for one action.

    No observation is returned here. Reading the page after an action is
    itself an observation, and it has to pass the gate like any other, so the
    loop asks for it separately. ``page_state`` carries session metadata only.

    ``detail`` describes the mechanism that failed and stays in the live
    transcript. ``side_effects`` names what the adapter refused, such as a
    popup under ``allow_new_windows: false``.

    ``dialog`` is the dialog waiting when the action returned. A click that
    opened one reports ``OK``, because the click was delivered, and reports
    the dialog here, because whatever the click started has not finished.

    ``screen`` is, for a read of a painted line, the text of every line the
    same picture shows, so the caller can tell a label shown once from a
    list. Like ``extracted`` it is live working state and never journaled.
    """

    outcome: Outcome
    page_state: PageState
    extracted: str | None = None
    detail: str | None = None
    side_effects: tuple[str, ...] = ()
    dialog: PendingDialog | None = None
    screen: tuple[str, ...] = ()


def unbound(
    observation: Observation, relation: Relation, target: AxNode, source: AxNode
) -> str | None:
    """Return why ``source`` does not identify the record ``target`` belongs to.

    Examples
    --------
    >>> button = AxNode("button", "Approve", tag="button", ancestors=("el-2", "el-1"))
    >>> number = AxNode("definition", "10001", tag="dd", slot="dd|Member number",
    ...                 ancestors=("el-3", "el-2", "el-1"))
    >>> screen = Observation("obs-1", ObservationMode.STRUCTURED,
    ...                      ObservationStatus.COMPLETE, PageState("https://h/q"),
    ...                      nodes=(button, number))
    >>> unbound(screen, Relation.CONTAINER, button, number) is None
    True
    >>> other = AxNode("button", "Approve", tag="button", ancestors=("el-5", "el-1"))
    >>> crowded = dataclasses.replace(screen, nodes=(button, number, other))
    >>> unbound(crowded, Relation.CONTAINER, other, number)
    'the boundary holding both holds 2 controls like the target'
    """
    if relation is Relation.LABELLED:
        ids = [value for name, value in source.attributes if name == "id"]
        named = " ".join(
            value
            for name, value in target.attributes
            if name in {"aria-describedby", "aria-labelledby"}
        ).split(" ")
        if ids and ids[0] in named:
            return None
        return "the target does not name the record evidence as its description"
    if relation is Relation.ROW:
        boundary = target.row if target.row == source.row else None
    else:
        boundary = next(
            (key for key in target.ancestors if key in source.ancestors), None
        )
    if boundary is None:
        return f"the target and the record evidence share no {relation.value}"
    inside = [node for node in observation.nodes if boundary in node.ancestors]
    likes = [node for node in inside if _like(node) == _like(target)]
    slots = [n for n in inside if (n.tag, n.slot) == (source.tag, source.slot)]
    if len(likes) != 1:
        return f"the boundary holding both holds {len(likes)} controls like the target"
    if len(slots) != 1:
        return f"the boundary holding both holds {len(slots)} values like the evidence"
    return None


def _like(node: AxNode) -> tuple[str, str, str]:
    return (node.tag, node.role, node.name)
