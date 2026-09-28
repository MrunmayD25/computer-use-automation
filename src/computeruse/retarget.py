"""Resolve a capability's reusable targets on a live surface.

Replay never asks a model where a control is. A structural target becomes an
accessibility or DOM locator, with its references filled in from this
invocation's inputs and variables, and it must match exactly one visible
control in a fresh structured observation. The match uses ``names_node``, the
same predicate discovery uses to tie a locator to an observed control, so a
recorded target means what it meant when it was seen.

A target belongs to the screen it was recorded on. It is resolved only while
the surface is on that allow-route template, so a control with the same role
and name on another permitted screen is never taken for it.

An observation that did not cover the whole screen cannot prove uniqueness or
absence. A missing target may be in the omitted part, and one visible match
may have a twin there. Such an observation reports ``INCOMPLETE``, which
replay treats as unknown. Two visible matches prove ambiguity regardless of
what the omitted part contains, so the resolver reports ``AMBIGUOUS``.

A visual target is a stored template matched with ``visual.locate`` against a
capture taken immediately before the action. The match must be unique by the
matcher's margin. The resulting point lives only as long as the capture it was
found in, and it is handed to the surface as a ``ScreenTarget``, which the
surface checks for freshness again before any input.

Neither kind of match proves which record a control acts on. That is record
evidence, which the gate and the surface enforce separately.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

from computeruse import matching, reading, visual
from computeruse.actions import (
    AxLocator,
    AxNode,
    DomLocator,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Point,
    Scope,
    ScopeKind,
    ScreenTarget,
    names_node,
    row_names_at,
    word_positions,
)
from computeruse.capability import (
    LocatorForm,
    Match,
    RadioTarget,
    Ref,
    RefKind,
    StructuralTarget,
    Template,
    TextTarget,
)
from computeruse.reading import Line
from computeruse.surface import Surface

type Text = Callable[[Ref], str]
"""Resolves a reference to its text for this invocation, or raises ``KeyError``."""


class Found(StrEnum):
    """The result of resolving a target by match count or route."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"
    OFF_ROUTE = "off_route"
    INCOMPLETE = "incomplete"


@dataclasses.dataclass(frozen=True, slots=True)
class Resolved:
    """A structural target resolved against one observation.

    ``node`` is the single observed control, and ``locator`` is what the
    surface resolves again immediately before acting.
    """

    found: Found
    locator: AxLocator | DomLocator | None = None
    node: AxNode | None = None


def locator_for(target: StructuralTarget, text: Text) -> AxLocator | DomLocator:
    """Fill a structural target's references and return the live locator.

    Raises
    ------
    KeyError
        If a reference names a value this invocation does not hold.
    ValueError
        If the filled locator is not one the action vocabulary accepts, or
        the target matches by ``CONTAINS`` and so needs an observation.
    """
    if target.match is Match.CONTAINS:
        raise ValueError("a target that contains a value is found with resolve")
    scope = None
    if target.scope is not None:
        scope = Scope(target.scope.kind, text(target.scope.name))
    if target.form is LocatorForm.ACCESSIBILITY:
        name = "" if target.name is None else text(target.name)
        return AxLocator(target.role, name, target.frame, scope)
    if target.attribute is None or target.value is None:
        raise ValueError("a dom target needs an attribute and a value")
    return DomLocator(
        target.tag, target.attribute, text(target.value), target.frame, scope
    )


def complete(observation: Observation | None) -> bool:
    """Report whether an observation covered the whole screen it describes.

    A partial status is incomplete. So is a window over part of a long list,
    which some adapters report in an optional ``window`` attribute with a
    ``start``, a ``shown`` count, a ``total``, and whether the total was
    ``counted`` in full. An adapter without that attribute reports coverage
    through its status alone.
    """
    if observation is None or observation.status is not ObservationStatus.COMPLETE:
        return False
    window = getattr(observation, "window", None)
    if window is None:
        return True
    return bool(
        window.start == 0
        and window.counted
        and window.shown >= window.total
        and getattr(window, "frame", None) is None
        and getattr(window, "scope", None) is None
    )


def resolve(
    target: StructuralTarget,
    observation: Observation | None,
    text: Text,
    route: str | None,
) -> Resolved:
    """Find ``target`` in a structured observation, requiring exactly one match.

    ``route`` is the allow-route template the surface is on. A target
    recorded on another screen is ``OFF_ROUTE`` without being looked for. Two
    matching controls are ``AMBIGUOUS``, never the first of them. A hidden
    control does not count. An incomplete observation proves neither absence
    nor uniqueness, so it gives ``INCOMPLETE`` unless it already shows two
    matches.

    Examples
    --------
    >>> from computeruse.actions import ObservationMode, ObservationStatus, PageState
    >>> from computeruse.capability import constant
    >>> seen = Observation("o1", ObservationMode.STRUCTURED,
    ...     ObservationStatus.COMPLETE, PageState("https://host.test/"),
    ...     nodes=(AxNode("button", "Save"), AxNode("button", "Save")))
    >>> save = StructuralTarget("save", "/", LocatorForm.ACCESSIBILITY, "button",
    ...     constant("Save"), "", None, None, (), None)
    >>> resolve(save, seen, lambda value: value.value, "/").found
    <Found.AMBIGUOUS: 'ambiguous'>
    >>> resolve(save, seen, lambda value: value.value, "/desk").found
    <Found.OFF_ROUTE: 'off_route'>
    """
    if route is None or route != target.route:
        return Resolved(Found.OFF_ROUTE)
    if observation is None or not observation.usable:
        return Resolved(Found.UNAVAILABLE)
    if target.match is Match.CONTAINS:
        locator = None
        matches = [
            node
            for node in observation.nodes
            if node.visible and _holds(target, node, text)
        ]
    else:
        locator = locator_for(target, text)
        matches = [
            node
            for node in observation.nodes
            if node.visible and names_node(locator, node)
        ]
    if len(matches) > 1:
        return Resolved(Found.AMBIGUOUS, locator)
    if not complete(observation):
        return Resolved(Found.INCOMPLETE, locator)
    if not matches:
        return Resolved(Found.NOT_FOUND, locator)
    if locator is None:
        locator = _seen(target, matches[0], text)
    return Resolved(Found.FOUND, locator, matches[0])


def _holds(target: StructuralTarget, node: AxNode, text: Text) -> bool:
    """Report whether ``node`` is a control a ``CONTAINS`` target describes.

    The role and frame compare exactly, as they do for any target, and a
    DOM target's tag and attribute value too. A name that is a reference
    must hold its value once as whole words. A row scope may be met by any
    of the row's names.
    """
    if target.frame != node.frame:
        return False
    if target.form is LocatorForm.DOM:
        found = locator_for(_unscoped(target), text)
        if not names_node(found, node):
            return False
    elif target.role != node.role or not _named(target.name, node.name, text):
        return False
    if target.scope is None:
        return True
    if node.scope is None or node.scope.kind is not target.scope.kind:
        return False
    if target.scope.column:
        # A row found by a cell that holds the request's value is bound to that
        # cell's column; a cell elsewhere that mentions the value is no match.
        names = row_names_at(node, target.scope.column)
    elif node.scope.kind is ScopeKind.ROW and node.row_names:
        names = node.row_names
    else:
        names = (node.scope.name,)
    return any(_named(target.scope.name, name, text) for name in names)


def _unscoped(target: StructuralTarget) -> StructuralTarget:
    """Return the target alone, matched exactly, with its scope checked apart."""
    return dataclasses.replace(target, scope=None, match=Match.EQUALS)


def _named(pattern: Ref | None, name: str, text: Text) -> bool:
    if pattern is None:
        return not name
    if pattern.kind is RefKind.CONSTANT:
        return name == pattern.value
    return len(word_positions(name, text(pattern))) == 1


def _seen(target: StructuralTarget, node: AxNode, text: Text) -> AxLocator | DomLocator:
    """Name the one control a ``CONTAINS`` target matched by its whole text.

    The surface resolves this exact locator again before acting, so the text
    around the value is read from the live page and lives only in memory. A
    DOM target keeps its own attribute and takes the row's first name.
    """
    scope = node.scope if target.scope else None
    if target.form is LocatorForm.DOM:
        found = locator_for(_unscoped(target), text)
        if isinstance(found, DomLocator):
            return dataclasses.replace(found, scope=scope)
    return AxLocator(node.role, node.name, node.frame, scope)


def shown(node: AxNode) -> str | None:
    """Return the text a control shows, or None for a field holding a secret.

    A field's value is what it holds. Anything else shows its accessible
    name, which for a cell or a definition is its text.
    """
    if node.secret:
        return None
    return node.value if node.value is not None else node.name


@dataclasses.dataclass(frozen=True, slots=True)
class Scene:
    """One live capture of a frame, taken for a single visual match.

    ``left`` and ``top`` place the captured image in the coordinate space of
    the capture ``capture_id`` names, so a point found in the image can be
    handed to the surface as that capture's coordinates. The scene lives in
    memory for one action and is never written anywhere.
    """

    capture_id: str
    image: bytes
    left: float = 0.0
    top: float = 0.0


class Scenes(Protocol):
    """The protocol for supplying a fresh capture to a visual match.

    A browser adapter would take a masked screenshot of the frame, or of the
    page when ``frame`` is empty, and issue a capture id its screen input
    accepts. The replay passes every capture through
    ``policy.evaluate_observation`` as a visual observation, with its risk,
    before it calls this. A provider grants nothing: it is called only after
    the gate has allowed the look.
    """

    def capture(self, frame: tuple[str, ...]) -> Scene | None:
        """Return a fresh capture of ``frame``, or None when none can be taken."""
        ...


@dataclasses.dataclass
class SurfaceScenes:
    """Capture through the guarded surface after replay permits the observation."""

    surface: Surface

    def capture(self, frame: tuple[str, ...]) -> Scene | None:
        """Use a fresh masked viewport. Frame-local crops remain unsupported."""
        if frame:
            return None
        observed = self.surface.observe(ObservationRequest(ObservationMode.VISUAL))
        if not observed.usable or observed.image is None:
            return None
        return Scene(observed.observation_id, observed.image)


@dataclasses.dataclass(frozen=True, slots=True)
class Spotted:
    """A visual match and the live input target derived from it."""

    found: Found
    target: ScreenTarget | None = None


def spot(template: Template, scene: Scene | None) -> Spotted:
    """Match a stored template in a fresh capture, requiring a unique match.

    The point is computed from where the template matched in this capture,
    and from nothing stored. A control that moved since recording is found
    where it is now; two copies of it are ``AMBIGUOUS``.
    """
    if scene is None:
        return Spotted(Found.UNAVAILABLE)
    match = visual.locate(template.image(), scene.image)
    if match.result is visual.MatchResult.AMBIGUOUS:
        return Spotted(Found.AMBIGUOUS)
    if match.result is not visual.MatchResult.FOUND or match.box is None:
        return Spotted(Found.NOT_FOUND)
    x, y = match.box.centre
    return Spotted(
        Found.FOUND,
        ScreenTarget(scene.capture_id, Point(scene.left + x, scene.top + y)),
    )


def text_line(
    target: TextTarget, lines: tuple[Line, ...], text: Text
) -> tuple[Found, Line | None]:
    """Find the one line a text target names among the lines a capture showed.

    A label names the line that starts with it. A reference names the line
    that holds its value once as whole words. A constant names the line that
    is exactly it. Spacing and case are ignored, because recognition can drop
    a space. No line is ``NOT_FOUND`` and two are ``AMBIGUOUS``.

    Examples
    --------
    >>> from computeruse.capability import constant
    >>> read = (Line("Status: active", 40, 220, 120, 20, 0.99),
    ...         Line("Open account", 40, 290, 110, 20, 0.99))
    >>> status = TextTarget("t1", "/", (), constant("Status:"), label=True)
    >>> text_line(status, read, lambda value: value.value)[0]
    <Found.FOUND: 'found'>
    >>> button = TextTarget("t2", "/", (), constant("Openaccount"))
    >>> text_line(button, read, lambda value: value.value)[1].text
    'Open account'
    """
    wanted = text(target.text)
    if target.label:
        matches = reading.anchored(lines, wanted)
    elif target.match is Match.CONTAINS:
        matches = [
            line
            for line in lines
            if matching.contains(line.text, wanted, painted=True, once=True)
        ]
    else:
        matches = [
            line
            for line in lines
            if reading.squash(line.text) == reading.squash(wanted)
        ]
    if not matches:
        return Found.NOT_FOUND, None
    if len(matches) > 1:
        return Found.AMBIGUOUS, None
    return Found.FOUND, matches[0]


def text_spot(
    target: TextTarget, scene: Scene | None, lines: tuple[Line, ...], text: Text
) -> Spotted:
    """Aim a text target at its line in a fresh capture, or beside it.

    The point is where the line is in this capture, moved by the target's
    offset when a field is drawn beside its label. Nothing stored holds a
    position; the line is found again each time.
    """
    if scene is None:
        return Spotted(Found.UNAVAILABLE)
    found, line = text_line(target, lines, text)
    if line is None:
        return Spotted(found)
    if target.dx or target.dy:
        x, y = line.left + target.dx, line.top + target.dy
    else:
        x, y = line.centre
    return Spotted(
        Found.FOUND,
        ScreenTarget(scene.capture_id, Point(scene.left + x, scene.top + y)),
    )


def radio_spot(
    target: RadioTarget, scene: Scene | None, reader: reading.Reader, text: Text
) -> Spotted:
    """Find one radio label and aim at its independently detected marker."""
    if scene is None:
        return Spotted(Found.UNAVAILABLE)
    try:
        choices = reading.radio_labels(scene.image, reader)
    except reading.RecognitionError:
        return Spotted(Found.UNAVAILABLE)
    found, line = text_line(target, tuple(choice.label for choice in choices), text)
    if line is None:
        return Spotted(found)
    choice = next(choice for choice in choices if choice.label is line)
    x, y = choice.marker.centre
    return Spotted(
        Found.FOUND,
        ScreenTarget(scene.capture_id, Point(scene.left + x, scene.top + y)),
    )
