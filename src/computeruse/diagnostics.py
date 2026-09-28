"""A structural failure snapshot that never copies text or field values."""

from __future__ import annotations

import dataclasses

from computeruse.actions import Observation, ObservationStatus

ROLES = frozenset(
    {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "checkbox",
        "radio",
        "spinbutton",
        "heading",
        "status",
        "alert",
        "cell",
        "rowheader",
        "columnheader",
        "table",
        "row",
        "dialog",
        "text",
        "img",
        "option",
        "listbox",
        "slider",
        "tab",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class ControlShape:
    """Closed widget category and anonymous relationships in one observation."""

    role: str
    frame: int
    group: int
    enabled: bool
    visible: bool
    protected: bool
    field: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """At most 512 controls, with coverage and dialog state but no page text."""

    status: ObservationStatus
    shown: int
    total: int
    counted: bool
    dialog: bool
    truncated: bool
    controls: tuple[ControlShape, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CheckShape:
    """The kind, result, and target of one check from the failing step.

    ``held`` is None when an earlier failure prevented this check from running.
    The target is described by its form, a role from ``ROLES`` or
    ``other``, how its name matches, what kind of reference names it, its
    scope's kind, and its frame depth. No text or value is kept.
    """

    tag: str
    held: bool | None
    form: str = ""
    role: str = ""
    match: str = ""
    name: str = ""
    scope: str = ""
    frames: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class FailureEvidence:
    """The failing location, condition categories, and safe structural snapshot."""

    step: int
    node: str
    reason: str
    expected: tuple[str, ...]
    observed: Snapshot | None
    checks: tuple[CheckShape, ...] = ()


def snapshot(observation: Observation | None) -> Snapshot | None:
    """Convert live nodes directly to categories, without serializing their text."""
    if observation is None:
        return None
    frames = {}
    groups = {}
    shapes = []
    for node in observation.nodes[:512]:
        frame = frames.setdefault(node.frame, len(frames) + 1)
        group = (
            groups.setdefault((node.frame, node.scope), len(groups) + 1)
            if node.scope is not None
            else 0
        )
        shapes.append(
            ControlShape(
                node.role if node.role in ROLES else "other",
                frame,
                group,
                node.enabled,
                node.visible,
                node.secret,
                node.value is not None,
            )
        )
    window = observation.window
    return Snapshot(
        observation.status,
        len(observation.nodes),
        window.total if window else len(observation.nodes),
        window.counted if window else observation.status is ObservationStatus.COMPLETE,
        observation.dialog is not None,
        len(observation.nodes) > 512,
        tuple(shapes),
    )
