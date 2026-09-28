"""Resolve operator-declared operation identities from observed controls."""

from __future__ import annotations

import dataclasses
import hashlib
import json

from computeruse.actions import Action, AxNode, Operation, ScreenTarget
from computeruse.profile import (
    ActionKind,
    Limit,
    ObservationMode,
    OperationBinding,
    Profile,
)
from computeruse.reading import Line, at, squash


@dataclasses.dataclass(frozen=True, slots=True)
class BoundOperation:
    """An alias and its canonical operation under one exact set of declarations."""

    name: str
    business: str
    limit: Limit | None


def declared(profile: Profile, name: str) -> BoundOperation | None:
    """Return a named binding, including the fingerprint of every alias."""
    binding = next((item for item in profile.operations if item.name == name), None)
    if binding is None:
        return None
    aliases = [dataclasses.asdict(item) for item in profile.operations]
    encoded = json.dumps(
        {
            "operations": aliases,
            "pickers": [dataclasses.asdict(item) for item in profile.pickers],
        }
        if profile.pickers
        else aliases,
        sort_keys=True,
    )
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return BoundOperation(name, f"{digest}:{binding.operation}", binding.limit)


def required_selections(profile: Profile, operation: Operation) -> frozenset[str]:
    """Return the required selection groups for an attested or unidentified input."""
    if operation.kind not in {
        ActionKind.CLICK,
        ActionKind.PRESS_KEY,
        ActionKind.TYPE,
        ActionKind.SELECT,
    }:
        return frozenset()
    held = declared(profile, operation.binding)
    names = {
        item.operation
        for item in profile.operations
        if (
            held is not None
            and held.business == operation.business
            and item.name == held.name
        )
        or (held is None and item.route == operation.route)
    }
    return frozenset(
        item.operation for item in profile.pickers if item.operation in names
    )


def selections_ready(
    profile: Profile, operation: Operation, node: AxNode | None
) -> bool:
    """Require each declared selection group from the gated observation."""
    required = required_selections(profile, operation)
    if not required:
        return True
    selected = dict(node.selections) if node is not None else {}
    return all(selected.get(name) is True for name in required)


def limit(profile: Profile, operation: Operation) -> Limit | None:
    """Apply a binding's limit, or the route's strictest limit when unidentified."""
    if operation.kind in {
        ActionKind.READ,
        ActionKind.OBSERVE,
        ActionKind.ASSERT,
        ActionKind.WAIT,
        ActionKind.WAIT_FOR,
    }:
        return None
    binding = declared(profile, operation.binding)
    if binding is not None and binding.business == operation.business:
        limits = {
            item.limit
            for item in profile.operations
            if item.operation == operation.business.rpartition(":")[2]
        }
    else:
        limits = {
            item.limit for item in profile.operations if item.route == operation.route
        }
    return (
        Limit.DENY
        if Limit.DENY in limits
        else Limit.RISKY
        if Limit.RISKY in limits
        else None
    )


def identify(
    profile: Profile,
    action: Action,
    route: str,
    *,
    node: AxNode | None = None,
    lines: tuple[Line, ...] = (),
    frame: tuple[str, ...] = (),
    scale: float = 1.0,
) -> BoundOperation | None:
    """Identify exactly one declaration from the evidence of the aimed control.

    Overlapping declarations are ambiguous even when they name one business
    operation. Nothing falls back from an ambiguous specific target to a
    general one. A painted binding requires its declared context lines once.
    """
    found = [
        item
        for item in profile.operations
        if matches(
            item,
            action,
            route,
            node=node,
            lines=lines,
            frame=frame,
            scale=scale,
        )
    ]
    return declared(profile, found[0].name) if len(found) == 1 else None


def matches(
    binding: OperationBinding,
    action: Action,
    route: str,
    *,
    node: AxNode | None,
    lines: tuple[Line, ...] = (),
    frame: tuple[str, ...] = (),
    scale: float = 1.0,
) -> bool:
    """Apply one declaration's explicit context, target, and input constraints."""
    if action.kind not in binding.kinds or route != binding.route:
        return False
    if binding.key and (
        action.kind is not ActionKind.PRESS_KEY or action.value != binding.key
    ):
        return False
    if binding.mode is ObservationMode.STRUCTURED:
        submits = bool(node and Operation.of(action, route, node=node).submission)
        return bool(
            node is not None
            and not node.secret
            and node.visible
            and node.frame == binding.frame
            and all(item in node.context for item in binding.context)
            and (binding.role == "*" or node.role == binding.role)
            and (binding.target == "*" or node.name == binding.target)
            and (
                binding.submission == "any"
                or submits == (binding.submission == "native")
            )
        )
    if not isinstance(action.target, ScreenTarget) or frame != binding.frame:
        return False
    if any(
        sum(squash(line.text) == squash(marker) for line in lines) != 1
        for marker in binding.context
    ):
        return False
    if binding.target == "*":
        return True
    point = action.target.point
    if point is None:
        return False
    line = at(lines, point.x * scale, point.y * scale)
    return line is not None and squash(line.text) == squash(binding.target)
