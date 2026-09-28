"""Plain descriptions of a capability's steps, controls, and checks.

The review summary, replay report, and replay help request all describe a step
with the same wording. This module does not read pages. It describes a saved
capability by names. Only a live replay supplies current values, such as the
member or account for the step.
"""

from __future__ import annotations

from collections.abc import Mapping

from computeruse.capability import (
    Absent,
    ActionNode,
    AtRoute,
    Bound,
    Capability,
    Completed,
    Condition,
    DialogOpen,
    FocusTarget,
    HumanNode,
    Present,
    Ref,
    RefKind,
    Shows,
    StructuralTarget,
    Target,
    TextTarget,
    ValueIs,
)

TAGS = {"a": "link", "td": "cell", "th": "header", "select": "list", "input": "field"}
"""Plain names for the page tags a control may be found by."""


def describe_value(value: Ref | None, values: Mapping[str, str] | None = None) -> str:
    """Name a reference: a saved text in quotes, anything else by name.

    During a live replay, ``values`` holds the invocation's values, and a
    reference it names is shown by its value, so a person sees the member
    or the account the step is about. A saved capability never has them.
    """
    if value is None:
        return ""
    if value.kind is RefKind.CONSTANT:
        return repr(value.value)
    if values and value.name in values:
        return repr(values[value.name])
    return value.name


def describe_target(
    target: Target | None, values: Mapping[str, str] | None = None
) -> str:
    """Name a control the way a person would look for it.

    Examples
    --------
    >>> from computeruse.capability import DomAttribute, LocatorForm, ScopeKind
    >>> from computeruse.capability import ScopeSpec, constant, ref
    >>> row = ScopeSpec(ScopeKind.ROW, ref(RefKind.VARIABLE, "fact_1"))
    >>> number = StructuralTarget("t1", "/accounts", LocatorForm.DOM, "", None, "a",
    ...     DomAttribute.SLOT, constant("a|Number"), (), row)
    >>> describe_target(number)
    "link 'Number' in the row for fact_1"
    >>> describe_target(number, {"fact_1": "AC7"})
    "link 'Number' in the row for 'AC7'"
    """
    if isinstance(target, StructuralTarget):
        kind = target.role or TAGS.get(target.tag, target.tag)
        named = describe_value(target.name, values)
        if not named and target.value is not None:
            # A control found by its slot, such as "a|Number", is named by
            # the label after the bar.
            label = target.value.value.partition("|")[2]
            named = repr(label) if label else ""
        said = f"{kind} {named}".strip()
        if target.scope is not None:
            where = describe_value(target.scope.name, values)
            said += f" in the {target.scope.kind.value} for {where}"
        return said
    if isinstance(target, TextTarget):
        return f"the painted text {describe_value(target.text, values)}"
    if isinstance(target, FocusTarget):
        return "the focused field"
    return "a reviewed image" if target is not None else "the page"


def describe_step(
    node: ActionNode,
    targets: Mapping[str, Target],
    values: Mapping[str, str] | None = None,
) -> str:
    """Describe one action node in a short line."""
    where = describe_target(targets.get(node.target or ""), values)
    value = describe_value(node.value, values)
    match node.kind.value:
        case "navigate":
            said = f"go to {node.destination.route if node.destination else node.route}"
        case "type":
            said = f"type {value} into {where}"
        case "select":
            said = f"choose {value} in {where}"
        case "press_key":
            said = f"press {value} in {where}"
        case "read":
            into = describe_value(node.into) or "a value"
            said = f"read {where} into {into}"
        case kind:
            said = f"{kind.replace('_', ' ')} {where}"
    if node.approval.value != "none":
        said += ", with approval on every run"
    return said


def steps(capability: Capability) -> list[tuple[str, str]]:
    """Return each step's node id and a short line saying what it does."""
    targets = {target.target_id: target for target in capability.targets}
    listed: list[tuple[str, str]] = []
    for node in capability.nodes:
        if isinstance(node, ActionNode):
            listed.append((node.node_id, describe_step(node, targets)))
        elif isinstance(node, HumanNode):
            reason = node.reason.value.replace("_", " ")
            listed.append((node.node_id, f"a person: {reason}"))
    return listed


def describe_condition(
    condition: Condition,
    capability: Capability,
    values: Mapping[str, str] | None = None,
) -> str:
    """Describe what one check expects and name its control.

    Live values appear only when a replay passes ``values``.
    """
    targets = {item.target_id: item for item in capability.targets}
    target = describe_target(targets.get(getattr(condition, "target", "")), values)
    match condition:
        case AtRoute():
            return f"the page is {condition.route}"
        case Present():
            return f"{target} is on the page"
        case Absent():
            return f"{target} is gone"
        case Shows():
            return f"{target} shows {describe_value(condition.value, values)}"
        case DialogOpen():
            return f"a {condition.kind} dialog is open"
        case Bound():
            return f"{describe_value(condition.value, values)} has a value"
        case ValueIs():
            value, expected = condition.value, condition.expected
            said = describe_value(expected, values)
            return f"{describe_value(value, values)} is {said}"
        case Completed():
            return f"step {condition.node} completed"
    return type(condition).TAG.replace("_", " ")
