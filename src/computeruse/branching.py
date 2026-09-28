"""Add explicit business outcomes and bounded recovery to a discovered draft.

An operator supplies the predicates. One successful discovery cannot infer the
meaning of an unseen error screen. Recovery waits or hands over, then checks the
result again without repeating the operation that may already have happened.
"""

from __future__ import annotations

import dataclasses
import json
from enum import StrEnum
from pathlib import Path
from typing import cast

from computeruse.capability import (
    ActionNode,
    Capability,
    CapabilityError,
    CheckNode,
    Condition,
    Edge,
    EdgeOrigin,
    HelpReason,
    HumanNode,
    Issue,
    Node,
    ResultKind,
    ResultNode,
    Review,
    Target,
    _decode,
    _unique_keys,
    identifier,
    validate,
)


class Recovery(StrEnum):
    """The action to take for a known exceptional state."""

    OUTCOME = "outcome"
    WAIT = "wait"
    HUMAN = "human"


@dataclasses.dataclass(frozen=True)
class Branch:
    """One operator-authored alternative after an observed action."""

    after: str
    when: tuple[Condition, ...]
    recovery: Recovery
    outcome: str
    limit: int


@dataclasses.dataclass(frozen=True)
class Plan:
    """Versioned branch declarations independent of an application name."""

    version: int
    targets: tuple[Target, ...]
    branches: tuple[Branch, ...]


def load_plan(path: Path) -> Plan:
    """Read a strict plan without including its data in errors."""
    if path.stat().st_size > 1_000_000:
        raise ValueError("branch plan is too large")
    issues: list[Issue] = []
    document = json.loads(path.read_text(), object_pairs_hook=_unique_keys)
    plan = cast("Plan", _decode(Plan, document, "branches", issues))
    if issues:
        raise CapabilityError(issues)
    if plan.version != 1:
        raise ValueError("unknown branch plan version")
    return plan


def extend(capability: Capability, plan: Plan) -> Capability:
    """Return a draft with ``plan`` applied, leaving ``capability`` unchanged."""
    nodes = list(capability.nodes)
    outcomes = list(capability.outcomes)
    for index, branch in enumerate(plan.branches):
        if not branch.when or not 1 <= branch.limit <= 100:
            raise ValueError("a branch needs predicates and a finite limit")
        node = next(
            (item for item in capability.nodes if item.node_id == branch.after), None
        )
        if (
            not isinstance(node, ActionNode)
            or not node.verify
            or len(node.transitions) != 1
        ):
            raise ValueError("a branch needs one observed action with a success check")
        if branch.recovery is Recovery.OUTCOME and not identifier(branch.outcome):
            raise ValueError("a business outcome needs a name")
        if branch.recovery is not Recovery.OUTCOME and branch.outcome:
            raise ValueError("recovery does not declare a business outcome")
        prefix = f"branch_{index + 1}"
        success = dataclasses.replace(
            node.transitions[0], when=node.verify, origin=EdgeOrigin.AUTHORED
        )
        alternative = Edge(prefix, branch.when, EdgeOrigin.AUTHORED, branch.limit)
        index_in_graph = next(
            i for i, item in enumerate(nodes) if item.node_id == node.node_id
        )
        current = nodes[index_in_graph]
        if not isinstance(current, ActionNode):
            raise TypeError("a branch source must remain an action")
        transitions = (
            (success, alternative)
            if current == node
            else (*current.transitions, alternative)
        )
        nodes[index_in_graph] = dataclasses.replace(
            node, verify=(), transitions=transitions
        )
        nodes.extend(_recovery(branch, node, prefix, success, alternative))
        if branch.outcome and branch.outcome not in outcomes:
            outcomes.append(branch.outcome)
    result = dataclasses.replace(
        capability,
        nodes=tuple(nodes),
        targets=capability.targets + plan.targets,
        outcomes=tuple(outcomes),
        provenance=dataclasses.replace(capability.provenance, review=Review.DRAFT),
    )
    issues = validate(result)
    if issues:
        raise CapabilityError(issues)
    return result


def _recovery(
    branch: Branch,
    action: ActionNode,
    prefix: str,
    success: Edge,
    alternative: Edge,
) -> tuple[Node, ...]:
    if branch.recovery is Recovery.OUTCOME:
        return (ResultNode(prefix, ResultKind.OUTCOME, branch.outcome, branch.when),)
    if branch.recovery is Recovery.WAIT:
        return (CheckNode(prefix, (success, alternative), delay_ms=100),)
    check = CheckNode(prefix + "_check", (success, alternative))
    # A hand-back must verify the success state before continuing.
    continuation = (
        Edge(check.node_id, action.verify, EdgeOrigin.AUTHORED, branch.limit),
    )
    recovery: Node = HumanNode(
        prefix, action.route, HelpReason.MANUAL_STEP, None, None, False, continuation
    )
    return recovery, check
