"""Build business-outcome branches from short follow-up discoveries.

One discovery sees only its own result. A capability built from a successful
case does not know how to end when the application rejects a request, such as
for a missing record, insufficient permissions, or an invalid record state. A
sandbox profile may name test inputs that produce each outcome. After the
main discovery, one short discovery per case
runs with those inputs. The model recognizes the application's answer in its
own words and finishes with that outcome, such as ``record_not_found`` or
``permission_denied``, which the loop verifies on the live page like any
claim.

This module turns each such run into one branch of the main capability, with
the operator-authored branch format of ``branching``. A branch sits after
each step of the main capability that matches the second run's last step:
the same action type, route, target, and value. When its checks hold there,
a replay ends with the outcome instead of asking a person. Nothing here reads
a page or calls a model; the model's veto on saved texts is passed in.

A branch saves the texts its checks name, such as "0 members found". For
``record_not_found`` they were read for a record the application does not
hold, so they cannot name a customer. Every other outcome is about a record
that exists, so its texts must also pass the model's veto. All of them are
still refused when they hold any input or output value, and each is added to
the site's word list as ``outcome``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Protocol

from computeruse.branching import Branch, Plan, Recovery, extend
from computeruse.capability import (
    Absent,
    ActionNode,
    Capability,
    CapabilityError,
    Condition,
    Present,
    Ref,
    RefKind,
    ResultKind,
    ResultNode,
    StructuralTarget,
    Target,
    ValueIs,
    validate,
)
from computeruse.profile import RECORD_NOT_FOUND, ActionKind

LIMIT = 1
"""How often a replay may take the outcome edge; an ending is taken once."""

PREFIX = "outcome_"
"""The start of every target id the branch adds, apart from the main ones."""


def case_goal(
    goal: str, inputs: Mapping[str, str], case: Mapping[str, str]
) -> str | None:
    """Return the goal for an outcome's test case, or None when it cannot be written.

    Each input the case names must appear in the goal exactly as the
    discovered value, and is replaced by the case's value. No other text
    changes, and no pattern is used.

    Examples
    --------
    >>> found, absent = {"member_id": "NM54"}, {"member_id": "NM99"}
    >>> case_goal("Look up member NM54.", found, absent)
    'Look up member NM99.'
    >>> case_goal("Look up the member.", found, absent)
    """
    written = goal
    for name, value in case.items():
        found = inputs.get(name)
        if not found or found not in written:
            return None
        written = written.replace(found, value)
    return written


def outcome_plan(main: Capability, other: Capability) -> Plan | None:
    """Return the branch that ends ``main`` with the outcome ``other`` verified.

    ``other`` is the capability recorded from the second discovery. It must
    end in exactly one outcome, reached from an action that ``main`` also
    takes, with one success check and one transition there; every such step
    of ``main`` gets the branch. A check of a value the second run kept is
    left out, and the other checks may name inputs and constants only.
    Anything else gives None: the second run shows nothing ``main`` can
    safely branch on.
    """
    endings = [
        node
        for node in other.nodes
        if isinstance(node, ResultNode) and node.result is ResultKind.OUTCOME
    ]
    if len(endings) != 1 or not endings[0].outcome or not endings[0].checks:
        return None
    ending = endings[0]
    # A check of a value the second run kept, such as the operator it read
    # from the header, proves that run's context. That value exists only in
    # that run, and the main capability proves its context on its own steps,
    # so the branch keeps the checks about the outcome itself.
    checks = tuple(
        condition for condition in ending.checks if not _kept_value(condition)
    )
    if not checks:
        return None
    step = _last_act(other, ending.node_id)
    if step is None:
        return None
    places = _matching(main, other, step)
    if not places:
        return None
    renamed: dict[str, str] = {}
    targets: list[Target] = []
    for condition in checks:
        target_id = getattr(condition, "target", None)
        if target_id is None or target_id in renamed:
            continue
        target = other.target(target_id)
        renamed[target_id] = f"{PREFIX}{len(renamed) + 1}"
        targets.append(dataclasses.replace(target, target_id=renamed[target_id]))
    when: list[Condition] = []
    for condition in checks:
        if not _plain(condition):
            return None
        target_id = getattr(condition, "target", None)
        when.append(
            dataclasses.replace(condition, target=renamed[target_id])
            if target_id is not None
            else condition
        )
    branches = tuple(
        Branch(
            after,
            (*when, *_no_record_row(main, after, ending.outcome)),
            Recovery.OUTCOME,
            ending.outcome,
            LIMIT,
        )
        for after in places
    )
    return Plan(1, tuple(targets), branches)


def _no_record_row(main: Capability, after: str, outcome: str) -> tuple[Condition, ...]:
    """Require, for a missing record, that the found path's record target is absent.

    Discovery refused "not found" while a result row showed the record. The
    step ``main`` takes next finds the record's row or link by an input,
    so the branch also requires that target to find nothing on a complete
    look. The two paths then cannot both hold, and an incomplete look holds
    neither (rule 15).
    """
    if outcome != RECORD_NOT_FOUND:
        return ()
    node = main.node(after)
    if not isinstance(node, ActionNode):
        return ()
    following = [edge.to for edge in node.transitions if not edge.when]
    conditions: list[Condition] = []
    for name in following:
        step = main.node(name)
        if not isinstance(step, ActionNode) or step.target is None:
            continue
        target = main.target(step.target)
        if isinstance(target, StructuralTarget) and _names_an_input(target):
            conditions.append(Absent(target.target_id))
    return tuple(conditions)


def _names_an_input(target: StructuralTarget) -> bool:
    """Report whether a target finds its control by one of the request's inputs."""
    refs = (target.name, target.value, target.scope.name if target.scope else None)
    return any(ref is not None and ref.kind is RefKind.INPUT for ref in refs)


class Veto(Protocol):
    """Reject texts that could contain personal or record data."""

    def keep_words(self, texts: tuple[str, ...]) -> frozenset[str]:
        """Return the texts it judges to be the website's own."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class Adopted:
    """A capability and texts adopted from one outcome test, or the reason none were."""

    capability: Capability | None
    texts: tuple[str, ...] = ()
    note: str = ""


def adopt(
    main: Capability,
    recorded: Capability | None,
    outcome: str,
    veto: Veto | None,
) -> Adopted:
    """Add the outcome ``recorded`` verified to ``main``, or say why not.

    ``recorded`` is the capability from the test case's discovery, which was
    run for ``outcome``. A run that ended with another outcome teaches
    nothing, so a permission case can never teach "not found". Texts from any
    outcome but ``record_not_found`` must all pass ``veto``, because the
    record they were read for exists; without a veto they are not saved.
    """
    if recorded is None:
        return Adopted(None, note="the run did not finish with a recordable outcome")
    plan = outcome_plan(main, recorded)
    if plan is None:
        return Adopted(
            None, note="no step of the capability matches the run's last step"
        )
    if plan.branches[0].outcome != outcome:
        return Adopted(
            None, note="the run ended with an outcome different from its case"
        )
    texts = outcome_texts(plan)
    if outcome != RECORD_NOT_FOUND and texts:
        kept = veto.keep_words(texts) if veto is not None else frozenset()
        if not set(texts) <= kept:
            return Adopted(
                None,
                note="the model judged a text could be a person's or a record's data",
            )
    try:
        learned = learn(main, plan)
    except (CapabilityError, ValueError):
        return Adopted(None, note="the branch does not fit the capability")
    return Adopted(learned, texts)


def learn(main: Capability, plan: Plan) -> Capability:
    """Add a learned outcome to ``main``, telling its two paths apart.

    ``extend`` makes the step's own success check the condition of the path
    that goes on. A step whose success check also holds for a missing record,
    such as typing into a search that filters as it goes, would then leave
    both paths open, and the replay would stop rather than choose. So the path
    that goes on also requires the next step's preconditions and its control,
    such as the record's status cell, which only a record the application
    holds shows. The next step needs them anyway; here they are checked one
    step earlier.

    Raises
    ------
    CapabilityError
        If the result does not validate.
    ValueError
        If the plan does not fit ``main``.
    """
    learned = extend(main, plan)
    places = {branch.after for branch in plan.branches}
    nodes = []
    for node in learned.nodes:
        if isinstance(node, ActionNode) and node.node_id in places:
            onward, *others = node.transitions
            following = learned.node(onward.to)
            if isinstance(following, ActionNode):
                needed = following.requires
                if following.target is not None:
                    # The control the next step acts on or reads, such as the
                    # record's status cell, appears only for a record the
                    # application holds.
                    needed = (*needed, Present(following.target))
                added = tuple(c for c in needed if c not in onward.when)
                onward = dataclasses.replace(onward, when=onward.when + added)
                node = dataclasses.replace(node, transitions=(onward, *others))
        nodes.append(node)
    result = dataclasses.replace(learned, nodes=tuple(nodes))
    issues = validate(result)
    if issues:
        raise CapabilityError(issues)
    return result


def outcome_texts(plan: Plan) -> tuple[str, ...]:
    """Return the constant texts a branch plan would save, once each.

    Examples
    --------
    >>> from computeruse.capability import LocatorForm, Present, constant
    >>> target = StructuralTarget("outcome_1", "/", LocatorForm.ACCESSIBILITY,
    ...     "status", constant("0 members found"), "", None, None, (), None)
    >>> branch = Branch("s2", (Present("outcome_1"),), Recovery.OUTCOME, "x", 1)
    >>> outcome_texts(Plan(1, (target,), (branch,)))
    ('0 members found',)
    """
    texts: list[str] = []
    refs: list[Ref] = []
    for target in plan.targets:
        if isinstance(target, StructuralTarget):
            refs.extend(ref for ref in (target.name, target.value) if ref is not None)
            if target.scope is not None:
                refs.append(target.scope.name)
    for branch in plan.branches:
        for condition in branch.when:
            refs.extend(
                value
                for value in (getattr(condition, name, None) for name in ("value",))
                if isinstance(value, Ref)
            )
    for ref in refs:
        if ref.kind is RefKind.CONSTANT and ref.value not in texts:
            texts.append(ref.value)
    return tuple(texts)


LOOKS = frozenset({ActionKind.READ, ActionKind.OBSERVE, ActionKind.WAIT_FOR})
"""Steps that only look, which a run may add before claiming its outcome."""


def _last_act(capability: Capability, node_id: str) -> ActionNode | None:
    """Return the last step before ``node_id`` that did more than look.

    A run often reads the "nothing matched" message before it claims the
    outcome. That read is not where the paths part; the step before it is.
    """
    step = _before(capability, node_id)
    while step is not None and step.kind in LOOKS:
        step = _before(capability, step.node_id)
    return step


def _before(capability: Capability, node_id: str) -> ActionNode | None:
    """Return the one action whose transition leads to ``node_id``, if one does."""
    found = [
        node
        for node in capability.nodes
        if isinstance(node, ActionNode)
        and any(edge.to == node_id for edge in node.transitions)
    ]
    return found[0] if len(found) == 1 else None


def _matching(main: Capability, other: Capability, step: ActionNode) -> tuple[str, ...]:
    """Return the ids of every step of ``main`` that ``step`` repeats.

    A capability may take the same step twice, such as a search before and
    after choosing an operator. The outcome may show after either, so each
    gets the branch.
    """
    wanted = _shape(other, step)
    return tuple(
        node.node_id
        for node in main.nodes
        if isinstance(node, ActionNode)
        and node.verify
        and len(node.transitions) == 1
        and _shape(main, node) == wanted
    )


def _shape(capability: Capability, node: ActionNode) -> tuple[object, ...]:
    """Describe a step by the operation it performs, without the ids issued.

    The action type, route, target, and value say which operation it is, as
    they do for a restriction. The effect label is the model's own word for
    it and may differ between two runs of the same step, so it is left out.
    """
    target = (
        None
        if node.target is None
        else dataclasses.replace(capability.target(node.target), target_id="")
    )
    return (node.kind, node.route, target, node.value)


def _kept_value(condition: Condition) -> bool:
    """Report whether a condition compares a value its run kept, and nothing else."""
    return isinstance(condition, ValueIs) and condition.value.kind is RefKind.VARIABLE


def _plain(condition: Condition) -> bool:
    """Report whether a condition names only inputs and constants."""
    for name in ("value", "expected"):
        value = getattr(condition, name, None)
        if isinstance(value, Ref) and value.kind not in {
            RefKind.INPUT,
            RefKind.CONSTANT,
        }:
            return False
    return getattr(condition, "record", None) is None
