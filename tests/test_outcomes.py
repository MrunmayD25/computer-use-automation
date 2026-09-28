"""A second discovery for a missing record teaches a capability its outcome."""

import dataclasses

import pytest
from replay_fakes import (
    BALANCE,
    MEMBER,
    PROFILE,
    Clock,
    People,
    bank,
    go,
    transfer_capability,
)

from computeruse.branching import extend
from computeruse.capability import (
    ActionNode,
    Bound,
    Present,
    RefKind,
    ResultKind,
    ResultNode,
    Shows,
    ValueIs,
    ref,
    validate,
)
from computeruse.capability import Match as TextMatch
from computeruse.outcomes import adopt, case_goal, outcome_plan, outcome_texts
from computeruse.profile import ActionKind, load_profile
from computeruse.replay import MemoryReplayLog, Status, replay

NOT_FOUND = {("search", ActionKind.CLICK, "Search"): "not_found"}


def _lookup(ending: ResultNode, verify) -> object:
    """Search for a member, end with ``ending``, and prove ``verify``."""
    capability = transfer_capability()
    kept = ("enter_member", "search", "read_balance")
    nodes = []
    for node in capability.nodes:
        if node.node_id not in kept:
            continue
        if node.node_id == "search":
            assert isinstance(node, ActionNode)
            node = dataclasses.replace(
                node,
                verify=verify,
                transitions=(
                    go("read_balance" if ending.node_id == "found" else ending.node_id),
                ),
            )
        if node.node_id == "read_balance":
            if ending.node_id != "found":
                continue
            node = dataclasses.replace(node, transitions=(go("found"),))
        nodes.append(node)
    nodes.append(ending)
    lookup = dataclasses.replace(
        capability,
        inputs=capability.inputs[:1],
        secrets=(),
        nodes=tuple(nodes),
        outcomes=(ending.outcome,) if ending.outcome else (),
    )
    assert validate(lookup) == (), validate(lookup)
    return lookup


def _main():
    found = ResultNode("found", ResultKind.SUCCESS, "", (Bound(BALANCE),))
    return _lookup(found, (Shows("member_shown", MEMBER, TextMatch.EQUALS),))


def _missing():
    ending = ResultNode(
        "missing", ResultKind.OUTCOME, "record_not_found", (Present("no_member"),)
    )
    return _lookup(ending, (Present("no_member"),))


def test_the_goal_for_an_outcome_case_changes_only_its_values():
    goal = "Look up member 10001 and read the balance."
    assert (
        case_goal(goal, {"member_id": "10001"}, {"member_id": "99999"})
        == "Look up member 99999 and read the balance."
    )


def test_a_branch_is_learned_after_the_step_both_runs_take():
    plan = outcome_plan(_main(), _missing())
    assert plan is not None
    (branch,) = plan.branches
    assert branch.after == "search"
    assert branch.outcome == "record_not_found"
    assert outcome_texts(plan) == ("No member found",)


def test_the_learned_outcome_ends_a_replay_for_a_missing_record(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    profile = load_profile(path)
    plan = outcome_plan(_main(), _missing())
    assert plan is not None
    learned = extend(_main(), plan)
    clock = Clock()
    people = People([])
    result = replay(
        learned,
        {"member_id": "99999"},
        profile=profile,
        surface=bank("99999", rules=NOT_FOUND),
        control=people.control(clock),
        log=MemoryReplayLog(),
        clock=clock,
        sleep=clock.advance,
        accept_draft=True,
    )
    assert result.status is Status.OUTCOME, result
    assert result.outcome == "record_not_found"
    assert people.requests == []
    # The found path still works for a member the application holds.
    found = replay(
        learned,
        {"member_id": "10001"},
        profile=profile,
        surface=bank(),
        control=People([]).control(clock),
        log=MemoryReplayLog(),
        clock=clock,
        sleep=clock.advance,
        accept_draft=True,
    )
    assert found.status is Status.SUCCEEDED, found


def test_no_branch_when_the_second_run_took_a_step_the_main_one_did_not():
    other = _missing()
    search = other.node("search")
    assert isinstance(search, ActionNode)
    changed = dataclasses.replace(search, route="/elsewhere")
    nodes = tuple(changed if n.node_id == "search" else n for n in other.nodes)
    assert outcome_plan(_main(), dataclasses.replace(other, nodes=nodes)) is None


def test_the_model_may_name_the_same_step_differently_in_each_run():
    other = _missing()
    search = other.node("search")
    assert isinstance(search, ActionNode)
    renamed = dataclasses.replace(search, effect="find_member")
    nodes = tuple(renamed if n.node_id == "search" else n for n in other.nodes)
    plan = outcome_plan(_main(), dataclasses.replace(other, nodes=nodes))
    assert plan is not None
    assert [branch.after for branch in plan.branches] == ["search"]


def test_every_repeat_of_the_shared_step_gets_the_branch():
    main = _main()
    search = main.node("search")
    enter = main.node("enter_member")
    assert isinstance(search, ActionNode)
    assert isinstance(enter, ActionNode)
    # The same search, taken a second time later in the capability.
    again = dataclasses.replace(search, node_id="search_again")
    first = dataclasses.replace(search, transitions=(go("search_again"),))
    nodes = tuple(first if n.node_id == "search" else n for n in main.nodes)
    twice = dataclasses.replace(main, nodes=(*nodes[:2], again, *nodes[2:]))
    plan = outcome_plan(twice, _missing())
    assert plan is not None
    assert sorted(branch.after for branch in plan.branches) == [
        "search",
        "search_again",
    ]


def test_a_read_before_the_claim_does_not_hide_the_shared_step():
    other = _missing()
    search = other.node("search")
    assert isinstance(search, ActionNode)
    # The run read the "nothing matched" message before it claimed the outcome.
    read = ActionNode(
        "read_message",
        ActionKind.READ,
        search.route,
        target="no_member",
        value=None,
        destination=None,
        effect=None,
        record=None,
        result_record=None,
        into=None,
        approval=search.approval,
        mandatory=False,
        requires=(),
        verify=(),
        transitions=(go("missing"),),
    )
    first = dataclasses.replace(search, transitions=(go("read_message"),))
    nodes = []
    for node in other.nodes:
        nodes.append(first if node.node_id == "search" else node)
        if node.node_id == "search":
            nodes.append(read)
    plan = outcome_plan(_main(), dataclasses.replace(other, nodes=tuple(nodes)))
    assert plan is not None
    assert [branch.after for branch in plan.branches] == ["search"]


def _overlapping():
    """A lookup whose search check also holds when nothing matched.

    The search field stays on every screen, as it would for a filter that
    updates while you type. A missing member therefore still satisfies "the
    field is there."
    """
    main = _main()
    search = main.node("search")
    read = main.node("read_balance")
    assert isinstance(search, ActionNode)
    assert isinstance(read, ActionNode)
    # The next step, reading the balance, states no precondition of its own.
    # the control it reads is what only a held member shows.
    changed = {
        "search": dataclasses.replace(search, verify=(Present("member_field"),)),
        "read_balance": dataclasses.replace(read, requires=()),
    }
    nodes = tuple(changed.get(node.node_id, node) for node in main.nodes)
    return dataclasses.replace(main, nodes=nodes)


def _with_field_everywhere(app):
    from replay_fakes import Page, field

    member = app.screens["member"]
    app.screens["member"] = Page(member.path, (field("Member number"), *member.nodes))
    return app


def _replay(capability, member, app, profile):
    clock = Clock()
    return replay(
        capability,
        {"member_id": member},
        profile=profile,
        surface=app,
        control=People([]).control(clock),
        log=MemoryReplayLog(),
        clock=clock,
        sleep=clock.advance,
        accept_draft=True,
    )


def test_the_onward_path_also_needs_the_next_steps_preconditions(tmp_path):
    from computeruse.outcomes import learn

    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    profile = load_profile(path)
    main = _overlapping()
    plan = outcome_plan(main, _missing())
    assert plan is not None
    missing = _with_field_everywhere(bank("99999", rules=NOT_FOUND))
    # Without the next step's preconditions both paths hold, and the replay
    # stops rather than choose.
    overlapped = _replay(extend(main, plan), "99999", missing, profile)
    assert overlapped.status is not Status.OUTCOME
    learned = learn(main, plan)
    ended = _replay(
        learned,
        "99999",
        _with_field_everywhere(bank("99999", rules=NOT_FOUND)),
        profile,
    )
    assert ended.status is Status.OUTCOME, ended
    assert ended.outcome == "record_not_found"
    # The onward path still succeeds for a member the application holds.
    ordinary = outcome_plan(_main(), _missing())
    assert ordinary is not None
    found = _replay(learn(_main(), ordinary), "10001", bank(), profile)
    assert found.status is Status.SUCCEEDED, found


class Keeps:
    """A model that keeps only the named texts, and records what it saw."""

    def __init__(self, *kept: str) -> None:
        self.kept = frozenset(kept)
        self.seen: list[tuple[str, ...]] = []

    def keep_words(self, texts: tuple[str, ...]) -> frozenset[str]:
        self.seen.append(texts)
        return self.kept & frozenset(texts)


def _refused(outcome: str):
    """The missing-record run, ending with ``outcome`` instead."""
    other = _missing()
    nodes = tuple(
        dataclasses.replace(node, outcome=outcome)
        if isinstance(node, ResultNode) and node.result is ResultKind.OUTCOME
        else node
        for node in other.nodes
    )
    return dataclasses.replace(other, nodes=nodes, outcomes=(outcome,))


def test_a_run_that_ends_with_another_outcome_teaches_nothing():
    adopted = adopt(_main(), _refused("permission_denied"), "record_not_found", None)
    assert adopted.capability is None
    assert adopted.note == "the run ended with an outcome different from its case"


def test_texts_about_a_record_that_exists_must_pass_the_models_veto():
    refused = _refused("record_ineligible")
    veto = Keeps()
    adopted = adopt(_main(), refused, "record_ineligible", veto)
    assert veto.seen == [("No member found",)]
    assert adopted.capability is None
    kept = adopt(_main(), refused, "record_ineligible", Keeps("No member found"))
    assert kept.capability is not None
    assert kept.texts == ("No member found",)


def test_texts_about_a_record_that_does_not_exist_need_no_veto():
    veto = Keeps()
    adopted = adopt(_main(), _missing(), "record_not_found", veto)
    assert veto.seen == []
    assert adopted.capability is not None


def test_a_check_of_a_value_the_second_run_kept_stays_with_that_run():
    """The second run proved its operator from a value it read and kept.

    That value exists only in the second run. The branch keeps the checks
    about the outcome, and the main capability proves its own operator.
    """
    other = _missing()
    kept = ValueIs(
        ref(RefKind.VARIABLE, "fact_1"),
        ref(RefKind.INPUT, "member_id"),
        TextMatch.CONTAINS,
    )
    other = dataclasses.replace(
        other,
        nodes=tuple(
            dataclasses.replace(node, checks=(*node.checks, kept))
            if isinstance(node, ResultNode)
            else node
            for node in other.nodes
        ),
    )
    plan = outcome_plan(_main(), other)
    assert plan is not None
    (branch,) = plan.branches
    assert all(not isinstance(condition, ValueIs) for condition in branch.when)
    assert outcome_texts(plan) == ("No member found",)


@pytest.mark.rule(15)
def test_a_not_found_branch_needs_the_records_row_to_be_absent():
    """Finding 7: discovery refused "not found" while a row showed the record.

    The found path uses an input to identify the record link in its next step.
    The learned branch also requires that link to find nothing. The two paths
    cannot both hold, and an incomplete observation satisfies neither path.
    """
    from computeruse.capability import (
        Absent,
        LocatorForm,
        RefKind,
        StructuralTarget,
        ref,
    )

    main = _main()
    link = StructuralTarget(
        "member_link",
        "/members/:id",
        LocatorForm.ACCESSIBILITY,
        "link",
        ref(RefKind.INPUT, "member_id"),
        "",
        None,
        None,
        (),
        None,
    )
    main = dataclasses.replace(
        main,
        targets=(*main.targets, link),
        nodes=tuple(
            dataclasses.replace(node, target="member_link")
            if node.node_id == "read_balance"
            else node
            for node in main.nodes
        ),
    )
    plan = outcome_plan(main, _missing())
    assert plan is not None
    (branch,) = plan.branches
    assert Absent("member_link") in branch.when
