"""The capability contract: serialization, static validation, and profile checks.

Every capability here is a synthetic fixture. These tests check only the
contract and do not drive a surface.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from replay_fakes import (
    MEMBER,
    PROFILE,
    go,
    limits,
    transfer_capability,
)

from computeruse.actions import Relation
from computeruse.capability import (
    SCHEMA_VERSION,
    ActionNode,
    Approval,
    Bound,
    CapabilityError,
    Destination,
    EdgeOrigin,
    Field,
    HelpReason,
    HumanNode,
    IssueCode,
    Match,
    Param,
    Present,
    RecordSpec,
    RefKind,
    RestrictionScope,
    ResultNode,
    SavedRestriction,
    Shows,
    StoragePermit,
    ValueType,
    VisualTarget,
    check_profile,
    dumps,
    effective_budgets,
    from_document,
    load_capability,
    loads,
    ref,
    template,
    to_document,
    validate,
)
from computeruse.policy import Source
from computeruse.profile import ActionKind, Limit, load_profile

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "capabilities"


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


def codes(capability) -> set[IssueCode]:
    return {issue.code for issue in validate(capability)}


def replace_node(capability, which, **changes):
    return dataclasses.replace(
        capability,
        nodes=tuple(
            dataclasses.replace(node, **changes) if node.node_id == which else node
            for node in capability.nodes
        ),
    )


def test_the_fixture_is_valid_against_the_contract_and_profile(profile) -> None:
    capability = transfer_capability()

    assert validate(capability) == ()
    assert check_profile(capability, profile) == ()


def test_a_capability_round_trips_through_json() -> None:
    capability = transfer_capability()

    assert loads(dumps(capability)) == capability


def test_the_synthetic_example_file_matches_the_fixture() -> None:
    """The example is regenerated from the fixture, so the two cannot drift."""
    stored = load_capability(EXAMPLE / "member_transfer.synthetic.json")

    assert stored == transfer_capability()
    assert stored.provenance.kind.value == "synthetic"


def test_an_unknown_schema_version_is_refused() -> None:
    document = to_document(transfer_capability())
    document["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(CapabilityError) as refused:
        from_document(document)

    assert {i.code for i in refused.value.issues} == {IssueCode.UNKNOWN_SCHEMA}


@pytest.mark.parametrize("older", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_an_older_artifact_is_refused_not_read_under_new_rules(older) -> None:
    """Summarize the contract changes introduced by each version.

    Versions 2 and 3 changed meanings. Version 6 added match modes, version 7
    added option matching, and version 8 added each check's purpose,
    cardinality, and row columns.
    """
    document = to_document(transfer_capability())
    document["schema_version"] = older

    with pytest.raises(CapabilityError) as caught:
        from_document(document)

    assert {issue.code for issue in caught.value.issues} == {IssueCode.UNKNOWN_SCHEMA}
    assert SCHEMA_VERSION == 10


def test_a_result_record_must_name_the_same_record() -> None:
    capability = transfer_capability()
    submit = capability.node("submit")
    assert isinstance(submit, ActionNode)
    assert submit.record is not None
    same = RecordSpec("member_shown", MEMBER, Relation.CONTAINER)
    other = RecordSpec("member_shown", ref(RefKind.INPUT, "amount"), Relation.CONTAINER)

    assert codes(replace_node(capability, "submit", result_record=same)) == set()
    assert IssueCode.INVALID_REFERENCE in codes(
        replace_node(capability, "submit", result_record=other)
    )
    assert IssueCode.INVALID_REFERENCE in codes(
        replace_node(capability, "open_transfer", result_record=same)
    )


@pytest.mark.parametrize(("prefix", "suffix"), [("NM", ""), ("", "1"), ("A", "B")])
def test_a_record_label_cannot_split_the_identifier(prefix, suffix) -> None:
    capability = transfer_capability()
    submit = capability.node("submit")
    assert isinstance(submit, ActionNode)
    assert submit.record is not None
    split = dataclasses.replace(submit.record, prefix=prefix, suffix=suffix)
    whole = dataclasses.replace(submit.record, prefix="Member: ", suffix=" (open)")

    assert IssueCode.INVALID_TARGET in codes(
        replace_node(capability, "submit", record=split)
    )
    assert codes(replace_node(capability, "submit", record=whole)) == set()


def test_only_a_navigation_takes_a_destination() -> None:
    destination = Destination("/members/:id", (Param("id", MEMBER),))
    clicking = replace_node(transfer_capability(), "search", destination=destination)

    assert IssueCode.INVALID_ROUTE in codes(clicking)


def test_a_human_effect_needs_the_operation_it_names() -> None:
    capability = transfer_capability()
    step = HumanNode(
        "confirm",
        "/members",
        HelpReason.MANUAL_STEP,
        None,
        "search_member",
        True,
        (go("read_balance", Present("member_field")),),
    )
    with_step = dataclasses.replace(
        capability,
        nodes=(*capability.nodes, step),
    )
    named = replace_node(with_step, "confirm", performs=ActionKind.CLICK)

    assert IssueCode.INVALID_ACTION in codes(with_step)
    assert IssueCode.INVALID_ACTION not in codes(named)


def test_an_unknown_node_type_is_refused() -> None:
    document = to_document(transfer_capability())
    document["nodes"][0]["type"] = "script"

    with pytest.raises(CapabilityError) as refused:
        from_document(document)

    assert IssueCode.UNKNOWN_VALUE in {i.code for i in refused.value.issues}


def test_an_unknown_field_is_refused_not_ignored() -> None:
    document = to_document(transfer_capability())
    document["nodes"][0]["run"] = "rm -rf /"

    with pytest.raises(CapabilityError) as refused:
        from_document(document)

    assert IssueCode.UNKNOWN_FIELD in {i.code for i in refused.value.issues}


def test_a_missing_field_is_refused_rather_than_defaulted() -> None:
    document = to_document(transfer_capability())
    del document["nodes"][0]["approval"]

    with pytest.raises(CapabilityError) as refused:
        from_document(document)

    assert IssueCode.MISSING_FIELD in {i.code for i in refused.value.issues}


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version": 1, "schema_version": 1}',
        '{"schema_version": NaN}',
        "not json",
    ],
)
def test_malformed_json_is_refused(text) -> None:
    with pytest.raises(CapabilityError):
        loads(text)


def test_diagnostics_do_not_echo_artifact_contents() -> None:
    document = to_document(transfer_capability())
    document["nodes"][0]["node_id"] = "Member 10001 Jane"
    document["capability_id"] = "Jane Doe 10001"

    with pytest.raises(CapabilityError) as refused:
        from_document(document)

    message = str(refused.value) + repr(refused.value.issues)
    assert "10001" not in message
    assert "Jane" not in message


def test_duplicate_node_ids_are_refused() -> None:
    capability = transfer_capability()
    doubled = replace_node(capability, "read_balance", node_id="search")

    assert IssueCode.DUPLICATE_IDENTIFIER in codes(doubled)


def test_a_missing_entry_is_refused() -> None:
    capability = dataclasses.replace(transfer_capability(), entry="nowhere")

    assert IssueCode.MISSING_ENTRY in codes(capability)


def test_a_reference_to_an_undeclared_input_is_refused() -> None:
    capability = replace_node(
        transfer_capability(), "enter_member", value=ref(RefKind.INPUT, "member")
    )

    assert IssueCode.UNKNOWN_REFERENCE in codes(capability)


def test_a_reference_to_an_unknown_target_is_refused() -> None:
    capability = replace_node(transfer_capability(), "search", target="nowhere")

    assert IssueCode.UNKNOWN_TARGET in codes(capability)


def test_a_secret_is_only_ever_typed() -> None:
    capability = replace_node(
        transfer_capability(),
        "search",
        transitions=(
            go(
                "read_balance",
                Shows("member_shown", ref(RefKind.SECRET, "teller_pin"), Match.EQUALS),
            ),
            go("not_found", Present("no_member")),
        ),
    )

    assert IssueCode.INVALID_REFERENCE in codes(capability)


def test_a_success_needs_completion_checks() -> None:
    capability = replace_node(transfer_capability(), "done", checks=())

    assert IssueCode.MISSING_COMPLETION_CHECK in codes(capability)


def test_an_output_missing_on_one_path_to_success_is_refused() -> None:
    capability = transfer_capability()
    required = dataclasses.replace(
        capability,
        outputs=(Field("balance", ValueType.DECIMAL, True, 1, 20, ()),),
    )
    assert validate(required) == ()

    skip = replace_node(
        required,
        "search",
        transitions=(
            go("open_transfer", Shows("member_shown", MEMBER, Match.EQUALS)),
            go("not_found", Present("no_member")),
        ),
    )
    skipped = dataclasses.replace(
        skip, nodes=tuple(n for n in skip.nodes if n.node_id != "read_balance")
    )

    assert IssueCode.MISSING_OUTPUT in codes(skipped)


def test_a_variable_assigned_on_only_one_branch_is_unavailable_after_they_meet() -> (
    None
):
    capability = transfer_capability()
    with_variable = dataclasses.replace(
        capability,
        variables=(Field("shown_balance", ValueType.DECIMAL, True, 1, 20, ()),),
    )
    shown = ref(RefKind.VARIABLE, "shown_balance")
    reads = replace_node(with_variable, "read_balance", into=shown)
    assert validate(reads) == ()

    uses = replace_node(
        reads,
        "enter_amount",
        requires=(Shows("balance", shown, Match.EQUALS),),
    )
    assert validate(uses) == ()

    branch = replace_node(
        uses,
        "search",
        transitions=(
            go("read_balance", Shows("member_shown", MEMBER, Match.EQUALS)),
            go("open_transfer", Present("no_member"), origin=EdgeOrigin.AUTHORED),
        ),
    )
    branch = dataclasses.replace(
        branch, nodes=tuple(n for n in branch.nodes if n.node_id != "not_found")
    )

    assert IssueCode.UNAVAILABLE_VARIABLE in codes(branch)


def test_a_bound_check_may_test_a_variable_that_may_be_missing() -> None:
    capability = transfer_capability()

    done = capability.node("done")
    assert isinstance(done, ResultNode)
    assert any(isinstance(c, Bound) for c in done.checks)
    assert validate(capability) == ()


def test_an_unbounded_cycle_is_refused_and_a_bounded_one_is_not() -> None:
    capability = replace_node(
        transfer_capability(),
        "search",
        transitions=(
            go("read_balance", Shows("member_shown", MEMBER, Match.EQUALS)),
            go("enter_member", Present("no_member"), origin=EdgeOrigin.AUTHORED),
        ),
    )
    capability = dataclasses.replace(
        capability,
        nodes=tuple(n for n in capability.nodes if n.node_id != "not_found"),
    )
    assert IssueCode.UNBOUNDED_CYCLE in codes(capability)

    bounded = replace_node(
        capability,
        "search",
        transitions=(
            go("read_balance", Shows("member_shown", MEMBER, Match.EQUALS)),
            go(
                "enter_member",
                Present("no_member"),
                origin=EdgeOrigin.AUTHORED,
                limit=2,
            ),
        ),
    )
    assert validate(bounded) == ()


def test_an_edge_that_skips_a_mandatory_step_is_refused() -> None:
    capability = replace_node(
        transfer_capability(),
        "enter_pin",
        transitions=(
            go("submit", Present("submit")),
            go("done", Present("submitted"), origin=EdgeOrigin.AUTHORED),
        ),
    )

    assert IssueCode.BYPASSES_INTERVENTION in codes(capability)


def test_an_unconditional_edge_must_be_the_only_edge() -> None:
    capability = replace_node(
        transfer_capability(),
        "read_balance",
        transitions=(go("open_transfer"), go("done", Present("submitted"))),
    )

    assert IssueCode.INVALID_TRANSITION in codes(capability)


def test_a_submission_without_a_check_is_refused() -> None:
    capability = replace_node(transfer_capability(), "submit", verify=())

    assert IssueCode.UNVERIFIED_ACTION in codes(capability)


def test_a_saved_restriction_cannot_be_skipped_or_denied() -> None:
    capability = transfer_capability()
    risky = SavedRestriction(
        RestrictionScope.TARGET,
        "/members/:id",
        (ActionKind.CLICK,),
        "transfer_link",
        "",
        Limit.RISKY,
        Source.FINDING,
    )
    assert IssueCode.BYPASSES_RESTRICTION in codes(
        dataclasses.replace(capability, restrictions=(risky,))
    )
    approved = replace_node(
        dataclasses.replace(capability, restrictions=(risky,)),
        "open_transfer",
        approval=Approval.EACH_RUN,
    )
    assert validate(approved) == ()

    denied = dataclasses.replace(risky, limit=Limit.DENY)
    assert IssueCode.DENIED_OPERATION in codes(
        dataclasses.replace(capability, restrictions=(denied,))
    )


def visual(capability, kind: ActionKind):
    from replay_fakes import png

    stored = template("button", png((40, 20), ((0, 0),)), StoragePermit.SYNTHETIC)
    return replace_node(
        dataclasses.replace(
            capability,
            templates=(stored,),
            targets=(
                *capability.targets,
                VisualTarget("painted", "/members", "button", ()),
            ),
        ),
        "enter_member",
        kind=kind,
        target="painted",
        value=MEMBER if kind is ActionKind.TYPE else None,
        verify=(Present("search"),) if kind is ActionKind.CLICK else (),
        effect="press" if kind is ActionKind.CLICK else None,
    )


def test_a_visual_target_may_only_be_clicked() -> None:
    capability = transfer_capability()

    assert IssueCode.UNSUPPORTED_TARGET in codes(visual(capability, ActionKind.TYPE))
    assert validate(visual(capability, ActionKind.CLICK)) == ()


def test_a_template_must_match_its_digest() -> None:
    capability = visual(transfer_capability(), ActionKind.CLICK)
    tampered = dataclasses.replace(
        capability,
        templates=(dataclasses.replace(capability.templates[0], sha256="0" * 64),),
    )

    assert IssueCode.INVALID_TEMPLATE in codes(tampered)


def test_a_capability_cannot_ask_for_what_the_profile_does_not_grant(
    tmp_path,
) -> None:
    capability = transfer_capability()
    narrower = PROFILE.replace("  type: {any: safe}\n", "").replace(
        "  teller_pin: {env: TELLER_PIN}\n", "  other: {env: OTHER}\n"
    )
    path = tmp_path / "narrow.yaml"
    path.write_text(narrower)

    found = {issue.code for issue in check_profile(capability, load_profile(path))}

    assert {IssueCode.NOT_GRANTED, IssueCode.UNDECLARED_SECRET} <= found


def test_a_record_bound_step_without_record_evidence_is_refused(profile) -> None:
    capability = replace_node(transfer_capability(), "submit", record=None)

    found = {issue.code for issue in check_profile(capability, profile)}

    assert IssueCode.RECORD_EVIDENCE_UNAVAILABLE in found


def test_a_denied_route_is_refused_by_the_profile(tmp_path) -> None:
    path = tmp_path / "deny.yaml"
    path.write_text(
        PROFILE.replace("deny_routes: []", "deny_routes: [/members/:id/transfer]")
    )

    found = check_profile(transfer_capability(), load_profile(path))

    assert IssueCode.ROUTE_NOT_PERMITTED in {issue.code for issue in found}


def test_the_profile_caps_the_capability_limits(profile) -> None:
    generous = transfer_capability(limits=limits(max_steps=900, max_wall_clock_s=9000))

    budgets = effective_budgets(generous, profile)

    assert budgets.max_steps == profile.budgets.max_steps
    assert budgets.max_wall_clock_s == profile.budgets.max_wall_clock_s


@pytest.mark.parametrize(
    ("kind", "good", "bad"),
    [
        (ValueType.DIGITS, "00123", "12a"),
        (ValueType.INTEGER, "-12", "1.5"),
        (ValueType.DECIMAL, "12.50", "12."),
        (ValueType.BOOLEAN, "true", "yes"),
        (ValueType.TEXT, "Jane", " Jane"),
    ],
)
def test_field_types(kind, good, bad) -> None:
    declared = Field("value", kind, True, 1, 10, ())

    assert declared.accepts(good)
    assert not declared.accepts(bad)


def test_persisted_capability_holds_no_invocation_values() -> None:
    text = dumps(transfer_capability())

    for value in ("10001", "250.00", "1204.50"):
        assert value not in text
    assert json.loads(text)["inputs"][0]["name"] == "member_id"


def test_results_do_not_use_outcome_names_they_do_not_declare() -> None:
    capability = replace_node(transfer_capability(), "not_found", outcome="gone")

    assert IssueCode.UNDECLARED_OUTCOME in codes(capability)


def test_navigation_refuses_permission_wildcards():
    capability = transfer_capability()
    node = capability.nodes[0]
    capability = replace_node(
        capability,
        node.node_id,
        kind=ActionKind.NAVIGATE,
        target=None,
        value=None,
        destination=Destination("/members/**", ()),
        verify=(Present("member_field"),),
    )
    assert IssueCode.INVALID_ROUTE in codes(capability)


@pytest.mark.parametrize(
    ("label", "text", "match", "dx", "valid"),
    [
        (False, "constant", "equals", 0, True),
        (False, "input", "contains", 0, True),
        (False, "constant", "contains", 0, False),
        (True, "constant", "equals", 0, False),
        (False, "constant", "equals", 900, False),
    ],
)
def test_a_text_target_is_a_line_a_reference_or_a_label_it_reads_after(
    label, text, match, dx, valid
) -> None:
    """A click aims at a line, or beside it. Only a read takes a label's value."""
    from computeruse.capability import TextTarget, constant

    capability = transfer_capability()
    painted = TextTarget(
        "search",
        "/members",
        (),
        ref(RefKind.INPUT, "member_id") if text == "input" else constant("Search"),
        Match(match),
        label=label,
        dx=dx,
    )
    changed = dataclasses.replace(
        capability,
        targets=tuple(
            painted if target.target_id == "search" else target
            for target in capability.targets
        ),
    )
    issues = validate(changed)
    assert (issues == ()) is valid, issues


@pytest.mark.rule(6)
def test_a_saved_screen_restriction_leaves_replay_looks_alone() -> None:
    """Finding 43: replay's own looks are not gated by a restriction on input."""
    from computeruse.capability import (
        ActionNode,
        Approval,
        RestrictionScope,
        SavedRestriction,
        strictest,
    )
    from computeruse.policy import LOOKS, Source

    inputs = tuple(kind for kind in ActionKind if kind not in LOOKS)
    saved = SavedRestriction(
        RestrictionScope.ROUTE, "/", inputs, None, "", Limit.RISKY, Source.FINDING
    )

    def node(kind, target="t1"):
        return ActionNode(
            "n1", kind, "/", target, None, None, None, None, None, None,
            Approval.NONE, False, (), (), (),
        )  # fmt: skip

    assert strictest((saved,), node(ActionKind.CLICK)) is Limit.RISKY
    assert strictest((saved,), node(ActionKind.OBSERVE, None)) is None
    assert strictest((saved,), node(ActionKind.READ)) is None


@pytest.mark.rule(6)
def test_a_step_aimed_by_pixels_is_covered_by_its_routes_restrictions() -> None:
    """Finding 47: as in discovery, pixels cannot prove a step is another control."""
    from computeruse.capability import (
        ActionNode,
        Approval,
        RestrictionScope,
        SavedRestriction,
        strictest,
    )
    from computeruse.policy import Source

    saved = SavedRestriction(
        RestrictionScope.TARGET,
        "/",
        (ActionKind.CLICK,),
        "t9",
        "",
        Limit.RISKY,
        Source.FINDING,
    )
    click = ActionNode(
        "n1", ActionKind.CLICK, "/", "t1", None, None, None, None, None, None,
        Approval.NONE, False, (), (), (),
    )  # fmt: skip
    assert strictest((saved,), click) is None
    assert strictest((saved,), click, pixel=True) is Limit.RISKY


@pytest.mark.rule(5)
def test_a_change_is_sent_once_across_graph_visits(tmp_path) -> None:
    # Graduated from tests/test_known_gaps.py (R6). The validator refuses the
    # cycle. Validation keeps the rule, and replay never sees the cycle.
    from replay_fakes import (
        PROFILE,
        App,
        Clock,
        Page,
        People,
        _ax,
        _node,
        button,
        status,
    )

    from computeruse.capability import AtRoute, ResultKind, check_profile
    from computeruse.replay import MemoryReplayLog, replay

    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    profile = load_profile(path)

    def at_desk(targets, nodes, **changes):
        base = transfer_capability()
        return dataclasses.replace(
            base,
            application=dataclasses.replace(
                base.application, entry_route="/desk", markers=(AtRoute("/desk"),)
            ),
            inputs=(),
            outputs=(),
            variables=(),
            secrets=(),
            outcomes=(),
            targets=tuple(targets),
            nodes=tuple(nodes),
            entry=nodes[0].node_id,
            **changes,
        )

    def execute(capability, app, profile):
        assert validate(capability) == ()
        assert check_profile(capability, profile) == ()
        clock = Clock()
        result = replay(
            capability,
            {},
            profile=profile,
            surface=app,
            control=People([]).control(clock),
            log=MemoryReplayLog(),
            clock=clock,
        )
        return result, None

    commit = _ax("commit", "/desk", "button", "Commit")
    again = _ax("again", "/desk", "status", "Again")
    finished = _ax("finished", "/desk", "status", "Finished")
    capability = at_desk(
        [commit, again, finished],
        [
            _node(
                "commit",
                ActionKind.CLICK,
                "/desk",
                target="commit",
                effect="create_entry",
                verify=(AtRoute("/desk"),),
                transitions=(
                    go("commit", Present("again"), limit=1),
                    go("done", Present("finished")),
                ),
            ),
            ResultNode("done", ResultKind.SUCCESS, "", (Present("finished"),)),
        ],
    )
    if validate(capability):
        return  # refusing the cycle at validation also keeps the rule
    app = App(
        {
            "first": Page("/desk", (button("Commit"), status("Again"))),
            "second": Page("/desk", (button("Commit"), status("Again"))),
            "third": Page("/desk", (status("Finished"),)),
        },
        "first",
        {
            ("first", ActionKind.CLICK, "Commit"): "second",
            ("second", ActionKind.CLICK, "Commit"): "third",
        },
    )
    execute(capability, app, profile)
    assert app.acted_on("Commit") <= 1


@pytest.mark.rule(5)
def test_only_an_authored_retry_may_loop_through_a_change() -> None:
    """A loop back to a click could send its change twice (R6).

    A bounded retry written by a person is allowed. A loop created by the
    recording is not. No loop may pass through a step that needs approval on
    every run.
    """
    retry = go(
        "enter_member", Present("no_member"), origin=EdgeOrigin.AUTHORED, limit=2
    )
    loop = dataclasses.replace(retry, origin=EdgeOrigin.OBSERVED)
    found = go("read_balance", Shows("member_shown", MEMBER, Match.EQUALS))

    def looping(edge, **changes):
        capability = replace_node(
            transfer_capability(), "search", transitions=(found, edge), **changes
        )
        return dataclasses.replace(
            capability,
            nodes=tuple(n for n in capability.nodes if n.node_id != "not_found"),
        )

    assert IssueCode.REPEATING_CHANGE not in codes(looping(retry))
    assert IssueCode.REPEATING_CHANGE in codes(looping(loop))
    careful = looping(retry, approval=Approval.EACH_RUN, mandatory=True)
    assert IssueCode.REPEATING_CHANGE in codes(careful)
