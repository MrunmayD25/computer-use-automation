"""Preserve trace boundaries through capability export with scripted events."""

import dataclasses
from pathlib import Path

import pytest
from replay_fakes import png

from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    Observation,
    ObservationMode,
    ObservationStatus,
    Outcome,
    PageState,
)
from computeruse.capability import ActionNode, HumanNode, dumps
from computeruse.decider import CheckKind, ResultCheck, Task, TaskRecord
from computeruse.escalation import (
    Ask,
    Handoff,
    HandoffOutcome,
    Interruption,
    InterventionRequest,
    Mode,
    Trigger,
)
from computeruse.loop import CheckResult, Ending, RunResult, Verification
from computeruse.manual import (
    ManualEvent,
    ManualKind,
    ManualSegment,
    ManualStep,
    Requirement,
)
from computeruse.profile import ActionKind, load_profile
from computeruse.recorder import Gap
from computeruse.recording import DiscoveryTrace, field


@pytest.fixture
def trace_case():
    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(AxNode("button", "Post", tag="button"),),
    )
    after = dataclasses.replace(
        before,
        observation_id="after",
        nodes=(
            AxNode("heading", "Posted", tag="h1"),
            AxNode("text", "10001", tag="dd", attributes=(("id", "member"),)),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        2,
        Action(ActionKind.CLICK, AxLocator("button", "Post")),
        ActionResult(Outcome.UNCERTAIN, state),
        state.location,
        before,
    )
    request = InterventionRequest(
        Trigger.DELIVERY_UNCERTAIN,
        "Post for 10001",
        profile.profile_id,
        2,
        "/members",
        "Check delivery",
        10,
    )
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(CheckKind.RECORD, AxLocator("text", "10001"), "10001"),
                "10001",
                True,
            ),
        ),
        task=Task(records=(TaskRecord("member", "10001"),)),
    )
    return profile, trace, request, after, result


def build(case, segments=(), inputs=None):
    profile, trace, request, after, result = case
    trace.handoff(request, Handoff(HandoffOutcome.RESUMED), segments)
    trace.look(after)
    return trace.build(
        result,
        profile=profile,
        inputs={"member": "10001"} if inputs is None else inputs,
        capability_id="scripted_post",
        run="scripted",
        safe_text=frozenset({"Post", "Posted", "member"}),
    )


def test_uncertain_action_is_replaced_by_one_human_node(trace_case):
    recording = build(trace_case)
    assert recording.complete, (recording.issues, recording.artifact_issues)
    nodes = recording.capability.nodes
    assert not any(
        isinstance(n, ActionNode) and n.kind is ActionKind.CLICK for n in nodes
    )
    people = [n for n in nodes if isinstance(n, HumanNode)]
    assert len(people) == 1
    assert people[0].performs is ActionKind.CLICK


def test_forbidden_manual_work_cannot_be_bridged_in_export(trace_case):
    segment = ManualSegment(
        "scripted",
        "iv-1",
        Mode.DISCOVERY,
        Ask.PERSON,
        Trigger.DELIVERY_UNCERTAIN,
        True,
        (ManualStep(ManualEvent(1, 1, ManualKind.CLICK, True), Requirement.FORBIDDEN),),
        (),
        Interruption.UNCERTAIN,
    )
    recording = build(trace_case, (segment,))
    assert not recording.complete
    assert Gap.DENIED_STEP in {issue.gap for issue in recording.issues}


def test_record_identifier_must_be_a_declared_input(trace_case):
    recording = build(trace_case, inputs={})
    assert not recording.complete
    assert Gap.UNBOUND_RECORD in {issue.gap for issue in recording.issues}


def test_surface_failure_stays_an_uncertain_execution_event(trace_case):
    from fakes import Screen, ScriptedDecider, ScriptedEscalator, ScriptedSurface

    from computeruse.actions import ObservationRequest
    from computeruse.decider import Observe, Propose
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover
    from computeruse.recording import ActionTaken
    from computeruse.surface import SurfaceError

    class FailedSurface(ScriptedSurface):
        def act(self, action, *, expect=None):
            del action, expect
            raise SurfaceError("connection lost after dispatch")

    profile = trace_case[0]
    surface = FailedSurface([Screen("http://127.0.0.1:8787/members")])
    trace = DiscoveryTrace()
    result = discover(
        "Search",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [
                Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                Propose(Action(ActionKind.CLICK, AxLocator("button", "Search"))),
            ]
        ),
        journal=MemoryJournal(),
        clock=lambda: 0,
        escalator=ScriptedEscalator([]),
        trace=trace,
    )
    assert result.ending is Ending.FAILED
    sent = [e for e in trace.entries if isinstance(e, ActionTaken)]
    assert len(sent) == 1
    assert sent[0].result.outcome is Outcome.SURFACE_ERROR
    recording = trace.build(
        result, profile=profile, inputs={}, capability_id="failed", run="scripted"
    )
    assert not recording.complete
    assert Gap.UNCERTAIN_DELIVERY in {issue.gap for issue in recording.issues}


def test_unapproved_dynamic_page_text_never_becomes_a_saved_constant(trace_case):
    from computeruse.capability import dumps

    profile, trace, request, after, result = trace_case
    private = "Alex Example address 123 Test Lane"
    after = dataclasses.replace(
        after, nodes=(dataclasses.replace(after.nodes[0], name=private), after.nodes[1])
    )
    recording = build((profile, trace, request, after, result))
    assert recording.complete, recording.issues
    assert private not in dumps(recording.capability)


@pytest.mark.parametrize("uncertain", [False, True])
def test_model_effect_text_is_never_saved_without_a_storage_declaration(
    trace_case, uncertain
):
    from computeruse.capability import dumps
    from computeruse.recording import ActionTaken

    profile, trace, request, after, result = trace_case
    private = "submit_for_private_customer_10001"
    entry = trace.entries[1]
    assert isinstance(entry, ActionTaken)
    trace.entries[1] = dataclasses.replace(
        entry,
        action=dataclasses.replace(entry.action, effect=private),
        result=dataclasses.replace(
            entry.result, outcome=Outcome.UNCERTAIN if uncertain else Outcome.OK
        ),
    )
    if not uncertain:
        trace.verified(
            2,
            (ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),),
            after,
        )
        trace.look(after)
        recording = trace.build(
            result,
            profile=profile,
            inputs={"member": "10001"},
            capability_id="scripted_post",
            run="scripted",
            safe_text=frozenset({"Post", "Posted", "member"}),
        )
    else:
        recording = build((profile, trace, request, after, result))
    assert recording.complete, (recording.issues, recording.artifact_issues)
    assert recording.capability is not None
    assert private not in dumps(recording.capability)
    if not uncertain:
        assert any(
            isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
            for node in recording.capability.nodes
        )


def test_unapproved_target_text_refuses_export(trace_case):
    profile, trace, request, after, result = trace_case
    private = "Alex Example address 123 Test Lane"
    after = dataclasses.replace(
        after,
        nodes=(
            after.nodes[0],
            dataclasses.replace(after.nodes[1], attributes=(("id", private),)),
        ),
    )
    recording = build((profile, trace, request, after, result))
    assert recording.complete
    from computeruse.capability import dumps

    assert private not in dumps(recording.capability)


@pytest.mark.rule(3)
def test_final_checks_preserve_live_record_evidence(trace_case):
    from computeruse.actions import RecordEvidence, Relation
    from computeruse.capability import ResultNode, Shows

    profile, trace, request, after, result = trace_case
    identity = AxLocator("text", "10001")
    evidence = RecordEvidence(identity, "10001", Relation.CONTAINER)
    checked = dataclasses.replace(result.checks[0].check, record=evidence)
    result = dataclasses.replace(
        result, checks=(dataclasses.replace(result.checks[0], check=checked),)
    )
    recording = build((profile, trace, request, after, result))
    assert recording.complete, recording.issues
    final = next(
        node for node in recording.capability.nodes if isinstance(node, ResultNode)
    )
    check = final.checks[0]
    assert isinstance(check, Shows)
    assert check.record is not None
    assert check.record.value.name == "member"


def test_absent_next_target_never_verifies_a_navigation(trace_case):
    from computeruse.capability import IssueCode
    from computeruse.recording import ActionTaken

    profile, trace, _, after, result = trace_case
    entry = trace.entries[1]
    assert isinstance(entry, ActionTaken)
    trace.entries[1] = dataclasses.replace(
        entry, result=dataclasses.replace(entry.result, outcome=Outcome.OK)
    )
    moved = dataclasses.replace(
        after, page_state=PageState("http://127.0.0.1:8787/members/10001")
    )
    trace.action(
        3,
        Action(ActionKind.READ, AxLocator("combobox", "Missing")),
        ActionResult(Outcome.NOT_FOUND, moved.page_state),
        moved.location,
        moved,
    )
    trace.look(moved)
    recording = trace.build(
        result,
        profile=profile,
        inputs={"member": "10001"},
        capability_id="missing_checkpoint",
        run="scripted",
        safe_text=frozenset({"Post", "Posted", "Missing", "member"}),
    )
    assert not recording.complete
    assert IssueCode.UNVERIFIED_ACTION in {
        issue.code for issue in recording.artifact_issues
    }


def test_custom_role_text_needs_explicit_storage_permission(trace_case):
    profile, trace, request, after, result = trace_case
    private = "customer-private-account-123"
    after = dataclasses.replace(
        after,
        nodes=(after.nodes[0], AxNode(private, "10001", tag="custom-value")),
    )
    check = dataclasses.replace(
        result.checks[0].check, target=AxLocator(private, "10001")
    )
    result = dataclasses.replace(
        result, checks=(dataclasses.replace(result.checks[0], check=check),)
    )
    recording = build((profile, trace, request, after, result))
    assert not recording.complete
    assert recording.capability is None


def test_failed_completion_fact_does_not_crash_export(trace_case):
    from computeruse.decider import FactRef

    profile, trace, _, _, result = trace_case
    result = dataclasses.replace(
        result,
        ending=Ending.HANDED_OFF,
        verification=None,
        outputs={},
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.RESULT, FactRef("status"), "Active", output="status"
                ),
                None,
                False,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"member": "10001"},
        capability_id="unfinished",
        run="scripted",
    )
    assert not recording.complete
    assert recording.capability is None


@pytest.mark.rule(3)
def test_a_final_check_whose_record_cannot_be_saved_is_a_gap(trace_case, monkeypatch):
    from computeruse.actions import RecordEvidence, Relation
    from computeruse.recording import _Export

    profile, trace, request, after, result = trace_case
    identity = AxLocator("text", "10001")
    evidence = RecordEvidence(identity, "10001", Relation.CONTAINER)
    checked = dataclasses.replace(result.checks[0].check, record=evidence)
    result = dataclasses.replace(
        result, checks=(dataclasses.replace(result.checks[0], check=checked),)
    )
    # A source no saved target can name, such as a label equal to an output.
    monkeypatch.setattr(_Export, "_record", lambda *_: None)
    recording = build((profile, trace, request, after, result))
    assert not recording.complete
    assert any(issue.gap is Gap.UNREPRESENTABLE_CHECK for issue in recording.issues)


def _collected(case, outputs=None):
    profile, trace, request, after, result = case
    if outputs is not None:
        result = dataclasses.replace(result, outputs=outputs)
    trace.handoff(request, Handoff(HandoffOutcome.RESUMED), ())
    trace.look(after)
    return trace.build(
        result,
        profile=profile,
        inputs={"member": "10001"},
        capability_id="scripted_post",
        run="scripted",
        collect=True,
    )


def test_collecting_keeps_unconfirmed_texts_as_candidates_with_their_place(
    trace_case,
):
    from computeruse.recorder import Place

    recording = _collected(trace_case)
    assert recording.complete, (recording.issues, recording.artifact_issues)
    places = {item.text: item.places for item in recording.candidates}
    # The heading is not a control's own label, so only a comparison or a
    # person may confirm it. The model never sees it.
    assert places["Posted"] == frozenset({Place.RECORD})
    assert all(item.text != "10001" for item in recording.candidates)


def test_the_first_screen_is_checked_only_by_confirmed_text(trace_case):
    """Found in a virtualized run: a customer's name became a saved check.

    The application opens with a member already shown. Its first screen reads
    the same for every record, so comparison with a second record incorrectly
    confirmed the member's name as website text.
    """
    profile, _, request, after, result = trace_case
    state = PageState("http://127.0.0.1:8787/members")
    first = Observation(
        "first",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(
            AxNode("heading", "Alex Morgan", tag="h1"),
            AxNode("button", "Post", tag="button"),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(first)
    trace.action(
        2,
        Action(ActionKind.CLICK, AxLocator("button", "Post")),
        ActionResult(Outcome.UNCERTAIN, state),
        state.location,
        first,
    )
    recording = _collected((profile, trace, request, after, result))

    assert recording.complete, (recording.issues, recording.artifact_issues)
    assert "Alex Morgan" not in {item.text for item in recording.candidates}
    assert "Alex Morgan" not in dumps(recording.capability)


@pytest.mark.rule(8)
@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("source", ["input", "output", "fact", "secret", "comparison"])
def test_known_private_values_override_entry_markers_and_candidates(
    trace_case, collect, source
):
    from computeruse.actions import DomAttribute, DomLocator
    from computeruse.decider import Fact

    profile, _, request, after, result = trace_case
    private = "PRIVATE-TRACE-7462"
    exposed = f"Receipt {private.lower()} ready"
    before = dataclasses.replace(
        after,
        observation_id="before",
        nodes=(
            AxNode("heading", exposed, tag="h1"),
            AxNode("button", "Post", tag="button"),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        2,
        Action(ActionKind.CLICK, AxLocator("button", "Post")),
        ActionResult(Outcome.UNCERTAIN, before.page_state),
        before.page_state.location,
        before,
    )
    inputs = {"member": "10001"}
    excluded = ()
    if source == "input":
        inputs["note"] = private
    elif source == "fact":
        trace.kept(Fact("receipt", private, 2, "/members"))
    elif source == "secret":
        excluded = (private,)
    elif source == "comparison":
        profile = dataclasses.replace(profile, confirmation={"member": private})
    else:
        output = DomLocator("dd", DomAttribute.ID, "receipt")
        after = dataclasses.replace(
            after,
            nodes=(
                *after.nodes,
                AxNode("text", private, tag="dd", attributes=(("id", "receipt"),)),
            ),
        )
        result = dataclasses.replace(
            result,
            outputs={"receipt": private},
            checks=(
                *result.checks,
                CheckResult(
                    ResultCheck(CheckKind.RESULT, output, private, output="receipt"),
                    private,
                    True,
                ),
            ),
        )
    trace.handoff(request, Handoff(HandoffOutcome.RESUMED), ())
    trace.look(after)
    if source == "output":
        trace.action(
            3,
            Action(ActionKind.READ, output),
            ActionResult(Outcome.OK, after.page_state, extracted=private),
            after.page_state.location,
            after,
        )
    recording = trace.build(
        result,
        profile=profile,
        inputs=inputs,
        capability_id="scripted_post",
        run="scripted",
        safe_text=frozenset({"Post", "Posted", "member", "receipt", exposed}),
        collect=collect,
        excluded=excluded,
    )

    assert recording.complete, (recording.issues, recording.artifact_issues)
    assert recording.capability is not None
    assert private.casefold() not in dumps(recording.capability).casefold()
    assert exposed not in {candidate.text for candidate in recording.candidates}


def test_a_click_is_proved_by_the_input_it_brings_onto_the_screen():
    """Use the operator shown in the header to prove the click.

    No control of the header's role held the operator before the click. The
    header is saved by that input alone, never by the name beside it.
    """
    from computeruse.capability import Match, RefKind, Shows, StructuralTarget
    from computeruse.decider import Match as ClaimMatch

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(
            AxNode("text", "Alex Kim (OP0001)", tag="span"),
            AxNode("menuitem", "Casey Patel (OP0002)", tag="span"),
            AxNode("button", "Messages", tag="button"),
        ),
    )
    after = dataclasses.replace(
        before,
        observation_id="after",
        nodes=(
            AxNode("text", "Casey Patel (OP0002)", tag="span"),
            AxNode("button", "Messages", tag="button"),
        ),
    )
    header = AxLocator("text", "Casey Patel (OP0002)")
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("menuitem", "Casey Patel (OP0002)")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.REQUIREMENT,
                    header,
                    "OP0002",
                    ClaimMatch.CONTAINS,
                    requirement="operator",
                ),
                "Casey Patel (OP0002)",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"operator": "OP0002"},
        capability_id="operator_switch",
        run="scripted",
        safe_text=frozenset({"Messages", "text", "menuitem"}),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    chosen = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
    )
    assert len(chosen.verify) == 1
    proof = chosen.verify[0]
    assert isinstance(proof, Shows)
    assert proof.match is Match.CONTAINS
    assert proof.value.kind is RefKind.INPUT
    shown = capability.target(proof.target)
    assert isinstance(shown, StructuralTarget)
    assert shown.role == "text"
    assert shown.match is Match.CONTAINS
    assert "Casey Patel" not in dumps(capability)


def _grid(with_slot: bool = True) -> Observation:
    """Two table rows whose status cells carry ids built from each row's record."""
    from computeruse.actions import Scope, ScopeKind

    def cell(key: str, member: str, status: str) -> AxNode:
        return AxNode(
            "gridcell",
            status,
            tag="div",
            attributes=(("id", f"cell-{key}-status"),),
            scope=Scope(ScopeKind.ROW, f"{member} Jordan {status}"),
            row=f"row-{key}",
            slot="div|Status" if with_slot else "",
        )

    return Observation(
        "grid",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState("http://127.0.0.1:8787/members"),
        nodes=(cell("54", "10001", "active"), cell("55", "10002", "closed")),
    )


def test_a_cell_in_a_repeated_row_is_saved_by_its_column_never_its_id():
    from computeruse.actions import DomAttribute, DomLocator, ScopeKind
    from computeruse.recorder import TargetSample
    from computeruse.recording import target_sample

    seen = _grid()
    picked = DomLocator("div", DomAttribute.ID, "cell-54-status")
    sample = target_sample(
        picked,
        "/members",
        seen,
        inputs=("10001",),
        safe_text=frozenset({"div|Status", "cell-54-status"}),
    )
    assert isinstance(sample, TargetSample)
    assert sample.attribute is DomAttribute.SLOT
    assert sample.value == "div|Status"
    assert sample.scope is not None
    assert sample.scope.kind is ScopeKind.ROW
    assert "10001" in sample.scope.name


def test_a_cell_in_a_repeated_row_without_a_column_is_not_saved_by_its_id():
    from computeruse.actions import DomAttribute, DomLocator
    from computeruse.recorder import ScreenSample
    from computeruse.recording import target_sample

    picked = DomLocator("div", DomAttribute.ID, "cell-54-status")
    sample = target_sample(
        picked,
        "/members",
        _grid(with_slot=False),
        inputs=("10001",),
        safe_text=frozenset({"cell-54-status"}),
    )
    assert isinstance(sample, ScreenSample)


def _clicked(opens_menu: bool):
    """Click a button, then read a status that was already on the screen."""
    from computeruse.decider import Match as ClaimMatch

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    ready = AxNode("status", "Ready", tag="p")
    menu = AxNode("button", "Menu", tag="button")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(menu, ready),
    )
    opened = (AxNode("menuitem", "Settings", tag="a"),) if opens_menu else ()
    after = dataclasses.replace(
        before, observation_id="after", nodes=(menu, *opened, ready)
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Menu")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    trace.action(
        2,
        Action(ActionKind.READ, AxLocator("status", "Ready")),
        ActionResult(Outcome.OK, state, extracted="Ready"),
        state.location,
        after,
    )
    trace.look(after)
    check = ResultCheck(
        CheckKind.STATE,
        AxLocator("status", "Ready"),
        "Ready",
        ClaimMatch.EQUALS,
    )
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        verification=Verification.EXECUTOR,
        checks=(CheckResult(check, "Ready", True),),
    )
    return trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="clicked",
        run="scripted",
        safe_text=frozenset({"Menu", "Settings", "Ready", "menuitem", "status"}),
    )


def test_a_click_is_proved_by_the_interface_control_it_brings_onto_the_screen():
    from computeruse.capability import Present, StructuralTarget

    recording = _clicked(opens_menu=True)
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    click = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
    )
    proof = click.verify[0]
    assert isinstance(proof, Present)
    shown = capability.target(proof.target)
    assert isinstance(shown, StructuralTarget)
    assert shown.role == "menuitem"


@pytest.mark.rule(4)
def test_an_unchanged_next_target_does_not_remove_the_previous_click():
    recording = _clicked(opens_menu=False)
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    assert any(
        isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
        for node in capability.nodes
    )


def _persons_click(label: tuple[str, bool] | None):
    """A person clicks Open member once, and the run labels it or not."""
    from computeruse.actions import Operation

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    button = AxNode("button", "Open member", tag="button")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(button,),
    )
    after = dataclasses.replace(
        before, observation_id="after", nodes=(AxNode("heading", "Member", tag="h1"),)
    )
    locator = AxLocator("button", "Open member")
    click = ManualEvent(
        1,
        1.0,
        ManualKind.CLICK,
        True,
        route="/members",
        target=locator,
        location=state.location,
        operation=Operation.of(
            Action(ActionKind.CLICK, locator), "/members", node=button
        ),
    )
    segment = ManualSegment(
        "scripted",
        "iv-1",
        Mode.DISCOVERY,
        Ask.PERSON,
        Trigger.NO_PROGRESS,
        True,
        (ManualStep(click, Requirement.VALIDATE_AND_BIND),),
        (),
        Interruption.NONE,
    )
    request = InterventionRequest(
        Trigger.NO_PROGRESS,
        "Open the member",
        profile.profile_id,
        1,
        "/members",
        "",
        10,
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.handoff(
        request,
        Handoff(HandoffOutcome.RESUMED, intervention="iv-1", changed=True),
        (segment,),
    )
    if label is not None:
        trace.labelled("iv-1", label[0], careful=label[1])
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        2,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(CheckKind.STATE, AxLocator("heading", "Member"), "Member"),
                "Member",
                True,
            ),
        ),
    )
    return trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="persons_click",
        run="scripted",
        safe_text=frozenset({"Open member", "Member", "open_member"}),
    )


def test_a_persons_labelled_click_becomes_an_automated_step():
    from computeruse.capability import Approval

    recording = _persons_click(("open_member", False))
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    assert not any(isinstance(node, HumanNode) for node in capability.nodes)
    click = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
    )
    assert click.effect == "open_member"
    assert click.approval is Approval.NONE
    assert click.verify


def test_a_risky_label_automates_the_click_only_with_approval_each_run():
    from computeruse.capability import Approval

    recording = _persons_click(("open_member", True))
    capability = recording.capability
    assert capability is not None
    click = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
    )
    assert click.approval is Approval.EACH_RUN


def test_an_unlabelled_click_stays_a_persons_step():
    recording = _persons_click(None)
    capability = recording.capability
    assert capability is not None
    assert any(isinstance(node, HumanNode) for node in capability.nodes)


def test_a_not_found_run_is_recorded_under_the_output_contract():
    """The run ends on a search that matched nothing, and returns no outputs."""
    from computeruse.capability import ResultKind, ResultNode
    from computeruse.contract import load_contract
    from computeruse.decider import Match as ClaimMatch
    from computeruse.decider import TaskRecord

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    field = AxNode("textbox", "Search", value="NM999998", tag="input")
    button = AxNode("button", "Search", tag="button")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(field, button),
    )
    empty = AxNode("status", "No members match your search.", tag="p")
    after = dataclasses.replace(
        before, observation_id="after", nodes=(field, button, empty)
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Search")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    nothing = ResultCheck(
        CheckKind.STATE,
        AxLocator("status", "No members match your search."),
        "No members match your search.",
        ClaimMatch.EQUALS,
    )
    searched = ResultCheck(CheckKind.RECORD, AxLocator("textbox", "Search"), "NM999998")
    result = RunResult(
        Ending.COMPLETED,
        2,
        "the page showed the outcome record_not_found",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(nothing, "No members match your search.", True),
            CheckResult(searched, "NM999998", True),
        ),
        task=Task(records=(TaskRecord("member", "NM999998"),)),
        outcome="record_not_found",
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"member_id": "NM999998", "operator_id": "OP0002"},
        capability_id="member_status",
        run="scripted",
        safe_text=frozenset({"Search", "No members match your search."}),
        contract=load_contract(Path("evaluation/sites/member-status.contract.json")),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    ending = [n for n in capability.nodes if isinstance(n, ResultNode)]
    assert [(n.result, n.outcome) for n in ending] == [
        (ResultKind.OUTCOME, "record_not_found")
    ]
    click = next(n for n in capability.nodes if isinstance(n, ActionNode))
    assert click.verify


def _window(*names: str) -> Observation:
    return Observation(
        "window",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState("http://127.0.0.1:8787/members"),
        nodes=tuple(AxNode("button", name, tag="button") for name in names),
    )


def test_a_control_matched_by_its_input_must_be_the_only_one_holding_it():
    from computeruse.recorder import ScreenSample, TargetSample
    from computeruse.recording import target_sample

    # A window's buttons all carry the record in their names.
    buttons = _window("Minimize Members 10001", "Close Members 10001")
    words = frozenset({"Minimize Members 10001", "Close Members 10001"})
    close = AxLocator("button", "Close Members 10001")
    refused = target_sample(
        close, "/members", buttons, inputs=("10001",), safe_text=words
    )
    assert isinstance(refused, ScreenSample)
    # One button holding it is still found again by the input alone.
    alone = _window("Open member 10001", "Refresh")
    opened = AxLocator("button", "Open member 10001")
    kept = target_sample(
        opened, "/members", alone, inputs=("10001",), safe_text=frozenset()
    )
    assert isinstance(kept, TargetSample)


@pytest.mark.parametrize(
    ("output_name", "declared_output", "saved_output"),
    [
        ("private_output_canary_7462", None, "output_1"),
        ("account_number", "account_number", "account_number"),
        ("private_output_canary_7462", "output_1", None),
    ],
)
def test_a_link_named_by_a_kept_value_is_saved_by_that_value(
    output_name: str, declared_output: str | None, saved_output: str | None
) -> None:
    """A run reads the new record's number, then opens the record by its link.

    The recorder saves the link using the output that holds the number. Replay
    then opens the record it created rather than the record discovery created.
    """
    from computeruse.actions import DomAttribute, DomLocator
    from computeruse.capability import RefKind, StructuralTarget
    from computeruse.contract import Contract
    from computeruse.decider import Fact, FactRef

    fact_name = "private_fact_canary_9821"
    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    shown = DomLocator("dd", DomAttribute.ID, "reference")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(
            AxNode("text", "AC0000165", tag="dd", attributes=(("id", "reference"),)),
            AxNode("link", "AC0000165", tag="a"),
        ),
    )
    after = dataclasses.replace(
        before, observation_id="after", nodes=(AxNode("heading", "Posted", tag="h1"),)
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.READ, shown),
        ActionResult(Outcome.OK, state, "AC0000165"),
        state.location,
        before,
    )
    trace.kept(Fact(fact_name, "AC0000165", 1, "/members", shown, may_change=False))
    trace.action(
        2,
        Action(ActionKind.CLICK, AxLocator("link", "AC0000165")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        outputs={output_name: "AC0000165"},
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.RESULT,
                    FactRef(fact_name),
                    "AC0000165",
                    output=output_name,
                ),
                "AC0000165",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="open_created",
        run="scripted",
        safe_text=frozenset({"Posted", "link", "text", "reference"}),
        contract=Contract(2, (), (field(declared_output),))
        if declared_output
        else None,
    )
    if saved_output is None:
        assert not recording.complete
        assert recording.capability is None
        assert Gap.INVALID_CAPABILITY in {issue.gap for issue in recording.issues}
        return
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    opened = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
    )
    link = capability.target(opened.target or "")
    assert isinstance(link, StructuralTarget)
    assert link.name is not None
    assert (link.name.kind, link.name.name) == (RefKind.OUTPUT, saved_output)
    assert [output.name for output in capability.outputs] == [saved_output]
    serialized = dumps(capability)
    assert "AC0000165" not in serialized
    assert "private_output_canary_7462" not in serialized
    assert fact_name not in serialized


@pytest.mark.rule(3, 9, 11)
@pytest.mark.parametrize(
    ("scenario", "line"),
    [
        ("variable", "AC0000165."),
        ("recognized_case", "ac0000165."),
        ("output", "AC0000165."),
        ("input_collision", "AC0000165."),
        ("repeated", "AC0000165 AC0000165"),
        ("before_read", "AC0000165."),
        ("evolving_fact", "AC0000166."),
    ],
)
def test_a_painted_record_link_requires_one_available_reference(
    scenario: str, line: str
) -> None:
    """A painted link follows the identifier its earlier read assigned.

    An input collision, repeated identifier, later read, or changed fact
    cannot supply that link's identity. Recognition may change case or
    edge punctuation.
    """
    from computeruse.actions import DomAttribute, DomLocator, Point, ScreenTarget
    from computeruse.capability import HelpReason, Match, RefKind, TextTarget
    from computeruse.decider import Fact, FactRef
    from computeruse.reading import Line

    profile = load_profile(Path("evaluation/profile.yaml"))
    read_first = scenario != "before_read"
    as_output = scenario == "output"
    evolving = scenario == "evolving_fact"
    state = PageState("http://127.0.0.1:8787/members")
    shown = DomLocator("dd", DomAttribute.ID, "reference")
    search = AxLocator("textbox", "Search")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(
            AxNode(
                "text",
                "Account AC0000165",
                tag="dd",
                attributes=(("id", "reference"),),
            ),
            AxNode("textbox", "Search", value="", tag="input"),
        ),
    )
    capture = Observation(
        "capture",
        ObservationMode.VISUAL,
        ObservationStatus.COMPLETE,
        state,
        image=png((640, 520), ()),
    )
    trace = DiscoveryTrace()
    trace.look(before)

    def read_and_search(
        step: int, seen: Observation, value: str = "AC0000165"
    ) -> Observation:
        trace.action(
            step,
            Action(ActionKind.READ, shown),
            ActionResult(Outcome.OK, state, f"Account {value}"),
            state.location,
            seen,
        )
        trace.kept(
            Fact(
                "account",
                value,
                step,
                "/members",
                shown,
                may_change=evolving,
                partial=True,
            )
        )
        trace.used(step + 1, "account")
        trace.action(
            step + 1,
            Action(ActionKind.TYPE, search, value),
            ActionResult(Outcome.OK, state),
            state.location,
            seen,
        )
        searched = dataclasses.replace(
            seen,
            observation_id="searched",
            nodes=tuple(
                dataclasses.replace(node, value=value)
                if node.role == "textbox"
                else node
                for node in seen.nodes
            ),
        )
        trace.look(searched)
        return searched

    current = before
    if read_first:
        current = read_and_search(1, before)
    trace.look(capture)
    clicked_at = 3 if read_first else 1
    trace.action(
        clicked_at,
        Action(ActionKind.CLICK, ScreenTarget("capture", Point(50, 471))),
        ActionResult(Outcome.OK, state),
        state.location,
        current,
    )
    after = dataclasses.replace(
        current,
        observation_id="after",
        nodes=(
            *(
                dataclasses.replace(node, name="Account AC0000166")
                if evolving and node.role == "text"
                else node
                for node in current.nodes
            ),
            AxNode("heading", "Posted", tag="h1"),
        ),
    )
    trace.look(after)
    posted = ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted")
    trace.verified(clicked_at, (posted,), after)
    if not read_first:
        read_and_search(2, after)
    if evolving:
        read_and_search(4, after, "AC0000166")
    checks = (CheckResult(posted, "Posted", True),)
    if as_output:
        checks += (
            CheckResult(
                ResultCheck(
                    CheckKind.RESULT,
                    FactRef("account"),
                    "AC0000165",
                    output="account_number",
                ),
                "AC0000165",
                True,
            ),
        )
    result = RunResult(
        Ending.COMPLETED,
        6 if evolving else 4,
        "done",
        outputs={"account_number": "AC0000165"} if as_output else {},
        verification=Verification.EXECUTOR,
        checks=checks,
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"member_id": "AC0000165"} if scenario == "input_collision" else {},
        capability_id="open_created",
        run="scripted",
        safe_text=frozenset({"Posted", "text", "reference", "Search"}),
        reader=_Lines(Line(line, 40, 460, 300, 22, 0.99)),
        collect=True,
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    serialized = dumps(capability)
    assert "AC0000165" not in serialized
    assert "AC0000166" not in serialized
    if scenario not in {"variable", "recognized_case", "output"}:
        (human,) = [node for node in capability.nodes if isinstance(node, HumanNode)]
        assert human.reason is HelpReason.UNREPRESENTABLE_TARGET
        assert human.performs is ActionKind.CLICK
        assert not any(
            isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
            for node in capability.nodes
        )
        if evolving:
            reads = [
                node.into
                for node in capability.nodes
                if isinstance(node, ActionNode) and node.kind is ActionKind.READ
            ]
            uses = [
                node.value
                for node in capability.nodes
                if isinstance(node, ActionNode) and node.kind is ActionKind.TYPE
            ]
            assert len(reads) == 2
            assert reads == uses
        return
    assert not any(isinstance(node, HumanNode) for node in capability.nodes)
    clicked = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
    )
    target = capability.target(clicked.target or "")
    assert isinstance(target, TextTarget)
    assert target.text.kind is (RefKind.OUTPUT if as_output else RefKind.VARIABLE)
    assert target.match is Match.CONTAINS


@pytest.mark.rule(3)
@pytest.mark.parametrize("outcome", [Outcome.OK, Outcome.UNCERTAIN])
def test_every_recorded_check_must_be_representable(outcome):
    """Delivery does not excuse losing the record-specific result check."""

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(AxNode("button", "Post", tag="button"),),
    )
    after = dataclasses.replace(
        before,
        observation_id="after",
        nodes=(
            AxNode("heading", "Posted", tag="h1"),
            AxNode("text", "AC0000165", tag="span"),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Post")),
        ActionResult(outcome, state),
        state.location,
        before,
    )
    trace.verified(
        1,
        (
            ResultCheck(CheckKind.STATE, AxLocator("text", "AC0000165"), "AC0000165"),
            ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
        ),
        after,
    )
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        2,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
                "Posted",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="announced",
        run="scripted",
        safe_text=frozenset({"Post", "Posted", "heading"}),
    )
    assert not recording.complete
    assert recording.capability is None
    assert Gap.UNREPRESENTABLE_CHECK in {issue.gap for issue in recording.issues}


@pytest.mark.rule(3)
def test_unconfirmed_check_text_keeps_the_rebuilt_capability_incomplete():
    """A receipt-specific check cannot disappear when its text is unconfirmed."""

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(AxNode("button", "Post", tag="button"),),
    )
    after = dataclasses.replace(
        before,
        observation_id="after",
        nodes=(
            AxNode("heading", "Posted", tag="h1"),
            AxNode("link", "R-17", tag="a"),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Post")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.verified(
        1,
        (
            ResultCheck(CheckKind.STATE, AxLocator("link", "R-17"), "R-17"),
            ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
        ),
        after,
    )
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        2,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
                "Posted",
                True,
            ),
        ),
    )

    def build(words: frozenset[str], *, collect: bool = False):
        return trace.build(
            result,
            profile=profile,
            inputs={},
            capability_id="posted",
            run="scripted",
            safe_text=words,
            collect=collect,
        )

    collected = build(frozenset(), collect=True)
    assert {item.text for item in collected.candidates} >= {"R-17", "Posted"}
    words = frozenset(item.text for item in collected.candidates if item.text != "R-17")
    strict = build(words)
    assert not strict.complete
    assert strict.capability is None
    assert Gap.UNREPRESENTABLE_CHECK in {issue.gap for issue in strict.issues}


@pytest.mark.parametrize("moved", [True, False])
def test_a_partial_look_proves_a_click_only_when_the_page_moved(moved):
    """A long list is seen in part.

    Its next control proves a filter only after the page's address changed.
    """
    from computeruse.recorder import Unneeded

    profile = load_profile(Path("evaluation/profile.yaml"))
    listed = PageState("http://127.0.0.1:8787/members")
    filtered = PageState(
        "http://127.0.0.1:8787/members?q=10001" if moved else listed.location
    )
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.PARTIAL,
        listed,
        nodes=(AxNode("button", "Apply", tag="button"),),
    )
    after = Observation(
        "after",
        ObservationMode.STRUCTURED,
        ObservationStatus.PARTIAL,
        filtered,
        nodes=(AxNode("link", "Open", tag="a"),),
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Apply")),
        ActionResult(Outcome.OK, filtered),
        filtered.location,
        before,
    )
    trace.look(after)
    trace.action(
        2,
        Action(ActionKind.CLICK, AxLocator("link", "Open")),
        ActionResult(Outcome.OK, filtered),
        filtered.location,
        after,
    )
    recorded = []

    import computeruse.recording as recording

    original = recording.Recorder.record

    def keep(self, event):
        recorded.append(event)
        return original(self, event)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(recording.Recorder, "record", keep)
        trace.build(
            RunResult(Ending.COMPLETED, 3, "done", verification=Verification.EXECUTOR),
            profile=profile,
            inputs={},
            capability_id="filtered",
            run="scripted",
            safe_text=frozenset({"Apply", "Open"}),
        )
    proofs = [
        event
        for event in recorded
        if type(event).__name__ == "Verified" and event.step == 1
    ]
    assert not any(isinstance(event, Unneeded) for event in recorded)
    if moved:
        (proof,) = proofs
        kinds = {check.kind.value for check in proof.checks}
        assert kinds == {"at_route", "present"}
    else:
        assert proofs == []


@pytest.mark.rule(3)
def test_a_check_before_its_value_is_available_prevents_export():
    """A search shows the new number, and the run reads it one step later.

    The search's check names the number before the read fills the variable.
    Export cannot refer to a future value or remove the obligation.
    """
    from computeruse.actions import DomAttribute, DomLocator
    from computeruse.decider import Fact, FactRef

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    shown = DomLocator("dd", DomAttribute.ID, "reference")
    search = Observation(
        "search",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(AxNode("button", "Search", tag="button"),),
    )
    found = dataclasses.replace(
        search,
        observation_id="found",
        nodes=(
            AxNode("heading", "Posted", tag="h1"),
            AxNode("text", "AC0000165", tag="dd", attributes=(("id", "reference"),)),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(search)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Search")),
        ActionResult(Outcome.OK, state),
        state.location,
        search,
    )
    trace.verified(
        1,
        (
            ResultCheck(CheckKind.STATE, AxLocator("text", "AC0000165"), "AC0000165"),
            ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
        ),
        found,
    )
    trace.look(found)
    trace.action(
        2,
        Action(ActionKind.READ, shown),
        ActionResult(Outcome.OK, state, "AC0000165"),
        state.location,
        found,
    )
    trace.kept(Fact("account", "AC0000165", 2, "/members", shown, may_change=False))
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        outputs={"account_number": "AC0000165"},
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.RESULT,
                    FactRef("account"),
                    "AC0000165",
                    output="account_number",
                ),
                "AC0000165",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="search_then_read",
        run="scripted",
        safe_text=frozenset({"Search", "Posted", "heading", "reference", "text"}),
    )
    assert not recording.complete
    assert recording.capability is None
    assert Gap.UNREPRESENTABLE_CHECK in {issue.gap for issue in recording.issues}


def _signed_on(contract, name="Operator"):
    """Record a run that only clicks Sign on, on a form that chose OP0002.

    ``name`` is the field's accessible name. An empty name means the page
    never linked the label, so the collector keeps it only in the field's slot.
    """
    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    operator = AxNode(
        "combobox",
        name,
        value="OP0002",
        tag="select",
        control="d1:c1",
        form="d1:f1",
        slot="select|Operator",
    )
    sign_on = AxNode(
        "button",
        "Sign on",
        tag="button",
        control="d1:c2",
        form="d1:f1",
        submits="d1:f1>d1:c2",
    )
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(operator, sign_on),
    )
    after = dataclasses.replace(
        before,
        observation_id="after",
        nodes=(AxNode("heading", "Posted", tag="h1"),),
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.CLICK, AxLocator("button", "Sign on")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        2,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
                "Posted",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"operator_id": "OP0002"},
        capability_id="sign_on",
        run="scripted",
        safe_text=frozenset(
            {"Operator", "Sign on", "Posted", "combobox", "select|Operator"}
        ),
        contract=contract,
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    return [node for node in capability.nodes if isinstance(node, ActionNode)]


@pytest.mark.rule(4)
@pytest.mark.parametrize("name", ["Operator", ""])
def test_a_field_a_form_showed_set_to_an_input_becomes_a_step(name):
    """The sign-on page selects the requested operator before the run does.

    Discovery only clicks Sign on. The contract binds the operator input to
    the Operator field. The saved capability first sets the field from its
    input, so replay signs on as the new operator. The step is marked as added
    because discovery never took it.
    """
    from computeruse.capability import RefKind
    from computeruse.contract import Binding, Contract

    contract = Contract(
        2, (field("operator_id"),), (), (Binding("operator_id", "Operator"),)
    )
    steps = _signed_on(contract, name)
    assert [step.kind for step in steps][:2] == [ActionKind.SELECT, ActionKind.CLICK]
    assert steps[0].value is not None
    assert (steps[0].value.kind, steps[0].value.name) == (RefKind.INPUT, "operator_id")
    assert steps[0].introduced == "binding:operator_id"
    assert not steps[1].introduced


@pytest.mark.rule(4)
def test_an_unbound_preset_field_becomes_a_precondition_not_a_step():
    """Equal values alone never tie a field to an input.

    Without a binding, the submit requires the field to show the input
    already, so a replay for another operator stops instead of signing on
    as the one the form chose.
    """
    from computeruse.capability import RefKind, Shows

    steps = _signed_on(None)
    assert [step.kind for step in steps][:1] == [ActionKind.CLICK]
    shows = [check for check in steps[0].requires if isinstance(check, Shows)]
    assert [(check.value.kind, check.value.name) for check in shows] == [
        (RefKind.INPUT, "operator_id")
    ]


class _Lines:
    """A reader that returns set lines for any capture."""

    def __init__(self, *lines) -> None:
        self.found = lines

    def lines(self, image: bytes) -> tuple:
        del image
        return self.found


@pytest.mark.rule(3, 8)
def test_generated_painted_checkpoint_prefers_reviewed_labels_to_unknown_text():
    from computeruse.actions import Point, ScreenTarget
    from computeruse.decider import Match
    from computeruse.reading import Line

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    before = Observation(
        "before",
        ObservationMode.VISUAL,
        ObservationStatus.COMPLETE,
        state,
        image=b"before",
    )
    after = dataclasses.replace(before, observation_id="after", image=b"after")

    class Reader:
        def lines(self, image):
            if image == b"before":
                return (Line("Prepare", 40, 120, 100, 20, 0.99),)
            return (
                Line("User name: Sam Lee", 40, 30, 200, 20, 0.99),
                Line("Review complete", 40, 180, 180, 20, 0.99),
            )

    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(
            ActionKind.CLICK, ScreenTarget("before", Point(70, 130)), effect="prepare"
        ),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    check = ResultCheck(
        CheckKind.STATE,
        ScreenTarget("after", Point(90, 190)),
        "Review complete",
        Match.CONTAINS,
    )
    built = trace.build(
        RunResult(
            Ending.COMPLETED,
            2,
            "done",
            verification=Verification.EXECUTOR,
            checks=(CheckResult(check, "Review complete", True),),
        ),
        profile=profile,
        inputs={},
        capability_id="prepare",
        run="scripted",
        safe_text=frozenset({"Prepare", "Review complete", "prepare"}),
        collect=True,
        reader=Reader(),
    )
    assert built.complete, (built.issues, built.artifact_issues)
    assert built.capability is not None
    assert "Sam Lee" not in dumps(built.capability)
    assert not built.candidates


@pytest.mark.rule(4)
@pytest.mark.parametrize("dismissal", ["disappears", "remains", "ambiguous_before"])
def test_a_disappearing_option_click_survives_changed_input_replay(
    pages, monkeypatch, dismissal: str
) -> None:
    """An already visible next field does not make a picker click unnecessary."""
    import time

    from replay_fakes import People

    from computeruse.actions import ObservationRequest
    from computeruse.recorder import CheckKind as RecordedCheck
    from computeruse.recorder import ExecutionEvent, Recorder, Verified
    from computeruse.replay import MemoryReplayLog, Status, replay

    alternate = (
        '<button type="button" role="option" data-product="Gold savings">'
        "ALT-G: Gold savings</button>"
        if dismissal == "ambiguous_before"
        else ""
    )
    markup = (
        '<h1>Prepare account</h1><div role="listbox">'
        '<button type="button" role="option" data-product="Gold savings">'
        "PLAN-G: Gold savings</button>"
        '<button type="button" role="option" data-product="Silver savings">'
        "PLAN-S: Silver savings</button>"
        + alternate
        + "</div><label>Nickname<input></label><script>"
        "window.selectedProduct=null;window.optionClicks=0;"
        "for (const option of document.querySelectorAll('[role=option]')) {"
        "option.onclick=()=>{window.optionClicks++;"
        "window.selectedProduct=option.dataset.product;"
        + ("" if dismissal == "remains" else "option.parentElement.replaceChildren();")
        + "};}</script>"
    )
    events: list[ExecutionEvent] = []
    original = Recorder.record

    def capture_event(recorder: Recorder, event: ExecutionEvent) -> None:
        events.append(event)
        original(recorder, event)

    monkeypatch.setattr(Recorder, "record", capture_event)
    with pages({"/": markup}) as (surface, profile):
        request = ObservationRequest(ObservationMode.STRUCTURED)
        before = surface.observe(request)
        assert before.status is ObservationStatus.COMPLETE
        assert any(node.name == "Nickname" for node in before.nodes)
        trace = DiscoveryTrace()
        trace.look(before)
        choice = Action(ActionKind.CLICK, AxLocator("option", "PLAN-G: Gold savings"))
        clicked = surface.act(choice)
        assert clicked.outcome is Outcome.OK
        trace.action(1, choice, clicked, before.location, before)
        after_choice = surface.observe(request)
        assert after_choice.status is ObservationStatus.COMPLETE
        trace.look(after_choice)
        nickname = AxLocator("textbox", "Nickname")
        typing = Action(ActionKind.TYPE, nickname, "Rainy day")
        typed = surface.act(typing)
        assert typed.outcome is Outcome.OK
        trace.input_used(2, "nickname")
        trace.action(2, typing, typed, after_choice.location, after_choice)
        after = surface.observe(request)
        trace.look(after)
        reading = surface.act(Action(ActionKind.READ, nickname))
        assert reading.outcome is Outcome.OK
        assert reading.extracted == "Rainy day"
        assert surface._page.evaluate("window.selectedProduct") == "Gold savings"
        assert surface._page.evaluate("window.optionClicks") == 1
        result = RunResult(
            Ending.COMPLETED,
            3,
            "done",
            verification=Verification.EXECUTOR,
            checks=(
                CheckResult(
                    ResultCheck(CheckKind.STATE, nickname, "Rainy day"),
                    reading.extracted,
                    True,
                ),
            ),
        )
        recording = trace.build(
            result,
            profile=profile,
            inputs={"product": "Gold savings", "nickname": "Rainy day"},
            capability_id="choose_product",
            run="browser",
            safe_text=frozenset({"Prepare account", "Nickname"}),
            collect=True,
        )
        absence = [
            check
            for event in events
            if isinstance(event, Verified) and event.step == 1
            for check in event.checks
            if check.kind is RecordedCheck.ABSENT
        ]
        assert bool(absence) is (dismissal == "disappears")
        if dismissal != "disappears":
            if recording.capability is not None:
                assert any(
                    isinstance(node, (ActionNode, HumanNode))
                    and (
                        node.kind is ActionKind.CLICK
                        if isinstance(node, ActionNode)
                        else node.performs is ActionKind.CLICK
                    )
                    for node in recording.capability.nodes
                )
            return
        assert recording.complete, (recording.issues, recording.artifact_issues)
        capability = recording.capability
        assert capability is not None
        assert (
            sum(
                isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
                for node in capability.nodes
            )
            == 1
        )
        assert not any(isinstance(node, HumanNode) for node in capability.nodes)
        serialized = dumps(capability)
        assert "Gold savings" not in serialized
        assert "Rainy day" not in serialized
        surface._page.reload()
        person = People([])
        replayed = replay(
            capability,
            {"product": "Silver savings", "nickname": "Holiday reserve"},
            profile=profile,
            surface=surface,
            control=person.control(time.monotonic),
            log=MemoryReplayLog(),
            clock=time.monotonic,
            accept_draft=True,
        )
        assert replayed.status is Status.SUCCEEDED, replayed
        assert not person.requests
        assert surface._page.evaluate("window.selectedProduct") == "Silver savings"
        assert surface._page.evaluate("window.optionClicks") == 1
        assert surface._page.get_by_label("Nickname").input_value() == "Holiday reserve"


@pytest.mark.rule(4, 6)
@pytest.mark.parametrize(
    "attestation",
    [
        "current",
        "unknown",
        "stale_page",
        "stale_name",
        "stale_role",
        "stale_scope",
        "stale_alias",
        "stale_slot",
        "stale_attribute",
        "stale_class",
        "intervening_observation",
        "missing_metadata",
    ],
)
def test_a_pixel_row_click_replays_by_its_attested_structural_control(
    pages, attestation: str
) -> None:
    """The query and row share a value, but the adapter identifies the clicked cell."""
    import time

    from replay_fakes import People

    from computeruse.actions import (
        Expectation,
        ObservationRequest,
        Operation,
        Point,
        ScreenTarget,
    )
    from computeruse.capability import StructuralTarget
    from computeruse.replay import MemoryReplayLog, Status, replay

    markup = """<h1>Accounts</h1><label>Search<input></label>
      <table><thead><tr><th>Account</th><th>Reference</th></tr></thead><tbody>
        <tr data-account="A-71"><td class="account-link" data-testid="account-link">
          A-71 · Rainy day</td><td>Ref 71</td></tr>
        <tr data-account="A-82"><td class="account-link" data-testid="account-link">
          A-82 · Holiday reserve</td><td>Ref 82</td></tr>
      </tbody></table><label>Opened account<input readonly></label>
      <style>td {padding: 20px; min-width: 300px}</style>
      <script>
      window.openedAccount=null;window.rowClicks=0;
      for (const row of document.querySelectorAll('tr[data-account]')) {
        row.onclick=()=>{
          window.rowClicks++;
          window.openedAccount=row.dataset.account;
          document.querySelector('input[readonly]').value=row.dataset.account;
        };
      }
      </script>"""
    with pages({"/": markup}) as (surface, profile):
        request = ObservationRequest(ObservationMode.STRUCTURED)
        before = surface.observe(request)
        trace = DiscoveryTrace()
        trace.look(before)
        search = AxLocator("textbox", "Search")
        typing = Action(ActionKind.TYPE, search, "A-71")
        typed = surface.act(typing)
        assert typed.outcome is Outcome.OK
        trace.input_used(1, "account")
        trace.action(1, typing, typed, before.location, before)
        searched = surface.observe(request)
        assert searched.status is ObservationStatus.COMPLETE
        trace.look(searched)
        cell = surface._page.locator('tr[data-account="A-71"] td').first
        mutations = {
            "stale_name": "el => el.setAttribute('aria-label', 'Replaced account')",
            "stale_role": "el => el.setAttribute('role', 'button')",
            "stale_scope": """el => el.parentElement.insertAdjacentHTML(
              'afterbegin', '<th>Moved group</th>')""",
            "stale_alias": (
                "el => el.nextElementSibling.textContent = 'Other reference'"
            ),
            "stale_slot": """el => el.closest('table').querySelector('th')
              .textContent = 'Changed column'""",
            "stale_attribute": "el => el.setAttribute('data-testid', 'changed-link')",
            "stale_class": "el => el.classList.remove('account-link')",
            "intervening_observation": (
                "el => el.setAttribute('aria-label', 'Replaced account')"
            ),
        }
        if attestation in mutations:
            cell.evaluate(mutations[attestation])
        if attestation == "intervening_observation":
            surface.observe(request)
        capture = surface.observe(ObservationRequest(ObservationMode.VISUAL))
        trace.look(capture)
        box = cell.bounding_box()
        assert box is not None
        target = ScreenTarget(
            capture.observation_id,
            Point(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2),
        )
        control = surface.control_at(target)
        assert control
        node = next(node for node in searched.nodes if node.control == control)
        assert node.tag == "td"
        assert node.role == "cell"
        click = Action(ActionKind.CLICK, target)
        route = profile.scope.route(searched.location)
        assert route is not None
        operation = Operation.of(click, route, node=node)
        expected = Expectation(
            searched.page_state,
            control=node.control,
            submits_as=node.submits_as,
            observed_control=None if attestation == "missing_metadata" else node,
        )
        clicked = surface.act(click, expect=expected)
        if attestation in mutations or attestation == "missing_metadata":
            assert clicked.outcome is Outcome.STALE
            assert surface._page.evaluate("window.rowClicks") == 0
            assert surface._page.evaluate("window.openedAccount") is None
            refreshed = surface.observe(request)
            current = next(item for item in refreshed.nodes if item.control == control)
            fresh_capture = surface.observe(ObservationRequest(ObservationMode.VISUAL))
            fresh_target = dataclasses.replace(
                target, capture_id=fresh_capture.observation_id
            )
            repeated = surface.act(
                dataclasses.replace(click, target=fresh_target),
                expect=dataclasses.replace(
                    expected,
                    page_state=refreshed.page_state,
                    observed_control=current,
                    submits_as=current.submits_as,
                ),
            )
            assert repeated.outcome is Outcome.OK
            assert surface._page.evaluate("window.rowClicks") == 1
            assert surface._page.evaluate("window.openedAccount") == "A-71"
            return
        assert clicked.outcome is Outcome.OK
        recorded_seen = searched
        if attestation == "unknown":
            operation = dataclasses.replace(operation, control="unknown-control")
        elif attestation == "stale_page":
            recorded_seen = dataclasses.replace(
                searched,
                page_state=dataclasses.replace(searched.page_state, page="other-page"),
            )
        trace.action(
            2, click, clicked, searched.location, recorded_seen, operation=operation
        )
        after = surface.observe(request)
        trace.look(after)
        opened = AxLocator("textbox", "Opened account")
        read = surface.act(Action(ActionKind.READ, opened))
        assert read.outcome is Outcome.OK
        assert read.extracted == "A-71"
        assert surface._page.evaluate("window.rowClicks") == 1
        check = ResultCheck(CheckKind.STATE, opened, "A-71")
        trace.verified(2, (check,), after)
        recording = trace.build(
            RunResult(
                Ending.COMPLETED,
                3,
                "done",
                verification=Verification.EXECUTOR,
                checks=(CheckResult(check, read.extracted, True),),
            ),
            profile=profile,
            inputs={"account": "A-71"},
            capability_id="open_account",
            run="browser",
            safe_text=frozenset({"Accounts", "Search", "Opened account", "td|Account"}),
            collect=True,
        )
        assert recording.complete, (recording.issues, recording.artifact_issues)
        capability = recording.capability
        assert capability is not None
        if attestation != "current":
            assert any(isinstance(node, HumanNode) for node in capability.nodes)
            assert not any(
                isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
                for node in capability.nodes
            )
            return
        assert not any(isinstance(node, HumanNode) for node in capability.nodes)
        (saved_click,) = [
            node
            for node in capability.nodes
            if isinstance(node, ActionNode) and node.kind is ActionKind.CLICK
        ]
        assert isinstance(capability.target(saved_click.target or ""), StructuralTarget)
        assert all(
            isinstance(target, StructuralTarget) for target in capability.targets
        )
        assert "A-71" not in dumps(capability)
        surface._page.reload()
        person = People([])
        replayed = replay(
            capability,
            {"account": "A-82"},
            profile=profile,
            surface=surface,
            control=person.control(time.monotonic),
            log=MemoryReplayLog(),
            clock=time.monotonic,
            accept_draft=True,
        )
        assert replayed.status is Status.SUCCEEDED, replayed
        assert not person.requests
        assert surface._page.get_by_label("Search", exact=True).input_value() == "A-82"
        assert surface._page.evaluate("window.openedAccount") == "A-82"
        assert surface._page.evaluate("window.rowClicks") == 1


@pytest.mark.rule(4)
def test_the_click_that_chose_the_field_is_kept() -> None:
    """Export a painted field click, focused typing and labelled submission."""
    from computeruse.actions import Point, ScreenTarget
    from computeruse.capability import FocusTarget, RefKind, TextTarget
    from computeruse.decider import Match as CheckMatch
    from computeruse.reading import Line

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    before_image = png((640, 520), ())
    typed_image = png((640, 521), ())
    submitted_image = png((640, 522), ())

    class Lines:
        def lines(self, image: bytes) -> tuple[Line, ...]:
            lines = (
                Line("Nickname", 40, 300, 80, 20, 0.99),
                Line("Find member", 50, 400, 100, 20, 0.99),
            )
            if image != before_image:
                lines += (Line("Ready", 40, 440, 80, 20, 0.99),)
            if image == submitted_image:
                lines += (Line("Submitted", 40, 480, 100, 20, 0.99),)
            return lines

    before = Observation(
        "c1",
        ObservationMode.VISUAL,
        ObservationStatus.COMPLETE,
        state,
        image=before_image,
    )
    focused = dataclasses.replace(before, observation_id="c2")
    typed = dataclasses.replace(before, observation_id="c3", image=typed_image)
    submitted = dataclasses.replace(before, observation_id="c4", image=submitted_image)
    trace = DiscoveryTrace()
    for step, seen, action in (
        (1, before, Action(ActionKind.CLICK, ScreenTarget("c1", Point(120, 345)))),
        (2, focused, Action(ActionKind.TYPE, ScreenTarget("c2"), "Rainy day")),
        (3, typed, Action(ActionKind.CLICK, ScreenTarget("c3", Point(100, 410)))),
    ):
        trace.look(seen)
        trace.action(
            step, action, ActionResult(Outcome.OK, state), state.location, seen
        )
        if step == 2:
            trace.verified(
                step,
                (
                    ResultCheck(
                        CheckKind.STATE,
                        ScreenTarget("c3", Point(60, 450)),
                        "Ready",
                        match=CheckMatch.CONTAINS,
                    ),
                ),
                typed,
            )
    trace.look(submitted)
    check = ResultCheck(
        CheckKind.STATE,
        ScreenTarget("c4", Point(60, 490)),
        "Submitted",
        match=CheckMatch.CONTAINS,
    )
    recording = trace.build(
        RunResult(
            Ending.COMPLETED,
            4,
            "done",
            verification=Verification.EXECUTOR,
            checks=(CheckResult(check, "Submitted", True),),
        ),
        profile=profile,
        inputs={"nickname": "Rainy day"},
        capability_id="focus",
        run="scripted",
        safe_text=frozenset({"Nickname", "Find member", "Ready", "Submitted"}),
        reader=Lines(),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    steps = [node for node in capability.nodes if isinstance(node, ActionNode)]
    assert [step.kind for step in steps] == [
        ActionKind.CLICK,
        ActionKind.TYPE,
        ActionKind.CLICK,
    ]
    field = capability.target(steps[0].target or "")
    focus = capability.target(steps[1].target or "")
    button = capability.target(steps[2].target or "")
    assert isinstance(field, TextTarget)
    assert (field.text.value, field.dx, field.dy) == ("Nickname", 80, 45)
    assert isinstance(focus, FocusTarget)
    assert isinstance(button, TextTarget)
    assert (button.text.value, button.dx, button.dy) == ("Find member", 0, 0)
    assert steps[1].value is not None
    assert (steps[1].value.kind, steps[1].value.name) == (RefKind.INPUT, "nickname")
    assert not any(isinstance(node, HumanNode) for node in capability.nodes)
    assert "Rainy day" not in dumps(capability)


@pytest.mark.rule(11)
@pytest.mark.parametrize("change_after_read", [False, True])
def test_a_changing_value_is_read_again_on_replay(
    pages, change_after_read: bool
) -> None:
    # Graduated from tests/test_known_gaps.py (R3).
    import time
    from typing import cast

    from fakes import ScriptedDecider, ScriptedEscalator
    from replay_fakes import People

    from computeruse.actions import (
        DomAttribute,
        DomLocator,
        ObservationMode,
        ObservationRequest,
    )
    from computeruse.browser import BrowserSurface
    from computeruse.decider import CheckKind, Finish, Observe, Propose, Remember
    from computeruse.decider import ResultCheck as Check
    from computeruse.journal import MemoryJournal
    from computeruse.loop import Ending, discover
    from computeruse.recording import ActionTaken
    from computeruse.replay import MemoryReplayLog, Status, replay

    markup = '<h1>Quote</h1><span id="quote">7.00</span><label>Amount<input></label>'
    source = DomLocator("span", DomAttribute.ID, "quote")
    fact_name = "private_fact_canary_9821"
    output_name = "private_output_canary_7462"
    with pages({"/": markup}) as (surface, profile):
        trace = DiscoveryTrace()
        result = discover(
            "Copy the current quote",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Remember(fact_name, "7.00", source=source, may_change=True),
                    Propose(
                        Action(ActionKind.TYPE, AxLocator("textbox", "Amount"), "7.00"),
                        fact=fact_name,
                    ),
                    Finish(
                        {output_name: "7.00"},
                        checks=(
                            Check(
                                CheckKind.RESULT,
                                AxLocator("textbox", "Amount"),
                                "7.00",
                                output=output_name,
                            ),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
        )
        assert result.ending is Ending.COMPLETED
        recording = trace.build(
            result,
            profile=profile,
            inputs={},
            capability_id="copy_quote",
            run="gap",
            safe_text=frozenset({"Quote", "quote", "Amount"}),
        )
        assert recording.capability is not None
        serialized = dumps(recording.capability)
        assert fact_name not in serialized
        assert output_name not in serialized
        assert [output.name for output in recording.capability.outputs] == ["output_1"]
        reads = [
            entry
            for entry in trace.entries
            if isinstance(entry, ActionTaken) and entry.action.kind is ActionKind.READ
        ]
        assert len(reads) >= 2
        surface._page.reload()
        original = surface.act

        class ChangesAfterTheFirstRead:
            def __getattr__(self, name):
                return getattr(surface, name)

            def act(self, action, *, expect=None):
                taken = original(action, expect=expect)
                if (
                    change_after_read
                    and action.kind is ActionKind.READ
                    and action.target == source
                ):
                    surface._page.locator("#quote").evaluate(
                        "el => el.textContent = '9.00'"
                    )
                return taken

        replayed = replay(
            recording.capability,
            {},
            profile=profile,
            surface=cast(BrowserSurface, ChangesAfterTheFirstRead()),
            control=People([]).control(time.monotonic),
            log=MemoryReplayLog(),
            clock=time.monotonic,
            accept_draft=True,
        )
        typed = surface._page.get_by_label("Amount").input_value()
        if change_after_read:
            assert replayed.status is not Status.SUCCEEDED or typed == "9.00"
        else:
            assert replayed.status is Status.SUCCEEDED
            assert typed == "7.00"
            assert replayed.outputs == {"output_1": "7.00"}


@pytest.mark.rule(9)
def test_a_text_that_is_both_an_input_and_a_kept_fact_is_refused():
    """Equal strings do not say where a value came from (rule 9).

    The run kept a reference number that happens to equal the member input,
    then opened a link named by it. Saving the link as the input, only
    because inputs are checked first, would guess. The step is a gap.
    """
    from computeruse.actions import DomAttribute, DomLocator
    from computeruse.decider import Fact

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    shown = DomLocator("dd", DomAttribute.ID, "reference")
    before = Observation(
        "before",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(
            AxNode("text", "10001", tag="dd", attributes=(("id", "reference"),)),
            AxNode("link", "10001", tag="a"),
        ),
    )
    after = dataclasses.replace(
        before, observation_id="after", nodes=(AxNode("heading", "Posted", tag="h1"),)
    )
    trace = DiscoveryTrace()
    trace.look(before)
    trace.action(
        1,
        Action(ActionKind.READ, shown),
        ActionResult(Outcome.OK, state, "10001"),
        state.location,
        before,
    )
    trace.kept(Fact("reference", "10001", 1, "/members", shown, may_change=False))
    trace.action(
        2,
        Action(ActionKind.CLICK, AxLocator("link", "10001")),
        ActionResult(Outcome.OK, state),
        state.location,
        before,
    )
    trace.look(after)
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
                "Posted",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"member_id": "10001"},
        capability_id="collide",
        run="scripted",
        safe_text=frozenset({"Posted", "link", "text", "reference"}),
    )
    assert not recording.complete


@pytest.mark.rule(15)
def test_a_painted_not_found_answer_is_saved_as_its_words_and_the_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The answer "No member with number 99999." holds the searched input.

    Both checks of the claim point at the answer, one for the state and one for
    the searched record. The recorder saves each as the words before the input
    plus the input. The learned branch then works for any member, while the
    painted field containing the typed number cannot satisfy it.
    """
    from contextlib import nullcontext

    from computeruse import browser, cli, loop, model, reading
    from computeruse.actions import Point, ScreenTarget
    from computeruse.capability import RefKind, ResultNode, Shows, TextTarget
    from computeruse.decider import CheckKind, Match, ResultCheck, Task, TaskRecord
    from computeruse.loop import CheckResult
    from computeruse.reading import Line

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")

    before = png((500, 800), ())
    after = png((500, 800), ((440, 10),))

    def capture(name: str) -> Observation:
        return Observation(
            name,
            ObservationMode.VISUAL,
            ObservationStatus.COMPLETE,
            state,
            image=after if name == "cap-3" else before,
        )

    class AnswerLines(_Lines):
        def lines(self, image: bytes) -> tuple[Line, ...]:
            return self.found if image == after else self.found[:-1]

    def answer_lines(member: str, label: str) -> AnswerLines:
        return AnswerLines(
            Line(member, 50, 190, 60, 20, 0.99),
            Line("Find member", 50, 250, 100, 20, 0.99),
            Line(f"{label} {member}.", 40, 740, 300, 20, 0.99),
        )

    def record(
        trace: DiscoveryTrace, member: str, label: str, outcome: str = ""
    ) -> tuple[RunResult, _Lines]:
        answer = f"{label} {member}."
        trace.look(capture("cap-1"))
        trace.action(
            1,
            Action(ActionKind.TYPE, ScreenTarget("cap-1"), member),
            ActionResult(Outcome.OK, state),
            state.location,
            capture("cap-1"),
        )
        trace.look(capture("cap-2"))
        trace.action(
            2,
            Action(
                ActionKind.CLICK, ScreenTarget("cap-2", Point(100, 260)), effect="find"
            ),
            ActionResult(Outcome.OK, state),
            state.location,
            capture("cap-2"),
        )
        trace.look(capture("cap-3"))
        on_answer = ScreenTarget("cap-3", Point(300, 750))
        checks = (
            CheckResult(
                ResultCheck(CheckKind.STATE, on_answer, answer, Match.CONTAINS),
                answer,
                True,
            ),
            CheckResult(
                ResultCheck(CheckKind.RECORD, on_answer, member, Match.EQUALS),
                answer,
                True,
            ),
        )
        result = RunResult(
            Ending.COMPLETED,
            4,
            "done",
            verification=Verification.EXECUTOR,
            checks=checks,
            task=Task(records=(TaskRecord("member", member),)),
            outcome=outcome,
        )
        return result, answer_lines(member, label)

    known = frozenset({"Find member", "find", "No member with number", "Member number"})
    success_trace = DiscoveryTrace()
    success, success_reader = record(success_trace, "12345", "Member number")
    main = success_trace.build(
        success,
        profile=profile,
        inputs={"member_id": "12345"},
        capability_id="missing",
        run="scripted",
        safe_text=known,
        reader=success_reader,
    )
    assert main.complete, main.issues
    assert main.capability is not None

    def discover(*_args: object, trace: DiscoveryTrace, **_kwargs: object) -> RunResult:
        result, _ = record(trace, "99999", "No member with number", "record_not_found")
        return result

    monkeypatch.setattr(
        browser, "open_session", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(model, "LunaDecider", lambda **_kwargs: None)
    monkeypatch.setattr(loop, "discover", discover)
    monkeypatch.setattr(
        reading, "LocalReader", lambda: answer_lines("99999", "No member with number")
    )
    args = cli._parser().parse_args(
        [
            "discover",
            "--goal",
            "Find member 12345",
            "--website",
            state.location,
            "--profile",
            "evaluation/profile.yaml",
        ]
    )
    args.profile = tmp_path / "profile.yaml"
    args.export_contract = None
    capability = cli._learn_outcome(
        args,
        profile,
        main.capability,
        {"member_id": "12345"},
        [],
        known,
        "record_not_found",
        {"member_id": "99999"},
    )
    ending = [
        node
        for node in capability.nodes
        if isinstance(node, ResultNode) and node.outcome
    ]
    assert [node.outcome for node in ending] == ["record_not_found"]
    saved = [check for check in ending[0].checks if isinstance(check, Shows)]
    assert len(saved) == 2
    for check in saved:
        target = capability.target(check.target)
        assert isinstance(target, TextTarget)
        assert target.label
        assert target.text.value == "No member with number"
        assert (check.value.kind, check.value.name) == (RefKind.INPUT, "member_id")


@pytest.mark.rule(15)
def test_a_painted_lookup_the_loop_tied_saves_its_record_check_and_no_person():
    """The loop tied the painted result to the member its screen names once.

    The capability keeps the record check on the member label, which replay
    reads for the new member, and saves no person's step to confirm the tie.
    """
    from computeruse.actions import Point, ScreenTarget
    from computeruse.capability import (
        HumanNode,
        RefKind,
        ResultNode,
        Shows,
        TextTarget,
    )
    from computeruse.decider import (
        CheckKind,
        Match,
        ResultCheck,
        Task,
        TaskOutput,
        TaskRecord,
    )
    from computeruse.loop import CheckResult
    from computeruse.reading import Line

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")

    def capture(name: str) -> Observation:
        return Observation(
            name,
            ObservationMode.VISUAL,
            ObservationStatus.COMPLETE,
            state,
            image=b"canvas",
        )

    status = ScreenTarget("cap-2", Point(60, 230))
    trace = DiscoveryTrace()
    trace.look(capture("cap-1"))
    trace.action(
        1,
        Action(ActionKind.TYPE, ScreenTarget("cap-1"), "12345"),
        ActionResult(Outcome.OK, state),
        state.location,
        capture("cap-1"),
    )
    trace.look(capture("cap-2"))
    trace.action(
        2,
        Action(ActionKind.READ, status),
        ActionResult(Outcome.OK, state, extracted="Status: active"),
        state.location,
        capture("cap-2"),
    )
    checks = (
        CheckResult(
            ResultCheck(
                CheckKind.RESULT,
                status,
                "active",
                Match.EQUALS,
                output="membership_status",
            ),
            "Status: active",
            True,
            bound="12345",
        ),
        CheckResult(
            ResultCheck(
                CheckKind.RECORD,
                ScreenTarget("cap-2", Point(60, 160)),
                "12345",
                Match.CONTAINS,
            ),
            "Member number: 12345",
            True,
            bound="12345",
        ),
    )
    result = RunResult(
        Ending.COMPLETED,
        3,
        "done",
        outputs={"membership_status": "active"},
        verification=Verification.EXECUTOR,
        checks=checks,
        task=Task(
            records=(TaskRecord("member", "12345"),),
            outputs=(TaskOutput("membership_status", "member"),),
        ),
    )
    built = trace.build(
        result,
        profile=profile,
        inputs={"member_id": "12345"},
        capability_id="tied",
        run="scripted",
        safe_text=frozenset({"Member number:", "Status:"}),
        collect=True,
        reader=_Lines(
            Line("Member number: 12345", 40, 150, 200, 20, 0.99),
            Line("Status: active", 40, 220, 120, 20, 0.99),
        ),
    )
    assert built.issues == ()
    capability = built.capability
    assert capability is not None
    assert not any(isinstance(node, HumanNode) for node in capability.nodes)
    (done,) = [node for node in capability.nodes if isinstance(node, ResultNode)]
    records = [
        check
        for check in done.checks
        if isinstance(check, Shows) and check.value.kind is RefKind.INPUT
    ]
    targets = [capability.target(check.target) for check in records]
    assert [
        target.text.value for target in targets if isinstance(target, TextTarget)
    ] == ["Member number:"]


@pytest.mark.rule(9)
def test_a_painted_option_in_another_case_is_saved_as_its_input():
    """Found in canvas batch 56: the option "Paper" for the input "paper".

    A painted comparison ignores case. The recorder had refused the line as
    an unnamed input, which turned the choice into a person's step with no
    check. The recorder now saves it as the input through a target that the
    validator accepts.
    """
    import computeruse.recording as recording
    from computeruse.actions import Point, ScreenTarget
    from computeruse.capability import IssueCode
    from computeruse.reading import Line
    from computeruse.recorder import Executed, Gap, TextSample

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    capture = Observation(
        "cap-1",
        ObservationMode.VISUAL,
        ObservationStatus.COMPLETE,
        state,
        image=png((640, 520), ()),
    )
    trace = DiscoveryTrace()
    trace.look(capture)
    trace.action(
        1,
        Action(ActionKind.CLICK, ScreenTarget("cap-1", Point(50, 471))),
        ActionResult(Outcome.OK, state),
        state.location,
        capture,
    )
    events = []
    original = recording.Recorder.record

    def keep(self, event):
        events.append(event)
        return original(self, event)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(recording.Recorder, "record", keep)
        built = trace.build(
            RunResult(Ending.COMPLETED, 2, "done", verification=Verification.EXECUTOR),
            profile=profile,
            inputs={"statement_delivery": "paper"},
            capability_id="painted",
            run="scripted",
            safe_text=frozenset(),
            collect=True,
            reader=_Lines(
                Line("Paper", 40, 460, 80, 22, 0.99),
                Line("Electronic", 200, 460, 110, 22, 0.99),
            ),
        )
    clicked = [
        event
        for event in events
        if isinstance(event, Executed) and isinstance(event.target, TextSample)
    ]
    assert [event.target.text for event in clicked] == ["Paper"]
    # The trace holds no result, so only the step and its target are judged.
    assert Gap.UNREPRESENTABLE_STEP not in {issue.gap for issue in built.issues}
    assert not [
        issue
        for issue in built.artifact_issues
        if issue.code is IssueCode.INVALID_TARGET
    ]
