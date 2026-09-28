"""A value inside a longer reading survives recording and changed-input replay."""

import dataclasses

import pytest
from fakes import FakeClock, ScriptedSeat, doing, when
from replay_fakes import PROFILE, App, Clock, Page, People

from computeruse import matching
from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    DomAttribute,
    DomLocator,
    Observation,
    ObservationMode,
    ObservationStatus,
    Outcome,
    PageState,
)
from computeruse.capability import (
    ActionNode,
    Extraction,
    IssueCode,
    RefKind,
    ResultConfirmationNode,
    Review,
    constant,
    dumps,
    loads,
    ref,
    validate,
)
from computeruse.contract import Contract
from computeruse.control import Control
from computeruse.decider import (
    CheckKind,
    Fact,
    FactRef,
    ResultCheck,
    Task,
    TaskRecord,
    TaskRequirement,
)
from computeruse.decider import Match as CheckMatch
from computeruse.escalation import (
    Ask,
    Command,
    HandoffOutcome,
    InterventionRequest,
    Mode,
    State,
    Trigger,
)
from computeruse.loop import CheckResult, Ending, RunResult, Verification
from computeruse.manual import Gap, GapKind
from computeruse.profile import ActionKind, load_profile
from computeruse.recording import DiscoveryTrace, field
from computeruse.replay import MemoryReplayLog, NodeVisited, Reason, Status, replay


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


def receipt(text):
    return AxNode("definition", text, tag="dd", attributes=(("id", "receipt"),))


def recorded(
    profile,
    text="O-900 · Vacation reserve",
    *,
    safe_text=None,
    inputs=None,
    collect=False,
    source=None,
    evidence=None,
    observation=None,
    direct=False,
    task=None,
    verification=Verification.EXECUTOR,
    extra_checks=(),
):
    inputs = inputs or {"nickname": "Vacation reserve"}
    source = source or DomLocator("dd", DomAttribute.ID, "receipt")
    state = (
        observation.page_state
        if observation
        else PageState("https://bank.test/members")
    )
    observation = observation or Observation(
        "receipt",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(receipt(text),),
    )
    trace = DiscoveryTrace()
    trace.look(observation)
    trace.action(
        1,
        Action(ActionKind.READ, source, evidence=evidence),
        ActionResult(Outcome.OK, state, text),
        state.location,
        observation,
    )
    if not direct:
        trace.kept(
            Fact(
                key="created",
                value="O-900",
                step=1,
                route=profile.scope.route(state.location),
                source=source,
                may_change=False,
                location=state.location,
                partial=True,
                record=evidence,
            )
        )
    result = RunResult(
        Ending.COMPLETED,
        2,
        "done",
        outputs={"number": "O-900"},
        verification=verification,
        task=task or Task(),
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.RESULT,
                    source if direct else FactRef("created"),
                    "O-900",
                    match=CheckMatch.CONTAINS if direct else CheckMatch.EQUALS,
                    output="number",
                    record=evidence if direct else None,
                ),
                "O-900",
                True,
            ),
            *extra_checks,
        ),
    )
    return trace.build(
        result,
        profile=profile,
        inputs=inputs,
        contract=Contract(2, tuple(field(name) for name in inputs), (field("number"),)),
        capability_id="composite",
        run="scripted",
        safe_text=frozenset({"receipt", " · "} if safe_text is None else safe_text),
        collect=collect,
    )


@pytest.mark.rule(3, 9, 16)
@pytest.mark.parametrize(
    ("answer", "changed", "replaced", "takeover", "succeeds"),
    [
        (HandoffOutcome.APPROVED, False, False, "", True),
        (HandoffOutcome.RESUMED, False, False, "", False),
        (HandoffOutcome.REJECTED, False, False, "", False),
        (HandoffOutcome.APPROVED, True, False, "", False),
        (HandoffOutcome.APPROVED, False, True, "", False),
        (HandoffOutcome.APPROVED, False, False, "postchecks", False),
        (HandoffOutcome.APPROVED, False, False, "final_checks", False),
        (HandoffOutcome.APPROVED, False, False, "final_live", False),
    ],
)
def test_missing_result_values_are_confirmed_with_current_inputs(
    profile, monkeypatch, answer, changed, replaced, takeover, succeeds
):
    recording = recorded(
        profile,
        inputs={"nickname": "Vacation reserve", "delivery": "Paper"},
        verification=Verification.PERSON,
        task=Task(
            requirements=(
                TaskRequirement("nickname", "Vacation reserve"),
                TaskRequirement("delivery", "Paper"),
            )
        ),
        extra_checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.REQUIREMENT,
                    DomLocator("dd", DomAttribute.ID, "receipt"),
                    "Vacation reserve",
                    match=CheckMatch.CONTAINS,
                    requirement="nickname",
                ),
                "O-900 · Vacation reserve",
                True,
            ),
        ),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    saved = dumps(recording.capability)
    assert "Vacation reserve" not in saved
    assert "Paper" not in saved
    capability = loads(saved)
    confirmation = capability.node("confirm")
    assert isinstance(confirmation, ResultConfirmationNode)
    assert confirmation.confirm_inputs == (ref(RefKind.INPUT, "delivery"),)
    capability = dataclasses.replace(
        capability,
        provenance=dataclasses.replace(capability.provenance, review=Review.REVIEWED),
    )
    app = App(
        {"receipt": Page("/members", (receipt("O-901 · Rainy day"),))},
        "receipt",
        {},
    )
    clock = Clock()
    log = MemoryReplayLog()
    approved = False
    taken = False
    final_read = False

    def respond(_request):
        nonlocal approved
        approved = answer is HandoffOutcome.APPROVED
        if replaced:
            app.screens["receipt"] = Page(
                "/members", (receipt("O-902 · Another customer"),)
            )
        return answer

    people = People([respond, HandoffOutcome.RESUMED])
    control = people.control(clock)
    intervene = control.intervene

    def handoff(request, checks=None):
        return dataclasses.replace(intervene(request, checks), changed=changed)

    def current_node():
        return next(
            (
                event.node
                for event in reversed(log.events)
                if isinstance(event, NodeVisited)
            ),
            "",
        )

    def read_result(_app, action):
        nonlocal final_read
        if current_node() == "done" and action.kind is ActionKind.READ:
            final_read = True

    halted = control.halted

    def take_session():
        nonlocal taken
        node = current_node()
        due = (
            (takeover == "postchecks" and node == "confirm")
            or (takeover == "final_checks" and node == "done" and not final_read)
            or (takeover == "final_live" and node == "done" and final_read)
        )
        if approved and not taken and due:
            taken = True
            previous = control.hand_back
            intervened = intervene(
                InterventionRequest(
                    trigger=Trigger.NO_PROGRESS,
                    goal="inspect the result",
                    profile_id=profile.profile_id,
                    step=0,
                    route="/members",
                    reason="operator takes and returns the unchanged receipt",
                    timeout_s=10,
                    ask=Ask.PERSON,
                    mode=Mode.REPLAY,
                )
            )
            assert intervened.outcome is HandoffOutcome.RESUMED
            assert control.hand_back is not previous
        return halted()

    monkeypatch.setattr(control, "intervene", handoff)
    monkeypatch.setattr(control, "halted", take_session)
    app.on_act = read_result
    result = replay(
        capability,
        {"nickname": "Rainy day", "delivery": "Electronic"},
        profile=profile,
        surface=app,
        control=control,
        log=log,
        clock=clock,
        sleep=clock.advance,
    )
    assert (result.status is Status.SUCCEEDED) is succeeds
    request = people.requests[0]
    assert request.ask is Ask.APPROVAL
    assert "required values: delivery=Electronic." in request.reason
    assert "Paper" not in request.reason
    if succeeds:
        assert dict(result.outputs) == {"number": "O-901"}
    else:
        assert not result.outputs
    if takeover:
        assert taken
        assert people.requests[1].ask is Ask.PERSON
        assert app.screens["receipt"].nodes == (receipt("O-901 · Rainy day"),)
    assert "Electronic" not in repr(log.events)


@pytest.mark.rule(14, 16)
@pytest.mark.parametrize("gap_stage", ["none", "initial", "later"])
def test_result_confirmation_requires_complete_initial_and_later_approvals(
    profile, monkeypatch, gap_stage
):
    recording = recorded(
        profile,
        inputs={"nickname": "Vacation reserve", "delivery": "Paper"},
        verification=Verification.PERSON,
        task=Task(requirements=(TaskRequirement("delivery", "Paper"),)),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = loads(dumps(recording.capability))
    capability = dataclasses.replace(
        capability,
        provenance=dataclasses.replace(capability.provenance, review=Review.REVIEWED),
    )
    app = App(
        {"receipt": Page("/members", (receipt("O-901 · Rainy day"),))},
        "receipt",
        {},
    )
    clock = FakeClock()
    seat = ScriptedSeat(clock, location=app.location())

    class Window:
        def attach(self, control):
            del control

        def show(self, status):
            del status

        def listening(self):
            return True

    def record_initial(session):
        if gap_stage == "initial":
            session.gaps.append(Gap(GapKind.NOT_DRAINED))

    def record_later(session):
        if gap_stage == "later":
            session.gaps.append(Gap(GapKind.NOT_DRAINED))

    seat.moves = [
        doing(State.AWAITING_APPROVAL, record_initial),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
        doing(State.AWAITING_APPROVAL, record_later),
        when(State.AWAITING_APPROVAL, Command.APPROVE),
    ]
    control = Control(
        mode=Mode.REPLAY, clock=clock, seat=seat, channels=[Window()], worker=False
    )
    seat.control = control
    halted = control.halted
    later = False

    def approve_later_read():
        nonlocal later
        if gap_stage != "initial" and control.segments and not later:
            later = True
            control.intervene(
                InterventionRequest(
                    trigger=Trigger.RISKY_ACTION,
                    goal="read the result",
                    profile_id=profile.profile_id,
                    step=0,
                    route="/members",
                    reason="approve a read without changing the result",
                    timeout_s=10,
                    action=Action(
                        ActionKind.READ, DomLocator("dd", DomAttribute.ID, "receipt")
                    ),
                    ask=Ask.APPROVAL,
                    mode=Mode.REPLAY,
                )
            )
        return halted()

    monkeypatch.setattr(control, "halted", approve_later_read)
    result = replay(
        capability,
        {"nickname": "Rainy day", "delivery": "Electronic"},
        profile=profile,
        surface=app,
        control=control,
        log=MemoryReplayLog(),
        clock=clock,
        sleep=clock.advance,
    )
    assert later is (gap_stage != "initial")
    segment = control.segments[0 if gap_stage == "initial" else 1]
    assert segment.ask is Ask.APPROVAL
    assert not segment.taken
    assert not segment.steps
    assert segment.complete is (gap_stage == "none")
    assert (result.status is Status.SUCCEEDED) is (gap_stage == "none")
    assert dict(result.outputs) == ({"number": "O-901"} if gap_stage == "none" else {})
    assert app.screens["receipt"].nodes == (receipt("O-901 · Rainy day"),)


@pytest.mark.rule(3, 9, 16)
@pytest.mark.parametrize(
    ("inputs", "records", "owner", "requirements"),
    [
        ({"nickname": "Vacation reserve"}, (), "", ("delivery",)),
        (
            {"nickname": "Vacation reserve", "delivery": "Paper", "copy": "Paper"},
            (),
            "",
            ("delivery",),
        ),
        (
            {
                "nickname": "Vacation reserve",
                "delivery": "Paper",
                "member": "M-100",
                "other": "M-200",
            },
            (TaskRecord("member", "M-100"), TaskRecord("other", "M-200")),
            "member",
            ("delivery",),
        ),
        (
            {"nickname": "Vacation reserve", "delivery": "Paper", "member": "M-100"},
            (TaskRecord("member", "M-100"),),
            "unknown",
            ("delivery",),
        ),
        (
            {"nickname": "Vacation reserve", "delivery": "Paper"},
            (),
            "",
            ("delivery", "confirmation_method"),
        ),
    ],
)
def test_a_missing_result_value_without_one_input_or_record_prevents_export(
    profile, inputs, records, owner, requirements
):
    recording = recorded(
        profile,
        inputs=inputs,
        verification=Verification.PERSON,
        task=Task(
            records=records,
            requirements=tuple(
                TaskRequirement(name, "Paper", of=owner) for name in requirements
            ),
        ),
    )
    assert not recording.complete
    assert any(issue.gap.value == "unrepresentable_check" for issue in recording.issues)


@pytest.mark.rule(8, 9)
def test_unknown_surrounding_data_is_not_saved(profile):
    recording = recorded(profile, "O-900 · Customer detail")
    assert not recording.complete


@pytest.mark.rule(8)
def test_a_new_composite_boundary_needs_record_text_confirmation(profile):
    from computeruse.recorder import Place

    recording = recorded(profile, safe_text={"receipt"}, collect=True)
    assert recording.capability is not None
    candidate = next(item for item in recording.candidates if item.text == " · ")
    assert candidate.places == frozenset({Place.RECORD})


@pytest.mark.rule(3, 9, 11)
@pytest.mark.parametrize(
    "boundary", [ref(RefKind.SECRET, "teller_pin"), ref(RefKind.OUTPUT, "number")]
)
def test_composite_boundaries_cannot_read_a_secret_or_their_own_output(
    profile, boundary
):
    capability = recorded(profile).capability
    assert capability is not None
    read = next(n for n in capability.nodes if isinstance(n, ActionNode))
    changed = dataclasses.replace(read, extraction=Extraction((boundary,), ()))
    capability = dataclasses.replace(
        capability,
        nodes=tuple(changed if n == read else n for n in capability.nodes),
    )
    assert IssueCode.INVALID_REFERENCE in {issue.code for issue in validate(capability)}


@pytest.mark.rule(11)
@pytest.mark.parametrize("kind", [ActionKind.CLICK, ActionKind.TYPE])
def test_only_a_structured_read_may_extract_a_value(profile, kind):
    capability = recorded(profile).capability
    assert capability is not None
    read = next(n for n in capability.nodes if isinstance(n, ActionNode))
    changed = dataclasses.replace(
        read, kind=kind, value=constant("test") if kind is ActionKind.TYPE else None
    )
    capability = dataclasses.replace(
        capability,
        nodes=tuple(changed if n == read else n for n in capability.nodes),
    )
    assert IssueCode.INVALID_ACTION in {issue.code for issue in validate(capability)}


@pytest.mark.rule(2, 11)
def test_composite_extraction_preserves_signs_and_identifier_boundaries():
    assert matching.extract("Due -4.00 total", "Due ", " total") == "-4.00"
    assert matching.extract("Order O-91-02 created", "Order ", " created") == "O-91-02"
    assert matching.extract("Order O-91-02 created", "Order ", "-02 created") is None
    assert matching.extract("Due (4.00) total", "Due ", " total") is None


@pytest.mark.rule(3, 7, 9, 11)
def test_a_composite_list_read_keeps_its_record_when_nicknames_repeat(pages):
    from computeruse.actions import (
        ObservationRequest,
        RecordEvidence,
        Relation,
        Scope,
        ScopeKind,
    )

    markup = """<table><tr><th>Order</th><th>Customer</th></tr>
      <tr><td class="number">O-900 · Vacation reserve</td>
      <td class="member">M-11</td></tr>
      <tr><td class="number">O-899 · Vacation reserve</td>
      <td class="member">M-10</td></tr></table>"""
    with pages({"/": markup}) as (surface, profile):
        observation = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        scope = Scope(ScopeKind.ROW, "M-11", column=1)
        source = DomLocator("td", DomAttribute.CSS_CLASS, "number", scope=scope)
        identity = DomLocator("td", DomAttribute.CSS_CLASS, "member", scope=scope)
        evidence = RecordEvidence(identity, "M-11", Relation.ROW)
        recording = recorded(
            profile,
            source=source,
            evidence=evidence,
            observation=observation,
            inputs={"member": "M-11", "nickname": "Vacation reserve"},
            safe_text={"number", "member", " · "},
        )
        assert recording.complete, (recording.issues, recording.artifact_issues)
        capability = loads(dumps(recording.capability))
        read = next(node for node in capability.nodes if isinstance(node, ActionNode))
        assert read.record is not None
        assert read.record.value == ref(RefKind.INPUT, "member")
        assert read.record.relation is Relation.ROW
        capability = dataclasses.replace(
            capability,
            provenance=dataclasses.replace(
                capability.provenance, review=Review.REVIEWED
            ),
        )
        surface._page.locator("table").evaluate(
            "el => {el.innerHTML = '<tr><th>Order</th><th>Customer</th></tr>' +"
            "'<tr><td class=number>O-902 · Rainy day</td>' +"
            "'<td class=member>M-13</td></tr>' +"
            "'<tr><td class=number>O-901 · Rainy day</td>' +"
            "'<td class=member>M-12</td></tr>';}"
        )
        clock = Clock()
        person = People([])
        result = replay(
            capability,
            {"member": "M-12", "nickname": "Rainy day"},
            profile=profile,
            surface=surface,
            control=person.control(clock),
            log=MemoryReplayLog(),
            clock=clock,
            sleep=clock.advance,
        )
        assert result.status is Status.SUCCEEDED, result
        assert dict(result.outputs) == {"number": "O-901"}


@pytest.mark.rule(2, 3, 7, 9, 11)
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "changed_input",
        "duplicate_source",
        "duplicate_input",
        "duplicate_product",
        "changed_boundary",
        "duplicate_boundary",
        "missing_output",
    ],
)
def test_a_composite_paragraph_uses_its_explicit_record_identity(pages, case, direct):
    from computeruse.actions import ObservationRequest, RecordEvidence, Relation
    from computeruse.capability import Match, StructuralTarget

    original = "Created Reserve order O-900 for member M-11."
    markup = f"<div><p>{original}</p></div><div><p>No other orders.</p></div>"
    if case == "duplicate_source":
        markup += "<div><p>Another Reserve order for member M-11.</p></div>"
    with pages({"/": markup}) as (surface, profile):
        observation = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        source = AxLocator("paragraph", original)
        evidence = RecordEvidence(
            source,
            "M-11",
            Relation.CONTAINER,
            prefix="Created Reserve order O-900 for member ",
            suffix=".",
        )
        inputs = {"member": "M-11", "product": "Reserve"}
        if case == "duplicate_input":
            inputs["other"] = "M-11"
        if case == "duplicate_product":
            inputs["other_product"] = "Reserve"
        boundaries = frozenset({"Created ", " order ", " for member ", "."})
        recording = recorded(
            profile,
            text=original,
            source=source,
            evidence=evidence,
            observation=observation,
            inputs=inputs,
            safe_text=boundaries,
            direct=direct,
        )
        if case in {"duplicate_source", "duplicate_input", "duplicate_product"}:
            assert not recording.complete
            if case == "duplicate_product":
                assert any(
                    issue.gap.value == "unrepresentable_step"
                    for issue in recording.issues
                )
            return
        assert recording.complete, (recording.issues, recording.artifact_issues)
        serialized = dumps(recording.capability)
        assert all(value not in serialized for value in (*inputs.values(), "O-900"))
        capability = loads(serialized)
        read = next(node for node in capability.nodes if isinstance(node, ActionNode))
        target = next(
            item for item in capability.targets if item.target_id == read.target
        )
        assert isinstance(target, StructuralTarget)
        assert target.name == ref(RefKind.INPUT, "member")
        assert target.match is Match.CONTAINS
        assert read.record is not None
        assert read.record.value == ref(RefKind.INPUT, "member")
        capability = dataclasses.replace(
            capability,
            provenance=dataclasses.replace(
                capability.provenance, review=Review.REVIEWED
            ),
        )
        updated = "Created Everyday order O-901 for member M-12."
        if case == "changed_boundary":
            updated = "Created Something else order O-901 for member M-12."
        elif case == "duplicate_boundary":
            updated = (
                "Created Everyday order Created Everyday order O-901 for member M-12."
            )
        elif case == "missing_output":
            updated = "Created Everyday order  for member M-12."
        surface._page.locator("p").first.evaluate(
            "(element, value) => {element.textContent = value}", updated
        )
        clock = Clock()
        result = replay(
            capability,
            {"member": "M-12", "product": "Everyday"},
            profile=profile,
            surface=surface,
            control=People([]).control(clock),
            log=MemoryReplayLog(),
            clock=clock,
            sleep=clock.advance,
            probing=boundaries,
        )
        if case in {"changed_boundary", "duplicate_boundary", "missing_output"}:
            assert result.status is Status.FAILED
            assert not result.outputs
            assert result.reason is Reason.OUTPUT_INVALID
        else:
            assert result.status is Status.SUCCEEDED, result
            assert dict(result.outputs) == {"number": "O-901"}
            assert boundaries <= result.seen
