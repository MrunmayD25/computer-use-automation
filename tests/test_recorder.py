"""Record a capability from typed execution events.

Every event stream here is written by hand. The recorder labels the result
``synthetic``, and no test presents it as discovered.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from replay_fakes import PROFILE, crop, limits, png

from computeruse.actions import (
    DomAttribute,
    Operation,
    Outcome,
    Relation,
    Scope,
    ScopeKind,
)
from computeruse.capability import (
    ActionNode,
    Approval,
    AtRoute,
    Field,
    HelpReason,
    HumanNode,
    IssueCode,
    LocatorForm,
    Match,
    Present,
    ProvenanceKind,
    RefKind,
    RestrictionScope,
    ResultConfirmationNode,
    Review,
    Shows,
    StoragePermit,
    StructuralTarget,
    ValueType,
    constant,
    dumps,
    loads,
    ref,
    validate,
)
from computeruse.policy import Restriction, Source
from computeruse.profile import ActionKind, Limit, Risk, load_profile
from computeruse.recorder import (
    CheckKind,
    CheckSample,
    DiscoveryRun,
    Dispatch,
    Executed,
    Finished,
    Gap,
    HumanSegment,
    Learned,
    Proposed,
    Recorder,
    RecordingIssue,
    RecordSample,
    ScreenSample,
    Synthetic,
    TargetSample,
    ValueSample,
    Verified,
    Verifier,
    VisualSample,
)


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


MEMBERS = "/members"
MEMBER = "/members/:id"
TRANSFER = "/members/:id/transfer"


def ax(route, role, name, scope=None) -> TargetSample:
    return TargetSample(
        route, LocatorForm.ACCESSIBILITY, role=role, name=name, scope=scope
    )


def dom(route, element) -> TargetSample:
    return TargetSample(
        route, LocatorForm.DOM, tag="dd", attribute=DomAttribute.ID, value=element
    )


def shows(target, value) -> CheckSample:
    return CheckSample(CheckKind.SHOWS, target, value)


def present(target) -> CheckSample:
    return CheckSample(CheckKind.PRESENT, target)


INPUT_MEMBER = ValueSample(RefKind.INPUT, "member_id")
SUBMITTED = ax(TRANSFER, "status", "Transfer submitted")
SUBMIT_OPERATION = Operation(
    ActionKind.CLICK,
    TRANSFER,
    "|button|button|Submit transfer",
    "d3:c9",
    submission="d3:c4>d3:c9",
    form="d3:c4",
    submission_as="|form#transfer>|button|button|Submit transfer",
    form_as="|form#transfer",
)
LINK_OPERATION = Operation(ActionKind.CLICK, MEMBER, "|a|link|Transfer", "d2:c7")


def transfer_events():
    """What a discovery run of the transfer workflow would publish. Synthetic."""
    return [
        Verified(0, (present(ax(MEMBERS, "textbox", "Member number")),)),
        Proposed(1, ActionKind.TYPE),
        Executed(
            1,
            ActionKind.TYPE,
            MEMBERS,
            Outcome.OK,
            target=ax(MEMBERS, "textbox", "Member number"),
            value=ValueSample(RefKind.CONSTANT, text="10001"),
        ),
        Executed(
            2,
            ActionKind.CLICK,
            MEMBERS,
            Outcome.OK,
            target=ax(MEMBERS, "button", "Search"),
            effect="search_member",
            operation=Operation(
                ActionKind.CLICK, MEMBERS, "|button|button|Search", "d1:c2"
            ),
        ),
        Verified(2, (shows(dom(MEMBER, "member-number"), INPUT_MEMBER),)),
        Executed(
            3,
            ActionKind.READ,
            MEMBER,
            Outcome.OK,
            target=dom(MEMBER, "balance"),
            into=ValueSample(RefKind.OUTPUT, "balance"),
        ),
        Proposed(4, ActionKind.CLICK),
        Executed(
            4,
            ActionKind.CLICK,
            MEMBER,
            Outcome.NOT_FOUND,
            target=ax(MEMBER, "link", "Transfers"),
            effect="open_transfer",
        ),
        Executed(
            5,
            ActionKind.CLICK,
            MEMBER,
            Outcome.OK,
            target=ax(MEMBER, "link", "Transfer"),
            effect="open_transfer",
            operation=LINK_OPERATION,
        ),
        Verified(5, (CheckSample(CheckKind.AT_ROUTE, route=TRANSFER),)),
        Executed(
            6,
            ActionKind.TYPE,
            TRANSFER,
            Outcome.OK,
            target=ax(TRANSFER, "textbox", "Amount"),
            value=ValueSample(RefKind.CONSTANT, text="250.00"),
        ),
        Executed(
            7,
            ActionKind.TYPE,
            TRANSFER,
            Outcome.OK,
            target=ax(TRANSFER, "textbox", "Teller PIN"),
            value=ValueSample(RefKind.SECRET, "teller_pin"),
        ),
        Executed(
            8,
            ActionKind.CLICK,
            TRANSFER,
            Outcome.OK,
            target=ax(TRANSFER, "button", "Submit transfer"),
            effect="submit_transfer",
            record=RecordSample(
                dom(TRANSFER, "member-number"), INPUT_MEMBER, Relation.CONTAINER
            ),
            operation=SUBMIT_OPERATION,
            risk=Risk.RISKY,
            approved=True,
        ),
        Verified(8, (present(SUBMITTED),)),
        Learned(
            Restriction(
                SUBMIT_OPERATION, "submit_transfer", Limit.RISKY, Source.OPERATOR, 8
            )
        ),
        Learned(
            Restriction(
                LINK_OPERATION,
                "open_transfer",
                Limit.RISKY,
                Source.FINDING,
                5,
                "the model's words",
            )
        ),
        Finished(
            Verifier.EXECUTOR,
            (present(SUBMITTED), shows(dom(TRANSFER, "member-number"), INPUT_MEMBER)),
        ),
    ]


def recorder(
    profile, *, origin=None, outputs=None, excluded=(), safe_text=(), collect=False
) -> Recorder:
    return Recorder(
        capability_id="member_transfer",
        version=1,
        profile=profile,
        origin=origin or Synthetic(),
        safe_text=frozenset(safe_text)
        | frozenset(
            {
                "Member number",
                "Search",
                "search_member",
                "member-number",
                "balance",
                "Transfer",
                "open_transfer",
                "Amount",
                "Teller PIN",
                "Submit transfer",
                "submit_transfer",
                "Transfer submitted",
            }
        ),
        inputs=(
            Field("member_id", ValueType.DIGITS, True, 5, 10, ()),
            Field("amount", ValueType.DECIMAL, True, 1, 12, ()),
        ),
        outputs=outputs or (Field("balance", ValueType.DECIMAL, False, 1, 20, ()),),
        variables=(),
        outcomes=(),
        limits=limits(),
        samples={"member_id": "10001", "amount": "250.00"},
        excluded=excluded,
        collect=collect,
    )


def record(profile, events, **options):
    making = recorder(profile, **options)
    for event in events:
        making.record(event)
    return making.finish()


def test_executed_behaviour_becomes_a_parameterized_draft(profile) -> None:
    recording = record(profile, transfer_events())

    assert recording.complete, recording
    capability = recording.capability
    assert capability is not None
    assert capability.provenance.kind is ProvenanceKind.SYNTHETIC
    assert capability.provenance.review is Review.DRAFT
    kinds = [n.kind for n in capability.nodes if isinstance(n, ActionNode)]
    assert kinds == [
        ActionKind.TYPE,
        ActionKind.CLICK,
        ActionKind.READ,
        ActionKind.CLICK,
        ActionKind.TYPE,
        ActionKind.TYPE,
        ActionKind.CLICK,
    ]
    typed = capability.nodes[0]
    assert isinstance(typed, ActionNode)
    assert typed.value is not None
    assert (typed.value.kind, typed.value.name) == (RefKind.INPUT, "member_id")
    assert capability.secrets == ("teller_pin",)
    assert recording.proposals == 2
    assert recording.unperformed == 1


def test_a_recording_stores_no_values_ids_or_model_words(profile) -> None:
    capability = record(profile, transfer_events()).capability
    assert capability is not None

    text = dumps(capability)

    for leaked in ("10001", "250.00", "d3:c9", "d2:c7", "d3:c4", "the model's words"):
        assert leaked not in text
    assert loads(text) == capability


@pytest.mark.rule(8)
@pytest.mark.parametrize("place", ["target", "effect"])
@pytest.mark.parametrize("collect", [False, True])
def test_confirmed_text_cannot_override_known_private_values(profile, place, collect):
    private = "PRIVATE_RECORD_7462"
    exposed = f"record_{private.lower()}_ready"
    outputs = (Field("balance", ValueType.CHOICE, False, 1, 30, (private, "unknown")),)
    events = transfer_events()
    action = events[3]
    assert isinstance(action, Executed)
    events[3] = dataclasses.replace(
        action,
        target=ax(MEMBERS, "button", exposed) if place == "target" else action.target,
        effect=exposed if place == "effect" else action.effect,
    )
    making = recorder(
        profile,
        origin=DiscoveryRun("scripted"),
        outputs=outputs,
        excluded=(private,),
        safe_text=(exposed,),
        collect=collect,
    )
    for event in events:
        making.record(event)
    recording = making.finish()

    assert exposed not in {item.text for item in recording.candidates}
    if recording.capability is not None:
        assert recording.capability.outputs == outputs
        document = json.loads(dumps(recording.capability))
        document.pop("outputs")
        assert private.casefold() not in json.dumps(document).casefold()
    else:
        assert recording.issues


def test_risky_and_restricted_steps_need_a_person_on_every_replay(profile) -> None:
    capability = record(profile, transfer_events()).capability
    assert capability is not None
    by_effect = {
        n.effect: n for n in capability.nodes if isinstance(n, ActionNode) and n.effect
    }

    assert by_effect["submit_transfer"].approval is Approval.EACH_RUN
    assert by_effect["submit_transfer"].mandatory
    assert by_effect["open_transfer"].approval is Approval.EACH_RUN
    assert by_effect["search_member"].approval is Approval.NONE


def test_restrictions_lose_element_ids_but_not_scope(profile) -> None:
    capability = record(profile, transfer_events()).capability
    assert capability is not None
    saved = {(r.scope, r.route): r for r in capability.restrictions}

    submission = saved[RestrictionScope.ROUTE, TRANSFER]
    assert set(submission.kinds) == {ActionKind.CLICK, ActionKind.PRESS_KEY}
    assert submission.key == "Enter"
    assert submission.source is Source.OPERATOR
    link = saved[RestrictionScope.TARGET, MEMBER]
    assert link.kinds == (ActionKind.CLICK,)
    assert capability.target(link.target or "").route == MEMBER


@pytest.mark.rule(6)
@pytest.mark.parametrize("kind", [ActionKind.CLICK, ActionKind.READ])
def test_a_screen_restriction_preserves_its_input_or_read_scope(profile, kind) -> None:
    from computeruse.policy import LOOKS

    screen = Operation(kind, MEMBER, "screen")
    events = [
        *transfer_events()[:-1],
        Learned(Restriction(screen, None, Limit.RISKY, Source.FINDING, 3)),
        transfer_events()[-1],
    ]

    capability = record(profile, events).capability
    assert capability is not None

    (saved,) = (
        r
        for r in capability.restrictions
        if r.route == MEMBER and r.scope is RestrictionScope.ROUTE
    )
    if kind is ActionKind.CLICK:
        assert ActionKind.CLICK in saved.kinds
        assert not set(saved.kinds) & LOOKS
    else:
        assert saved.kinds == ()
    read = next(
        n
        for n in capability.nodes
        if isinstance(n, ActionNode) and n.kind is ActionKind.READ
    )
    assert read.approval is (
        Approval.NONE if kind is ActionKind.CLICK else Approval.EACH_RUN
    )


@pytest.mark.rule(9)
@pytest.mark.parametrize(
    "name",
    [
        # Inside a longer identifier, the input is not a whole word.
        "Search 100012",
        # Twice, the text names no single place to read the input from.
        "Search 10001 or 10001",
        # With another input, the text holds more than one value.
        "Send 250.00 to 10001",
    ],
)
def test_text_that_embeds_an_input_value_becomes_a_persons_step(profile, name) -> None:
    events = transfer_events()
    events[3] = dataclasses.replace(events[3], target=ax(MEMBERS, "button", name))

    capability = record(profile, events).capability
    assert capability is not None

    step = capability.nodes[1]
    assert isinstance(step, HumanNode)
    assert step.reason is HelpReason.UNREPRESENTABLE_TARGET
    assert "10001" not in dumps(capability)


@pytest.mark.rule(9)
def test_text_holding_an_input_once_is_saved_as_that_input_alone(profile) -> None:
    events = transfer_events()
    events[3] = dataclasses.replace(
        events[3], target=ax(MEMBERS, "button", "Look up (10001) now")
    )

    capability = record(profile, events).capability
    assert capability is not None

    step = capability.nodes[1]
    assert isinstance(step, ActionNode)
    target = capability.target(step.target or "")
    assert isinstance(target, StructuralTarget)
    assert target.match is Match.CONTAINS
    assert target.name == ref(RefKind.INPUT, "member_id")
    # Neither the input's value nor the words around it are written.
    assert "10001" not in dumps(capability)
    assert "Look up" not in dumps(capability)


@pytest.mark.rule(9)
@pytest.mark.parametrize("match", [Match.EQUALS, Match.CONTAINS])
def test_a_check_on_text_holding_an_input_checks_that_input_alone(
    profile, match
) -> None:
    # A member box shows more than the number typed into it. The check keeps
    # the input, once and as whole words, and never the name beside it.
    events = transfer_events()
    at = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, Verified) and event.step == 8
    )
    shown = ValueSample(RefKind.CONSTANT, text="10001 Alex Morgan")
    events[at] = Verified(
        8,
        (
            present(SUBMITTED),
            CheckSample(CheckKind.SHOWS, dom(TRANSFER, "member-number"), shown, match),
        ),
    )

    capability = record(profile, events).capability
    assert capability is not None
    checks = [
        condition
        for node in capability.nodes
        if isinstance(node, ActionNode)
        for condition in node.verify
        if isinstance(condition, Shows)
    ]
    assert any(
        check.value == ref(RefKind.INPUT, "member_id")
        and check.match is Match.CONTAINS
        and check.once
        for check in checks
    )
    assert "Alex Morgan" not in dumps(capability)


def test_a_refused_check_says_which_kind_and_why(profile) -> None:
    events = transfer_events()
    at = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, Verified) and event.step == 8
    )
    # A text holding the member number twice has no single reading.
    shown = ValueSample(RefKind.CONSTANT, text="10001 and 10001 again")
    events[at] = Verified(
        8,
        (CheckSample(CheckKind.SHOWS, dom(TRANSFER, "member-number"), shown),),
    )

    recording = record(profile, events)

    assert not recording.complete
    assert any(
        issue.detail.startswith(
            "a shows check: its literal text holds an invocation value"
        )
        for issue in recording.issues
    )


def test_an_unrepresentable_step_with_no_check_makes_the_recording_incomplete(
    profile,
) -> None:
    events = transfer_events()
    events[5] = dataclasses.replace(
        events[5], target=ax(MEMBER, "definition", "Balance for 100019")
    )

    recording = record(profile, events)

    assert not recording.complete
    assert RecordingIssue(Gap.UNREPRESENTABLE_STEP, 3) in recording.issues


def test_a_live_coordinate_is_never_a_target(profile) -> None:
    events = transfer_events()
    events[8] = dataclasses.replace(events[8], target=ScreenSample(MEMBER))

    capability = record(profile, events).capability
    assert capability is not None

    step = capability.nodes[3]
    assert isinstance(step, HumanNode)
    assert step.transitions[0].when == (AtRoute(TRANSFER),)


def test_a_visual_crop_needs_a_storage_permit(profile) -> None:
    canvas = png((200, 120), ((30, 40),))
    button_crop = crop(canvas, 30, 40, 40, 20)
    unpermitted = VisualSample(MEMBER, button_crop)
    permitted = VisualSample(MEMBER, button_crop, permit=StoragePermit.SYNTHETIC)

    def with_target(target):
        events = transfer_events()
        events[8] = dataclasses.replace(events[8], target=target)
        return events

    refused = record(profile, with_target(unpermitted)).capability
    assert refused is not None
    assert isinstance(refused.nodes[3], HumanNode)
    assert refused.templates == ()

    stored = record(profile, with_target(permitted)).capability
    assert stored is not None
    assert isinstance(stored.nodes[3], ActionNode)
    assert stored.templates[0].permit is StoragePermit.SYNTHETIC

    real = record(profile, with_target(permitted), origin=DiscoveryRun("run7"))
    assert real.capability is not None
    assert real.capability.templates == ()
    assert real.capability.provenance.kind is ProvenanceKind.DISCOVERED


def test_a_manual_detour_is_kept_as_a_persons_step(profile) -> None:
    events = transfer_events()
    code = ax(MEMBERS, "textbox", "Member number")
    events.insert(
        1, HumanSegment(0, MEMBERS, HelpReason.AUTHENTICATION, (present(code),), gaps=2)
    )

    capability = record(profile, events).capability
    assert capability is not None

    first = capability.node(capability.entry)
    assert isinstance(first, HumanNode)
    assert first.reason is HelpReason.AUTHENTICATION
    assert first.mandatory
    assert first.transitions[0].when == (Present("t1"),)


def test_a_manual_detour_with_no_check_is_not_silently_dropped(profile) -> None:
    events = transfer_events()
    events.insert(1, HumanSegment(0, MEMBERS, HelpReason.MANUAL_STEP, ()))

    recording = record(profile, events)

    assert not recording.complete
    assert {issue.gap for issue in recording.issues} == {
        Gap.MANUAL_WITHOUT_CONTINUATION
    }


def test_a_step_the_profile_requires_evidence_for_goes_to_a_person(profile) -> None:
    events = transfer_events()
    events[12] = dataclasses.replace(events[12], record=None)

    capability = record(profile, events).capability
    assert capability is not None

    last = capability.nodes[-2]
    assert isinstance(last, HumanNode)
    assert last.reason is HelpReason.RECORD_EVIDENCE


AMOUNT_STEP = 10
"""The index of the amount's typing event in ``transfer_events``."""
SUBMIT_STEP = 12
"""The index of the submission's event in ``transfer_events``."""


@pytest.mark.parametrize(
    ("outcome", "dispatch"),
    [
        (Outcome.UNCERTAIN, None),
        (Outcome.SURFACE_ERROR, None),
        (Outcome.STALE, Dispatch.SENT),
    ],
)
def test_an_uncertain_delivery_nothing_checked_is_reported(
    profile, outcome, dispatch
) -> None:
    events = transfer_events()
    events[AMOUNT_STEP] = dataclasses.replace(
        events[AMOUNT_STEP], outcome=outcome, dispatch=dispatch
    )

    recording = record(profile, events)

    assert not recording.complete
    assert RecordingIssue(Gap.UNCERTAIN_DELIVERY, 6) in recording.issues


def test_a_surface_error_known_not_to_have_dispatched_is_left_out(profile) -> None:
    events = transfer_events()
    failed = dataclasses.replace(
        events[AMOUNT_STEP], outcome=Outcome.SURFACE_ERROR, dispatch=Dispatch.NOT_SENT
    )
    events.insert(AMOUNT_STEP, failed)

    recording = record(profile, events)

    assert recording.complete, recording
    assert recording.unperformed == 2
    capability = recording.capability
    assert capability is not None
    amounts = [
        n
        for n in capability.nodes
        if isinstance(n, ActionNode)
        and n.value is not None
        and n.value.name == "amount"
    ]
    assert len(amounts) == 1


def test_an_uncertain_submission_the_run_then_verified_is_kept_with_its_check(
    profile,
) -> None:
    events = transfer_events()
    events[SUBMIT_STEP] = dataclasses.replace(
        events[SUBMIT_STEP], outcome=Outcome.UNCERTAIN
    )

    capability = record(profile, events).capability
    assert capability is not None

    submit = next(
        n
        for n in capability.nodes
        if isinstance(n, ActionNode) and n.effect == "submit_transfer"
    )
    assert len(submit.verify) == 1
    assert isinstance(submit.verify[0], Present)
    assert submit.approval is Approval.EACH_RUN


def test_a_denied_operation_is_not_turned_into_a_persons_step(profile) -> None:
    """Regression: a denied screenshot click became a scheduled human task."""
    events = transfer_events()
    events[8] = dataclasses.replace(events[8], target=ScreenSample(MEMBER))
    screen = Operation(ActionKind.CLICK, MEMBER, "screen")
    events.insert(-1, Learned(Restriction(screen, None, Limit.DENY, Source.FINDING, 5)))

    recording = record(profile, events)

    assert not recording.complete
    assert RecordingIssue(Gap.DENIED_STEP, 5) in recording.issues


def test_a_persons_step_names_the_operation_it_stands_in_for(profile) -> None:
    events = transfer_events()
    events[8] = dataclasses.replace(events[8], target=ScreenSample(MEMBER))
    screen = Operation(ActionKind.CLICK, MEMBER, "screen")
    events.insert(
        -1, Learned(Restriction(screen, None, Limit.RISKY, Source.FINDING, 5))
    )

    capability = record(profile, events).capability
    assert capability is not None

    step = capability.nodes[3]
    assert isinstance(step, HumanNode)
    assert (step.performs, step.effect) == (ActionKind.CLICK, "open_transfer")


def test_a_click_nobody_checked_fails_validation(profile) -> None:
    events = [
        e for e in transfer_events() if not (isinstance(e, Verified) and e.step == 5)
    ]

    recording = record(profile, events)

    assert {issue.gap for issue in recording.issues} == {Gap.INVALID_CAPABILITY}
    assert IssueCode.UNVERIFIED_ACTION in {i.code for i in recording.artifact_issues}


def test_a_run_that_never_finished_is_incomplete(profile) -> None:
    recording = record(profile, transfer_events()[:-1])

    assert {issue.gap for issue in recording.issues} == {Gap.NOT_FINISHED}


def test_a_choice_whose_label_holds_an_input_is_saved_as_that_input(profile) -> None:
    events = transfer_events()
    events[7] = Executed(
        6,
        ActionKind.SELECT,
        TRANSFER,
        Outcome.OK,
        target=ax(TRANSFER, "combobox", "Amount"),
        value=ValueSample(RefKind.CONSTANT, text="STD-250 250.00 standard amount"),
    )

    capability = record(profile, events).capability
    assert capability is not None

    step = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.SELECT
    )
    assert step.value == ref(RefKind.INPUT, "amount")
    assert step.value_match is Match.CONTAINS
    # The label's other words are never written, and the draft loads again.
    assert "standard" not in dumps(capability)
    assert "STD-250" not in dumps(capability)
    assert loads(dumps(capability)) == capability


def test_a_choice_saved_by_contains_takes_the_one_option_holding_the_input() -> None:
    from computeruse.actions import AxNode, SelectOption
    from computeruse.replay import Reason, _End, _option_holding

    def listing(*labels: str) -> AxNode:
        return AxNode(
            "combobox",
            "Product",
            options=tuple(SelectOption(label, label) for label in labels),
        )

    chosen = listing("USD-CHECKING USD checking", "USD-SAVINGS USD savings")
    assert _option_holding(chosen, "USD savings") == "USD-SAVINGS USD savings"
    # No option, or two, stops the replay rather than choosing the first.
    with pytest.raises(_End) as missing:
        _option_holding(listing("EUR savings"), "USD savings")
    assert missing.value.reason is Reason.TARGET_NOT_FOUND
    with pytest.raises(_End) as twice:
        _option_holding(listing("A USD savings", "B USD savings"), "USD savings")
    assert twice.value.reason is Reason.TARGET_AMBIGUOUS


@pytest.mark.parametrize("match", [Match.EQUALS, Match.CONTAINS])
def test_a_step_that_leaves_its_page_names_where_its_result_shows_the_record(
    profile, match
) -> None:
    """The submit's record is checked on the page it leads to.

    A whole-word check proves the record as well as an exact check. The
    recorder saves the same comparison.
    """
    events = transfer_events()
    at = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, Verified) and event.step == 8
    )
    events[at] = Verified(
        8,
        (
            present(SUBMITTED),
            CheckSample(
                CheckKind.SHOWS, dom(TRANSFER, "member-number"), INPUT_MEMBER, match
            ),
        ),
    )

    capability = record(profile, events).capability
    assert capability is not None
    submit = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.effect == "submit_transfer"
    )
    assert submit.result_record is not None
    assert submit.result_record.match is match
    shown = capability.target(submit.result_record.source)
    assert isinstance(shown, StructuralTarget)
    assert shown.value is not None
    assert shown.value.value == "member-number"


@pytest.mark.parametrize(
    ("shown", "match"),
    [
        (
            ax(MEMBER, "definition", "Balance", Scope(ScopeKind.ROW, "10001")),
            Match.EQUALS,
        ),
        (
            dataclasses.replace(
                dom(MEMBER, "balance"),
                scope=Scope(ScopeKind.ROW, "Balance held for 10001 today"),
            ),
            Match.CONTAINS,
        ),
    ],
)
def test_a_scoped_row_parameterizes_the_row_name(profile, shown, match) -> None:
    events = transfer_events()
    at = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, Executed) and event.kind is ActionKind.READ
    )
    events[at] = dataclasses.replace(events[at], target=shown)

    capability = record(profile, events).capability
    assert capability is not None
    read = next(
        node
        for node in capability.nodes
        if isinstance(node, ActionNode) and node.kind is ActionKind.READ
    )
    target = capability.target(read.target or "")
    assert isinstance(target, StructuralTarget)
    assert target.scope is not None
    assert target.scope.name == ref(RefKind.INPUT, "member_id")
    assert target.match is match
    assert Shows in {type(c) for c in capability.node("done").checks}
    assert "Balance held" not in dumps(capability)


def test_a_tie_a_person_confirmed_is_asked_again_on_each_replay(profile) -> None:
    """Every check passed the executor's reading. Only the tie needed a person.

    The capability keeps the checks, and a person's step before the result
    confirms the tie on each replay.
    """
    events = transfer_events()
    at = next(
        index for index, event in enumerate(events) if isinstance(event, Finished)
    )
    events[at] = dataclasses.replace(
        events[at], verifier=Verifier.PERSON, confirmed=True
    )

    capability = record(profile, events).capability
    assert capability is not None
    confirm = capability.node("confirm")
    assert isinstance(confirm, HumanNode)
    assert confirm.reason is HelpReason.RECORD_EVIDENCE
    assert confirm.transitions[0].to == "done"
    assert "confirm_inputs" not in dumps(capability)
    # Without the executor's reading of every check, nothing is saved.
    events[at] = dataclasses.replace(events[at], confirmed=False)
    unconfirmed = record(profile, events)
    assert unconfirmed.capability is None
    assert {issue.gap for issue in unconfirmed.issues} == {Gap.UNVERIFIED_RESULT}


@pytest.mark.rule(3, 9, 16)
@pytest.mark.parametrize(
    "changes",
    [
        {"confirm_inputs": ()},
        {"confirm_inputs": (constant("250.00"),)},
        {"confirm_inputs": (ref(RefKind.SECRET, "teller_pin"),)},
        {"confirm_inputs": (ref(RefKind.OUTPUT, "balance"),)},
        {"confirm_inputs": (ref(RefKind.INPUT, "undeclared"),)},
        {"confirm_inputs": (ref(RefKind.INPUT, "amount"),) * 2},
        {"mandatory": False},
        {"performs": ActionKind.CLICK},
        {"effect": "submit_transfer"},
        {"permissions": (ActionKind.TYPE,)},
        {"reason": HelpReason.MANUAL_STEP},
        {"successor": "entry"},
    ],
)
def test_result_confirmation_cannot_save_values_or_perform_an_action(
    profile, changes
) -> None:
    events = transfer_events()
    finished = events[-1]
    assert isinstance(finished, Finished)
    events[-1] = dataclasses.replace(
        finished,
        verifier=Verifier.PERSON,
        confirmed=True,
        confirm_inputs=(ref(RefKind.INPUT, "amount"),),
    )
    capability = record(profile, events).capability
    assert capability is not None
    confirmation = capability.node("confirm")
    assert isinstance(confirmation, ResultConfirmationNode)
    if "successor" in changes:
        changes = {
            "transitions": (
                dataclasses.replace(confirmation.transitions[0], to=capability.entry),
            )
        }
    invalid = dataclasses.replace(confirmation, **changes)
    changed = dataclasses.replace(
        capability,
        nodes=tuple(
            invalid if node == confirmation else node for node in capability.nodes
        ),
    )
    issues = validate(changed)
    if "transitions" in changes:
        index = capability.nodes.index(confirmation)
        assert any(
            issue.code is IssueCode.INVALID_TRANSITION
            and issue.where == f"nodes[{index}].transitions"
            for issue in issues
        )
    else:
        assert any(
            issue.code in {IssueCode.INVALID_FIELD, IssueCode.INVALID_REFERENCE}
            for issue in issues
        )


@pytest.mark.rule(6)
def test_a_persons_labelled_click_keeps_the_submission_rule() -> None:
    """Finding 46: judge a submission by its control, regardless of its label."""
    from computeruse.actions import AxLocator, Operation
    from computeruse.loop import _judged_label
    from computeruse.manual import ManualEvent, ManualKind
    from computeruse.profile import Limit, Submissions

    profile = load_profile(Path(__file__).parents[1] / "examples" / "profile.yaml")
    sends = Operation(
        ActionKind.CLICK,
        "/members",
        "|button|button|Go",
        "d1:c2",
        submission="d1:c1>d1:c2",
        form="d1:c1",
    )
    click = ManualEvent(
        1,
        1.0,
        ManualKind.CLICK,
        True,
        route="/members",
        target=AxLocator("button", "Go"),
        location="https://sandbox.example.test/members",
        operation=sends,
    )
    denied = dataclasses.replace(profile, submissions=Submissions(Limit.DENY))
    assert _judged_label(denied, click, "open_member", None) is None
    risky = dataclasses.replace(profile, submissions=Submissions(Limit.RISKY))
    assert _judged_label(risky, click, "open_member", None) is True
