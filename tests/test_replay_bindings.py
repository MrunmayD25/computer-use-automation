"""Regressions for the third review of PR #6, against scripted surfaces.

The three sections cover separate findings. Completion evidence must belong
to the intended record. A person's step that replaces an uncertain operation
keeps that operation's restrictions. Replay judges every look, capture, and
internal read again after approval. The scripted surfaces enforce nothing by
themselves, so each test isolates replay or recorder behavior from adapter
protections. Every fixture is synthetic.
"""

from __future__ import annotations

import dataclasses

import pytest
from replay_fakes import (
    MEMBER,
    PROFILE,
    Captures,
    Page,
    People,
    bank,
    crop,
    dd,
    limits,
    png,
    status,
    transfer_capability,
)
from test_recorder import (
    AMOUNT_STEP,
    SUBMIT_OPERATION,
    SUBMIT_STEP,
    SUBMITTED,
    TRANSFER,
    present,
    record,
    transfer_events,
)
from test_recorder import (
    ax as sample,
)
from test_replay import canvas_capability, desk, run
from test_replay_regressions import always, approve, ax, load, replace_node

from computeruse.actions import Operation, Outcome, Relation
from computeruse.capability import (
    ActionNode,
    HelpReason,
    HumanNode,
    Present,
    RecordSpec,
    RestrictionScope,
    ResultKind,
    ResultNode,
    SavedRestriction,
    validate,
)
from computeruse.escalation import Ask, HandoffOutcome
from computeruse.policy import Restriction, Source
from computeruse.profile import ActionKind, Limit
from computeruse.recorder import Dispatch, Gap, HumanSegment, Learned, RecordingIssue
from computeruse.replay import Actor, Delivery, Reason, Status


@pytest.fixture
def profile(tmp_path):
    return load(tmp_path)


def status_only():
    """The transfer, with a final check that reads only the submission status.

    Nothing after the submission repeats the member check, so only the
    submission's own evidence can catch a confirmation for another member.
    """
    capability = transfer_capability()
    return dataclasses.replace(
        capability,
        nodes=tuple(
            ResultNode("done", ResultKind.SUCCESS, "", (Present("submitted"),))
            if node.node_id == "done"
            else node
            for node in capability.nodes
        ),
    )


def with_other_confirmation(app, member="10002", *, number=True):
    nodes = (status("Transfer submitted"),)
    if number:
        nodes = (dd("Member number", member, "member-number"), *nodes)
    app.screens["other"] = Page(f"/members/{member}/transfer", nodes)
    return app


def submitted(result) -> list[tuple[Delivery, Actor | None]]:
    return [(a.delivery, a.actor) for a in result.attempts if a.node == "submit"]


# 1. Completion evidence belongs to the intended record.


@pytest.mark.parametrize(
    ("member", "number"),
    [("10001", False), ("10002", True)],
    ids=["missing", "wrong"],
)
def test_a_confirmation_without_the_intended_record_cannot_settle_approval(
    profile, member, number
) -> None:
    """Regression: a 10002 confirmation completed the 10001 submission."""
    app = with_other_confirmation(bank(), member, number=number)

    def elsewhere(request):
        assert request.ask is Ask.APPROVAL
        app.state = "other"
        return HandoffOutcome.APPROVED

    people = People([elsewhere, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED])

    result, _ = run(status_only(), app, people, profile)

    assert result.status is not Status.SUCCEEDED
    assert result.status is Status.NEEDS_HELP
    assert (Delivery.COMPLETED, Actor.PERSON) not in submitted(result)
    assert app.acted_on("Submit transfer") == 0


def test_another_members_confirmation_never_settles_an_uncertain_submission(
    profile,
) -> None:
    app = with_other_confirmation(bank())
    # The submission times out, and the page it leaves shows another member.
    app.rules[("transfer", ActionKind.CLICK, "Submit transfer")] = "other"
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.UNCERTAIN, True)]
    people = People([approve, HandoffOutcome.RESUMED])

    result, _ = run(status_only(), app, people, profile)

    # A decline reads the page again. Another member's confirmation settles
    # nothing, so the replay stops rather than asking again.
    assert result.reason is Reason.DELIVERY_UNCERTAIN
    assert submitted(result) == [(Delivery.UNCERTAIN, Actor.AUTOMATION)]
    assert app.acted_on("Submit transfer") == 1
    assert [r.reason for r in result.history][1:] == [Reason.DELIVERY_UNCERTAIN]


def receipt_capability(*, bound: bool):
    """Submit, and land on the member screen with a receipt."""
    base = status_only()
    receipt = ax("receipt", "/members/:id", "status", "Receipt issued")
    spec = RecordSpec("member_shown", MEMBER, Relation.CONTAINER) if bound else None
    capability = replace_node(
        dataclasses.replace(base, targets=(*base.targets, receipt)),
        "submit",
        verify=(Present("receipt"),),
        result_record=spec,
    )
    return dataclasses.replace(
        capability,
        nodes=tuple(
            ResultNode("done", ResultKind.SUCCESS, "", (Present("receipt"),))
            if node.node_id == "done"
            else node
            for node in capability.nodes
        ),
    )


def receipt_bank(member="10001"):
    app = bank()
    app.screens["receipt"] = Page(
        f"/members/{member}",
        (dd("Member number", member, "member-number"), status("Receipt issued")),
    )
    app.rules[("transfer", ActionKind.CLICK, "Submit transfer")] = "receipt"
    return app


@pytest.mark.parametrize(
    ("bound", "member", "status", "reason", "delivery"),
    [
        (True, "10001", Status.SUCCEEDED, Reason.COMPLETED, Delivery.COMPLETED),
        (
            False,
            "10001",
            Status.NEEDS_HELP,
            Reason.DELIVERY_UNCERTAIN,
            Delivery.UNCERTAIN,
        ),
        (
            True,
            "10002",
            Status.NEEDS_HELP,
            Reason.DELIVERY_UNCERTAIN,
            Delivery.UNCERTAIN,
        ),
    ],
    ids=["bound", "unbound", "wrong-record"],
)
def test_a_navigating_submission_requires_its_intended_result_record(
    profile, bound, member, status, reason, delivery
) -> None:
    capability = receipt_capability(bound=bound)
    assert validate(capability) == ()
    app = receipt_bank(member)
    people = People([approve, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED])

    result, _ = run(capability, app, people, profile)

    assert result.status is status
    assert result.reason is reason
    assert app.acted_on("Submit transfer") == 1
    assert submitted(result) == [(delivery, Actor.AUTOMATION)]


# 2. A person's step that replaces an uncertain operation keeps its restrictions.


def takeover(events, index, outcome, *, reason=HelpReason.MANUAL_STEP, step=None):
    """Make the event at ``index`` uncertain and insert a person's takeover."""
    event = events[index]
    events[index] = dataclasses.replace(event, outcome=outcome)
    checks = (present(SUBMITTED),) if event.step == 8 else ()
    if event.step == 6:
        checks = (present(sample(TRANSFER, "textbox", "Amount")),)
    at = event.step if step is None else step
    events.insert(index + 1, HumanSegment(at, event.route, reason, checks))
    return events


def deny_submission(events):
    denied = Restriction(
        SUBMIT_OPERATION, "submit_transfer", Limit.DENY, Source.OPERATOR, 8
    )
    events.insert(-1, Learned(denied))
    return events


@pytest.mark.parametrize("outcome", [Outcome.UNCERTAIN, Outcome.SURFACE_ERROR])
def test_a_denied_operation_a_person_took_over_is_not_recorded(
    profile, outcome
) -> None:
    """Regression: the takeover's human node lost the denied operation."""
    events = deny_submission(takeover(transfer_events(), SUBMIT_STEP, outcome))

    recording = record(profile, events)

    assert not recording.complete
    assert RecordingIssue(Gap.DENIED_STEP, 8) in recording.issues


def test_a_risky_operation_a_person_took_over_stays_a_persons_step(profile) -> None:
    events = takeover(transfer_events(), SUBMIT_STEP, Outcome.UNCERTAIN)

    capability = record(profile, events).capability
    assert capability is not None

    step = next(n for n in capability.nodes if isinstance(n, HumanNode))
    assert (step.performs, step.effect) == (ActionKind.CLICK, "submit_transfer")
    assert step.mandatory
    assert step.transitions[0].when
    assert validate(capability) == ()


def test_permitted_work_a_person_took_over_is_recorded_with_its_check(
    profile,
) -> None:
    events = takeover(transfer_events(), AMOUNT_STEP, Outcome.UNCERTAIN)

    capability = record(profile, events).capability
    assert capability is not None

    (step,) = (n for n in capability.nodes if isinstance(n, HumanNode))
    assert step.performs is ActionKind.TYPE
    assert step.reason is HelpReason.MANUAL_STEP
    assert len(step.transitions[0].when) == 1
    typed = [
        n
        for n in capability.nodes
        if isinstance(n, ActionNode) and n.kind is ActionKind.TYPE
    ]
    assert len(typed) == 2


def test_authentication_right_after_an_uncertain_operation_does_not_replace_it(
    profile,
) -> None:
    events = takeover(
        transfer_events(),
        AMOUNT_STEP,
        Outcome.UNCERTAIN,
        reason=HelpReason.AUTHENTICATION,
    )

    recording = record(profile, events)

    assert RecordingIssue(Gap.UNCERTAIN_DELIVERY, 6) in recording.issues


def test_a_person_at_another_step_does_not_replace_the_uncertain_operation(
    profile,
) -> None:
    events = takeover(transfer_events(), AMOUNT_STEP, Outcome.UNCERTAIN, step=5)

    recording = record(profile, events)

    assert RecordingIssue(Gap.UNCERTAIN_DELIVERY, 6) in recording.issues


def test_a_takeover_of_two_different_operations_is_ambiguous(profile) -> None:
    events = takeover(transfer_events(), SUBMIT_STEP, Outcome.UNCERTAIN)
    press = dataclasses.replace(
        events[SUBMIT_STEP],
        kind=ActionKind.PRESS_KEY,
        value=None,
        target=None,
        effect="submit_transfer",
        dispatch=Dispatch.UNKNOWN,
    )
    events.insert(SUBMIT_STEP, press)

    recording = record(profile, events)

    assert RecordingIssue(Gap.AMBIGUOUS_TAKEOVER, 8) in recording.issues


def test_a_deny_on_another_control_there_still_reaches_the_persons_step(
    profile,
) -> None:
    other = Operation(ActionKind.CLICK, TRANSFER, "|button|button|Other", "d3:c11")
    events = takeover(transfer_events(), SUBMIT_STEP, Outcome.UNCERTAIN)
    events.insert(-1, Learned(Restriction(other, None, Limit.DENY, Source.FINDING, 8)))

    recording = record(profile, events)

    # A deny on a different control on the same screen reaches the person's
    # step only through its saved form, which covers every click there.
    assert not recording.complete
    assert RecordingIssue(Gap.DENIED_STEP, 8) in recording.issues


# 3. Looks, captures, and completion reads are judged again after approval.


def observe_restriction(route: str, limit: Limit) -> SavedRestriction:
    return SavedRestriction(
        RestrictionScope.ROUTE,
        route,
        (ActionKind.OBSERVE,),
        None,
        "",
        limit,
        Source.OPERATOR,
    )


def test_a_saved_observation_deny_stops_every_look_there(profile) -> None:
    """Regression: a saved deny on observe was ignored by the replay's looks."""
    capability = transfer_capability(
        restrictions=(observe_restriction("/members", Limit.DENY),)
    )
    assert validate(capability) == ()
    app = bank()

    result, _ = run(capability, app, People([]), profile)

    assert result.reason is Reason.RESTRICTION_DENIED
    assert app.looks == 0
    assert app.acted == []


def test_a_saved_risky_observation_asks_before_each_look(profile) -> None:
    capability = transfer_capability(
        restrictions=(observe_restriction("/members", Limit.RISKY),),
        limits=limits(max_help_requests=100),
    )
    app = bank()
    people = People([always(HandoffOutcome.APPROVED)] * 100)

    result, _ = run(capability, app, people, profile)

    looks = [
        r for r in people.requests if r.action and r.action.kind.value == "observe"
    ]
    assert result.status is Status.SUCCEEDED
    assert len(looks) == app.observed_at.count("/members")
    assert looks


def test_a_saved_risky_observation_asks_before_each_capture(profile) -> None:
    recorded = png((200, 120), ((30, 40),))
    capability = dataclasses.replace(
        canvas_capability(crop(recorded, 30, 40, 40, 20)),
        restrictions=(observe_restriction("/desk", Limit.RISKY),),
        limits=limits(max_help_requests=100),
    )
    captures = Captures([recorded])
    app = desk()
    app.rules[("start", ActionKind.CLICK, "screen")] = "b"
    people = People([always(HandoffOutcome.APPROVED)] * 100)

    result, _ = run(capability, app, people, profile, inputs={}, scenes=captures)

    looks = [
        r for r in people.requests if r.action and r.action.kind.value == "observe"
    ]
    assert result.status is Status.SUCCEEDED
    assert captures.taken >= 1
    assert len(looks) == app.looks + captures.taken


def risky_looks(tmp_path):
    return load(
        tmp_path, PROFILE.replace("observe: {any: safe}", "observe: {any: risky}")
    )


def test_a_look_approved_before_the_page_left_the_profile_is_not_taken(
    tmp_path,
) -> None:
    """Regression: the replay observed /forbidden after an approval for /members."""
    app = bank()
    app.screens["forbidden"] = Page("/forbidden", app.screens["search"].nodes)

    def leave(request):
        app.state = "forbidden"
        return approve(request)

    result, _ = run(transfer_capability(), app, People([leave]), risky_looks(tmp_path))

    assert "/forbidden" not in app.observed_at
    assert app.looks == 0
    assert result.status is Status.FAILED


def roomy():
    """The transfer, allowed enough requests for a check read's approval.

    A check that compares text is a read (rule 19), so with reads declared
    risky the search step's check asks once too.
    """
    capability = transfer_capability()
    limits = dataclasses.replace(capability.limits, max_help_requests=6)
    return dataclasses.replace(capability, limits=limits)


def risky_reads(tmp_path):
    return load(tmp_path, PROFILE.replace("read: {any: safe}", "read: {any: risky}"))


def read_requests(people):
    return [
        r
        for r in people.requests
        if r.route == "/members/:id/transfer"
        and r.action
        and r.action.kind is ActionKind.READ
    ]


def test_a_completion_read_whose_control_changed_during_approval_asks_again(
    tmp_path,
) -> None:
    app = bank()

    def redraw_at_the_end(request):
        if (
            app.state == "confirmed"
            and request.action is not None
            and request.action.kind is ActionKind.READ
            and not read_requests(people)[1:]
        ):
            app.screens["confirmed"].nodes = tuple(
                dataclasses.replace(node, control="c21")
                if node.name == "10001"
                else node
                for node in app.screens["confirmed"].nodes
            )
        return HandoffOutcome.APPROVED

    people = People([redraw_at_the_end] * 6)
    capability = transfer_capability(limits=limits(max_help_requests=6))

    result, _ = run(capability, app, people, risky_reads(tmp_path))

    assert result.status is Status.SUCCEEDED
    assert [r.reason for r in result.history if r.node == "done"] == [
        Reason.APPROVAL_REQUIRED,
        Reason.APPROVAL_INVALIDATED,
    ]


def test_a_completion_read_is_not_made_after_the_page_moved(tmp_path) -> None:
    app = bank()
    reads_before: list[int] = []

    def move_at_the_end(request):
        if (
            app.state == "confirmed"
            and request.action is not None
            and request.action.kind is ActionKind.READ
        ):
            reads_before.append(sum(a.kind is ActionKind.READ for a in app.acted))
            app.state = "member"
        return HandoffOutcome.APPROVED

    people = People([move_at_the_end] * 6)

    result, _ = run(roomy(), app, people, risky_reads(tmp_path))

    reads_after = sum(a.kind is ActionKind.READ for a in app.acted)
    assert result.reason is Reason.COMPLETION_CHECK_FAILED
    assert reads_before == [reads_after]


def test_a_record_bound_node_is_unchanged_by_a_saved_look_restriction_elsewhere(
    profile,
) -> None:
    capability = transfer_capability(
        restrictions=(observe_restriction("/desk", Limit.DENY),)
    )
    submit = capability.node("submit")
    assert isinstance(submit, ActionNode)

    result, _ = run(capability, bank(), People([approve]), profile)

    assert result.status is Status.SUCCEEDED
