"""Deterministic replay against a scripted application and scripted people.

Every capability here is a synthetic test fixture. These tests show what the
replay engine does with a surface, a profile, and an operator channel. They do
not establish support for a live application. That requires the integration
evidence listed in the capability document.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import socket
import subprocess
import sys
from typing import cast

import httpx
import pytest
from fakes import FakeClock, ScriptedSeat, doing, when
from import_contract import assert_module_import_contract
from replay_fakes import (
    BALANCE,
    INPUTS,
    MEMBER,
    PROFILE,
    App,
    Captures,
    Clock,
    Page,
    People,
    bank,
    button,
    crop,
    go,
    limits,
    png,
    status,
    transfer_capability,
)

import computeruse.capability
import computeruse.recorder
import computeruse.replay
import computeruse.replay_control
import computeruse.retarget
from computeruse.actions import (
    AxNode,
    Outcome,
    Relation,
    ScreenTarget,
    SecretRef,
    TargetForm,
)
from computeruse.capability import (
    ActionNode,
    Application,
    Approval,
    Bound,
    CheckNode,
    EdgeOrigin,
    HelpReason,
    HumanNode,
    LocatorForm,
    Present,
    RecordSpec,
    RestrictionScope,
    ResultKind,
    ResultNode,
    Review,
    SavedRestriction,
    Shows,
    StoragePermit,
    StructuralTarget,
    VisualTarget,
    constant,
    dumps,
    template,
    validate,
)
from computeruse.capability import (
    Match as TextMatch,
)
from computeruse.control import Control, ControlChanged
from computeruse.escalation import (
    Ask,
    Command,
    Handoff,
    HandoffOutcome,
    Mode,
    State,
    Trigger,
    Verdict,
)
from computeruse.journal import ManualAction, MemoryJournal, Paused
from computeruse.manual import ManualKind
from computeruse.policy import Source
from computeruse.profile import ActionKind, Limit, load_profile
from computeruse.replay import Actor, Delivery, MemoryReplayLog, Reason, Status, replay
from computeruse.replay import Ask as ReplayAsk
from computeruse.surface import SurfaceError


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


def run(capability, app, people, profile, *, inputs=None, clock=None, scenes=None):
    log = MemoryReplayLog()
    clock = clock or Clock()
    result = replay(
        capability,
        INPUTS if inputs is None else inputs,
        profile=profile,
        surface=app,
        control=people.control(clock),
        log=log,
        clock=clock,
        scenes=scenes,
        sleep=clock.advance,
    )
    return result, log


def approve(request):
    assert request.ask is Ask.APPROVAL
    return HandoffOutcome.APPROVED


# The whole workflow.


def test_replay_succeeds_with_typed_outputs_and_one_approval(profile) -> None:
    app = bank()
    people = People([approve])

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.SUCCEEDED
    assert result.reason is Reason.COMPLETED
    assert dict(result.outputs) == {"balance": "1204.50"}
    assert app.state == "confirmed"
    assert app.acted_on("Submit transfer") == 1
    assert [request.reason for request in result.history] == [Reason.APPROVAL_REQUIRED]
    submit = next(a for a in app.acted if a.effect == "submit_transfer")
    assert submit.evidence is not None
    assert submit.evidence.value == "10001"


def test_a_declared_business_outcome_is_not_a_failure(profile) -> None:
    app = bank(rules={("search", ActionKind.CLICK, "Search"): "not_found"})

    result, _ = run(transfer_capability(), app, People([]), profile)

    assert result.status is Status.OUTCOME
    assert result.outcome == "member_not_found"
    assert result.reason is Reason.BUSINESS_OUTCOME
    assert dict(result.outputs) == {}


def test_inputs_are_typed_before_anything_runs(profile) -> None:
    app = bank()

    result, _ = run(
        transfer_capability(),
        app,
        People([]),
        profile,
        inputs={"member_id": "10O01", "amount": "250.00"},
    )

    assert result.status is Status.FAILED
    assert result.reason is Reason.INVALID_INPUT
    assert app.acted == []


def test_an_unreviewed_draft_is_not_replayed(profile) -> None:
    capability = transfer_capability()
    draft = dataclasses.replace(
        capability,
        provenance=dataclasses.replace(capability.provenance, review=Review.DRAFT),
    )
    app = bank()

    result, _ = run(draft, app, People([]), profile)

    assert result.reason is Reason.UNREVIEWED
    assert app.acted == []


def test_the_application_must_match_before_the_first_node(profile) -> None:
    app = bank()
    app.state = "member"

    result, _ = run(transfer_capability(), app, People([]), profile)

    assert result.reason is Reason.INCOMPATIBLE_APPLICATION
    assert app.acted == []


def test_a_capability_for_another_profile_is_refused(profile) -> None:
    capability = transfer_capability()
    other = dataclasses.replace(
        capability,
        application=dataclasses.replace(
            capability.application, profile_id="bank/other"
        ),
    )

    result, _ = run(other, bank(), People([]), profile)

    assert result.reason is Reason.PROFILE_CONFLICT
    assert result.issues


def test_the_capability_cannot_raise_the_profile_budget(profile) -> None:
    capability = transfer_capability(limits=limits(max_steps=1000))
    result, _ = run(capability, bank(), People([approve]), profile)
    assert result.status is Status.SUCCEEDED

    tight = transfer_capability(limits=limits(max_steps=3))
    result, _ = run(tight, bank(), People([approve]), profile)
    assert result.reason is Reason.BUDGET_EXHAUSTED


# Risk and approval.


def test_a_risky_step_asks_again_on_every_replay(profile) -> None:
    capability = transfer_capability()
    first, second = bank(), bank()
    people = People([approve, approve])

    one, _ = run(capability, first, people, profile)
    two, _ = run(capability, second, people, profile)

    asks = [request for request in people.requests if request.ask is Ask.APPROVAL]
    assert len(asks) == 2
    assert one.run != two.run
    assert people.controls[0].run != people.controls[1].run
    assert len(one.history) == len(two.history) == 1
    assert first.acted_on("Submit transfer") == second.acted_on("Submit transfer") == 1


def test_resume_is_not_approval(profile) -> None:
    app = bank()
    people = People([HandoffOutcome.RESUMED, approve])

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.SUCCEEDED
    assert [request.ask for request in people.requests] == [
        Ask.APPROVAL,
        Ask.APPROVAL,
    ]
    assert app.acted_on("Submit transfer") == 1


def test_a_rejected_approval_ends_the_replay_without_acting(profile) -> None:
    app = bank()

    result, _ = run(
        transfer_capability(), app, People([HandoffOutcome.REJECTED]), profile
    )

    assert result.status is Status.TERMINATED
    assert result.reason is Reason.APPROVAL_REJECTED
    assert app.acted_on("Submit transfer") == 0


@pytest.mark.rule(5)
def test_a_replay_asked_to_stop_before_a_change_never_sends_it(profile) -> None:
    # The comparison that confirms website text stops here, so it never asks
    # anyone to approve a change and never makes one.
    app = bank()
    people = People([])
    clock = Clock()
    control = people.control(clock)

    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        log=MemoryReplayLog(),
        clock=clock,
        sleep=clock.advance,
        stop_before_change=True,
    )

    assert result.status is Status.STOPPED
    assert result.reason is Reason.STOPPED_BEFORE_CHANGE
    assert app.acted_on("Submit transfer") == 0
    assert app.acted_on("Search") == 1
    assert result.history == ()
    # A planned stop is not a failure in the window or the terminal.
    assert control.status().state is State.COMPLETED
    assert control.status().ending == "stopped"


def test_an_effect_the_profile_declares_risky_asks_for_approval_on_its_own(
    profile,
) -> None:
    # The step itself asks for no approval. Only the profile's effect does.
    capability = transfer_capability()
    unmarked = tuple(
        dataclasses.replace(node, approval=Approval.NONE, mandatory=False)
        if node.node_id == "submit"
        else node
        for node in capability.nodes
    )
    app = bank()
    people = People([HandoffOutcome.REJECTED])

    result, _ = run(
        dataclasses.replace(capability, nodes=unmarked), app, people, profile
    )

    assert [request.ask for request in people.requests] == [Ask.APPROVAL]
    assert result.reason is Reason.APPROVAL_REJECTED
    assert app.acted_on("Submit transfer") == 0


class _Answers:
    """A control that answers every intervention with one fixed hand-off."""

    def __init__(self, handoff: Handoff) -> None:
        self.handoff = handoff
        self.requests: list = []

    def intervene(self, request):
        self.requests.append(request)
        return self.handoff


@pytest.mark.parametrize(
    ("handoff", "answer"),
    [
        (Handoff(HandoffOutcome.APPROVED), HandoffOutcome.APPROVED),
        (Handoff(HandoffOutcome.APPROVED, changed=True), HandoffOutcome.RESUMED),
        (Handoff(HandoffOutcome.RESUMED), HandoffOutcome.RESUMED),
        (Handoff(HandoffOutcome.RESUMED, changed=True), HandoffOutcome.RESUMED),
    ],
)
def test_only_an_approval_about_the_unchanged_screen_approves(
    profile, handoff, answer
) -> None:
    control = _Answers(handoff)
    operator = computeruse.replay_control.ControlledOperator(
        cast("Control", control), profile
    )
    request = computeruse.replay.HelpRequest(
        "rp-1-1",
        "rp-1",
        computeruse.replay.Ask.APPROVAL,
        Reason.APPROVAL_REQUIRED,
        "member_transfer@1",
        "submit",
        "/members/:id/transfer",
        600,
    )

    assert operator.request(request).answer.value == answer.value
    assert control.requests[0].trigger is Trigger.RISKY_ACTION


def _denying(tmp_path, effect: str):
    path = tmp_path / "denying.yaml"
    path.write_text(
        PROFILE.replace(
            "      submit_transfer: risky\n",
            f"      submit_transfer: risky\n      {effect}: deny\n",
        )
    )
    return load_profile(path)


def test_a_step_with_an_effect_the_profile_denies_asks_nobody(tmp_path) -> None:
    app = bank()
    people = People([])

    result, _ = run(
        transfer_capability(), app, people, _denying(tmp_path, "search_member")
    )

    assert result.reason is Reason.PROFILE_CONFLICT
    assert "effect_denied" in {issue.code for issue in result.issues}
    assert people.requests == []
    assert app.acted == []


def test_a_persons_step_with_an_effect_the_profile_denies_asks_nobody(
    tmp_path,
) -> None:
    capability = login_capability()
    by_hand = tuple(
        dataclasses.replace(node, performs=ActionKind.CLICK, effect="post_entry")
        if node.node_id == "code"
        else node
        for node in capability.nodes
    )
    app = desk()
    people = People([leave(app, "b")])

    result, _ = run(
        dataclasses.replace(capability, nodes=by_hand),
        app,
        people,
        _denying(tmp_path, "post_entry"),
        inputs={},
    )

    assert result.reason is Reason.PROFILE_CONFLICT
    assert "effect_denied" in {issue.code for issue in result.issues}
    assert people.requests == []
    assert app.acted == []


def test_a_saved_restriction_needs_a_person_even_when_the_profile_does_not(
    profile,
) -> None:
    capability = transfer_capability()
    safe = tuple(
        dataclasses.replace(node, approval=Approval.NONE, mandatory=False)
        if node.node_id == "open_transfer"
        else node
        for node in capability.nodes
    )
    saved = SavedRestriction(
        RestrictionScope.TARGET,
        "/members/:id",
        (ActionKind.CLICK,),
        "transfer_link",
        "",
        Limit.RISKY,
        Source.FINDING,
    )
    ignored = dataclasses.replace(capability, nodes=safe, restrictions=(saved,))

    result, _ = run(ignored, bank(), People([]), profile)

    assert result.reason is Reason.INVALID_ARTIFACT
    assert "bypasses_restriction" in {issue.code for issue in result.issues}


# Record identity.


def test_a_wrong_member_on_a_similar_page_is_not_acted_on(profile) -> None:
    app = bank(shown="10002")
    people = People(
        [HandoffOutcome.RESUMED, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED]
    )

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.DELIVERY_UNCERTAIN
    assert result.node == "search"
    assert app.acted_on("Search") == 1
    assert app.acted_on("Transfer") == 0
    assert "balance" not in result.outputs


def test_approval_cannot_stand_in_for_record_evidence(profile) -> None:
    app = bank(swapped="10002")
    people = People(
        [HandoffOutcome.APPROVED, HandoffOutcome.APPROVED, HandoffOutcome.APPROVED]
    )

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.RECORD_MISMATCH
    assert all(request.ask is Ask.PERSON for request in people.requests)
    assert app.acted_on("Submit transfer") == 0


def _named(app: App) -> App:
    """Show each member number beside the member's name, as many screens do."""
    for key in ("transfer", "confirmed"):
        page = app.screens[key]
        nodes = tuple(
            dataclasses.replace(node, role="definition", name=f"{node.name} · J. Doe")
            if node.slot == "Member number"
            else node
            for node in page.nodes
        )
        app.screens[key] = Page(page.path, nodes)
    return app


def _by_words() -> computeruse.capability.Capability:
    """The transfer, finding the member cell by the member number it holds.

    Nothing but the reference is saved: the name beside the number is read
    from the live page on each replay.
    """
    capability = transfer_capability()
    held = dataclasses.replace(
        capability.target("transfer_member"),
        form=LocatorForm.ACCESSIBILITY,
        role="definition",
        name=MEMBER,
        tag="",
        attribute=None,
        value=None,
        match=TextMatch.CONTAINS,
    )
    targets = tuple(
        held if target.target_id == "transfer_member" else target
        for target in capability.targets
    )
    link = RecordSpec(
        "transfer_member", MEMBER, Relation.CONTAINER, match=TextMatch.CONTAINS
    )
    submit = capability.node("submit")
    done = capability.node("done")
    assert isinstance(submit, ActionNode)
    assert isinstance(done, ResultNode)
    changed = {
        "submit": dataclasses.replace(submit, record=link),
        "done": dataclasses.replace(
            done,
            checks=(
                Present("submitted"),
                Shows("transfer_member", MEMBER, TextMatch.CONTAINS),
                Bound(BALANCE),
            ),
        ),
    }
    nodes = tuple(changed.get(node.node_id, node) for node in capability.nodes)
    words = dataclasses.replace(capability, targets=targets, nodes=nodes)
    assert validate(words) == ()
    return words


def test_a_member_named_by_words_is_found_and_its_name_is_never_saved(profile):
    app = _named(bank())
    people = People([approve])

    result, _ = run(_by_words(), app, people, profile)

    assert result.status is Status.SUCCEEDED, result
    submit = next(a for a in app.acted if a.effect == "submit_transfer")
    assert submit.evidence is not None
    assert submit.evidence.value == "10001"
    assert submit.evidence.suffix == " · J. Doe"
    assert "J. Doe" not in dumps(_by_words())


def test_a_different_member_beside_the_same_name_is_not_acted_on(profile):
    app = _named(bank(swapped="10002"))
    people = People(
        [HandoffOutcome.APPROVED, HandoffOutcome.APPROVED, HandoffOutcome.APPROVED]
    )

    result, _ = run(_by_words(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert app.acted_on("Submit transfer") == 0


def _record_bound_search(shown: str) -> tuple[App, computeruse.capability.Capability]:
    """The search result is proved by a balance tied to the member on the page."""
    app = bank(shown=shown)
    member = app.screens["member"]
    app.screens["member"] = Page(
        member.path,
        tuple(dataclasses.replace(node, ancestors=("el-9",)) for node in member.nodes),
    )
    capability = transfer_capability()
    tied = Shows(
        "balance",
        constant("1204.50"),
        TextMatch.EQUALS,
        RecordSpec("member_shown", MEMBER, Relation.CONTAINER),
    )
    search = capability.node("search")
    assert isinstance(search, ActionNode)
    searched = dataclasses.replace(
        search, transitions=(go("read_balance", tied), *search.transitions[1:])
    )
    nodes = tuple(
        searched if node.node_id == "search" else node for node in capability.nodes
    )
    return app, dataclasses.replace(capability, nodes=nodes)


def test_a_transition_check_holds_only_for_the_record_it_names(profile) -> None:
    app, capability = _record_bound_search("10002")
    people = People(
        [HandoffOutcome.RESUMED, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED]
    )

    result, _ = run(capability, app, people, profile)

    # The balance matches, but it belongs to another member.
    assert result.status is Status.NEEDS_HELP
    assert result.node == "search"
    assert app.acted_on("Transfer") == 0


def test_a_record_bound_transition_passes_for_the_named_record(profile) -> None:
    app, capability = _record_bound_search("10001")
    people = People([HandoffOutcome.APPROVED, HandoffOutcome.APPROVED])

    run(capability, app, people, profile)

    assert app.acted_on("Transfer") == 1


def test_a_record_past_the_first_page_is_found_by_complete_coverage(profile) -> None:
    app = bank()
    app.page_size = 2
    member = app.screens["member"]
    filler = tuple(status(f"Notice {index}") for index in range(3))
    app.screens["member"] = Page(member.path, (*filler, *member.nodes))
    people = People([approve])

    result, _ = run(transfer_capability(), app, people, profile)

    # The member number is on the third page of the member screen.
    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Submit transfer") == 1


@pytest.mark.rule(14, 16)
def test_a_person_taking_the_session_between_pages_pauses_the_replay(
    profile, monkeypatch
) -> None:
    app = bank()
    app.page_size = 2
    control, seat, _ = _controlled(
        app,
        [
            when(State.HUMAN_CONTROL, Command.RESUME),
            when(State.AWAITING_APPROVAL, Command.APPROVE),
        ],
    )
    seat.wait_s = 0.01
    taken = []
    looked = app.observe

    def observe(request):
        if request.start and not taken:
            taken.append(request.start)
            seat.send(Command.TAKE_CONTROL, control.status())
            raise ControlChanged
        return looked(request)

    monkeypatch.setattr(app, "observe", observe)
    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        log=MemoryReplayLog(),
        clock=seat.clock,
        journal=MemoryJournal(),
    )

    assert taken
    assert any(segment.taken for segment in control.segments)
    assert result.status is Status.SUCCEEDED


# Delivery.


def test_uncertain_delivery_is_checked_not_repeated(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.UNCERTAIN, True)]

    result, log = run(transfer_capability(), app, People([approve]), profile)

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Submit transfer") == 1
    dispatched = [e for e in log.events if isinstance(e, computeruse.replay.Dispatched)]
    assert dispatched[-1].outcome is Outcome.UNCERTAIN
    assert dispatched[-1].delivery is Delivery.COMPLETED


@pytest.mark.rule(5)
def test_uncertain_delivery_goes_to_a_person_and_is_never_resent(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.UNCERTAIN, False)]
    people = People([approve, HandoffOutcome.RESUMED])

    result, _ = run(transfer_capability(), app, people, profile)

    # Declining leaves the page unable to show it, so the replay stops rather
    # than asking again, and the change is never sent a second time.
    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.DELIVERY_UNCERTAIN
    assert app.acted_on("Submit transfer") == 1
    assert [r.reason for r in result.history][1:] == [Reason.DELIVERY_UNCERTAIN]
    assert result.history[1].ask is ReplayAsk.APPROVAL


@pytest.mark.rule(5)
def test_approving_an_uncertain_change_says_it_went_through(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.UNCERTAIN, False)]

    result, _ = run(transfer_capability(), app, People([approve, approve]), profile)

    # The person's approval settles the step: it counts as done by them, the
    # replay goes on past it, and it is never sent a second time. Later
    # checks still read the page, so a wrong "yes" cannot pass as success.
    assert result.reason is not Reason.DELIVERY_UNCERTAIN
    assert app.acted_on("Submit transfer") == 1
    assert any(
        attempt.delivery is Delivery.COMPLETED and attempt.actor is Actor.PERSON
        for attempt in result.attempts
    )


@pytest.mark.rule(16)
def test_a_person_who_finished_the_uncertain_submission_lets_the_replay_go_on(
    profile,
) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.UNCERTAIN, False)]

    def finish(_request):
        app.state = "confirmed"
        return HandoffOutcome.RESUMED

    result, _ = run(transfer_capability(), app, People([approve, finish]), profile)

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Submit transfer") == 1
    submit = [a for a in result.attempts if a.node == "submit"]
    assert [a.delivery for a in submit] == [Delivery.UNCERTAIN, Delivery.COMPLETED]
    assert submit[-1].actor is Actor.PERSON


def test_resume_does_not_repeat_what_a_person_already_completed(profile) -> None:
    app = bank()

    def perform(request):
        assert request.ask is Ask.APPROVAL
        app.state = "confirmed"
        return HandoffOutcome.RESUMED

    result, _ = run(transfer_capability(), app, People([perform]), profile)

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Submit transfer") == 0
    assert result.attempts[-1].actor is Actor.PERSON


def test_a_surface_failure_during_input_is_reported_as_uncertain(profile) -> None:
    app = bank()

    def die(app, action):
        if action.effect == "submit_transfer":
            app.fail_with = SurfaceError("driver died")

    app.on_act = die

    result, log = run(transfer_capability(), app, People([approve]), profile)

    assert result.status is Status.FAILED
    assert result.reason is Reason.SURFACE_FAILED
    assert result.attempts[-1].node == "submit"
    assert result.attempts[-1].delivery is Delivery.UNCERTAIN
    last = [e for e in log.events if isinstance(e, computeruse.replay.Dispatched)][-1]
    assert last.delivery is Delivery.UNCERTAIN


def test_a_stale_target_is_retried_within_its_bound(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.STALE, False), (Outcome.OK, True)]

    result, _ = run(transfer_capability(), app, People([approve]), profile)

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Member number") == 2
    first = [a for a in result.attempts if a.node == "enter_member"]
    assert [a.delivery for a in first] == [
        Delivery.NOT_DISPATCHED,
        Delivery.COMPLETED,
    ]


# Transitions.


def test_two_matching_transitions_are_reported_not_guessed(profile) -> None:
    capability = transfer_capability()
    # Both branches read the member screen, so both can hold at once there.
    targets = tuple(
        dataclasses.replace(target, route="/members/:id")
        if target.target_id == "no_member"
        else target
        for target in capability.targets
    )
    app = bank()
    app.screens["member"].nodes += (status("No member found"),)

    result, _ = run(
        dataclasses.replace(capability, targets=targets), app, People([]), profile
    )

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.AMBIGUOUS_STATE
    assert app.acted_on("Transfer") == 0


def test_an_ambiguous_target_is_never_the_first_match(profile) -> None:
    app = bank()
    app.screens["search"].nodes += (button("Search"),)
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.reason is Reason.TARGET_AMBIGUOUS
    assert app.acted_on("Search") == 0


def test_bounded_recovery_stops_at_its_limit(profile) -> None:
    capability = transfer_capability()
    again = go(
        "enter_member", Present("no_member"), origin=EdgeOrigin.AUTHORED, limit=2
    )
    nodes = tuple(
        dataclasses.replace(
            node,
            transitions=(
                go(
                    "read_balance",
                    computeruse.capability.Shows(
                        "member_shown", MEMBER, computeruse.capability.Match.EQUALS
                    ),
                ),
                again,
            ),
        )
        if node.node_id == "search"
        else node
        for node in capability.nodes
    )
    retrying = dataclasses.replace(
        capability, nodes=tuple(n for n in nodes if n.node_id != "not_found")
    )
    app = bank(rules={("search", ActionKind.CLICK, "Search"): "not_found"})
    app.screens["not_found"] = Page(
        "/members",
        (
            AxNode("textbox", "Member number", tag="input"),
            button("Search"),
            status("No member found"),
        ),
    )
    app.rules[("not_found", ActionKind.CLICK, "Search")] = "not_found"

    result, _ = run(retrying, app, People([]), profile)

    assert result.status is Status.RECOVERY_EXHAUSTED
    assert app.acted_on("Search") == 3


# People and the state they leave behind.


def login_capability() -> computeruse.capability.Capability:
    """Sign in, a person enters a one-time code, then B or a supported C."""
    base = transfer_capability()
    desk = "/desk"
    targets = (
        StructuralTarget(
            "sign_in",
            desk,
            LocatorForm.ACCESSIBILITY,
            "button",
            constant("Sign in"),
            "",
            None,
            None,
            (),
            None,
        ),
        StructuralTarget(
            "code",
            desk,
            LocatorForm.ACCESSIBILITY,
            "textbox",
            constant("One-time code"),
            "",
            None,
            None,
            (),
            None,
        ),
        StructuralTarget(
            "queue",
            desk,
            LocatorForm.ACCESSIBILITY,
            "heading",
            constant("Work queue"),
            "",
            None,
            None,
            (),
            None,
        ),
        StructuralTarget(
            "expired",
            desk,
            LocatorForm.ACCESSIBILITY,
            "status",
            constant("Password expired"),
            "",
            None,
            None,
            (),
            None,
        ),
    )
    sign_in = ActionNode(
        "sign_in",
        ActionKind.CLICK,
        desk,
        "sign_in",
        None,
        None,
        "sign_in",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (Present("code"),),
        (go("code"),),
    )
    code = HumanNode(
        "code",
        desk,
        HelpReason.AUTHENTICATION,
        None,
        None,
        True,
        (
            go("done", Present("queue")),
            go("expired", Present("expired"), origin=EdgeOrigin.AUTHORED),
        ),
    )
    return dataclasses.replace(
        base,
        capability_id="desk_sign_in",
        application=Application(
            base.application.profile_id,
            base.application.surface,
            base.application.origin,
            desk,
            (Present("sign_in"),),
        ),
        inputs=(),
        outputs=(),
        secrets=(),
        outcomes=("password_expired",),
        targets=targets,
        entry="sign_in",
        nodes=(
            sign_in,
            code,
            ResultNode("done", ResultKind.SUCCESS, "", (Present("queue"),)),
            ResultNode(
                "expired", ResultKind.OUTCOME, "password_expired", (Present("expired"),)
            ),
        ),
    )


def desk() -> App:
    pages = {
        "start": Page("/desk", (button("Sign in"),)),
        "code": Page("/desk", (AxNode("textbox", "One-time code", tag="input"),)),
        "b": Page("/desk", (AxNode("heading", "Work queue", tag="h1"),)),
        "c": Page("/desk", (status("Password expired"),)),
        "unknown": Page("/desk", (status("Maintenance window"),)),
        "both": Page(
            "/desk",
            (AxNode("heading", "Work queue", tag="h1"), status("Password expired")),
        ),
    }
    return App(pages, "start", {("start", ActionKind.CLICK, "Sign in"): "code"})


def leave(app: App, state: str, answer: HandoffOutcome = HandoffOutcome.RESUMED):
    def act(request):
        assert request.ask is Ask.PERSON
        app.state = state
        return answer

    return act


def test_a_person_returning_to_b_continues_through_b(profile) -> None:
    app = desk()

    people = People([leave(app, "b")])
    result, _ = run(login_capability(), app, people, profile, inputs={})

    assert result.status is Status.SUCCEEDED
    assert result.history[0].reason is Reason.PLANNED_STEP
    assert result.history[0].evidence == people.requests[0].intervention


def test_a_persons_step_needs_no_support_from_the_surface(profile) -> None:
    # The surface only clicks. The person types the code that replay cannot.
    class ClicksOnly(App):
        def capabilities(self):
            return {ActionKind.CLICK: frozenset({TargetForm.ACCESSIBILITY})}

    base = desk()
    app = ClicksOnly(base.screens, base.state, base.rules)
    capability = login_capability()
    typed = tuple(
        dataclasses.replace(node, performs=ActionKind.TYPE)
        if node.node_id == "code"
        else node
        for node in capability.nodes
    )

    result, _ = run(
        dataclasses.replace(capability, nodes=typed),
        app,
        People([leave(app, "b")]),
        profile,
        inputs={},
    )

    assert result.status is Status.SUCCEEDED


def test_a_person_returning_to_a_supported_c_follows_that_branch(profile) -> None:
    app = desk()

    result, _ = run(
        login_capability(), app, People([leave(app, "c")]), profile, inputs={}
    )

    assert result.status is Status.OUTCOME
    assert result.outcome == "password_expired"


def test_an_unknown_state_keeps_the_replay_paused(profile) -> None:
    app = desk()
    people = People(
        [leave(app, "unknown"), HandoffOutcome.RESUMED, HandoffOutcome.RESUMED]
    )

    result, _ = run(login_capability(), app, people, profile, inputs={})

    assert result.status is Status.NEEDS_HELP
    assert [r.reason for r in result.history] == [
        Reason.PLANNED_STEP,
        Reason.NO_TRANSITION,
        Reason.NO_TRANSITION,
    ]
    assert app.acted_on("Sign in") == 1


def test_an_unknown_state_resolved_later_continues(profile) -> None:
    app = desk()
    people = People([leave(app, "unknown"), leave(app, "b")])

    result, _ = run(login_capability(), app, people, profile, inputs={})

    assert result.status is Status.SUCCEEDED


def test_several_matching_continuations_are_ambiguous(profile) -> None:
    app = desk()

    result, _ = run(
        login_capability(), app, People([leave(app, "both")]), profile, inputs={}
    )

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.AMBIGUOUS_STATE


def test_a_late_answer_is_a_timeout(profile) -> None:
    app = desk()
    clock = Clock()

    def slow(request):
        clock.advance(request.timeout_s + 1)
        app.state = "b"
        return HandoffOutcome.RESUMED

    result, _ = run(
        login_capability(), app, People([slow]), profile, inputs={}, clock=clock
    )

    assert result.status is Status.HELP_TIMED_OUT


def test_an_answer_to_another_request_is_refused(profile) -> None:
    app = desk()
    receipts = []

    def wrong_request(seat, status):
        if status.state is not State.PAUSED:
            return False
        wrong = dataclasses.replace(
            status,
            offer=dataclasses.replace(status.offer, intervention="iv-other"),
        )
        receipts.append(seat.send(Command.RESUME, wrong))
        seat.send(Command.TERMINATE, status)
        return True

    control, seat, _ = _controlled(app, [wrong_request])
    result = replay(
        login_capability(),
        {},
        profile=profile,
        surface=app,
        control=control,
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert receipts[0].verdict is Verdict.STALE
    assert result.status is Status.TERMINATED
    assert app.acted_on("Open B") == app.acted_on("Open C") == 0


def test_a_step_a_person_already_did_later_is_never_sent(profile) -> None:
    # Opening the transfer is uncertain, and the person finishes the whole
    # transfer. The amount, the PIN, and the submission are all done.
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 3 + [(Outcome.UNCERTAIN, False)]

    def finish(request):
        assert request.trigger is Trigger.DELIVERY_UNCERTAIN
        app.state = "confirmed"
        return HandoffOutcome.RESUMED

    people = People([finish, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED])

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert app.acted_on("Amount") == app.acted_on("Teller PIN") == 0
    assert app.acted_on("Submit transfer") == 0


class _Watching:
    """An operator channel that keeps every status the run showed it."""

    def __init__(self) -> None:
        self.shown: list = []

    def attach(self, control) -> None:
        del control

    def show(self, status) -> None:
        self.shown.append(status)

    def listening(self) -> bool:
        return True


def _controlled(app: App, moves: list) -> tuple[Control, ScriptedSeat, _Watching]:
    clock = FakeClock()
    seat = ScriptedSeat(clock, moves=moves, wait_s=1.0, location=app.location())
    watching = _Watching()
    control = Control(
        mode=Mode.REPLAY, clock=clock, seat=seat, channels=[watching], worker=False
    )
    seat.control = control
    return control, seat, watching


def _uncertain_submission() -> App:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.UNCERTAIN, False)]
    return app


def test_what_a_person_does_in_a_replay_is_kept_under_its_intervention(
    profile,
) -> None:
    app = _uncertain_submission()

    def finish(seat) -> None:
        seat.person(ManualKind.CLICK, role="button", name="Submit transfer")
        app.state = "confirmed"

    control, seat, watching = _controlled(
        app,
        [
            when(State.AWAITING_APPROVAL, Command.APPROVE),
            when(State.AWAITING_APPROVAL, Command.TAKE_CONTROL),
            doing(State.HUMAN_CONTROL, finish),
            when(State.HUMAN_CONTROL, Command.RESUME),
        ],
    )
    journal = MemoryJournal()

    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        journal=journal,
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Submit transfer") == 1
    (help_pause,) = [
        event
        for event in journal.events
        if isinstance(event, Paused) and event.trigger is Trigger.DELIVERY_UNCERTAIN
    ]
    clicks = [event for event in journal.events if isinstance(event, ManualAction)]
    assert [event.intervention for event in clicks] == [help_pause.intervention]
    assert all(segment.mode is Mode.REPLAY for segment in control.segments)
    assert help_pause.intervention in {s.intervention for s in control.segments}
    assert all(status.mode is Mode.REPLAY for status in watching.shown)
    offered = {s.offer.context for s in watching.shown if s.offer is not None}
    assert offered == {"member_transfer@1"}


def test_a_replay_nobody_answers_ends_when_the_hand_off_times_out(tmp_path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE.replace("handoff_timeout_s: 600", "handoff_timeout_s: 30"))
    app = _uncertain_submission()
    control, seat, _ = _controlled(
        app, [when(State.AWAITING_APPROVAL, Command.APPROVE)]
    )

    result = replay(
        transfer_capability(),
        INPUTS,
        profile=load_profile(path),
        surface=app,
        control=control,
        journal=MemoryJournal(),
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert result.status is Status.HELP_TIMED_OUT
    assert app.acted_on("Submit transfer") == 1


def test_termination_stops_the_replay(profile) -> None:
    result, _ = run(
        login_capability(),
        desk(),
        People([HandoffOutcome.TERMINATED]),
        profile,
        inputs={},
    )

    assert result.status is Status.TERMINATED
    assert result.reason is Reason.OPERATOR_TERMINATED


def test_waiting_for_a_person_is_not_spent_from_the_budget(profile) -> None:
    app = desk()
    clock = Clock()

    def slow(_request):
        clock.advance(500)
        app.state = "b"
        return HandoffOutcome.RESUMED

    result, _ = run(
        login_capability(), app, People([slow]), profile, inputs={}, clock=clock
    )

    assert result.status is Status.SUCCEEDED
    assert result.paused_s == pytest.approx(500)
    assert result.active_s < 120


@pytest.mark.rule(12)
def test_replay_interventions_leave_the_capability_unchanged(profile) -> None:
    capability = login_capability()
    before = dumps(capability)
    app = desk()

    run(
        capability,
        app,
        People([leave(app, "unknown"), leave(app, "c")]),
        profile,
        inputs={},
    )

    assert dumps(capability) == before
    code = capability.node("code")
    assert isinstance(code, HumanNode)
    assert [edge.origin for edge in code.transitions] == [
        EdgeOrigin.OBSERVED,
        EdgeOrigin.AUTHORED,
    ]


# Visual targets.


def canvas_capability(image: bytes) -> computeruse.capability.Capability:
    base = login_capability()
    stored = template("pay_button", image, StoragePermit.SYNTHETIC)
    press = ActionNode(
        "press",
        ActionKind.CLICK,
        "/desk",
        "pay",
        None,
        None,
        "press",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (Present("queue"),),
        (go("done"),),
    )
    return dataclasses.replace(
        base,
        capability_id="canvas_press",
        templates=(stored,),
        targets=(
            *base.targets,
            VisualTarget("pay", "/desk", "pay_button", ()),
        ),
        entry="press",
        application=dataclasses.replace(base.application, markers=()),
        nodes=(
            press,
            ResultNode("done", ResultKind.SUCCESS, "", (Present("queue"),)),
        ),
        outcomes=(),
    )


def test_a_visual_target_is_found_where_it_is_now(profile) -> None:
    recorded = png((200, 120), ((30, 40),))
    button_crop = crop(recorded, 30, 40, 40, 20)
    capability = canvas_capability(button_crop)
    moved = png((200, 120), ((120, 70),))
    app = desk()
    app.rules[("start", ActionKind.CLICK, "screen")] = "b"

    result, _ = run(
        capability, app, People([]), profile, inputs={}, scenes=Captures([moved])
    )

    assert result.status is Status.SUCCEEDED
    target = app.acted[0].target
    assert isinstance(target, ScreenTarget)
    assert target.point is not None
    assert (target.point.x, target.point.y) == (140.0, 80.0)
    document = json.loads(dumps(capability))
    assert "pay_button" in {item["template_id"] for item in document["templates"]}
    assert not {"x", "y", "left", "top", "point", "box"} & set(document["templates"][0])


def test_two_copies_of_a_visual_target_are_ambiguous(profile) -> None:
    recorded = png((200, 120), ((30, 40),))
    capability = canvas_capability(crop(recorded, 30, 40, 40, 20))
    twins = png((200, 120), ((30, 40), (120, 70)))
    app = desk()

    result, _ = run(
        capability,
        app,
        People([HandoffOutcome.RESUMED] * 3),
        profile,
        inputs={},
        scenes=Captures([twins]),
    )

    assert result.reason is Reason.TARGET_AMBIGUOUS
    assert app.acted == []


def test_a_visual_target_without_a_capture_goes_to_a_person(profile) -> None:
    recorded = png((200, 120), ((30, 40),))
    capability = canvas_capability(crop(recorded, 30, 40, 40, 20))
    app = desk()

    result, _ = run(
        capability, app, People([HandoffOutcome.RESUMED] * 3), profile, inputs={}
    )

    assert result.reason is Reason.TARGET_UNAVAILABLE
    assert app.acted == []


# What replay never does.


REPLAY_MODULES = (
    computeruse.capability,
    computeruse.recorder,
    computeruse.replay,
    computeruse.replay_control,
    computeruse.retarget,
)


@pytest.mark.parametrize("module", REPLAY_MODULES, ids=lambda m: m.__name__)
def test_replay_modules_import_no_model_and_no_driver(module) -> None:
    assert_module_import_contract(module)


def test_loading_the_replay_modules_loads_no_model_at_all() -> None:
    # A fresh interpreter, so modules this test session loaded do not count.
    names = ", ".join(module.__name__ for module in REPLAY_MODULES)
    forbidden = {"computeruse.decider", "computeruse.model", "computeruse.loop"}
    probe = f"import sys, {names}; print(sorted(set(sys.modules) & {forbidden!r}))"
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"


def test_replay_requires_explicit_session_control() -> None:
    signature = inspect.signature(replay)
    assert signature.parameters["control"].default is inspect.Parameter.empty


def test_a_whole_replay_makes_no_network_call(profile, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("replay tried to reach the network")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    result, _ = run(transfer_capability(), bank(), People([approve]), profile)

    assert result.status is Status.SUCCEEDED


def test_logs_carry_no_values_and_secrets_stay_named(profile) -> None:
    app = bank()

    result, log = run(transfer_capability(), app, People([approve]), profile)

    written = json.dumps(
        [{"event": type(e).__name__, **dataclasses.asdict(e)} for e in log.events]
    )
    for value in ("10001", "250.00", "1204.50", "Submit transfer", "Teller PIN"):
        assert value not in written
    typed = [a.value for a in app.acted if a.kind is ActionKind.TYPE]
    assert SecretRef("teller_pin") in typed
    assert result.outputs["balance"] == "1204.50"
    assert "10001" not in dumps(transfer_capability())


def test_help_requests_carry_the_capability_and_node(profile) -> None:
    people = People([approve])

    run(transfer_capability(), bank(), people, profile)

    request = people.requests[0]
    assert request.capability == "member_transfer@1"
    assert request.reason.splitlines()[1].startswith("Step submit: ")
    assert request.route == "/members/:id/transfer"
    assert request.timeout_s == 600
    assert request.action is not None
    assert request.action.effect == "submit_transfer"


def test_a_check_node_branches_on_declared_conditions(profile) -> None:
    capability = login_capability()
    check = CheckNode(
        "which", (go("done", Present("queue")), go("expired", Present("expired")))
    )
    sign_in = dataclasses.replace(
        capability.node("sign_in"),
        verify=(Present("expired"),),
        transitions=(go("which"),),
    )
    branching = dataclasses.replace(
        capability,
        nodes=(sign_in, check, capability.node("done"), capability.node("expired")),
    )
    app = desk()
    app.rules[("start", ActionKind.CLICK, "Sign in")] = "c"

    result, log = run(branching, app, People([]), profile, inputs={})

    assert result.status is Status.OUTCOME
    moves = [e for e in log.events if isinstance(e, computeruse.replay.Transitioned)]
    assert [(e.node, e.to) for e in moves] == [
        ("sign_in", "which"),
        ("which", "expired"),
    ]


def test_failure_evidence_names_the_check_that_did_not_hold(profile) -> None:
    from computeruse.diagnostics import FailureEvidence

    app = bank(shown="10002")
    people = People(
        [HandoffOutcome.RESUMED, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED]
    )

    result, log = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    (evidence,) = [item for item in log.events if isinstance(item, FailureEvidence)]
    failed = [shape for shape in evidence.checks if shape.held is False]
    assert failed
    assert failed[0].tag in {"shows", "present"}
    assert failed[0].form in {"dom", "accessibility"}
    # Nothing but structure is kept: no member number anywhere in the shapes.
    assert "10001" not in repr(evidence.checks)
    assert "10002" not in repr(evidence.checks)


@pytest.mark.parametrize("landed", [True, False])
def test_a_submission_whose_reply_was_lost_is_uncertain_not_failed(
    profile, landed: bool
) -> None:
    """The submit left, and no permitted page came back, as when a reply is lost.

    The surface reports that as blocked and sent elsewhere. The replay checks
    the page, finishes if the submission shows, and otherwise asks a person.
    It never sends the submission again.
    """
    from computeruse.control import SENT_ELSEWHERE

    class LostReply(App):
        def act(self, action, *, expect=None):
            result = super().act(action, expect=expect)
            if action.effect == "submit_transfer":
                return dataclasses.replace(
                    result, outcome=Outcome.BLOCKED, detail=SENT_ELSEWHERE
                )
            return result

    held = bank()
    app = LostReply(
        **{field.name: getattr(held, field.name) for field in dataclasses.fields(held)}
    )
    app.outcomes = [(Outcome.OK, True)] * 6 + [(Outcome.OK, landed)]
    people = People([approve, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED])

    result, _ = run(transfer_capability(), app, people, profile)

    assert app.acted_on("Submit transfer") == 1
    if landed:
        assert result.status is Status.SUCCEEDED
    else:
        assert result.status is Status.NEEDS_HELP
        assert result.reason is Reason.DELIVERY_UNCERTAIN


# Text targets, for an application painted on a canvas.


class Lines:
    """A reader that returns set lines for each fake capture."""

    def __init__(self, by_image: dict[bytes, tuple]) -> None:
        self.by_image = by_image

    def lines(self, image: bytes) -> tuple:
        return self.by_image.get(image, ())


class Painted(App):
    """A canvas whose read at any point returns one painted line."""

    painted: str = ""

    def act(self, action, *, expect=None):
        from computeruse.actions import ActionResult, PageState

        if action.kind is ActionKind.READ and isinstance(action.target, ScreenTarget):
            self.acted.append(action)
            return ActionResult(Outcome.OK, PageState(self.location()), self.painted)
        return super().act(action, expect=expect)


def _painted_capability(target, kind=ActionKind.CLICK, into=None, outputs=()):
    from computeruse.capability import Bound

    base = login_capability()
    act = ActionNode(
        "press",
        kind,
        "/desk",
        target.target_id,
        None,
        None,
        "press" if kind is ActionKind.CLICK else None,
        None,
        None,
        into,
        Approval.NONE,
        False,
        (),
        (Present("queue"),) if kind is ActionKind.CLICK else (),
        (go("done"),),
    )
    checks = (Bound(into),) if into is not None else (Present("queue"),)
    return dataclasses.replace(
        base,
        capability_id="painted",
        targets=(*base.targets, target),
        entry="press",
        outputs=outputs,
        application=dataclasses.replace(base.application, markers=()),
        nodes=(act, ResultNode("done", ResultKind.SUCCESS, "", checks)),
        outcomes=(),
    )


def _replay_painted(capability, app, profile, captures, reader):
    clock = Clock()
    return replay(
        capability,
        {},
        profile=profile,
        surface=app,
        control=People([HandoffOutcome.RESUMED] * 3).control(clock),
        log=MemoryReplayLog(),
        clock=clock,
        scenes=captures,
        sleep=clock.advance,
        reader=reader,
    )


def test_a_text_target_is_clicked_where_its_line_is_now(profile) -> None:
    from computeruse.capability import TextTarget
    from computeruse.reading import Line

    target = TextTarget("pay", "/desk", (), constant("Pay now"))
    capability = _painted_capability(target)
    assert validate(capability) == ()
    app = desk()
    app.rules[("start", ActionKind.CLICK, "screen")] = "b"
    reader = Lines(
        {
            b"canvas": (
                Line("Pay now", 100, 50, 60, 20, 0.99),
                Line("Cancel", 100, 90, 50, 20, 0.99),
            )
        }
    )

    result = _replay_painted(capability, app, profile, Captures([b"canvas"]), reader)

    assert result.status is Status.SUCCEEDED
    clicked = app.acted[0].target
    assert isinstance(clicked, ScreenTarget)
    assert clicked.point is not None
    assert (clicked.point.x, clicked.point.y) == (130.0, 60.0)


@pytest.mark.rule(10)
@pytest.mark.parametrize("agree", [True, False])
def test_a_painted_value_is_taken_only_when_two_captures_agree(
    profile, agree: bool
) -> None:
    from computeruse.capability import Field, RefKind, TextTarget, ValueType, ref
    from computeruse.reading import Line

    status = ref(RefKind.OUTPUT, "status")
    target = TextTarget("status", "/desk", (), constant("Status:"), label=True)
    capability = _painted_capability(
        target,
        ActionKind.READ,
        into=status,
        outputs=(Field("status", ValueType.TEXT, True, 1, 40, ()),),
    )
    assert validate(capability) == ()
    held = desk()
    app = Painted(
        **{field.name: getattr(held, field.name) for field in dataclasses.fields(held)}
    )
    app.painted = "Status: active"
    reader = Lines(
        {
            b"first": (Line("Status: active", 40, 220, 120, 20, 0.99),),
            b"second": (
                Line(
                    "Status: active" if agree else "Status: closed",
                    40,
                    220,
                    120,
                    20,
                    0.99,
                ),
            ),
        }
    )

    result = _replay_painted(
        capability, app, profile, Captures([b"first", b"second", b"second"]), reader
    )

    if agree:
        assert result.status is Status.SUCCEEDED
        assert dict(result.outputs) == {"status": "active"}
    else:
        assert result.status is Status.NEEDS_HELP
        assert result.reason is Reason.READING_UNCONFIRMED


def test_typing_to_the_focus_goes_to_a_fresh_captures_focus(profile) -> None:
    """A painted form moves the focus, and replay types at that position."""
    from computeruse.capability import FocusTarget

    target = FocusTarget("focus", "/desk", ())
    capability = _painted_capability(target, ActionKind.TYPE)
    typed = next(node for node in capability.nodes if node.node_id == "press")
    capability = dataclasses.replace(
        capability,
        inputs=(),
        nodes=tuple(
            dataclasses.replace(node, value=constant("OP0002"))
            if node.node_id == "press"
            else node
            for node in capability.nodes
        ),
    )
    assert typed.kind is ActionKind.TYPE
    assert validate(capability) == ()
    app = desk()

    _replay_painted(capability, app, profile, Captures([b"canvas"]), Lines({}))

    sent = app.acted[0]
    assert isinstance(sent.target, ScreenTarget)
    assert sent.target.point is None
    assert sent.value == "OP0002"


@pytest.mark.rule(4)
def test_a_click_that_chooses_a_painted_field_is_kept_and_sent_first(profile) -> None:
    """A focusing click has no check of its own. The following typing has one.

    The validator accepts the pair, and replay clicks the field before it
    types, so the keys go where discovery sent them (R4).
    """
    from computeruse.capability import FocusTarget, TextTarget
    from computeruse.reading import Line

    field = TextTarget("field", "/desk", (), constant("Nickname"), dx=120, dy=10)
    focus = FocusTarget("focus", "/desk", ())
    base = _painted_capability(focus, ActionKind.TYPE)
    typed = next(node for node in base.nodes if node.node_id == "press")
    click = ActionNode(
        "choose",
        ActionKind.CLICK,
        "/desk",
        "field",
        None,
        None,
        "choose_field",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (),
        (go("press"),),
    )
    capability = dataclasses.replace(
        base,
        inputs=(),
        targets=(*base.targets, field),
        entry="choose",
        nodes=(
            click,
            dataclasses.replace(typed, value=constant("Rainy day")),
            *(node for node in base.nodes if node.node_id != "press"),
        ),
    )
    assert validate(capability) == ()
    app = desk()
    reader = Lines({b"canvas": (Line("Nickname", 40, 30, 80, 20, 0.99),)})

    _replay_painted(capability, app, profile, Captures([b"canvas"]), reader)

    first, second = app.acted[0], app.acted[1]
    assert first.kind is ActionKind.CLICK
    assert isinstance(first.target, ScreenTarget)
    assert first.target.point is not None
    assert second.kind is ActionKind.TYPE
    assert isinstance(second.target, ScreenTarget)
    assert second.target.point is None


@pytest.mark.rule(4)
def test_a_click_with_no_check_that_does_not_lead_to_typing_is_refused() -> None:
    """Only a click the next typing proves may go without its own check."""
    from computeruse.capability import FocusTarget, IssueCode, TextTarget

    field = TextTarget("field", "/desk", (), constant("Nickname"), dx=120, dy=10)
    focus = FocusTarget("focus", "/desk", ())
    base = _painted_capability(focus, ActionKind.TYPE)
    click = ActionNode(
        "choose",
        ActionKind.CLICK,
        "/desk",
        "field",
        None,
        None,
        "choose_field",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (),
        (go("done"),),
    )
    capability = dataclasses.replace(
        base,
        inputs=(),
        targets=(*base.targets, field),
        entry="choose",
        nodes=(click, *(node for node in base.nodes if node.node_id != "press")),
    )
    codes = {issue.code for issue in validate(capability)}
    assert IssueCode.UNVERIFIED_ACTION in codes


def test_a_painted_result_is_checked_on_a_fresh_capture(profile) -> None:
    from computeruse.capability import Field, RefKind, TextTarget, ValueType, ref
    from computeruse.reading import Line

    status = ref(RefKind.OUTPUT, "status")
    target = TextTarget("status", "/desk", (), constant("Status:"), label=True)
    base = _painted_capability(
        target,
        ActionKind.READ,
        into=status,
        outputs=(Field("status", ValueType.TEXT, True, 1, 40, ()),),
    )
    capability = dataclasses.replace(
        base,
        nodes=tuple(
            ResultNode(
                "done",
                ResultKind.SUCCESS,
                "",
                (Shows("status", status, TextMatch.EQUALS),),
            )
            if node.node_id == "done"
            else node
            for node in base.nodes
        ),
    )
    assert validate(capability) == ()
    held = desk()
    app = Painted(
        **{field.name: getattr(held, field.name) for field in dataclasses.fields(held)}
    )
    app.painted = "Status: active"
    reader = Lines({b"canvas": (Line("Status: active", 40, 220, 120, 20, 0.99),)})

    result = _replay_painted(capability, app, profile, Captures([b"canvas"]), reader)

    assert result.status is Status.SUCCEEDED
    assert dict(result.outputs) == {"status": "active"}


class _ChangingReader:
    """Reads the painted status as approved once, then as rejected."""

    def __init__(self) -> None:
        self.calls = 0

    def lines(self, image):  # noqa: ARG002  the reader's protocol takes the image
        self.calls += 1
        text = "Status: Approved" if self.calls == 1 else "Status: Rejected"
        from computeruse.reading import Line

        return (Line(text, 0, 0, 100, 20, 1.0),)


@pytest.mark.rule(10)
def test_a_painted_finish_needs_two_agreeing_captures(profile) -> None:
    # Graduated from tests/test_known_gaps.py (the painted part of R1).
    from replay_fakes import App, Captures, Page, People, png

    from computeruse.capability import (
        AtRoute,
        Match,
        ResultKind,
        ResultNode,
        Shows,
        TextTarget,
        check_profile,
        validate,
    )
    from computeruse.capability import constant as saved_constant

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

    def execute(capability, app, profile, **options):
        assert validate(capability) == ()
        assert check_profile(capability, profile) == ()
        people = People([])
        clock = Clock()
        result = replay(
            capability,
            {},
            profile=profile,
            surface=app,
            control=people.control(clock),
            log=MemoryReplayLog(),
            clock=clock,
            **options,
        )
        return result, people

    target = TextTarget("state", "/desk", (), saved_constant("Status:"), label=True)
    capability = at_desk(
        [target],
        [
            ResultNode(
                "done",
                ResultKind.SUCCESS,
                "",
                (Shows("state", saved_constant("Approved"), Match.EQUALS),),
            ),
        ],
    )
    app = App({"desk": Page("/desk", ())}, "desk", {})
    reader = _ChangingReader()
    result, _ = execute(
        capability, app, profile, reader=reader, scenes=Captures([png((120, 40), ())])
    )
    assert result.status is not Status.SUCCEEDED
    assert reader.calls >= 2


@pytest.mark.rule(16)
def test_a_request_names_the_operation_the_record_and_the_question(profile) -> None:
    """R8: a person is told what to settle, for which record, not only a code."""
    people = People([HandoffOutcome.APPROVED])
    clock = Clock()
    result = replay(
        transfer_capability(),
        {"member_id": "10001", "amount": "250.00"},
        profile=profile,
        surface=bank(),
        control=people.control(clock),
        log=MemoryReplayLog(),
        clock=clock,
    )
    assert result.status is Status.SUCCEEDED, result
    (asked,) = people.requests
    question, step = asked.reason.splitlines()
    assert question == "Approve this one operation, once, for this record."
    assert step.startswith("Step submit: click")
    assert "member_id=10001" in step


@pytest.mark.rule(16)
def test_the_control_window_shows_the_question() -> None:
    """The replay's operator bridge puts the question in the request's reason."""
    from typing import cast

    from computeruse.escalation import InterventionRequest
    from computeruse.profile import Profile
    from computeruse.replay import HelpRequest, Reason
    from computeruse.replay_control import ControlledOperator

    shown: list[InterventionRequest] = []

    class Seen:
        def intervene(self, request):
            shown.append(request)
            from computeruse.escalation import Handoff, HandoffOutcome

            return Handoff(HandoffOutcome.TIMED_OUT, request.step)

    class Named:
        profile_id = "p"

    request = HelpRequest(
        "r-1",
        "r",
        computeruse.replay.Ask.APPROVAL,
        Reason.APPROVAL_REQUIRED,
        "transfer@1",
        "submit",
        "/pay",
        5.0,
        question="click submit_transfer for member_id=10001: approve this one "
        "operation, once, for this record",
    )
    control = cast("Control", Seen())
    named = cast("Profile", Named())
    ControlledOperator(control, named).request(request)
    question, step = shown[0].reason.splitlines()
    assert step.startswith("Step submit")
    assert "click submit_transfer" in question


@pytest.mark.rule(15)
@pytest.mark.parametrize("agree", [True, False])
def test_a_painted_not_found_answer_ends_the_replay_as_its_outcome(
    profile, agree: bool
) -> None:
    """A learned branch saved as the answer's words and the searched input.

    The answer names the member searched for, so the branch holds for any
    member, and it is read from two captures that must agree (rule 10).
    """
    from computeruse.capability import CheckNode, Purpose, RefKind, TextTarget, ref
    from computeruse.reading import Line

    answer = TextTarget(
        "answer", "/desk", (), constant("No member with number"), label=True
    )
    member = ref(RefKind.INPUT, "member_id")
    said = Shows("answer", member, TextMatch.CONTAINS, purpose=Purpose.STATE)
    base = transfer_capability()
    capability = dataclasses.replace(
        base,
        inputs=base.inputs[:1],
        secrets=(),
        targets=(answer,),
        application=dataclasses.replace(
            base.application, entry_route="/desk", markers=()
        ),
        entry="branch",
        nodes=(
            CheckNode("branch", (go("missing", said),)),
            ResultNode("missing", ResultKind.OUTCOME, "member_not_found", (said,)),
        ),
    )
    assert validate(capability) == ()
    line = "No member with number 10009."
    reader = Lines(
        {
            b"first": (Line(line, 40, 740, 300, 20, 0.99),),
            b"second": (
                Line(
                    line if agree else "No member with number 10008.",
                    40,
                    740,
                    300,
                    20,
                    0.99,
                ),
            ),
        }
    )

    clock = Clock()
    result = replay(
        capability,
        {"member_id": "10009"},
        profile=profile,
        surface=desk(),
        control=People([]).control(clock),
        log=MemoryReplayLog(),
        clock=clock,
        scenes=Captures([b"first", b"second"] * 4),
        sleep=clock.advance,
        reader=reader,
    )

    if agree:
        assert result.status is Status.OUTCOME
        assert result.outcome == "member_not_found"
    else:
        assert result.status is not Status.OUTCOME
