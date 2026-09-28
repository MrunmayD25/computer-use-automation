"""One section per replay or recording defect found in review of PR #6.

Each test reproduces a failure from the first replay implementation and fails
if its fix is removed. Every capability and event stream is a synthetic
fixture. ``test_replay_browser.py`` contains the native-dialog and
approval-state cases that need a browser.
"""

from __future__ import annotations

import dataclasses

import pytest
from fakes import ScriptedSeat, doing, when
from replay_fakes import (
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
    dd,
    go,
    limits,
    png,
    status,
    transfer_capability,
)
from test_replay import (
    _controlled,
    canvas_capability,
    desk,
    login_capability,
    run,
)

from computeruse.actions import (
    ActionResult,
    AxNode,
    Observation,
    ObservationMode,
    ObservationStatus,
    Outcome,
    PageState,
    TargetForm,
)
from computeruse.capability import (
    Absent,
    ActionNode,
    Approval,
    AtRoute,
    CheckNode,
    Destination,
    DialogOpen,
    HumanNode,
    Issue,
    IssueCode,
    LocatorForm,
    Match,
    Param,
    Present,
    RestrictionScope,
    ResultKind,
    ResultNode,
    SavedRestriction,
    Shows,
    StructuralTarget,
    check_profile,
    constant,
    text_matches,
    validate,
)
from computeruse.control import NOT_SENT, Control
from computeruse.escalation import Ask, Command, HandoffOutcome, State, Verdict
from computeruse.journal import MemoryJournal
from computeruse.manual import ManualKind
from computeruse.policy import Source
from computeruse.profile import ActionKind, Limit, load_profile
from computeruse.replay import Actor, Delivery, MemoryReplayLog, Reason, Status, replay
from computeruse.retarget import complete


def load(tmp_path, text: str = PROFILE):
    path = tmp_path / "profile.yaml"
    path.write_text(text)
    return load_profile(path)


@pytest.fixture
def profile(tmp_path):
    return load(tmp_path)


def approve(request):
    assert request.ask is Ask.APPROVAL
    return HandoffOutcome.APPROVED


def always(answer: HandoffOutcome):
    def reply(_request):
        return answer

    return reply


def replace_node(capability, node_id, **changes):
    return dataclasses.replace(
        capability,
        nodes=tuple(
            dataclasses.replace(node, **changes) if node.node_id == node_id else node
            for node in capability.nodes
        ),
    )


def ax(target_id, route, role, name) -> StructuralTarget:
    return StructuralTarget(
        target_id,
        route,
        LocatorForm.ACCESSIBILITY,
        role,
        constant(name),
        "",
        None,
        None,
        (),
        None,
    )


# 1. Uncertain delivery stays uncertain until the page shows otherwise.


def test_uncertain_typing_that_changed_nothing_does_not_reach_the_submission(
    profile,
) -> None:
    """Regression: an unverified type node counted an uncertain input as done."""
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 4 + [(Outcome.UNCERTAIN, False)]
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.DELIVERY_UNCERTAIN
    assert app.acted_on("Amount") == 1
    assert app.acted_on("Teller PIN") == 0
    assert app.acted_on("Submit transfer") == 0
    assert {r.reason for r in result.history} == {Reason.DELIVERY_UNCERTAIN}
    amount = [a for a in result.attempts if a.node == "enter_amount"]
    assert [a.delivery for a in amount] == [Delivery.UNCERTAIN]


def test_uncertain_typing_the_field_shows_is_verified_by_the_field(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 4 + [(Outcome.UNCERTAIN, True)]

    result, _ = run(transfer_capability(), app, People([approve]), profile)

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Amount") == 1
    amount = [a for a in result.attempts if a.node == "enter_amount"]
    assert [a.delivery for a in amount] == [Delivery.COMPLETED]


def test_uncertain_secret_typing_needs_evidence_after_handback(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 5 + [(Outcome.UNCERTAIN, True)]
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.DELIVERY_UNCERTAIN
    assert app.acted_on("Teller PIN") == 1
    assert app.acted_on("Submit transfer") == 0
    pin = [a for a in result.attempts if a.node == "enter_pin"]
    assert [a.delivery for a in pin] == [Delivery.UNCERTAIN]


def branch_capability():
    """Pick a branch in a select, then save. A synthetic fixture."""
    base = login_capability()
    pick = ActionNode(
        "pick",
        ActionKind.SELECT,
        "/desk",
        "branch",
        constant("North"),
        None,
        None,
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (),
        (go("save"),),
    )
    save = ActionNode(
        "save",
        ActionKind.CLICK,
        "/desk",
        "save",
        None,
        None,
        "save_branch",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (Present("saved"),),
        (go("done"),),
    )
    return dataclasses.replace(
        base,
        capability_id="pick_branch",
        application=dataclasses.replace(base.application, markers=(Present("branch"),)),
        outcomes=(),
        targets=(
            ax("branch", "/desk", "combobox", "Branch"),
            ax("save", "/desk", "button", "Save"),
            ax("saved", "/desk", "status", "Saved"),
        ),
        entry="pick",
        nodes=(
            pick,
            save,
            ResultNode("done", ResultKind.SUCCESS, "", (Present("saved"),)),
        ),
    )


def branch_app() -> App:
    branch = AxNode("combobox", "Branch", value="", tag="select")
    pages = {
        "form": Page("/desk", (branch, button("Save"))),
        "saved": Page("/desk", (branch, status("Saved"))),
    }
    return App(pages, "form", {("form", ActionKind.CLICK, "Save"): "saved"})


def test_uncertain_selection_that_changed_nothing_does_not_save(profile) -> None:
    app = branch_app()
    app.outcomes = [(Outcome.UNCERTAIN, False)]
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(branch_capability(), app, people, profile, inputs={})

    assert result.reason is Reason.DELIVERY_UNCERTAIN
    assert app.acted_on("Save") == 0
    assert app.acted_on("Branch") == 1


def test_an_uncertain_selection_the_field_shows_goes_on(profile) -> None:
    app = branch_app()
    app.outcomes = [(Outcome.UNCERTAIN, True)]

    result, _ = run(branch_capability(), app, People([]), profile, inputs={})

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Branch") == 1


# 2. Approval is bound to a fresh proposal and a unique run.


def ready_capability():
    """The transfer, with the submission requiring the account to show Ready."""
    base = transfer_capability()
    ready = ax("ready", "/members/:id/transfer", "status", "Account Ready")
    return replace_node(
        dataclasses.replace(base, targets=(*base.targets, ready)),
        "submit",
        requires=(Present("ready"),),
    )


def ready_bank() -> App:
    app = bank()
    app.screens["transfer"].nodes += (status("Account Ready"),)
    return app


def test_a_precondition_that_changes_during_approval_stops_the_dispatch(
    profile,
) -> None:
    """Regression: the replay dispatched on checks made before the approval."""
    app = ready_bank()

    def freeze(request):
        assert request.ask is Ask.APPROVAL
        nodes = app.screens["transfer"].nodes
        app.screens["transfer"].nodes = tuple(
            status("Account Frozen") if node.name == "Account Ready" else node
            for node in nodes
        )
        return HandoffOutcome.APPROVED

    people = People([freeze, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED])

    result, _ = run(ready_capability(), app, people, profile)

    assert app.acted_on("Submit transfer") == 0
    assert result.status is Status.NEEDS_HELP
    assert [r.reason for r in result.history][1:] == [Reason.PRECONDITION_UNMET] * 2
    # The request tells the person what the replay needs, not only why.
    lines = people.requests[1].reason.splitlines()
    assert lines[0] == "Help the replay: the page was not in the state this step needs."
    assert lines[1].startswith("Next step: click button 'Submit transfer'")
    assert lines[2].startswith("It expects: ")
    assert "'Account Ready'" in lines[2]
    assert lines[3].startswith("Do the step yourself, or bring the page to that state")
    assert lines[-1].startswith("Step ")


@pytest.mark.rule(5)
def test_an_operation_completed_during_approval_is_not_repeated(profile) -> None:
    app = bank()

    def someone_else_submits(request):
        assert request.ask is Ask.APPROVAL
        app.state = "confirmed"
        return HandoffOutcome.APPROVED

    result, _ = run(transfer_capability(), app, People([someone_else_submits]), profile)

    assert result.status is Status.SUCCEEDED
    assert app.acted_on("Submit transfer") == 0
    submit = [a for a in result.attempts if a.node == "submit"]
    assert [(a.delivery, a.actor) for a in submit] == [
        (Delivery.COMPLETED, Actor.PERSON)
    ]


def test_a_redrawn_control_invalidates_the_approval_and_asks_again(profile) -> None:
    app = bank()
    app.screens["transfer"].nodes = tuple(
        dataclasses.replace(node, control="c9")
        if node.name == "Submit transfer"
        else node
        for node in app.screens["transfer"].nodes
    )

    def redraw(request):
        app.screens["transfer"].nodes = tuple(
            dataclasses.replace(node, control="c12")
            if node.name == "Submit transfer"
            else node
            for node in app.screens["transfer"].nodes
        )
        return approve(request)

    people = People([redraw, approve])

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.SUCCEEDED
    assert [r.reason for r in result.history] == [
        Reason.APPROVAL_REQUIRED,
        Reason.APPROVAL_INVALIDATED,
    ]
    assert app.acted_on("Submit transfer") == 1
    submitted = next(
        i for i, a in enumerate(app.acted) if a.effect == "submit_transfer"
    )
    submit_expect = app.expected[submitted]
    assert submit_expect is not None
    assert submit_expect.control == "c12"


@pytest.mark.rule(6)
def test_a_changed_record_during_approval_is_never_submitted(profile) -> None:
    app = bank()

    def swap(request):
        app.screens["transfer"].nodes = tuple(
            dd("Member number", "10002", "member-number")
            if node.name == "10001"
            else node
            for node in app.screens["transfer"].nodes
        )
        return approve(request)

    people = People([swap, HandoffOutcome.APPROVED, HandoffOutcome.APPROVED])

    result, _ = run(transfer_capability(), app, people, profile)

    assert app.acted_on("Submit transfer") == 0
    assert result.reason is Reason.RECORD_MISMATCH


def test_looking_again_at_an_unchanged_screen_keeps_the_approval(profile) -> None:
    app = bank()
    people = People([approve])

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.SUCCEEDED
    assert len(people.requests) == 1
    assert app.looks > 10


def test_an_approval_from_another_replay_is_refused(profile) -> None:
    earlier = []

    def approve_first(request):
        earlier.append(first.controls[0].status())
        return approve(request)

    first = People([approve_first])
    run(transfer_capability(), bank(), first, profile)
    app = bank()
    receipts = []

    def stale_approval(seat, status):
        if status.state is not State.AWAITING_APPROVAL:
            return False
        receipts.append(seat.send(Command.APPROVE, earlier[0]))
        seat.send(Command.TERMINATE, status)
        return True

    control, seat, _ = _controlled(app, [stale_approval])
    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert receipts[0].verdict is Verdict.STALE
    assert result.status is Status.TERMINATED
    assert app.acted_on("Submit transfer") == 0


def test_a_second_answer_to_an_answered_request_is_refused(profile) -> None:
    app = bank()
    app.outcomes = [(Outcome.OK, True)] * 4 + [(Outcome.UNCERTAIN, False)]
    answered = []
    receipts = []

    def first_answer(seat, status):
        if status.state is not State.AWAITING_APPROVAL:
            return False
        answered.append(status)
        seat.send(Command.RESUME, status)
        return True

    control, seat, _ = _controlled(app, [first_answer])
    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        log=MemoryReplayLog(),
        clock=seat.clock,
    )
    # The same answer again, for the request it already settled.
    receipts.append(seat.send(Command.RESUME, answered[0]))

    assert receipts[0].verdict in {Verdict.STALE, Verdict.DUPLICATE}
    assert result.status is Status.NEEDS_HELP
    assert app.acted_on("Teller PIN") == 0


# 3. A target is resolved only on the screen it was recorded on.


def test_matching_controls_on_another_allowed_route_are_not_used(profile) -> None:
    """Regression: same-named controls under /desk passed for the member page."""
    app = bank()
    app.screens["desk"] = Page("/desk", app.screens["member"].nodes)
    app.rules[("search", ActionKind.CLICK, "Search")] = "desk"
    app.rules[("desk", ActionKind.CLICK, "Transfer")] = "transfer"
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(transfer_capability(), app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert not [a for a in app.acted if a.kind is ActionKind.READ]
    assert app.acted_on("Transfer") == 0


def test_an_action_waits_for_its_own_screen(profile) -> None:
    app = bank()
    # The search lands on a desk page that shows the member screen's controls.
    app.screens["member"] = Page("/desk", app.screens["member"].nodes)
    capability = replace_node(
        transfer_capability(),
        "search",
        transitions=(
            go("read_balance", Present("transfer_link")),
            go("not_found", Present("no_member")),
        ),
    )
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(capability, app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert not [a for a in app.acted if a.kind is ActionKind.READ]


def test_different_records_on_one_route_template_are_told_apart(profile) -> None:
    app = bank(swapped="10002")
    people = People([HandoffOutcome.APPROVED] * 3)

    result, _ = run(transfer_capability(), app, people, profile)

    assert app.screens["transfer"].path == "/members/10002/transfer"
    assert result.reason is Reason.RECORD_MISMATCH
    assert app.acted_on("Submit transfer") == 0


def test_navigation_starts_on_its_source_screen_not_its_destination(profile) -> None:
    base = transfer_capability()
    open_member = ActionNode(
        "open_member",
        ActionKind.NAVIGATE,
        "/members",
        None,
        None,
        Destination("/members/:id", (Param("id", MEMBER),)),
        None,
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (),
        (go("read_balance"),),
    )
    capability = dataclasses.replace(
        base,
        entry="open_member",
        nodes=(
            open_member,
            *(
                n
                for n in base.nodes
                if n.node_id not in {"enter_member", "search", "not_found"}
            ),
        ),
        outcomes=(),
    )
    assert validate(capability) == ()
    app = bank()

    def arrive(_app, action):
        if action.kind is ActionKind.NAVIGATE:
            _app.state = "member"

    app.on_act = arrive

    result, _ = run(capability, app, People([approve]), profile)

    assert result.status is Status.SUCCEEDED
    navigation = next(a for a in app.acted if a.kind is ActionKind.NAVIGATE)
    assert navigation.destination == "https://bank.test/members/10001"


# 4. A denied operation never becomes a person's task.


def test_an_artifact_with_a_person_doing_a_denied_operation_is_invalid() -> None:
    capability = login_capability()
    code = capability.node("code")
    assert isinstance(code, HumanNode)
    denied = SavedRestriction(
        RestrictionScope.ROUTE,
        "/desk",
        (ActionKind.CLICK,),
        None,
        "",
        Limit.DENY,
        Source.FINDING,
    )
    clicking = replace_node(
        dataclasses.replace(capability, restrictions=(denied,)),
        "code",
        performs=ActionKind.CLICK,
    )
    typing = replace_node(clicking, "code", performs=ActionKind.TYPE)
    everything = dataclasses.replace(
        typing, restrictions=(dataclasses.replace(denied, kinds=()),)
    )

    person = Issue(IssueCode.DENIED_OPERATION, "nodes[1]")
    assert person in validate(clicking)
    assert person not in validate(typing)
    assert person in validate(everything)


def test_a_persons_step_is_held_to_the_profiles_effect_exceptions(profile) -> None:
    capability = replace_node(
        login_capability(),
        "code",
        performs=ActionKind.PRESS_KEY,
        effect="post_entry",
    )
    ungranted = replace_node(capability, "code", performs=ActionKind.DRAG)

    assert check_profile(capability, profile) == ()
    assert IssueCode.NOT_GRANTED in {i.code for i in check_profile(ungranted, profile)}


def test_a_denied_operation_as_a_persons_step_is_refused_before_replay(
    profile,
) -> None:
    denied = SavedRestriction(
        RestrictionScope.ROUTE,
        "/desk",
        (ActionKind.TYPE,),
        None,
        "",
        Limit.DENY,
        Source.OPERATOR,
    )
    capability = replace_node(
        dataclasses.replace(login_capability(), restrictions=(denied,)),
        "code",
        performs=ActionKind.TYPE,
    )
    app = desk()

    result, _ = run(capability, app, People([]), profile, inputs={})

    assert result.reason is Reason.INVALID_ARTIFACT
    assert app.acted == []


# 5. Identity comparisons and incomplete observations.


@pytest.mark.rule(2)
@pytest.mark.parametrize(
    ("shown", "expected", "holds"),
    [
        ("1123456", "12345", False),
        ("Member 123456", "12345", False),
        ("Member 12345", "12345", True),
        ("Member 12345, J. Doe", "12345", True),
        # A hyphen joins an identifier's parts: "A-12345-B" is another
        # identifier, as discovery always judged it (rule 2).
        ("A-12345-B", "12345", False),
        ("Acct 12345-01", "12345", False),
        ("Balance -$4.00", "$4.00", False),
        ("Member (12345)", "12345", True),
        ("Transfer submitted", "submitted", True),
        ("Transfer resubmitted", "submitted", False),
    ],
)
def test_contains_matches_whole_words_only(shown, expected, holds) -> None:
    assert text_matches(shown, expected, Match.CONTAINS) is holds


def test_a_member_number_inside_another_is_not_that_member(profile) -> None:
    """Regression: CONTAINS accepted member 12345 inside 1123456."""
    capability = transfer_capability()
    contains = Shows("member_shown", MEMBER, Match.CONTAINS)
    capability = replace_node(
        capability,
        "search",
        transitions=(
            go("read_balance", contains),
            go("not_found", Present("no_member")),
        ),
    )
    app = bank(member="12345", shown="1123456")
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(
        capability, app, people, profile, inputs={**INPUTS, "member_id": "12345"}
    )

    assert result.status is Status.NEEDS_HELP
    assert not [a for a in app.acted if a.kind is ActionKind.READ]


def absent_capability():
    """Sign in, then succeed when no warning shows. A synthetic fixture."""
    base = login_capability()
    sign_in = base.node("sign_in")
    assert isinstance(sign_in, ActionNode)
    which = CheckNode(
        "which",
        (go("done", Absent("expired")), go("expired", Present("expired"))),
    )
    return dataclasses.replace(
        base,
        nodes=(
            dataclasses.replace(
                sign_in, verify=(AtRoute("/desk"),), transitions=(go("which"),)
            ),
            which,
            ResultNode("done", ResultKind.SUCCESS, "", (Absent("expired"),)),
            base.node("expired"),
        ),
    )


def test_absence_is_not_proved_by_a_partial_observation(profile) -> None:
    """Regression: a truncated tree that left out the warning counted as absence."""
    app = desk()
    app.rules[("start", ActionKind.CLICK, "Sign in")] = "c"
    app.omit = {"Password expired"}

    def truncate(app):
        if app.state == "c":
            app.coverage = ObservationStatus.PARTIAL

    app.on_observe = truncate
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(absent_capability(), app, people, profile, inputs={})

    assert result.status is not Status.SUCCEEDED
    assert result.status is Status.NEEDS_HELP
    assert result.reason in {Reason.NO_TRANSITION, Reason.HELP_UNRESOLVED}


def test_absence_on_a_complete_observation_still_holds(profile) -> None:
    app = desk()
    app.rules[("start", ActionKind.CLICK, "Sign in")] = "b"

    result, _ = run(absent_capability(), app, People([]), profile, inputs={})

    assert result.status is Status.SUCCEEDED


@dataclasses.dataclass
class _Truncating(App):
    """Leaves out a repeated Search button from every look after the first."""

    seen: int = 0

    def observe(self, request):
        self.seen += 1
        observation = super().observe(request)
        if self.seen == 1:
            return observation
        kept: list[AxNode] = []
        for node in observation.nodes:
            if node.name == "Search" and any(k.name == "Search" for k in kept):
                continue
            kept.append(node)
        return dataclasses.replace(
            observation, status=ObservationStatus.PARTIAL, nodes=tuple(kept)
        )


def test_one_match_in_a_partial_observation_is_not_proof_it_is_the_only_one(
    profile,
) -> None:
    base = bank()
    app = _Truncating(base.screens, base.state, base.rules)
    # The application has two Search buttons. After the entry check, every
    # observation is truncated and shows only one of them.
    app.screens["search"].nodes += (button("Search"),)
    people = People([HandoffOutcome.RESUMED] * 3)

    result, _ = run(transfer_capability(), app, people, profile)

    assert app.acted_on("Search") == 0
    assert Reason.OBSERVATION_INCOMPLETE in {r.reason for r in result.history}
    assert result.status is Status.NEEDS_HELP


def test_hidden_duplicates_do_not_count_as_matches(profile) -> None:
    app = bank()
    app.screens["search"].nodes += (
        AxNode("button", "Search", tag="button", visible=False),
    )

    result, _ = run(transfer_capability(), app, People([approve]), profile)

    assert result.status is Status.SUCCEEDED


@dataclasses.dataclass(frozen=True, slots=True)
class _Window:
    start: int
    shown: int
    total: int
    counted: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class _Windowed(Observation):
    window: _Window | None = None


@pytest.mark.parametrize(
    ("window", "whole"),
    [
        (None, True),
        (_Window(0, 10, 10), True),
        (_Window(0, 10, 40), False),
        (_Window(10, 10, 20), False),
        (_Window(0, 10, 10, counted=False), False),
    ],
)
def test_a_window_over_part_of_a_screen_is_incomplete(window, whole) -> None:
    seen = _Windowed(
        "o1",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState("https://bank.test/members"),
        window=window,
    )

    assert complete(seen) is whole


# 6. Every look and every internal read carries the full policy verdict.


def test_risky_observation_asks_before_every_look(tmp_path) -> None:
    risky = load(
        tmp_path, PROFILE.replace("observe: {any: safe}", "observe: {any: risky}")
    )
    capability = transfer_capability(limits=limits(max_help_requests=200))
    app = bank()
    people = People([always(HandoffOutcome.APPROVED)] * 200)

    result, _ = run(capability, app, people, risky)

    looks = [
        r for r in people.requests if r.action and r.action.kind.value == "observe"
    ]
    assert result.status is Status.SUCCEEDED
    assert len(looks) == app.looks
    assert all(r.ask is Ask.APPROVAL for r in looks)


def test_a_declined_risky_look_reads_nothing(tmp_path) -> None:
    risky = load(
        tmp_path, PROFILE.replace("observe: {any: safe}", "observe: {any: risky}")
    )
    app = bank()

    result, _ = run(transfer_capability(), app, People([HandoffOutcome.RESUMED]), risky)

    assert app.looks == 0
    assert app.acted == []
    assert result.reason is Reason.INCOMPATIBLE_APPLICATION


def roomy():
    """The transfer, allowed enough requests for a check read's approval.

    A check that compares text is a read (rule 19), so with reads declared
    risky the search step's check asks once too.
    """
    capability = transfer_capability()
    limits = dataclasses.replace(capability.limits, max_help_requests=6)
    return dataclasses.replace(capability, limits=limits)


def test_risky_completion_reads_need_approval(tmp_path) -> None:
    """Regression: completion reads ignored a risky verdict."""
    risky = load(tmp_path, PROFILE.replace("read: {any: safe}", "read: {any: risky}"))
    app = bank()
    people = People([approve] * 4)

    result, _ = run(roomy(), app, people, risky)

    reads = [
        r for r in people.requests if r.action and r.action.kind is ActionKind.READ
    ]
    assert result.status is Status.SUCCEEDED
    # The search step's check compares text, so it is a read too (rule 19).
    assert len(reads) == 3
    assert [r.node for r in result.history if r.node != "submit"] == [
        "search",
        "read_balance",
        "done",
    ]


def test_a_declined_completion_read_is_not_a_success(tmp_path) -> None:
    risky = load(tmp_path, PROFILE.replace("read: {any: safe}", "read: {any: risky}"))
    people = People([approve, approve, approve, HandoffOutcome.RESUMED])

    result, _ = run(roomy(), bank(), people, risky)

    assert result.reason is Reason.COMPLETION_CHECK_FAILED


def test_safe_permitted_reads_and_looks_ask_nobody(profile) -> None:
    people = People([approve])

    result, _ = run(transfer_capability(), bank(), people, profile)

    assert result.status is Status.SUCCEEDED
    kinds = [r.action.kind for r in people.requests if r.action]
    assert kinds == [ActionKind.CLICK]


def test_a_saved_restriction_applies_to_a_completion_read(profile) -> None:
    def restricted(limit):
        saved = SavedRestriction(
            RestrictionScope.TARGET,
            "/members/:id/transfer",
            (ActionKind.READ,),
            "transfer_member",
            "",
            limit,
            Source.OPERATOR,
        )
        return transfer_capability(restrictions=(saved,))

    denied, _ = run(restricted(Limit.DENY), bank(), People([approve]), profile)
    people = People([approve, approve])
    risky, _ = run(restricted(Limit.RISKY), bank(), people, profile)

    assert denied.reason is Reason.RESTRICTION_DENIED
    assert risky.status is Status.SUCCEEDED
    assert [r.node for r in risky.history] == ["submit", "done"]


def test_a_visual_capture_passes_the_gate_first(tmp_path) -> None:
    risky = load(
        tmp_path, PROFILE.replace("observe: {any: safe}", "observe: {any: risky}")
    )
    recorded = png((200, 120), ((30, 40),))
    capability = canvas_capability(crop(recorded, 30, 40, 40, 20))
    captures = Captures([recorded])
    app = desk()
    app.rules[("start", ActionKind.CLICK, "screen")] = "b"
    looked = 0

    def approve_structured_only(_request):
        nonlocal looked
        looked += 1
        return HandoffOutcome.APPROVED if looked == 1 else HandoffOutcome.RESUMED

    people = People([approve_structured_only] * 10)

    run(capability, app, people, risky, inputs={}, scenes=captures)

    assert captures.taken == 0
    assert app.acted == []


@dataclasses.dataclass
class _Declaring(App):
    """Declares what it performs the way the discovery branch's surfaces do."""

    def capabilities(self):
        return {
            ActionKind.TYPE: frozenset({TargetForm.ACCESSIBILITY}),
            ActionKind.READ: frozenset({TargetForm.DOM}),
        }


def test_a_surface_that_declares_no_support_for_an_action_is_not_sent_it(
    profile,
) -> None:
    base = bank()
    app = _Declaring(base.screens, base.state, base.rules)

    result, _ = run(transfer_capability(), app, People([]), profile)

    assert result.reason is Reason.UNSUPPORTED_BY_SURFACE
    assert app.acted_on("Search") == 0


# 7. The budget is checked immediately before anything reaches the surface.


def test_an_observation_that_spends_the_budget_stops_the_next_input(profile) -> None:
    """Regression: typing was sent after an observation used the remaining time."""
    app = bank()
    clock = Clock()
    looks = 0

    def slow(app):
        # The third look at the transfer screen is the typing node's own look,
        # taken after the node began and before its input is sent.
        nonlocal looks
        if app.state == "transfer":
            looks += 1
            if looks == 3:
                clock.advance(1000)

    app.on_observe = slow

    result, _ = run(transfer_capability(), app, People([]), profile, clock=clock)

    assert result.reason is Reason.BUDGET_EXHAUSTED
    assert app.acted_on("Amount") == 0


def test_a_retry_after_the_budget_is_spent_is_not_sent(profile) -> None:
    app = bank()
    clock = Clock()
    app.outcomes = [(Outcome.STALE, False)]

    def slow(_app, action):
        if action.kind is ActionKind.TYPE:
            clock.advance(1000)

    app.on_act = slow

    result, _ = run(transfer_capability(), app, People([]), profile, clock=clock)

    assert result.reason is Reason.BUDGET_EXHAUSTED
    assert app.acted_on("Member number") == 1


def test_a_wait_for_a_person_neither_spends_nor_resets_the_budget(profile) -> None:
    app = bank()
    clock = Clock()
    worked = False

    def work(app):
        nonlocal worked
        if app.state == "member" and not worked:
            worked = True
            clock.advance(100)

    def slow(request):
        clock.advance(590)
        return approve(request)

    app.on_observe = work

    result, _ = run(transfer_capability(), app, People([slow]), profile, clock=clock)

    assert result.status is Status.SUCCEEDED
    assert result.paused_s == pytest.approx(590)
    assert result.active_s == pytest.approx(100)


def test_attempts_within_one_node_are_bounded(profile) -> None:
    capability = transfer_capability(limits=limits(max_help_requests=5))
    app = bank()
    app.screens["search"].nodes = app.screens["search"].nodes[:1]
    people = People([HandoffOutcome.RESUMED] * 50)

    result, _ = run(capability, app, people, profile)

    assert result.status is Status.NEEDS_HELP
    assert result.reason is Reason.TARGET_NOT_FOUND
    assert len(people.requests) == 5
    # Two looks per round of retries, one per hand-back, and the entry check.
    assert app.looks <= 1 + 1 + 5 * 3


# 8. A dialog answer names the dialog it was decided about.


def dialog_capability(kind: ActionKind):
    """Post an entry, which opens a confirmation dialog, then answer it."""
    base = login_capability()
    post = ActionNode(
        "post",
        ActionKind.CLICK,
        "/desk",
        "post",
        None,
        None,
        "post_entry",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (),
        (go("answer", DialogOpen("confirm", constant("Post this entry?"))),),
    )
    answer = ActionNode(
        "answer",
        kind,
        "/desk",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Approval.EACH_RUN,
        True,
        (DialogOpen("confirm", constant("Post this entry?")),),
        (Present("result"),),
        (go("done"),),
    )
    return dataclasses.replace(
        base,
        capability_id="post_entry",
        application=dataclasses.replace(base.application, markers=(Present("post"),)),
        outcomes=(),
        targets=(
            ax("post", "/desk", "button", "Post"),
            ax("result", "/desk", "status", "Done"),
        ),
        entry="post",
        nodes=(
            post,
            answer,
            ResultNode("done", ResultKind.SUCCESS, "", (Present("result"),)),
        ),
    )


def dialog_app() -> App:
    pages = {
        "start": Page("/desk", (button("Post"),)),
        "done": Page("/desk", (status("Done"),)),
    }
    return App(
        pages,
        "start",
        {
            ("start", ActionKind.ACCEPT_DIALOG, "dialog"): "done",
            ("start", ActionKind.DISMISS_DIALOG, "dialog"): "done",
        },
        opens={("start", ActionKind.CLICK, "Post"): "confirm"},
    )


@pytest.mark.parametrize("kind", [ActionKind.ACCEPT_DIALOG, ActionKind.DISMISS_DIALOG])
def test_a_dialog_answer_names_the_observed_dialog(profile, kind) -> None:
    """Regression: the answer's expectation named no dialog, so a browser refused it."""
    app = dialog_app()

    result, _ = run(dialog_capability(kind), app, People([approve]), profile, inputs={})

    assert result.status is Status.SUCCEEDED
    assert app.answered == [(kind, "dialog-1")]
    answer = app.expected[-1]
    assert answer is not None
    assert answer.dialog == "dialog-1"


def test_a_dialog_replaced_during_approval_needs_its_own_approval(profile) -> None:
    app = dialog_app()

    def replace(request):
        app.open_dialog()
        return approve(request)

    people = People([replace, approve])

    result, _ = run(
        dialog_capability(ActionKind.ACCEPT_DIALOG), app, people, profile, inputs={}
    )

    assert result.status is Status.SUCCEEDED
    assert [r.reason for r in result.history] == [
        Reason.APPROVAL_REQUIRED,
        Reason.APPROVAL_INVALIDATED,
    ]
    assert app.answered == [(ActionKind.ACCEPT_DIALOG, "dialog-2")]


def test_a_dialog_that_disappeared_during_approval_is_not_answered(profile) -> None:
    app = dialog_app()

    def gone(request):
        app.dialog = None
        return approve(request)

    people = People([gone, HandoffOutcome.RESUMED, HandoffOutcome.RESUMED])

    result, _ = run(
        dialog_capability(ActionKind.ACCEPT_DIALOG), app, people, profile, inputs={}
    )

    assert app.answered == []
    assert result.status is Status.NEEDS_HELP


def test_a_changed_confirmation_message_is_never_answered(profile, monkeypatch):
    app = dialog_app()
    original = app.act

    def replaced(action, *, expect=None):
        result = original(action, expect=expect)
        if app.dialog is not None:
            app.dialog = dataclasses.replace(app.dialog, message="Delete all entries?")
        return result

    monkeypatch.setattr(app, "act", replaced)

    def agrees(request):
        # Every approval is granted, so only the message can refuse the answer.
        return (
            HandoffOutcome.APPROVED
            if request.ask is Ask.APPROVAL
            else HandoffOutcome.RESUMED
        )

    people = People([agrees] * 6)
    result, _ = run(
        dialog_capability(ActionKind.ACCEPT_DIALOG), app, people, profile, inputs={}
    )
    assert result.status is not Status.SUCCEEDED
    assert app.answered == []


# 9. Found in a coverage audit of the earlier replay harness: a Stop that the
# surface honoured at the last moment, before any input, ended the replay as
# a failure instead of pausing it.


class _StopsBeforeTyping(App):
    """Takes a Stop just as the replay is about to type, as a browser would.

    The browser checks the run's control at the last moment before input, and
    reports that nothing was sent.
    """

    control: Control | None = None
    seat: ScriptedSeat | None = None
    stopped: bool = False

    def act(self, action, *, expect=None):
        if action.kind is ActionKind.TYPE and not self.stopped:
            assert self.control is not None
            assert self.seat is not None
            self.stopped = True
            self.seat.send(Command.STOP, self.control.status())
            if not self.control.permits():
                return ActionResult(
                    Outcome.BLOCKED, PageState(self.location()), detail=NOT_SENT
                )
        return super().act(action, expect=expect)


def test_a_stop_before_an_input_is_sent_pauses_and_sends_it_once(profile) -> None:
    base = bank()
    app = _StopsBeforeTyping(base.screens, base.state, base.rules)
    control, seat, _ = _controlled(
        app,
        [
            when(State.PAUSED, Command.RESUME),
            when(State.AWAITING_APPROVAL, Command.APPROVE),
        ],
    )
    app.control, app.seat = control, seat

    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        journal=MemoryJournal(),
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert app.stopped
    assert result.status is Status.SUCCEEDED, result
    assert app.acted_on("Member number") == 1
    assert app.acted_on("Submit transfer") == 1


# 10. Found in the same audit: a person who took the session and clicked the
# step's own control, before its result showed, had that step sent again by
# the replay after the hand-back.


def _takes_the_submission(app: App, *, name: str):
    """Stop at the transfer screen and let a person click ``name`` there."""
    flags = {"approved": False, "stopped": False}

    def approve(seat, status):
        if status.state is not State.AWAITING_APPROVAL or flags["approved"]:
            return False
        seat.send(Command.APPROVE, status)
        flags["approved"] = True
        return True

    def by_hand(seat) -> None:
        # The application takes the click, but its result shows only later.
        seat.person(ManualKind.CLICK, role="button", name=name)

    control, seat, _ = _controlled(
        app,
        [
            approve,
            when(State.PAUSED, Command.TAKE_CONTROL),
            doing(State.HUMAN_CONTROL, by_hand),
            when(State.HUMAN_CONTROL, Command.RESUME),
            # Anything the replay asks after this, a person answers by resuming.
            *[when(State.AWAITING_APPROVAL, Command.APPROVE) for _ in range(3)],
        ],
    )

    def stop_on_the_transfer_screen(current: App) -> None:
        if flags["approved"] and not flags["stopped"] and current.state == "transfer":
            flags["stopped"] = True
            seat.send(Command.STOP, control.status())

    app.on_observe = stop_on_the_transfer_screen
    return control, seat, flags


def test_a_step_a_person_may_have_done_is_never_sent_again(profile) -> None:
    app = bank()
    control, seat, flags = _takes_the_submission(app, name="Submit transfer")

    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        journal=MemoryJournal(),
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert flags["stopped"]
    assert app.acted_on("Submit transfer") == 0
    assert result.status is not Status.SUCCEEDED


@pytest.mark.rule(5, 14)
def test_a_person_clicking_another_control_does_not_stop_the_step(profile) -> None:
    app = bank()
    control, seat, flags = _takes_the_submission(app, name="Cancel")

    result = replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        journal=MemoryJournal(),
        log=MemoryReplayLog(),
        clock=seat.clock,
    )

    assert flags["stopped"]
    assert result.status is Status.SUCCEEDED, result
    assert app.acted_on("Submit transfer") == 1
    assert (
        sum(
            item.answer.value == HandoffOutcome.APPROVED.value
            for item in result.history
        )
        == 2
    )


# 11. Found in a blind review: a saved click that submits a form asked for
# approval only when its effect label or its node said so. Replay now reads
# the operator's rule for submissions from the control the target found.


def _submitting_search() -> App:
    app = bank()
    search = app.screens["search"]
    submit = AxNode(
        "button",
        "Search",
        tag="button",
        control="d1:c4",
        form="d1:c2",
        submits="d1:c2>d1:c4",
    )
    app.screens["search"] = Page(search.path, (*search.nodes[:1], submit))
    return app


@pytest.mark.parametrize(
    ("submissions", "asked"), [("risky", True), ("by_effect", False)]
)
def test_a_saved_submission_follows_the_profiles_rule(tmp_path, submissions, asked):
    profile = load(
        tmp_path,
        PROFILE.replace(
            "version: 4\n", f"version: 6\norigins: []\nsubmissions: {submissions}\n"
        ),
    )
    app = _submitting_search()
    people = People([HandoffOutcome.REJECTED, approve_all, approve_all])

    result, _ = run(transfer_capability(), app, people, profile)

    search = [r for r in result.history if r.node == "search"]
    assert bool(search) is asked
    if asked:
        assert search[0].ask.value == Ask.APPROVAL.value
        assert result.reason is Reason.APPROVAL_REJECTED
        assert app.acted_on("Search") == 0


def approve_all(request):
    return (
        HandoffOutcome.APPROVED
        if request.ask is Ask.APPROVAL
        else HandoffOutcome.RESUMED
    )
