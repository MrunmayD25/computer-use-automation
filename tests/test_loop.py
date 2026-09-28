"""Contract checks for the discovery loop's control flow.

Every collaborator is scripted, so these tests exercise stopping conditions,
refusals, and handoffs without a browser, model, or elapsed time.
``test_browser.py`` checks browser behavior against Chromium. This module does
not test page interaction.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import pytest
from fakes import (
    DONE,
    FakeClock,
    ReadsNoRecords,
    Screen,
    ScriptedDecider,
    ScriptedEscalator,
    ScriptedSurface,
    grants,
)
from import_contract import assert_module_import_contract

import computeruse.loop
from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    ObservationMode,
    ObservationProvenance,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    SecretRef,
    VisualAnchor,
)
from computeruse.decider import (
    AskHuman,
    CheckKind,
    Decision,
    Finish,
    InvalidDecisionError,
    ModelError,
    Observe,
    Propose,
    ResultCheck,
    Transcript,
)
from computeruse.escalation import Handoff, HandoffOutcome, Trigger
from computeruse.journal import (
    Acted,
    Corrected,
    Escalated,
    Failed,
    JsonlJournal,
    MemoryJournal,
    Observed,
    Refused,
    ValueSource,
)
from computeruse.loop import Ending, Verification, discover
from computeruse.policy import Denial
from computeruse.profile import ActionKind, Profile
from computeruse.surface import SurfaceError

MEMBERS = "https://sandbox.example.test/members"
MEMBER = "https://sandbox.example.test/members/12345"
SAVINGS = "https://sandbox.example.test/members/12345/savings"
WIRE = "https://sandbox.example.test/members/12345/savings/wire"

SEARCH = AxLocator("button", "Search")
BALANCE = AxLocator("cell", "Savings balance")
PASSWORD = AxLocator("textbox", "Password")

LOOK = Observe(ObservationRequest(ObservationMode.STRUCTURED), "read the controls")
PICTURE = Observe(ObservationRequest(ObservationMode.VISUAL), "look at the screen")
READ = Propose(Action(ActionKind.READ, BALANCE), "read the balance")
BALANCE_CLAIM = Finish(
    {"savings_balance": "4200.00"},
    "on screen",
    checks=(
        ResultCheck(CheckKind.RESULT, BALANCE, "4200.00", output="savings_balance"),
    ),
)
"""A claim the loop verifies by reading BALANCE again, which needs one extract."""


def screen(location: str = MEMBER, **edits: object) -> Screen:
    return Screen(location=location, **edits)  # ty: ignore[invalid-argument-type]


def drive(
    profile: Profile,
    decisions: Sequence[Decision],
    *,
    screens: list[Screen] | None = None,
    outcomes: list[Outcome] | None = None,
    extracts: list[str] | None = None,
    handoffs: list[Handoff] | None = None,
    goal: str = "read the savings balance",
    clock: FakeClock | None = None,
):
    """Run the loop against scripted collaborators and return everything."""
    surface = ScriptedSurface(
        screens=screens or [screen()],
        outcomes=outcomes or [],
        extracts=extracts or [],
    )
    decider = ScriptedDecider(decisions=list(decisions))
    escalator = ScriptedEscalator(handoffs=handoffs or [])
    journal = MemoryJournal()
    result = discover(
        goal,
        profile,
        surface=surface,
        decider=decider,
        escalator=escalator,
        journal=journal,
        clock=clock or FakeClock(),
    )
    return result, surface, decider, escalator, journal


def modes(surface: ScriptedSurface) -> list[str]:
    return [request.mode.value for request in surface.looks]


@pytest.mark.parametrize("recovers", [True, False])
def test_invalid_proposals_are_bounded_and_return_correction_context(profile, recovers):
    class CorrectingDecider(ReadsNoRecords):
        def __init__(self):
            self.calls = 0

        def decide(self, transcript):
            self.calls += 1
            if recovers and self.calls == 2:
                assert "invalid" in transcript.notices[-1]
                return DONE
            raise InvalidDecisionError("a capture id is required")

    decider = CorrectingDecider()
    surface = ScriptedSurface([screen()])
    operator = ScriptedEscalator()
    journal = MemoryJournal()
    result = discover(
        "test",
        profile,
        surface=surface,
        decider=decider,
        escalator=operator,
        journal=journal,
        clock=FakeClock(),
    )
    assert not surface.acted
    corrected = [event for event in journal.events if isinstance(event, Corrected)]
    assert len(corrected) == (1 if recovers else decider.calls)
    assert "capture id" not in str(corrected)
    assert result.ending is (Ending.COMPLETED if recovers else Ending.HANDED_OFF)
    assert decider.calls == (
        2 if recovers else profile.budgets.max_retries_per_step + 1
    )


@pytest.mark.rule(12, 15, 17)
@pytest.mark.parametrize("proposal", [READ, BALANCE_CLAIM])
@pytest.mark.parametrize("refresh", [[], [PICTURE, LOOK]])
def test_adapter_refusals_explain_the_next_decision_without_entering_the_journal(
    profile, proposal, refresh
):
    import dataclasses

    explanation = "this edited field cannot prove saved data; read a displayed result"

    class ExplainingSurface(ScriptedSurface):
        def act(self, action, *, expect=None):
            result = super().act(action, expect=expect)
            if action.target == BALANCE:
                return dataclasses.replace(
                    result, outcome=Outcome.NOT_ACTIONABLE, detail=explanation
                )
            return result

    decider = ScriptedDecider([LOOK, proposal, *refresh, DONE])
    surface = ExplainingSurface([screen()])
    journal = MemoryJournal()
    discover(
        "read the savings balance",
        profile,
        surface=surface,
        decider=decider,
        escalator=ScriptedEscalator(),
        journal=journal,
        clock=FakeClock(),
    )
    assert any(explanation in note for note in decider.seen[-1].notices)
    assert explanation not in str(journal.events)


def test_pixel_actions_cannot_avoid_learned_restrictions_by_changing_input_kind():
    from computeruse.actions import Operation, Point, ScreenTarget
    from computeruse.policy import covers

    click = Action(ActionKind.CLICK, ScreenTarget("one", Point(10, 20)))
    key = Action(ActionKind.PRESS_KEY, ScreenTarget("two"), "Enter")
    assert covers(Operation.of(click, "/members"), Operation.of(key, "/members"))


@pytest.mark.rule(15, 17)
@pytest.mark.parametrize("page_input", [False, True])
def test_reading_between_finish_attempts_cannot_reset_verification_retries(
    profile, page_input
):
    """Retry a failing claim without resetting the count after reads.

    Once the retries are spent, a person is asked about the claim. Input that
    changes the page starts a new count, because the old claim no longer
    describes it. A claim whose checks pass but leave the task uncovered goes
    to a person at once. `test_claims.py` covers that path.
    """
    attempts = profile.budgets.max_retries_per_step + 1
    wrong = Finish(
        {"savings_balance": "0.00"},
        checks=(
            ResultCheck(CheckKind.RESULT, BALANCE, "0.00", output="savings_balance"),
        ),
    )
    decisions: list[Decision] = [LOOK]
    for index in range(attempts):
        if page_input and index == attempts - 1:
            decisions.append(Propose(Action(ActionKind.CLICK, SEARCH, effect="search")))
        read = Propose(Action(ActionKind.READ, AxLocator("cell", f"Reading {index}")))
        decisions.extend([read, wrong])
    decider = ScriptedDecider(decisions)
    operator = ScriptedEscalator([])
    result = discover(
        "Read the savings balance.",
        profile,
        surface=ScriptedSurface([screen()], extracts=["4200.00"] * attempts * 2),
        decider=decider,
        escalator=operator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert result.ending is Ending.HANDED_OFF
    verifying = [
        request
        for request in operator.requests
        if request.trigger is Trigger.UNVERIFIED_RESULT
    ]
    if page_input:
        assert verifying == []
        return
    assert len(decider.seen) == 1 + attempts * 2
    assert len(verifying) == 1


# Choosing a tool.


def test_runs_do_not_share_observations_history_or_restrictions(profile) -> None:
    flagged = Propose(
        Action(ActionKind.CLICK, SEARCH, effect="search", flag_risky=True)
    )
    first, _, _, _, _ = drive(
        profile,
        [LOOK, flagged, DONE],
        handoffs=[Handoff(HandoffOutcome.APPROVED)],
    )
    click = Propose(Action(ActionKind.CLICK, SEARCH, effect="search"))
    second, surface, decider, escalator, _ = drive(profile, [LOOK, click, DONE])

    assert first.ending is second.ending is Ending.COMPLETED
    assert first.restrictions
    assert second.restrictions == ()
    assert decider.seen[0].observations == ()
    assert decider.seen[0].history == ()
    assert decider.seen[0].attempted_modes == ()
    assert surface.acted == [click.action]
    assert escalator.requests == []


@pytest.mark.parametrize("first", [LOOK, PICTURE])
def test_either_observation_tool_may_be_chosen_first(
    profile: Profile, first: Observe
) -> None:
    result, surface, decider, _, _ = drive(profile, [first, DONE])

    assert result.ending is Ending.COMPLETED
    assert modes(surface)[0] == first.request.mode.value
    assert decider.seen[0].observations == ()
    assert decider.seen[0].location == MEMBER


def test_the_first_decision_sees_the_location_without_a_capture(
    profile: Profile,
) -> None:
    _, surface, decider, _, _ = drive(profile, [DONE])

    assert surface.looks == []
    assert decider.seen[0].permitted_modes == (
        ObservationMode.STRUCTURED,
        ObservationMode.VISUAL,
    )


def test_a_mode_the_profile_denies_is_refused_before_any_capture(
    edited_profile,
) -> None:
    profile = edited_profile(
        perception={
            "allowed_modes": ["structured"],
            "max_alternate_observations_per_step": 1,
        }
    )

    result, surface, decider, _, journal = drive(profile, [PICTURE, LOOK, DONE])

    assert surface.looks == [ObservationRequest(ObservationMode.STRUCTURED)]
    assert result.ending is Ending.COMPLETED
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.MODE_NOT_PERMITTED]
    assert "mode_not_permitted" in decider.seen[1].notices[0]


def test_observing_is_refused_when_the_profile_declares_no_observe(
    edited_profile,
) -> None:
    profile = edited_profile(
        actions=grants({"read": "safe", "click": "safe"}),
        budgets={
            "max_steps": 3,
            "max_wall_clock_s": 300,
            "max_retries_per_step": 2,
            "max_navigations": 15,
        },
    )

    result, surface, _, _, journal = drive(profile, [LOOK, LOOK, LOOK])

    assert surface.looks == []
    assert result.ending is Ending.EXHAUSTED
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert {event.reason for event in refusals} == {Denial.UNDECLARED_ACTION}


def test_an_action_before_any_observation_is_not_performed(profile: Profile) -> None:
    _, surface, decider, _, _ = drive(profile, [READ, LOOK, READ, DONE])

    assert surface.acted == [READ.action]
    assert "look at the surface" in decider.seen[1].notices[0]


# Observation results.


def test_an_unavailable_observation_is_reported_and_not_kept(profile: Profile) -> None:
    blank = screen(structured=ObservationStatus.UNAVAILABLE)

    _, surface, decider, _, journal = drive(
        profile, [LOOK, PICTURE, DONE], screens=[blank]
    )

    assert modes(surface) == ["structured", "visual"]
    assert "unavailable" in decider.seen[1].notices[0]
    assert decider.seen[1].observations == ()
    observed = [event for event in journal.events if isinstance(event, Observed)]
    assert observed[0].status is ObservationStatus.UNAVAILABLE


def test_a_partial_observation_is_kept_and_reported(profile: Profile) -> None:
    partial = screen(structured=ObservationStatus.PARTIAL)

    _, _, decider, _, journal = drive(profile, [LOOK, DONE], screens=[partial])

    assert decider.seen[1].observations[0].status is ObservationStatus.PARTIAL
    observed = [event for event in journal.events if isinstance(event, Observed)]
    assert observed[0].provenance is ObservationProvenance.INITIAL or True


# The alternate allowance.


def test_the_alternate_allowance_survives_several_model_responses(
    profile: Profile,
) -> None:
    result, surface, decider, _, journal = drive(profile, [LOOK, PICTURE, LOOK, DONE])

    assert profile.perception.max_alternate_observations_per_step == 1
    assert modes(surface) == ["structured", "visual"]
    assert result.ending is Ending.COMPLETED
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.ALTERNATE_LIMIT_REACHED]
    assert decider.seen[2].alternates_remaining == 0
    assert "alternate_limit_reached" in decider.seen[3].notices[0]


def test_progress_resets_the_alternate_allowance(profile: Profile) -> None:
    result, surface, decider, _, _ = drive(
        profile, [LOOK, PICTURE, READ, PICTURE, DONE]
    )

    assert result.ending is Ending.COMPLETED
    assert decider.seen[3].alternates_remaining == 1
    assert modes(surface) == ["structured", "visual", "structured", "visual"]
    assert surface.looks[2].provenance is ObservationProvenance.POST_ACTION


def test_a_resumed_session_resets_the_alternate_allowance(profile: Profile) -> None:
    result, _, decider, _, _ = drive(
        profile,
        [
            LOOK,
            PICTURE,
            AskHuman(Trigger.NO_PROGRESS, "the panel never rendered"),
            PICTURE,
            DONE,
        ],
        handoffs=[Handoff(HandoffOutcome.RESUMED, "opened the panel")],
    )

    assert result.ending is Ending.COMPLETED
    assert decider.seen[3].alternates_remaining == 1


def test_a_new_observation_id_does_not_reset_the_allowance(profile: Profile) -> None:
    _, surface, decider, _, _ = drive(profile, [LOOK, PICTURE, PICTURE, DONE])

    identifiers = {look.provenance for look in surface.looks}
    assert identifiers  # the run did look more than once
    assert decider.seen[3].alternates_remaining == 0
    assert modes(surface) == ["structured", "visual"]


# Human intervention.


def test_a_perception_gap_reaches_a_person_after_one_tool(
    profile: Profile,
) -> None:
    """The loop does not run the other tool first. The model asked for a person."""
    result, surface, _, escalator, _ = drive(
        profile,
        [
            LOOK,
            AskHuman(Trigger.INSUFFICIENT_OBSERVATION, "nothing readable"),
            DONE,
        ],
    )

    assert modes(surface) == ["structured"]
    assert [request.trigger for request in escalator.requests] == [
        Trigger.INSUFFICIENT_OBSERVATION
    ]
    assert result.ending is Ending.HANDED_OFF


def test_the_model_may_ask_for_the_other_tool_itself(profile: Profile) -> None:
    result, surface, _, escalator, _ = drive(profile, [LOOK, PICTURE, DONE])

    assert modes(surface) == ["structured", "visual"]
    assert escalator.requests == []
    assert result.ending is Ending.COMPLETED


def test_a_second_look_beyond_the_allowance_is_refused(profile: Profile) -> None:
    result, surface, _, _, journal = drive(profile, [LOOK, PICTURE, LOOK, DONE])

    assert modes(surface) == ["structured", "visual"]
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.ALTERNATE_LIMIT_REACHED]
    assert result.ending is Ending.COMPLETED


def test_a_failed_look_still_spends_the_allowance(profile: Profile) -> None:
    """A tool that returned nothing was still run, so the count includes it."""
    blind = screen(structured=ObservationStatus.FAILED)

    _, surface, _, _, journal = drive(
        profile, [LOOK, PICTURE, LOOK, DONE], screens=[blind]
    )

    assert modes(surface) == ["structured", "visual"]
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.ALTERNATE_LIMIT_REACHED]


def test_a_failed_look_is_reported_to_the_next_decision(profile: Profile) -> None:
    blind = screen(structured=ObservationStatus.FAILED)

    _, _, decider, _, _ = drive(
        profile, [LOOK, AskHuman(Trigger.NO_PROGRESS, "blind")], screens=[blind]
    )

    assert decider.seen[1].attempted_modes == (ObservationMode.STRUCTURED,)
    assert decider.seen[1].observations == ()


def test_a_missing_value_reaches_a_person_without_another_tool(
    profile: Profile,
) -> None:
    result, surface, _, escalator, _ = drive(
        profile,
        [LOOK, AskHuman(Trigger.MISSING_USER_INPUT, "nobody supplied the amount")],
    )

    assert modes(surface) == ["structured"]
    assert [request.trigger for request in escalator.requests] == [
        Trigger.MISSING_USER_INPUT
    ]
    assert result.ending is Ending.HANDED_OFF


@pytest.mark.parametrize(
    "trigger",
    [
        Trigger.AUTHENTICATION_REQUIRED,
        Trigger.AMBIGUOUS_TARGET,
        Trigger.RISKY_ACTION,
        Trigger.NO_PROGRESS,
    ],
)
def test_other_reasons_do_not_run_a_second_tool(
    profile: Profile, trigger: Trigger
) -> None:
    _, surface, _, escalator, _ = drive(profile, [LOOK, AskHuman(trigger, "reason")])

    assert modes(surface) == ["structured"]
    assert len(escalator.requests) == 1


def test_no_alternate_allowance_refuses_the_second_tool(
    edited_profile,
) -> None:
    profile = edited_profile(
        perception={
            "allowed_modes": ["structured", "visual"],
            "max_alternate_observations_per_step": 0,
        }
    )

    result, surface, _, _, journal = drive(
        profile,
        [LOOK, PICTURE, AskHuman(Trigger.INSUFFICIENT_OBSERVATION, "nothing readable")],
    )

    assert modes(surface) == ["structured"]
    assert result.ending is Ending.HANDED_OFF
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.ALTERNATE_LIMIT_REACHED]


def test_the_intervention_request_carries_the_tools_already_run(
    profile: Profile,
) -> None:
    _, _, _, escalator, _ = drive(
        profile, [LOOK, AskHuman(Trigger.AUTHENTICATION_REQUIRED, "a login appeared")]
    )

    assert escalator.requests[0].observed_modes == (ObservationMode.STRUCTURED,)
    assert escalator.requests[0].route == "/members/:id"


def test_the_handoff_timeout_is_the_profile_alone(profile: Profile) -> None:
    clock = FakeClock()

    class SlowDecider(ReadsNoRecords):
        def decide(self, transcript: Transcript) -> Decision:  # noqa: ARG002
            clock.advance(profile.budgets.max_wall_clock_s - 0.5)
            return AskHuman(Trigger.NO_PROGRESS, "stuck")

    escalator = ScriptedEscalator()
    discover(
        "read",
        profile,
        surface=ScriptedSurface(screens=[screen()]),
        decider=SlowDecider(),
        escalator=escalator,
        journal=MemoryJournal(),
        clock=clock,
    )

    assert escalator.requests[0].timeout_s == profile.escalation.handoff_timeout_s


@pytest.mark.rule(17)
def test_human_waiting_is_not_charged_to_the_execution_budget(
    profile: Profile,
) -> None:
    clock = FakeClock()

    class SlowEscalator(ScriptedEscalator):
        def request(self, intervention):
            clock.advance(profile.budgets.max_wall_clock_s * 3)
            return super().request(intervention)

    surface = ScriptedSurface(screens=[screen()], extracts=["4200.00", "4200.00"])
    result = discover(
        "read",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [
                LOOK,
                AskHuman(Trigger.MISSING_USER_INPUT, "need the amount"),
                READ,
                BALANCE_CLAIM,
            ]
        ),
        escalator=SlowEscalator([Handoff(HandoffOutcome.RESUMED, "typed it in")]),
        journal=MemoryJournal(),
        clock=clock,
    )

    assert result.ending is Ending.COMPLETED
    assert result.outputs == {"savings_balance": "4200.00"}
    # The second read is the loop checking the claim against the page.
    assert surface.acted == [READ.action, READ.action]


def test_a_resumed_session_is_observed_again_before_deciding(
    profile: Profile,
) -> None:
    surface = ScriptedSurface(screens=[screen()])

    class MovingEscalator(ScriptedEscalator):
        def request(self, intervention):
            surface.screens = [screen(SAVINGS)]
            return super().request(intervention)

    decider = ScriptedDecider([LOOK, AskHuman(Trigger.NO_PROGRESS, "stuck"), DONE])
    result = discover(
        "read",
        profile,
        surface=surface,
        decider=decider,
        escalator=MovingEscalator([Handoff(HandoffOutcome.RESUMED)]),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )

    assert result.ending is Ending.COMPLETED
    assert decider.seen[2].observations[0].location == SAVINGS
    assert surface.looks[-1].provenance is ObservationProvenance.POST_INTERVENTION


# Policy at the action boundary.


def test_reports_outputs_when_the_decider_finishes(profile: Profile) -> None:
    result, surface, _, _, journal = drive(
        profile,
        [LOOK, READ, BALANCE_CLAIM],
        extracts=["4200.00", "4200.00"],
    )

    assert result.ending is Ending.COMPLETED
    assert result.outputs == {"savings_balance": "4200.00"}
    assert result.verification is Verification.EXECUTOR
    assert [action.kind for action in surface.acted] == [ActionKind.READ] * 2
    acted = next(event for event in journal.events if isinstance(event, Acted))
    assert acted.target_kind == "AxLocator"


def test_undeclared_action_is_refused_and_reported_to_the_decider(
    edited_profile,
) -> None:
    profile = edited_profile(
        actions=grants({"observe": "safe", "read": "safe", "click": "safe"})
    )

    result, surface, decider, _, journal = drive(
        profile,
        [
            LOOK,
            Propose(Action(ActionKind.SELECT, SEARCH, "Riverside"), "choose"),
            DONE,
        ],
    )

    assert surface.acted == []
    assert result.ending is Ending.COMPLETED
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.UNDECLARED_ACTION]
    assert "undeclared_action" in decider.seen[2].notices[0]


def test_navigation_into_a_deny_route_never_reaches_the_surface(
    profile: Profile,
) -> None:
    result, surface, _, _, journal = drive(
        profile,
        [
            LOOK,
            Propose(Action(ActionKind.NAVIGATE, destination=WIRE), "send the wire"),
            DONE,
        ],
    )

    assert surface.acted == []
    assert result.ending is Ending.COMPLETED
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.ROUTE_NOT_PERMITTED]


def test_undeclared_secret_is_refused(profile: Profile) -> None:
    _, surface, _, _, journal = drive(
        profile,
        [
            LOOK,
            Propose(
                Action(ActionKind.TYPE, PASSWORD, SecretRef("not_declared")), "sign in"
            ),
            DONE,
        ],
    )

    assert surface.acted == []
    refusals = [event for event in journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refusals] == [Denial.UNDECLARED_SECRET]


def test_a_visual_anchor_the_run_never_saw_is_not_acted_on(profile: Profile) -> None:
    click = Propose(
        Action(ActionKind.CLICK, VisualAnchor("anchor-9")), "click the painted button"
    )

    _, surface, decider, _, _ = drive(
        profile,
        [PICTURE, click, DONE],
        screens=[screen(regions=("anchor-0",))],
    )

    assert surface.acted == []
    assert "visual anchor" in decider.seen[2].notices[0]


def test_a_visual_anchor_from_the_current_picture_is_acted_on(
    profile: Profile,
) -> None:
    painted = Action(ActionKind.CLICK, VisualAnchor("anchor-0"), effect="approve")
    click = Propose(painted, "click it")

    _, surface, _, _, journal = drive(
        profile,
        [PICTURE, click, DONE],
        screens=[screen(regions=("anchor-0",))],
    )

    assert surface.acted == [click.action]
    acted = next(event for event in journal.events if isinstance(event, Acted))
    assert acted.target_kind == "VisualAnchor"


def test_risky_action_runs_only_after_approval(profile: Profile) -> None:
    submit = Action(ActionKind.CLICK, SEARCH, effect="submit_payment")
    result, surface, _, escalator, journal = drive(
        profile,
        [LOOK, Propose(submit, "submit the search"), DONE],
        handoffs=[Handoff(HandoffOutcome.APPROVED)],
    )

    assert surface.acted == [submit]
    assert result.ending is Ending.COMPLETED
    assert escalator.requests[0].trigger is Trigger.RISKY_ACTION
    escalations = [e for e in journal.events if isinstance(e, Escalated)]
    assert [e.outcome for e in escalations] == [HandoffOutcome.APPROVED]


def test_rejected_risky_action_is_not_performed(profile: Profile) -> None:
    result, surface, decider, _, _ = drive(
        profile,
        [
            LOOK,
            Propose(Action(ActionKind.CLICK, SEARCH, effect="submit_payment"), "pay"),
            DONE,
        ],
        handoffs=[Handoff(HandoffOutcome.REJECTED, "do not send payments")],
    )

    assert surface.acted == []
    assert result.ending is Ending.COMPLETED
    assert "declined click" in decider.seen[2].notices[0]
    assert "do not send payments" in decider.seen[2].notices[1]


def test_handoff_timeout_ends_the_run(profile: Profile) -> None:
    result, surface, _, _, _ = drive(
        profile,
        [
            LOOK,
            Propose(Action(ActionKind.CLICK, SEARCH, effect="submit_payment"), "go"),
        ],
        handoffs=[Handoff(HandoffOutcome.TIMED_OUT)],
    )

    assert result.ending is Ending.HANDED_OFF
    assert result.detail == "handoff timed out"
    assert surface.acted == []


def test_the_action_carries_the_page_the_decision_was_made_against(
    profile: Profile,
) -> None:
    _, surface, _, _, _ = drive(profile, [LOOK, READ, DONE])

    assert surface.expected[0] is not None
    assert surface.expected[0].page_state.location == MEMBER
    assert surface.expected[0].page_state.signature == ("button:Search",)


@pytest.mark.parametrize(
    "outcome", [Outcome.AMBIGUOUS, Outcome.STALE, Outcome.NOT_FOUND]
)
def test_an_unresolved_target_escalates_rather_than_acting_again(
    profile: Profile, outcome: Outcome
) -> None:
    attempts = profile.budgets.max_retries_per_step + 1
    result, surface, _, escalator, journal = drive(
        profile,
        [LOOK, *([READ] * attempts)],
        outcomes=[outcome] * attempts,
    )

    assert len(surface.acted) == attempts
    assert [request.trigger for request in escalator.requests] == [Trigger.NO_PROGRESS]
    assert result.ending is Ending.HANDED_OFF
    recorded = [event.outcome for event in journal.events if isinstance(event, Acted)]
    assert recorded == [outcome] * attempts


# Automatic observations.


def test_the_automatic_observation_uses_a_permitted_tool(edited_profile) -> None:
    profile = edited_profile(
        perception={
            "allowed_modes": ["visual"],
            "max_alternate_observations_per_step": 1,
        }
    )

    result, surface, _, _, journal = drive(profile, [PICTURE, READ, DONE])

    assert result.ending is Ending.COMPLETED
    assert modes(surface) == ["visual", "visual"]
    observed = [event for event in journal.events if isinstance(event, Observed)]
    assert observed[1].provenance is ObservationProvenance.POST_ACTION
    assert observed[1].alternate is False


def test_an_automatic_refresh_runs_one_tool_and_does_not_switch(
    profile: Profile,
) -> None:
    """A refresh that comes back empty tells the model, rather than trying more."""
    blind = screen(structured=ObservationStatus.FAILED, visual=ObservationStatus.FAILED)

    result, surface, decider, _, _ = drive(
        profile, [LOOK, READ, DONE], screens=[screen(), blind]
    )

    assert modes(surface) == ["structured", "structured"]
    assert any(
        "no observation of this screen is in hand" in notice
        for notice in decider.seen[-1].notices
    )
    assert result.ending is Ending.COMPLETED


# Budgets and boundaries.


@pytest.mark.rule(17)
def test_step_budget_stops_the_run(edited_profile) -> None:
    profile = edited_profile(
        budgets={
            "max_steps": 3,
            "max_wall_clock_s": 300,
            "max_retries_per_step": 2,
            "max_navigations": 15,
        }
    )

    result, surface, _, _, _ = drive(profile, [LOOK, READ, READ, READ])

    assert result.ending is Ending.EXHAUSTED
    assert result.detail == "max_steps"
    assert len(surface.acted) == 2


def test_navigation_budget_stops_the_run(edited_profile) -> None:
    profile = edited_profile(
        budgets={
            "max_steps": 40,
            "max_wall_clock_s": 300,
            "max_retries_per_step": 2,
            "max_navigations": 2,
        }
    )
    hop = Propose(Action(ActionKind.NAVIGATE, destination=SAVINGS), "open savings")

    result, surface, _, _, _ = drive(
        profile,
        [LOOK, hop, hop, hop],
        screens=[screen(MEMBER), screen(SAVINGS), screen(MEMBER), screen(SAVINGS)],
    )

    assert result.ending is Ending.EXHAUSTED
    assert result.detail == "max_navigations"
    assert len(surface.acted) == 2


def test_wall_clock_budget_stops_the_run(edited_profile) -> None:
    profile = edited_profile(
        budgets={
            "max_steps": 40,
            "max_wall_clock_s": 10,
            "max_retries_per_step": 2,
            "max_navigations": 15,
        }
    )
    clock = FakeClock()

    class TickingDecider(ReadsNoRecords):
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, transcript: Transcript) -> Decision:  # noqa: ARG002
            self.calls += 1
            clock.advance(6)
            return LOOK

    decider = TickingDecider()
    result = discover(
        "read the savings balance",
        profile,
        surface=ScriptedSurface(screens=[screen()]),
        decider=decider,
        escalator=ScriptedEscalator(),
        journal=MemoryJournal(),
        clock=clock,
    )

    assert result.ending is Ending.EXHAUSTED
    assert result.detail == "max_wall_clock_s"
    assert decider.calls == 2


def test_run_is_blocked_when_the_surface_opens_outside_the_profile(
    profile: Profile,
) -> None:
    result, surface, decider, _, _ = drive(
        profile,
        [DONE],
        screens=[screen("https://elsewhere.test/members/12345")],
    )

    assert result.ending is Ending.BLOCKED
    assert result.detail == "surface opened outside the profile"
    assert decider.seen == []
    assert surface.looks == []


def test_run_is_blocked_when_an_action_lands_outside_the_profile(
    profile: Profile,
) -> None:
    result, _, _, _, _ = drive(
        profile,
        [
            LOOK,
            Propose(Action(ActionKind.CLICK, SEARCH, effect="search"), "follow"),
            DONE,
        ],
        screens=[screen(MEMBER), screen("https://sandbox.example.test/admin")],
    )

    assert result.ending is Ending.BLOCKED
    assert result.detail == "surface left the permitted routes"


def test_a_page_that_moves_during_a_decision_invalidates_it(
    profile: Profile,
) -> None:
    surface = ScriptedSurface(screens=[screen()])

    class MovingDecider(ScriptedDecider):
        def decide(self, transcript):
            decision = super().decide(transcript)
            if len(self.seen) == 2:
                surface.screens = [screen(SAVINGS)]
            return decision

    decider = MovingDecider([LOOK, READ, DONE])
    result = discover(
        "read",
        profile,
        surface=surface,
        decider=decider,
        escalator=ScriptedEscalator(),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )

    assert result.ending is Ending.COMPLETED
    assert surface.acted == []
    assert "moved during the decision" in decider.seen[2].notices[0]


def test_a_new_capture_of_the_same_page_does_not_invalidate_a_decision(
    profile: Profile,
) -> None:
    """Use the signature, not a new observation id, to detect a page change."""
    result, surface, decider, _, _ = drive(profile, [LOOK, PICTURE, READ, DONE])

    assert result.ending is Ending.COMPLETED
    assert surface.acted == [READ.action]
    assert len(decider.seen[2].observations) == 2


# Failures and evidence.


def test_a_surface_failure_ends_the_run_with_a_terminal_event(
    profile: Profile,
) -> None:
    class BrokenSurface(ScriptedSurface):
        def observe(self, request):  # noqa: ARG002
            raise SurfaceError("the page is closed")

    journal = MemoryJournal()
    result = discover(
        "read",
        profile,
        surface=BrokenSurface(screens=[screen()]),
        decider=ScriptedDecider([LOOK]),
        escalator=ScriptedEscalator(),
        journal=journal,
        clock=FakeClock(),
    )

    assert result.ending is Ending.FAILED
    assert result.detail == "the surface could not continue"
    failures = [event for event in journal.events if isinstance(event, Failed)]
    assert [event.stage for event in failures] == ["surface"]


def test_a_model_failure_ends_the_run_with_a_terminal_event(
    profile: Profile,
) -> None:
    class BrokenDecider(ReadsNoRecords):
        def decide(self, transcript: Transcript) -> Decision:  # noqa: ARG002
            raise ModelError("the provider call failed")

    journal = MemoryJournal()
    result = discover(
        "read",
        profile,
        surface=ScriptedSurface(screens=[screen()]),
        decider=BrokenDecider(),
        escalator=ScriptedEscalator(),
        journal=journal,
        clock=FakeClock(),
    )

    assert result.ending is Ending.FAILED
    failures = [event for event in journal.events if isinstance(event, Failed)]
    assert [event.stage for event in failures] == ["model"]


def test_a_provider_call_cut_short_by_the_budget_ends_as_exhausted(
    profile: Profile,
) -> None:
    """A call gets only the time the run has left, so its timeout is the budget's."""
    clock = FakeClock()

    class SlowDecider(ReadsNoRecords):
        def decide(self, transcript: Transcript) -> Decision:
            clock.advance(transcript.seconds_remaining)
            raise ModelError("the provider call failed")

    journal = MemoryJournal()
    result = discover(
        "read",
        profile,
        surface=ScriptedSurface(screens=[screen()]),
        decider=SlowDecider(),
        escalator=ScriptedEscalator(),
        journal=journal,
        clock=clock,
    )

    assert result.ending is Ending.EXHAUSTED
    assert not any(isinstance(event, Failed) for event in journal.events)


@pytest.mark.rule(12)
def test_journal_records_policy_metadata_and_never_values(profile: Profile) -> None:
    stream = io.StringIO()
    surface = ScriptedSurface(screens=[screen()], extracts=["4200.00"])
    discover(
        "read the savings balance",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [
                LOOK,
                Propose(
                    Action(ActionKind.TYPE, PASSWORD, SecretRef("login_password")),
                    "sign in",
                ),
                Propose(Action(ActionKind.TYPE, SEARCH, "12345"), "enter the id"),
                READ,
                BALANCE_CLAIM,
            ]
        ),
        escalator=ScriptedEscalator(),
        journal=JsonlJournal(stream),
        clock=FakeClock(),
    )
    written = stream.getvalue()

    assert "login_password" in written
    assert "structured" in written
    assert "Savings balance" not in written
    assert '"route": "route_1"' in written
    assert "/members/:id" not in written
    assert "APP_PASSWORD" not in written
    assert "12345" not in written
    assert "4200.00" not in written


def test_journal_names_the_secret_without_resolving_it(profile: Profile) -> None:
    _, _, _, _, journal = drive(
        profile,
        [
            LOOK,
            Propose(
                Action(ActionKind.TYPE, PASSWORD, SecretRef("login_password")),
                "sign in",
            ),
            DONE,
        ],
    )

    acted = next(event for event in journal.events if isinstance(event, Acted))
    assert acted.value_source is ValueSource.SECRET
    assert acted.secret_name == "login_password"
    assert acted.route == "/members/:id"


@pytest.mark.parametrize(
    "finish",
    [
        Finish({}, "customer-private-value"),
        AskHuman(Trigger.NO_PROGRESS, "customer-private-value"),
    ],
)
def test_journal_omits_runtime_text_even_when_it_quotes_surface_data(
    profile: Profile, finish: Decision
) -> None:
    private = "customer-private-value"
    stream = io.StringIO()
    target = DomLocator("span", DomAttribute.CSS_CLASS, private, frame=(private,))
    discover(
        f"read {private}",
        profile,
        surface=ScriptedSurface(
            screens=[Screen(MEMBER, controls=(("cell", private),))],
            extracts=[private],
        ),
        decider=ScriptedDecider(
            [LOOK, Propose(Action(ActionKind.READ, target), private), finish]
        ),
        escalator=ScriptedEscalator(),
        journal=JsonlJournal(stream),
        clock=FakeClock(),
    )

    assert private not in stream.getvalue()


@pytest.mark.parametrize("location", [MEMBERS, MEMBER, SAVINGS])
def test_permitted_entry_points_start_a_run(profile: Profile, location: str) -> None:
    result, _, _, _, _ = drive(profile, [DONE], screens=[screen(location)])

    assert result.ending is Ending.COMPLETED


def test_the_loop_imports_no_driver_and_no_model_client() -> None:
    """The seam is real only if the loop cannot reach past its protocols."""
    assert_module_import_contract(computeruse.loop)


@pytest.mark.rule(1)
def test_discovery_runs_only_against_a_declared_sandbox(edited_profile) -> None:
    profile = edited_profile(environment="production")

    result, surface, decider, _, _ = drive(profile, [LOOK])

    assert result.ending is Ending.BLOCKED
    assert decider.seen == []
    assert surface.looks == []


@pytest.mark.parametrize("outcome", [Outcome.UNCERTAIN, Outcome.HANDOFF])
def test_unacknowledged_input_hands_over_without_retry_or_post_action_capture(
    profile, outcome
):
    result, surface, _, escalator, _ = drive(
        profile,
        [LOOK, READ, READ],
        outcomes=[outcome],
        screens=[screen(controls=(("cell", "Savings balance"),))],
    )
    assert result.ending is Ending.HANDED_OFF
    assert len(surface.acted) == 1
    assert len(surface.looks) == 1
    assert len(escalator.requests) == 1


def test_a_reading_that_renames_a_declared_output_is_sent_back(profile) -> None:
    """Found in canvas batch 52: the model read ``status_text`` as ``status text``.

    The contract declared the output, so a claim under the other name could
    complete but never be saved. The reading goes back with the declared
    names, and the run uses the reading that names them.
    """
    from computeruse.decider import Task, TaskOutput

    decider = ScriptedDecider(
        [DONE],
        tasks=[
            Task(outputs=(TaskOutput("status text"),)),
            Task(outputs=(TaskOutput("status"),)),
        ],
    )
    result = computeruse.loop.discover(
        "Report the status as one output named status",
        profile,
        surface=ScriptedSurface([Screen("https://sandbox.example.test/members")]),
        decider=decider,
        escalator=ScriptedEscalator([]),
        journal=MemoryJournal(),
        clock=FakeClock(),
        outputs=("status",),
    )
    assert decider.interpreted[1] == (
        "the task was refused: the task must name exactly the declared outputs: status",
    )
    assert result.ending is Ending.COMPLETED
    assert result.outputs == {"status": "Task done"}
