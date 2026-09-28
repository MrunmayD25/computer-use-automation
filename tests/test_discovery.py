"""Run the real discovery loop end to end against Chromium.

The decider is scripted because these tests do not cover model choices. The
browser, served pages, policy gate, journal, and clock are real. Each result
assertion is checked against the live page. A run fails these tests if it
reports success without changing the page.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable, Iterator

import pytest
from fakes import ReadsNoRecords, ScriptedEscalator, grants

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    ObservationMode,
    ObservationRequest,
    Outcome,
    Scope,
    ScopeKind,
    VisualAnchor,
)
from computeruse.browser import BrowserSurface, open_session
from computeruse.decider import (
    AskHuman,
    CheckKind,
    Decision,
    Finish,
    Match,
    Observe,
    Propose,
    ResultCheck,
    Transcript,
)
from computeruse.escalation import Handoff, HandoffOutcome, Trigger
from computeruse.journal import Acted, MemoryJournal, Observed, as_record
from computeruse.loop import Ending, RunResult, discover
from computeruse.profile import ActionKind, Profile

GOAL = "Open member 12345, read the savings balance, and post the approval"
BALANCE = DomLocator("td", DomAttribute.TEXT, "$4,212.55", frame=("ledger",))
APPROVAL_STATE = DomLocator("output", DomAttribute.ID, "approval-state")
OPEN_12345 = AxLocator("button", "Open", scope=Scope(ScopeKind.ROW, "12345"))
LOOK = ObservationRequest(ObservationMode.STRUCTURED)
SEE = ObservationRequest(ObservationMode.VISUAL)

type Move = Callable[[Transcript], Decision]


@dataclasses.dataclass
class Script(ReadsNoRecords):
    """A decider that replays moves, each free to read the transcript.

    A painted target names a region the adapter reported moments earlier, so a
    move is a function rather than a fixed decision. That is also the point:
    the anchor has to arrive through the transcript, or the move cannot build
    a target at all.
    """

    moves: list[Move]
    seen: list[Transcript] = dataclasses.field(default_factory=list)

    def decide(self, transcript: Transcript) -> Decision:
        self.seen.append(transcript)
        if not self.moves:
            return AskHuman(Trigger.NO_PROGRESS, "the script ran out")
        return self.moves.pop(0)(transcript)


def look(_: Transcript) -> Decision:
    return Observe(LOOK, "read the controls")


def see(_: Transcript) -> Decision:
    return Observe(SEE, "the approval pad is painted")


def do(action: Action) -> Move:
    return lambda _: Propose(action, "next step")


def finish(*names: str) -> Move:
    """Claim what earlier reads returned, checked against the controls read.

    The member the goal names is checked too, by the page heading, so the
    claim is only complete on that member's page.
    """

    def _finish(transcript: Transcript) -> Decision:
        read = [
            (turn.action.target, turn.extracted)
            for turn in transcript.history
            if turn.action is not None
            and isinstance(turn.action.target, (AxLocator, DomLocator))
            and turn.extracted
        ]
        found = {name: value for name in names for _, value in read}
        checks = [
            ResultCheck(CheckKind.RESULT, target, value, output=name)
            for name in names
            for target, value in read
        ]
        checks.append(
            ResultCheck(
                CheckKind.RECORD,
                AxLocator("heading", "Member 12345"),
                "12345",
                match=Match.CONTAINS,
            )
        )
        return Finish(found, "the goal is met", checks=tuple(checks))

    return _finish


def post_the_approval(transcript: Transcript) -> Decision:
    """Click the control painted into the canvas, by the region last reported."""
    regions = [
        region
        for observation in transcript.observations
        for region in observation.regions
    ]
    assert regions, "the visual tool reported no painted region"
    return Propose(Action(ActionKind.CLICK, VisualAnchor(regions[0].anchor_id)))


@pytest.fixture
def run(site: str, site_profile):
    """Return a helper that runs one scripted discovery against the pages."""

    @contextlib.contextmanager
    def _run(
        moves: list[Move],
        *,
        path: str = "/members",
        handoffs: list[Handoff] | None = None,
        **edits: object,
    ) -> Iterator[_Check]:
        profile: Profile = site_profile(**edits)
        decider = Script(list(moves))
        escalator = ScriptedEscalator(handoffs or [])
        journal = MemoryJournal()
        with open_session(profile, f"{site}{path}") as page:
            result = discover(
                GOAL,
                profile,
                surface=page,
                decider=decider,
                escalator=escalator,
                journal=journal,
                clock=time.monotonic,
            )
            yield _Check(result, decider, escalator, journal, page)

    return _run


@dataclasses.dataclass
class _Check:
    """The run's result, next to the live page it ran against."""

    result: RunResult
    decider: Script
    escalator: ScriptedEscalator
    journal: MemoryJournal
    page: BrowserSurface


def _last(page: BrowserSurface, target: DomLocator) -> str | None:
    """Read one value off the live page, after the run has stopped."""
    return page.act(Action(ActionKind.READ, target)).extracted


def test_a_scripted_run_reaches_the_goal_through_a_real_browser(run) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_12345)),
        do(Action(ActionKind.READ, BALANCE)),
        see,
        post_the_approval,
        finish("balance"),
    ]

    with run(moves) as check:
        result = check.result
        assert result.ending is Ending.COMPLETED
        assert result.outputs == {"balance": "$4,212.55"}
        assert check.page.location().endswith("/members/12345")
        assert _last(check.page, APPROVAL_STATE) == "posted"


def test_the_run_reads_the_frame_it_was_pointed_at(run) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_12345)),
        do(Action(ActionKind.READ, BALANCE)),
        finish("balance"),
    ]

    with run(moves) as check:
        assert check.result.outputs == {"balance": "$4,212.55"}


def test_a_click_into_a_denied_route_stops_the_run(run) -> None:
    wire = AxLocator("link", "Wire funds")
    moves: list[Move] = [look, do(Action(ActionKind.CLICK, wire))]

    with run(moves, path="/members/12345") as check:
        assert check.result.ending is not Ending.COMPLETED
        assert check.page.location().endswith("/members/12345")


def test_a_risky_action_reaches_a_person_before_it_runs(run) -> None:
    verify = AxLocator("button", "Verify")
    submit = Action(ActionKind.CLICK, verify, effect="verify_approver")
    moves: list[Move] = [look, do(submit)]
    refused = [Handoff(HandoffOutcome.REJECTED, "not for a synthetic passcode")]
    actions = grants({"observe": "safe", "read": "safe"})
    actions["click"] = {"any": "safe", "effects": {"verify_approver": "risky"}}

    with run(moves, path="/members/12345", handoffs=refused, actions=actions) as check:
        first = check.escalator.requests[0]
        assert first.trigger is Trigger.RISKY_ACTION
        assert first.action == submit


def test_an_action_the_profile_never_declared_is_refused(run) -> None:
    moves: list[Move] = [look, do(Action(ActionKind.CLICK, OPEN_12345))]
    allowed = {"observe": "safe", "read": "safe"}

    with run(moves, actions=grants(allowed)) as check:
        assert check.page.location().endswith("/members")
        assert any("policy refused" in notice for notice in _notices(check))


@pytest.mark.rule(19)
def test_an_observation_mode_the_profile_denies_is_refused(run) -> None:
    perception = {
        "allowed_modes": ["structured"],
        "max_alternate_observations_per_step": 1,
    }

    with run([see, see], perception=perception) as check:
        assert not [
            event
            for event in check.journal.events
            if isinstance(event, Observed) and event.mode is ObservationMode.VISUAL
        ]


@pytest.mark.rule(12)
def test_the_journal_carries_no_page_text_from_a_real_run(run) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_12345)),
        do(Action(ActionKind.READ, BALANCE)),
        finish("balance"),
    ]

    with run(moves) as check:
        written = repr([as_record(event) for event in check.journal.events])
        assert "4,212.55" not in written
        assert "Dale Okonkwo" not in written
        assert GOAL not in written
        assert "Open" not in written


def test_the_journal_records_the_target_kind_of_each_action(run) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_12345)),
        do(Action(ActionKind.READ, BALANCE)),
        finish("balance"),
    ]

    with run(moves) as check:
        kinds = [
            event.target_kind
            for event in check.journal.events
            if isinstance(event, Acted)
        ]
        assert kinds == ["AxLocator", "DomLocator"]


def test_an_unresolvable_target_does_not_move_the_page(run) -> None:
    ghost = AxLocator("button", "Approve everything")
    moves: list[Move] = [look, do(Action(ActionKind.CLICK, ghost))]

    with run(moves) as check:
        outcomes = [
            event.outcome for event in check.journal.events if isinstance(event, Acted)
        ]
        assert outcomes == [Outcome.NOT_FOUND]
        assert check.page.location().endswith("/members")


def _notices(check: _Check) -> list[str]:
    return [
        notice for transcript in check.decider.seen for notice in transcript.notices
    ]
