"""Progress, continuation, boundaries, and evidence across generic applications."""

import dataclasses
import json
import time

import pytest
from fakes import ScriptedDecider, ScriptedEscalator
from pages import serve_pages

from computeruse.actions import (
    Action,
    AxLocator,
    AxNode,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    PageState,
)
from computeruse.capability import HumanNode
from computeruse.decider import (
    AskHuman,
    CheckKind,
    FactRef,
    Finish,
    Observe,
    Propose,
    Remember,
    ResultCheck,
    Task,
    TaskRecord,
    TaskRequirement,
)
from computeruse.diagnostics import snapshot
from computeruse.escalation import Handoff, HandoffOutcome, Trigger
from computeruse.journal import MemoryJournal
from computeruse.loop import Ending, Verification, discover
from computeruse.profile import ActionKind
from computeruse.recording import DiscoveryTrace

STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)


@pytest.mark.parametrize("toggle", [False, True])
def test_successful_adapter_calls_cannot_hide_a_cycle(pages, toggle):
    change = (
        "document.querySelector('h1').textContent = window.flip ? 'A' : 'B'; "
        "window.flip = !window.flip"
        if toggle
        else ""
    )
    with pages({"/": f'<h1>A</h1><button onclick="{change}">Next</button>'}) as (
        surface,
        profile,
    ):
        person = ScriptedEscalator([])
        decider = ScriptedDecider(
            [Observe(STRUCTURED)]
            + [
                Propose(Action(ActionKind.CLICK, AxLocator("button", "Next")))
                for _ in range(18)
            ]
        )
        result = discover(
            "Reach a new page",
            profile,
            surface=surface,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert result.ending is Ending.HANDED_OFF
        assert person.requests[0].trigger is Trigger.NO_PROGRESS
        assert decider.decisions
        assert result.steps <= 10


def test_scrolling_that_brings_new_controls_into_view_is_progress(pages):
    rows = "".join(f"<p><button>Row {index}</button></p>" for index in range(80))
    with pages({"/": f"<h1>Rows</h1>{rows}"}) as (surface, profile):
        person = ScriptedEscalator([])
        decider = ScriptedDecider(
            [Observe(STRUCTURED)]
            + [Propose(Action(ActionKind.SCROLL, None, "down")) for _ in range(4)]
            + [AskHuman(Trigger.NO_PROGRESS, "stop here")]
        )
        discover(
            "Reach the last row",
            profile,
            surface=surface,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        # Only the decider's own request reaches a person, after all four
        # scrolls. The loop never judged a scroll that moved the view a cycle.
        assert len(person.requests) == 1
        assert person.requests[0].reason == "stop here"


DONE_PAGE = {
    "/": '<h1>Ready</h1><button onclick="'
    "document.querySelector('h1').textContent='Done'\">Complete</button>"
}


class Completes:
    """A person who finishes the step on the page and hands the session back."""

    def __init__(self, surface) -> None:
        self.surface = surface
        self.calls = 0

    def request(self, intervention):
        del intervention
        self.calls += 1
        if self.calls == 1:
            self.surface._page.locator("button").click()
        return Handoff(HandoffOutcome.RESUMED, changed=True)


def test_a_bare_repeat_request_is_checked_before_asking_again(pages):
    with pages(DONE_PAGE) as (surface, profile):
        person = Completes(surface)
        after = (ResultCheck(CheckKind.STATE, AxLocator("heading", "Done"), "Done"),)
        first = AskHuman(Trigger.NO_PROGRESS, "Complete the step", after=after)
        # The model asks again without naming what the person should do.
        bare = AskHuman(Trigger.NO_PROGRESS, "Complete the step")
        result = discover(
            "Complete the step",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [Observe(STRUCTURED), first, bare, Finish({}, checks=after)]
            ),
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert result.ending is Ending.COMPLETED, result
        assert person.calls == 1


def test_a_later_hand_back_does_not_reread_an_earlier_continuation(pages, monkeypatch):
    from computeruse.browser import BrowserSurface

    with pages(DONE_PAGE) as (surface, profile):
        person = Completes(surface)
        after = (ResultCheck(CheckKind.STATE, AxLocator("heading", "Done"), "Done"),)
        first = AskHuman(Trigger.NO_PROGRESS, "Complete the step", after=after)
        # Clicks on a control that is not there make the loop itself hand over.
        missing = Propose(Action(ActionKind.CLICK, AxLocator("button", "Missing")))
        decider = ScriptedDecider(
            [Observe(STRUCTURED), first, *[missing] * 6, Finish({}, checks=after)]
        )
        reads = []
        original = BrowserSurface.act

        def act(self, action, *, expect=None):
            if action.kind is ActionKind.READ:
                reads.append(action.target)
            return original(self, action, expect=expect)

        monkeypatch.setattr(BrowserSurface, "act", act)
        discover(
            "Complete the step",
            profile,
            surface=surface,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        # One read follows the first hand-back, and another checks the finish claim.
        # The loop's own later hand-backs read nothing of the first request.
        assert person.calls >= 2
        assert len(reads) == 2


def test_checked_human_continuation_prevents_duplicate_manual_requests(pages):
    with pages(
        {
            "/": '<h1>Ready</h1><button onclick="'
            "document.querySelector('h1').textContent='Done'\">Complete</button>"
        }
    ) as (surface, profile):

        class Person:
            calls = 0

            def request(self, intervention):
                del intervention
                self.calls += 1
                surface._page.locator("button").click()
                return Handoff(HandoffOutcome.RESUMED, changed=True)

        person = Person()
        after = (ResultCheck(CheckKind.STATE, AxLocator("heading", "Done"), "Done"),)
        handoff = AskHuman(Trigger.NO_PROGRESS, "Complete the step", after=after)
        trace = DiscoveryTrace()
        result = discover(
            "Complete the step",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [Observe(STRUCTURED), handoff, handoff, Finish({}, checks=after)]
            ),
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
        )
        assert result.ending is Ending.COMPLETED, result
        assert person.calls == 1
        recording = trace.build(
            result,
            profile=profile,
            inputs={},
            capability_id="manual",
            run="scripted",
            safe_text=frozenset({"Ready", "Done"}),
        )
        assert recording.complete, (recording.issues, recording.artifact_issues)
        assert recording.capability is not None
        assert (
            sum(isinstance(node, HumanNode) for node in recording.capability.nodes) == 1
        )


def test_declared_peer_frame_is_observed_but_other_origins_stay_blocked(pages):
    with serve_pages({"/": "<button>Peer action</button>"}) as peer:
        markup = f'<h1>Shell</h1><iframe name="workspace" src="{peer}"></iframe>'
        with pages(
            {"/": markup},
            version=5,
            origins=[
                {
                    "origin": peer,
                    "allow_routes": ["/"],
                    "deny_routes": [],
                }
            ],
        ) as (surface, _):
            observation = surface.observe(STRUCTURED)
            target = AxLocator("button", "Peer action", frame=("workspace",))
            assert any(node.name == "Peer action" for node in observation.nodes)
            assert surface.act(Action(ActionKind.CLICK, target)).outcome.value == "ok"


def test_failure_snapshot_keeps_structure_without_runtime_text():
    private = "private-value-123"
    observed = Observation(
        "capture",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState("https://private.example/"),
        nodes=(
            AxNode(private, private, value=private, attributes=(("id", private),)),
            AxNode("textbox", private, secret=True),
        ),
    )
    saved = snapshot(observed)
    assert saved is not None
    encoded = json.dumps(dataclasses.asdict(saved))
    assert private not in encoded
    assert "private.example" not in encoded
    assert saved.controls[0].role == "other"
    assert saved.controls[1].protected


def test_discovery_journal_never_overwrites_an_earlier_run(tmp_path):
    import contextlib

    from computeruse.cli import _journal
    from computeruse.evidence import EvidenceError

    path = tmp_path / "events.jsonl"
    path.write_text("earlier evidence\n")
    with (
        contextlib.ExitStack() as stack,
        pytest.raises(
            EvidenceError, match="evidence destination unavailable"
        ) as raised,
    ):
        _journal(stack, path)
    assert isinstance(raised.value.__cause__, FileExistsError)
    assert path.read_text() == "earlier evidence\n"


def test_screen_clicks_at_different_points_are_not_a_cycle():
    from pathlib import Path

    from fakes import DONE, FakeClock, ReadsNoRecords, Screen, ScriptedSurface

    from computeruse.actions import Point, ScreenTarget
    from computeruse.profile import load_profile

    visual = ObservationRequest(ObservationMode.VISUAL)

    class Picker(ReadsNoRecords):
        def __init__(self) -> None:
            self.picks = 0

        def decide(self, transcript):
            captures = [o for o in transcript.observations if o.image is not None]
            if not captures:
                return Observe(visual)
            if self.picks == 5:
                return DONE
            self.picks += 1
            # A painted list: each pick is a different control at a new place.
            point = Point(40 + 60 * self.picks, 40)
            target = ScreenTarget(captures[-1].observation_id, point)
            return Propose(Action(ActionKind.CLICK, target, effect="pick"))

    location = "https://sandbox.example.test/members/12345"
    surface = ScriptedSurface(
        [Screen(location, controls=(("button", "Search"), ("status", "Task done")))]
    )
    person = ScriptedEscalator([])
    result = discover(
        "Pick the entries",
        load_profile(Path("examples/profile.yaml")),
        surface=surface,
        decider=Picker(),
        escalator=person,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert [request.trigger for request in person.requests] == []
    assert result.ending is Ending.COMPLETED
    targets = [action.target for action in surface.acted]
    assert all(isinstance(target, ScreenTarget) for target in targets)
    assert (
        len({target.point for target in targets if isinstance(target, ScreenTarget)})
        == 5
    )


# A requirement shown on one page and proved on another.

_SIGNED_ON = (
    "<h1>Operator</h1><table><tr><th>Signed on</th><td>OP0002</td></tr></table>"
    '<a href="/search?q=NM999998">Member search</a>'
)
_NOT_FOUND = (
    "<h1>Members</h1>"
    '<input type="search" aria-label="Member number" value="NM999998">'
    '<p role="status">0 members found</p>'
)


def _prove_the_operator_elsewhere(pages, claim: Finish, task: Task):
    with pages({"/": _SIGNED_ON, "/search": _NOT_FOUND}) as (surface, profile):
        decider = ScriptedDecider(
            [
                Observe(STRUCTURED),
                Remember(
                    "operator",
                    "OP0002",
                    may_change=False,
                    source=AxLocator("cell", "OP0002"),
                ),
                Propose(Action(ActionKind.CLICK, AxLocator("link", "Member search"))),
                Observe(STRUCTURED),
                claim,
            ],
            tasks=[task],
        )
        return discover(
            "As operator OP0002, look up member NM999998.",
            profile,
            surface=surface,
            decider=decider,
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )


_OPERATOR = ResultCheck(
    CheckKind.REQUIREMENT, FactRef("operator"), "OP0002", requirement="operator"
)


def test_a_requirement_kept_on_one_page_proves_a_claim_on_another(pages):
    task = Task(requirements=(TaskRequirement("operator", "OP0002", context=True),))
    result = _prove_the_operator_elsewhere(pages, Finish({}, checks=(_OPERATOR,)), task)
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.verification is Verification.EXECUTOR


def test_a_not_found_outcome_can_cite_a_requirement_kept_elsewhere(pages):
    task = Task(
        records=(TaskRecord("member", "NM999998"),),
        requirements=(TaskRequirement("operator", "OP0002", context=True),),
    )
    claim = Finish(
        {},
        checks=(
            _OPERATOR,
            ResultCheck(
                CheckKind.STATE,
                AxLocator("status", "0 members found"),
                "0 members found",
            ),
            ResultCheck(
                CheckKind.RECORD, AxLocator("searchbox", "Member number"), "NM999998"
            ),
        ),
        outcome="record_not_found",
    )
    result = _prove_the_operator_elsewhere(pages, claim, task)
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.outcome == "record_not_found"


# A value inside a sentence, as legacy screens often show it.

_EVENT = (
    "<h1>Receipt</h1><table><tr><th>Summary</th></tr>"
    "<tr><td>Account AC0000161 opened for NM000054.</td></tr></table>"
)


def _kept(pages, *, may_change: bool) -> dict[str, str]:
    source = AxLocator("cell", "Account AC0000161 opened for NM000054.")
    with pages({"/": _EVENT}) as (surface, profile):
        decider = ScriptedDecider(
            [
                Observe(STRUCTURED),
                Remember("account", "AC0000161", may_change=may_change, source=source),
                Observe(STRUCTURED),
            ]
        )
        discover(
            "Report the new account's number.",
            profile,
            surface=surface,
            decider=decider,
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
    return {fact.key: fact.value for fact in decider.seen[-1].memory}


def test_an_unchanging_value_inside_a_sentence_can_be_kept(pages):
    assert _kept(pages, may_change=False) == {"account": "AC0000161"}


def test_a_changeable_value_must_be_the_whole_text_it_is_kept_from(pages):
    # The loop reads a changeable fact again before it is used, and compares
    # the whole text, so only an exact reading is kept.
    assert _kept(pages, may_change=True) == {}
