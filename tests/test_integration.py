"""Real browser checks across discovery, recording, replay and live control."""

import time
from pathlib import Path

import pytest
import yaml
from fakes import ScriptedDecider, ScriptedEscalator
from pages import serve_pages
from replay_fakes import People

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    ObservationMode,
    ObservationRequest,
)
from computeruse.browser import open_session
from computeruse.capability import dumps, loads
from computeruse.contract import Contract
from computeruse.control import Control
from computeruse.decider import CheckKind, Finish, Observe, Propose, ResultCheck
from computeruse.escalation import Handoff, HandoffOutcome, Mode
from computeruse.journal import MemoryJournal
from computeruse.loop import Ending, discover
from computeruse.profile import ActionKind, load_profile
from computeruse.recording import DiscoveryTrace, field
from computeruse.replay import MemoryReplayLog, Status, replay

pytest_plugins = ("test_takeover",)

LOOKUP = Contract(2, (field("member_id"),), (field("standing"),))

SEARCH = """<!doctype html><h1>Member search</h1>
<form onsubmit="event.preventDefault();
location.href='/members/'+document.getElementById('member').value">
<label>Member number <input id="member"></label><button>Search</button></form>"""
DETAIL = """<!doctype html><h1>Member details</h1><section>
<dl><dt>Member number</dt><dd id="record">{member}</dd>
<dt>Standing</dt><dd id="standing">{standing}</dd></dl></section>"""


@pytest.fixture
def application(tmp_path):
    with serve_pages(
        {
            "/members": SEARCH,
            "/members/10001": DETAIL.format(member="10001", standing="Active"),
            "/members/10002": DETAIL.format(member="10002", standing="Suspended"),
        }
    ) as base:
        document = yaml.safe_load(Path("evaluation/profile.yaml").read_text())
        document["base_url"] = base
        document["actions"]["click"] = {"any": "risky"}
        path = tmp_path / "profile.yaml"
        path.write_text(yaml.safe_dump(document))
        yield base, load_profile(path)


def discovered(base, profile, failed_postcondition=False):
    trace = DiscoveryTrace()
    identity = DomLocator("dd", DomAttribute.ID, "record")
    standing = DomLocator("dd", DomAttribute.ID, "standing")
    script = ScriptedDecider(
        [
            Observe(ObservationRequest(ObservationMode.STRUCTURED)),
            Propose(
                Action(ActionKind.TYPE, AxLocator("textbox", "Member number"), "10001")
            ),
            Propose(
                Action(ActionKind.CLICK, AxLocator("button", "Search")),
                after=(
                    ResultCheck(
                        CheckKind.STATE, AxLocator("status", "Missing"), "Ready"
                    ),
                )
                if failed_postcondition
                else (),
            ),
            Finish(
                {"standing": "Active"},
                checks=(
                    ResultCheck(CheckKind.RECORD, identity, "10001"),
                    ResultCheck(
                        CheckKind.RESULT, standing, "Active", output="standing"
                    ),
                ),
            ),
        ]
    )
    with open_session(profile, base + "/members", record=True) as surface:
        result = discover(
            "Read standing for member 10001",
            profile,
            surface=surface,
            decider=script,
            journal=MemoryJournal(),
            clock=time.monotonic,
            escalator=ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)]),
            trace=trace,
        )
    assert result.ending is Ending.COMPLETED
    recording = trace.build(
        result,
        profile=profile,
        inputs={"member_id": "10001"},
        contract=LOOKUP,
        capability_id="member_standing",
        run="integration-discovery",
        safe_text=frozenset(
            {
                "Member search",
                "Member details",
                "member",
                "Search",
                "button|",
                "record",
                "standing",
            }
        ),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    return recording.capability


@pytest.mark.parametrize("failed_postcondition", [False, True])
def test_real_discovery_artifact_replays_for_another_member_with_fresh_approval(
    application,
    failed_postcondition,
):
    base, profile = application
    capability = discovered(base, profile, failed_postcondition)
    serialized = dumps(capability)
    assert "10001" not in serialized
    assert "Active" not in serialized
    saved = loads(serialized)
    for member, expected in [("10002", "Suspended"), ("10001", "Active")]:
        people = People([HandoffOutcome.APPROVED])
        with open_session(profile, base + "/members", record=True) as surface:
            result = replay(
                saved,
                {"member_id": member},
                profile=profile,
                surface=surface,
                control=people.control(time.monotonic),
                log=MemoryReplayLog(),
                clock=time.monotonic,
                sleep=surface.idle,
                accept_draft=True,
            )
            assert result.status is Status.SUCCEEDED, result
            assert result.outputs == {"standing": expected}
            assert surface.location().endswith("/members/" + member)
        assert len(people.requests) == 1
    assert dumps(saved) == serialized


def human_control(surface, mode, member):
    from test_takeover import Sitting, Watching, acting, command

    from computeruse.escalation import Command, State

    def perform(page):
        page.goto(surface.location().split("/members")[0] + "/members/" + member)

    sitting = Sitting(
        surface,
        [
            command(State.PAUSED, Command.TAKE_CONTROL),
            acting(State.HUMAN_CONTROL, perform),
            command(State.HUMAN_CONTROL, Command.RESUME),
            command(State.PAUSED, Command.TERMINATE),
        ],
    )
    control = Control(
        mode=mode,
        clock=time.monotonic,
        seat=sitting,
        channels=[Watching()],
        worker=False,
    )
    sitting.control = control
    return control


def manual_discovery(base, profile):
    from computeruse.decider import AskHuman
    from computeruse.escalation import Trigger

    trace = DiscoveryTrace()
    script = ScriptedDecider(
        [
            Observe(ObservationRequest(ObservationMode.STRUCTURED)),
            AskHuman(Trigger.NO_PROGRESS, "Open the requested member"),
            Finish(
                {"standing": "Active"},
                checks=(
                    ResultCheck(
                        CheckKind.RECORD,
                        DomLocator("dd", DomAttribute.ID, "record"),
                        "10001",
                    ),
                    ResultCheck(
                        CheckKind.RESULT,
                        DomLocator("dd", DomAttribute.ID, "standing"),
                        "Active",
                        output="standing",
                    ),
                ),
            ),
        ]
    )
    journal = MemoryJournal()
    with open_session(profile, base + "/members", record=True) as surface:
        control = human_control(surface, Mode.DISCOVERY, "10001")
        result = discover(
            "Read standing of member 10001",
            profile,
            surface=surface,
            decider=script,
            control=control,
            journal=journal,
            clock=time.monotonic,
            trace=trace,
        )
    assert result.ending is Ending.COMPLETED, result
    assert result.segments
    assert result.segments[0].steps
    recorded = trace.build(
        result,
        profile=profile,
        inputs={"member_id": "10001"},
        contract=LOOKUP,
        capability_id="manual_lookup",
        run=control.run,
        safe_text=frozenset(
            {
                "Member search",
                "Member details",
                "member",
                "Search",
                "button|",
                "record",
                "standing",
            }
        ),
    )
    assert recorded.complete, (recorded.issues, recorded.artifact_issues)
    return recorded.capability


@pytest.mark.parametrize("left_at", ["10002", "10001"])
def test_recorded_takeover_replays_and_refuses_the_wrong_resume_state(
    application, left_at
):
    from computeruse.capability import HumanNode
    from computeruse.journal import ManualAction

    base, profile = application
    capability = manual_discovery(base, profile)
    assert any(isinstance(node, HumanNode) for node in capability.nodes)
    before = dumps(capability)
    journal = MemoryJournal()
    with open_session(profile, base + "/members", record=True) as surface:
        control = human_control(surface, Mode.REPLAY, left_at)
        result = replay(
            capability,
            {"member_id": "10002"},
            profile=profile,
            surface=surface,
            control=control,
            journal=journal,
            log=MemoryReplayLog(),
            clock=time.monotonic,
            sleep=surface.idle,
            accept_draft=True,
        )
    if left_at == "10002":
        assert result.status is Status.SUCCEEDED, result
        assert result.outputs == {"standing": "Suspended"}
    else:
        assert result.status is Status.TERMINATED, result
        assert not result.outputs
    assert any(isinstance(event, ManualAction) for event in journal.events)
    assert dumps(capability) == before


def test_repeated_dialog_messages_never_invent_an_answer(recorded):
    from test_takeover import bounded, hand_over

    from computeruse.manual import Detail, ManualKind

    with bounded(30), recorded({"/": "<h1>Dialogs</h1>"}) as (surface, _):
        hand_over(surface)
        for answer in ("dismiss", "accept"):
            surface._page.evaluate("setTimeout(() => confirm('Same message'), 0)")
            surface.idle(0.1)
            getattr(surface._pending["page-1"].dialog, answer)()
            surface.idle(0.1)
        recording = surface.hand_back()
    answers = [event for event in recording.events if event.kind is ManualKind.DIALOG]
    assert len(answers) == 2
    assert all(event.detail is Detail.NONE for event in answers)
    assert recording.gaps or all(event.dialog for event in answers)


def test_popup_dialog_answered_in_browser_resumes_discovery(recorded):
    from test_takeover import Sitting, Watching, acting, bounded, command

    from computeruse.escalation import Command, State

    written = {
        "/": "<button onclick=\"window.open('/popup')\">Open</button><h1>Ready</h1>",
        "/popup": "<script>confirm('Continue?')</script><h1>Popup ready</h1>",
    }
    with bounded(30), recorded(written, allow_new_windows=True) as (surface, profile):

        def answer(page):
            del page
            surface._pending["page-2"].dialog.accept()
            surface.idle(0.1)

        sitting = Sitting(
            surface,
            [
                command(State.PAUSED, Command.TAKE_CONTROL),
                acting(State.HUMAN_CONTROL, answer),
                command(State.HUMAN_CONTROL, Command.RESUME),
            ],
        )
        control = Control(
            mode=Mode.DISCOVERY,
            clock=time.monotonic,
            seat=sitting,
            channels=[Watching()],
            worker=False,
        )
        sitting.control = control
        script = ScriptedDecider(
            [
                Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                Propose(Action(ActionKind.CLICK, AxLocator("button", "Open"))),
                Finish(
                    {"title": "Ready"},
                    checks=(
                        ResultCheck(
                            CheckKind.RESULT,
                            AxLocator("heading", "Ready"),
                            "Ready",
                            output="title",
                        ),
                    ),
                ),
            ]
        )
        result = discover(
            "Read the page",
            profile,
            surface=surface,
            decider=script,
            control=control,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert not surface._pending
        assert surface.observe(ObservationRequest(ObservationMode.STRUCTURED)).usable
    assert result.ending is Ending.COMPLETED, result
    assert len(result.segments) == 1


def test_dialog_interrupting_collector_is_bounded(recorded, monkeypatch):
    from test_takeover import bounded

    from computeruse import browser

    with bounded(20), recorded({"/": "<h1>Ready</h1>"}) as (surface, _):
        # A page getter may synchronously open a dialog while the collector runs.
        original = browser.SCRIPT
        monkeypatch.setattr(
            browser, "SCRIPT", "() => { confirm('During collection'); return true; }"
        )
        started = time.monotonic()
        observed = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        assert time.monotonic() - started < 10
        assert observed.dialog is not None or not observed.usable
        surface._pending["page-1"].dialog.dismiss()
        surface.idle(0.1)
        monkeypatch.setattr(browser, "SCRIPT", original)
        assert surface.observe(ObservationRequest(ObservationMode.STRUCTURED)).usable


def test_stop_after_replay_click_does_not_send_it_again(application, monkeypatch):
    from test_takeover import Sitting, Watching, bounded, command

    from computeruse.browser import BrowserSurface
    from computeruse.control import Order
    from computeruse.escalation import Command, State

    base, profile = application
    capability = discovered(base, profile)
    actual = BrowserSurface.act
    clicked = []
    with bounded(30), open_session(profile, base + "/members", record=True) as surface:
        sitting = Sitting(
            surface,
            [
                command(State.AWAITING_APPROVAL, Command.APPROVE),
                command(State.PAUSED, Command.RESUME),
            ],
        )
        control = Control(
            mode=Mode.REPLAY,
            clock=time.monotonic,
            seat=sitting,
            channels=[Watching()],
            worker=False,
        )
        sitting.control = control

        def stopped(adapter, action, *, expect=None):
            result = actual(adapter, action, expect=expect)
            if action.kind is ActionKind.CLICK:
                clicked.append(action)
                status = control.status()
                control.submit(Order(Command.STOP, status.run, status.revision))
            return result

        monkeypatch.setattr(BrowserSurface, "act", stopped)
        result = replay(
            capability,
            {"member_id": "10002"},
            profile=profile,
            surface=surface,
            control=control,
            journal=MemoryJournal(),
            log=MemoryReplayLog(),
            clock=time.monotonic,
            sleep=surface.idle,
            accept_draft=True,
        )
    assert result.status is Status.SUCCEEDED, result
    assert result.outputs == {"standing": "Suspended"}
    assert len(clicked) == 1
