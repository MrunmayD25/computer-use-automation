"""Recorded action checks reject success for another record on a live page."""

import json
import time

import pytest
from fakes import ScriptedDecider, ScriptedEscalator
from pages import serve_pages

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    ObservationMode,
    ObservationRequest,
)
from computeruse.browser import open_session
from computeruse.capability import dumps
from computeruse.control import Control
from computeruse.decider import CheckKind, Finish, Observe, Propose, ResultCheck
from computeruse.escalation import Mode
from computeruse.journal import MemoryJournal
from computeruse.loop import Ending, discover
from computeruse.profile import ActionKind
from computeruse.recording import DiscoveryTrace
from computeruse.replay import MemoryReplayLog, Status, replay


@pytest.mark.rule(3)
def test_recorded_postcondition_rejects_generic_success_for_another_record(
    site_profile,
    tmp_path,
):
    markup = """<h1>Posting</h1><label>Record<input></label>
    <button onclick="document.querySelector('h1').textContent = 'Posted';
    document.querySelector('#receipt').textContent = window.receiptOverride ||
    document.querySelector('input').value;
    window.posts = (window.posts || 0) + 1">Post</button>
    <span id="receipt"></span>"""
    field = AxLocator("textbox", "Record")
    posted = ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted")
    receipt = ResultCheck(
        CheckKind.STATE, DomLocator("span", DomAttribute.ID, "receipt"), "record-one"
    )
    with serve_pages({"/": markup}) as base:
        profile = site_profile(base_url=base, allow_routes=["/**"])
        with open_session(profile, base) as surface:
            trace = DiscoveryTrace()
            result = discover(
                "Post record-one",
                profile,
                surface=surface,
                decider=ScriptedDecider(
                    [
                        Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                        Propose(
                            Action(ActionKind.TYPE, field, "record-one"),
                            input_name="record",
                        ),
                        Propose(
                            Action(ActionKind.CLICK, AxLocator("button", "Post")),
                            after=(posted, receipt),
                        ),
                        Finish({}, checks=(posted,)),
                    ]
                ),
                escalator=ScriptedEscalator(),
                journal=MemoryJournal(),
                clock=time.monotonic,
                trace=trace,
                inputs={"record": "record-one"},
            )
            assert result.ending is Ending.COMPLETED, result
            assert surface._page.locator("#receipt").inner_text() == "record-one"
            assert surface._page.evaluate("window.posts") == 1
            recorded = trace.build(
                result,
                profile=profile,
                inputs={"record": "record-one"},
                capability_id="post_record",
                run="scripted",
                safe_text=frozenset({"Posting", "Record", "Post", "Posted", "receipt"}),
            )
            assert recorded.complete, (recorded.issues, recorded.artifact_issues)
            assert recorded.capability is not None
            original = dumps(recorded.capability)
            assert "record-one" not in original
            (tmp_path / "capability.json").write_text(original)
        for wrong_record in (True, False):
            with open_session(profile, profile.scope.base_url) as replay_surface:
                if wrong_record:
                    replay_surface._page.evaluate(
                        "window.receiptOverride = 'unrelated-record'"
                    )
                log = MemoryReplayLog()
                replayed = replay(
                    recorded.capability,
                    {"record": "record-two"},
                    profile=profile,
                    surface=replay_surface,
                    control=Control(
                        mode=Mode.REPLAY,
                        clock=time.monotonic,
                        seat=replay_surface,
                        escalator=ScriptedEscalator(),
                    ),
                    log=log,
                    clock=time.monotonic,
                    sleep=replay_surface.idle,
                    accept_draft=True,
                )
                assert replay_surface._page.locator("h1").inner_text() == "Posted"
                assert replay_surface._page.evaluate("window.posts") == 1
                assert replay_surface._page.locator("#receipt").inner_text() == (
                    "unrelated-record" if wrong_record else "record-two"
                )
                assert (replayed.status is Status.SUCCEEDED) is not wrong_record, (
                    replayed
                )
                assert "record-two" not in repr(log.events)
                assert "unrelated-record" not in repr(log.events)
                assert dumps(recorded.capability) == original
                case = "wrong-record" if wrong_record else "correct-record"
                replay_surface._page.screenshot(path=tmp_path / f"{case}.png")
                (tmp_path / f"{case}.json").write_text(
                    json.dumps(
                        {
                            "status": replayed.status.value,
                            "posts": replay_surface._page.evaluate("window.posts"),
                            "receipt": replay_surface._page.locator(
                                "#receipt"
                            ).inner_text(),
                            "journal": [type(event).__name__ for event in log.events],
                        },
                        indent=2,
                    )
                    + "\n"
                )
