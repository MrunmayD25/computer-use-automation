"""Discovery exports reusable data and visual actions through production replay."""

import dataclasses
import time

import pytest
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
    Outcome,
    Point,
    ScreenTarget,
)
from computeruse.capability import ValueType, dumps
from computeruse.contract import Contract
from computeruse.decider import (
    CheckKind,
    FactRef,
    Finish,
    Observe,
    Propose,
    Remember,
    ResultCheck,
)
from computeruse.escalation import HandoffOutcome
from computeruse.journal import MemoryJournal
from computeruse.loop import CheckResult, Ending, RunResult, Verification, discover
from computeruse.profile import ActionKind
from computeruse.recording import DiscoveryTrace, field
from computeruse.replay import MemoryReplayLog, Status, replay
from computeruse.visual import Box, cut_out

STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)
VISUAL = ObservationRequest(ObservationMode.VISUAL)


def execute(capability, surface, profile, inputs=None):
    return replay(
        capability,
        inputs or {},
        profile=profile,
        surface=surface,
        control=People([HandoffOutcome.TIMED_OUT]).control(time.monotonic),
        log=MemoryReplayLog(),
        clock=time.monotonic,
        sleep=surface.idle,
        accept_draft=True,
    )


@pytest.mark.parametrize("input_is_label", [False, True])
def test_native_selection_replays_with_another_id_and_display_name(
    pages, input_is_label
):
    markup = """<h1>Context</h1><label>Operator<select>
    <option value="OP1">First Person (OP1)</option>
    <option value="OP2">Second Person (OP2)</option>
    <option value="OP3">Third Person (OP3)</option>
    </select></label>"""
    target = AxLocator("combobox", "Operator")
    operator = "Second Person (OP2)" if input_is_label else "OP2"
    replacement = "Third Person (OP3)" if input_is_label else "OP3"
    with pages({"/": markup}) as (surface, profile):
        trace = DiscoveryTrace()
        result = discover(
            "Choose operator OP2 and read the selection",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(STRUCTURED),
                    Propose(
                        Action(ActionKind.SELECT, target, "OP2"),
                        input_name="" if input_is_label else "operator",
                    ),
                    Finish(
                        {"selection": "Second Person (OP2)"},
                        checks=(
                            ResultCheck(
                                CheckKind.RESULT,
                                target,
                                "Second Person (OP2)",
                                output="selection",
                            ),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
            inputs={"operator": operator},
        )
        assert result.ending is Ending.COMPLETED, result
        record = trace.build(
            result,
            profile=profile,
            inputs={"operator": operator},
            contract=Contract(2, (field("operator"),), (field("selection"),)),
            capability_id="choose_operator",
            run="scripted",
            safe_text=frozenset({"Operator", "Context"}),
        )
        assert record.complete, (record.issues, record.artifact_issues)
        assert record.capability is not None
        assert "Second Person" not in dumps(record.capability)
        surface._page.reload()
        replayed = execute(
            record.capability, surface, profile, {"operator": replacement}
        )
        assert replayed.status is Status.SUCCEEDED, replayed
        assert replayed.outputs == {"selection": "Third Person (OP3)"}


def test_cross_screen_fact_is_read_again_for_each_replay(pages):
    written = {
        "/": '<h1>Source</h1><span id="number">17290</span><a href="/form">Next</a>',
        "/form": "<h1>Destination</h1><label>Number<input></label>",
    }
    source = DomLocator("span", DomAttribute.ID, "number")
    destination = AxLocator("textbox", "Number")
    with pages(written) as (surface, profile):
        trace = DiscoveryTrace()
        result = discover(
            "Copy the displayed number into the next form and report the number",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(STRUCTURED),
                    Remember("number", "17290", source=source, may_change=False),
                    Propose(Action(ActionKind.CLICK, AxLocator("link", "Next"))),
                    Propose(
                        Action(ActionKind.TYPE, destination, "17290"), fact="number"
                    ),
                    Finish(
                        {"number": "17290"},
                        checks=(
                            ResultCheck(
                                CheckKind.RESULT,
                                FactRef("number"),
                                "17290",
                                output="number",
                            ),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
        )
        assert result.ending is Ending.COMPLETED, result
        contract = Contract(
            1, (), (dataclasses.replace(field("number"), type=ValueType.INTEGER),)
        )
        record = trace.build(
            result,
            profile=profile,
            inputs={},
            capability_id="copy_number",
            run="scripted",
            safe_text=frozenset({"Source", "number", "Next", "Destination", "Number"}),
            contract=contract,
        )
        assert record.complete, (record.issues, record.artifact_issues)
        assert record.capability is not None
        assert "17290" not in dumps(record.capability)
        surface._page.goto(profile.scope.base_url)
        surface._page.locator("#number").evaluate("el => el.textContent = '81935'")
        replayed = execute(record.capability, surface, profile)
        assert replayed.status is Status.SUCCEEDED, replayed
        assert replayed.outputs == {"number": "81935"}
        assert surface._page.locator("input").input_value() == "81935"


def test_equal_discovery_inputs_keep_their_distinct_names(pages):
    first = AxLocator("textbox", "First")
    second = AxLocator("textbox", "Second")
    with pages(
        {"/": "<h1>Form</h1><label>First<input></label><label>Second<input></label>"}
    ) as (surface, profile):
        trace = DiscoveryTrace()
        result = discover(
            "Fill both fields and report them",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(STRUCTURED),
                    Propose(Action(ActionKind.TYPE, first, "same"), input_name="first"),
                    Propose(
                        Action(ActionKind.TYPE, second, "same"), input_name="second"
                    ),
                    Finish(
                        {"first_value": "same", "second_value": "same"},
                        checks=(
                            ResultCheck(
                                CheckKind.RESULT, first, "same", output="first_value"
                            ),
                            ResultCheck(
                                CheckKind.RESULT, second, "same", output="second_value"
                            ),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
            inputs={"first": "same", "second": "same"},
        )
        assert result.ending is Ending.COMPLETED, result
        contract = Contract(
            1,
            (field("first"), field("second")),
            (field("first_value"), field("second_value")),
        )
        record = trace.build(
            result,
            profile=profile,
            inputs={"first": "same", "second": "same"},
            capability_id="two_inputs",
            run="scripted",
            safe_text=frozenset({"Form", "First", "Second"}),
            contract=contract,
        )
        assert record.complete, (record.issues, record.artifact_issues)
        surface._page.reload()
        replayed = execute(
            record.capability, surface, profile, {"first": "alpha", "second": "beta"}
        )
        assert replayed.status is Status.SUCCEEDED, replayed
        assert replayed.outputs == {"first_value": "alpha", "second_value": "beta"}


def test_two_facts_refreshed_in_one_decision_keep_separate_outputs(pages):
    first = DomLocator("span", DomAttribute.ID, "first")
    second = DomLocator("span", DomAttribute.ID, "second")
    written = {
        "/": '<h1>Facts</h1><span id="first">17290</span><span id="second">81935</span>'
    }
    with pages(written) as (surface, profile):
        trace = DiscoveryTrace()
        result = discover(
            "Report both displayed numbers",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(STRUCTURED),
                    Remember("first", "17290", source=first),
                    Remember("second", "81935", source=second),
                    Finish(
                        {"first_number": "17290", "second_number": "81935"},
                        checks=(
                            ResultCheck(
                                CheckKind.RESULT,
                                FactRef("first"),
                                "17290",
                                output="first_number",
                            ),
                            ResultCheck(
                                CheckKind.RESULT,
                                FactRef("second"),
                                "81935",
                                output="second_number",
                            ),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
        )
        assert result.ending is Ending.COMPLETED, result
        from computeruse.recording import ActionTaken

        refreshed = [
            entry
            for entry in trace.entries
            if isinstance(entry, ActionTaken) and entry.step == result.steps
        ]
        assert {entry.action.target for entry in refreshed} == {first, second}
        record = trace.build(
            result,
            profile=profile,
            inputs={},
            contract=Contract(2, (), (field("first_number"), field("second_number"))),
            capability_id="two_facts",
            run="scripted",
            safe_text=frozenset({"Facts", "first", "second"}),
        )
        assert record.complete, (record.issues, record.artifact_issues)
        assert record.capability is not None
        saved = dumps(record.capability)
        assert "17290" not in saved
        assert "81935" not in saved
        surface._page.locator("#first").evaluate("el => el.textContent = '30406'")
        surface._page.locator("#second").evaluate("el => el.textContent = '50907'")
        replayed = execute(record.capability, surface, profile)
        assert replayed.status is Status.SUCCEEDED, replayed
        assert replayed.outputs == {"first_number": "30406", "second_number": "50907"}


def test_recorded_peer_navigation_keeps_query_parameters_and_policy(pages):
    with (
        serve_pages(
            {
                "/next/12345": '<h1><span id="title">Done</span></h1>',
                "/next/67890": '<h1><span id="title">Done</span></h1>',
            }
        ) as peer,
        pages(
            {"/": "<h1>Home</h1>"},
            version=5,
            origins=[
                {"origin": peer, "allow_routes": ["/next/:id"], "deny_routes": []}
            ],
        ) as (surface, profile),
    ):
        trace = DiscoveryTrace()
        result = discover(
            "Open the permitted next page for 12345 and read its title",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(STRUCTURED),
                    Propose(
                        Action(
                            ActionKind.NAVIGATE,
                            destination=peer + "/next/12345?member=12345#section",
                        )
                    ),
                    Finish(
                        {"title": "Done"},
                        checks=(
                            ResultCheck(
                                CheckKind.RESULT,
                                DomLocator("span", DomAttribute.ID, "title"),
                                "Done",
                                output="title",
                            ),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
            trace=trace,
        )
        assert result.ending is Ending.COMPLETED, result
        record = trace.build(
            result,
            profile=profile,
            inputs={"member_id": "12345"},
            contract=Contract(2, (field("member_id"),), (field("title"),)),
            capability_id="peer_navigation",
            run="scripted",
            safe_text=frozenset({"Home", "title", "member", "section"}),
        )
        assert record.complete, (record.issues, record.artifact_issues)
        assert record.capability is not None
        surface._page.goto(profile.scope.base_url)
        replayed = execute(record.capability, surface, profile, {"member_id": "67890"})
        assert replayed.status is Status.SUCCEEDED, replayed
        assert surface.location() == peer + "/next/67890?member=67890#section"


@pytest.mark.parametrize("duplicate", [False, True])
def test_permitted_visual_click_finds_moved_control_and_refuses_twins(pages, duplicate):
    markup = """<h1><span id="state">Visual form</span></h1><p role="status"></p>
    <button style="position:absolute;left:40px;top:100px;
    width:120px;height:50px;background:#209070;color:white;border:3px solid black"
    onclick="document.querySelector('#state').textContent='Finished';
    document.querySelector('p').textContent='Choice accepted'">Continue</button>"""
    with pages({"/": markup}) as (surface, profile):
        trace = DiscoveryTrace()
        before = surface.observe(STRUCTURED)
        trace.look(before)
        picture = surface.observe(VISUAL)
        trace.look(picture)
        crop = cut_out(picture.image, Box(40, 100, 120, 50))
        assert crop is not None
        action = Action(
            ActionKind.CLICK, ScreenTarget(picture.observation_id, Point(100, 125))
        )
        acted = surface.act(action)
        assert acted.outcome is Outcome.OK
        trace.action(1, action, acted, before.location, before)
        after = surface.observe(STRUCTURED)
        trace.look(after)
        target = DomLocator("span", DomAttribute.ID, "state")
        check = ResultCheck(CheckKind.RESULT, target, "Finished", output="state")
        trace.verified(
            1,
            (
                ResultCheck(
                    CheckKind.STATE,
                    AxLocator("status", "Choice accepted"),
                    "Choice accepted",
                ),
            ),
            after,
        )
        result_read = surface.act(Action(ActionKind.READ, target))
        trace.action(
            2, Action(ActionKind.READ, target), result_read, before.location, after
        )
        result = RunResult(
            Ending.COMPLETED,
            2,
            "completed",
            outputs={"state": "Finished"},
            verification=Verification.EXECUTOR,
            checks=(CheckResult(check, "Finished", True),),
        )
        record = trace.build(
            result,
            profile=profile,
            inputs={},
            contract=Contract(2, (), (field("state"),)),
            capability_id="visual_continue",
            run="scripted",
            safe_text=frozenset({"Visual form", "state", "Choice accepted"}),
            visual_templates={1: crop},
        )
        assert record.complete, (record.issues, record.artifact_issues)
        assert record.capability is not None
        assert len(record.capability.templates) == 1
        serialized = dumps(record.capability)
        assert "capture_id" not in serialized
        surface._page.reload()
        surface._page.locator("button").evaluate("el => el.style.left = '260px'")
        if duplicate:
            surface._page.locator("button").evaluate("""el => {
                const copy = el.cloneNode(true); copy.style.left = '480px';
                document.body.append(copy);
            }""")
        replayed = execute(record.capability, surface, profile)
        if duplicate:
            assert replayed.status is not Status.SUCCEEDED
            assert surface._page.locator("h1").text_content() == "Visual form"
        else:
            assert replayed.status is Status.SUCCEEDED, replayed
            assert surface._page.locator("h1").text_content() == "Finished"
