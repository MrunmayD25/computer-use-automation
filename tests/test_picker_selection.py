"""A query in a picker never substitutes for the component's chosen state."""

import dataclasses

import pytest
from test_operation_bindings import browser_profile, expectation

from computeruse.actions import (
    Action,
    AxLocator,
    ObservationMode,
    ObservationRequest,
    Operation,
    Outcome,
)
from computeruse.escalation import HandoffOutcome
from computeruse.profile import ActionKind

CUSTOM = """<h1>Review</h1><form onsubmit="event.preventDefault();window.commits++">
<label>Choice<test-picker id="picker"></test-picker></label><button>Post</button></form>
<script>
window.commits=0; picker.ready=false;
picker.attachShadow({mode:'open'}).innerHTML=
  '<input role="combobox" aria-expanded="false">';
const input=picker.shadowRoot.querySelector('input');
input.value='Query';input.oninput=()=>{picker.ready=false;};
window.choose=()=>{picker.ready=true;input.value='Chosen item';};
</script>"""


def config():
    return browser_profile() | {
        "version": 8,
        "pickers": [
            {
                "operation": "post_order",
                "slot": "input|Choice",
                "source": "host",
                "property": "ready",
            }
        ],
    }


@pytest.mark.rule(1, 6)
@pytest.mark.parametrize("chosen", [False, True])
def test_only_a_chosen_picker_can_pass_the_dispatch_check(pages, chosen):
    with pages({"/": CUSTOM}, **config()) as (surface, profile):
        if chosen:
            surface._page.evaluate("window.choose()")
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        from computeruse import operations

        held = surface.operation_binding(
            Action(ActionKind.CLICK, AxLocator("button", "Post")), seen, node
        )
        operation = dataclasses.replace(
            Operation.of(
                Action(ActionKind.CLICK, AxLocator("button", "Post")), "/**", node=node
            ),
            binding=held.name,
            business=held.business,
        )
        assert operations.selections_ready(profile, operation, node) is chosen
        action = Action(ActionKind.CLICK, AxLocator("button", "Post"))
        result = surface.act(action, expect=expectation(surface, action, seen, node))
        assert result.outcome is (Outcome.OK if chosen else Outcome.NOT_ACTIONABLE)
        assert surface._page.evaluate("window.commits") == (1 if chosen else 0)


@pytest.mark.rule(6)
@pytest.mark.parametrize("during_input", [False, True])
def test_a_picker_deselected_after_authorization_never_commits(pages, during_input):
    with pages({"/": CUSTOM}, **config()) as (surface, _):
        surface._page.evaluate("window.choose()")
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        action = Action(ActionKind.CLICK, AxLocator("button", "Post"))
        expect = expectation(surface, action, seen, node)
        surface._page.evaluate(
            "() => {document.querySelector('button').onpointerdown="
            "()=>{picker.ready=false;};}"
            if during_input
            else "() => {picker.ready=false;}"
        )
        result = surface.act(action, expect=expect)
        assert result.outcome is (
            Outcome.UNCERTAIN if during_input else Outcome.NOT_ACTIONABLE
        )
        assert surface._page.evaluate("window.commits") == 0


@pytest.mark.rule(1, 6)
@pytest.mark.parametrize("chosen", [False, True])
def test_discovery_requires_a_selection_before_it_requests_commit_approval(
    pages, chosen
):
    import time

    from fakes import ScriptedDecider, ScriptedEscalator

    from computeruse.decider import Observe, Propose
    from computeruse.escalation import Handoff, HandoffOutcome, Trigger
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover

    person = ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)])
    with pages({"/": CUSTOM}, **config()) as (surface, profile):
        if chosen:
            surface._page.evaluate("window.choose()")
        discover(
            "Post the chosen item",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Propose(
                        Action(
                            ActionKind.CLICK,
                            AxLocator("button", "Post"),
                            effect="navigate",
                        )
                    ),
                ]
            ),
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert surface._page.evaluate("window.commits") == (1 if chosen else 0)
        assert sum(
            request.trigger is Trigger.RISKY_ACTION for request in person.requests
        ) == (1 if chosen else 0)


@pytest.mark.rule(1, 6, 12)
@pytest.mark.parametrize("chosen", [False, True])
def test_replay_requires_a_selection_before_commit_approval(tmp_path, chosen):
    import yaml
    from replay_fakes import PROFILE, App, Clock, Page, People
    from test_operation_bindings import binding, recorded_operations

    from computeruse import operations
    from computeruse.actions import AxNode
    from computeruse.capability import Review
    from computeruse.profile import load_profile
    from computeruse.replay import MemoryReplayLog, Reason, Status, replay

    doc = yaml.safe_load(PROFILE)
    doc.update(
        version=8,
        origins=[],
        submissions="by_effect",
        operations=[
            binding("commit", "post_order", "Post"),
            binding("back", "prepare", "Back", limit="by_effect"),
        ],
        pickers=config()["pickers"],
    )
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(doc))
    profile = load_profile(path)

    class SelectedApp(App):
        writes = 0

        def operation_binding(self, action, observation, node=None):
            assert observation is not None
            return operations.identify(profile, action, "/desk", node=node)

        def _apply(self, action, found):
            assert action.kind is ActionKind.CLICK
            if found.name == "Post":
                self.writes += 1
                self.state = "after"

    nodes = tuple(
        AxNode(
            "button",
            name,
            tag="button",
            context=("heading:Review",),
            selections=(("post_order", chosen),),
        )
        for name in ("Back", "Post")
    )
    app = SelectedApp(
        {
            "page": Page("/desk", nodes),
            "after": Page("/desk", (*nodes, AxNode("status", "Posted"))),
        },
        "page",
        {},
    )
    recorded = recorded_operations(profile, ("Post",))
    assert recorded.complete, recorded.artifact_issues
    capability = dataclasses.replace(
        recorded.capability,
        provenance=dataclasses.replace(
            recorded.capability.provenance, review=Review.REVIEWED
        ),
    )
    clock = Clock()
    person = People([HandoffOutcome.APPROVED] if chosen else [])
    result = replay(
        capability,
        {},
        profile=profile,
        surface=app,
        control=person.control(clock),
        log=MemoryReplayLog(),
        clock=clock,
        sleep=clock.advance,
    )
    assert app.writes == (1 if chosen else 0)
    if chosen:
        assert result.status is Status.SUCCEEDED, result
    else:
        assert result.status is not Status.SUCCEEDED
        assert result.history[0].reason is Reason.SELECTION_UNCONFIRMED


@pytest.mark.rule(1, 6)
@pytest.mark.parametrize("source", ["native", "aria"])
@pytest.mark.parametrize("chosen", [False, True])
def test_standard_selection_sources_do_not_treat_a_query_as_selection(
    pages, source, chosen
):
    field = (
        '<select aria-label="Choice"><option value="">Choose</option>'
        "<option>Item</option></select>"
        if source == "native"
        else '<input role="combobox" aria-label="Choice" aria-controls="choices" '
        'aria-expanded="false" value="Item">'
        '<div role="listbox" id="choices"><div role="option" '
        'aria-selected="false">Item</div></div>'
    )
    markup = "<h1>Review</h1><form>" + field + "<button>Post</button></form>"
    declared = config()
    declared["pickers"] = [
        dict(
            operation="post_order",
            source=source,
            property="",
            slot=("select" if source == "native" else "input") + "|Choice",
        )
    ]
    with pages({"/": markup}, **declared) as (surface, _):
        if chosen:
            surface._page.evaluate(
                "() => {document.querySelector('select').selectedIndex=1;}"
                if source == "native"
                else "() => {document.querySelector('[role=option]')"
                ".setAttribute('aria-selected','true');}"
            )
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        assert dict(node.selections)["post_order"] is chosen


@pytest.mark.rule(1, 6)
@pytest.mark.parametrize(
    "change",
    [
        "picker.ready='yes'",
        "delete picker.ready",
        "picker.remove()",
        "picker.shadowRoot.innerHTML += '<input role=combobox>'",
        "Object.defineProperty(picker,'ready',{get(){throw Error('unavailable')}})",
    ],
)
def test_missing_ambiguous_or_non_boolean_picker_state_never_counts(pages, change):
    with pages({"/": CUSTOM}, **config()) as (surface, _):
        surface._page.evaluate("window.choose()")
        surface._page.evaluate("() => {" + change + ";}")
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        assert dict(node.selections)["post_order"] is False


@pytest.mark.rule(1, 6)
def test_a_portalled_option_can_be_chosen_before_the_commit(pages):
    import time

    from fakes import ScriptedDecider, ScriptedEscalator
    from test_operation_bindings import binding

    from computeruse.decider import Observe, Propose
    from computeruse.escalation import Handoff, HandoffOutcome, Trigger
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover

    declared = config()
    declared["operations"].append(
        binding("choose", "choose_item", "*", limit="by_effect")
        | {
            "route": "/**",
            "role": "option",
            "context": ["popup:listbox"],
        }
    )
    markup = (
        CUSTOM + '<div role="listbox"><button role="option" onclick="choose()">'
        "Item</button></div>"
    )
    person = ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)])
    with pages({"/": markup}, **declared) as (surface, profile):
        discover(
            "Post the chosen item",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Propose(
                        Action(
                            ActionKind.CLICK,
                            AxLocator("option", "Item"),
                            effect="choose_item",
                        )
                    ),
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Propose(
                        Action(
                            ActionKind.CLICK,
                            AxLocator("button", "Post"),
                            effect="post_item",
                        )
                    ),
                ]
            ),
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert surface._page.evaluate("window.commits") == 1
        assert (
            sum(request.trigger is Trigger.RISKY_ACTION for request in person.requests)
            == 1
        )


@pytest.mark.rule(1, 6)
@pytest.mark.parametrize(
    "change",
    [
        {"source": "guess"},
        {"source": "host", "property": ""},
        {"source": "aria", "property": "ready"},
        {"operation": "undeclared"},
        {"slot": ""},
    ],
)
def test_invalid_selection_declarations_are_rejected(tmp_path, change):
    import yaml
    from replay_fakes import PROFILE

    from computeruse.profile import ProfileError, load_profile

    doc = yaml.safe_load(PROFILE)
    declared = config()
    declared["operations"][0]["route"] = "/desk"
    declared["pickers"][0].update(change)
    doc.update(declared)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ProfileError):
        load_profile(path)
