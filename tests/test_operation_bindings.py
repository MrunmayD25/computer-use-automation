"""Operator declarations identify business operations without granting permission."""

import dataclasses

import pytest
import yaml
from replay_fakes import PROFILE

from computeruse.actions import Action, AxLocator, AxNode, Operation
from computeruse.escalation import HandoffOutcome
from computeruse.policy import covers
from computeruse.profile import ActionKind, ProfileError, load_profile


def binding(name, operation, target, *, kind="click", mode="structured", limit="risky"):
    return {
        "name": name,
        "operation": operation,
        "route": "/desk",
        "frame": [],
        "mode": mode,
        "context": ["heading:Review" if mode == "structured" else "Review"],
        "kinds": [kind],
        "target": target,
        "role": "button" if mode == "structured" else "canvas",
        "key": "Enter" if kind == "press_key" else "",
        "submission": "any",
        "limit": limit,
    }


@pytest.fixture
def profile(tmp_path):
    doc = yaml.safe_load(PROFILE)
    doc.update(
        version=7,
        origins=[],
        submissions="risky",
        operations=[
            binding("commit", "post_order", "Post"),
            binding("alternate", "post_order", "Confirm"),
            binding("enter", "post_order", "Post", kind="press_key"),
            binding("back", "edit_order", "Back", limit="by_effect"),
        ],
    )
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_profile(path)


def resolved(profile, label, *, kind=ActionKind.CLICK, context=("heading:Review",)):
    from computeruse.operations import identify

    action = Action(
        kind,
        AxLocator("button", label),
        "Enter" if kind is ActionKind.PRESS_KEY else None,
    )
    node = AxNode("button", label, tag="button", control="d1:c4", context=context)
    held = identify(profile, action, "/desk", node=node)
    assert held is not None
    return dataclasses.replace(
        Operation.of(action, "/desk", node=node),
        binding=held.name,
        business=held.business,
    )


@pytest.mark.rule(1, 5, 6)
def test_alternate_controls_and_enter_share_the_declared_business_operation(profile):
    post = resolved(profile, "Post")
    alternate = resolved(profile, "Confirm")
    enter = resolved(profile, "Post", kind=ActionKind.PRESS_KEY)
    assert covers(post, alternate)
    assert covers(post, enter)
    from computeruse.loop import _change_key

    assert _change_key(post) == _change_key(alternate) == _change_key(enter)


@pytest.mark.rule(6)
def test_a_different_declared_operation_does_not_inherit_a_missing_control(profile):
    post = resolved(profile, "Post")
    back = resolved(profile, "Back")
    assert not covers(post, back, lost=frozenset({post.control}))
    unknown = Operation(ActionKind.CLICK, "/desk", "screen")
    assert covers(post, unknown)
    assert covers(unknown, back)


@pytest.mark.rule(1, 6)
def test_a_binding_cannot_be_claimed_by_the_model_or_reused_in_another_context(profile):
    from computeruse.operations import identify

    action = Action(ActionKind.CLICK, AxLocator("button", "Post"), effect="post_order")
    node = AxNode("button", "Post", tag="button", context=("heading:Edit",))
    assert identify(profile, action, "/desk", node=node) is None
    assert identify(profile, action, "/desk", node=None) is None


@pytest.mark.rule(1)
@pytest.mark.parametrize(
    "change",
    [
        {"limit": "safe"},
        {"kinds": ["read"]},
        {"context": []},
        {"route": "/forbidden"},
        {"role": ""},
        {"target": ""},
        {"submission": "sometimes"},
    ],
)
def test_invalid_operation_declarations_are_rejected(tmp_path, change):
    doc = yaml.safe_load(PROFILE)
    item = binding("commit", "post_order", "Post") | change
    doc.update(version=7, origins=[], submissions="risky", operations=[item])
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ProfileError):
        load_profile(path)


@pytest.mark.rule(1, 6)
def test_an_unknown_target_inherits_the_declared_limit(profile):
    from computeruse import operations
    from computeruse.profile import Limit

    unknown = Operation(ActionKind.CLICK, "/desk", "screen")
    assert operations.limit(profile, unknown) is Limit.RISKY
    assert operations.limit(profile, resolved(profile, "Back")) is None
    denied = dataclasses.replace(
        profile,
        operations=tuple(
            dataclasses.replace(item, limit=Limit.DENY)
            if item.name == "commit"
            else item
            for item in profile.operations
        ),
    )
    assert operations.limit(denied, unknown) is Limit.DENY
    assert operations.limit(denied, resolved(denied, "Confirm")) is Limit.DENY


def browser_profile(*, painted=False):
    rule = binding(
        "commit", "post_order", "Post", mode="visual" if painted else "structured"
    )
    rule["route"] = "/**"
    return {
        "version": 7,
        "origins": [],
        "submissions": "by_effect",
        "operations": [rule],
    }


def expectation(surface, action, seen, node=None):
    from computeruse.actions import Expectation

    held = surface.operation_binding(action, seen, node)
    assert held is not None
    return Expectation(seen.page_state, binding=held.name, business=held.business)


@pytest.mark.rule(6)
@pytest.mark.parametrize("change", [False, True])
def test_browser_checks_operation_context_at_dispatch(pages, change):
    from computeruse.actions import ObservationMode, ObservationRequest, Outcome

    markup = """<h1>Review</h1><button onclick="window.commits++">Post</button>
      <script>window.commits=0</script>"""
    with pages({"/": markup}, **browser_profile()) as (surface, _):
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        action = Action(ActionKind.CLICK, AxLocator("button", "Post"))
        expect = expectation(surface, action, seen, node)
        if change:
            surface._page.locator("h1").evaluate("el => el.textContent = 'Edit'")
        result = surface.act(action, expect=expect)
        assert surface._page.evaluate("window.commits") == (0 if change else 1), result
        assert result.outcome is (Outcome.STALE if change else Outcome.OK)


@pytest.mark.rule(6)
def test_a_binding_changed_during_trusted_input_never_receives_the_click(pages):
    from computeruse.actions import ObservationMode, ObservationRequest, Outcome

    markup = """<h1>Review</h1><button
      onpointerdown="document.querySelector('h1').textContent='Edit'"
      onclick="window.commits++">Post</button><script>window.commits=0</script>"""
    with pages({"/": markup}, **browser_profile()) as (surface, _):
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        action = Action(ActionKind.CLICK, AxLocator("button", "Post"))
        result = surface.act(action, expect=expectation(surface, action, seen, node))
        assert result.outcome is Outcome.UNCERTAIN
        assert surface._page.evaluate("window.commits") == 0


@pytest.mark.rule(2, 6)
@pytest.mark.parametrize("nested", [False, True])
def test_shadow_context_is_shared_by_observation_resolution_and_dispatch(pages, nested):
    from computeruse.actions import ObservationMode, ObservationRequest, Outcome

    markup = """<h1>Workspace</h1><div id="host"></div><script>
      window.commits=0;
      const root=host.attachShadow({mode:'open'});
      root.innerHTML='<h2>Review</h2><div id="inner"></div>';
      const inside=root.getElementById('inner');
      const target=NESTED ? inside.attachShadow({mode:'open'}) : inside;
      target.innerHTML='<button onclick="window.commits++">Post</button>';
      window.changeContext=()=>root.querySelector('h2').textContent='Edit';
      </script>""".replace("NESTED", "true" if nested else "false")
    with pages({"/": markup}, **browser_profile()) as (surface, _):
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        assert "heading:Review" in node.context
        action = Action(ActionKind.CLICK, AxLocator("button", "Post"))
        expect = expectation(surface, action, seen, node)
        assert surface.act(action, expect=expect).outcome is Outcome.OK
        assert surface._page.evaluate("window.commits") == 1
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        node = next(item for item in seen.nodes if item.name == "Post")
        expect = expectation(surface, action, seen, node)
        surface._page.evaluate("window.changeContext()")
        assert surface.act(action, expect=expect).outcome is Outcome.STALE
        assert surface._page.evaluate("window.commits") == 1


@pytest.mark.rule(1, 5, 6)
def test_enter_in_a_nested_picker_is_distinct_from_enter_that_submits(pages):
    from computeruse.actions import ObservationMode, ObservationRequest, Outcome

    markup = """<h1>Review</h1><form onsubmit="event.preventDefault();window.commits++">
      <input aria-label="Memo"><button>Post</button><div id="picker"></div>
      </form><script>window.commits=0;
      picker.attachShadow({mode:'open'}).innerHTML='<input aria-label="Query">';
      </script>"""
    config = browser_profile()
    for name, operation, submission in (
        ("submit_key", "post_order", "native"),
        ("picker_key", "search", "none"),
    ):
        item = binding(name, operation, "*", kind="press_key")
        item.update(route="/**", role="textbox", submission=submission)
        config["operations"].append(item)
    with pages({"/": markup}, **config) as (surface, _):
        seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        keys = {}
        for name, expected in (("Memo", "submit_key"), ("Query", "picker_key")):
            node = next(item for item in seen.nodes if item.name == name)
            action = Action(ActionKind.PRESS_KEY, AxLocator("textbox", name), "Enter")
            expect = expectation(surface, action, seen, node)
            assert expect.binding == expected
            keys[name] = (action, expect)
        action, expect = keys["Query"]
        assert surface.act(action, expect=expect).outcome is Outcome.OK
        assert surface._page.evaluate("window.commits") == 0
        action, expect = keys["Memo"]
        assert surface.act(action, expect=expect).outcome is Outcome.OK
        assert surface._page.evaluate("window.commits") == 1


@pytest.mark.rule(6, 10)
@pytest.mark.parametrize("change", [False, True])
def test_painted_binding_freezes_pixels_until_the_click_arrives(pages, change):
    from computeruse.actions import (
        ObservationMode,
        ObservationRequest,
        Outcome,
        Point,
        ScreenTarget,
    )
    from computeruse.reading import Line

    class Reader:
        def lines(self, image):
            assert image
            return (Line("Review", 30, 10, 90, 20, 1), Line("Post", 30, 60, 90, 20, 1))

    markup = """<style>body {margin:0}</style>
      <canvas id="drawing" width="300" height="200" tabindex="0"></canvas>
      <script>window.commits=0;
      const drawing=document.getElementById('drawing'), paint=drawing.getContext('2d');
      paint.fillStyle='white';paint.fillRect(0,0,300,200);paint.fillStyle='black';
      paint.fillText('Review',30,20);paint.fillText('Post',30,70);
      drawing.onclick=()=>window.commits++;
      </script>"""
    with pages({"/": markup}, **browser_profile(painted=True)) as (surface, _):
        errors = []
        surface._page.on("pageerror", lambda error: errors.append(str(error)))
        surface._reader = Reader()
        seen = surface.observe(ObservationRequest(ObservationMode.VISUAL))
        action = Action(
            ActionKind.CLICK, ScreenTarget(seen.observation_id, Point(50, 70))
        )
        expect = expectation(surface, action, seen)
        if change:
            surface._page.evaluate(
                "() => { drawing.onpointerdown=()=>paint.fillRect(200,150,20,20); }"
            )
        result = surface.act(action, expect=expect)
        assert not errors
        assert surface._page.evaluate("window.commits") == (0 if change else 1), result
        assert result.outcome is (Outcome.UNCERTAIN if change else Outcome.OK)


@pytest.mark.rule(1, 5, 6, 14)
@pytest.mark.parametrize("known_alias", [False, True])
def test_discovery_can_prepare_after_a_write_but_cannot_send_its_alias(
    pages, known_alias
):
    import time

    from fakes import ScriptedDecider, ScriptedEscalator

    from computeruse.actions import ObservationMode, ObservationRequest
    from computeruse.decider import Observe, Propose
    from computeruse.escalation import Handoff, HandoffOutcome, Trigger
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover

    markup = """<h1>Review</h1>
      <button onclick="window.commits++">Post</button>
      <button onclick="window.commits++">Confirm</button>
      <button onclick="window.backs++">Back</button>
      <script>window.commits=0;window.backs=0</script>"""
    config = browser_profile()
    config["operations"] += [
        binding("back", "edit_order", "Back", limit="by_effect") | {"route": "/**"},
    ]
    if known_alias:
        config["operations"].append(
            binding("alias", "post_order", "Confirm") | {"route": "/**"}
        )
    person = ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)])
    with pages({"/": markup}, **config) as (surface, profile):
        result = discover(
            "Post the order",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    *(
                        Propose(
                            Action(
                                ActionKind.CLICK,
                                AxLocator("button", label),
                                effect="navigate",
                            )
                        )
                        for label in ("Post", "Confirm", "Back")
                    ),
                ]
            ),
            escalator=person,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert surface._page.evaluate("window.commits") == 1
        assert surface._page.evaluate("window.backs") == 1
        assert all(item.operation.business for item in result.restrictions)
    assert (
        sum(request.trigger is Trigger.RISKY_ACTION for request in person.requests) == 1
    )


def recorded_operations(profile, labels=("Back", "Post")):
    from replay_fakes import limits

    from computeruse.actions import Outcome
    from computeruse.capability import LocatorForm
    from computeruse.policy import Restriction, Source
    from computeruse.profile import Limit
    from computeruse.recorder import (
        CheckKind,
        CheckSample,
        Executed,
        Finished,
        Learned,
        Recorder,
        Synthetic,
        TargetSample,
        Verified,
        Verifier,
    )

    recorder = Recorder(
        capability_id="declared_write",
        version=1,
        profile=profile,
        origin=Synthetic(),
        inputs=(),
        outputs=(),
        variables=(),
        outcomes=(),
        limits=limits(),
        samples={},
    )
    shown = TargetSample(
        "/desk", LocatorForm.ACCESSIBILITY, role="status", name="Posted"
    )
    check = CheckSample(CheckKind.PRESENT, shown)
    for step, label in enumerate(labels, 1):
        recorder.record(
            Executed(
                step,
                ActionKind.CLICK,
                "/desk",
                Outcome.OK,
                target=TargetSample(
                    "/desk", LocatorForm.ACCESSIBILITY, role="button", name=label
                ),
                operation=resolved(profile, label),
                effect="navigate",
            )
        )
        recorder.record(Verified(step, (check,)))
    recorder.record(
        Learned(
            Restriction(
                resolved(profile, "Post"), "navigate", Limit.RISKY, Source.PROPOSAL, 2
            )
        )
    )
    recorder.record(Finished(Verifier.EXECUTOR, (check,)))
    return recorder.finish()


@pytest.mark.rule(3, 6)
def test_export_keeps_the_declared_operation_scope(profile):
    from computeruse.capability import ActionNode, Approval, RestrictionScope

    recorded = recorded_operations(profile)
    assert recorded.complete, (recorded.issues, recorded.artifact_issues)
    assert recorded.capability.restrictions[0].scope is RestrictionScope.OPERATION
    nodes = [node for node in recorded.capability.nodes if isinstance(node, ActionNode)]
    assert [node.approval for node in nodes] == [Approval.NONE, Approval.EACH_RUN]
    assert all(node.binding and node.business for node in nodes)


@pytest.mark.rule(1, 6)
def test_changing_any_alias_requires_review_and_recording_again(profile):
    from computeruse.capability import IssueCode, check_profile

    recorded = recorded_operations(profile)
    assert recorded.complete
    changed = dataclasses.replace(
        profile,
        operations=tuple(
            dataclasses.replace(item, operation="another_change")
            if item.name == "alternate"
            else item
            for item in profile.operations
        ),
    )
    assert IssueCode.PROFILE_MISMATCH in {
        issue.code for issue in check_profile(recorded.capability, changed)
    }


@pytest.mark.rule(1, 6)
@pytest.mark.parametrize("changes", [{"route": "/members"}, {"kind": ActionKind.READ}])
def test_a_saved_binding_cannot_move_to_another_route_or_action(profile, changes):
    from computeruse.capability import ActionNode, IssueCode, check_profile

    capability = recorded_operations(profile).capability
    assert capability is not None
    node = next(item for item in capability.nodes if isinstance(item, ActionNode))
    capability = dataclasses.replace(
        capability,
        nodes=tuple(
            dataclasses.replace(item, **changes) if item is node else item
            for item in capability.nodes
        ),
    )
    assert IssueCode.PROFILE_MISMATCH in {
        issue.code for issue in check_profile(capability, profile)
    }


@pytest.mark.rule(1, 6)
def test_overlapping_bindings_remain_unidentified_and_keep_the_route_limit(profile):
    from computeruse import operations
    from computeruse.profile import Limit

    changed = dataclasses.replace(
        profile,
        operations=(
            *profile.operations,
            dataclasses.replace(profile.operations[0], name="overlap"),
        ),
    )
    action = Action(ActionKind.CLICK, AxLocator("button", "Post"))
    node = AxNode("button", "Post", context=("heading:Review",))
    assert operations.identify(changed, action, "/desk", node=node) is None
    assert (
        operations.limit(changed, Operation.of(action, "/desk", node=node))
        is Limit.RISKY
    )


@pytest.mark.rule(5, 6, 12)
@pytest.mark.parametrize("known_alias", [False, True])
def test_replay_sends_alternate_controls_for_one_business_change_only_once(
    profile, known_alias
):
    from replay_fakes import App, Clock, Page, People

    from computeruse import operations
    from computeruse.capability import ActionNode, Review, dumps, loads
    from computeruse.replay import MemoryReplayLog, Status, replay

    class BusinessApp(App):
        writes = 0

        def operation_binding(self, action, observation, node=None):
            assert observation is not None
            return operations.identify(profile, action, "/desk", node=node)

        def _apply(self, action, found):
            assert found is not None
            if action.kind is ActionKind.CLICK:
                self.writes += 1
                self.state = "after"

    nodes = tuple(
        AxNode("button", name, tag="button", context=("heading:Review",))
        for name in ("Post", "Confirm")
    )
    app = BusinessApp(
        {
            "before": Page("/desk", nodes),
            "after": Page("/desk", (*nodes, AxNode("status", "Posted"))),
        },
        "before",
        {},
    )
    recorded = recorded_operations(profile, ("Post", "Confirm"))
    assert recorded.complete, (recorded.issues, recorded.artifact_issues)
    capability = loads(dumps(recorded.capability))
    capability = dataclasses.replace(
        capability,
        provenance=dataclasses.replace(capability.provenance, review=Review.REVIEWED),
    )
    if not known_alias:
        action_nodes = [
            node for node in capability.nodes if isinstance(node, ActionNode)
        ]
        alias = action_nodes[1]
        capability = dataclasses.replace(
            capability,
            nodes=tuple(
                dataclasses.replace(node, binding="", business="")
                if node is alias
                else node
                for node in capability.nodes
            ),
        )
    before = dumps(capability)
    clock = Clock()
    person = People([HandoffOutcome.APPROVED])
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
    assert result.status is Status.SUCCEEDED, result
    assert app.writes == 1
    assert len(person.requests) == 1
    assert dumps(capability) == before
