"""Painted choices need a specific review before the bound write can run."""

import dataclasses

import pytest
import yaml
from replay_fakes import PROFILE, App, Clock, Page, People, go, transfer_capability

from computeruse import operations
from computeruse.actions import ActionKind, AxNode, ScreenTarget
from computeruse.capability import (
    ActionNode,
    Application,
    Approval,
    Field,
    FocusTarget,
    HelpReason,
    HumanNode,
    IssueCode,
    Match,
    Present,
    RefKind,
    RestrictionScope,
    ResultKind,
    ResultNode,
    SavedRestriction,
    TextTarget,
    ValueType,
    constant,
    dumps,
    loads,
    ref,
    validate,
)
from computeruse.escalation import Ask, HandoffOutcome
from computeruse.policy import Source
from computeruse.profile import Limit, load_profile
from computeruse.reading import Line
from computeruse.replay import MemoryReplayLog, Status, replay
from computeruse.retarget import Scene


@pytest.fixture
def profile(tmp_path):
    doc = yaml.safe_load(PROFILE)
    doc.update(
        version=7,
        origins=[],
        submissions="by_effect",
        operations=[
            {
                "name": name,
                "operation": operation,
                "route": "/desk",
                "frame": [],
                "mode": "visual",
                "context": [context],
                "kinds": ["click"],
                "target": target,
                "role": "canvas",
                "key": "",
                "submission": "any",
                "limit": limit,
            }
            for name, operation, context, target, limit in (
                ("prepare", "edit_order", "Edit", "*", "by_effect"),
                ("commit", "post_order", "Review", "Post", "risky"),
            )
        ],
    )
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_profile(path)


def capability(profile):
    prepare = operations.declared(profile, "prepare")
    commit = operations.declared(profile, "commit")
    assert prepare is not None
    assert commit is not None
    targets = (
        TextTarget(
            "choice", "/desk", (), ref(RefKind.INPUT, "frequency"), Match.CONTAINS
        ),
        TextTarget("review", "/desk", (), constant("Review")),
        TextTarget("post", "/desk", (), constant("Post")),
        TextTarget("posted", "/desk", (), constant("Posted")),
    )
    choose = ActionNode(
        "choose",
        ActionKind.CLICK,
        "/desk",
        "choice",
        None,
        None,
        "choose_frequency",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (Present("choice"),),
        (),
        (go("review"),),
        binding=prepare.name,
        business=prepare.business,
    )
    review = dataclasses.replace(
        choose,
        node_id="review",
        target="review",
        effect="review_order",
        requires=(Present("review"),),
        verify=(Present("post"),),
        transitions=(go("post"),),
    )
    post = dataclasses.replace(
        choose,
        node_id="post",
        target="post",
        effect="post_order",
        requires=(Present("post"),),
        verify=(Present("posted"),),
        approval=Approval.EACH_RUN,
        mandatory=True,
        binding=commit.name,
        business=commit.business,
        transitions=(go("done"),),
    )
    base = transfer_capability()
    return dataclasses.replace(
        base,
        application=Application(
            base.application.profile_id,
            base.application.surface,
            base.application.origin,
            "/desk",
            (),
        ),
        inputs=(Field("frequency", ValueType.TEXT, True, 1, 40, ()),),
        outputs=(),
        variables=(),
        secrets=(),
        outcomes=(),
        targets=targets,
        entry="choose",
        nodes=(
            choose,
            review,
            post,
            ResultNode("done", ResultKind.SUCCESS, "", (Present("posted"),)),
        ),
        restrictions=(),
    )


def change_node(cap, name, **changes):
    return dataclasses.replace(
        cap,
        nodes=tuple(
            dataclasses.replace(node, **changes) if node.node_id == name else node
            for node in cap.nodes
        ),
    )


@pytest.mark.rule(3, 4, 16)
def test_a_bound_painted_choice_keeps_its_click_and_requires_commit_review(profile):
    cap = capability(profile)
    assert validate(cap) == ()
    restored = loads(dumps(cap))
    assert restored == cap
    choice = restored.node("choose")
    assert isinstance(choice, ActionNode)
    assert choice.verify == ()
    assert choice.transitions == (go("review"),)


@pytest.mark.rule(3, 4, 5, 6, 16)
@pytest.mark.parametrize(
    ("node", "changes"),
    [
        ("choose", {"binding": "", "business": ""}),
        ("choose", {"approval": Approval.EACH_RUN, "mandatory": True}),
        ("review", {"binding": "", "business": ""}),
        ("review", {"transitions": (go("done"),)}),
        ("review", {"transitions": (go("choose"),)}),
        ("review", {"transitions": (go("post"), go("done"))}),
        ("post", {"approval": Approval.NONE}),
        ("post", {"mandatory": False}),
        ("post", {"verify": ()}),
        ("post", {"binding": "", "business": ""}),
        ("post", {"kind": ActionKind.READ, "effect": None}),
    ],
)
def test_an_unchecked_choice_cannot_bypass_a_bound_commit_review(
    profile, node, changes
):
    cap = change_node(capability(profile), node, **changes)
    assert any(
        issue.code is IssueCode.UNVERIFIED_ACTION and issue.where == "nodes[0].verify"
        for issue in validate(cap)
    )


@pytest.mark.rule(3, 4, 6)
@pytest.mark.parametrize("risky", [False, True])
def test_typing_after_a_choice_does_not_prove_the_choice(profile, risky):
    cap = capability(profile)
    cap = dataclasses.replace(
        cap, targets=(*cap.targets, FocusTarget("focus", "/desk", ()))
    )
    cap = change_node(
        cap,
        "review",
        kind=ActionKind.TYPE,
        target="focus",
        value=constant("memo"),
    )
    cap = change_node(cap, "post", approval=Approval.NONE, mandatory=False)
    if risky:
        cap = change_node(cap, "choose", approval=Approval.EACH_RUN, mandatory=True)
    assert any(
        issue.code is IssueCode.UNVERIFIED_ACTION and issue.where == "nodes[0].verify"
        for issue in validate(cap)
    )


@pytest.mark.rule(3, 6, 16)
@pytest.mark.parametrize("limit", [Limit.RISKY, Limit.DENY])
def test_a_saved_restriction_cannot_use_the_deferred_selection_review(profile, limit):
    cap = capability(profile)
    cap = dataclasses.replace(
        cap,
        restrictions=(
            SavedRestriction(
                RestrictionScope.ROUTE,
                "/desk",
                (ActionKind.CLICK,),
                None,
                "",
                limit,
                Source.PROPOSAL,
            ),
        ),
    )
    assert any(
        issue.code is IssueCode.UNVERIFIED_ACTION and issue.where == "nodes[0].verify"
        for issue in validate(cap)
    )


@pytest.mark.rule(3, 4, 16)
def test_a_human_step_before_review_leaves_the_choice_unverified(profile):
    cap = capability(profile)
    person = HumanNode(
        "review",
        "/desk",
        HelpReason.UNREPRESENTABLE_TARGET,
        ActionKind.CLICK,
        "review_order",
        True,
        (go("post", Present("post")),),
    )
    cap = dataclasses.replace(
        cap,
        nodes=tuple(person if node.node_id == "review" else node for node in cap.nodes),
    )
    assert any(
        issue.code is IssueCode.UNVERIFIED_ACTION and issue.where == "nodes[0].verify"
        for issue in validate(cap)
    )


@pytest.mark.rule(3, 6, 16)
def test_review_in_another_frame_cannot_verify_the_choice(profile):
    cap = capability(profile)
    cap = dataclasses.replace(
        cap,
        targets=tuple(
            dataclasses.replace(target, frame=("other",))
            if target.target_id == "post"
            else target
            for target in cap.targets
        ),
    )
    assert any(
        issue.code is IssueCode.UNVERIFIED_ACTION and issue.where == "nodes[0].verify"
        for issue in validate(cap)
    )


class PaintedOrder(App):
    """A choice changes only a marker. Review displays the selected value."""

    selected = ""
    writes = 0
    miss_choice = False

    def lines(self, image):
        assert image == self.state.encode()
        texts = {
            "edit": ("Edit", "Quarterly", "Yearly", "Review"),
            "review": ("Review", self.selected, "Post"),
            "done": ("Posted",),
        }[self.state]
        return tuple(
            Line(text, 40, 20 + index * 40, 80, 20, 0.99)
            for index, text in enumerate(texts)
            if text
        )

    def capture(self, frame):
        assert frame == ()
        return Scene(f"cap-{self.looks}", self.state.encode())

    def _apply(self, action, found):
        assert found is not None
        assert isinstance(action.target, ScreenTarget)
        assert action.target.point is not None
        if self.state == "edit":
            if action.target.point.y > 130:
                self.state = "review"
            elif not self.miss_choice:
                self.selected = "Quarterly" if action.target.point.y < 90 else "Yearly"
        elif self.state == "review":
            self.writes += 1
            self.state = "done"


@pytest.mark.rule(3, 4, 5, 12, 16)
@pytest.mark.parametrize("frequency", ["Quarterly", "Yearly"])
@pytest.mark.parametrize("miss_choice", [False, True])
def test_replay_reviews_the_current_choice_before_sending_one_write(
    profile,
    frequency,
    miss_choice,
):
    class BoundOrder(PaintedOrder):
        def operation_binding(self, action, observation, node=None):
            assert observation is not None
            return operations.identify(
                profile,
                action,
                "/desk",
                node=node,
                lines=self.lines(self.state.encode()),
            )

    app = BoundOrder(
        {
            name: Page("/desk", (AxNode("canvas", "Application", tag="canvas"),))
            for name in ("edit", "review", "done")
        },
        "edit",
        {},
    )
    app.miss_choice = miss_choice

    def inspect(request):
        assert request.ask is Ask.APPROVAL
        question, step = request.reason.splitlines()
        assert question.startswith("Verify the selected choices")
        assert f"frequency={frequency}" in question
        assert step.startswith("Step post: ")
        return (
            HandoffOutcome.APPROVED
            if app.selected == frequency
            else HandoffOutcome.REJECTED
        )

    person = People([inspect])
    cap = capability(profile)
    original = dumps(cap)
    log = MemoryReplayLog()
    clock = Clock()
    result = replay(
        cap,
        {"frequency": frequency},
        profile=profile,
        surface=app,
        control=person.control(clock),
        log=log,
        clock=clock,
        sleep=clock.advance,
        scenes=app,
        reader=app,
    )
    assert result.status is (Status.TERMINATED if miss_choice else Status.SUCCEEDED), (
        result
    )
    assert app.writes == (0 if miss_choice else 1)
    assert len(person.requests) == 1
    assert dumps(cap) == original
    assert frequency not in repr(log.events)
