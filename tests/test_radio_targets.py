"""Painted radio labels survive recording without changing factual readings."""

import dataclasses
import json

import pytest
from replay_fakes import Clock, People, go, transfer_capability

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
from computeruse.capability import (
    ActionNode,
    Approval,
    IssueCode,
    Match,
    Present,
    RadioTarget,
    RefKind,
    ResultKind,
    ResultNode,
    Review,
    Shows,
    TextTarget,
    constant,
    dumps,
    loads,
    ref,
    validate,
)
from computeruse.decider import CheckKind, ResultCheck
from computeruse.loop import CheckResult, Ending, RunResult, Verification
from computeruse.profile import ActionKind
from computeruse.reading import LocalReader, radio_labels, squash
from computeruse.recording import DiscoveryTrace
from computeruse.replay import MemoryReplayLog, Status, replay
from computeruse.retarget import Found, Scene, radio_spot, text_line


def painted(choices, *, chosen=False, scale=1, left=40, columns=1):
    markup = """<!doctype html><style>body { margin: 0; }</style>
    <canvas id="screen" width="900" height="620"></canvas>
    <p id="receipt" role="status"></p><p id="chosen"></p>
    <p id="clicks">0</p><button id="reset">Reset</button>
    <script>
    const canvas = document.getElementById('screen');
    const paint = canvas.getContext('2d');
    const choices = __CHOICES__;
    const columns = __COLUMNS__;
    let scale = __SCALE__;
    let left = __LEFT__;
    let selected = __CHOSEN__;
    let clicks = 0;
    function draw() {
      paint.setTransform(1, 0, 0, 1, 0, 0);
      paint.fillStyle = '#ffffff';
      paint.fillRect(0, 0, canvas.width, canvas.height);
      paint.scale(scale, scale);
      paint.font = '18px Helvetica, Arial, sans-serif';
      choices.forEach((choice, index) => {
        const x = left + (index % columns) * 180;
        const y = 70 + Math.floor(index / columns) * 60;
        if (choice.marker) {
          paint.strokeStyle = '#52606d';
          paint.lineWidth = 1.5;
          paint.beginPath();
          paint.arc(x + 10, y - 6, 9, 0, Math.PI * 2);
          paint.stroke();
          if (selected) {
            paint.fillStyle = '#1d4ed8';
            paint.beginPath();
            paint.arc(x + 10, y - 6, 5, 0, Math.PI * 2);
            paint.fill();
          }
        }
        paint.fillStyle = '#1f2933';
        paint.fillText(choice.label, x + 28, y);
      });
    }
    canvas.addEventListener('click', event => {
      const x = event.offsetX / scale;
      const y = event.offsetY / scale;
      const index = choices.findIndex((choice, index) => choice.marker &&
        Math.abs(x - left - (index % columns) * 180 - 10) < 12 &&
        Math.abs(y - 64 - Math.floor(index / columns) * 60) < 12);
      if (index < 0) return;
      document.getElementById('chosen').textContent = choices[index].label;
      document.getElementById('receipt').textContent = 'Choice recorded';
      document.getElementById('clicks').textContent = String(++clicks);
      selected = true;
      draw();
    });
    document.getElementById('reset').addEventListener('click', () => {
      left = 150;
      scale = 1.5;
      selected = false;
      clicks = 0;
      for (const id of ['receipt', 'chosen']) {
        document.getElementById(id).textContent = '';
      }
      document.getElementById('clicks').textContent = '0';
      draw();
    });
    draw();
    </script>"""
    for token, value in (
        ("__CHOICES__", choices),
        ("__COLUMNS__", columns),
        ("__CHOSEN__", chosen),
        ("__SCALE__", scale),
        ("__LEFT__", left),
    ):
        markup = markup.replace(token, json.dumps(value))
    return markup


@pytest.fixture(scope="module")
def reader():
    return LocalReader()


def capture(surface):
    observed = surface.observe(ObservationRequest(ObservationMode.VISUAL))
    assert observed.image is not None
    return observed, Scene(observed.observation_id, observed.image)


@pytest.mark.rule(2, 9)
@pytest.mark.parametrize(
    ("selected", "scale", "left"),
    [(False, 1, 40), (True, 1, 220), (False, 1.5, 80), (True, 2, 100)],
)
def test_radio_resolution_preserves_letters_across_selection_and_layout(
    pages, reader, selected, scale, left
):
    labels = ("Paper", "Orange", "OPaper")
    choices = [{"label": label, "marker": True} for label in labels]
    choices.append({"label": "OPaper", "marker": False})
    markup = painted(choices, chosen=selected, scale=scale, left=left)
    with pages({"/": markup}) as (surface, _):
        _, scene = capture(surface)
        raw = reader.lines(scene.image)
        recognized = radio_labels(scene.image, reader, raw)
        assert {squash(choice.label.text) for choice in recognized} == {
            squash(label) for label in labels
        }
        for index, label in enumerate(labels):
            target = RadioTarget("choice", "/", (), constant(label))
            spotted = radio_spot(target, scene, reader, lambda value: value.value)
            assert spotted.found is Found.FOUND, (label, recognized)
            assert spotted.target is not None
            assert spotted.target.point is not None
            assert spotted.target.point.x == pytest.approx((left + 10) * scale, abs=2)
            assert spotted.target.point.y == pytest.approx(
                (64 + index * 60) * scale, abs=2
            )
        assert reader.lines(scene.image) == raw
        plain = tuple(
            line for line in raw if 230 * scale < line.centre[1] < 260 * scale
        )
        assert any(squash(line.text) == "opaper" for line in plain)
        target = TextTarget(
            "word", "/", (), ref(RefKind.INPUT, "choice"), Match.CONTAINS
        )
        assert text_line(target, plain, lambda _: "paper")[0] is Found.NOT_FOUND


@pytest.mark.rule(2, 6)
@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        ([{"label": "Paper", "marker": False}], Found.NOT_FOUND),
        ([{"label": "OPaper", "marker": False}], Found.NOT_FOUND),
        (
            [{"label": "Paper", "marker": True}] * 2,
            Found.AMBIGUOUS,
        ),
    ],
)
def test_radio_target_requires_one_marker_and_its_own_label(
    pages, reader, choices, expected
):
    with pages({"/": painted(choices)}) as (surface, _):
        _, scene = capture(surface)
        target = RadioTarget("choice", "/", (), constant("Paper"))
        spotted = radio_spot(target, scene, reader, lambda value: value.value)
        assert spotted.found is expected


@pytest.mark.rule(2, 6)
@pytest.mark.parametrize(("selected", "scale"), [(False, 1), (True, 1.5)])
def test_horizontal_radio_options_remain_separate_click_targets(
    pages, reader, selected, scale
):
    labels = ("Paper", "Electronic")
    choices = [{"label": label, "marker": True} for label in labels]
    markup = painted(choices, chosen=selected, scale=scale, columns=2)
    with pages({"/": markup}) as (surface, _):
        for label in labels:
            _, scene = capture(surface)
            target = RadioTarget("choice", "/", (), constant(label))
            spotted = radio_spot(target, scene, reader, lambda value: value.value)
            assert spotted.found is Found.FOUND
            assert spotted.target is not None
            clicked = surface.act(Action(ActionKind.CLICK, spotted.target))
            assert clicked.outcome is Outcome.OK
            result = surface.act(
                Action(ActionKind.READ, DomLocator("p", DomAttribute.ID, "chosen"))
            )
            assert result.extracted == label


@pytest.mark.rule(2, 6)
@pytest.mark.parametrize("selected", [False, True])
def test_multiword_radio_labels_remain_one_control(pages, reader, selected):
    labels = ("EUR checking", "USD savings", "Electronic delivery")
    choices = [{"label": label, "marker": True} for label in labels]
    with pages({"/": painted(choices, chosen=selected)}) as (surface, _):
        for label in labels:
            _, scene = capture(surface)
            target = RadioTarget("choice", "/", (), constant(label))
            spotted = radio_spot(target, scene, reader, lambda value: value.value)
            assert spotted.found is Found.FOUND
            assert spotted.target is not None
            assert (
                surface.act(Action(ActionKind.CLICK, spotted.target)).outcome
                is Outcome.OK
            )
            result = surface.act(
                Action(ActionKind.READ, DomLocator("p", DomAttribute.ID, "chosen"))
            )
            assert result.extracted == label


@pytest.mark.rule(17)
@pytest.mark.parametrize(
    "failure",
    ["image", "empty_label", "ambiguous_label", "multiple_rows", "distant_words"],
)
def test_an_unreadable_radio_capture_is_unavailable(pages, reader, failure):
    scene = Scene("broken", b"not a PNG image")
    if failure != "image":
        markup = painted([{"label": "Paper", "marker": True}])
        with pages({"/": markup}) as (surface, _):
            _, scene = capture(surface)

    class Reader:
        def lines(self, image):
            if image == scene.image:
                return reader.lines(image)
            if failure == "empty_label":
                return ()
            words = reader.lines(image)
            assert len(words) == 1
            if failure == "multiple_rows":
                return words[0], dataclasses.replace(
                    words[0],
                    left=words[0].left + words[0].width,
                    top=words[0].top + words[0].height,
                )
            if failure == "distant_words":
                return words[0], dataclasses.replace(
                    words[0], left=words[0].left + words[0].width + words[0].height * 3
                )
            return words[0], dataclasses.replace(words[0], text="Other")

    target = RadioTarget("choice", "/", (), constant("Paper"))
    spotted = radio_spot(target, scene, Reader(), lambda value: value.value)
    assert spotted.found is Found.UNAVAILABLE
    assert spotted.target is None


@pytest.mark.rule(3, 4, 8, 9)
def test_a_recorded_radio_click_replays_a_changed_input_on_the_live_canvas(
    pages, reader, tmp_path
):
    choices = [{"label": label, "marker": True} for label in ("Paper", "Orange")]
    with pages({"/": painted(choices)}) as (surface, profile):
        before, _ = capture(surface)
        action = Action(
            ActionKind.CLICK,
            ScreenTarget(before.observation_id, Point(50, 64)),
            effect="choose",
        )
        trace = DiscoveryTrace()
        trace.look(before)
        dispatched = surface.act(action)
        assert dispatched.outcome is Outcome.OK
        trace.action(1, action, dispatched, before.page_state.location, before)
        after = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
        trace.look(after)
        source = DomLocator("p", DomAttribute.ID, "receipt")
        checked = surface.act(Action(ActionKind.READ, source))
        assert checked.outcome is Outcome.OK
        assert checked.extracted == "Choice recorded"
        check = ResultCheck(CheckKind.STATE, source, "Choice recorded")
        built = trace.build(
            RunResult(
                Ending.COMPLETED,
                2,
                "done",
                verification=Verification.EXECUTOR,
                checks=(CheckResult(check, checked.extracted, True),),
            ),
            profile=profile,
            inputs={"choice": "paper"},
            capability_id="radio_choice",
            run="browser",
            safe_text=frozenset({"Choice recorded", "choose", "receipt", "Reset"}),
            reader=reader,
        )
        assert built.complete, (built.issues, built.artifact_issues)
        assert built.capability is not None
        capability = loads(dumps(built.capability))
        target = next(
            item for item in capability.targets if isinstance(item, RadioTarget)
        )
        assert target.text == ref(RefKind.INPUT, "choice")
        assert '"radio"' in dumps(capability)
        capability = dataclasses.replace(
            capability,
            provenance=dataclasses.replace(
                capability.provenance, review=Review.REVIEWED
            ),
        )
        (tmp_path / "radio-capability.json").write_text(dumps(capability))
        reset = surface.act(Action(ActionKind.CLICK, AxLocator("button", "Reset")))
        assert reset.outcome is Outcome.OK
        clock = Clock()
        people = People([])
        result = replay(
            capability,
            {"choice": "orange"},
            profile=profile,
            surface=surface,
            control=people.control(clock),
            log=MemoryReplayLog(),
            clock=clock,
            sleep=clock.advance,
            reader=reader,
        )
        assert result.status is Status.SUCCEEDED, result
        assert not people.requests
        for element, expected in (("chosen", "Orange"), ("clicks", "1")):
            reading = surface.act(
                Action(ActionKind.READ, DomLocator("p", DomAttribute.ID, element))
            )
            assert reading.extracted == expected


def radio_capability():
    base = transfer_capability()
    target = RadioTarget("choice", "/desk", (), constant("Paper"))
    action = ActionNode(
        "choose",
        ActionKind.CLICK,
        "/desk",
        "choice",
        None,
        None,
        "choose",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (Present("choice"),),
        (go("done"),),
    )
    return dataclasses.replace(
        base,
        application=dataclasses.replace(
            base.application, entry_route="/desk", markers=()
        ),
        inputs=(),
        outputs=(),
        secrets=(),
        outcomes=(),
        targets=(target,),
        entry="choose",
        nodes=(
            action,
            ResultNode("done", ResultKind.SUCCESS, "", (Present("choice"),)),
        ),
    )


@pytest.mark.rule(2, 10, 11)
def test_radio_target_round_trips_but_cannot_become_factual_evidence():
    capability = radio_capability()
    assert validate(capability) == ()
    assert loads(dumps(capability)) == capability
    target = capability.targets[0]
    for change in ({"label": True}, {"dx": 1}, {"dy": 1}):
        invalid = dataclasses.replace(
            capability, targets=(dataclasses.replace(target, **change),)
        )
        assert IssueCode.INVALID_TARGET in {issue.code for issue in validate(invalid)}
    action, done = capability.nodes
    assert isinstance(action, ActionNode)
    assert isinstance(done, ResultNode)
    invalid_read = dataclasses.replace(action, kind=ActionKind.READ, effect=None)
    issues = validate(dataclasses.replace(capability, nodes=(invalid_read, done)))
    assert any(
        issue.code is IssueCode.UNSUPPORTED_TARGET and issue.where == "nodes[0].target"
        for issue in issues
    )
    invalid_done = dataclasses.replace(
        done, checks=(Shows("choice", constant("Paper"), Match.EQUALS),)
    )
    issues = validate(dataclasses.replace(capability, nodes=(action, invalid_done)))
    assert IssueCode.UNSUPPORTED_TARGET in {issue.code for issue in issues}
