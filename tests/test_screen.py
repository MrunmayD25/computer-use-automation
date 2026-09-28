"""Screen input acts on pixels without requiring a semantic locator."""

import dataclasses

import pytest

from computeruse.actions import (
    Action,
    MouseButton,
    MouseInput,
    ObservationMode,
    ObservationRequest,
    Outcome,
    Point,
    ScreenTarget,
    SecretRef,
)
from computeruse.browser import open_session
from computeruse.profile import ActionKind


@pytest.fixture
def screen(site, site_profile):
    with open_session(site_profile(), f"{site}/members") as surface:
        surface._page.set_content("""
            <style>body {margin: 30px}
              input, button {display:block; margin:20px}</style>
            <label>Username<input id="user"></label>
            <label>Password<input id="password" type="password"></label>
            <button onclick="document.querySelector('output').textContent='clicked'">
              Go</button>
            <output>waiting</output>
        """)
        yield surface


def capture(surface, selector=None):
    observation = surface.observe(ObservationRequest(ObservationMode.VISUAL))
    point = None
    if selector is not None:
        box = surface._page.locator(selector).bounding_box()
        point = Point(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    return ScreenTarget(observation.observation_id, point)


def test_pixels_click_type_and_use_a_key_combination(screen):
    assert not screen.observe(ObservationRequest(ObservationMode.VISUAL)).regions
    assert (
        screen.act(Action(ActionKind.CLICK, capture(screen, "#user"))).outcome
        is Outcome.OK
    )
    assert (
        screen.act(Action(ActionKind.TYPE, capture(screen), "first")).outcome
        is Outcome.OK
    )
    assert (
        screen.act(
            Action(ActionKind.PRESS_KEY, capture(screen), "ControlOrMeta+a")
        ).outcome
        is Outcome.OK
    )
    assert (
        screen.act(Action(ActionKind.TYPE, capture(screen), "second")).outcome
        is Outcome.OK
    )
    assert screen._page.locator("#user").input_value() == "second"
    assert (
        screen.act(Action(ActionKind.CLICK, capture(screen, "button"))).outcome
        is Outcome.OK
    )
    assert screen._page.locator("output").inner_text() == "clicked"


def test_pixels_refuse_stale_captures_movement_and_out_of_bounds(screen):
    target = capture(screen, "button")
    capture(screen)
    assert screen.act(Action(ActionKind.CLICK, target)).outcome is Outcome.STALE
    target = capture(screen, "button")
    screen._page.locator("button").evaluate("e => e.style.marginLeft = '250px'")
    assert screen.act(Action(ActionKind.CLICK, target)).outcome is Outcome.STALE
    target = dataclasses.replace(capture(screen), point=Point(9999, 9999))
    assert (
        screen.act(Action(ActionKind.CLICK, target)).outcome is Outcome.NOT_ACTIONABLE
    )
    assert screen._page.locator("output").inner_text() == "waiting"


def test_pixel_typing_detects_focus_changes(screen):
    screen._page.locator("#user").focus()
    target = capture(screen)
    screen._page.locator("#password").focus()
    assert (
        screen.act(Action(ActionKind.TYPE, target, "wrong place")).outcome
        is Outcome.STALE
    )
    assert screen._page.locator("#password").input_value() == ""


def test_pixel_secret_input_is_protected(screen, monkeypatch):
    monkeypatch.setenv("APPROVER_PASSCODE", "private-value")
    screen._page.locator("#user").focus()
    action = Action(ActionKind.TYPE, capture(screen), SecretRef("approver_passcode"))
    assert screen.act(action).outcome is Outcome.OK
    seen = screen.observe(ObservationRequest(ObservationMode.STRUCTURED))
    assert all(node.value != "private-value" for node in seen.nodes)
    assert screen._page.locator("#user").input_value() == "private-value"


def test_screen_coordinates_are_finite():
    with pytest.raises(ValueError) as raised:  # noqa: PT011  compare the message literally
        Point(float("nan"), 0)
    assert "finite" in str(raised.value)


def test_pixel_input_reaches_permitted_iframe_without_a_frame_locator(screen):
    screen._page.set_content('<iframe src="/members/12345/savings"></iframe>')
    frame = screen._page.frame(url=screen.location() + "/12345/savings")
    if frame is None:
        screen._page.locator("iframe").content_frame.locator("body").wait_for()
        frame = screen._page.frames[1]
    frame.set_content('<input id="inside"><output></output>')
    field = frame.locator("#inside")
    box = field.bounding_box()
    target = dataclasses.replace(
        capture(screen), point=Point(box["x"] + 10, box["y"] + 10)
    )
    assert screen.act(Action(ActionKind.CLICK, target)).outcome is Outcome.OK
    assert (
        screen.act(Action(ActionKind.TYPE, capture(screen), "member")).outcome
        is Outcome.OK
    )
    assert field.input_value() == "member"


def test_pixel_input_refuses_an_out_of_policy_frame(screen):
    screen._page.set_content('<iframe srcdoc="<button>Forbidden</button>"></iframe>')
    screen._page.frames[1].locator("button").wait_for()
    target = capture(screen, "iframe")
    assert screen.act(Action(ActionKind.CLICK, target)).outcome is Outcome.BLOCKED


def test_pixels_can_right_click_double_click_hover_scroll_and_drag(screen):
    screen._page.set_content("""
      <style>body {margin:0} canvas {display:block}</style>
      <canvas width="400" height="300" tabindex="0"></canvas><output></output>
      <div style="height:1600px"></div>
      <script>
        const canvas = document.querySelector('canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#aabbcc'; ctx.fillRect(0,0,400,300);
        window.events = [];
        const names = ['contextmenu', 'dblclick', 'mousemove', 'mousedown', 'mouseup'];
        for (const name of names) {
          canvas.addEventListener(name, e => {
            e.preventDefault(); window.events.push([name,e.buttons,e.shiftKey]);
          });
        }
      </script>
    """)
    right = Action(
        ActionKind.CLICK,
        capture(screen, "canvas"),
        mouse=MouseInput(button=MouseButton.RIGHT),
    )
    assert screen.act(right).outcome is Outcome.OK
    assert (
        screen.act(Action(ActionKind.DOUBLE_CLICK, capture(screen, "canvas"))).outcome
        is Outcome.OK
    )
    assert (
        screen.act(Action(ActionKind.MOVE, capture(screen, "canvas"))).outcome
        is Outcome.OK
    )
    drag = Action(
        ActionKind.DRAG,
        capture(screen, "canvas"),
        mouse=MouseInput(modifiers=("Shift",), path=(Point(240, 180), Point(290, 220))),
    )
    assert screen.act(drag).outcome is Outcome.OK
    events = screen._page.evaluate("window.events")
    assert any(e[0] == "contextmenu" for e in events)
    assert any(e[0] == "dblclick" for e in events)
    assert any(e == ["mousemove", 1, True] for e in events)
    screen._page.mouse.move(200, 150)
    assert screen._page.evaluate("window.events.at(-1)") == ["mousemove", 0, False]
    scroll = Action(
        ActionKind.SCROLL,
        capture(screen, "canvas"),
        "pixels",
        mouse=MouseInput(delta=Point(0, 450)),
    )
    assert screen.act(scroll).outcome is Outcome.OK
    assert screen._page.evaluate("window.scrollY") > 0


def test_pixel_capture_cannot_be_reused_after_any_action(screen):
    from computeruse.actions import AxLocator

    target = capture(screen, "button")
    assert (
        screen.act(Action(ActionKind.CLICK, AxLocator("button", "Go"))).outcome
        is Outcome.OK
    )
    assert screen.act(Action(ActionKind.CLICK, target)).outcome is Outcome.STALE


def test_pixel_typing_sends_keyboard_events_to_a_canvas(screen):
    screen._page.set_content("""
      <canvas tabindex="0"></canvas><output></output>
      <script>
        document.querySelector('canvas').addEventListener('keydown', e => {
          document.querySelector('output').textContent += e.key;
        });
      </script>
    """)
    screen._page.locator("canvas").focus()
    assert (
        screen.act(Action(ActionKind.TYPE, capture(screen), "hello")).outcome
        is Outcome.OK
    )
    assert screen._page.locator("output").inner_text() == "hello"


def test_pixel_typing_stops_when_focus_moves_during_input(screen):
    screen._page.locator("#user").evaluate("""el => {
      el.addEventListener('input', () => document.querySelector('#password').focus());
      el.focus();
    }""")
    assert (
        screen.act(Action(ActionKind.TYPE, capture(screen), "abc")).outcome
        is Outcome.STALE
    )
    assert screen._page.locator("#user").input_value() == "a"
    assert screen._page.locator("#password").input_value() == ""


def test_pixel_typing_keeps_the_captured_target_until_delivery(screen, monkeypatch):
    screen._page.locator("#user").focus()
    target = capture(screen)
    ready = screen._screen_ready

    def move_focus(_self, target):
        result = ready(target)
        screen._page.locator("#password").focus()
        return result

    monkeypatch.setattr(type(screen), "_screen_ready", move_focus)
    assert (
        screen.act(Action(ActionKind.TYPE, target, "wrong field")).outcome
        is Outcome.STALE
    )
    assert screen._page.locator("#user").input_value() == ""
    assert screen._page.locator("#password").input_value() == ""


def test_navigation_during_capture_returns_a_failed_observation(screen, monkeypatch):
    from playwright.sync_api import Error

    from computeruse.browser import BrowserSurface

    def detached(_self):
        raise Error("Frame was detached")

    with monkeypatch.context() as patch:
        patch.setattr(BrowserSurface, "_masks", detached)
        result = screen.observe(ObservationRequest(ObservationMode.VISUAL))
    assert not result.usable
    assert result.image is None
    assert screen.observe(ObservationRequest(ObservationMode.VISUAL)).usable


@pytest.mark.parametrize("change", ["focus", "scroll", "navigation"])
def test_capture_refuses_context_that_changes_while_taking_the_picture(
    screen, monkeypatch, change
):
    screen._page.locator("#user").focus()
    screen._page.evaluate("document.body.style.height = '2000px'")
    screenshot = screen._page.screenshot

    def changing_screenshot(**kwargs):
        image = screenshot(**kwargs)
        if change == "focus":
            screen._page.locator("#password").focus()
        elif change == "scroll":
            screen._page.evaluate("window.scrollTo(0, 500)")
        else:
            screen._page.reload()
        return image

    with monkeypatch.context() as patch:
        patch.setattr(screen._page, "screenshot", changing_screenshot)
        seen = screen.observe(ObservationRequest(ObservationMode.VISUAL))
    assert not seen.usable
    assert seen.image is None
    assert (
        screen.act(
            Action(ActionKind.TYPE, ScreenTarget(seen.observation_id), "x")
        ).outcome
        is Outcome.STALE
    )
    assert screen.observe(ObservationRequest(ObservationMode.VISUAL)).usable


@pytest.mark.rule(6)
@pytest.mark.parametrize("changes_during_input", [False, True])
def test_a_screen_point_names_the_observed_control_under_it(
    screen, changes_during_input
):
    """A screenshot click on page content is judged as that control (rule 6)."""
    from computeruse.actions import Expectation, PageState

    seen = screen.observe(ObservationRequest(ObservationMode.STRUCTURED))
    button = next(node for node in seen.nodes if node.role == "button")
    field = next(node for node in seen.nodes if node.name == "Username")
    target = capture(screen, "button")
    assert screen.control_at(target) == button.control
    assert screen.control_at(ScreenTarget(target.capture_id)) == ""

    # Authorized as another control, the click is refused before input.
    other = Expectation(PageState(screen.location()), control=field.control)
    result = screen.act(Action(ActionKind.CLICK, target), expect=other)
    assert result.outcome is Outcome.STALE
    assert screen._page.locator("output").inner_text() == "waiting"

    # Authorized as the control under the point, it goes through.
    same = Expectation(
        PageState(screen.location()),
        control=button.control,
        submits_as=button.submits_as,
        observed_control=button,
    )
    if changes_during_input:
        screen._page.mouse.move(0, 0)
        screen._page.locator("button").evaluate("""el => {
          el.addEventListener('pointermove', () => {
            el.setAttribute('aria-label', 'Changed button');
          }, {once: true});
        }""")
    result = screen.act(
        Action(ActionKind.CLICK, capture(screen, "button")), expect=same
    )
    if changes_during_input:
        assert result.outcome is Outcome.STALE
        assert screen._page.locator("output").inner_text() == "waiting"
        fresh = screen.observe(ObservationRequest(ObservationMode.STRUCTURED))
        current = next(node for node in fresh.nodes if node.control == button.control)
        result = screen.act(
            Action(ActionKind.CLICK, capture(screen, "button")),
            expect=dataclasses.replace(same, observed_control=current),
        )
    assert result.outcome is Outcome.OK
    assert screen._page.locator("output").inner_text() == "clicked"
