"""Where the control window goes beside a visible session, on any screen area."""

import pytest

from computeruse.browser import WindowRect, fit

LAPTOP = WindowRect(0, 33, 1512, 892)
PANEL = 500


def test_the_panel_goes_beside_the_size_a_person_chose():
    chosen = WindowRect(0, 33, 900, 700)
    application, panel = fit(chosen, LAPTOP, PANEL)
    assert application == chosen
    assert panel == WindowRect(900, 33, PANEL, 700)


def test_the_panel_uses_the_left_side_when_only_that_side_has_room():
    chosen = WindowRect(600, 33, 900, 700)
    application, panel = fit(chosen, LAPTOP, PANEL)
    assert application == chosen
    assert panel.right == chosen.left


@pytest.mark.parametrize(
    "chosen",
    [
        # Maximized, then a window wider than the room either side allows.
        WindowRect(0, 33, 1512, 892),
        WindowRect(200, 33, 1100, 892),
    ],
)
def test_without_room_the_application_narrows_and_nothing_is_covered(chosen):
    application, panel = fit(chosen, LAPTOP, PANEL)
    assert application.right <= panel.left
    assert application.left >= LAPTOP.left
    assert panel.right <= LAPTOP.right


def test_a_second_screen_keeps_its_own_origin():
    second = WindowRect(1512, 0, 1920, 1080)
    chosen = WindowRect(1600, 40, 1000, 900)
    application, panel = fit(chosen, second, PANEL)
    assert application == chosen
    assert panel.left == chosen.right
    assert panel.right <= second.right


def test_a_window_knows_which_screen_holds_it():
    assert LAPTOP.holds(WindowRect(0, 33, 900, 700))
    assert not LAPTOP.holds(WindowRect(1600, 40, 1000, 900))
