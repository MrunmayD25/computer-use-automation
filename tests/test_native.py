"""The native control window: its commands, its helper, and the docked highlight."""

import contextlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import cast

import pytest

import computeruse.native as native
from computeruse.actions import Action, AxLocator, ObservationMode, ObservationRequest
from computeruse.browser import TargetMarker, open_session
from computeruse.budget import Budget
from computeruse.control import Control, Status
from computeruse.escalation import Mode, State
from computeruse.journal import MemoryJournal
from computeruse.native import NativeWindow, NativeWindowError
from computeruse.panel import ControlWindow
from computeruse.profile import ActionKind, load_profile

LOOK = ObservationRequest(ObservationMode.STRUCTURED)
AMBER = "rgb(233, 158, 18)"

# A stand-in helper. It answers the first status it is shown with one click
# on Stop, and writes the receipt it gets back to the file it is given.
CLICKS_STOP = """
import json, sys
from pathlib import Path
from computeruse.native import message_for
for line in sys.stdin:
    message = json.loads(line)
    if "status" in message and not Path(sys.argv[1]).with_suffix(".sent").exists():
        Path(sys.argv[1]).with_suffix(".sent").write_text("")
        print(json.dumps({"command": message_for(message["status"], "stop", "")}),
              flush=True)
    elif "receipt" in message:
        Path(sys.argv[1]).write_text(json.dumps(message["receipt"]))
        break
"""


@contextlib.contextmanager
def helper(script: str, *args: str):
    process = subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        yield NativeWindow(process)
    finally:
        process.kill()
        process.wait()


def started(window) -> Control:
    profile = load_profile(Path("evaluation/profile.yaml"))
    control = Control(mode=Mode.DISCOVERY, clock=time.monotonic, channels=[window])
    assert control.begin(
        profile=profile,
        budget=Budget(profile.budgets, time.monotonic),
        journal=MemoryJournal(),
        context="look up the member",
    )
    return control


def test_a_click_in_the_window_reaches_the_run_as_a_checked_order(tmp_path):
    receipt = tmp_path / "receipt.json"
    with helper(CLICKS_STOP, str(receipt)) as window:
        control = started(window)
        deadline = time.monotonic() + 10
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        answered = json.loads(receipt.read_text())
    assert answered["verdict"] == "accepted"
    assert control.status().state is State.STOPPING


@pytest.mark.parametrize(
    "line",
    [
        # Not a command at all, or a command with a field the window never sends.
        "not json",
        json.dumps({"status": {}}),
        json.dumps({"command": {"command": "stop"}, "via": "terminal"}),
    ],
)
def test_a_line_that_is_not_one_command_is_ignored(line):
    with helper("import sys; sys.stdin.read()") as window:
        control = started(window)
        assert window._command(line) is None
        assert control.status().state is State.RUNNING


def test_a_command_the_run_cannot_read_changes_nothing():
    with helper("import sys; sys.stdin.read()") as window:
        control = started(window)
        status = control.status()
        malformed = {"command": "approve", "run": status.run, "revision": "4"}
        receipt = window._command(json.dumps({"command": malformed}))
        assert receipt is not None
        assert receipt["verdict"] == "invalid"
        # A well-formed command for another run is judged, and refused, by the run.
        other = native.message_for({"run": "run-other", "revision": 1}, "stop", "")
        refused = window._command(json.dumps({"command": other}))
        assert refused is not None
        assert refused["verdict"] != "accepted"
        assert control.status().state is State.RUNNING


def test_the_helper_is_told_where_tk_keeps_its_scripts(tmp_path, monkeypatch):
    tkinter = pytest.importorskip("tkinter")
    (tmp_path / "lib" / f"tcl{tkinter.TclVersion}").mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    monkeypatch.delenv("TCL_LIBRARY", raising=False)
    monkeypatch.setenv("TK_LIBRARY", "/chosen")
    env = native._environment()
    assert env["TCL_LIBRARY"] == str(tmp_path / "lib" / f"tcl{tkinter.TclVersion}")
    # A folder the operator named is kept, and a missing one is not invented.
    assert env["TK_LIBRARY"] == "/chosen"


def test_a_helper_that_never_opens_its_window_is_an_error(monkeypatch):
    monkeypatch.setattr(native, "READY_S", 0.5)

    class Silent:
        stdout = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdout=subprocess.PIPE,
            text=True,
        ).stdout

    with pytest.raises(NativeWindowError):
        native._ready(cast("subprocess.Popen[str]", Silent()))


def test_without_a_native_window_the_run_uses_the_browser_control_window(
    site, site_profile, monkeypatch
):
    from computeruse.cli import _control_window

    def unavailable():
        raise NativeWindowError("no display")

    monkeypatch.setattr(native, "open_native_window", unavailable)
    with (
        open_session(site_profile(), site + "/queue", record=True) as surface,
        contextlib.ExitStack() as stack,
    ):
        channels = _control_window(surface, stack)
        assert [type(channel) for channel in channels] == [ControlWindow, TargetMarker]


def test_a_docked_session_marks_its_target_where_no_look_sees_it(site, site_profile):
    with open_session(site_profile(), site + "/queue", record=True) as surface:
        before = surface.observe(LOOK)
        # A visible session sets this when it docks a control window.
        surface._marking = True
        approve = Action(ActionKind.CLICK, AxLocator("button", "Approve"))
        surface.preview(approve)
        page = surface.browser.contexts[0].pages[0]
        glow = page.locator("[data-highlight=true]")
        assert glow.count() == 1
        assert glow.evaluate("el => getComputedStyle(el).pointerEvents") == "none"
        assert surface.observe(LOOK).nodes == before.nodes
        # While a person decides, the same target turns amber.
        control = types.SimpleNamespace(proposal=approve)
        marker = TargetMarker(surface)
        marker.attach(cast("Control", control))
        marker.show(cast("Status", types.SimpleNamespace(offer=object())))
        assert glow.evaluate("el => getComputedStyle(el).borderTopColor") == AMBER
        assert not marker.listening()


def test_an_undocked_session_draws_no_highlight(site, site_profile):
    with open_session(site_profile(), site + "/queue", record=True) as surface:
        surface.preview(Action(ActionKind.CLICK, AxLocator("button", "Approve")))
        page = surface.browser.contexts[0].pages[0]
        assert page.locator("[data-highlight=true]").count() == 0
