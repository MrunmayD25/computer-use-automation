"""The operator's control panel in a native window that stays above other windows.

A browser window does not stay in front on macOS. When a person clicks the
application's window, only that window comes forward, and a control window
beside it can end up behind another app. So a headed run draws its controls in
a small Tk window in a helper process, ``python -m computeruse.native``, and
keeps that window above every other window, over a full-screen app as well.

On macOS, the helper uses the Objective-C runtime through ``ctypes`` to apply
three settings that Tk does not expose:

- The helper runs as an accessory app, with no Dock icon. macOS shows another
  app's window over a full-screen app only for such an app.
- The window joins every Space, including full-screen ones.
- The window sits at the floating level, above ordinary windows.

The run and the helper talk over the helper's standard input and output, one
JSON object per line. The run sends the status to draw, where to place the
window, and the receipt for each command. The helper sends each command a
person clicks, with the run, the status revision it drew, and the request id.
The run checks every command with ``panel.order_from``, the same check the
browser control window uses, and hands it to ``Control.submit``, which is safe
from the reader thread. Only this process holds the helper's pipes, so no
other program can send commands through it.

The helper sets every label with plain text, because the reason and the
proposal can quote page text and model text. A person cannot close the window
while the run lasts; the run closes it when it ends.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from computeruse.control import Control, Status
from computeruse.panel import MAX_NOTE, order_from, payload

if TYPE_CHECKING:
    import tkinter as tk

WIDTH = 420
"""The window's width in screen points, beside the application window."""

READY_S = 10.0
"""How long the helper may take to open its window before the run gives up."""

CLOSE_S = 2.0
"""How long the helper may take to close before it is stopped."""

CONFIRM_S = 3.0
"""How long a first click on Terminate waits for the confirming second click."""

POLL_MS = 50
"""How often the helper's window takes in what the run sent."""

TITLE = "computeruse control"
"""The window's title, which also finds it among the helper's windows."""

TONES = {"start": "#1f8a4c", "resume": "#1f8a4c", "stop": "#b26a00"}
"""The main button's color for each command it sends."""

MODES = {
    "discovery": ("#1d4ed8", "DISCOVERY", "The model is working out the steps"),
    "replay": ("#6d28d9", "REPLAY", "The saved capability runs, with no model"),
}
"""The banner's color, title, and line for each mode a run can be in."""

MUTED = "#6b7280"
APPROVE = "#1f8a4c"
TERMINATE = "#b3261e"


class NativeWindowError(RuntimeError):
    """The native control window could not open. The run uses another channel."""


class NativeWindow:
    """A run's channel to the native control window in the helper process.

    ``show`` runs on the owner thread and writes the status to the helper.
    A reader thread reads the commands a person clicks, checks each, submits
    it, and writes the receipt back. Writes from both threads share a lock,
    so no two messages interleave.
    """

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._control: Control | None = None
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @classmethod
    def open(cls) -> contextlib.AbstractContextManager[NativeWindow]:
        """Open the native control window, and close it on exit.

        Raises
        ------
        NativeWindowError
            If the helper cannot start or does not open its window in time.
        """
        return open_native_window()

    def attach(self, control: Control) -> None:
        """Send this control the commands the person clicks."""
        self._control = control

    def show(self, status: Status) -> None:
        """Draw ``status`` in the window. Owner thread only."""
        self._send({"status": payload(status)})

    def listening(self) -> bool:
        """Report whether the helper still runs its window."""
        return self._process.poll() is None

    def place(self, left: int, top: int, width: int, height: int) -> None:
        """Move the window's outer frame to this rectangle, in screen points."""
        self._send({"place": [left, top, width, height]})

    def _send(self, message: Mapping[str, object]) -> None:
        stream = self._process.stdin
        if stream is None or not self.listening():
            return
        line = json.dumps(message, ensure_ascii=True)
        with self._lock, contextlib.suppress(OSError, ValueError):
            stream.write(line + "\n")
            stream.flush()

    def _read(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        for line in stream:
            receipt = self._command(line)
            if receipt is not None:
                self._send({"receipt": receipt})

    def _command(self, line: str) -> dict[str, object] | None:
        """Judge one line the helper sent, and return the receipt to draw."""
        control = self._control
        try:
            message = json.loads(line)
        except ValueError:
            return None
        if not isinstance(message, dict) or set(message) != {"command"}:
            return None
        if control is None:
            return {"verdict": "invalid", "reason": "the window is not attached"}
        order = order_from(message["command"])
        if order is None:
            return {
                "verdict": "invalid",
                "reason": "the window sent a command the run cannot read",
                "status": payload(control.status()),
            }
        receipt = control.submit(order)
        return {
            "verdict": receipt.verdict.value,
            "reason": receipt.reason,
            "status": payload(receipt.status),
        }


def _environment() -> dict[str, str]:
    """Return the helper's environment, with Tk's library folders named.

    A Python built by uv finds Tcl's own scripts next to its executable. Run
    through a virtual environment's link it sometimes does not, so the folders
    are named when they exist.
    """
    env = dict(os.environ)
    try:
        import tkinter
    except ImportError:
        return env
    base = Path(sys.base_prefix) / "lib"
    for variable, folder in (
        ("TCL_LIBRARY", f"tcl{tkinter.TclVersion}"),
        ("TK_LIBRARY", f"tk{tkinter.TkVersion}"),
    ):
        if variable not in env and (base / folder).is_dir():
            env[variable] = str(base / folder)
    return env


@contextlib.contextmanager
def open_native_window() -> Iterator[NativeWindow]:
    """Start the helper, wait for its window, and close both on exit."""
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "computeruse.native"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=_environment(),
        )
    except OSError as error:
        raise NativeWindowError("the control window's helper did not start") from error
    try:
        _ready(process)
        yield NativeWindow(process)
    finally:
        _close(process)


def _ready(process: subprocess.Popen[str]) -> int:
    """Wait for the helper's ready line, and return its title bar height."""
    stream = process.stdout
    if stream is None:
        raise NativeWindowError("the control window's helper has no output")
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True).start()
    try:
        first = lines.get(timeout=READY_S)
        message = json.loads(first)
    except (queue.Empty, ValueError):
        raise NativeWindowError("the control window did not open") from None
    ready = message.get("ready") if isinstance(message, dict) else None
    if not isinstance(ready, dict) or not isinstance(ready.get("bar"), int):
        raise NativeWindowError("the control window did not open")
    return ready["bar"]


def _close(process: subprocess.Popen[str]) -> None:
    with contextlib.suppress(OSError, ValueError):
        if process.stdin is not None:
            process.stdin.write(json.dumps({"close": True}) + "\n")
            process.stdin.flush()
            process.stdin.close()
    try:
        process.wait(timeout=CLOSE_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def plain(text: str) -> str:
    r"""Return a note as a person may send it: printable, trimmed, and short.

    Examples
    --------
    >>> plain("  checked\tthe record  ")
    'checked the record'
    """
    kept = "".join(
        " " if ord(character) < 32 or 127 <= ord(character) <= 159 else character
        for character in text
    )
    return kept[:MAX_NOTE].strip()


def words(value: object) -> str:
    """Return an enum value as words.

    Examples
    --------
    >>> words("human_control")
    'human control'
    """
    return str(value or "").replace("_", " ")


def duration(seconds: object) -> str:
    """Return seconds as minutes and seconds.

    Examples
    --------
    >>> duration(75.4)
    '1 min 15 s'
    >>> duration(None)
    '0 s'
    """
    total = max(0, round(seconds)) if isinstance(seconds, (int, float)) else 0
    minutes, rest = divmod(total, 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def message_for(
    status: Mapping[str, Any], command: str, note: str
) -> dict[str, object]:
    """Build the command message for ``status``, as ``order_from`` reads it.

    Examples
    --------
    >>> drawn = {"run": "r1", "revision": 4, "offer": {"intervention": "rq1"}}
    >>> message_for(drawn, "resume", " done ")
    {'command': 'resume', 'run': 'r1', 'intervention': 'rq1', 'revision': 4, \
'note': 'done'}
    >>> message_for({"run": "r1", "revision": 2, "offer": None}, "stop", "x")
    {'command': 'stop', 'run': 'r1', 'intervention': '', 'revision': 2}
    """
    offer = status.get("offer")
    message: dict[str, object] = {
        "command": command,
        "run": status.get("run", ""),
        "intervention": offer.get("intervention", "")
        if isinstance(offer, dict)
        else "",
        "revision": status.get("revision", 0),
    }
    text = plain(note)
    if command in {"resume", "approve", "reject"} and text:
        message["note"] = text
    return message


PURPOSES = {
    "inputs": ("#1d4ed8", "DISCOVERY", "Filling in the task's inputs before the run"),
    "comparison": (
        "#b45309",
        "COMPARISON",
        "Checking the draft with a second member",
    ),
    "outcome_check": (
        "#0e7490",
        "OUTCOME CHECK",
        "A short discovery for a member that does not exist",
    ),
    "checks": ("#b45309", "CHECKS", "Checking the draft before it is saved"),
    "saving": ("#1f8a4c", "SAVING", "Confirming the texts the capability will save"),
    "review": ("#1f8a4c", "REVIEW", "Decide whether replay may run the capability"),
}
"""The banner for a short control that is not the run the person started."""


def banner(mode: object, purpose: object = "") -> tuple[str, str, str]:
    """Return the banner's color, title, and line for a run's mode.

    Examples
    --------
    >>> banner("replay")[1:]
    ('REPLAY', 'The saved capability runs, with no model')
    >>> banner(None)[1:]
    ('WAITING', 'Waiting for the run')
    >>> banner("replay", "comparison")[1]
    'COMPARISON'
    """
    if str(purpose or "") in PURPOSES:
        return PURPOSES[str(purpose)]
    return MODES.get(str(mode or ""), (MUTED, "WAITING", "Waiting for the run"))


def newer(drawn: Mapping[str, Any] | None, status: Mapping[str, Any]) -> bool:
    """Report whether ``status`` may replace the one drawn.

    A late status of the same run never replaces a newer one, so a redraw
    cannot bring back a button a receipt has disabled.

    Examples
    --------
    >>> newer({"run": "r", "revision": 5}, {"run": "r", "revision": 4})
    False
    >>> newer({"run": "r", "revision": 5}, {"run": "s", "revision": 1})
    True
    """
    return (
        drawn is None
        or drawn.get("run") != status.get("run")
        or status.get("revision", 0) >= drawn.get("revision", 0)
    )


OWNERSHIP = {
    "ready": ("#374151", "Press Start to begin"),
    "running": ("#0f766e", "Automation is working"),
    "stopping": ("#b26a00", "Stopping at the next safe point"),
    "paused": ("#b26a00", "Waiting for you"),
    "awaiting_approval": ("#b26a00", "Waiting for your decision"),
    "human_control": ("#1f8a4c", "You have control of the browser"),
    "checking": ("#b26a00", "Checking the session you handed back"),
    "completed": ("#1f8a4c", "Run completed"),
    "failed": ("#b3261e", "Run failed"),
    "terminated": ("#b3261e", "Run terminated"),
}
"""The ownership line's color and words for each state of a run."""

DECLINE = "#4b5563"
"""The main button's color while it would decline a waiting approval."""

REVIEW = "capability_review"
"""The trigger of the request that shows a saved capability for review."""

RESULT = "replay_result"
"""The trigger of the request that shows how a finished replay ended."""

RESULTS = {RESULT: "Replay", "discovery_result": "Discovery"}
"""The triggers that show how a finished run ended, and which run it was."""

CHECKS = "run_checks"
"""The trigger of the request that asks whether saving may run its checks."""

CLOSE = "#374151"
"""The color of the button that closes a finished replay's window."""

ANSWERS = {
    "delivery_uncertain": (
        "Yes, it went through",
        "No, stop the replay",
        "No, stop the replay",
    ),
    "unverified_result": (
        "Yes, the result is correct",
        "No, keep working",
        "No, stop the replay",
    ),
}
"""The buttons for a yes or no question: yes, then no in discovery and replay.

Each still sends approve or resume; only the words name the answer."""


def ownership(state: object) -> tuple[str, str]:
    """Return the color and words saying who holds the session now.

    Examples
    --------
    >>> ownership("human_control")[1]
    'You have control of the browser'
    >>> ownership("something new")[1]
    'Waiting for the run'
    """
    return OWNERSHIP.get(str(state or ""), (MUTED, "Waiting for the run"))


def heading(offer: Mapping[str, Any]) -> tuple[str, str]:
    """Return the request card's color and heading for what a request asks.

    Examples
    --------
    >>> heading({"ask": "approval"})[1]
    'Needs your approval'
    >>> heading({"ask": "person", "trigger": "missing_user_input"})[1]
    'Needs a value from you'
    >>> heading({"ask": "person", "trigger": "no_progress"})[1]
    'Needs your help'
    >>> heading({"ask": "approval", "trigger": "capability_review"})[1]
    'Review the saved capability'
    """
    ask, trigger = offer.get("ask"), offer.get("trigger")
    if trigger in RESULTS:
        return CLOSE, f"{RESULTS[trigger]} result"
    if trigger == CHECKS:
        return "#b45309", "Run the checks?"
    if trigger in ANSWERS:
        return "#b26a00", "Needs your answer"
    if trigger == REVIEW:
        return "#1f8a4c", "Review the saved capability"
    if ask == "approval":
        return "#b26a00", "Needs your approval"
    if ask == "person" and trigger == "missing_user_input":
        return "#1d4ed8", "Needs a value from you"
    if ask == "person":
        return "#b26a00", "Needs your help"
    return MUTED, "Paused by you"


def question(offer: Mapping[str, Any]) -> tuple[str, str]:
    """Return what a request asks, first, and why, second.

    An approval asks about its proposal, so the proposal leads and the
    reason explains it. Any other request asks in its reason.

    Examples
    --------
    >>> question({"ask": "approval", "proposal": "click 'Open'", "reason": "risky"})
    ("click 'Open'", 'risky')
    >>> question({"ask": "person", "proposal": None, "reason": "sign in"})
    ('sign in', '')
    >>> question({"trigger": "capability_review", "reason": "Steps (2)"})[0]
    'Approve it for replay without a model?'
    >>> question({"trigger": "replay_result", "reason": "Replay failed."})
    ('Replay failed.', '')
    >>> steps = "Approve this one operation." + chr(10) + "Step s20: click"
    >>> asked = {"ask": "approval", "proposal": "click Commit", "reason": steps}
    >>> question(asked)[0]
    'Approve this one operation.'
    >>> question(asked)[1].splitlines()
    ['click Commit', 'Step s20: click']
    """
    reason = str(offer.get("reason") or "")
    proposal = str(offer.get("proposal") or "")
    if offer.get("trigger") == REVIEW:
        return "Approve it for replay without a model?", reason
    if offer.get("trigger") in RESULTS:
        first, _, rest = reason.partition("\n")
        return first, rest
    if offer.get("trigger") == CHECKS:
        return "Check the draft before it is saved?", reason
    if "\n" in reason:
        # A replay puts its question on the first line. The proposal and
        # related step follow.
        first, _, rest = reason.partition("\n")
        shown = proposal if proposal and not proposal.startswith("not verified") else ""
        return first, "\n".join(part for part in (shown, rest) if part)
    if offer.get("ask") == "approval" and proposal:
        return proposal, reason
    return reason, ""


def primary_face(status: Mapping[str, Any]) -> tuple[str, str]:
    """Return the main button's words and color.

    While an approval waits, resume declines it, so the button says so and
    loses the color of going ahead. It still sends the same command.

    Examples
    --------
    >>> waiting = {"primary": {"command": "resume", "label": "Resume"},
    ...            "offered": ["approve", "resume", "terminate"]}
    >>> primary_face(waiting)
    ('Decline and resume', '#4b5563')
    >>> primary_face({"primary": {"command": "stop", "label": "Stop"}})
    ('Stop', '#b26a00')
    >>> gone = {"primary": {"command": "resume"}, "offered": ["approve", "resume"],
    ...         "offer": {"ask": "approval", "trigger": "delivery_uncertain"}}
    >>> primary_face(gone)[0]
    'No, stop the replay'
    >>> helping = {"primary": {"command": "resume"}, "offered": ["resume"],
    ...            "offer": {"ask": "person", "trigger": "no_progress"}}
    >>> primary_face(helping)[0]
    'Done, continue'
    """
    primary = status.get("primary") or {}
    command = str(primary.get("command") or "")
    offer = _offer_of(status)
    trigger, ask = offer.get("trigger"), offer.get("ask")
    if command == "resume" and trigger == REVIEW:
        return "Keep it as a draft", DECLINE
    if command == "resume" and trigger in RESULTS:
        return "OK, close", CLOSE
    if command == "resume" and trigger == CHECKS:
        return "Skip the checks", DECLINE
    if command == "resume" and ask == "approval" and trigger in ANSWERS:
        replaying = status.get("mode") == "replay"
        return ANSWERS[trigger][2 if replaying else 1], DECLINE
    if command == "resume" and "approve" in (status.get("offered") or []):
        return "Decline and resume", DECLINE
    if command == "resume" and ask == "person":
        # A person was asked to do a step; resuming says it is done, and the
        # run checks the page before it goes on.
        return "Done, continue", TONES["resume"]
    return str(primary.get("label") or "Waiting"), TONES.get(command, MUTED)


def _offer_of(status: Mapping[str, Any]) -> Mapping[str, Any]:
    offer = status.get("offer")
    return offer if isinstance(offer, dict) else {}


def approve_face(status: Mapping[str, Any]) -> str:
    """Return the approve button's words for the question the window shows.

    Examples
    --------
    >>> approve_face({"offer": {"trigger": "delivery_uncertain"}})
    'Yes, it went through'
    >>> approve_face({"offer": {"trigger": "risky_action"}})
    'Approve once'
    """
    trigger = _offer_of(status).get("trigger")
    if trigger == REVIEW:
        return "Approve for replay"
    if trigger == CHECKS:
        return "Run the checks"
    if trigger in ANSWERS:
        return ANSWERS[trigger][0]
    return "Approve once"


def note_title(status: Mapping[str, Any]) -> str:
    """Return the note box's title, which asks for a value when one is missing.

    Examples
    --------
    >>> note_title({"offer": {"ask": "person", "trigger": "missing_user_input"}})
    'Type your answer, then click Done, continue'
    >>> note_title({"offer": None})
    'Note (optional)'
    """
    offer = status.get("offer")
    if isinstance(offer, dict) and offer.get("trigger") == "missing_user_input":
        return "Type your answer, then click Done, continue"
    return "Note (optional)"


def details(status: Mapping[str, Any]) -> str:
    """Return the run's small print on one line.

    Examples
    --------
    >>> details({"step": 9, "remaining_s": 612, "paused_s": 0, "run": "run-1"})
    'step 9 · 10 min 12 s left · run-1'
    """
    parts = [
        f"step {status.get('step', 0)}",
        f"{duration(status.get('remaining_s'))} left",
    ]
    if status.get("paused_s"):
        parts.append(f"paused {duration(status.get('paused_s'))}")
    parts.append(str(status.get("run", "")))
    return " · ".join(part for part in parts if part)


# The helper process. Nothing below runs in the run's own process.


def _float_above_all(title: str) -> None:  # pragma: no cover  macOS window server
    """Keep the window titled ``title`` above every window, on every Space."""
    import ctypes
    import ctypes.util

    runtime = ctypes.util.find_library("objc")
    appkit = ctypes.util.find_library("AppKit")
    if runtime is None or appkit is None:
        return
    objc = ctypes.cdll.LoadLibrary(runtime)
    ctypes.cdll.LoadLibrary(appkit)
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    send = objc.objc_msgSend
    pointer, count, flag = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_bool
    type Result = type[
        ctypes.c_void_p | ctypes.c_ulong | ctypes.c_bool | ctypes.c_char_p
    ]

    def call(
        target: object, name: str, *args: int, restype: Result | None = None
    ) -> Any:  # noqa: ANN401  the type ``restype`` names
        send.restype = restype
        send.argtypes = [pointer, pointer, *(ctypes.c_long for _ in args)]
        return send(target, objc.sel_registerName(name.encode()), *args)

    accessory, floating = 1, 3
    all_spaces, beside_full_screen = 1, 256
    app = call(
        objc.objc_getClass(b"NSApplication"), "sharedApplication", restype=pointer
    )
    call(app, "setActivationPolicy:", accessory, restype=flag)
    windows = call(app, "windows", restype=pointer)
    for index in range(call(windows, "count", restype=count)):
        window = call(windows, "objectAtIndex:", index, restype=pointer)
        name = call(window, "title", restype=pointer)
        text = call(name, "UTF8String", restype=ctypes.c_char_p) if name else None
        if text is None or text.decode("utf-8", "replace") != title:
            continue
        call(window, "setCollectionBehavior:", all_spaces | beside_full_screen)
        call(window, "setLevel:", floating)
        call(window, "orderFrontRegardless")


BACKGROUND = "#f7f7f8"
OFFER = "#fff7e6"
DISABLED = "#c4c7cc"
INK = "#111827"
FONT = "Helvetica"


class _Button:  # pragma: no cover  drawn by Tk in the helper process
    """A colored label that acts on a click while enabled.

    A label rather than a native button, because macOS draws native buttons
    in its own colors.
    """

    def __init__(self, parent: tk.Misc, text: str, action: Callable[[], None]) -> None:
        import tkinter

        self.label = tkinter.Label(
            parent,
            text=text,
            fg="white",
            bg=MUTED,
            pady=10,
            cursor="hand2",
            font=(FONT, 14, "bold"),
        )
        self.enabled = False
        self.label.bind("<Button-1>", lambda _event: self.enabled and action())

    def set(self, text: str, color: str, *, enabled: bool) -> None:
        self.label.configure(text=text, bg=color if enabled else DISABLED)
        self.enabled = enabled


class _Helper:  # pragma: no cover  drawn by Tk in the helper process
    """The window itself: Tk widgets, the run's messages, and the clicks.

    From the top: the mode banner, who holds the session, the run's small
    print, the open request with its question first, the notices, the note,
    and the buttons, with the one that goes ahead first.
    """

    def __init__(self, write: Callable[[dict[str, object]], None]) -> None:
        import tkinter

        self.write = write
        self.inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.drawn: dict[str, Any] | None = None
        self.busy = False
        self.armed = 0.0
        self.bar = 28
        root = tkinter.Tk()
        self.root = root
        root.title(TITLE)
        root.configure(bg=BACKGROUND)
        root.geometry(f"{WIDTH}x640+40+60")
        root.minsize(320, 360)
        root.attributes("-topmost", True)
        # Only the run closes the window; a person cannot close it by mistake.
        root.protocol("WM_DELETE_WINDOW", lambda: None)
        with contextlib.suppress(tkinter.TclError):
            root.tk.call("console", "hide")
        # Show the run type in a large label.
        self.banner = tkinter.Frame(root, bg=MUTED, padx=16, pady=10)
        self.banner.pack(fill="x")
        self.banner_title = self._text(
            self.banner, "WAITING", size=19, weight="bold", fg="white", bg=MUTED
        )
        self.banner_title.pack(fill="x")
        self.banner_line = self._text(
            self.banner, "Waiting for the run", size=12, fg="white", bg=MUTED
        )
        self.banner_line.pack(fill="x")
        # Show who currently controls the session.
        self.owner = tkinter.Frame(root, bg=MUTED, padx=16, pady=8)
        self.owner.pack(fill="x")
        self.owner_line = self._text(
            self.owner, "Waiting for the run", size=15, weight="bold", fg="white"
        )
        self.owner_line.configure(bg=MUTED)
        self.owner_line.pack(fill="x")
        body = tkinter.Frame(root, bg=BACKGROUND, padx=16, pady=10)
        body.pack(fill="both", expand=True)
        self.details = self._text(body, "", size=13, fg="#374151")
        self.details.pack(fill="x")
        # Draw the open request with a colored edge, heading, and question.
        self.card = tkinter.Frame(body, bg=MUTED)
        inner = tkinter.Frame(self.card, bg=OFFER, padx=12, pady=10)
        inner.pack(fill="both", expand=True, padx=(5, 0))
        top = tkinter.Frame(inner, bg=OFFER)
        top.pack(fill="x")
        self.card_heading = self._text(top, "", size=13, weight="bold", bg=OFFER)
        self.card_heading.pack(side="left")
        self.card_left = self._text(top, "", size=11, fg=MUTED, bg=OFFER)
        self.card_left.pack(side="right")
        self.card_question = self._text(
            inner, "", size=15, weight="bold", fg=INK, bg=OFFER
        )
        self.card_question.pack(fill="x", pady=(6, 2))
        self.card_why = self._text(inner, "", size=12, fg=INK, bg=OFFER)
        self.card_context = self._text(inner, "", size=11, fg=MUTED, bg=OFFER)
        self.card_small = self._text(inner, "", size=10, fg=MUTED, bg=OFFER)
        self.lines = tkinter.Frame(body, bg=BACKGROUND)
        self.lines.pack(fill="x")
        self.fields: dict[str, tk.Label] = {}
        for name in ("notice", "settling", "interrupted", "ending"):
            self.fields[name] = self._text(self.lines, "", size=12, fg="#8a4b00")
        self.note_title = self._text(body, "Note (optional)", size=12, fg=INK)
        self.note_title.pack(fill="x", pady=(10, 2))
        self.note = tkinter.Entry(
            body,
            relief="solid",
            borderwidth=1,
            bg="white",
            fg=INK,
            font=(FONT, 13),
            insertbackground=INK,
            highlightthickness=2,
            highlightbackground=BACKGROUND,
            highlightcolor="#1d4ed8",
        )
        self.note.pack(fill="x", ipady=4)
        self.buttons = tkinter.Frame(body, bg=BACKGROUND, pady=10)
        self.buttons.pack(fill="x")
        self.approve = _Button(
            self.buttons, "Approve once", lambda: self._send("approve")
        )
        self.primary = _Button(self.buttons, "Waiting", self._primary)
        self.terminate = _Button(self.buttons, "Terminate", self._terminate)
        self.primary.label.pack(fill="x", pady=3)
        self.terminate.label.pack(fill="x", pady=(10, 3))
        self.message = self._text(body, "", size=11, fg=MUTED)
        self.message.pack(fill="x")
        self._paint()

    def _text(
        self,
        parent: tk.Misc,
        text: str,
        *,
        size: int,
        weight: str = "normal",
        fg: str = INK,
        bg: str = BACKGROUND,
    ) -> tk.Label:
        import tkinter

        # Text inside the request card has its edge and padding to fit in too.
        inside = bg == OFFER
        return tkinter.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            anchor="w",
            justify="left",
            font=(FONT, size, weight),
            wraplength=WIDTH - (90 if inside else 60),
        )

    def _offered(self, command: str) -> bool:
        return command in ((self.drawn or {}).get("offered") or [])

    def _paint(self) -> None:
        status = self.drawn or {}
        primary = status.get("primary") or {}
        command = primary.get("command")
        text, color = primary_face(status)
        self.primary.set(
            text,
            color,
            enabled=not self.busy
            and bool(command)
            and bool(primary.get("enabled"))
            and self._offered(command),
        )
        # Approve once exists only while the run asks for one approval, and
        # then it comes first, above the button that declines.
        if self._offered("approve"):
            self.approve.set(approve_face(status), APPROVE, enabled=not self.busy)
            self.approve.label.pack(fill="x", pady=3, before=self.primary.label)
        else:
            self.approve.label.pack_forget()
        # A finished replay's window has only the button that closes it.
        offer = status.get("offer")
        if isinstance(offer, dict) and offer.get("trigger") in RESULTS:
            self.terminate.label.pack_forget()
        else:
            self.terminate.label.pack(fill="x", pady=(10, 3))
        confirming = self.armed > time.monotonic()
        self.terminate.set(
            "Click again to terminate" if confirming else "Terminate",
            TERMINATE,
            enabled=not self.busy and self._offered("terminate"),
        )

    def draw(self, status: dict[str, Any]) -> None:
        if not newer(self.drawn, status):
            return
        self.drawn = status
        color, title, line = banner(status.get("mode"), status.get("purpose"))
        self.banner.configure(bg=color)
        self.banner_title.configure(text=title, bg=color)
        self.banner_line.configure(text=line, bg=color)
        tone, words_now = ownership(status.get("state"))
        if status.get("ending") == "stopped":
            tone, words_now = CLOSE, "Stopped before the first change, as planned"
        shown = status.get("offer")
        if isinstance(shown, dict) and shown.get("trigger") in RESULTS:
            tone, words_now = CLOSE, f"{RESULTS[shown['trigger']]} finished"
        self.owner.configure(bg=tone)
        self.owner_line.configure(text=words_now, bg=tone)
        self.details.configure(text=details(status))
        offer = status.get("offer")
        if isinstance(offer, dict):
            edge, title_text = heading(offer)
            asked, why = question(offer)
            review = offer.get("trigger") == REVIEW
            finished = offer.get("trigger") in {*RESULTS, CHECKS}
            context = str(offer.get("context") or "")
            name = "Capability" if status.get("mode") == "replay" else "Goal"
            small = " · ".join(
                part
                for part in (
                    f"request {offer.get('intervention', '')}",
                    words(offer.get("trigger")),
                    str(offer.get("session") or ""),
                )
                if part
            )
            self.card.configure(bg=edge)
            self.card_heading.configure(text=title_text, fg=edge)
            self.card_left.configure(
                text=""
                if finished
                else f"answer within {duration(offer.get('left_s'))}"
            )
            self.card_question.configure(text=asked)
            for label, text in (
                (
                    self.card_why,
                    why
                    if review or finished or "\n" in str(offer.get("reason") or "")
                    else f"Why: {why}"
                    if why
                    else "",
                ),
                (self.card_context, f"{name}: {context}" if context else ""),
                (self.card_small, small),
            ):
                label.configure(text=text)
                if text:
                    label.pack(fill="x", pady=(2, 0))
                else:
                    label.pack_forget()
            self.card.pack(fill="x", before=self.lines, pady=(10, 4))
        else:
            self.card.pack_forget()
        for name in ("notice", "settling", "interrupted", "ending"):
            text = str(status.get(name) or "")
            if name == "ending" and text:
                text = f"ended: {words(text)}"
            self.fields[name].configure(text=text)
            if text:
                self.fields[name].pack(fill="x", pady=(4, 0))
            else:
                self.fields[name].pack_forget()
        asking = note_title(status)
        self.note_title.configure(
            text=asking,
            fg="#1d4ed8" if asking.startswith("Your answer") else INK,
            font=(FONT, 12, "bold" if asking.startswith("Your answer") else "normal"),
        )
        # A finished replay takes no note; its window only closes.
        offer = status.get("offer")
        if isinstance(offer, dict) and offer.get("trigger") in RESULTS:
            self.note_title.pack_forget()
            self.note.pack_forget()
        else:
            self.note_title.pack(fill="x", pady=(10, 2), before=self.buttons)
            self.note.pack(fill="x", ipady=4, before=self.buttons)
        self._paint()

    def _primary(self) -> None:
        primary = (self.drawn or {}).get("primary") or {}
        if primary.get("command"):
            self._send(str(primary["command"]))

    def _terminate(self) -> None:
        if self.armed > time.monotonic():
            self.armed = 0.0
            self._send("terminate")
            return
        self.armed = time.monotonic() + CONFIRM_S
        self.message.configure(text="Click Terminate again within 3 seconds")
        self._paint()
        self.root.after(int(CONFIRM_S * 1000) + 50, self._paint)

    def _send(self, command: str) -> None:
        if self.drawn is None or self.busy:
            return
        self.busy = True
        self._paint()
        self.message.configure(text=f"sending {words(command)}")
        self.write({"command": message_for(self.drawn, command, self.note.get())})

    def receipt(self, receipt: dict[str, Any]) -> None:
        self.busy = False
        status = receipt.get("status")
        if isinstance(status, dict):
            self.draw(status)
        verdict = words(receipt.get("verdict"))
        reason = str(receipt.get("reason") or "")
        self.message.configure(text=f"{verdict}: {reason}" if reason else verdict)
        if receipt.get("verdict") == "accepted":
            self.note.delete(0, "end")
        self._paint()

    def place(self, left: int, top: int, width: int, height: int) -> None:
        # Tk sizes the content; the title bar sits above it inside the frame.
        inner = max(200, height - self.bar)
        self.root.geometry(f"{width}x{inner}+{left}+{top}")

    def pump(self) -> None:
        while True:
            try:
                message = self.inbox.get_nowait()
            except queue.Empty:
                break
            if message is None or message.get("close"):
                self.root.destroy()
                return
            place = message.get("place")
            if isinstance(message.get("status"), dict):
                self.draw(message["status"])
            elif isinstance(message.get("receipt"), dict):
                self.receipt(message["receipt"])
            elif isinstance(place, list) and len(place) == 4:
                self.place(*(int(value) for value in place))
        self.root.after(POLL_MS, self.pump)

    def start(self) -> None:
        root = self.root
        root.update()
        if sys.platform == "darwin":
            _float_above_all(TITLE)
        root.update()
        self.bar = max(0, root.winfo_rooty() - root.winfo_y()) or self.bar
        self.write({"ready": {"bar": self.bar}})
        root.after(POLL_MS, self.pump)
        root.mainloop()


def _listen(  # pragma: no cover  the helper process
    stream: IO[str], inbox: queue.Queue[dict[str, Any] | None]
) -> None:
    for line in stream:
        with contextlib.suppress(ValueError):
            message = json.loads(line)
            if isinstance(message, dict):
                inbox.put(message)
    inbox.put(None)


def main() -> None:  # pragma: no cover  the helper process
    """Run the helper: draw what the run sends, and send what a person clicks."""
    out = sys.stdout

    def write(message: dict[str, object]) -> None:
        out.write(json.dumps(message, ensure_ascii=True) + "\n")
        out.flush()

    helper = _Helper(write)
    reader = threading.Thread(
        target=_listen, args=(sys.stdin, helper.inbox), daemon=True
    )
    reader.start()
    helper.start()


if __name__ == "__main__":  # pragma: no cover
    main()
