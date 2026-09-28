"""The operator's control panel in a separate browser context.

The window is a separate browser context in the session's browser, never an
overlay in the session's pages. Page scripts in the session cannot see it or
call it. The model never observes it and the recorder never records it,
because neither looks outside the session's context. A native dialog in the
session does not block it either, because it runs in another renderer.

The window's one command binding is exposed on the window's page only. Its
handler accepts a call only from that page's main frame, while the page still
shows the document this module wrote. The document loads nothing and runs one
inline script, which a Content-Security-Policy pins by its hash. The script
sets every field with ``textContent``, because the reason and the proposal can
quote page text and model text.

Two execution contexts reach this module. ``show`` runs on the owner thread
and drives the window's page. The binding handler runs inside Playwright's
dispatcher while the owner thread may be inside another driver call. A
Playwright call from the handler would deadlock, so the handler only reads the
message, calls the pure-Python ``Control.submit``, and returns the receipt as
plain data. The window's script draws that receipt at once, so "Stopping..."
appears while the owner thread is still busy in a browser call.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import math
import unicodedata
from collections.abc import Iterator

from playwright.sync_api import Browser, BrowserContext, Error, Page, Route

from computeruse.control import Control, Offer, Order, Status
from computeruse.escalation import Command, Via

BINDING = "computeruseControl"
"""The name the window's script calls to send a command."""

TITLE = "computeruse control"
"""The window's document title, which a headed browser shows on the window."""

WINDOW_WIDTH = 400
"""The window's viewport width in CSS pixels."""

WINDOW_HEIGHT = 640
"""The window's viewport height in CSS pixels."""

RENDER_MS = 2000
"""How long drawing one status may take before the window counts as broken."""

LOAD_MS = 5000
"""How long opening the window's page and writing its document may take."""

MAX_ID = 64
"""The longest run or intervention id a window message may carry."""

MAX_NOTE = 400
"""The longest operator note a window message may carry."""

_KEYS = frozenset({"command", "run", "intervention", "revision", "note"})
_REQUIRED = _KEYS - {"note"}
_COMMANDS = frozenset(command.value for command in Command)
_DRAW = "status => { window.render(status); return true; }"

_SCRIPT = """
(() => {
  "use strict";
  const BINDING = "computeruseControl";
  const NOTED = new Set(["resume", "approve", "reject"]);
  const CONFIRM_MS = 3000;
  const NOTE_LIMIT = 400;
  const TONES = {start: "go", resume: "go", stop: "hold"};
  const field = {};
  for (const node of document.querySelectorAll("[data-field]")) {
    field[node.dataset.field] = node;
  }
  const button = {};
  for (const node of document.querySelectorAll("button[data-command]")) {
    button[node.dataset.command] = node;
  }
  const offerPart = document.querySelector("[data-part=offer]");
  const contextKey = document.querySelector("[data-part=context]");
  const note = document.querySelector("[data-part=note]");
  let drawn = null;
  let busy = false;
  let armed = 0;
  let disarm = 0;
  let deadline = null;

  const words = value => String(value ?? "").replaceAll("_", " ");
  const duration = value => {
    const total = Math.max(0, Math.round(Number(value) || 0));
    const minutes = Math.floor(total / 60);
    const rest = total % 60;
    return minutes ? `${minutes} min ${rest} s` : `${rest} s`;
  };
  const put = (name, text) => {
    field[name].textContent = String(text ?? "");
  };
  const line = (name, text) => {
    put(name, text);
    field[name].hidden = !text;
  };
  const say = text => {
    put("message", text);
    field.message.title = String(text ?? "");
  };
  const offered = command =>
    drawn !== null && Array.isArray(drawn.offered) && drawn.offered.includes(command);
  const plain = text => Array.from(text, character => {
    const code = character.charCodeAt(0);
    return code < 32 || (code >= 127 && code <= 159) ? " " : character;
  }).slice(0, NOTE_LIMIT).join("").trim();

  const countdown = () => {
    if (deadline === null) {
      put("left", "");
      return;
    }
    put("left", duration((deadline - performance.now()) / 1000));
  };

  const buttons = () => {
    const primary = drawn ? drawn.primary : null;
    const main = button.primary;
    main.textContent = primary ? primary.label : "Waiting";
    main.className = ((primary && TONES[primary.command]) || "muted") + " wide";
    main.disabled = busy || !primary || !primary.enabled || !offered(primary.command);
    // Approve once exists only while the run asks for one approval.
    button.approve.hidden = !offered("approve");
    button.approve.disabled = busy || !offered("approve");
    const confirming = armed > performance.now();
    button.terminate.textContent = confirming ? "Confirm terminate" : "Terminate";
    button.terminate.disabled = busy || !offered("terminate");
  };

  const render = status => {
    if (!status || typeof status !== "object") return false;
    if (drawn && drawn.run === status.run && status.revision < drawn.revision) {
      return false;
    }
    drawn = status;
    put("state", words(status.state));
    if (field.activity) {
      put("activity", status.activity || "Waiting for the next action");
    }
    put("run", status.run);
    put("mode", status.mode);
    put("owner", words(status.owner));
    put("step", status.step);
    put("remaining", duration(status.remaining_s));
    put("paused", duration(status.paused_s));
    const offer = typeof status.offer === "object" ? status.offer : null;
    offerPart.hidden = offer === null;
    contextKey.textContent = status.mode === "replay" ? "Capability" : "Goal";
    deadline = null;
    if (offer !== null) {
      put("intervention", offer.intervention);
      put("ask", words(offer.ask));
      put("trigger", offer.trigger ? words(offer.trigger) : "none");
      put("reason", offer.reason);
      put("context", offer.context);
      put("proposal", offer.proposal || "none");
      put("session", offer.session || "No operator can see the session");
      deadline = performance.now() + Math.max(0, Number(offer.left_s) || 0) * 1000;
    }
    countdown();
    line("notice", status.notice);
    line("settling", status.settling);
    line("interrupted", status.interrupted);
    line("ending", status.ending ? `Ended: ${words(status.ending)}` : "");
    buttons();
    return true;
  };
  window.render = render;

  const send = async command => {
    const sender = window[BINDING];
    if (drawn === null || busy || typeof sender !== "function") return;
    const message = {
      command,
      run: drawn.run,
      intervention: drawn.offer ? drawn.offer.intervention : "",
      revision: drawn.revision,
    };
    const text = plain(note.value);
    if (NOTED.has(command) && text) message.note = text;
    busy = true;
    buttons();
    say(`Sending ${words(command)}`);
    let receipt = null;
    try {
      receipt = await sender(message);
    } catch (error) {
      receipt = null;
    }
    busy = false;
    if (!receipt || typeof receipt !== "object") {
      buttons();
      say("The command did not reach the run");
      return;
    }
    render(receipt.status);
    buttons();
    const verdict = words(receipt.verdict);
    say(receipt.reason ? `${verdict}: ${receipt.reason}` : verdict);
    if (receipt.verdict === "accepted" && "note" in message) note.value = "";
  };

  for (const node of Object.values(button)) {
    node.addEventListener("mousedown", event => event.preventDefault());
    node.addEventListener("click", () => node.blur());
  }
  button.primary.addEventListener("click", event => {
    if (!event.isTrusted) return;
    const primary = drawn ? drawn.primary : null;
    if (primary && primary.command) void send(primary.command);
  });
  button.approve.addEventListener("click", event => {
    if (event.isTrusted) void send("approve");
  });
  button.terminate.addEventListener("click", event => {
    if (!event.isTrusted || event.detail > 1) return;
    clearTimeout(disarm);
    if (armed > performance.now()) {
      armed = 0;
      void send("terminate");
      return;
    }
    armed = performance.now() + CONFIRM_MS;
    disarm = setTimeout(() => {
      armed = 0;
      buttons();
    }, CONFIRM_MS);
    buttons();
    say("Click Terminate again within 3 seconds to end the run");
  });
  setInterval(countdown, 500);
  buttons();
})();
"""

_DIGEST = base64.b64encode(hashlib.sha256(_SCRIPT.encode()).digest()).decode("ascii")

_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; "
    f"script-src 'sha256-{_DIGEST}'; base-uri 'none'; form-action 'none'"
)

_STYLE = """
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html, body {
  margin: 0;
  background: #151a22;
  color: #f5f7fb;
  font: 13px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.panel { display: flex; flex-direction: column; gap: 8px; padding: 0 12px 12px; }
.title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0 -12px;
  padding: 10px 12px;
  background: #242b37;
  font-weight: 700;
}
.state {
  font-size: 11px;
  font-weight: 600;
  color: #b8c0ce;
  text-transform: uppercase;
  letter-spacing: .05em;
}
.grid { display: grid; grid-template-columns: 104px 1fr; gap: 3px 8px; margin: 0; }
dt { color: #929dad; }
dd { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wrap {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: 6.5em;
  overflow-y: auto;
}
.offer { border-top: 1px solid #3a4352; padding-top: 8px; }
.line { margin: 0; overflow-wrap: anywhere; color: #d7dce5; }
.notice { color: #facc15; }
.note { display: flex; gap: 8px; align-items: center; color: #929dad; }
.note input {
  flex: 1;
  min-width: 0;
  padding: 5px 7px;
  border: 1px solid #3a4352;
  border-radius: 6px;
  background: #0f1319;
  color: #f5f7fb;
  font: inherit;
}
.buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
button {
  border: 0;
  border-radius: 7px;
  padding: 8px;
  color: #101319;
  font: 700 13px ui-sans-serif, system-ui, sans-serif;
  cursor: pointer;
}
button:disabled { cursor: default; filter: saturate(.35); opacity: .55; }
.wide { grid-column: 1 / -1; }
.go { background: #4ade80; }
.hold { background: #facc15; }
.stop { background: #fb7185; }
.take { background: #60a5fa; }
.muted { background: #9ca3af; }
.message {
  margin: 0;
  min-height: 1.4em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #b8c0ce;
}
"""

_BODY = """
<main class="panel">
<header class="title">
<span>Agent control</span><span class="state" data-field="state">waiting</span>
</header>
<dl class="grid">
<dt>Run</dt><dd data-field="run"></dd>
<dt>Mode</dt><dd data-field="mode"></dd>
<dt>Owner</dt><dd data-field="owner"></dd>
<dt>Step</dt><dd data-field="step"></dd>
<dt>Time left</dt><dd data-field="remaining"></dd>
<dt>Paused for</dt><dd data-field="paused"></dd>
</dl>
<section class="offer" data-part="offer" hidden>
<dl class="grid">
<dt>Request</dt><dd data-field="intervention"></dd>
<dt>Asks for</dt><dd data-field="ask"></dd>
<dt>Trigger</dt><dd data-field="trigger"></dd>
<dt>Reason</dt><dd class="wrap" data-field="reason"></dd>
<dt data-part="context">Goal</dt><dd class="wrap" data-field="context"></dd>
<dt>Proposal</dt><dd class="wrap" data-field="proposal"></dd>
<dt>Session</dt><dd class="wrap" data-field="session"></dd>
<dt>Answer within</dt><dd data-field="left"></dd>
</dl>
</section>
<p class="line notice" data-field="notice" hidden></p>
<p class="line" data-field="settling" hidden></p>
<p class="line" data-field="interrupted" hidden></p>
<p class="line" data-field="ending" hidden></p>
<label class="note">Note
<input data-part="note" type="text" maxlength="400" autocomplete="off"
 spellcheck="false">
</label>
<div class="buttons">
<button type="button" class="muted wide" data-command="primary" disabled
 >Waiting</button>
<button type="button" class="go wide" data-command="approve" hidden disabled
 >Approve once</button>
<button type="button" class="stop wide" data-command="terminate"
 disabled>Terminate</button>
</div>
<p class="message" data-field="message" role="status"></p>
</main>
"""

PANEL_HTML = "".join(
    (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n',
        f'<meta http-equiv="Content-Security-Policy" content="{_POLICY}">\n',
        f"<title>{TITLE}</title>\n<style>{_STYLE}</style>\n</head>\n<body>",
        _BODY,
        f"<script>{_SCRIPT}</script>\n</body>\n</html>\n",
    )
)
"""The window's whole document: markup, style, and its one pinned script."""


def payload(status: Status) -> dict[str, object]:
    """Return ``status`` as plain data the window's script can draw.

    Enums become their values and the offered commands a sorted list, so the
    result survives a JSON round trip unchanged. Text passes through as it
    is, markup included, because the window sets it as text and never parses
    it. Seconds are rounded to tenths.

    Parameters
    ----------
    status
        What the control reports now, or what a receipt carried.

    Returns
    -------
    dict
        Strings, numbers, booleans, lists, None, and nested dicts only.
    """
    primary = status.primary
    return {
        "run": status.run,
        "mode": status.mode.value,
        "purpose": status.purpose,
        "revision": status.revision,
        "state": status.state.value,
        "owner": status.owner.value,
        "step": status.step,
        "primary": {
            "label": primary.label,
            "command": primary.command.value if primary.command is not None else None,
            "enabled": primary.enabled,
        },
        "offered": sorted(command.value for command in status.offered),
        "remaining_s": _seconds(status.remaining_s),
        "paused_s": _seconds(status.paused_s),
        "notice": status.notice,
        "settling": status.settling,
        "interrupted": status.interrupted,
        "ending": status.ending,
        "offer": _offer(status.offer) if status.offer is not None else None,
    }


def _offer(offer: Offer) -> dict[str, object]:
    return {
        "intervention": offer.intervention,
        "ask": offer.ask.value,
        "trigger": offer.trigger.value if offer.trigger is not None else None,
        "reason": offer.reason,
        "context": offer.context,
        "proposal": offer.proposal,
        "route": offer.route,
        "session": offer.session,
        "timeout_s": _seconds(offer.timeout_s),
        "left_s": _seconds(offer.left_s),
    }


def _seconds(value: float) -> float:
    return round(value, 1) if math.isfinite(value) else 0.0


def order_from(message: object) -> Order | None:
    """Build the order a window message asks for, or None if it is malformed.

    A message is a dict with exactly the keys the window sends. ``command``
    is one of the command values. ``run`` and ``intervention`` are strings of
    at most 64 characters. ``revision`` is a non-negative integer and not a
    boolean. ``note`` is absent or a string of at most 400 characters, and an
    empty or blank note means no note. No string may hold a control
    character. Whether the order holds is the control's to judge, so a
    command for another run still becomes an order and is refused there.

    Examples
    --------
    >>> order_from({"command": "stop", "run": "run-1", "intervention": "",
    ...             "revision": 3}).command
    <Command.STOP: 'stop'>
    >>> order_from({"command": "stop", "run": "run-1", "intervention": "",
    ...             "revision": 3, "via": "terminal"}) is None
    True
    """
    if not isinstance(message, dict) or not _REQUIRED <= set(message) <= _KEYS:
        return None
    command = message["command"]
    run = message["run"]
    intervention = message["intervention"]
    revision = message["revision"]
    note = message.get("note", "")
    if not (
        isinstance(command, str)
        and command in _COMMANDS
        and isinstance(run, str)
        and _readable(run, MAX_ID)
        and isinstance(intervention, str)
        and _readable(intervention, MAX_ID)
        and isinstance(note, str)
        and _readable(note, MAX_NOTE)
    ):
        return None
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        return None
    return Order(
        Command(command),
        run,
        revision,
        intervention=intervention,
        via=Via.PANEL,
        note=note.strip() or None,
    )


def _readable(text: str, limit: int) -> bool:
    """Report whether ``text`` fits ``limit`` and holds no control character."""
    return len(text) <= limit and not any(
        unicodedata.category(character) == "Cc" for character in text
    )


class ControlWindow:
    """A control window for one run, in its own browser context.

    It implements the ``Channel`` protocol. Open it with
    ``open_control_window`` or ``ControlWindow.open``, which own the context
    and close it on exit.

    Parameters
    ----------
    context
        A browser context that holds nothing but this window.
    render_ms
        How long drawing one status may take before the window counts as
        broken.
    """

    def __init__(
        self, context: BrowserContext, *, render_ms: float = RENDER_MS
    ) -> None:
        self._context = context
        self._render_ms = render_ms
        self._control: Control | None = None
        self._page: Page | None = None
        self._reopened = False
        self._abandoned = False

    @classmethod
    def open(cls, browser: Browser) -> contextlib.AbstractContextManager[ControlWindow]:
        """Open a control window beside the session in ``browser``.

        The same as ``open_control_window(browser)``.
        """
        return open_control_window(browser)

    @property
    def page(self) -> Page | None:
        """The window's page, or None once the window has given up."""
        return self._page

    def attach(self, control: Control) -> None:
        """Send this control the commands the operator clicks."""
        self._control = control

    def show(self, status: Status) -> None:
        """Draw ``status`` in the window. Owner thread only.

        The window never draws a status older than one it already drew, so
        a late redraw cannot bring back a button a receipt has disabled. A
        window the operator closed, or one that cannot draw, is opened again
        once. After a second failure the window is closed for good and
        ``listening`` reports False. Any other page the operator opened in
        the window's context is closed first.
        """
        if self._abandoned:
            return
        self._close_strays()
        data = payload(status)
        if self._draw(data):
            self._reopened = False
            return
        if self._reopened:
            self._abandon()
            return
        self._reopened = True
        try:
            self._load()
        except Error:
            self._abandon()
            return
        if not self._draw(data):
            self._abandon()

    def listening(self) -> bool:
        """Report whether an operator can use this window.

        A window the operator closed or reloaded still counts, because the
        next draw opens it again; only a window given up on does not.
        """
        return not self._abandoned

    def _load(self) -> None:
        """Open a fresh page, expose the binding on it, and write the document."""
        stale = self._page
        self._page = None
        if stale is not None and not stale.is_closed():
            with contextlib.suppress(Error):
                stale.close()
        page = self._context.new_page()
        self._page = page
        page.expose_binding(BINDING, self._receive)
        page.set_content(PANEL_HTML, timeout=LOAD_MS)

    def _draw(self, data: dict[str, object]) -> bool:
        """Hand ``data`` to the window's script, within ``render_ms``."""
        page = self._page
        if page is None or page.is_closed():
            return False
        try:
            handle = page.wait_for_function(
                _DRAW, arg=data, timeout=self._render_ms, polling=100
            )
            handle.dispose()
        except Error:
            return False
        return True

    def _close_strays(self) -> None:
        for page in self._context.pages:
            if page is not self._page:
                with contextlib.suppress(Error):
                    page.close()

    def _abandon(self) -> None:
        self._abandoned = True
        page = self._page
        self._page = None
        if page is not None and not page.is_closed():
            with contextlib.suppress(Error):
                page.close()

    def _receive(self, source: dict[str, object], *args: object) -> dict[str, object]:
        """Judge one command the window sent. Runs inside Playwright's dispatcher.

        Nothing here calls Playwright. The page, its main frame, and the
        frame's address are read from what the client already holds. A call
        from any other page, from a child frame, or from a document this
        module did not write is refused without reaching the control.
        """
        control = self._control
        page = self._page
        if control is None:
            return _refusal("the window is not attached to a run", None)
        if page is None:
            return _refusal("the window is closed", payload(control.status()))
        frame = page.main_frame
        if (
            source.get("page") is not page
            or source.get("frame") is not frame
            or frame.url != "about:blank"
        ):
            return _refusal(
                "only the control window can send commands", payload(control.status())
            )
        order = order_from(args[0]) if len(args) == 1 else None
        if order is None:
            return _refusal(
                "the window sent a command the run cannot read",
                payload(control.status()),
            )
        receipt = control.submit(order)
        return {
            "verdict": receipt.verdict.value,
            "reason": receipt.reason,
            "status": payload(receipt.status),
        }


def _refusal(reason: str, status: dict[str, object] | None) -> dict[str, object]:
    return {"verdict": "invalid", "reason": reason, "status": status}


@contextlib.contextmanager
def open_control_window(browser: Browser) -> Iterator[ControlWindow]:
    """Open the control window in a new context of ``browser``, and close it on exit.

    The context aborts every request, so the window loads nothing, and a tab
    the operator opens in it cannot reach anything either. The binding is
    exposed on the window's page, never on a context the session uses.

    Parameters
    ----------
    browser
        The browser the session runs in. A headed browser shows the window
        beside the session's own windows.

    Yields
    ------
    ControlWindow
        The window, not yet attached to a control.
    """
    context = browser.new_context(
        viewport={"width": WINDOW_WIDTH, "height": WINDOW_HEIGHT}
    )
    try:
        context.set_default_timeout(LOAD_MS)
        context.set_default_navigation_timeout(LOAD_MS)
        context.route("**/*", _refuse)
        window = ControlWindow(context)
        window._load()
        yield window
    finally:
        with contextlib.suppress(Error):
            context.close()


def _refuse(route: Route) -> None:
    """Abort one request from the window's context, which loads nothing."""
    route.abort("blockedbyclient")
