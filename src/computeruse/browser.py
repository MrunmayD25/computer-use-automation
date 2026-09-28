"""The Playwright adapter: everything that knows what a browser is.

This module is the only one that imports a driver. The loop and gate accept
typed actions, including live screenshot coordinates.

The adapter enforces these boundaries:

- Origin and route. A document request outside the profile's origin or routes
  is aborted in the network layer, before it is sent, so the destination is
  never fetched and never rendered. What the browser does show afterwards is
  its own error page, so the adapter then returns the session to the page it
  was on and reports the action as blocked. A subresource from another origin
  is aborted too, which is also what keeps a screenshot from pulling in
  pixels from a third party.
- Frame scope, from the frame's current state. A frame is read only when its
  own current URL is permitted and every frame above it is permitted too. The
  ``src`` attribute is not consulted, because a frame that has been replaced
  through ``srcdoc`` or a same-document write still carries the attribute it
  was born with. A frame that is not read is covered in every screenshot, and
  a capture whose boundary cannot be established is refused rather than
  taken.
- New windows and downloads. A popup is closed unless ``allow_new_windows``
  is declared, and downloads are refused at the context unless
  ``allow_downloads`` is declared. Both are reported as side effects. A
  permitted popup gets a page id, is listed by ``pages``, and is never made
  active by a decision: it goes to a person. While a window the caller does
  not know about is open, an action that names the windows it knows is
  refused before any input is sent. Route screening, download refusal, dialog
  holding, and secret masking apply to every page the session manages.
- Secret values. A password field, a one-time-code field, and any field this
  run has typed a declared secret into are reported without a value and
  without their ``value`` attribute. It makes that decision before reading
  anything and masks those fields in screenshots. A field that receives a declared
  secret is marked and held by element before anything is typed, and the
  same element receives the value, so renaming the field changes nothing. A
  held field that disappears while its document is still showing refuses
  every observation, because the value may have moved with it.
- Record context and record identity. A target is resolved immediately
  before acting, and the control's context path is compared with the path
  the decision was made against. Where the action carries record evidence,
  the evidence is read again from the page and must still hold the same
  value in the target's row or panel.
- Dialogs. A dialog is held open and reported under an id of its own. It is
  answered only by an action the profile declares, and only when the answer
  names that id, so an approval cannot be spent on a dialog that replaced it.
- A person in the same session. A recorded session reports what a person does
  in it as categories, never values, marks every field a person typed into
  as holding a secret before anything captures the screen again, and refuses
  automation input and capture while the person holds the session. Page
  script never runs while a dialog is open, because it would wait for the
  dialog, and every page call a hand-back makes has a time limit.

The adapter cannot determine whether a click has a business effect. The
operator declares risky effects, and the run can record findings about the
application. For a painted control, the adapter does verify that the target
canvas is on top at the click point. That check does not prove that the
application accepted the click.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import io
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Collection, Coroutine, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Any

from PIL import Image, ImageChops
from playwright._impl._sync_base import mapping
from playwright.sync_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Dialog,
    Download,
    ElementHandle,
    Error,
    FloatRect,
    Frame,
    JSHandle,
    Locator,
    Page,
    Playwright,
    Request,
    Route,
    sync_playwright,
)
from playwright.sync_api import TimeoutError as DriverTimeout

from computeruse import operations, policy, reading, visual
from computeruse._collect import (
    CHANGED,
    CONTEXT,
    CONTROL,
    COVERS,
    EVIDENCE,
    FOCUSED,
    GUARD,
    HELD,
    KNOWN_AS,
    OPERATION,
    PROMPTS,
    PROTECT,
    READ,
    RECORDED,
    RECORDER,
    RESOLVE,
    SCRIPT,
    SUBMISSION,
)
from computeruse.actions import (
    DIALOG_ACTIONS,
    ENTER,
    MAX_OCCURRENCE,
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    Capabilities,
    DomLocator,
    Expectation,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Operation,
    Outcome,
    PageInfo,
    PageState,
    PendingDialog,
    Point,
    Scope,
    ScopeKind,
    ScreenTarget,
    SecretRef,
    SelectOption,
    Target,
    TargetForm,
    VisualAnchor,
    VisualMeta,
    VisualRegion,
    Window,
    describe,
)
from computeruse.control import (
    NOT_SENT,
    SENT_ELSEWHERE,
    Control,
    PageProbe,
    Recording,
    SessionProbe,
    Status,
)
from computeruse.escalation import Owner
from computeruse.manual import Detail, Gap, GapKind, ManualEvent, ManualKind, Widget
from computeruse.operations import BoundOperation
from computeruse.profile import ActionKind, Profile
from computeruse.reading import LocalReader, Reader
from computeruse.surface import SurfaceError

ELEMENTS = (
    "a[href], button, input, select, textarea, canvas, [role], [onclick], "
    "[tabindex], summary, td, th, dd, dt, h1, h2, h3, h4, h5, h6, legend, "
    "label, output, caption, img[alt]"
)
"""The elements a structured observation always looks at.

The collector also describes any element with visible text of its own, and
any top-level editable region, whatever its tag. Everything else is page
furniture.
"""

ATTRIBUTES = (
    "id",
    "name",
    "type",
    "title",
    "placeholder",
    "value",
    "data-value",
    "data-testid",
    "href",
    "aria-label",
    "aria-describedby",
    "aria-labelledby",
)
"""The attributes reported, matching what a DOM locator may match on."""

SECRET_FIELDS = ", ".join(
    (
        'input[type="password"]',
        'input[autocomplete="current-password"]',
        'input[autocomplete="new-password"]',
        'input[autocomplete="one-time-code"]',
    )
)
"""Fields whose value is a credential, by type or by the page's own declaration."""

KEYS = frozenset(
    {
        "Enter",
        "Tab",
        "Escape",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "PageUp",
        "PageDown",
        "Home",
        "End",
    }
)
"""Keys ``press_key`` may send. A key that types a character is a ``type``."""

SETTLE_MS = 50
"""How long an action waits after loading, for the events it caused."""

VIEW = "() => [window.scrollX, window.scrollY, window.innerWidth, window.innerHeight]"
"""Where the page is scrolled to and how much of it is visible."""

ON_TOP = """
(el, point) => {
  const box = el.getBoundingClientRect();
  const x = box.left + point.dx;
  const y = box.top + point.dy;
  return el.ownerDocument.elementFromPoint(x, y) === el;
}
"""
"""Whether this element is the one a click at that offset would reach."""

_ELEMENT = frozenset({TargetForm.ACCESSIBILITY, TargetForm.DOM})
_SCREEN = frozenset({TargetForm.SCREEN})
_SESSION = frozenset({TargetForm.NONE})

SUPPORTED: Capabilities = {
    ActionKind.READ: _ELEMENT | _SCREEN,
    ActionKind.ASSERT: _ELEMENT,
    ActionKind.WAIT_FOR: _ELEMENT,
    ActionKind.NAVIGATE: _SESSION,
    ActionKind.SCROLL: _SESSION | _ELEMENT | _SCREEN,
    ActionKind.CLICK: _ELEMENT | _SCREEN | {TargetForm.VISUAL},
    ActionKind.DOUBLE_CLICK: _SCREEN,
    ActionKind.MOVE: _SCREEN,
    ActionKind.DRAG: _SCREEN,
    ActionKind.WAIT: _SCREEN,
    ActionKind.TYPE: _ELEMENT | _SCREEN,
    ActionKind.SELECT: _ELEMENT,
    ActionKind.PRESS_KEY: _SESSION | _ELEMENT | _SCREEN,
    ActionKind.ACCEPT_DIALOG: _SESSION,
    ActionKind.DISMISS_DIALOG: _SESSION,
}
"""Every action type this adapter performs, with the target forms it takes.

A screen ``read`` hit tests one point of the current screenshot and reads the
element there, which is how a run that only sees pixels can still have its
result checked. Anything absent here is never offered to a decision, and no
entry selects a window.
"""

MARK_ATTRIBUTE = "data-computeruse-protected"
"""The attribute a field carries once a declared secret has been typed into it.

Its value is a token issued per session, so a page cannot claim the mark in
advance. A screenshot mask is a locator, and a locator cannot be built from an
element handle, so the mark is what the mask is placed by.
"""

STOPPED = NOT_SENT
"""The result reported when a stop arrives before input is sent."""

HELD_BY = "a person holds the session, so the run neither reads nor operates it"
"""The result reported while a person holds the session."""

DRAIN_S = 0.5
"""How long a hand-back waits for events a page announced and has not delivered."""

READY_MS = 100
"""How long a probe waits for a page's document to finish loading."""

POLL_MS = 100
"""How often a bounded page call looks again. Every call here answers at once."""

PANEL_WIDTH = 420
"""The control window's width beside a visible session. Chrome may widen it."""

DOCK_SETTLE_MS = 300
"""How long the browser takes to apply one window placement."""

HOLD_S = 0.8
"""How long the highlight shows a target before a click, a key, or typing."""

REFIT_S = 0.5
"""How often a visible session checks whether a person resized its window."""

QUIET_MS = 300
"""How long every readable frame must go without a content change before a look."""

QUIET_LIMIT_MS = 2_500
"""The longest a look waits for a busy page to go quiet before it reads anyway."""

_QUIET = """arg => {
  const now = performance.now();
  let watch = window[arg.key];
  if (!watch) {
    watch = window[arg.key] = {last: now};
    watch.observer = new MutationObserver(() => {
      watch.last = performance.now();
    });
    watch.observer.observe(
      document, {subtree: true, childList: true, characterData: true});
  }
  const age = Math.round(now - watch.last);
  if (arg.stop) {
    watch.observer.disconnect();
    delete window[arg.key];
  }
  return String(age);
}"""
"""Report how long a frame's content has been unchanged, watching from the first call.

Only added, removed, and rewritten content counts. An attribute such as a
blinking cursor's class is not content, so it cannot keep a page busy forever.
"""

MAX_HEARD = 5_000
"""Reported input held until the run collects it. More is counted as a gap."""

UNCHECKED = -1
"""The document number of a frame whose restored values were never checked."""


@dataclasses.dataclass(frozen=True, slots=True)
class BrowserLimits:
    """Bounds on how much one adapter call may read, and for how long.

    An observation with no size bound is a token bill with no size bound, and
    an adapter call with no time bound cannot be interrupted by the loop,
    which checks the clock between calls and not during them.
    """

    max_nodes: int = 200
    max_count: int = 5_000
    max_frames: int = 8
    max_text: int = 160
    max_name: int = 1_000
    max_canvases: int = 4
    max_extract: int = 2000
    operation_ms: int = 5_000
    navigation_ms: int = 15_000
    viewport: tuple[int, int] = (1280, 800)


@dataclasses.dataclass(frozen=True, slots=True)
class _InputGuard:
    """Name a guard without resolving a JavaScript handle behind a dialog."""

    page: str
    frame: Frame
    key: str


@dataclasses.dataclass(frozen=True, slots=True)
class _Anchor:
    """A painted region kept in memory for the length of this session only.

    ``capture`` names the observation that produced the crop. An id is only
    honoured while that observation is the most recent visual one, so an id
    can never come to mean a different control than the picture the model was
    shown beside it.
    """

    frame: tuple[str, ...]
    canvas: int
    crop: bytes
    capture: str
    box: visual.Box


@dataclasses.dataclass(frozen=True, slots=True)
class _Pending:
    """The dialog the page is waiting on, and the id it was reported under."""

    info: PendingDialog
    dialog: Dialog


@dataclasses.dataclass(frozen=True, slots=True)
class WindowRect:
    """A window's or a screen area's position and size, in screen pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Return the x coordinate just past the rectangle's right edge."""
        return self.left + self.width

    @classmethod
    def of(cls, bounds: Mapping[str, Any]) -> WindowRect:
        """Read the bounds a browser reports for a window."""
        return cls(
            int(bounds["left"]),
            int(bounds["top"]),
            int(bounds["width"]),
            int(bounds["height"]),
        )

    def holds(self, other: WindowRect) -> bool:
        """Report whether ``other``'s centre lies inside this area."""
        x = other.left + other.width // 2
        y = other.top + other.height // 2
        return self.left <= x < self.right and self.top <= y < self.top + self.height


def fit(
    application: WindowRect, area: WindowRect, panel: int
) -> tuple[WindowRect, WindowRect]:
    """Place the control window beside ``application`` inside the screen ``area``.

    The size a person gave the application window is kept, and the control
    window goes to its right, or to its left when only that side has room. When
    neither side has room, as after maximizing, the application gets the area's
    width less the control window, so the controls never cover the site.

    Examples
    --------
    >>> area = WindowRect(0, 33, 1512, 892)
    >>> fit(WindowRect(0, 33, 900, 700), area, 500)[1]
    WindowRect(left=900, top=33, width=500, height=700)
    >>> fit(WindowRect(600, 33, 900, 700), area, 500)[1]
    WindowRect(left=100, top=33, width=500, height=700)
    >>> fit(WindowRect(0, 33, 1512, 892), area, 500)
    (WindowRect(left=0, top=33, width=1012, height=892), \
WindowRect(left=1012, top=33, width=500, height=892))
    """
    if area.right - application.right >= panel:
        beside = WindowRect(
            application.right, application.top, panel, application.height
        )
        return application, beside
    if application.left - area.left >= panel:
        beside = WindowRect(
            application.left - panel, application.top, panel, application.height
        )
        return application, beside
    docked = WindowRect(area.left, area.top, area.width - panel, area.height)
    return docked, WindowRect(docked.right, area.top, panel, area.height)


def _window_rect(session: CDPSession, window: int) -> WindowRect:
    return WindowRect.of(
        session.send("Browser.getWindowBounds", {"windowId": window})["bounds"]
    )


def _set_window(session: CDPSession, window: int, **bounds: object) -> None:
    session.send("Browser.setWindowBounds", {"windowId": window, "bounds": bounds})


type Place = Callable[[int, int, int, int], None]
"""Move a native control window to the given position and size."""


@dataclasses.dataclass(slots=True)
class _Dock:
    """State retained from the first fit for later dock adjustments.

    A browser control window is moved through its own ``side`` session. A
    native one has no page, and is moved by ``place``.
    """

    panel: Page | None
    application: CDPSession
    window: int
    side: CDPSession | None
    beside: int
    inset: tuple[int, int]
    panel_width: int
    area: WindowRect
    last: WindowRect
    place: Place | None = None
    checked: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class _Protected:
    """A field a declared secret was typed into, known by the element itself.

    ``generation`` counts the navigations of ``frame`` when the field was
    marked. A handle that stops answering after the frame has navigated
    belonged to a document that is gone. One that stops answering without a
    navigation is a question the adapter cannot answer.

    ``person`` says a person typed into it, rather than the run typing a
    declared secret. Both are held the same way, and neither is let go while
    its document still shows and the field cannot be found there.
    """

    frame: Frame
    handle: ElementHandle
    generation: int
    person: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _Heard:
    """One input the session reported, before it is judged a person's.

    ``at`` is milliseconds since the epoch, the clock both the page and this
    process read. ``foreign`` says the adapter already knows it was not its
    own input, which is true of everything it notices itself, such as a
    navigation while a person holds the session. ``auto`` says the page's
    recorder found it covered by one of the adapter's input claims.
    """

    arrival: int
    at: float
    kind: ManualKind
    page: str
    frame: tuple[str, ...]
    position: tuple[int, ...]
    url: str
    foreign: bool = False
    tag: str = ""
    role: str = ""
    name: str = ""
    secret: bool = False
    editable: bool = False
    detail: Detail = Detail.NONE
    auto: bool = False
    source: Frame | None = None
    seq: int = 0
    dialog: PendingDialog | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _Edit:
    """A field someone typed into, waiting to be marked as holding a secret.

    The page's recorder holds the element under ``seq`` in document ``doc``.
    Only that recorder can find it again, and only while the document shows.
    """

    frame: Frame
    doc: str
    seq: int
    at: float
    page: str
    generation: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Control:
    """One element a structured observation reported, and the id it was given.

    The id holds for this element for as long as its document is showing,
    whatever locator later names it. It is session state, not a locator: a
    later run has no way to find this element by it.
    """

    control_id: str
    frame: Frame
    handle: ElementHandle
    generation: int


class _Hold(StrEnum):
    """The current state of one protected field."""

    ACTIVE = "active"
    UNMARKED = "unmarked"
    LOST = "lost"
    GONE = "gone"


class _Refusal(Exception):  # noqa: N818  an outcome, not an error condition
    """A target-resolution failure that prevents safe input."""

    def __init__(self, outcome: Outcome, detail: str) -> None:
        super().__init__(detail)
        self.outcome = outcome
        self.detail = detail


_DRIVERS = threading.local()
"""The Playwright driver running on each thread, if any."""


@contextlib.contextmanager
def _driver() -> Iterator[Playwright]:
    """Yield this thread's Playwright driver, starting one only if none runs.

    Playwright's sync API allows one driver per thread. A discovery keeps its
    browser open while it saves, and saving opens further sessions for the
    comparison and the outcome cases, so those launch their browsers from the
    driver already running rather than starting a second one.
    """
    running = getattr(_DRIVERS, "current", None)
    if running is not None:
        yield running
        return
    with sync_playwright() as driver:
        _DRIVERS.current = driver
        try:
            yield driver
        finally:
            _DRIVERS.current = None


@contextlib.contextmanager
def open_session(
    profile: Profile,
    entry_url: str,
    *,
    headless: bool = True,
    limits: BrowserLimits | None = None,
    record: bool = False,
) -> Iterator[BrowserSurface]:
    """Open a browser at ``entry_url`` under ``profile`` and yield its surface.

    The context is built with the profile's window and download declarations
    already applied, and with request screening installed, so the session
    cannot be opened outside the policy and then corrected afterwards.

    ``record`` installs the recorder a person's input is reported by, before
    the entry page loads, so the first document already has it. Only a
    recorded session in a window someone can see can be handed to a person.
    """
    bounds = limits or BrowserLimits()
    if policy.route_for(profile, entry_url) is None:
        raise SurfaceError("the entry point is outside the profile")
    with _driver() as driver:
        browser = driver.chromium.launch(headless=headless)
        try:
            yield from _session(
                browser, profile, entry_url, bounds, visible=not headless, record=record
            )
        finally:
            with contextlib.suppress(Error):
                browser.close()


def _session(
    browser: Browser,
    profile: Profile,
    entry_url: str,
    limits: BrowserLimits,
    *,
    visible: bool,
    record: bool,
) -> Iterator[BrowserSurface]:
    scope = profile.scope
    context = browser.new_context(
        viewport={"width": limits.viewport[0], "height": limits.viewport[1]},
        device_scale_factor=1,
        accept_downloads=scope.allow_downloads,
    )
    context.set_default_timeout(limits.operation_ms)
    context.set_default_navigation_timeout(limits.navigation_ms)
    page = context.new_page()
    surface = BrowserSurface(page, profile, limits, visible=visible, record=record)
    surface.install(context)
    try:
        page.goto(entry_url, wait_until="domcontentloaded")
        yield surface
    finally:
        with contextlib.suppress(Error):
            context.close()


@dataclasses.dataclass(frozen=True, slots=True)
class _ScreenCapture:
    capture_id: str
    image: bytes
    view: tuple[int, ...]
    focus: tuple[ElementHandle, tuple[str, ...]] | None
    documents: tuple[tuple[Frame, int], ...]


class BrowserSurface:
    """Live Chromium pages, observed and operated under one profile.

    One page is active at a time. Observation and action use only the active
    page. A popup the profile permits is managed and listed, and no decision
    can make it active. A page that closes while active hands the session back
    to the page that was active when it opened.

    The surface is also the seat a person takes the session at, as the
    ``Seat`` protocol in ``control`` describes. The same pages and the same
    context are handed over and back; nothing is reopened.
    """

    __slots__ = (
        "_active",
        "_anchors",
        "_arrivals",
        "_binding",
        "_browser_scope",
        "_capture",
        "_changed",
        "_claims",
        "_context",
        "_controls",
        "_counts",
        "_dialogs",
        "_dock",
        "_docs",
        "_documents",
        "_edits",
        "_effects_from",
        "_ended",
        "_generations",
        "_guards",
        "_heard",
        "_human",
        "_input_capture",
        "_issued",
        "_limits",
        "_marker",
        "_marking",
        "_moved",
        "_numbered",
        "_opened",
        "_openers",
        "_orphans",
        "_overflow",
        "_owner",
        "_page",
        "_pages",
        "_pending",
        "_permitted",
        "_profile",
        "_protected",
        "_reader",
        "_record",
        "_sequence",
        "_session",
        "_side_effects",
        "_timeline",
        "_token",
        "_ui_token",
        "_unwatched",
        "_visible",
    )

    def __init__(
        self,
        page: Page,
        profile: Profile,
        limits: BrowserLimits,
        *,
        visible: bool = False,
        record: bool = False,
    ) -> None:
        self._page = page
        self._pages: dict[str, Page] = {"page-1": page}
        self._openers: dict[str, str] = {}
        self._opened = 1
        self._active = "page-1"
        self._pending: dict[str, _Pending] = {}
        self._profile = profile
        self._browser_scope = profile.scope
        self._limits = limits
        self._anchors: dict[str, _Anchor] = {}
        self._controls: dict[str, _Control] = {}
        self._documents: dict[tuple[Frame, int], int] = {}
        self._issued = 0
        self._protected: list[_Protected] = []
        # Fields this run typed into, selected in, or sent a key to. A field
        # here cannot prove a required state while its document still shows.
        self._changed: list[_Protected] = []
        self._generations: dict[Frame, int] = {}
        self._token = secrets.token_hex(8)
        self._side_effects: list[str] = []
        self._sequence = 0
        self._input_capture: _ScreenCapture | None = None
        self._dialogs = 0
        self._capture = ""
        self._context = page.context
        self._visible = visible
        self._dock: _Dock | None = None
        self._record = record
        # Text recognition, for a canvas that holds no text a script can read.
        self._reader: Reader | None = None
        self._ended = False
        self._human = False
        self._owner = Owner.AUTOMATION
        self._timeline: collections.deque[tuple[float, Owner]] = collections.deque(
            maxlen=64
        )
        self._effects_from = 0
        self._permitted: Callable[[], bool] | None = None
        self._claims = 0
        self._guards: list[_InputGuard] = []
        self._binding = "__cu_" + secrets.token_hex(8)
        self._marker = "__cu_" + secrets.token_hex(8)
        self._session = secrets.token_hex(16)
        self._heard: list[_Heard] = []
        self._arrivals = 0
        self._numbered = 0
        self._overflow = 0
        self._counts: dict[str, list[int]] = {}
        self._docs: dict[Frame, str] = {}
        self._edits: list[_Edit] = []
        self._orphans: list[_Edit] = []
        # Frames a person typed in, with the document last checked for
        # restored values; UNCHECKED means never checked.
        self._moved: dict[Frame, int] = {}
        self._unwatched: dict[Frame, int] = {}
        # A docked session shows the target it is about to act on.
        self._marking = False
        self._ui_token = secrets.token_hex(12)

    def preview(self, action: Action, *, waiting: bool = False) -> None:
        """Color a resolved target without changing its style or taking input.

        The highlight is one fixed box the pointer passes through, marked as
        the operator's own, so no look reads it and no screenshot shows it.
        Blue marks the target automation acts on next. Amber marks a target
        waiting for approval.
        """
        if not self._marking or self._blocked() or self._human:
            return
        try:
            box = None
            if isinstance(action.target, ScreenTarget):
                point = action.target.point
                if point is not None:
                    box = {
                        "x": point.x - 15,
                        "y": point.y - 15,
                        "width": 30,
                        "height": 30,
                    }
            elif isinstance(action.target, VisualAnchor):
                anchor = self._anchors.get(action.target.anchor_id)
                if anchor is not None and anchor.capture == self._capture:
                    canvas = (
                        self._frame_for(anchor.frame)
                        .locator("canvas")
                        .nth(anchor.canvas)
                    )
                    canvas_box = canvas.bounding_box()
                    if canvas_box is not None:
                        # The display uses the last captured rectangle. Only
                        # the gated action may capture pixels to match again.
                        box = {
                            "x": max(0, canvas_box["x"]) + anchor.box.left,
                            "y": max(0, canvas_box["y"]) + anchor.box.top,
                            "width": anchor.box.width,
                            "height": anchor.box.height,
                        }
            elif action.target is not None:
                box = self._locate(action.target).bounding_box()
            self._page.evaluate(
                """({mark, box, waiting}) => {
              const nodes = Array.from(
                document.querySelectorAll('[data-computeruse-ui]'));
              let glow = nodes.find(el =>
                el.getAttribute('data-computeruse-ui') === mark &&
                el.getAttribute('data-highlight') === 'true');
              if (!box) { if (glow) glow.remove(); return; }
              if (!glow) {
                glow = document.createElement('div');
                glow.setAttribute('data-computeruse-ui', mark);
                glow.setAttribute('data-highlight', 'true');
                glow.setAttribute('aria-hidden', 'true');
                document.documentElement.append(glow);
              }
              glow.style.cssText = 'position:fixed!important;' +
                'pointer-events:none!important;' +
                'z-index:2147483646!important;box-sizing:border-box!important;' +
                `left:${box.x - 6}px!important;top:${box.y - 6}px!important;` +
                `width:${box.width + 12}px!important;` +
                `height:${box.height + 12}px!important;` +
                'border-radius:6px!important;' +
                `border:4px solid ${waiting ? '#e99e12' : '#2683ff'}!important;` +
                // A white ring inside and a dark ring outside keep the border
                // visible on light, dark, and blue backgrounds alike.
                'box-shadow:inset 0 0 0 2px #ffffff,0 0 0 2px #ffffff,' +
                '0 0 0 5px rgba(17,24,39,0.55)!important;' +
                `background:${waiting ? '#e99e1226' : '#2683ff26'}!important;`;
              for (const host of nodes) {
                if (host === glow ||
                    host.getAttribute('data-computeruse-ui') !== mark) continue;
                const r = host.getBoundingClientRect();
                if (box.x < r.right && box.x + box.width > r.left &&
                    box.y < r.bottom && box.y + box.height > r.top) {
                  const left = box.x > innerWidth / 2;
                  host.style.setProperty('left', left ? '12px' : 'auto', 'important');
                  host.style.setProperty('right', left ? 'auto' : '12px', 'important');
                }
              }
            }""",
                {"mark": self._ui_token, "box": box, "waiting": waiting},
            )
        except (Error, _Refusal):
            return

    def _picture_style(self) -> str:
        return (
            f'[data-computeruse-ui="{self._ui_token}"] '
            "{ visibility: hidden !important; }"
        )

    def _preview_action(self, action: Action) -> None:
        if not self._marking:
            return
        self.preview(action)
        if action.kind in {
            ActionKind.CLICK,
            ActionKind.TYPE,
            ActionKind.PRESS_KEY,
            ActionKind.SELECT,
        }:
            self.idle(HOLD_S)

    def dock(self, panel: Page) -> None:
        """Fit the application and the control panel side by side on the screen.

        Nothing is added to the application's pages. The application gets a
        narrower window, so every site lays itself out for that width by its
        own code, as it would if a person resized the browser. The panel is a
        separate window flush against its right edge, so it can never cover
        the application or take its clicks.

        The screen's usable area is read by maximizing the window once. Chrome
        keeps a minimum window width, so the panel is placed first and the
        application gets exactly the width the panel leaves. What the fit
        measured is kept, so ``_refit`` can follow a person who later resizes
        or moves the window. A browser that cannot place windows leaves both
        where they are.
        """
        if not self._visible:
            return
        with contextlib.suppress(Error, KeyError):
            application = self._context.new_cdp_session(self._page)
            window = application.send("Browser.getWindowForTarget")["windowId"]
            side = panel.context.new_cdp_session(panel)
            beside = side.send("Browser.getWindowForTarget")["windowId"]
            area = self._screen_area(application, window, self._page)
            # The window is the viewport plus the browser's own toolbar.
            self._page.set_viewport_size({"width": 800, "height": 600})
            self._page.wait_for_timeout(DOCK_SETTLE_MS)
            probe = _window_rect(application, window)
            inset = (probe.width - 800, probe.height - 600)
            panel.set_viewport_size(
                {"width": PANEL_WIDTH - inset[0], "height": area.height - inset[1]}
            )
            self._page.wait_for_timeout(DOCK_SETTLE_MS)
            # Chrome may keep the panel wider than asked; the site gets the rest.
            panel_width = _window_rect(side, beside).width
            dock = _Dock(
                panel,
                application,
                window,
                side,
                beside,
                inset,
                panel_width,
                area,
                probe,
            )
            # The first fit fills the screen: the probe size was only a measure.
            placed, beside_rect = fit(area, area, panel_width)
            self._arrange(dock, placed, beside_rect)
            self._dock = dock
            self._marking = True

    def dock_native(self, place: Place, width: int) -> None:
        """Fit the application and a native control window side by side.

        The same fit as ``dock``, for a control window outside the browser.
        Nothing is added to the application's pages but the target highlight.
        The native window takes exactly ``width``, so no measure of it is
        needed, and ``place`` moves it wherever ``fit`` puts it.
        """
        if not self._visible:
            return
        with contextlib.suppress(Error, KeyError):
            application = self._context.new_cdp_session(self._page)
            window = application.send("Browser.getWindowForTarget")["windowId"]
            area = self._screen_area(application, window, self._page)
            self._page.set_viewport_size({"width": 800, "height": 600})
            self._page.wait_for_timeout(DOCK_SETTLE_MS)
            probe = _window_rect(application, window)
            inset = (probe.width - 800, probe.height - 600)
            dock = _Dock(
                None, application, window, None, 0, inset, width, area, probe, place
            )
            placed, beside = fit(area, area, width)
            self._arrange(dock, placed, beside)
            self._dock = dock
            self._marking = True

    def _screen_area(self, session: CDPSession, window: int, page: Page) -> WindowRect:
        """Read the usable area of ``window``'s screen by maximizing it once."""
        before = _window_rect(session, window)
        _set_window(session, window, windowState="maximized")
        page.wait_for_timeout(DOCK_SETTLE_MS)
        area = _window_rect(session, window)
        _set_window(session, window, windowState="normal")
        page.wait_for_timeout(DOCK_SETTLE_MS)
        _set_window(
            session,
            window,
            left=before.left,
            top=before.top,
            width=before.width,
            height=before.height,
        )
        return area

    def _arrange(self, dock: _Dock, application: WindowRect, panel: WindowRect) -> None:
        """Size the application's page to its window and put the panel beside it."""
        if dock.place is not None:
            dock.place(panel.left, panel.top, panel.width, panel.height)
        elif dock.side is not None and dock.panel is not None:
            _set_window(dock.side, dock.beside, windowState="normal")
            dock.panel.set_viewport_size(
                {
                    "width": panel.width - dock.inset[0],
                    "height": panel.height - dock.inset[1],
                }
            )
            _set_window(dock.side, dock.beside, left=panel.left, top=panel.top)
        _set_window(dock.application, dock.window, windowState="normal")
        # A headed page follows its viewport, so this also sizes the window.
        self._page.set_viewport_size(
            {
                "width": application.width - dock.inset[0],
                "height": application.height - dock.inset[1],
            }
        )
        _set_window(
            dock.application, dock.window, left=application.left, top=application.top
        )
        self._page.wait_for_timeout(DOCK_SETTLE_MS)
        dock.last = _window_rect(dock.application, dock.window)

    def _refit(self) -> None:
        """Follow a person who resized, moved, or maximized the application window.

        The page's size is set by the viewport, not by the window, so it does
        not notice a resize by itself. Between steps, at most every
        ``REFIT_S``, the window's bounds are read again. When they changed,
        the page is sized to the new window and the panel moves beside it.
        A window moved to another screen has that screen's area read through
        the panel window, so the application window keeps the size the person
        gave it. Only window placement is used; nothing runs in the
        application's pages. The next look sees the new layout, and a capture
        taken before the change is refused as stale.
        """
        dock = self._dock
        now = time.monotonic()
        if dock is None or now - dock.checked < REFIT_S or self._blocked():
            return
        dock.checked = now
        with contextlib.suppress(Error, KeyError):
            window = dock.application.send(
                "Browser.getWindowBounds", {"windowId": dock.window}
            )["bounds"]
            if window.get("windowState") == "minimized":
                return
            current = WindowRect.of(window)
            if current == dock.last:
                return
            if not dock.area.holds(current):
                if dock.side is not None and dock.panel is not None:
                    _set_window(
                        dock.side, dock.beside, left=current.left, top=current.top
                    )
                    dock.area = self._screen_area(dock.side, dock.beside, dock.panel)
                else:
                    # Maximizing reads the new screen; the window keeps its size.
                    dock.area = self._screen_area(
                        dock.application, dock.window, self._page
                    )
            placed, beside = fit(current, dock.area, dock.panel_width)
            self._arrange(dock, placed, beside)

    @property
    def browser(self) -> Browser:
        """The browser this session runs in, for a window beside the session.

        A control window opens its own context in it, so nothing it shows is
        part of the session the run observes and the recorder reports.
        """
        browser = self._context.browser
        if browser is None:
            raise SurfaceError("the session has no browser")
        return browser

    def install(self, context: BrowserContext) -> None:
        """Screen requests, popups, downloads, and dialogs for this session.

        Requests are screened on the context, so every page it opens is
        covered from its first request. Dialogs, downloads, navigations, and
        closing are watched per page, from the moment the page is announced,
        which is before the page can deliver any event of its own.

        Dialogs are held on the context rather than per page. A popup can open
        a dialog while it loads, before a listener on the popup itself exists,
        and the driver dismisses a dialog nobody listens for.

        The handlers are wrapped rather than passed as bound methods, because
        the driver stores bookkeeping on whatever object it is handed and this
        one declares its attributes.

        A recorded session also gets the recorder in every frame, before any
        page script runs, and the binding it reports through. The binding's
        name is random per session and the recorder removes it from the page.
        """

        def screen(route: Route, request: Request) -> None:
            self._screen(route, request)

        def popup(page: Page) -> None:
            self._popup(page)

        def dialog(shown: Dialog) -> None:
            self._handle_dialog(self._page_id(shown.page), shown)

        def ended(_: BrowserContext) -> None:
            self._ended = True

        def heard(source: Mapping[str, Any], *payload: object) -> None:
            self._hear(source, payload)

        self._context = context
        context.route("**/*", screen)
        context.on("page", popup)
        context.on("dialog", dialog)
        context.on("close", ended)
        if self._record:
            context.expose_binding(self._binding, heard)
            options = {
                "maxText": self._limits.max_text,
                "maxName": self._limits.max_name,
                "secretMark": self._mark(),
                "controlMark": self._ui_token,
                "binding": self._binding,
                "marker": self._marker,
                "token": self._session,
                "probe": f"computeruse-{secrets.token_hex(8)}",
            }
            context.add_init_script(script=f"({RECORDER})({json.dumps(options)});")
        self._manage("page-1", self._page)

    def _manage(self, page_id: str, page: Page) -> None:
        def closed_dialog(shown: Dialog) -> None:
            self._dialog_closed(page_id, shown)

        def download(started: Download) -> None:
            self._download(started)

        def navigated(frame: Frame) -> None:
            self._generations[frame] = self._generations.get(frame, 0) + 1
            if self._owner is not Owner.AUTOMATION:
                self._note(ManualKind.NAVIGATION, page_id, frame, frame.url)

        def closed(_: Page) -> None:
            self._closed(page_id)

        page.on("dialogclosed", closed_dialog)
        page.on("download", download)
        page.on("framenavigated", navigated)
        page.on("close", closed)

    def _page_id(self, page: Page | None) -> str:
        """Return the id of a managed page, or empty for any other page."""
        for page_id, managed in self._pages.items():
            if managed is page:
                return page_id
        return ""

    def location(self) -> str:
        """Return the active page's URL from session metadata, reading no content."""
        if self._page.is_closed():
            raise SurfaceError("the page is closed")
        return self._page.url

    def capabilities(self) -> Capabilities:
        """Return what this adapter performs, as ``SUPPORTED`` lists it."""
        return SUPPORTED

    def pages(self) -> tuple[PageInfo, ...]:
        """Return every managed page, with its route and any waiting dialog."""
        return self._page_list()

    # Operator control of the session. Everything here runs on the
    # thread that owns the driver. The recorder's binding and the page and
    # context handlers run inside the driver's own calls, so they only add to
    # the buffers below and never call the driver.

    def where(self) -> str:
        """Return where an operator can find the session, or "" if unavailable.

        A session nobody can see cannot be handed over, and neither can one
        that is not recorded, because nothing would protect what a person
        typed into it. Live text only.
        """
        if not self._visible or not self._record:
            return ""
        return f"the Chromium window the run opened ({self._active} is active)"

    def idle(self, seconds: float) -> None:
        """Deliver session events for about ``seconds``, running no page script.

        The wait is on the context, which a person cannot close from the
        window, so it keeps delivering the recorder's reports and a control
        window's commands after every page of the session has closed.
        """
        if self._ended:
            raise SurfaceError("the browser session closed")
        self._refit()
        try:
            self._context.wait_for_event("close", timeout=max(1, int(seconds * 1000)))
        except DriverTimeout:
            return
        except Error as error:
            raise SurfaceError("the browser session failed") from error
        raise SurfaceError("the browser session closed")

    def watch(self, permitted: Callable[[], bool]) -> None:
        """Ask ``permitted`` immediately before any input is sent."""
        self._permitted = permitted

    def transfer(self, owner: Owner, intervention: str) -> None:
        """Note that ``owner`` now holds the session.

        A person taking the session gets it with nothing the automation held
        still usable: every capture and screen target is revoked, and a
        guard left in a page by an input a dialog interrupted is removed, so
        it cannot cancel what the person does. The window is brought to the
        front. Handing it back to the automation drops side effects noticed
        since it let go, because they belong to the person's activity, which
        is recorded separately, not to the automation's next action.
        """
        del intervention
        was = self._owner
        if was is Owner.AUTOMATION and owner is not Owner.AUTOMATION:
            self._effects_from = len(self._side_effects)
        if owner is Owner.AUTOMATION and was is not Owner.AUTOMATION:
            del self._side_effects[self._effects_from :]
        self._owner = owner
        self._timeline.append((time.time(), owner))
        self._human = owner is Owner.HUMAN
        if was is Owner.AUTOMATION and owner is not Owner.AUTOMATION and self._record:
            self._forget_focus()
        if self._human:
            self._revoke()
            self._unguard()
            with contextlib.suppress(Error):
                self._page.bring_to_front()

    def _forget_focus(self) -> None:
        """Let each recorder report the next edit to any field, whoever typed.

        A recorder reports one edit per field per focus. A person who takes
        the session and keeps typing where the automation just typed would
        otherwise go unreported, and unprotected.
        """
        blocked = self._blocked()
        for page_id, page in list(self._pages.items()):
            if page.is_closed() or page_id in blocked:
                continue
            for frame in self._frame_paths(page):
                with contextlib.suppress(Error):
                    self._ask(frame, RECORDED, {"marker": self._marker, "op": "forget"})

    def activity(self) -> tuple[ManualEvent, ...]:
        """Return person input reported since the last call, in order.

        Input one of the adapter's own claims covered is its own and is left
        out. Everything else is a person's, whoever owned the session.
        """
        heard, self._heard = self._heard, []
        heard.sort(key=lambda item: (item.at, item.arrival))
        kept = [
            item for item in heard if item.foreign or not self._own(item.at, item.auto)
        ]
        identities = self._identities(kept)
        found: list[ManualEvent] = []
        for item in kept:
            self._numbered += 1
            operation, present = identities.get(item.arrival, (None, None))
            event = self._manual(item, self._numbered)
            if item.dialog is not None:
                event = dataclasses.replace(
                    event,
                    detail=self._answered(item.page, item.dialog),
                    dialog=item.dialog.dialog_id,
                )
            found.append(
                dataclasses.replace(event, operation=operation, present=present)
            )
        return tuple(found)

    def _answered(self, page_id: str, dialog: PendingDialog) -> Detail:
        """Return how a person answered ``dialog``, or report it as unknown.

        The page function that opened the dialog returns the answer, and the
        recorder keeps it with the dialog's kind and message. A dialog no page
        function opened, a frame whose recorder cannot answer, or a page a
        dialog still stops leaves the answer unknown, never guessed.
        """
        page = self._pages.get(page_id)
        if page is None or page.is_closed() or page_id in self._blocked():
            return Detail.NONE
        request = {
            "marker": self._marker,
            "op": "answered",
            "kind": dialog.kind,
            "message": dialog.message,
            "limit": self._limits.max_text,
        }
        for frame in self._frame_paths(page):
            try:
                answer = self._ask(frame, RECORDED, request)
            except Error:
                continue
            if answer == "accepted":
                return Detail.ACCEPTED
            if answer == "dismissed":
                return Detail.DISMISSED
        return Detail.NONE

    def _identities(
        self, items: Sequence[_Heard]
    ) -> dict[int, tuple[Operation | None, frozenset[str] | None]]:
        """Describe each control that received a person's input as observation would.

        Each frame's recorder says which of the controls this adapter issued
        ids to each input reached, and which native submission it could
        perform, so a restriction bound to an operation applies to a person's
        input as it would to the automation's, under whatever name the
        control shows now. A frame a dialog stops is not asked, and its
        inputs keep no identity, which the classifier treats as unresolved.
        """
        found: dict[int, tuple[Operation | None, frozenset[str] | None]] = {}
        waiting: dict[Frame, list[_Heard]] = {}
        for item in items:
            if item.kind in _OPERATED and item.source is not None and item.seq:
                waiting.setdefault(item.source, []).append(item)
        blocked = self._blocked()
        for frame, group in waiting.items():
            if frame.is_detached() or self._page_id(frame.page) in blocked:
                continue
            known = self._known(frame)
            request = {
                "marker": self._marker,
                "op": "identify",
                "seqs": [item.seq for item in group],
                "known": [control.handle for control in known],
            }
            try:
                answer = json.loads(self._ask(frame, RECORDED, request))
            except (Error, ValueError):
                continue
            ids = [control.control_id for control in known]
            present = frozenset(ids[at] for at in answer["present"])
            for item, identity in zip(group, answer["found"], strict=False):
                operation = (
                    _operation_of(item, identity, ids, self._profile)
                    if identity
                    else None
                )
                found[item.arrival] = (operation, present)
        return found

    def hand_back(self, *, revoke: bool = True) -> Recording:
        """Collect the recording and protect every field a person typed into.

        No page script runs on a page with a dialog open, or on a page that
        shares one with it through an opener, and every other page call has
        a time limit. Each frame's recorder is asked for the last event it
        sent, and the driver is given up to half a second to deliver any it
        has not; the rest are counted as a gap. A recorder that cannot be
        asked has already announced its last event in its own reports.

        A field that cannot be marked while it is still on the page, or that
        left the page while its document still shows, is counted as
        unprotected, on this hand-back and on every later one, until the field
        is back and marked or its document is gone. Nothing lets it go
        sooner, a person's word included. A frame with no working recorder is
        reported, and is read and captured as a frame outside the profile
        until it navigates. ``revoke`` false keeps the captures the automation
        held, for an approval nobody changed the screen for.
        """
        self._human = False
        if not self._record:
            if revoke:
                self._revoke()
            return Recording(self.activity(), (Gap(GapKind.NOT_RECORDED),))
        started = time.monotonic()
        reports = self._reports()
        gaps = self._drain(started)
        unprotected = self._protect_edits(self._blocked())
        unprotected += len(self._stranded())
        missing = [frame for frame, report in reports.items() if report == "missing"]
        self._unwatched = {
            frame: generation
            for frame, generation in self._unwatched.items()
            if not frame.is_detached()
        }
        for frame in missing:
            self._unwatched[frame] = self._generation(frame)
        if missing:
            gaps.append(Gap(GapKind.FRAME_UNWATCHED, len(missing)))
        if self._overflow:
            gaps.append(Gap(GapKind.EVENTS_DROPPED, self._overflow))
            self._overflow = 0
        if revoke:
            self._revoke()
        return Recording(self.activity(), tuple(gaps), unprotected)

    def probe(self) -> SessionProbe:
        """Report every open page, its dialog, readiness, and credential prompts.

        A page with a dialog open, and any page sharing it, is reported from
        what the adapter already knows. Readiness is asked of the driver, not
        the page. Credential prompts are counted by selector and visibility,
        and nothing is read from them.
        """
        pages: list[PageProbe] = []
        prompts = 0
        for page_id, page in list(self._pages.items()):
            if page.is_closed():
                continue
            pages.append(
                PageProbe(
                    page_id, page.url, page_id in self._pending, self._ready(page)
                )
            )
            if page_id not in self._blocked():
                prompts += self._prompts(page)
        return SessionProbe(tuple(pages), prompts, self._lost())

    def _ready(self, page: Page) -> bool:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=READY_MS)
        except Error:
            return False
        return True

    def _prompts(self, page: Page) -> int:
        """Count the visible credential fields in the page's permitted frames."""
        found = 0
        for _path, frame in self._frames(page)[0]:
            try:
                answer = self._ask(frame, PROMPTS, SECRET_FIELDS)
            except Error:
                continue
            count = answer.removeprefix("prompts ")
            found += int(count) if count.isdigit() else 0
        return found

    def _lost(self) -> bool:
        """Report whether a field holding a secret can no longer be accounted for."""
        if self._blocked():
            return bool(self._stranded())
        try:
            self._protections()
        except _Refusal:
            return True
        return False

    def _blocked(self) -> frozenset[str]:
        """Return the pages a dialog stops, which page script must not touch.

        A dialog stops its own page and every page that shares its process
        with it, which includes a popup it opened and the page that opened it.
        """
        held = set(self._pending)
        grown = bool(held)
        while grown:
            grown = False
            for page_id, opener in self._openers.items():
                if (page_id in held) != (opener in held):
                    held.update((page_id, opener))
                    grown = True
        return frozenset(held)

    def _revoke(self) -> None:
        """Forget every capture and screen target the automation held."""
        self._input_capture = None
        self._anchors = {}
        self._capture = ""

    def _unguard(self) -> None:
        """Remove every in-page guard still waiting on a page no dialog stops.

        A guard a dialog interrupted stays until its own time limit, because
        removing it would run page script on a page that cannot answer.
        """
        blocked = self._blocked()
        for guard in list(self._guards):
            if guard.page not in blocked:
                self._drop(guard)

    @contextlib.contextmanager
    def _sending(self, guard: _InputGuard | None = None) -> Iterator[None]:
        """Send input only if the run still permits it.

        An operation can wait seconds for its target. A stop accepted during
        that wait, or a person's input heard during it, is honoured here, at
        the last moment, and nothing is sent.
        """
        if self._permitted is not None and not self._permitted():
            if guard is not None:
                self._drop(guard)
            raise _Refusal(Outcome.BLOCKED, STOPPED)
        yield

    @contextlib.contextmanager
    def _claimed(
        self,
        target: ElementHandle | None,
        kinds: Collection[str],
        *,
        anywhere: bool = False,
        limits: Mapping[str, int] | None = None,
        value: str | None = None,
    ) -> Iterator[None]:
        """Tell the recorders which input the next call sends, and to what.

        Only an event a claim covers, by its kind and its element, counts as
        the adapter's own; the time it arrived proves nothing, because a
        person's input can land in the same millisecond. ``target`` is the
        element the call sends to. None claims the focused element in every
        frame of the active page, or with ``anywhere`` the whole document,
        which a scroll or a drag needs because its events reach whatever
        element lies under the pointer. ``limits`` caps how many keys or
        edits the call itself sends, and ``value`` is the value a fill
        leaves in a text field. Input past either, on the claimed element, is
        reported as uncertain and the field is marked.

        A frame whose recorder cannot take the claim reports the call's input
        as a person's, which stops the run rather than hiding anything. A
        claim left in a page a dialog stops lapses after twice the operation
        limit, and a person given the session clears every claim.
        """
        if not self._record:
            yield
            return
        if target is not None:
            owner = target.owner_frame()
            frames = [owner] if owner is not None else []
        else:
            frames = list(self._frame_paths(self._page))
        self._claims += 1
        claim = f"claim-{self._claims}"
        request = {
            "marker": self._marker,
            "op": "claim",
            "id": claim,
            "target": target,
            "anywhere": anywhere,
            "kinds": list(kinds),
            "limits": dict(limits or {}),
            "value": value,
            "lifetime": self._limits.operation_ms * 2,
        }
        placed: list[Frame] = []
        for frame in frames:
            if frame.is_detached():
                continue
            with contextlib.suppress(Error):
                self._ask(frame, RECORDED, request)
                placed.append(frame)
        try:
            yield
        finally:
            blocked = self._blocked()
            for frame in placed:
                if frame.is_detached() or self._page_id(frame.page) in blocked:
                    continue
                with contextlib.suppress(Error):
                    self._ask(
                        frame,
                        RECORDED,
                        {"marker": self._marker, "op": "release", "id": claim},
                    )

    def _own(self, at: float, auto: bool) -> bool:
        """Report whether a page event was the adapter's own input.

        It was only when a recorder found it covered by one of the adapter's
        claims and the automation held the session when it arrived. A page
        script can claim input too, so a claim never covers what arrives
        while a person or the operator holds the session.
        """
        return auto and self._owner_at(at / 1000) is Owner.AUTOMATION

    def _owner_at(self, at: float) -> Owner:
        """Return who held the session at ``at``, in seconds since the epoch."""
        for stamp, owner in reversed(self._timeline):
            if stamp <= at:
                return owner
        return Owner.AUTOMATION

    # What the recorder and the session report.

    def _hear(self, source: Mapping[str, Any], payload: tuple[object, ...]) -> None:
        """Keep one report from a page's recorder, if it is one.

        This runs inside the driver, so it only checks and appends. A report
        without the session's token, from a page the run does not manage, or
        from a frame the profile does not permit is dropped unread.
        """
        report = _parse(payload, self._session, self._limits.max_name)
        page_id = self._page_id(source.get("page"))
        frame = source.get("frame")
        if report is None or not page_id or not isinstance(frame, Frame):
            return
        if not self._frame_permitted(frame):
            return
        counts = self._counts.setdefault(report.doc, [0, 0])
        counts[0] = max(counts[0], report.seq)
        if report.kind is None:
            if report.hello:
                self._docs[frame] = report.doc
            return
        counts[1] += 1
        own = self._own(report.at, report.auto)
        if report.kind is ManualKind.EDIT and report.editable and not own:
            generation = self._generation(frame)
            self._edits.append(
                _Edit(frame, report.doc, report.seq, report.at, page_id, generation)
            )
        path, position = self._place(page_id, frame)
        self._keep(
            _Heard(
                0,
                report.at,
                report.kind,
                page_id,
                path,
                position,
                report.url,
                tag=report.tag,
                role=report.role,
                name=report.name,
                secret=report.secret,
                editable=report.editable,
                detail=report.detail,
                auto=report.auto,
                source=frame,
                seq=report.seq,
            )
        )

    def _note(
        self,
        kind: ManualKind,
        page_id: str,
        frame: Frame | None,
        url: str,
        detail: Detail = Detail.NONE,
        dialog: PendingDialog | None = None,
    ) -> None:
        """Keep a session event that did not come from adapter input."""
        path, position = self._place(page_id, frame)
        self._keep(
            _Heard(
                0,
                time.time() * 1000,
                kind,
                page_id,
                path,
                position,
                url[:2000],
                foreign=True,
                detail=detail,
                dialog=dialog,
                source=frame,
            )
        )

    def _keep(self, item: _Heard) -> None:
        """Append one report in arrival order, or count it if the buffer is full."""
        if len(self._heard) >= MAX_HEARD:
            self._overflow += 1
            return
        self._arrivals += 1
        self._heard.append(dataclasses.replace(item, arrival=self._arrivals))

    def _place(
        self, page_id: str, frame: Frame | None
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Return a frame's named path and its position among its siblings."""
        page = self._pages.get(page_id)
        if frame is None or page is None or frame.is_detached():
            return (), ()
        path = self._frame_paths(page).get(frame, ())
        position: list[int] = []
        current = frame
        while current.parent_frame is not None:
            parent = current.parent_frame
            siblings = [
                child for child in parent.child_frames if not child.is_detached()
            ]
            position.append(siblings.index(current) if current in siblings else -1)
            current = parent
        position.reverse()
        return path, tuple(position)

    def _manual(self, item: _Heard, sequence: int) -> ManualEvent:
        """Convert one report into a run event with only safe identifiers."""
        # The recorder leaves the name empty for anything that could be named
        # after what was typed into it, so a name here is a label.
        named = item.role and item.name and not item.secret
        control = (
            Widget.of(item.role, item.tag)
            if item.tag
            else _PAGE_KINDS.get(item.kind, Widget.OTHER)
        )
        return ManualEvent(
            sequence=sequence,
            at=item.at / 1000,
            kind=item.kind,
            owned=self._owner_at(item.at / 1000) is Owner.HUMAN,
            page=item.page,
            route=policy.route_for(self._profile, item.url) or "",
            frame=item.frame,
            position=item.position,
            control=control,
            target=AxLocator(item.role, item.name, frame=item.frame) if named else None,
            secret=item.secret,
            detail=item.detail,
            location=item.url,
        )

    # Marking the fields a person typed into.

    def _ask(self, frame: Frame, expression: str, arg: object) -> str:
        """Run a page function that answers with a string, within the time limit."""

        async def read() -> str:
            answer = await frame._impl_obj.wait_for_function(
                expression,
                arg=mapping.to_impl(arg),
                polling=POLL_MS,
                timeout=self._limits.operation_ms,
            )
            return str(await answer.json_value())

        return self._bounded(frame, read())

    def _quiet(self) -> None:
        """Wait until every readable frame stops changing, within a bound.

        The load event of a single-page application fired long before a
        search filters its list or a grid fills in. A look taken 50 ms after
        typing reads the page as it was, so each look first waits for
        ``QUIET_MS`` without a content change in any permitted frame, and
        never longer than ``QUIET_LIMIT_MS``. A frame that navigates or a
        dialog that opens ends the wait early; the look then reports what it
        can.
        """
        frames = [frame for _path, frame in self._frames()[0]]
        key = self._marker + "_quiet"
        deadline = time.monotonic() + QUIET_LIMIT_MS / 1000
        try:
            while not self._blocked():
                ages = [
                    int(self._ask(frame, _QUIET, {"key": key, "stop": False}))
                    for frame in frames
                ]
                if min(ages, default=QUIET_MS) >= QUIET_MS:
                    return
                if time.monotonic() >= deadline:
                    return
                self._page.wait_for_timeout(POLL_MS)
        except Error:
            return
        finally:
            for frame in frames:
                with contextlib.suppress(Error):
                    if not self._blocked() and not frame.is_detached():
                        self._ask(frame, _QUIET, {"key": key, "stop": True})

    def _bounded[T](self, frame: Frame, operation: Coroutine[Any, Any, T]) -> T:
        """Bound driver setup and result transfer as well as page evaluation.

        A dialog can block the renderer before Playwright installs its page
        timeout. Use its existing event loop so dialog callbacks still arrive.
        This small dependency on the driver's sync bridge is kept here.
        """

        async def bounded() -> T:
            task = asyncio.create_task(operation)
            done, _ = await asyncio.wait(
                {task}, timeout=self._limits.operation_ms / 1000
            )
            if task not in done:
                # The driver can wait for a dialog even while cancelling.
                # These calls only inspect or clean up; input never runs here.
                task.cancel()
                task.add_done_callback(
                    lambda finished: (
                        None if finished.cancelled() else finished.exception()
                    )
                )
                raise DriverTimeout(
                    "the page did not answer within the operation limit"
                )
            return task.result()

        return frame._sync(bounded())

    def _reports(self) -> dict[Frame, str]:
        """Ask each permitted frame's recorder which document it is and what it sent.

        A frame with no recorder, or with one that cannot run because the
        frame may not run script, is reported as ``missing``. A frame that
        could not be asked is left out.
        """
        reports: dict[Frame, str] = {}
        for page_id, page in list(self._pages.items()):
            if page.is_closed() or page_id in self._blocked():
                continue
            for frame in self._frame_paths(page):
                if not self._route_permitted(frame) or page_id in self._blocked():
                    continue
                try:
                    answer = self._ask(
                        frame, RECORDED, {"marker": self._marker, "op": "report"}
                    )
                except Error:
                    continue
                parts = answer.split(" ")
                if parts[0] == "report" and len(parts) == 3 and parts[2].isdigit():
                    self._docs[frame] = parts[1]
                    counts = self._counts.setdefault(parts[1], [0, 0])
                    counts[0] = max(counts[0], int(parts[2]))
                    reports[frame] = "watched"
                else:
                    reports[frame] = "missing"
        return reports

    def _drain(self, started: float) -> list[Gap]:
        """Wait a moment for reports a page sent and the driver has not delivered."""
        while self._outstanding() and time.monotonic() - started < DRAIN_S:
            with contextlib.suppress(Error):
                self._context.wait_for_event("close", timeout=50)
        missing = self._outstanding()
        for counts in self._counts.values():
            counts[1] = max(counts[1], counts[0])
        # A document no longer shown in any frame has nothing left to deliver.
        self._docs = {
            frame: doc for frame, doc in self._docs.items() if not frame.is_detached()
        }
        showing = set(self._docs.values())
        self._counts = {
            doc: counts for doc, counts in self._counts.items() if doc in showing
        }
        return [Gap(GapKind.NOT_DRAINED, missing)] if missing else []

    def _outstanding(self) -> int:
        return sum(max(0, sent - received) for sent, received in self._counts.values())

    def _protect_edits(self, blocked: Collection[str] = ()) -> int:
        """Mark every field a person typed into, and count those still unmarked.

        The recorder finds each field again by the number it reported it
        under. A field that is gone went with its document if the document
        navigated, and then nothing is owed. Gone without a navigation, the
        value may have moved into whatever replaced it, so the field is kept
        against capture until the document goes. Nothing a person says lets
        it go sooner, because nothing can show where the value went. The
        adapter's own edits never reach this list.
        """
        # Reports continue to arrive in a fresh list while the page is queried.
        pending, self._edits = self._edits, []
        waiting: list[_Edit] = []
        for edit in pending:
            if edit.frame.is_detached():
                continue
            if edit.page in blocked or edit.page in self._blocked():
                waiting.append(edit)
                continue
            try:
                answer = self._mark_edit(edit)
            except Error:
                if self._generation(edit.frame) == edit.generation:
                    waiting.append(edit)
                else:
                    self._moved[edit.frame] = UNCHECKED
                continue
            if answer == "failed":
                waiting.append(edit)
            elif answer == "marked":
                # The value stays with the browser's history of this frame,
                # and Back can bring it into a later document.
                self._moved.setdefault(edit.frame, self._generation(edit.frame))
            elif answer == "moved":
                # The document is gone, but Back can restore its value in a
                # later document.
                self._moved[edit.frame] = UNCHECKED
            elif answer == "gone":
                self._orphans.append(edit)
                if self._docs.get(edit.frame, edit.doc) != edit.doc:
                    self._moved[edit.frame] = UNCHECKED
        self._edits = waiting + self._edits
        return len(waiting) + self._mark_changed(blocked)

    def _mark_changed(self, blocked: Collection[str]) -> int:
        """Mark changed fields in frames whose person-typed document went.

        A person who typed into a field and then left its document can bring
        the value back with the browser's Back button, into a new document the
        recorder never saw them type in. That can happen in a later takeover
        too, so a frame stays listed for the whole session. On each hand-back
        a frame showing a document not yet checked has every field whose
        value is not the one its page set marked. Returns the frames that
        could not be asked, which count as unprotected.
        """
        failed = 0
        for frame, checked in list(self._moved.items()):
            if frame.is_detached():
                del self._moved[frame]
                continue
            if checked == self._generation(frame):
                continue
            if self._page_id(frame.page) in blocked:
                failed += 1
                continue
            try:
                found = frame.wait_for_function(
                    CHANGED,
                    arg=self._mark(),
                    polling=POLL_MS,
                    timeout=self._limits.operation_ms,
                )
            except Error:
                failed += 1
                continue
            generation = self._generation(frame)
            self._moved[frame] = generation
            for item in found.get_properties().values():
                element = item.as_element()
                if element is not None:
                    self._protected.append(
                        _Protected(frame, element, generation, person=True)
                    )
        return failed

    def _mark_edit(self, edit: _Edit) -> str:
        """Ask the recorder to mark one edited field, and hold it if it did."""
        found = edit.frame.wait_for_function(
            RECORDED,
            arg={
                "marker": self._marker,
                "op": "mark",
                "doc": edit.doc,
                "seq": edit.seq,
                "mark": self._mark(),
            },
            polling=POLL_MS,
            timeout=self._limits.operation_ms,
        )
        element = found.as_element()
        if element is None:
            return str(found.json_value())
        self._protected.append(
            _Protected(edit.frame, element, self._generation(edit.frame), person=True)
        )
        return "marked"

    def _stranded(self) -> list[_Edit]:
        """Return person-typed fields that left a document that still shows."""
        self._orphans = [
            edit
            for edit in self._orphans
            if not edit.frame.is_detached()
            and self._docs.get(edit.frame, edit.doc) == edit.doc
        ]
        return self._orphans

    @property
    def _dialog(self) -> _Pending | None:
        """Return the dialog waiting in the active page, if there is one."""
        return self._pending.get(self._active)

    def _dialog_elsewhere(self) -> str:
        """Return the id of another page with a dialog waiting, or empty.

        A dialog stops the scripts of every page that shares its process,
        which includes a same-origin popup and its opener. Reading or
        operating such a page would wait on a script that cannot run until
        the dialog is answered, so a waiting dialog anywhere is answered first.
        """
        for page_id in self._pending:
            if page_id != self._active:
                return page_id
        return ""

    def _page_state(self, signature: tuple[str, ...] = ()) -> PageState:
        return PageState(self._page.url, signature, self._active)

    def _page_list(self) -> tuple[PageInfo, ...]:
        """Describe every managed page, for a decision to choose between."""
        listed: list[PageInfo] = []
        for page_id, page in self._pages.items():
            pending = self._pending.get(page_id)
            route = None
            if not page.is_closed():
                route = policy.route_for(self._profile, page.url)
            listed.append(
                PageInfo(
                    page_id=page_id,
                    active=page_id == self._active,
                    route=route or "",
                    dialog=pending.info.dialog_id if pending is not None else "",
                    location=page.url if route else "",
                )
            )
        return tuple(listed)

    def _activate(self, page_id: str) -> None:
        """Make ``page_id`` the page every observation and action uses."""
        self._active = page_id
        self._page = self._pages[page_id]
        self._input_capture = None
        self._anchors = {}
        self._capture = ""

    def _closed(self, page_id: str) -> None:
        """Forget a closed page, and hand an active one's session back."""
        if self._owner is not Owner.AUTOMATION and page_id in self._pages:
            self._note(ManualKind.PAGE_CLOSED, page_id, None, self._pages[page_id].url)
        self._pages.pop(page_id, None)
        self._pending.pop(page_id, None)
        if page_id != self._active:
            return
        fallback = self._openers.get(page_id, "")
        if fallback not in self._pages:
            fallback = next(iter(self._pages), "")
        if not fallback:
            return
        self._activate(fallback)
        self._side_effects.append(f"{page_id} closed; {fallback} is active again")

    def observe(self, request: ObservationRequest) -> Observation:
        """Collect one observation with the requested tool."""
        self._refit()
        if self._human:
            return self._held(request)
        self._sequence += 1
        self._input_capture = None
        # The driver's child-frame list can lag behind DOMContentLoaded.
        # Bound readiness by time, then gate each frame's actual URL. This
        # never grants a loading frame permission from its src attribute.
        if not self._blocked():
            with contextlib.suppress(Error):
                self._page.wait_for_load_state("load", timeout=READY_MS * 5)
            self._quiet()
        if self._dialog is not None:
            return self._dialog_observation(request)
        elsewhere = self._dialog_elsewhere()
        if elsewhere:
            return Observation(
                observation_id=f"obs-{self._sequence}",
                mode=request.mode,
                status=ObservationStatus.UNAVAILABLE,
                page_state=self._page_state(),
                notes=(
                    (
                        f"{elsewhere} has a dialog waiting, which stops every page "
                        "of this session; a person must answer it or close "
                        f"{elsewhere}"
                    ),
                ),
                pages=self._page_list(),
            )
        try:
            if request.mode is ObservationMode.STRUCTURED:
                return self._read_structure(request)
            return self._take_picture()
        except Error as error:
            # A frame may navigate during any collection or masking operation.
            if self._page.is_closed():
                raise SurfaceError("the page closed during capture") from error
            return Observation(
                observation_id=f"obs-{self._sequence}",
                mode=request.mode,
                status=ObservationStatus.FAILED,
                page_state=self._page_state(),
                notes=("the surface changed during capture; look again",),
                pages=self._page_list(),
            )

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        """Resolve ``action``'s target against the live page and perform it."""
        if self._human:
            return ActionResult(Outcome.BLOCKED, self._page_state(), detail=HELD_BY)
        if self._page.is_closed():
            raise SurfaceError("the page is closed")
        if expect is not None and expect.page_state.location != self._page.url:
            return self._result(Outcome.STALE, detail="the page moved before acting")
        if (
            expect is not None
            and expect.page_state.page
            and expect.page_state.page != self._active
        ):
            return self._result(
                Outcome.STALE, detail="another page became active before acting"
            )
        unknown = self._unknown_window(expect)
        if unknown:
            return self._result(
                Outcome.STALE,
                detail=f"{unknown} opened before acting; it goes to a person",
            )
        if self._dialog is not None and action.kind not in DIALOG_ACTIONS:
            return self._result(
                Outcome.NOT_ACTIONABLE,
                detail="a dialog is waiting for a decision",
            )
        elsewhere = self._dialog_elsewhere()
        if elsewhere:
            return self._result(
                Outcome.NOT_ACTIONABLE,
                detail=f"{elsewhere} has a dialog waiting; a person must answer it",
            )
        before = self._page.url
        try:
            self._preview_action(action)
            result = self._perform(action, expect)
        except _Refusal as refusal:
            result = self._result(refusal.outcome, detail=refusal.detail)
        except Error as error:
            if self._page.is_closed():
                raise SurfaceError("the page closed during an action") from error
            result = self._dialog_interrupted(error)
        finally:
            # Reading changes nothing on the page, so the capture it checked
            # stays usable. Every other action ends that capture.
            if action.kind is not ActionKind.READ:
                self._input_capture = None
        return self._recover(result, before)

    def _held(self, request: ObservationRequest) -> Observation:
        """Report the screen as unreadable while a person holds the session.

        Nothing is captured. The control already refuses automation while a
        person holds the session; this is the same refusal one layer down.
        """
        self._sequence += 1
        return Observation(
            observation_id=f"obs-{self._sequence}",
            mode=request.mode,
            status=ObservationStatus.UNAVAILABLE,
            page_state=self._page_state(),
            notes=(HELD_BY,),
            pages=self._page_list(),
        )

    def _dialog_interrupted(self, error: Error) -> ActionResult:
        """Report that a dialog interrupted an action.

        A page that opens a dialog stops answering, so the driver call the
        action was in times out. The click did reach the control, which is why
        the dialog is there, so the outcome is not a surface failure. What did
        not happen is whatever the application does after the dialog is
        answered. The result carries the dialog, so the loop knows the
        operation is waiting on it rather than done.
        """
        if self._dialog is None:
            return self._result(Outcome.SURFACE_ERROR, detail=_brief(error))
        self._side_effects.append("a dialog opened and is waiting for a decision")
        return self._result(
            Outcome.OK,
            detail="the action opened a dialog and stopped there",
        )

    def _recover(self, result: ActionResult, before: str) -> ActionResult:
        """Return the session to the last permitted page after a refusal.

        Aborting a document request stops the destination being fetched, and
        leaves the browser showing its own error page. Staying there would end
        the run on a screen that is outside the profile because the profile
        worked, so the session goes back to where it was and the action is
        reported as blocked.
        """
        if policy.route_for(self._profile, self._page.url) is not None:
            return result
        if policy.route_for(self._profile, before) is None:
            return result
        with contextlib.suppress(Error):
            self._page.goto(before, wait_until="domcontentloaded")
        return dataclasses.replace(
            result,
            outcome=Outcome.BLOCKED if result.outcome is Outcome.OK else result.outcome,
            detail=SENT_ELSEWHERE if result.outcome is Outcome.OK else result.detail,
            page_state=self._page_state(),
            side_effects=(
                *result.side_effects,
                "the session was returned to the last permitted page",
            ),
        )

    # Policy enforcement at the session boundary.

    def _screen(self, route: Route, request: Request) -> None:
        url = request.url
        if request.resource_type == "document":
            if url != "about:blank" and policy.route_for(self._profile, url) is None:
                self._side_effects.append(
                    "a navigation outside the profile was blocked"
                )
                if self._owner is not Owner.AUTOMATION:
                    self._refused(request)
                route.abort()
                return
        elif not self._same_origin(url):
            self._side_effects.append("an off-origin subresource was blocked")
            route.abort()
            return
        route.continue_()

    def _refused(self, request: Request) -> None:
        """Note a document request a person made that the profile refused."""
        try:
            frame = request.frame
        except Error:
            return
        page_id = self._page_id(frame.page)
        self._note(ManualKind.NAVIGATION_REFUSED, page_id, frame, frame.url)

    def _popup(self, page: Page) -> None:
        """Close a new window, or take it under management with an id of its own.

        A permitted window is not made active, and nothing here makes it so.
        The run keeps the page it was on and hands the new one to a person,
        so a popup never receives input the automation sent it.
        """
        if not self._browser_scope.allow_new_windows:
            self._side_effects.append("a new window was closed")
            if self._owner is not Owner.AUTOMATION:
                self._note(
                    ManualKind.PAGE_CLOSED, "", None, "", Detail.CLOSED_BY_POLICY
                )
            with contextlib.suppress(Error):
                page.close()
            return
        self._opened += 1
        page_id = f"page-{self._opened}"
        self._pages[page_id] = page
        self._openers[page_id] = self._active
        self._manage(page_id, page)
        if self._owner is not Owner.AUTOMATION:
            self._note(ManualKind.PAGE_OPENED, page_id, None, page.url)
        self._side_effects.append(
            f"a new window opened as {page_id}; it goes to a person"
        )

    def _download(self, download: Download) -> None:
        if self._browser_scope.allow_downloads:
            return
        self._side_effects.append("a download was cancelled")
        with contextlib.suppress(Error):
            download.cancel()

    def _handle_dialog(self, owner: str, dialog: Dialog) -> None:
        """Hold the dialog open and report it under a new id.

        Dismissing it here would take a decision the operator never declared,
        and it would take it before anybody saw the message. The page stops
        answering while a dialog is open, which is why every other call checks
        for one first rather than blocking on a page that cannot reply.

        The id is never reused in the session. Two dialogs with the same
        message are still two decisions, and an answer names exactly one. A
        dialog is held for the page that opened it, so a popup's dialog is
        never answered from another page, and is listed with its page.
        A dialog from a page outside the profile is dismissed, because that
        page can never be selected to answer it, and the dialog would stop
        every page that shares its process.
        """
        page = self._pages.get(owner)
        if page is None or policy.route_for(self._profile, page.url) is None:
            self._side_effects.append(
                f"a dialog in {owner or 'a page the run does not manage'}, which "
                "is outside the profile, was dismissed"
            )
            with contextlib.suppress(Error):
                dialog.dismiss()
            return
        self._dialogs += 1
        self._pending[owner] = _Pending(
            PendingDialog(
                dialog_id=f"dialog-{self._dialogs}",
                kind=dialog.type,
                message=dialog.message[: self._limits.max_text],
            ),
            dialog,
        )

    def _dialog_closed(self, page_id: str, dialog: Dialog) -> None:
        """Note a dialog that closed without this adapter answering it.

        The adapter forgets a dialog before it answers one, so a dialog that
        closes while still held was answered by someone else, whoever owned
        the session. The event does not say which button they chose; the
        answer is asked of the page's recorder later, off the driver's
        callback, and stays unknown when the recorder cannot say.
        """
        pending = self._pending.get(page_id)
        if pending is None or pending.dialog is not dialog:
            return
        del self._pending[page_id]
        page = self._pages.get(page_id)
        self._note(
            ManualKind.DIALOG,
            page_id,
            None,
            page.url if page else "",
            dialog=pending.info,
        )

    def _dialog_info(self) -> PendingDialog | None:
        return None if self._dialog is None else self._dialog.info

    def _dialog_observation(self, request: ObservationRequest) -> Observation:
        """Report the open dialog without reading the page behind it.

        The status describes the page body, which is not read. The dialog is
        reported beside it, and it is enough to decide how to answer.
        """
        return Observation(
            observation_id=f"obs-{self._sequence}",
            mode=request.mode,
            status=ObservationStatus.UNAVAILABLE,
            page_state=self._page_state(),
            dialog=self._dialog_info(),
            notes=("a dialog is open, so the page behind it cannot be read",),
            pages=self._page_list(),
        )

    def _answer_dialog(self, accept: bool, expect: Expectation | None) -> ActionResult:
        """Answer the dialog the decision named, and only if it is still open.

        The check and the answer happen here, at the last moment. A dialog
        that was answered by someone else and replaced by another while a
        person approved the answer has a new id, and the approval does not
        transfer to it.
        """
        pending = self._dialog
        if pending is None:
            return self._result(
                Outcome.NOT_ACTIONABLE, detail="no dialog is waiting for a decision"
            )
        if expect is None or expect.dialog is None:
            return self._result(
                Outcome.NOT_ACTIONABLE,
                detail="an answer must name the dialog it was decided about",
            )
        if expect.dialog != pending.info.dialog_id:
            return self._result(
                Outcome.STALE,
                detail="the dialog waiting now is not the one the decision was about",
            )
        try:
            with self._sending():
                self._pending.pop(self._active, None)
                if accept:
                    pending.dialog.accept()
                else:
                    pending.dialog.dismiss()
        except Error as error:
            return self._result(Outcome.SURFACE_ERROR, detail=_brief(error))
        self._settle()
        return self._result(Outcome.OK)

    def _same_origin(self, url: str) -> bool:
        return self._browser_scope.contains(url) or url.startswith(("about:", "data:"))

    # Structured observation.

    def _read_structure(self, request: ObservationRequest) -> Observation:
        """Collect a window over the permitted frames, or refuse before reading.

        The protected fields are settled first. If one cannot be accounted
        for, nothing is collected at all, because the collector could only
        withhold a value from an element it has been told about.

        The window is counted across frames in order. Each frame reports how
        many visible candidates it holds, so a later request can start where
        this one stopped, and nothing is scrolled to find them.
        """
        try:
            protected = self._protections()
            permitted, refused = self._frames()
            if request.frame is not None:
                wanted = self._frame_for(request.frame)
                permitted = [item for item in permitted if item[1] is wanted]
        except _Refusal as refusal:
            return Observation(
                observation_id=f"obs-{self._sequence}",
                mode=ObservationMode.STRUCTURED,
                status=ObservationStatus.FAILED,
                page_state=self._page_state(),
                notes=(f"the structured read was refused: {refusal.detail}",),
                pages=self._page_list(),
            )
        nodes: list[AxNode] = []
        notes: list[str] = []
        status = ObservationStatus.COMPLETE
        if refused and request.frame is None:
            notes.append(f"{refused} frame(s) outside the profile were not read")
            status = ObservationStatus.PARTIAL
        skip, total, counted = request.start, 0, True
        for path, frame in permitted:
            held = [item.handle for item, _ in protected if item.frame is frame]
            window = _Slice(skip, self._limits.max_nodes - len(nodes), request.scope)
            report = self._collect(frame, path, nodes, notes, held, window)
            if report is None:
                status = ObservationStatus.PARTIAL
                continue
            seen, whole, closed = report
            if closed:
                status = ObservationStatus.PARTIAL
            total += seen
            counted = counted and whole
            skip = max(0, skip - seen)
        shown = Window(
            request.start,
            len(nodes),
            total,
            counted,
            request.frame,
            request.scope,
            collected=status is ObservationStatus.COMPLETE,
        )
        if shown.rest or not counted:
            status = ObservationStatus.PARTIAL
            more = f"{total}" if counted else f"more than {total}"
            notes.append(
                f"controls {request.start + 1} to {request.start + len(nodes)} of "
                f"{more} are shown; look again with start "
                f"{request.start + len(nodes)} for the next part"
            )
        if not nodes and status is ObservationStatus.COMPLETE:
            status = ObservationStatus.UNAVAILABLE
            notes.append("no control was readable on this screen")
        # A window over part of the screen is not the screen's identity, so
        # it carries no signature and never makes another observation stale.
        signature = () if request.narrowed else tuple(node.identity for node in nodes)
        return Observation(
            observation_id=f"obs-{self._sequence}",
            mode=ObservationMode.STRUCTURED,
            status=status,
            page_state=self._page_state(signature),
            nodes=tuple(nodes),
            dialog=self._dialog_info(),
            notes=tuple(notes),
            window=shown,
            pages=self._page_list(),
        )

    def _collect(
        self,
        frame: Frame,
        path: tuple[str, ...],
        nodes: list[AxNode],
        notes: list[str],
        held: list[ElementHandle],
        window: _Slice,
    ) -> tuple[int, bool, bool] | None:
        """Describe one frame's share of the window.

        Returns how many visible candidates the frame holds, whether that
        count is complete, and whether a closed shadow root hid part of the
        frame, or None when the frame could not be read.
        """
        known = self._known(frame)

        async def collect() -> tuple[dict[str, Any], dict[str, JSHandle]]:
            result = await frame._impl_obj.wait_for_function(
                SCRIPT,
                arg=mapping.to_impl(
                    {
                        **self._script_options(held),
                        "maxNodes": window.limit,
                        "skip": window.skip,
                        "countLimit": self._limits.max_count,
                        "scope": _scope_spec(window.scope),
                        "attributes": list(ATTRIBUTES),
                        "knownElements": [control.handle for control in known],
                    }
                ),
                timeout=self._limits.operation_ms,
                polling=POLL_MS,
            )
            report_handle = await result.get_property("report")
            report = await report_handle.json_value()
            fresh_handle = await result.get_property("fresh")
            fresh = await fresh_handle.get_properties()
            return report, {key: JSHandle(handle) for key, handle in fresh.items()}

        try:
            report, fresh = self._bounded(frame, collect())
        except Error as error:
            notes.append(f"a frame could not be read: {_brief(error)}")
            return None
        if self._blocked():
            # A dialog opened while the script ran, so the frame's report
            # may describe a page the dialog has already changed.
            notes.append("a dialog interrupted the observation")
            return None
        ids = self._identify(frame, known, fresh, report["gone"])
        nodes.extend(_node(item, path, ids) for item in report["nodes"])
        closed = bool(report["closedRoots"])
        if closed:
            notes.append("a closed shadow root could not be read")
        return int(report["total"]), bool(report["counted"]), closed

    def _script_options(self, held: list[ElementHandle]) -> dict[str, Any]:
        """Return the options every collector script reads its limits from."""
        return {
            "maxText": self._limits.max_text,
            "maxName": self._limits.max_name,
            "selector": ELEMENTS,
            "secretElements": held,
            "secretMark": self._mark(),
            "controlMark": self._ui_token,
            "pickerRules": [dataclasses.asdict(item) for item in self._profile.pickers],
        }

    # Control identity.

    def _known(self, frame: Frame) -> list[_Control]:
        """Return the controls held for the document ``frame`` shows now."""
        generation = self._generation(frame)
        for key, control in list(self._controls.items()):
            if control.frame is frame and control.generation != generation:
                del self._controls[key]
        return [c for c in self._controls.values() if c.frame is frame]

    def _identify(
        self,
        frame: Frame,
        known: list[_Control],
        fresh: dict[str, Any],
        gone: list[int],
    ) -> tuple[list[str], list[str]]:
        """Issue ids to the new elements, and forget the ones that left.

        An element keeps its id for as long as its document is showing. The
        id begins with a number for that document, so an id from before a
        navigation can never be confused with one after it, and the numbers
        after it only grow, so a later id means an element seen later.
        """
        generation = self._generation(frame)
        document = self._documents.setdefault(
            (frame, generation), len(self._documents) + 1
        )
        issued: list[str] = []
        for index in range(len(fresh)):
            element = fresh[str(index)].as_element()
            self._issued += 1
            control_id = f"d{document}:c{self._issued}"
            issued.append(control_id)
            if element is not None:
                self._controls[control_id] = _Control(
                    control_id, frame, element, generation
                )
        for index in gone:
            self._controls.pop(known[index].control_id, None)
        return [control.control_id for control in known], issued

    def _check_control(self, handle: ElementHandle, expect: Expectation | None) -> None:
        """Refuse to act on a control other than the one the decision was about.

        The loop decides restrictions for one observed control. A locator that
        now resolves to a different control the adapter already knows would
        carry that decision to it, so the action is stale. A control drawn
        since the observation is accepted as a layout change unless the loop
        asked for ``strict``, which it does when a restriction on this screen
        could apply to the new element.
        """
        if expect is None or not expect.control:
            return
        frame = handle.owner_frame()
        known = self._known(frame) if frame is not None else []
        try:
            index = handle.evaluate(KNOWN_AS, [c.handle for c in known])
        except Error as error:
            raise _Refusal(
                Outcome.STALE, "the control could not be identified"
            ) from error
        if index >= 0 and known[index].control_id == expect.control:
            return
        if index >= 0:
            raise _Refusal(
                Outcome.STALE,
                "the target now names a different control from the one decided on",
            )
        if expect.strict:
            raise _Refusal(
                Outcome.STALE,
                "the control was drawn again since it was observed; look again",
            )

    # Declared secrets.

    def _mark(self) -> dict[str, str]:
        return {"name": MARK_ATTRIBUTE, "token": self._token}

    def _protect(self, handle: ElementHandle) -> None:
        """Mark the field a declared secret is about to be typed into.

        The field is the pinned element the value will go into. It is marked
        and held, and only then is anything typed, into that same element. If
        the mark cannot be placed, nothing is typed. A fill that fails or is
        interrupted afterwards leaves the protection in place, because some
        or all of the value may already be in the field.
        """
        frame = handle.owner_frame()
        if frame is None:
            raise _Refusal(Outcome.NOT_FOUND, "the field could not be resolved")
        try:
            handle.evaluate(PROTECT, self._mark())
        except Error as error:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "the field could not be marked as holding a secret; nothing was typed",
            ) from error
        self._protected.append(_Protected(frame, handle, self._generation(frame)))

    def _protections(self) -> list[tuple[_Protected, _Hold]]:
        """Settle every protected field, or refuse whatever asked.

        A field whose frame is gone, or whose document was replaced by a
        navigation, is dropped: the value went with the document. A field
        that left its document while the document is still showing is not
        dropped and is not guessed about. The page may have put the value
        into a replacement, so every observation is refused until the
        document itself is gone.

        Fields a person typed into are marked first. One that cannot be
        marked while it is still on the page, or that left its document
        before it could be, refuses the same way.
        """
        self._marked()
        kept: list[tuple[_Protected, _Hold]] = []
        lost = bool(self._stranded())
        for item in self._protected:
            state = self._hold(item)
            if state is _Hold.GONE:
                continue
            kept.append((item, state))
            lost = lost or state is _Hold.LOST
        self._protected = [item for item, _ in kept]
        if lost:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "a field holding a declared secret left the page while the page "
                "is still showing, and the value may have moved with it",
            )
        return kept

    def _marked(self) -> None:
        """Mark every field a person typed into, or refuse whatever asked."""
        if self._protect_edits():
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "a field a person typed into could not be marked as holding a "
                "secret, so nothing on the screen is read",
            )

    def _hold(self, item: _Protected) -> _Hold:
        """Ask the page about one protected field, within the time limit.

        A call that runs out of time is a question the adapter cannot
        answer, which is how a field left on a page a dialog stopped is
        treated.
        """
        if item.frame.is_detached():
            return _Hold.GONE
        try:
            if self._page_id(item.frame.page) in self._blocked():
                answer = self._ask(item.frame, HELD, [item.handle, self._mark()])
            else:
                # A page without a blocking dialog answers immediately. A
                # handle from a closed document also fails immediately.
                answer = str(item.frame.evaluate(HELD, [item.handle, self._mark()]))
        except Error:
            if self._generation(item.frame) != item.generation:
                return _Hold.GONE
            return _Hold.LOST
        connected, _, marked = answer.partition(" ")
        if connected != "connected":
            return _Hold.LOST
        return _Hold.ACTIVE if marked == "marked" else _Hold.UNMARKED

    def _generation(self, frame: Frame) -> int:
        return self._generations.get(frame, 0)

    def _documents_now(self) -> tuple[tuple[Frame, int], ...]:
        """Return the navigation count of every frame in the active page."""
        return tuple(
            (frame, count)
            for frame, count in self._generations.items()
            if frame.page is self._page
        )

    # Visual observation.

    def _take_picture(self) -> Observation:
        """Capture the viewport as it stands, and segment the canvases in it.

        Nothing here scrolls, clicks, or types. An observation that moved the
        page would report a viewport the run never had, and the next click
        would be computed against a position that no longer exists. A canvas
        outside the viewport is reported as out of view instead, and the model
        can ask for a scroll, which is an action the profile declares.
        """
        try:
            masks, described = self._masks()
        except _Refusal as refusal:
            return self._blank_picture(refusal.detail)
        try:
            before = self._page.evaluate(VIEW)
            focus = self._screen_element(None)
            documents = self._documents_now()
            image = self._page.screenshot(
                style=self._picture_style(),
                mask=masks,
                mask_color="#101010",
                timeout=self._limits.operation_ms,
            )
        except Error as error:
            return self._blank_picture(_brief(error))
        try:
            # The masks are placed while the picture is taken. Checking them
            # again afterwards catches a field replaced during the capture.
            self._masks()
        except _Refusal as refusal:
            return self._blank_picture(refusal.detail)
        regions, notes = self._regions(image)
        view = self._page.evaluate(VIEW)
        if list(before) != list(view) or documents != self._documents_now():
            return self._blank_picture("the page moved or navigated during capture")
        current = self._screen_element(None)
        if (focus is None) != (current is None) or (
            focus is not None
            and current is not None
            and not current[0].evaluate("(el, held) => el === held", focus[0])
        ):
            return self._blank_picture("keyboard focus changed during capture")
        meta = VisualMeta(
            viewport_width=int(view[2]),
            viewport_height=int(view[3]),
            scroll_x=int(view[0]),
            scroll_y=int(view[1]),
            masked=tuple(described),
        )
        self._input_capture = _ScreenCapture(
            f"obs-{self._sequence}",
            image,
            tuple(view),
            focus,
            documents,
        )
        return Observation(
            observation_id=f"obs-{self._sequence}",
            mode=ObservationMode.VISUAL,
            status=ObservationStatus.PARTIAL if notes else ObservationStatus.COMPLETE,
            page_state=self._page_state(),
            regions=regions,
            image=image,
            visual=meta,
            dialog=self._dialog_info(),
            notes=tuple(notes),
            pages=self._page_list(),
        )

    def _blank_picture(self, detail: str) -> Observation:
        """Report a refused or broken capture, rather than a partial one.

        A screenshot that could not place every mask it owed is not a
        degraded screenshot, it is a leak, so no image is returned at all.
        """
        return Observation(
            observation_id=f"obs-{self._sequence}",
            mode=ObservationMode.VISUAL,
            status=ObservationStatus.FAILED,
            page_state=self._page_state(),
            notes=(f"the screenshot was refused: {detail}",),
            pages=self._page_list(),
        )

    def _masks(self) -> tuple[list[Locator], list[str]]:
        """Build the masks a capture owes, or refuse the capture.

        Every mask is a locator the driver paints over while it takes the
        picture, so the pixels are never in the image. A mask that cannot be
        built raises, and the caller returns no image at all.
        """
        masks: list[Locator] = []
        described: list[str] = []
        protected = self._protections()
        main = self._page.main_frame
        if self._unwatched.get(main, -1) == self._generation(main):
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "the page itself had no recorder while a person held it",
            )
        permitted, _ = self._frames()
        for _path, frame in permitted:
            fields = frame.locator(SECRET_FIELDS)
            if fields.count():
                masks.append(fields)
                described.append("credential fields")
            masks.extend(self._frame_masks(frame, described))
        for item, state in protected:
            if item.frame.page is not self._page:
                # Another page's field is masked when that page is captured.
                continue
            if not self._frame_permitted(item.frame):
                # Its parent-frame mask covers the whole frame.
                continue
            masks.append(self._secret_mask(item, state))
            described.append("a field a declared secret was typed into")
        return masks, described

    def _secret_mask(self, item: _Protected, state: _Hold) -> Locator:
        """Return a mask that covers this protected field, or refuse the capture.

        The mask is placed by the session's mark, and the locator is checked
        to resolve to the held element itself. A field whose mark the page
        removed cannot be covered, so the capture is refused rather than
        taken without it.
        """
        if state is not _Hold.ACTIVE:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "a field holding a declared secret lost the mark its mask is placed by",
            )
        marked = item.frame.locator(f'[{MARK_ATTRIBUTE}="{self._token}"]')
        try:
            covered = marked.evaluate_all(COVERS, item.handle)
        except Error as error:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "a secret field's mask could not be checked"
            ) from error
        if not covered:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "the mask for a secret field does not reach the field",
            )
        return marked

    def _frame_masks(self, frame: Frame, described: list[str]) -> list[Locator]:
        """Cover every frame the profile does not permit, or refuse the capture.

        The holder element is asked for the frame it is actually showing. An
        element that will not report one leaves the adapter unable to say
        whether the pixels are inside the profile, and an unanswerable
        boundary question refuses the capture instead of assuming the answer.
        """
        masks: list[Locator] = []
        holders = frame.locator("iframe, frame")
        for index in range(holders.count()):
            holder = holders.nth(index)
            try:
                handle = holder.element_handle(timeout=self._limits.operation_ms)
                child = handle.content_frame() if handle is not None else None
            except Error as error:
                raise _Refusal(
                    Outcome.SURFACE_ERROR, "a frame's contents could not be read"
                ) from error
            if child is None:
                raise _Refusal(
                    Outcome.SURFACE_ERROR,
                    "a frame would not report where it currently is",
                )
            if not self._frame_permitted(child):
                masks.append(holder)
                described.append("a frame outside the profile")
        return masks

    def _regions(self, image: bytes) -> tuple[tuple[VisualRegion, ...], list[str]]:
        """Segment every permitted canvas out of the picture already taken.

        The canvas is located through the DOM and cut out of the viewport
        image, so no canvas is photographed on its own and nothing is scrolled
        into view to photograph it. Each region carries its own crop, which is
        what the model is shown beside the id: an id next to a width says
        nothing about which button it names.

        Ids name this capture. An id from the capture before it is still
        recognised, and is refused as stale rather than silently matched,
        which is a different thing to tell a caller than never having seen it.
        Anything older is dropped, so this dictionary holds two captures.
        """
        found: list[VisualRegion] = []
        notes: list[str] = []
        previous = self._capture
        self._capture = f"obs-{self._sequence}"
        self._anchors = {
            key: held for key, held in self._anchors.items() if held.capture == previous
        }
        permitted, _ = self._frames()
        for index, (path, frame) in enumerate(permitted):
            canvases = frame.locator("canvas")
            total = canvases.count()
            if total > self._limits.max_canvases:
                notes.append("more canvases than the limit; the rest were not read")
            for canvas in range(min(total, self._limits.max_canvases)):
                shot = self._canvas_crop(canvases.nth(canvas), image, notes)
                if shot is None:
                    continue
                for number, candidate in enumerate(visual.segment(shot)):
                    anchor_id = f"anchor-{self._sequence}-{index}-{canvas}-{number}"
                    self._anchors[anchor_id] = _Anchor(
                        path, canvas, candidate.crop, self._capture, candidate.box
                    )
                    found.append(
                        VisualRegion(
                            anchor_id=anchor_id,
                            frame=path,
                            width=candidate.box.width,
                            height=candidate.box.height,
                            hint="a painted region inside a canvas",
                            image=candidate.crop,
                        )
                    )
        return tuple(found), notes

    def _canvas_crop(
        self, canvas: Locator, image: bytes, notes: list[str]
    ) -> bytes | None:
        """Crop a canvas, or return None and append the failure reason to notes."""
        try:
            box = canvas.bounding_box(timeout=self._limits.operation_ms)
        except Error:
            notes.append("a canvas could not be measured")
            return None
        if box is None:
            notes.append("a canvas is not rendered and was not read")
            return None
        shot = visual.cut_out(image, _box(box))
        if shot is None:
            notes.append("a canvas sits outside the viewport and was not read")
        return shot

    # Frames.

    def _frames(
        self, page: Page | None = None
    ) -> tuple[list[tuple[tuple[str, ...], Frame]], int]:
        """Return the frames this profile permits, by their current location."""
        permitted: list[tuple[tuple[str, ...], Frame]] = []
        refused = 0
        for frame, path in self._frame_paths(page).items():
            if len(permitted) >= self._limits.max_frames:
                break
            if not self._frame_permitted(frame):
                refused += 1
                continue
            permitted.append((path, frame))
        return permitted, refused

    def _frame_paths(self, page: Page | None = None) -> dict[Frame, tuple[str, ...]]:
        """Name every frame of a page, outermost first, breadth first.

        A frame whose name no sibling shares is named by its name, which is
        what a person reads and what survives a reload. A frame with no name,
        or with a name a sibling shares, is named by its name and the number
        of the document it is showing, such as ``ledger#4`` or ``#7``. That
        number changes when the frame navigates or is replaced, so an old id
        stops resolving instead of reaching whatever took its place. The
        page is the active one unless another is named.
        """
        main = (page or self._page).main_frame
        paths: dict[Frame, tuple[str, ...]] = {main: ()}
        waiting = [main]
        while waiting:
            parent = waiting.pop(0)
            # The driver can still list a frame its page has already removed.
            # A detached frame shows nothing and must not share a live
            # frame's name, so it is left out.
            children = [
                child for child in parent.child_frames if not child.is_detached()
            ]
            for child in children:
                paths[child] = (*paths[parent], self._segment(child, children))
                waiting.append(child)
        return paths

    def _segment(self, frame: Frame, siblings: Sequence[Frame]) -> str:
        name = frame.name
        shared = sum(1 for sibling in siblings if sibling.name == name)
        if name and "#" not in name and shared == 1:
            return name
        return f"{name}#{self._document(frame)}"

    def _document(self, frame: Frame) -> int:
        """Return the session's number for the document ``frame`` shows now."""
        key = (frame, self._generation(frame))
        return self._documents.setdefault(key, len(self._documents) + 1)

    def _frame_permitted(self, frame: Frame) -> bool:
        """Report whether this frame and every frame above it are permitted.

        The frame's own current URL is what decides, not the ``src`` its
        holder was written with, because a frame replaced through ``srcdoc``
        or a same-document write keeps the attribute it was born with while
        showing something else. A frame nested inside a refused frame is
        refused too, whatever its own URL says.

        In a recorded session, a frame that had no recorder when a person
        handed the session back is refused as well, until it navigates. What
        a person typed there is unknown, so nothing in it is read and every
        capture covers it.
        """
        current: Frame | None = frame
        while current is not None:
            if policy.route_for(self._profile, current.url) is None:
                return False
            unwatched = self._unwatched.get(current)
            if unwatched is not None and unwatched == self._generation(current):
                return False
            current = current.parent_frame
        return True

    def _route_permitted(self, frame: Frame) -> bool:
        """Report whether this frame and every frame above it are on a route."""
        current: Frame | None = frame
        while current is not None:
            if policy.route_for(self._profile, current.url) is None:
                return False
            current = current.parent_frame
        return True

    def _frame_for(self, path: Sequence[str]) -> Frame:
        """Return the one permitted frame ``path`` names, or refuse it.

        A path is compared with the ids this session issues now. A numbered
        segment that no longer matches belonged to a document that navigated
        or was replaced. A plain name that now belongs to several frames is
        ambiguous, and no frame is chosen for it by order.
        """
        wanted = tuple(path)
        permitted, _ = self._frames()
        for candidate, frame in permitted:
            if candidate == wanted:
                return frame
        if any("#" in segment for segment in wanted):
            raise _Refusal(
                Outcome.STALE,
                "that frame navigated or was replaced since it was observed; "
                "look again",
            )
        named = [
            frame
            for frame in self._frame_paths()
            if _frame_names(frame) == wanted and frame.parent_frame is not None
        ]
        if len(named) > 1:
            raise _Refusal(
                Outcome.AMBIGUOUS,
                f"{len(named)} frames share that name; use the frame id from a "
                "fresh observation",
            )
        raise _Refusal(Outcome.NOT_FOUND, "the named frame is not available")

    # Acting.

    def _perform(
        self, action: Action, expect: Expectation | None = None
    ) -> ActionResult:
        kind = action.kind
        if kind in DIALOG_ACTIONS:
            return self._answer_dialog(kind is ActionKind.ACCEPT_DIALOG, expect)
        if isinstance(action.target, ScreenTarget):
            return self._screen_action(action, action.target, expect)
        if kind is ActionKind.NAVIGATE:
            return self._navigate(action)
        if kind in _SESSION_ACTIONS and action.target is None:
            return self._session_action(action)
        if isinstance(action.target, VisualAnchor):
            return self._visual_action(action, expect)
        if kind is ActionKind.WAIT_FOR:
            return self._wait_for(action, expect)
        if kind not in _ELEMENT_ACTIONS:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                f"{kind.value} does not take a structured target here",
            )
        handle = self._locate(action.target)
        self._check_actionable(handle, kind)
        frame = () if action.target is None else action.target.frame
        match kind:
            case ActionKind.CLICK:
                return self._click(action, handle, expect, frame)
            case ActionKind.PRESS_KEY:
                return self._press(action, handle, expect, frame)
            case ActionKind.TYPE | ActionKind.SELECT:
                return self._fill(action, handle, expect, frame)
            case _:
                self._check_context(handle, expect)
                self._check_record(action, handle, frame, expect)
                return self._element_action(action, handle, frame, expect)

    def _click(
        self,
        action: Action,
        handle: ElementHandle,
        expect: Expectation | None,
        frame: tuple[str, ...],
    ) -> ActionResult:
        """Wait until the control can take a click, then check, then click.

        A trial click waits for everything the driver waits for, an overlay
        included, without clicking. Only then is the element pinned and its
        context and record checked, so no check is older than the wait. The
        driver's own click still waits if the page changes again, so the
        record is checked a third time inside the page, by a guard that sees
        each input event of the click before the page does and cancels them
        if the record changed. A click on a replaced element fails, because
        the pinned element is gone rather than found again.
        """
        timeout = self._limits.operation_ms
        try:
            # Attribute any input emitted during the readiness trial to this
            # control, as for the final click.
            with self._claimed(handle, _CLICKED):
                handle.click(trial=True, timeout=timeout)
        except Error as error:
            if not _connected(handle):
                raise _Refusal(
                    Outcome.STALE,
                    "the control was replaced while the click waited; look again",
                ) from error
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "the control never became ready for a click"
            ) from error
        # The trial click may have waited for an overlay, and the session can
        # have changed while it did.
        self._check_boundary(expect)
        self._check_control(handle, expect)
        self._check_context(handle, expect)
        self._check_submission(action, handle, frame, expect)
        guard = self._guard(action, handle, frame, _CLICK_EVENTS, expect=expect)
        self._check_boundary(expect, guard)
        try:
            with self._sending(guard), self._claimed(handle, _CLICKED):
                handle.click(timeout=timeout)
        except Error as error:
            if self._dialog is None:
                self._release(guard, error)
            raise
        self._release(guard)
        self._settle()
        return self._result(Outcome.OK)

    def _press(
        self,
        action: Action,
        handle: ElementHandle,
        expect: Expectation | None,
        frame: tuple[str, ...],
    ) -> ActionResult:
        """Send one key to the control the action names, and to no other.

        The action focuses its own target, then asks the page whether that
        element holds focus. A page that moves focus somewhere else on focus
        stops the action there, with nothing sent. The key's input events go
        through the same guard as a click's, which also cancels a key that
        reaches any other element of that document, so a focus that moves
        between the check and the key does not deliver it elsewhere.
        """
        if action.value not in KEYS:
            raise _Refusal(Outcome.NOT_ACTIONABLE, "that key is not permitted")
        self._check_control(handle, expect)
        self._check_context(handle, expect)
        self._check_submission(action, handle, frame, expect)
        try:
            handle.focus()
            focused = bool(handle.evaluate(FOCUSED))
        except Error as error:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "the control could not take focus"
            ) from error
        if not focused:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "focus moved away from the control; the key was not sent",
            )
        self._note_change(handle)
        self._check_boundary(expect)
        guard = self._guard(action, handle, frame, _KEY_EVENTS, expect=expect)
        self._check_boundary(expect, guard)
        try:
            with self._sending(guard), self._claimed(handle, _PRESSED, limits=_ONE_KEY):
                handle.press(str(action.value), timeout=self._limits.operation_ms)
        except Error as error:
            if self._dialog is None:
                self._release(guard, error)
            raise
        self._release(guard)
        self._settle()
        return self._result(Outcome.OK)

    def _fill(
        self,
        action: Action,
        handle: ElementHandle,
        expect: Expectation | None,
        frame: tuple[str, ...],
    ) -> ActionResult:
        """Type or select into a pinned field, checking the record last.

        Readiness is waited for first. Setting a value cannot be cancelled by
        a page guard the way an input event can, so the record check is the
        last thing before the value is set, on the same element.
        """
        timeout = self._limits.operation_ms
        state = "editable" if action.kind is ActionKind.TYPE else "enabled"
        try:
            handle.wait_for_element_state(state, timeout=timeout)
        except Error as error:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "the field never became ready"
            ) from error
        self._check_boundary(expect)
        self._check_control(handle, expect)
        self._check_context(handle, expect)
        self._check_record(action, handle, frame)
        self._check_boundary(expect)
        self._check_binding(action, handle, frame, expect)
        self._note_change(handle)
        if action.kind is ActionKind.TYPE:
            self._type(action, handle)
        else:
            selected = handle.evaluate_handle(
                """(el, value) => {
                  const matches = Array.from(el.options || []).filter(option =>
                    option.value === value || option.label === value);
                  if (matches.length !== 1) { return null; }
                  const option = matches[0];
                  return option.disabled || option.closest('optgroup[disabled]')
                    ? null : option;
                }""",
                str(action.value),
            ).as_element()
            if selected is None:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE,
                    "the option value or label is missing, disabled, or ambiguous",
                )
            self._check_record(action, handle, frame)
            self._check_boundary(expect)
            with self._sending(), self._claimed(handle, ("select",)):
                handle.select_option(element=selected, timeout=timeout)
        self._settle()
        return self._result(Outcome.OK)

    def _guard(
        self,
        action: Action,
        element: Locator | ElementHandle,
        frame: tuple[str, ...],
        events: tuple[str, ...],
        *,
        persistent: bool = False,
        expect: Expectation | None = None,
    ) -> _InputGuard:
        """Check the record now, and place the in-page check for the input.

        The guard is placed for every click and key, with or without record
        evidence, because it also catches input that reaches a replacement
        of the pinned element.
        """
        options = self._check_record(action, element, frame)
        guard = _InputGuard(
            self._active,
            self._frame_for(frame),
            f"computeruse-guard-{secrets.token_hex(16)}",
        )
        binding_options: dict[str, Any] = {}
        if self._profile.operations:
            canvas = element.evaluate("el => el.tagName") == "CANVAS"
            binding_options = {
                "paintedBinding": canvas,
                "bindingOnce": action.kind is ActionKind.TYPE,
                "operationName": expect.binding if expect else None,
                "operationKind": action.kind.value,
                "operationKey": action.value if isinstance(action.value, str) else "",
                "requiredSelections": sorted(
                    operations.required_selections(
                        self._profile,
                        Operation(
                            action.kind,
                            self._profile.scope.route(self.location()) or "",
                            binding=expect.binding or "" if expect else "",
                            business=expect.business if expect else "",
                        ),
                    )
                ),
            }
            if not canvas:
                route = self._profile.scope.route(self.location())
                binding_options["operationRules"] = [
                    {
                        "name": rule.name,
                        "target": rule.target,
                        "role": rule.role,
                        "context": list(rule.context),
                        "submission": rule.submission,
                    }
                    for rule in self._profile.operations
                    if rule.mode is ObservationMode.STRUCTURED
                    and rule.route == route
                    and rule.frame == frame
                    and action.kind in rule.kinds
                    and (not rule.key or rule.key == action.value)
                ]
        if (
            isinstance(action.target, ScreenTarget)
            and action.target.point is not None
            and expect is not None
            and expect.observed_control is not None
        ):
            binding_options["observedControl"] = _control_spec(expect.observed_control)
        try:
            element.evaluate(
                GUARD,
                {
                    **(options or self._script_options([])),
                    "events": list(events),
                    "last": "" if persistent else events[-1],
                    "check": options is not None,
                    "lifetime": self._limits.operation_ms * 2,
                    "guardKey": guard.key,
                    **binding_options,
                },
            )
        except Error as error:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "the check could not be placed in the page"
            ) from error
        self._guards.append(guard)
        try:
            self._check_binding(action, element, frame, expect)
        except _Refusal:
            self._release(guard)
            raise
        return guard

    def _check_binding(
        self,
        action: Action,
        element: Locator | ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None,
    ) -> None:
        """Check declared operation identity again after readiness and before input."""
        if not self._profile.operations:
            return
        if expect is None or expect.binding is None:
            raise _Refusal(Outcome.STALE, "operation identity was not judged")
        route = self._profile.scope.route(self.location()) or ""
        if element.evaluate("el => el.tagName") == "CANVAS":
            capture = self._input_capture
            if capture is None or not isinstance(action.target, ScreenTarget):
                raise _Refusal(
                    Outcome.STALE, "a declared operation needs a current capture"
                )
            if self._reader is None:
                self._reader = LocalReader()
            masks, _ = self._masks()
            fresh = self._page.screenshot(
                mask=masks, mask_color="#101010", style=self._picture_style()
            )
            self._masks()
            found = [
                operations.identify(
                    self._profile,
                    action,
                    route,
                    frame=frame,
                    lines=self._reader.lines(image),
                    scale=Image.open(io.BytesIO(image)).width / capture.view[2],
                )
                for image in (capture.image, fresh)
            ]
        else:
            live = element.evaluate(OPERATION, self._script_options([]))
            node = AxNode(
                live["role"],
                live["name"],
                frame=frame,
                context=tuple(live["context"]),
                secret=live["secret"],
                submits=live["submits"],
                enter=live["enter"],
                selections=tuple(
                    (name, bool(ready)) for name, ready in live["selections"]
                ),
            )
            found = [operations.identify(self._profile, action, route, node=node)]
            physical = Operation(
                action.kind, route, binding=expect.binding, business=expect.business
            )
            if not operations.selections_ready(self._profile, physical, node):
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE,
                    "choose an item in every required picker before committing; "
                    "typed search text does not prove a selection",
                )
        if any(
            (item.name if item else "") != expect.binding
            or (item.business if item else "") != expect.business
            for item in found
        ):
            raise _Refusal(
                Outcome.STALE, "the operation binding changed since authorization"
            )

    def _unknown_window(self, expect: Expectation | None) -> str:
        """Return a page the caller did not know about, from session metadata."""
        if expect is None or expect.windows is None:
            return ""
        return next((key for key in self._pages if key not in expect.windows), "")

    def _check_boundary(
        self, expect: Expectation | None, guard: _InputGuard | None = None
    ) -> None:
        """Refuse input when a window or dialog appeared while the action waited.

        This runs after every readiness wait and immediately before input is
        sent, from session metadata only: the page list and the dialogs the
        session holds. Nothing is sent when it refuses, so the outcome says
        the input was not sent. A driver call that waits again after this
        check, such as a click that waits for the control to be stable, can
        still see a window open while it waits. Such input is delivered, and
        the loop hands the session over as soon as the call returns.
        """
        unknown = self._unknown_window(expect)
        detail = ""
        if unknown:
            detail = f"{unknown} opened while the input waited; the input was not sent"
        elif self._dialog is not None or self._dialog_elsewhere():
            detail = "a dialog opened while the input waited; the input was not sent"
        if not detail:
            return
        if guard is not None:
            self._drop(guard)
        raise _Refusal(Outcome.STALE if unknown else Outcome.NOT_ACTIONABLE, detail)

    def _release(self, guard: _InputGuard, failed: Error | None = None) -> None:
        """Remove the guard, and refuse the action if it cancelled the input.

        A driver call that timed out because the guard kept cancelling its
        retries is reported as the stale action it is, not as a driver
        failure. Any other failure is left for the caller to raise.
        """
        blocked = self._remove_guard(guard)
        if blocked == "uncertain_operation":
            raise _Refusal(
                Outcome.UNCERTAIN,
                "input started before the operation changed; "
                "check delivery without retrying",
            ) from failed
        if blocked and blocked != "clear":
            raise _Refusal(
                Outcome.STALE,
                f"the {blocked} changed as the input arrived, and the page "
                "did not receive it",
            ) from failed

    def _drop(self, guard: _InputGuard) -> None:
        """Remove a guard whose input was never sent, or that a person inherits."""
        self._remove_guard(guard)

    def _remove_guard(self, guard: _InputGuard) -> str | None:
        """Defer cleanup behind any dialog and bound cleanup if one opens later."""
        if guard not in self._guards or guard.page in self._blocked():
            return None
        try:
            answer = self._ask(
                guard.frame,
                """(key) => {
                  const g = window[key];
                  if (!g) { return 'clear'; }
                  g.remove(); delete window[key]; return g.blocked || 'clear';
                }""",
                guard.key,
            )
        except Error:
            # A navigation destroys the guard. A new dialog defers its cleanup.
            if guard.page not in self._blocked():
                self._forget_guard(guard)
            return None
        self._forget_guard(guard)
        return answer

    def _forget_guard(self, guard: _InputGuard) -> None:
        self._guards = [item for item in self._guards if item is not guard]

    def _check_submission(
        self,
        action: Action,
        element: Locator | ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None,
    ) -> None:
        """Refuse input whose native submission is not the one the gate judged.

        A page can turn a plain button into a submit button, or give a field
        a form, after the observation the gate judged. The live element's
        submission is read a moment before input and compared with the one
        observed; a change makes the action stale, and the next decision is
        judged afresh (rule 6).
        """
        if expect is None:
            return
        if action.kind in {ActionKind.CLICK, ActionKind.DOUBLE_CLICK}:
            observed, side = expect.submits_as, "submits"
        elif action.kind is ActionKind.PRESS_KEY and action.value == "Enter":
            observed, side = expect.enter_as, "enter"
        else:
            return
        if observed is None:
            return
        try:
            live = element.evaluate(SUBMISSION, self._script_options([]))
        except Error as error:
            raise _Refusal(
                Outcome.STALE, "the control's submission could not be read"
            ) from error
        description = str((live or {}).get(side) or "")
        framed = f"{'/'.join(frame)}#{description}" if description else ""
        if framed != observed:
            raise _Refusal(
                Outcome.STALE, "the control's submission changed since it was judged"
            )

    def _check_context(
        self, locator: Locator | ElementHandle, expect: Expectation | None
    ) -> None:
        """Refuse a control that now sits under a different record.

        The decision was made against a control under a particular heading, in
        a particular region, row, and form. That path is asked of the control
        this call is about to operate, a moment before it operates it. A
        button that moved under the same heading still matches, and so does a
        page where something unrelated changed. A record swapped in at the
        same URL does not, and the action is reported stale rather than
        applied to whatever took its place.
        """
        if expect is None or not expect.context:
            return
        try:
            seen = locator.evaluate(CONTEXT, self._script_options([]))
        except Error as error:
            raise _Refusal(
                Outcome.STALE, "the control's context could not be read"
            ) from error
        if tuple(str(entry) for entry in seen) != tuple(expect.context):
            raise _Refusal(
                Outcome.STALE, "the control now sits under a different record"
            )

    def _check_record(
        self,
        action: Action,
        element: Locator | ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None = None,
    ) -> dict[str, Any] | None:
        """Refuse an action whose record is no longer the one it was decided about.

        The evidence control is resolved again, its value is read again the
        way the collector reports it, and it must still sit in the target's
        row or panel. Each question is answered from the live page a moment
        before the action, never from the observation. A layout change passes
        all three; a different record in the same place fails the value, and
        evidence that vanished, repeated, or moved away from the target fails
        the others. When ``expect`` asks for displayed evidence, a field is
        refused as the source, because it shows what was typed into it.
        """
        evidence = action.evidence
        if evidence is None:
            return None
        if evidence.source.frame != frame:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "the record evidence is not in the target's frame",
            )
        try:
            source = self._locate(evidence.source)
        except _Refusal as refusal:
            if refusal.outcome is Outcome.AMBIGUOUS:
                raise _Refusal(
                    Outcome.AMBIGUOUS,
                    "the record evidence now matches more than one control",
                ) from None
            raise _Refusal(
                Outcome.STALE, "the record evidence is no longer on the screen"
            ) from None
        if expect is not None and expect.displayed_evidence:
            self._refuse_field(source)
        document = self._frame_for(frame)
        protected = self._protections()
        held = [item.handle for item, _ in protected if item.frame is document]
        try:
            options: dict[str, Any] = {
                **self._script_options(held),
                "source": source,
                "relation": evidence.relation.value,
                "value": evidence.displayed,
            }
            seen = element.evaluate(EVIDENCE, options)
        except Error as error:
            raise _Refusal(
                Outcome.STALE, "the record evidence could not be read"
            ) from error
        if seen["secret"]:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "a withheld field cannot identify a record"
            )
        if seen["value"] != evidence.displayed:
            raise _Refusal(
                Outcome.STALE,
                "the record on the screen is not the one the decision was about",
            )
        if not seen["related"]:
            raise _Refusal(
                Outcome.AMBIGUOUS,
                f"no {evidence.relation.value} holds this record alone with the "
                "target, so the evidence cannot vouch for it",
            )
        return options

    def _navigate(self, action: Action) -> ActionResult:
        destination = action.destination or self._page.url
        with self._sending():
            self._page.goto(
                destination,
                wait_until="domcontentloaded",
                timeout=self._limits.navigation_ms,
            )
        return self._result(Outcome.OK)

    def _session_action(self, action: Action) -> ActionResult:
        if action.kind is ActionKind.SCROLL:
            if action.value not in {"up", "down"}:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "scroll takes up or down")
            step = -400 if action.value == "up" else 400
            # The wheel event reaches the page after the call returns, so the
            # claim lasts until the scroll comes to rest.
            with self._sending(), self._claimed(None, ("scroll",), anywhere=True):
                self._page.mouse.wheel(0, step)
                self._rest()
        else:
            if action.value not in KEYS:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "that key is not permitted")
            with self._sending(), self._claimed(None, _PRESSED, limits=_ONE_KEY):
                self._page.keyboard.press(str(action.value))
        return self._result(Outcome.OK)

    def _element_action(
        self,
        action: Action,
        handle: ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None = None,
    ) -> ActionResult:
        match action.kind:
            case ActionKind.READ:
                self._check_shown_only(handle, expect)
                return self._result(Outcome.OK, extracted=self._shown(handle, frame))
            case ActionKind.ASSERT:
                if self._shown(handle, frame) != " ".join(str(action.value).split()):
                    return self._result(Outcome.NOT_FOUND, detail="the text differs")
            case _:
                # A targeted scroll brings the control into view. This movement
                # comes from the explicit scroll action.
                if action.value != "into_view":
                    raise _Refusal(
                        Outcome.NOT_ACTIONABLE, "a scroll with a target takes into_view"
                    )
                handle.scroll_into_view_if_needed(timeout=self._limits.operation_ms)
                self._rest()
        self._settle()
        return self._result(Outcome.OK)

    def _check_shown_only(
        self, handle: ElementHandle, expect: Expectation | None
    ) -> None:
        """Refuse a field when the reading must be something the page displays.

        A field shows what was typed into it, including by this run, so it
        cannot confirm which record a page is about. A committed reading
        accepts a field only when neither this run nor a person changed it
        in the document that still shows it.
        """
        if expect is None:
            return
        if expect.displayed:
            self._refuse_field(handle)
        elif expect.committed and self._changed_here(handle):
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "this field was changed on this page, so it cannot show what the "
                "application holds; check a value the page displays",
            )

    def _note_change(self, handle: ElementHandle) -> None:
        """Hold a field this run is about to change, by the element itself."""
        frame = handle.owner_frame()
        if frame is not None:
            self._changed.append(_Protected(frame, handle, self._generation(frame)))

    def _changed_here(self, handle: ElementHandle) -> bool:
        """Report whether ``handle`` was changed in the document now showing it.

        A navigation of the field's frame replaces its document, and with it
        the change, so a field the application renders again counts as its
        own. Fields a person typed into count as changed too.
        """
        frame = handle.owner_frame()
        if frame is None:
            return True
        generation = self._generation(frame)
        held = [
            *self._changed,
            *(item for item in self._protected if item.person),
        ]
        held = [
            item
            for item in held
            if item.frame is frame and item.generation == generation
        ]
        for item in held:
            try:
                if handle.evaluate("(el, other) => el === other", item.handle):
                    return True
            except Error:
                continue
        return False

    def _refuse_field(self, handle: ElementHandle) -> None:
        """Refuse ``handle`` when it is a field or an editable region."""
        try:
            field = bool(
                handle.evaluate(
                    "el => el.matches('input, select, textarea') "
                    "|| el.isContentEditable"
                )
            )
        except Error as error:
            raise _Refusal(Outcome.STALE, "the control could not be read") from error
        if field:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "a field shows what was typed; check a value the page displays",
            )

    def _painted(
        self, capture: _ScreenCapture, point: Point
    ) -> tuple[str, tuple[str, ...]]:
        """Read the line of text a canvas paints at ``point`` in ``capture``.

        A canvas holds no text a script can read, so its picture is read on
        this machine. The line must hold the point; anything else reads as
        nothing, and a check against it fails rather than guessing.

        A second, fresh picture is read at the same point, and the two
        readings must agree, or the read is refused as stale (rule 10).
        Agreement detects a screen that changed between the two; it does not
        prove that recognition read the line correctly. The capture's other
        lines come back too, so the caller can tell whether a label is shown
        once.
        """
        if self._reader is None:
            self._reader = LocalReader()
        width = capture.view[2] if len(capture.view) > 2 else 0
        masks, _ = self._masks()
        fresh = self._page.screenshot(
            mask=masks, mask_color="#101010", style=self._picture_style()
        )
        self._masks()
        readings = []
        shown: tuple[str, ...] = ()
        for image in (capture.image, fresh):
            # Each picture has its own pixel size, so the point is scaled to
            # each one; the fresh screenshot need not match the capture's.
            scale = Image.open(io.BytesIO(image)).width / width if width else 1.0
            lines = self._reader.lines(image)
            shown = shown or tuple(line.text for line in lines)
            line = reading.at(lines, point.x * scale, point.y * scale)
            readings.append(line.text if line is not None else "")
        if readings[0] != readings[1]:
            raise _Refusal(
                Outcome.STALE, "two readings of the painted line disagree; look again"
            )
        return readings[0][: self._limits.max_extract], shown

    def _shown(self, handle: ElementHandle, frame: tuple[str, ...]) -> str:
        """Read what one element shows, the way an observation reports it.

        A field gives its value, a checkbox or radio its checked state, and a
        select its chosen options. A field that holds a secret is refused
        before its value is read, by the same test the collector uses.
        """
        document = self._frame_for(frame)
        protected = self._protections()
        held = [item.handle for item, _ in protected if item.frame is document]
        options = {**self._script_options(held), "maxName": self._limits.max_extract}
        try:
            seen = handle.evaluate(READ, options)
        except Error as error:
            raise _Refusal(Outcome.STALE, "the control could not be read") from error
        if seen["secret"]:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "that field holds a secret and is not read"
            )
        return str(seen["value"])

    def _wait_for(self, action: Action, expect: Expectation | None) -> ActionResult:
        """Wait until the target resolves to exactly one visible control.

        A control that is not there yet is waited for, up to the value in
        milliseconds or the operation timeout, and never past the navigation
        timeout. Resolution uses the same rules as every other action, so a
        match that stays ambiguous is reported as ambiguous at the deadline.
        """
        limit = self._limits.operation_ms
        if action.value is not None:
            try:
                limit = int(str(action.value))
            except ValueError as error:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE, "wait_for takes milliseconds"
                ) from error
            if not 0 < limit <= self._limits.navigation_ms:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE,
                    f"wait_for takes 1 to {self._limits.navigation_ms} milliseconds",
                )
        target = action.target
        if not isinstance(target, (AxLocator, DomLocator)):
            raise _Refusal(Outcome.NOT_ACTIONABLE, "wait_for takes a structured target")
        deadline = time.monotonic() + limit / 1000
        frame = target.frame
        while True:
            if self._dialog is not None:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "a dialog opened while waiting")
            try:
                handle = self._locate(target)
            except _Refusal as refusal:
                if refusal.outcome not in _STILL_WAITING:
                    raise
                if time.monotonic() >= deadline:
                    raise _Refusal(
                        refusal.outcome,
                        f"after {limit} ms: {refusal.detail}",
                    ) from None
                self._page.wait_for_timeout(100)
                continue
            self._check_context(handle, expect)
            self._check_record(action, handle, frame)
            return self._result(Outcome.OK)

    def _screen_element(
        self, point: Point | None
    ) -> tuple[ElementHandle, tuple[str, ...]] | None:
        """Hit test pixels or follow keyboard focus through frames and shadow roots."""
        frame = self._page.main_frame
        permitted = {frame: path for path, frame in self._frames()[0]}
        position = [point.x, point.y] if point else None
        while frame in permitted:
            handle = frame.evaluate_handle(
                """point => {
                  let root = document;
                  let el = point ? root.elementFromPoint(...point) : root.activeElement;
                  while (el && el.shadowRoot) {
                    root = el.shadowRoot;
                    const next = point
                      ? root.elementFromPoint(...point) : root.activeElement;
                    if (!next || next === el) break;
                    el = next;
                  }
                  return el;
                }""",
                position,
            ).as_element()
            if handle is None:
                return None
            if handle.get_attribute("data-computeruse-ui") == self._ui_token:
                return None
            child = handle.content_frame()
            if child is None:
                return handle, permitted[frame]
            if position is not None:
                position = handle.evaluate(
                    """(el, p) => {
                      const r = el.getBoundingClientRect();
                      return [(p[0]-r.x)*el.offsetWidth/r.width-el.clientLeft,
                              (p[1]-r.y)*el.offsetHeight/r.height-el.clientTop];
                    }""",
                    position,
                )
            frame = child
        return None

    def _screen_ready(self, target: ScreenTarget) -> _ScreenCapture:
        self._marked()
        capture = self._input_capture
        if capture is None or target.capture_id != capture.capture_id:
            raise _Refusal(Outcome.STALE, "look again before using screen input")
        if capture.documents != self._documents_now():
            raise _Refusal(Outcome.STALE, "a document navigated; look again")
        view = tuple(self._page.evaluate(VIEW))
        if view != capture.view:
            raise _Refusal(Outcome.STALE, "the viewport moved or resized; look again")
        point = target.point
        if point is None:
            current = self._screen_element(None)
            if capture.focus is None or current is None:
                raise _Refusal(
                    Outcome.BLOCKED, "keyboard focus is outside the permitted surface"
                )
            if not current[0].evaluate("(el, held) => el === held", capture.focus[0]):
                raise _Refusal(Outcome.STALE, "keyboard focus changed; look again")
            return capture
        if not 0 <= point.x < view[2] or not 0 <= point.y < view[3]:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "the point is outside the screenshot"
            )
        masks, _ = self._masks()
        image = self._page.screenshot(
            mask=masks, mask_color="#101010", style=self._picture_style()
        )
        self._masks()
        # Check the pointed region rather than unrelated animations elsewhere.
        box = (
            max(0, int(point.x) - 24),
            max(0, int(point.y) - 24),
            min(view[2], int(point.x) + 24),
            min(view[3], int(point.y) + 24),
        )
        before = Image.open(io.BytesIO(capture.image)).convert("RGB").crop(box)
        after = Image.open(io.BytesIO(image)).convert("RGB").crop(box)
        if ImageChops.difference(before, after).getbbox() is not None:
            raise _Refusal(
                Outcome.STALE, "the pixels at the target changed; look again"
            )
        return capture

    def control_at(self, target: ScreenTarget) -> str:
        """Return the observed control under a screenshot point, or "" (rule 6).

        Only the current input capture's points are answered. A canvas, a
        point on nothing, and an element no observation described give "",
        so the action keeps its conservative route-wide identity.
        """
        capture = self._input_capture
        if (
            target.point is None
            or capture is None
            or target.capture_id != capture.capture_id
        ):
            return ""
        found = self._screen_element(target.point)
        if found is None:
            return ""
        return self._control_of(found[0])

    def operation_binding(
        self,
        action: Action,
        observation: Observation | None,
        node: AxNode | None = None,
    ) -> BoundOperation | None:
        """Identify a declared operation only from an already gated observation."""
        if not self._profile.operations:
            return None
        route = self._profile.scope.route(
            observation.location if observation else self.location()
        )
        if route is None:
            return None
        if node is not None and observation is not None and node in observation.nodes:
            return operations.identify(self._profile, action, route, node=node)
        capture = self._input_capture
        target = action.target
        if (
            not isinstance(target, ScreenTarget)
            or capture is None
            or capture.capture_id != target.capture_id
        ):
            return None
        found = self._screen_element(target.point)
        if found is None or found[0].evaluate("el => el.tagName") != "CANVAS":
            return None
        if self._reader is None:
            self._reader = LocalReader()
        scale = Image.open(io.BytesIO(capture.image)).width / capture.view[2]
        return operations.identify(
            self._profile,
            action,
            route,
            lines=self._reader.lines(capture.image),
            frame=found[1],
            scale=scale,
        )

    def _check_point(
        self,
        action: Action,
        target: ScreenTarget,
        element: ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None,
    ) -> None:
        """Refuse screen input when the control under the point is not the one judged.

        The action was authorized as the observed control under its point.
        The same control, reusable target metadata, and submission must
        still be there a moment before input (rule 6).
        """
        if target.point is None or expect is None or not expect.control:
            return
        if self._control_of(element) != expect.control:
            raise _Refusal(Outcome.STALE, "a different control is under the point now")
        observed = expect.observed_control
        if (
            observed is None
            or observed.control != expect.control
            or observed.frame != frame
            or observed.secret
        ):
            raise _Refusal(Outcome.STALE, "the point has no observed control metadata")
        protected = self._protections()
        document = self._frame_for(frame)
        held = [item.handle for item, _ in protected if item.frame is document]

        async def matches() -> bool:
            return (
                await element._impl_obj.evaluate(
                    CONTROL,
                    arg=mapping.to_impl(
                        {
                            **self._script_options(held),
                            "observedControl": _control_spec(observed),
                        }
                    ),
                )
            ) is True

        try:
            current = self._bounded(document, matches())
        except Error as error:
            raise _Refusal(
                Outcome.STALE, "the control metadata could not be checked"
            ) from error
        if current is not True:
            raise _Refusal(
                Outcome.STALE, "the control metadata changed since observation"
            )
        self._check_submission(action, element, frame, expect)

    def _control_of(self, element: ElementHandle) -> str:
        """Return the id this adapter issued for ``element``, or ""."""
        if element.evaluate("el => el.tagName") == "CANVAS":
            return ""
        frame = element.owner_frame()
        known = self._known(frame) if frame is not None else []
        try:
            index = element.evaluate(KNOWN_AS, [c.handle for c in known])
        except Error:
            return ""
        return known[index].control_id if index >= 0 else ""

    def _screen_action(
        self, action: Action, target: ScreenTarget, expect: Expectation | None = None
    ) -> ActionResult:
        capture = self._screen_ready(target)
        if action.kind is ActionKind.WAIT:
            try:
                duration = int(str(action.value))
            except ValueError as error:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE, "wait requires milliseconds"
                ) from error
            if not 0 <= duration <= self._limits.operation_ms:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE, "wait exceeds the operation timeout"
                )
            self._page.wait_for_timeout(duration)
            return self._result(Outcome.OK)
        found = (
            capture.focus
            if target.point is None
            else self._screen_element(target.point)
        )
        if found is None:
            raise _Refusal(
                Outcome.BLOCKED, "the input target is outside the permitted surface"
            )
        element, frame = found
        if action.kind is ActionKind.READ:
            if target.point is None:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "read requires a point")
            if element.evaluate("el => el.tagName") == "CANVAS":
                # Every keystroke goes to the canvas, so the event reports a
                # changed field. The caller uses the expected value to judge
                # whether the painted line is a field or displayed text.
                text, shown = self._painted(capture, target.point)
                return self._result(Outcome.OK, extracted=text, screen=shown)
            self._check_shown_only(element, expect)
            return self._result(Outcome.OK, extracted=self._shown(element, frame))
        self._check_record(action, element, frame)
        self._check_point(action, target, element, frame, expect)
        if action.kind in {ActionKind.TYPE, ActionKind.PRESS_KEY}:
            if target.point is not None:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE,
                    "click to focus, then send keyboard input without a point",
                )
            self._screen_keyboard(action, element, frame, expect)
        else:
            if target.point is None:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "mouse input requires a point")
            self._screen_mouse(action, target.point, element, frame, expect)
        self._settle()
        return self._result(Outcome.OK)

    def _screen_keyboard(
        self,
        action: Action,
        element: ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None = None,
    ) -> None:
        if action.kind is ActionKind.TYPE and isinstance(action.value, SecretRef):
            self._secret_field(element)
            self._protect(element)
        if action.kind is ActionKind.TYPE:
            # Each character is at most one key and one edit.
            typed = len(self._value(action))
            claim = self._claimed(element, _TYPED, limits={"key": typed, "edit": typed})
        else:
            claim = self._claimed(element, _PRESSED, limits=_ONE_KEY)
        events = (*_KEY_EVENTS, "beforeinput", "input")
        guard = self._guard(
            action, element, frame, events, persistent=True, expect=expect
        )
        self._check_boundary(expect, guard)
        self._note_change(element)
        try:
            with self._sending(), claim:
                if action.kind is ActionKind.TYPE:
                    self._page.keyboard.type(self._value(action))
                else:
                    self._page.keyboard.press(str(action.value))
        finally:
            self._release(guard)

    def _screen_mouse(
        self,
        action: Action,
        point: Point,
        element: ElementHandle,
        frame: tuple[str, ...],
        expect: Expectation | None = None,
    ) -> None:
        mouse = self._page.mouse
        options = action.mouse
        pressed: list[str] = []
        events = (
            _CLICK_EVENTS
            if action.kind is ActionKind.CLICK
            else ("pointerdown", "mousedown")
        )
        if action.kind is ActionKind.DOUBLE_CLICK:
            events = (*_CLICK_EVENTS, "dblclick")
        guard = self._guard(action, element, frame, events, expect=expect)
        self._check_boundary(expect, guard)
        kinds, anywhere = _MOUSE_CLAIMS.get(action.kind, ((), False))
        try:
            with self._sending(), self._claimed(element, kinds, anywhere=anywhere):
                for key in options.modifiers:
                    self._page.keyboard.down(key)
                    pressed.append(key)
                match action.kind:
                    case ActionKind.CLICK | ActionKind.DOUBLE_CLICK:
                        count = 2 if action.kind is ActionKind.DOUBLE_CLICK else 1
                        mouse.click(
                            point.x,
                            point.y,
                            button=options.button.value,
                            click_count=count,
                        )
                    case ActionKind.MOVE:
                        mouse.move(point.x, point.y)
                    case ActionKind.SCROLL:
                        mouse.move(point.x, point.y)
                        mouse.wheel(options.delta.x, options.delta.y)
                        self._rest()
                    case ActionKind.DRAG:
                        self._screen_drag(action, point)
                    case _:
                        raise _Refusal(
                            Outcome.NOT_ACTIONABLE, "this action is not a mouse input"
                        )
        finally:
            for key in reversed(pressed):
                self._page.keyboard.up(key)
            self._release(guard)

    def _screen_drag(self, action: Action, start: Point) -> None:
        if not action.mouse.path:
            raise _Refusal(Outcome.NOT_ACTIONABLE, "drag requires a path")
        view = self._page.evaluate(VIEW)
        for point in action.mouse.path:
            if not 0 <= point.x < view[2] or not 0 <= point.y < view[3]:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "drag leaves the screenshot")
            if self._screen_element(point) is None:
                raise _Refusal(
                    Outcome.BLOCKED, "drag enters a surface outside the profile"
                )
        mouse = self._page.mouse
        mouse.move(start.x, start.y)
        mouse.down(button=action.mouse.button.value)
        try:
            for point in action.mouse.path:
                mouse.move(point.x, point.y)
        finally:
            mouse.up(button=action.mouse.button.value)

    def _visual_action(
        self, action: Action, expect: Expectation | None = None
    ) -> ActionResult:
        """Find a painted control again and click where it is now.

        Every step happens against the viewport as it currently stands. The
        canvas is found through the DOM, cut out of a picture of that
        viewport, and matched by pixels, so the offset and the click belong to
        the same moment. Nothing scrolls, because scrolling here would move
        the canvas after it had been measured.

        The point is hit tested before the click. A mouse event landing on
        something that covers the canvas is not the control being operated,
        and reporting that as done would be a claim this adapter cannot make.

        A matching crop says which painted button this is. It says nothing
        about which record the button acts on, so record evidence is checked
        against the canvas element exactly as it is for any other target.
        """
        target = action.target
        if not isinstance(target, VisualAnchor):
            raise _Refusal(Outcome.NOT_FOUND, "the action named no painted region")
        if action.kind is not ActionKind.CLICK:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE, "a painted region supports only a click"
            )
        anchor = self._anchors.get(target.anchor_id)
        if anchor is None:
            raise _Refusal(Outcome.NOT_FOUND, "that region was never captured")
        if anchor.capture != self._capture:
            raise _Refusal(Outcome.STALE, "that region belongs to an earlier capture")

        frame = self._frame_for(anchor.frame)
        canvas = frame.locator("canvas").nth(anchor.canvas)
        box = canvas.bounding_box(timeout=self._limits.operation_ms)
        if box is None:
            raise _Refusal(Outcome.NOT_ACTIONABLE, "the canvas is not rendered")
        offset_x, offset_y = self._match_in(anchor, _box(box))
        if not self._on_top(canvas, offset_x, offset_y):
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "something covers the painted control at that point",
            )
        guard = self._guard(action, canvas, anchor.frame, _CLICK_EVENTS, expect=expect)
        self._check_boundary(expect, guard)
        painted = canvas.element_handle(timeout=self._limits.operation_ms)
        with self._sending(guard), self._claimed(painted, _CLICKED):
            self._page.mouse.click(box["x"] + offset_x, box["y"] + offset_y)
        self._release(guard)
        self._settle()
        return self._result(Outcome.OK)

    def _match_in(self, anchor: _Anchor, box: visual.Box) -> tuple[float, float]:
        """Return where the stored crop sits inside the canvas, right now.

        The scene is cut from a picture of the current viewport, masked the
        same way an observation would be, so a canvas that has scrolled out of
        view refuses the action rather than being chased.
        """
        masks, _ = self._masks()
        image = self._page.screenshot(
            mask=masks,
            mask_color="#101010",
            timeout=self._limits.operation_ms,
            style=self._picture_style(),
        )
        scene = visual.cut_out(image, box)
        if scene is None:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "the canvas is not fully in view; scroll to it first",
            )
        match = visual.locate(anchor.crop, scene)
        if match.result is visual.MatchResult.AMBIGUOUS:
            raise _Refusal(Outcome.AMBIGUOUS, "the painted region matched twice")
        if match.result is not visual.MatchResult.FOUND or match.box is None:
            raise _Refusal(Outcome.NOT_FOUND, "the painted region is no longer there")
        return match.box.centre

    def _on_top(self, canvas: Locator, offset_x: float, offset_y: float) -> bool:
        """Report whether the canvas is the element a click there would reach."""
        try:
            return bool(
                canvas.evaluate(
                    ON_TOP,
                    {"dx": offset_x, "dy": offset_y},
                    timeout=self._limits.operation_ms,
                )
            )
        except Error:
            return False

    def _value(self, action: Action) -> str:
        if isinstance(action.value, SecretRef):
            variable = self._profile.secrets.get(action.value.name)
            if variable is None:
                raise _Refusal(Outcome.NOT_ACTIONABLE, "that secret is not declared")
            resolved = os.environ.get(variable)
            if not resolved:
                raise _Refusal(
                    Outcome.NOT_ACTIONABLE,
                    "the declared secret is not set in the environment",
                )
            return resolved
        return str(action.value)

    def _type(self, action: Action, field: ElementHandle) -> None:
        """Type a value, protecting the field first when the value is a secret."""
        value = self._value(action)
        secret = isinstance(action.value, SecretRef)
        if secret:
            self._secret_field(field)
            self._protect(field)
        # A fill sends one edit per line and a deletion key when it empties a
        # field. A text field must then hold the exact value. A secret field is
        # already protected, so its value is not returned.
        limits = {"key": 1, "edit": 2 * value.count("\n") + 1}
        claim = self._claimed(
            field, ("edit", "key"), limits=limits, value=None if secret else value
        )
        with self._sending(), claim:
            field.fill(value, timeout=self._limits.operation_ms)

    def _secret_field(self, field: ElementHandle) -> None:
        """Refuse a secret for anything but an input or a textarea.

        An input or textarea keeps its value in a property no text reading
        sees. An editable region keeps it as text in the document, where a
        name, an ancestor's text, or the page's own script would read it.
        """
        try:
            held = bool(field.evaluate("el => el.matches('input, textarea')"))
        except Error as error:
            raise _Refusal(Outcome.STALE, "the field could not be checked") from error
        if not held:
            raise _Refusal(
                Outcome.NOT_ACTIONABLE,
                "a secret can be typed only into an input or a textarea",
            )

    def _rest(self) -> None:
        """Wait for a scroll to come to rest before anything measures the page.

        A wheel event starts an animation and returns. Measuring a control
        while the page is still moving produces a position that was true for
        one frame, which is how a click ends up near a control instead of on
        it.
        """
        last: object = None
        for _ in range(10):
            current = self._page.evaluate("() => [window.scrollX, window.scrollY]")
            if current == last:
                return
            last = current
            self._page.wait_for_timeout(50)

    def _settle(self) -> None:
        """Wait for the page to load, and let the driver deliver its events.

        A popup, a dialog, or a download an action caused is announced by an
        event that can arrive just after the action returns. A short wait is
        when the driver delivers it, so the result that caused it can report it.
        """
        with contextlib.suppress(Error):
            self._page.wait_for_load_state(
                "domcontentloaded", timeout=self._limits.operation_ms
            )
            self._page.wait_for_timeout(SETTLE_MS)

    # Target resolution.

    def _locate(self, target: Target | None) -> ElementHandle:
        """Turn a target into exactly one control, or refuse it.

        Uniqueness is required rather than assumed. When a label repeats, the
        scope is what narrows it; taking the first match would make the run
        depend on document order, which is the thing that changes when a
        tenant reorders a table.
        """
        if target is None:
            raise _Refusal(Outcome.NOT_FOUND, "the action named no target")
        if isinstance(target, (VisualAnchor, ScreenTarget)):
            raise _Refusal(Outcome.NOT_ACTIONABLE, "a visual target is not a locator")
        found = self._resolve(target)
        count = len(found)
        if target.occurrence:
            if count <= target.occurrence:
                raise _Refusal(Outcome.NOT_FOUND, "that occurrence does not exist")
            return found[target.occurrence]
        if count == 0:
            raise _Refusal(Outcome.NOT_FOUND, "no control matched the target")
        if count > 1:
            raise _Refusal(Outcome.AMBIGUOUS, f"{count} controls matched the target")
        return found[0]

    def _resolve(self, target: AxLocator | DomLocator) -> list[ElementHandle]:
        """Return the visible elements ``target`` names, in observation order.

        The frame script compares the target with the role, name, scope,
        attribute, or class the collector reports, by the same functions, so
        every control an observation reports can be named back to it. No
        selector is built from a value, so a name with quotes, backslashes,
        or line breaks is matched like any other.
        """
        frame = self._frame_for(target.frame)
        # A secret field that left its page may have taken its value with it,
        # and a value locator could then test where it went, so nothing is
        # resolved until every protected field is accounted for.
        protected = self._protections()
        held = [item.handle for item, _ in protected if item.frame is frame]
        spec: dict[str, Any]
        if isinstance(target, AxLocator):
            spec = {"kind": "ax", "role": target.role, "name": target.name}
        else:
            spec = {
                "kind": "dom",
                "tag": target.tag,
                "attribute": target.attribute.value,
                "value": target.value,
            }
        spec["scope"] = _scope_spec(target.scope)
        try:
            result = frame.evaluate_handle(
                RESOLVE,
                {
                    **self._script_options(held),
                    "target": spec,
                    "limit": MAX_OCCURRENCE + 2,
                },
            )
            found = result.get_properties()
        except Error as error:
            raise _Refusal(
                Outcome.STALE, "the frame changed while the target was resolved"
            ) from error
        handles = [found[str(index)].as_element() for index in range(len(found))]
        return [handle for handle in handles if handle is not None]

    def _check_actionable(self, handle: ElementHandle, kind: ActionKind) -> None:
        try:
            if not handle.is_visible():
                raise _Refusal(Outcome.NOT_ACTIONABLE, "the control is not visible")
            if kind in _NEEDS_ENABLED and not handle.is_enabled():
                raise _Refusal(Outcome.NOT_ACTIONABLE, "the control is disabled")
        except Error as error:
            raise _Refusal(Outcome.NOT_ACTIONABLE, _brief(error)) from error

    def _result(
        self,
        outcome: Outcome,
        *,
        extracted: str | None = None,
        detail: str | None = None,
        screen: tuple[str, ...] = (),
    ) -> ActionResult:
        effects = tuple(self._side_effects)
        self._side_effects.clear()
        self._effects_from = 0
        return ActionResult(
            outcome=outcome,
            page_state=self._page_state(),
            extracted=extracted,
            detail=detail,
            side_effects=effects,
            dialog=self._dialog_info(),
            screen=screen,
        )


def _box(box: FloatRect) -> visual.Box:
    """Turn a driver bounding box into the integer box the matcher works in.

    Examples
    --------
    >>> _box({"x": 10.4, "y": 20.6, "width": 30.2, "height": 12.9})
    Box(left=10, top=20, width=30, height=13)
    """
    left, top = int(box["x"]), int(box["y"])
    right = int(box["x"] + box["width"])
    bottom = int(box["y"] + box["height"])
    return visual.Box(left=left, top=top, width=right - left, height=bottom - top)


_CLICK_EVENTS = ("pointerdown", "mousedown", "pointerup", "mouseup", "click")
"""The input events one click delivers, in order. The guard checks each."""

_KEY_EVENTS = ("keydown", "keypress")
"""The input events of a key press that decide what it does, in order.

A key's own default action, such as moving focus or submitting a form,
happens on these. The ``keyup`` goes wherever focus is by then, which after
Tab is the next field, so it is not judged.
"""

_SESSION_ACTIONS = frozenset({ActionKind.SCROLL, ActionKind.PRESS_KEY})
"""Actions this adapter performs on the page itself when they name no target."""

_ELEMENT_ACTIONS = frozenset(
    {
        ActionKind.CLICK,
        ActionKind.PRESS_KEY,
        ActionKind.TYPE,
        ActionKind.SELECT,
        ActionKind.READ,
        ActionKind.ASSERT,
        ActionKind.SCROLL,
    }
)
"""Actions this adapter performs on one element a structured target names."""

_NEEDS_ENABLED = frozenset(
    {ActionKind.CLICK, ActionKind.PRESS_KEY, ActionKind.TYPE, ActionKind.SELECT}
)

_STILL_WAITING = frozenset({Outcome.NOT_FOUND, Outcome.AMBIGUOUS, Outcome.STALE})
"""Resolution outcomes a wait_for keeps polling through until its deadline."""

_OPERATED = {
    ManualKind.CLICK: ActionKind.CLICK,
    ManualKind.KEY: ActionKind.PRESS_KEY,
    ManualKind.EDIT: ActionKind.TYPE,
    ManualKind.SELECT: ActionKind.SELECT,
}
"""The manual inputs that are an operation, and the action type each one is."""

_CLICKED = ("click", "select", "submit")
"""Recorder event types that one click can cause on its target element."""

_PRESSED = ("key", "click", "edit", "select", "submit")
"""Recorder event types that one key can cause on the focused element."""

_TYPED = ("key", "edit", "submit")
"""Recorder event types that typing can cause on the focused element."""

_ONE_KEY = {"key": 1, "edit": 1}
"""A single key sends one key and at most one edit."""

_MOUSE_CLAIMS: dict[ActionKind, tuple[tuple[str, ...], bool]] = {
    ActionKind.CLICK: (_CLICKED, False),
    ActionKind.DOUBLE_CLICK: (_CLICKED, False),
    ActionKind.SCROLL: (("scroll",), True),
    ActionKind.DRAG: (("click", "select"), True),
}
"""Input claims and element targeting for each mouse action."""

_KINDS = {
    "click": ManualKind.CLICK,
    "edit": ManualKind.EDIT,
    "select": ManualKind.SELECT,
    "key": ManualKind.KEY,
    "submit": ManualKind.SUBMIT,
    "scroll": ManualKind.SCROLL,
}
"""The manual-action category for each recorder report."""

_BEATS = frozenset({"hello", "heartbeat"})
"""Recorder reports that announce a document and its last event, not an input."""

_DETAILS = frozenset(
    {
        "",
        "confirm",
        "cancel",
        "move",
        "shortcut",
        "up",
        "down",
        "left",
        "right",
        "unprompted",
        "uncertain",
        "command",
    }
)
"""Every detail a recorder report may carry. Nothing else is accepted."""

_PAGE_KINDS = {
    ManualKind.NAVIGATION: Widget.PAGE,
    ManualKind.NAVIGATION_REFUSED: Widget.PAGE,
    ManualKind.PAGE_OPENED: Widget.PAGE,
    ManualKind.PAGE_CLOSED: Widget.PAGE,
    ManualKind.SCROLL: Widget.PAGE,
    ManualKind.DIALOG: Widget.DIALOG,
}
"""The control an event that names no element was made on."""

_COUNT = 1_000_000_000
"""The largest sequence number a recorder report may carry."""

_LATEST = 10**15
"""The latest time a recorder report may carry, in milliseconds since the epoch."""

_EVENT: dict[str, tuple[type | tuple[type, ...], int]] = {
    "token": (str, 64),
    "doc": (str, 64),
    "kind": (str, 16),
    "seq": (int, _COUNT),
    "at": ((int, float), _LATEST),
    "tag": (str, 64),
    "role": (str, 64),
    "name": (str, 0),
    "type": (str, 32),
    "secret": (bool, 0),
    "editable": (bool, 0),
    "auto": (bool, 0),
    "detail": (str, 16),
    "url": (str, 2000),
}
"""Every field of a recorder's input report, its type, and its longest length.

A name's longest length is the adapter's own name limit.
"""

_BEAT = {key: _EVENT[key] for key in ("token", "doc", "kind", "at")} | {
    "last": (int, _COUNT)
}
"""Every field of a recorder's announcement of its last event."""


class TargetMarker:
    """Colors the proposal a docked run waits on, amber, until a person answers.

    A channel that only draws. It sends no commands and never counts as a
    person who can answer, so ``listening`` is False. ``show`` runs on the
    owner thread, which is the only thread that may drive the page.
    """

    def __init__(self, surface: BrowserSurface) -> None:
        self._surface = surface
        self._control: Control | None = None

    def attach(self, control: Control) -> None:
        """Read the pending proposal from this control."""
        self._control = control

    def show(self, status: Status) -> None:
        """Mark the proposal an approval request is about."""
        control = self._control
        proposal = control.proposal if control is not None else None
        if proposal is not None and status.offer is not None:
            self._surface.preview(proposal, waiting=True)

    def listening(self) -> bool:
        """Report False: nobody sends commands through a highlight."""
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class _Report:
    """One recorder report that passed every check. ``kind`` is None for a beat.

    A beat announces the document and the last event it sent; ``seq`` holds
    that number, and ``hello`` says the document has just started.
    """

    doc: str
    seq: int
    at: float
    kind: ManualKind | None = None
    hello: bool = False
    tag: str = ""
    role: str = ""
    name: str = ""
    secret: bool = False
    editable: bool = False
    detail: Detail = Detail.NONE
    url: str = ""
    auto: bool = False


def _parse(payload: tuple[object, ...], token: str, max_name: int) -> _Report | None:
    """Return a recorder report if it has exactly the expected shape, or None.

    The fields, their types, their lengths, and the kind and detail values
    are all checked, and a report without the session's token is refused
    before anything else is looked at.

    Examples
    --------
    >>> _parse(({"token": "x", "kind": "hello"},), "t", 10) is None
    True
    >>> hello = {"token": "t", "kind": "hello", "doc": "d", "last": 0, "at": 5}
    >>> report = _parse((hello,), "t", 10)
    >>> report.hello, report.doc, report.kind
    (True, 'd', None)
    """
    if len(payload) != 1 or not isinstance(payload[0], dict):
        return None
    message: dict[str, Any] = payload[0]
    if message.get("token") != token:
        return None
    kind = message.get("kind")
    fields = _BEAT if kind in _BEATS else _EVENT if kind in _KINDS else None
    if fields is None or set(message) != set(fields):
        return None
    for key, (kinds, limit) in fields.items():
        value = message[key]
        if isinstance(value, bool) != (kinds is bool) or not isinstance(value, kinds):
            return None
        longest = max_name if key == "name" else limit
        if isinstance(value, str) and len(value) > longest:
            return None
        if kinds is not bool and kinds is not str and not 0 <= value <= limit:
            return None
    if not message["doc"] or message.get("detail", "") not in _DETAILS:
        return None
    if kind in _BEATS:
        return _Report(
            message["doc"], message["last"], float(message["at"]), hello=kind == "hello"
        )
    return _Report(
        message["doc"],
        message["seq"],
        float(message["at"]),
        _KINDS[str(kind)],
        tag=message["tag"],
        role=message["role"],
        name=message["name"],
        secret=message["secret"],
        editable=message["editable"],
        detail=Detail(message["detail"]),
        url=message["url"],
        auto=message["auto"],
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Slice:
    """The part of a structured screen one frame is asked to describe."""

    skip: int
    limit: int
    scope: Scope | None


def _connected(handle: ElementHandle) -> bool:
    """Report whether a pinned element is still in its document."""
    try:
        return bool(handle.evaluate("el => el.isConnected"))
    except Error:
        return False


def _scope_spec(scope: Scope | None) -> dict[str, str] | None:
    """Write a scope the way the frame scripts compare it."""
    if scope is None:
        return None
    return {"kind": scope.kind.value, "name": scope.name}


def _control_spec(node: AxNode) -> dict[str, Any]:
    return {
        "role": node.role,
        "name": node.name,
        "tag": node.tag,
        "scope": _scope_spec(node.scope),
        "rowNames": list(node.row_names),
        "rowColumns": list(node.row_columns),
        "slot": node.slot,
        "attributes": [
            (name, value) for name, value in node.attributes if name != "value"
        ],
        "classes": list(node.classes),
    }


def _frame_names(frame: Frame) -> tuple[str, ...]:
    """Return a frame's chain of plain names, outermost first."""
    names: list[str] = []
    current: Frame | None = frame
    while current is not None and current.parent_frame is not None:
        names.append(current.name)
        current = current.parent_frame
    names.reverse()
    return tuple(names)


def _node(
    item: dict[str, Any], path: tuple[str, ...], ids: tuple[list[str], list[str]]
) -> AxNode:
    known, fresh = ids

    def named(ref: list[Any] | None) -> str:
        if not ref:
            return ""
        return known[ref[1]] if ref[0] == "known" else fresh[ref[1]]

    def pair(refs: list[Any] | None) -> str:
        return f"{named(refs[0])}>{named(refs[1])}" if refs else ""

    def framed(description: str | None) -> str:
        return f"{'/'.join(path)}#{description}" if description else ""

    scope = item.get("scope")
    return AxNode(
        role=item["role"],
        name=item["name"],
        value=item["value"],
        frame=path,
        tag=item["tag"],
        attributes=tuple((name, value) for name, value in item["attributes"]),
        interactions=tuple(_kinds(item["interactions"])),
        scope=Scope(ScopeKind(scope["kind"]), scope["name"]) if scope else None,
        context=tuple(str(entry) for entry in item.get("context") or ()),
        row=named(item.get("row")) or None,
        ancestors=tuple(named(entry) for entry in item.get("ancestors") or ()),
        slot=str(item.get("slot") or ""),
        classes=tuple(str(name) for name in item.get("classes") or ()),
        in_view=bool(item.get("inView", True)),
        control=named(item.get("control")),
        form=named(item.get("form")),
        submits=pair(item.get("submits")),
        enter=pair(item.get("enter")),
        form_as=framed(item.get("formAs")),
        submits_as=framed(item.get("submitsAs")),
        enter_as=framed(item.get("enterAs")),
        enabled=bool(item["enabled"]),
        visible=bool(item["visible"]),
        shadow=bool(item["shadow"]),
        secret=bool(item["secret"]),
        options=tuple(SelectOption(**option) for option in item.get("options", ())),
        options_complete=bool(item.get("optionsComplete", True)),
        selections=tuple(
            (name, bool(ready)) for name, ready in item.get("selections", ())
        ),
        row_names=tuple(str(name) for name in (scope or {}).get("names") or ()),
        row_columns=tuple(
            int(column)
            for column in (scope or {}).get("columns") or ()
            if isinstance(column, int)
        ),
    )


def _operation_of(
    item: _Heard, identity: dict[str, Any], ids: list[str], profile: Profile
) -> Operation | None:
    """Describe a person's input as the operation an automated one would be.

    A control the adapter never observed has no id, and a submission that
    involves one is left out, so only the description can match it. A key
    is known only as a category, so only Enter and Escape name their key.
    """
    kind = _OPERATED.get(item.kind)
    if kind is None:
        return None

    def named(at: int | None) -> str:
        return ids[at] if at is not None and 0 <= at < len(ids) else ""

    def pair(refs: list[int | None] | None) -> str:
        # The collector writes a missing button as nothing after the ">". A
        # part the adapter never observed has no id to write at all.
        if not refs or any(at is not None and not named(at) for at in refs):
            return ""
        return f"{named(refs[0])}>{named(refs[1])}"

    def framed(description: str) -> str:
        return f"{'/'.join(item.frame)}#{description}" if description else ""

    key = {Detail.CONFIRM: ENTER, Detail.CANCEL: "Escape"}.get(item.detail, "")
    pressing = kind is ActionKind.PRESS_KEY
    node = AxNode(
        str(identity["role"]),
        str(identity["name"]),
        frame=item.frame,
        tag=str(identity["tag"]),
        control=named(identity["control"]),
    )
    submission = submission_as = ""
    if kind is ActionKind.CLICK:
        submission = pair(identity["submits"])
        submission_as = framed(str(identity["submitsAs"]))
    elif pressing and key == ENTER:
        submission = pair(identity["enter"])
        submission_as = framed(str(identity["enterAs"]))
    return Operation(
        kind=kind,
        route=policy.route_for(profile, item.url) or "",
        target=describe(node),
        control=node.control,
        key=key if pressing else "",
        submission=submission,
        form=named(identity["form"]),
        submission_as=submission_as,
        form_as=framed(str(identity["formAs"])),
    )


def _kinds(names: Sequence[str]) -> Iterator[ActionKind]:
    for name in names:
        try:
            yield ActionKind(name)
        except ValueError:
            continue


def _brief(error: Error) -> str:
    """Return a short, live-only description of a driver failure."""
    return str(error).splitlines()[0][:160]
