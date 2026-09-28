"""Test a person taking, using, and returning the live browser session.

The browser, pages, and recorder are real. Headless Chromium serves pages
written here, and the recorder reports through the session binding. A script
acts for the person through the driver on its owner thread, between the
surface's calls. Its input reaches the page as trusted input, as a person's
would. Every assertion about the surface also checks the page or recording.
"""

from __future__ import annotations

import contextlib
import dataclasses
import faulthandler
import io
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml
from pages import serve_pages
from PIL import Image, ImageChops
from playwright.sync_api import ElementHandle, Page

from computeruse import browser, policy
from computeruse.actions import (
    Action,
    AxLocator,
    AxNode,
    DomAttribute,
    DomLocator,
    Expectation,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Operation,
    Outcome,
    Point,
    RecordEvidence,
    Relation,
    ScreenTarget,
)
from computeruse.browser import STOPPED, BrowserSurface, open_session
from computeruse.budget import Budget
from computeruse.control import (
    Control,
    Order,
    Receipt,
    Recording,
    SessionProbe,
    Status,
)
from computeruse.decider import (
    AskHuman,
    CheckKind,
    Decision,
    Finish,
    Observe,
    Propose,
    ResultCheck,
    Task,
    Transcript,
)
from computeruse.escalation import (
    Ask,
    CheckFailure,
    Command,
    HandoffOutcome,
    InterventionRequest,
    Mode,
    Owner,
    State,
    Trigger,
    Verdict,
    Via,
)
from computeruse.journal import (
    Acted,
    Commanded,
    HandedBack,
    ManualAction,
    MemoryJournal,
    Observed,
)
from computeruse.loop import Ending, discover
from computeruse.manual import (
    Detail,
    Gap,
    GapKind,
    ManualEvent,
    ManualKind,
    Requirement,
    Widget,
    classify,
    gaps_in,
)
from computeruse.panel import ControlWindow
from computeruse.policy import Restriction, Source
from computeruse.profile import ActionKind, Limit, Profile, load_profile

parse = browser._parse
SITE_PROFILE = Path(__file__).resolve().parents[1] / "evaluation" / "profile.yaml"
STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)
VISUAL = ObservationRequest(ObservationMode.VISUAL)

SHORT_SECRET = "Kq7w"
LONG_SECRET = "Zp9xZp9xZp9x"

DESK = """<!doctype html><html><body><h1>Member desk</h1>
<button id="approve" onclick="window.approved = (window.approved || 0) + 1">
  Approve</button>
<input id="pin" type="password" aria-label="PIN">
<input id="code" type="password" aria-label="Code">
<input id="note" aria-label="Note"
       style="width: 320px; height: 48px; font-size: 32px">
<label><input id="agree" type="checkbox"> Agree</label>
<div id="member" role="S-1001" tabindex="0">Member 1001</div>
<a id="hash" href="#later">Later</a>
<a id="next" href="/next">Next</a>
<div style="height: 3000px"></div>
<p id="later">Later section</p>
</body></html>"""

NEXT = "<!doctype html><html><body><h1>Next</h1></body></html>"


@pytest.fixture
def recorded(tmp_path: Path):
    """Return a helper that opens a recorded session on written pages."""

    @contextlib.contextmanager
    def _open(
        written: dict[str, str], path: str = "/", **edits: object
    ) -> Iterator[tuple[BrowserSurface, Profile]]:
        with serve_pages(written) as base:
            document = yaml.safe_load(SITE_PROFILE.read_text())
            document["base_url"] = base
            document["allow_routes"] = ["/**"]
            document["deny_routes"] = ["/forbidden/**"]
            document.update(edits)
            destination = tmp_path / "recorded.yaml"
            destination.write_text(yaml.safe_dump(document))
            profile = load_profile(destination)
            with open_session(profile, f"{base}{path}", record=True) as surface:
                yield surface, profile

    return _open


@contextlib.contextmanager
def bounded(seconds: float) -> Iterator[None]:
    """Fail the whole run loudly, rather than hang, if a block overruns."""
    faulthandler.dump_traceback_later(seconds, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def hand_over(surface: BrowserSurface) -> None:
    """Stop the automation and give the person the session, as the control does."""
    surface.transfer(Owner.OPERATOR, "iv-1")
    surface.transfer(Owner.HUMAN, "iv-1")


def sanitized(event: ManualEvent) -> tuple[object, ...]:
    """Return what a journal entry for ``event`` carries besides its numbering."""
    return (
        event.kind,
        event.control,
        event.route,
        event.position,
        event.secret,
        event.owned,
        event.detail,
    )


def region(image: bytes, box: tuple[int, int, int, int]) -> Image.Image:
    return Image.open(io.BytesIO(image)).convert("RGB").crop(box)


# What the recorder reports.


def test_a_persons_input_is_reported_in_order_as_categories_only(recorded) -> None:
    with recorded({"/": DESK, "/next": NEXT}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#approve")
        page.click("#pin")
        page.keyboard.type(SHORT_SECRET)
        page.click("#code")
        page.keyboard.type(LONG_SECRET)
        page.click("#agree")
        page.click("#note")
        page.keyboard.press("Enter")
        page.keyboard.press("Escape")
        page.keyboard.press("Tab")
        page.mouse.wheel(0, 400)
        page.click("#hash")
        page.wait_for_timeout(100)
        page.evaluate("window.scrollTo(0, 0)")
        page.click("#next")
        page.wait_for_load_state("domcontentloaded")
        recording = surface.hand_back()

    events = recording.events
    assert [(event.kind, event.detail) for event in events] == [
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.EDIT, Detail.NONE),
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.EDIT, Detail.NONE),
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.SELECT, Detail.NONE),
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.KEY, Detail.CONFIRM),
        (ManualKind.KEY, Detail.CANCEL),
        (ManualKind.KEY, Detail.MOVE),
        (ManualKind.SCROLL, Detail.DOWN),
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.NAVIGATION, Detail.NONE),
        (ManualKind.CLICK, Detail.NONE),
        (ManualKind.NAVIGATION, Detail.NONE),
    ]
    sequences = [event.sequence for event in events]
    assert sequences == sorted(set(sequences))
    assert all(event.owned for event in events)
    assert events[0].target == AxLocator("button", "Approve")
    assert events[6].target == AxLocator("checkbox", "Agree")
    assert events[12].target == AxLocator("link", "Later")
    assert events[13].location.endswith("#later")
    assert events[15].location.endswith("/next")
    assert recording.gaps == ()
    assert recording.unprotected == 0


EDITORS = """<!doctype html><html><body><h1>Notes</h1>
<label for="memo">Memo</label><input id="memo" type="text">
<div id="rich" contenteditable="true" role="textbox" aria-label="Rich"></div>
</body></html>"""


def test_a_field_is_named_by_its_label_and_an_editable_region_is_not(
    recorded,
) -> None:
    """A field's label is not what was typed, so an edit there can be found again.

    An editable region's name is its own text, which is the typed value, so it
    is never named and the step goes to a person on every replay.
    """
    with recorded({"/": EDITORS}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#memo")
        page.keyboard.type("Paid in full")
        page.click("#rich")
        page.keyboard.type("Paid in full")
        recording = surface.hand_back()

    edits = [event for event in recording.events if event.kind is ManualKind.EDIT]
    assert [event.target for event in edits] == [AxLocator("textbox", "Memo"), None]
    assert "Paid in full" not in repr(recording)


NESTED = """<!doctype html><html><body><h1>Notes</h1>
<div role="main"><p>Memo</p><div id="rich" contenteditable="true"></div>
<p id="below">click here</p></div>
</body></html>"""


def test_text_typed_into_an_editable_region_does_not_name_what_holds_it(
    recorded,
) -> None:
    """A clicked ancestor would be named by its text, typed text included."""
    with recorded({"/": NESTED}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#rich")
        page.keyboard.type("OTP-3141")
        page.click("#below")
        recording = surface.hand_back()

    clicks = [event for event in recording.events if event.kind is ManualKind.CLICK]
    assert clicks
    assert "OTP-3141" not in repr(recording)
    assert all(event.target is None for event in clicks)


def test_a_short_and_a_long_secret_are_recorded_identically(recorded) -> None:
    with recorded({"/": DESK}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#pin")
        page.keyboard.type(SHORT_SECRET)
        page.click("#code")
        page.keyboard.type(LONG_SECRET)
        recording = surface.hand_back()

    short, long = recording.events[:2], recording.events[2:]
    assert [event.kind for event in short] == [ManualKind.CLICK, ManualKind.EDIT]
    assert [sanitized(event) for event in short] == [sanitized(event) for event in long]
    assert all(event.secret and event.target is None for event in recording.events)
    written = repr(recording)
    assert SHORT_SECRET not in written
    assert LONG_SECRET not in written


def test_the_automations_own_input_is_not_a_persons(recorded) -> None:
    with recorded({"/": DESK}) as (surface, _):
        note = AxLocator("textbox", "Note")
        results = [
            surface.act(Action(ActionKind.CLICK, AxLocator("button", "Approve"))),
            surface.act(Action(ActionKind.TYPE, note, "hello")),
            surface.act(Action(ActionKind.PRESS_KEY, note, "Tab")),
            surface.act(Action(ActionKind.SCROLL, value="down")),
        ]
        surface.idle(0.5)
        heard = surface.activity()
        observation = surface.observe(STRUCTURED)
        approved = surface._page.evaluate("window.approved")

    assert [result.outcome for result in results] == [Outcome.OK] * 4
    assert approved == 1
    assert heard == ()
    typed = next(node for node in observation.nodes if node.name == "Note")
    assert typed.value == "hello"
    assert not typed.secret


def test_a_trial_click_before_a_real_one_is_not_a_persons(site, tmp_path) -> None:
    """The driver's trial click delivers a trusted click the recorder hears.

    It comes before the real click's own window, so only fencing the whole
    action keeps it from being read as a person's click, which would stop a
    run that nobody touched.
    """
    document = yaml.safe_load(SITE_PROFILE.read_text())
    document["base_url"] = site
    destination = tmp_path / "site.yaml"
    destination.write_text(yaml.safe_dump(document))
    profile = load_profile(destination)
    payee = DomLocator("dd", DomAttribute.ID, "payee-number")
    proof = RecordEvidence(payee, "10001", Relation.CONTAINER)
    send = DomLocator("button", DomAttribute.ID, "send-payment")
    with open_session(profile, f"{site}/payments", record=True) as surface:
        surface.transfer(Owner.AUTOMATION, "")
        results = []
        for _ in range(3):
            seen = surface.observe(STRUCTURED)
            click = Action(ActionKind.CLICK, send, evidence=proof, effect="pay")
            results.append(surface.act(click, expect=Expectation(seen.page_state)))
        surface.idle(0.5)
        heard = surface.activity()
        sent = surface._page.locator("#payments-sent").inner_text()

    assert [result.outcome for result in results] == [Outcome.OK] * 3
    assert sent == "10001 10001 10001"
    assert heard == ()


def test_a_page_script_cannot_add_an_event(recorded, monkeypatch) -> None:
    forger = """() => {
      const forged = { kind: 'click', seq: 1, at: Date.now(), tag: 'button',
        role: 'button', name: 'Forged', type: '', secret: false,
        editable: false, detail: '', url: location.href, doc: 'forged' };
      let calls = 0;
      for (const name of Object.getOwnPropertyNames(window)) {
        let value;
        try { value = window[name]; } catch (error) { continue; }
        if (typeof value !== 'function' || name === 'close' || name === 'print' ||
            name === 'alert' || name === 'confirm' || name === 'prompt' ||
            name === 'open' || name === 'stop' || name === 'focus') { continue; }
        try { value(forged); calls += 1; } catch (error) { /* not callable */ }
      }
      const controller = window.__playwright__binding__controller__;
      if (controller && controller._bindings) {
        for (const name of controller._bindings.keys()) {
          for (const guess of ['', 'token', name]) {
            try {
              controller.callBinding(name, Object.assign({ token: guess }, forged))
                .catch(() => {});
              calls += 1;
            } catch (error) { /* refused */ }
          }
        }
      }
      document.getElementById('approve').click();
      return calls;
    }"""
    refused: list[object] = []

    def checked(payload, token, max_name):
        report = parse(payload, token, max_name)
        if report is None:
            refused.append(payload)
        return report

    monkeypatch.setattr(browser, "_parse", checked)
    with recorded({"/": DESK}) as (surface, _):
        hand_over(surface)
        calls = surface._page.evaluate(forger)
        surface.idle(0.5)
        heard = surface.activity()

    assert calls > 0
    assert any("Forged" in repr(payload) for payload in refused)
    assert heard == ()


def test_a_page_chosen_role_reaches_the_journal_only_as_other(
    recorded,
) -> None:
    journal = MemoryJournal()
    with recorded({"/": DESK}) as (surface, profile):
        control = Control(
            mode=Mode.DISCOVERY, clock=time.monotonic, seat=surface, worker=False
        )
        control.begin(
            profile=profile,
            budget=Budget(profile.budgets, time.monotonic),
            journal=journal,
            context="review the member",
        )
        surface._page.click("#member")
        surface.idle(0.5)
        control.halted()
        control.finish("completed")

    manual = [event for event in journal.events if isinstance(event, ManualAction)]
    assert len(manual) == 1
    assert manual[0].control is Widget.OTHER
    assert not manual[0].owned
    assert "S-1001" not in repr(journal.events)
    assert "Member 1001" not in repr(journal.events)


# Through the run's control and the real loop.


@dataclasses.dataclass
class Sitting:
    """The browser seat with a scripted operator glancing at it between slices.

    Each wait slice the control takes offers the next move the current status.
    A move sends a command or acts in the page through the driver, on the
    owner thread, the way a person at the control window and the session
    would. Everything else goes straight to the real seat.
    """

    seat: BrowserSurface
    moves: list[Callable[[Sitting, Status], bool]]
    control: Control | None = None

    def where(self) -> str:
        return "the test window"

    def idle(self, seconds: float) -> None:
        self.seat.idle(seconds)
        if (
            self.control is not None
            and self.moves
            and self.moves[0](self, self.control.status())
        ):
            self.moves.pop(0)

    def transfer(self, owner: Owner, intervention: str) -> None:
        self.seat.transfer(owner, intervention)

    def activity(self) -> tuple[ManualEvent, ...]:
        return self.seat.activity()

    def hand_back(self, *, revoke: bool = True) -> Recording:
        return self.seat.hand_back(revoke=revoke)

    def probe(self) -> SessionProbe:
        return self.seat.probe()

    def watch(self, permitted: Callable[[], bool]) -> None:
        self.seat.watch(permitted)


def command(state: State, sent: Command) -> Callable[[Sitting, Status], bool]:
    """Send ``sent`` once the run reaches ``state``."""

    def move(sitting: Sitting, status: Status) -> bool:
        if status.state is not state or sitting.control is None:
            return False
        offer = status.offer
        sitting.control.submit(
            Order(
                sent,
                status.run,
                status.revision,
                offer.intervention if offer is not None else "",
            )
        )
        return True

    return move


def typing(state: State, text: str) -> Callable[[Sitting, Status], bool]:
    """Type ``text`` into the Note field once the run reaches ``state``."""

    def move(sitting: Sitting, status: Status) -> bool:
        if status.state is not state:
            return False
        sitting.seat._page.click("#note")
        sitting.seat._page.keyboard.type(text)
        return True

    return move


@dataclasses.dataclass
class Looking:
    """A decider that looks at the page twice and then asks for a person."""

    seen: list[Transcript] = dataclasses.field(default_factory=list)

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        """Supply the task for this scripted control test."""
        del goal, notices
        return Task()

    def decide(self, transcript: Transcript) -> Decision:
        self.seen.append(transcript)
        if len(self.seen) < 3:
            return Observe(STRUCTURED, "read the desk")
        return AskHuman(Trigger.NO_PROGRESS, "nothing left to do")


class Watching:
    """A channel an operator is watching, so the run waits for them."""

    def attach(self, control: Control) -> None:
        del control

    def show(self, status: Status) -> None:
        del status

    def listening(self) -> bool:
        return True


def test_a_person_handed_the_run_leaves_nothing_readable_behind(recorded) -> None:
    journal = MemoryJournal()
    decider = Looking()
    with recorded(
        {"/": DESK}, escalation={"handoff_timeout_s": 30, "on_timeout": "abort"}
    ) as (
        surface,
        profile,
    ):
        sitting = Sitting(
            surface,
            [
                command(State.RUNNING, Command.TAKE_CONTROL),
                typing(State.HUMAN_CONTROL, "PERSON-SECRET"),
                command(State.HUMAN_CONTROL, Command.RESUME),
                command(State.PAUSED, Command.TERMINATE),
            ],
        )
        control = Control(
            mode=Mode.DISCOVERY,
            clock=time.monotonic,
            seat=sitting,
            channels=[Watching()],
            worker=False,
        )
        sitting.control = control
        result = discover(
            "review the desk",
            profile,
            surface=surface,
            decider=decider,
            journal=journal,
            clock=time.monotonic,
            control=control,
        )
        note = surface._page.locator("#note").input_value()

    assert note == "PERSON-SECRET"
    assert result.ending is Ending.TERMINATED
    handed = [event for event in journal.events if isinstance(event, HandedBack)]
    assert [event.passed for event in handed] == [True]
    manual = [event for event in journal.events if isinstance(event, ManualAction)]
    assert [event.kind for event in manual] == [ManualKind.CLICK, ManualKind.EDIT]
    assert all(event.owned for event in manual)
    assert any(
        isinstance(event, Observed) and event.status is ObservationStatus.COMPLETE
        for event in journal.events[journal.events.index(handed[0]) :]
    )
    assert "PERSON-SECRET" not in repr(decider.seen)
    assert "PERSON-SECRET" not in repr(journal.events)
    assert [step.event.kind for step in result.segments[0].steps] == [
        ManualKind.CLICK,
        ManualKind.EDIT,
    ]


# Protecting what a person typed.


def test_a_person_typed_field_is_withheld_and_masked_after_hand_back(
    recorded,
) -> None:
    with recorded({"/": DESK}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#note")
        page.keyboard.type("PLAIN9999")
        box = page.locator("#note").bounding_box()
        assert box is not None
        inside = (
            int(box["x"]) + 3,
            int(box["y"]) + 3,
            int(box["x"] + box["width"]) - 3,
            int(box["y"] + box["height"]) - 3,
        )
        unmasked = region(page.screenshot(), inside)
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)
        picture = surface.observe(VISUAL)

    assert recording.unprotected == 0
    assert structured.status is ObservationStatus.COMPLETE
    note = next(node for node in structured.nodes if node.name == "Note")
    assert note.secret
    assert note.value is None
    assert "PLAIN9999" not in repr(structured)
    assert picture.image is not None
    masked = region(picture.image, inside)
    assert ImageChops.difference(unmasked, masked).getbbox() is not None
    assert masked.getcolors() == [(masked.width * masked.height, (16, 16, 16))]


REDRAWN = """<!doctype html><html><body><h1>Search</h1>
<form id="search"><input id="member" aria-label="Member"></form>
<button id="redraw" onclick="document.getElementById('search').outerHTML =
  '<form id=search><input id=member aria-label=Member></form>'">Redraw</button>
</body></html>"""


def test_a_typed_field_a_redraw_removes_is_unprotected_until_the_page_goes(
    recorded,
) -> None:
    """A second hand-back is asked again, and fails again, until a reload.

    Nothing about handing back twice says where the value went, so only the
    document going can let the field go.
    """
    with recorded({"/": REDRAWN}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#member")
        page.keyboard.type("10001")
        page.click("#redraw")
        refused = surface.hand_back()
        lost = surface.probe().protection_lost
        surface.transfer(Owner.AUTOMATION, "iv-1")
        blind = surface.observe(STRUCTURED)
        hand_over(surface)
        again = surface.hand_back()
        still = surface.probe().protection_lost
        hand_over(surface)
        page.reload()
        page.wait_for_load_state("domcontentloaded")
        reloaded = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        after = surface.probe().protection_lost
        seeing = surface.observe(STRUCTURED)

    assert refused.unprotected == 1
    assert lost
    assert blind.status is ObservationStatus.FAILED
    assert again.unprotected == 1
    assert still
    assert reloaded.unprotected == 0
    assert not after
    assert seeing.status is ObservationStatus.COMPLETE


def test_a_marked_field_a_redraw_removes_is_lost_until_the_page_goes(
    recorded,
) -> None:
    with recorded({"/": REDRAWN}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#member")
        page.keyboard.type("10001")
        marked = surface.hand_back()
        hand_over(surface)
        page.click("#redraw")
        surface.idle(0.3)
        lost = surface.probe().protection_lost
        surface.hand_back()
        again = surface.probe().protection_lost
        hand_over(surface)
        page.reload()
        page.wait_for_load_state("domcontentloaded")
        surface.hand_back()
        after = surface.probe().protection_lost

    assert marked.unprotected == 0
    assert lost
    assert again
    assert not after


# Dialogs, idling, and stopping.

ASKING = """<!doctype html><html><body><h1>Queue</h1>
<button id="ask">Ask</button></body></html>"""


def test_a_dialog_a_person_answers_is_recorded_and_released(recorded) -> None:
    with recorded({"/": ASKING}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.evaluate(
            "setTimeout(() => { document.title = confirm('Proceed?') ? 'y' : 'n' })"
        )
        surface.idle(0.3)
        held = "page-1" in surface._pending
        with bounded(30):
            started = time.monotonic()
            probe = surface.probe()
            recording = surface.hand_back()
            waited = time.monotonic() - started
        hand_over(surface)
        surface._pending["page-1"].dialog.accept()
        surface.idle(0.3)
        answered = surface.activity()
        released = "page-1" not in surface._pending
        title = page.title()

    assert held
    assert probe.pages[0].dialog
    assert recording.unprotected == 0
    assert waited < 3
    assert [event.kind for event in answered] == [ManualKind.DIALOG]
    assert answered[0].control is Widget.DIALOG
    assert answered[0].detail is Detail.ACCEPTED
    assert answered[0].dialog == "dialog-1"
    assert released
    assert title == "y"


def test_idle_delivers_binding_calls_after_every_session_page_closes(
    recorded,
) -> None:
    got: list[object] = []
    with recorded({"/": ASKING}) as (surface, _):
        surface._page.close()
        side = surface.browser.new_context()
        try:
            side.expose_binding("__side", lambda _source, value: got.append(value))
            window = side.new_page()
            window.set_content("<p>side</p>")
            window.evaluate("setTimeout(() => window.__side('hello'), 100)")
            surface.idle(0.6)
        finally:
            side.close()

    assert got == ["hello"]


COVERED = """<!doctype html><html><body><h1>Queue</h1>
<button id="late" onclick="window.clicks = (window.clicks || 0) + 1">Send</button>
<div id="cover" style="position: fixed; inset: 0; background: white"></div>
<script>setTimeout(() => document.getElementById('cover').remove(), 900)</script>
</body></html>"""


def test_a_stop_during_a_clicks_wait_sends_nothing(recorded) -> None:
    allowed = [True]
    asked: list[bool] = []

    def permitted() -> bool:
        asked.append(allowed[0])
        return allowed[0]

    with recorded({"/": COVERED}) as (surface, _):
        surface.watch(permitted)
        stop = threading.Timer(0.3, allowed.__setitem__, (0, False))
        stop.start()
        try:
            result = surface.act(Action(ActionKind.CLICK, AxLocator("button", "Send")))
        finally:
            stop.cancel()
        surface._page.wait_for_timeout(300)
        clicks = surface._page.evaluate("window.clicks || 0")

    assert result.outcome is Outcome.BLOCKED
    assert result.detail == STOPPED
    assert asked == [False]
    assert clicks == 0


def test_the_same_pages_survive_a_hand_over_and_back(recorded) -> None:
    with recorded({"/": DESK}) as (surface, _):
        context = surface._page.context
        pages = list(context.pages)
        hand_over(surface)
        withheld = surface.observe(STRUCTURED)
        refused = surface.act(Action(ActionKind.CLICK, AxLocator("button", "Approve")))
        surface._page.click("#approve")
        surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        clicked = surface.act(Action(ActionKind.CLICK, AxLocator("button", "Approve")))
        approved = surface._page.evaluate("window.approved")
        same = surface._page.context is context and list(context.pages) == pages
        listed = context in surface.browser.contexts

    assert withheld.status is ObservationStatus.UNAVAILABLE
    assert withheld.nodes == ()
    assert refused.outcome is Outcome.BLOCKED
    assert clicked.outcome is Outcome.OK
    assert approved == 2
    assert same
    assert listed


SANDBOXED = """<!doctype html><html><body><h1>Outer</h1>
<iframe name="inner" sandbox src="/inner" style="width: 400px; height: 200px">
</iframe></body></html>"""
INNER = """<!doctype html><html><body><h2>Inner</h2>
<input id="field" aria-label="Field"></body></html>"""


def test_a_frame_that_cannot_record_is_reported_and_covered(recorded) -> None:
    with recorded({"/": SANDBOXED, "/inner": INNER}) as (surface, _):
        page = surface._page
        page.frame_locator("iframe").locator("#field").wait_for()
        hand_over(surface)
        page.frame_locator("iframe").locator("#field").click()
        page.keyboard.type("UNSEEN42")
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)
        picture = surface.observe(VISUAL)

    assert [(gap.kind, gap.count) for gap in recording.gaps] == [
        (GapKind.FRAME_UNWATCHED, 1)
    ]
    assert "UNSEEN42" not in repr(structured)
    assert "Field" not in [node.name for node in structured.nodes]
    assert structured.status is ObservationStatus.PARTIAL
    assert picture.visual is not None
    assert "a frame outside the profile" in picture.visual.masked


def test_a_session_nobody_can_see_is_not_offered(recorded) -> None:
    with recorded({"/": DESK}) as (surface, _):
        where = surface.where()

    assert where == ""


# Regressions.

EDITOR = """<!doctype html><html><body><h1>Memo</h1><table><tr><td>Note</td>
<td id="cell"><span>x</span><div id="ed" contenteditable="true"
  style="min-width: 200px; min-height: 20px; border: 1px solid"></div></td>
</tr></table><input id="pin" aria-label="PIN"></body></html>"""


def test_a_protected_editable_region_is_never_read(pages) -> None:
    with pages({"/": EDITOR}) as (surface, _):
        page = surface._page
        page.click("#ed")
        page.keyboard.type("SECRET-4321")
        page.keyboard.press("Enter")
        page.keyboard.type("LINE-TWO")
        page.click("#pin")
        page.keyboard.type("9876")
        surface._protect(page.query_selector("#ed"))
        surface._protect(page.query_selector("#pin"))
        observation = surface.observe(STRUCTURED)
        cell = surface.act(
            Action(ActionKind.READ, DomLocator("td", DomAttribute.ID, "cell"))
        )
        pin = surface.act(Action(ActionKind.READ, AxLocator("textbox", "PIN")))
        editor = surface.act(
            Action(ActionKind.READ, DomLocator("div", DomAttribute.ID, "ed"))
        )

    written = repr(observation)
    assert observation.status is ObservationStatus.COMPLETE
    assert "SECRET-4321" not in written
    assert "LINE-TWO" not in written
    assert "9876" not in written
    region_node = next(node for node in observation.nodes if node.tag == "div")
    assert region_node.secret
    assert region_node.name == ""
    assert cell.outcome is Outcome.NOT_ACTIONABLE
    assert pin.outcome is Outcome.NOT_ACTIONABLE
    assert editor.outcome is Outcome.NOT_ACTIONABLE


POPUP = """<!doctype html><html><body><h1>Queue</h1></body></html>"""
LOADING = """<!doctype html><html><body><script>alert('loading')</script>
<p>popup</p></body></html>"""


def test_a_dialog_a_popup_opens_while_loading_is_held(pages) -> None:
    with pages({"/": POPUP, "/pop": LOADING}, allow_new_windows=True) as (
        surface,
        _,
    ):
        surface._page.evaluate("window.open('/pop'); 0")
        surface._page.wait_for_timeout(800)
        held = surface._pending.get("page-2")
        observation = surface.observe(STRUCTURED)

    assert held is not None
    assert held.info.message == "loading"
    assert observation.status is ObservationStatus.UNAVAILABLE


def test_an_unrecorded_session_reports_that_nothing_was_recorded(pages) -> None:
    with pages({"/": DESK}) as (surface, _):
        hand_over(surface)
        recording = surface.hand_back()
        where = surface.where()

    assert [gap.kind for gap in recording.gaps] == [GapKind.NOT_RECORDED]
    assert where == ""


# Review regressions: approvals and captures, Back, the control window, and a
# person's navigation and popups, through the real control.

COUNTER = """<!doctype html><html><body><h1>Member desk</h1>
<button id="approve" style="width: 200px; height: 80px; font-size: 24px"
  onclick="const shown = document.getElementById('count');
           shown.textContent = String(Number(shown.textContent) + 1)">
  Approve</button>
<p>Approved <output id="count">0</output> times</p>
</body></html>"""
COUNT = DomLocator("output", DomAttribute.ID, "count")

EXITS = """<!doctype html><html><body><h1>Member desk</h1>
<a id="away" href="/forbidden/ledger">Ledger</a>
<button id="pop" onclick="window.open('/next')">Statement</button>
</body></html>"""


def with_risky(kind: ActionKind) -> dict[str, dict[str, str]]:
    """Return the evaluation profile's action grants with ``kind`` declared risky."""
    actions = yaml.safe_load(SITE_PROFILE.read_text())["actions"]
    actions[kind.value] = {"any": "risky"}
    return actions


def centre(surface: BrowserSurface, selector: str) -> Point:
    """Return the viewport point at the middle of the element ``selector`` finds."""
    box = surface._page.locator(selector).bounding_box()
    assert box is not None
    return Point(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def send(control: Control, sent: Command) -> Receipt:
    """Send ``sent`` against the status the control reports now."""
    status = control.status()
    offer = status.offer
    return control.submit(
        Order(
            sent,
            status.run,
            status.revision,
            offer.intervention if offer is not None else "",
        )
    )


def acting(
    state: State, act: Callable[[Page], object]
) -> Callable[[Sitting, Status], bool]:
    """Do ``act`` in the session's page once the run reaches ``state``."""

    def move(sitting: Sitting, status: Status) -> bool:
        if status.state is not state:
            return False
        act(sitting.seat._page)
        return True

    return move


@contextlib.contextmanager
def seated(
    surface: BrowserSurface,
    profile: Profile,
    moves: list[Callable[[Sitting, Status], bool]],
) -> Iterator[tuple[Control, MemoryJournal]]:
    """Begin a run's control on the real seat, with a scripted operator at it.

    No loop runs. The test calls the control's boundaries itself, and the
    control finishes when the block ends, which flushes its journal.
    """
    journal = MemoryJournal()
    sitting = Sitting(surface, moves)
    control = Control(
        mode=Mode.DISCOVERY,
        clock=time.monotonic,
        seat=sitting,
        channels=[Watching()],
        worker=False,
    )
    sitting.control = control
    control.begin(
        profile=profile,
        budget=Budget(profile.budgets, time.monotonic),
        journal=journal,
        context="review the desk",
    )
    try:
        yield control, journal
    finally:
        control.finish("terminated")


@dataclasses.dataclass
class ClickingAPoint:
    """A decider that looks at a picture, clicks one point of it, and claims the count.

    A click that did not land is never claimed. The decider asks for a person
    instead, so a broken approval ends the run rather than retrying it.
    """

    point: Point

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        """Supply the task for this scripted control test."""
        del goal, notices
        return Task()

    def decide(self, transcript: Transcript) -> Decision:
        clicks = [
            turn
            for turn in transcript.history
            if turn.action is not None and turn.action.kind is ActionKind.CLICK
        ]
        if clicks and clicks[-1].outcome is Outcome.OK:
            check = ResultCheck(CheckKind.RESULT, COUNT, "1", output="approved")
            return Finish({"approved": "1"}, "the member is approved", (check,))
        if clicks:
            return AskHuman(Trigger.NO_PROGRESS, "the click did not land")
        pictures = [
            observation
            for observation in transcript.observations
            if observation.mode is ObservationMode.VISUAL
        ]
        if not pictures:
            return Observe(VISUAL, "see the desk")
        target = ScreenTarget(pictures[-1].observation_id, self.point)
        click = Action(ActionKind.CLICK, target, effect="approve_member")
        return Propose(click, "approve the member")


def test_approving_a_risky_click_on_a_screenshot_point_performs_it(
    recorded,
) -> None:
    """An approval is about the screen its proposal was made on.

    Nobody changed that screen, so the capture the point belongs to is kept
    through the approval, and the click is sent rather than refused as stale.
    """
    journal = MemoryJournal()
    with recorded({"/": COUNTER}, actions=with_risky(ActionKind.CLICK)) as (
        surface,
        profile,
    ):
        sitting = Sitting(
            surface,
            [
                command(State.AWAITING_APPROVAL, Command.APPROVE),
                command(State.PAUSED, Command.TERMINATE),
            ],
        )
        control = Control(
            mode=Mode.DISCOVERY,
            clock=time.monotonic,
            seat=sitting,
            channels=[Watching()],
            worker=False,
        )
        sitting.control = control
        result = discover(
            "approve the member once",
            profile,
            surface=surface,
            decider=ClickingAPoint(centre(surface, "#approve")),
            journal=journal,
            clock=time.monotonic,
            control=control,
        )
        count = surface._page.locator("#count").inner_text()

    clicks = [
        event
        for event in journal.events
        if isinstance(event, Acted) and event.kind is ActionKind.CLICK
    ]
    approvals = [
        event.verdict
        for event in journal.events
        if isinstance(event, Commanded) and event.command is Command.APPROVE
    ]
    handed = [event for event in journal.events if isinstance(event, HandedBack)]
    assert count == "1"
    assert approvals == [Verdict.ACCEPTED]
    assert [event.passed for event in handed] == [True]
    assert [event.outcome for event in clicks] == [Outcome.OK]
    assert result.ending is Ending.COMPLETED


def test_a_capture_taken_before_take_control_and_resume_is_refused_as_stale(
    recorded,
) -> None:
    """A person held the session, so a point on the screen before it means nothing."""
    with recorded({"/": COUNTER}) as (surface, profile):
        point = centre(surface, "#approve")
        with seated(
            surface, profile, [command(State.HUMAN_CONTROL, Command.RESUME)]
        ) as (control, _):
            picture = surface.observe(VISUAL)
            taken = send(control, Command.TAKE_CONTROL)
            halted = control.halted()
            handoff = control.hold()
            target = ScreenTarget(picture.observation_id, point)
            result = surface.act(Action(ActionKind.CLICK, target))
            count = surface._page.locator("#count").inner_text()

    assert picture.status is ObservationStatus.COMPLETE
    assert taken.verdict is Verdict.ACCEPTED
    assert halted
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert result.outcome is Outcome.STALE
    assert count == "0"


def test_a_capture_taken_before_a_stop_and_resume_is_refused_as_stale(
    recorded,
) -> None:
    """Only an answer keeps captures. Resume revokes even unused captures.

    Nobody took the session here, so nothing revoked the capture when the
    run stopped. The hand-back for the resume is what refuses it.
    """
    with recorded({"/": COUNTER}) as (surface, profile):
        point = centre(surface, "#approve")
        with seated(surface, profile, [command(State.PAUSED, Command.RESUME)]) as (
            control,
            _,
        ):
            picture = surface.observe(VISUAL)
            stopped = send(control, Command.STOP)
            halted = control.halted()
            handoff = control.hold()
            target = ScreenTarget(picture.observation_id, point)
            result = surface.act(Action(ActionKind.CLICK, target))
            count = surface._page.locator("#count").inner_text()

    assert picture.status is ObservationStatus.COMPLETE
    assert stopped.verdict is Verdict.ACCEPTED
    assert halted
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert result.outcome is Outcome.STALE
    assert count == "0"


def inside_of(surface: BrowserSurface, selector: str) -> tuple[int, int, int, int]:
    """Return the box just inside the element's border, in viewport pixels."""
    box = surface._page.locator(selector).bounding_box()
    assert box is not None
    return (
        int(box["x"]) + 3,
        int(box["y"]) + 3,
        int(box["x"] + box["width"]) - 3,
        int(box["y"] + box["height"]) - 3,
    )


def test_a_value_a_person_brings_back_with_back_is_withheld_and_masked(
    recorded,
) -> None:
    """The browser restores a typed value into a document the recorder never saw.

    The field has no autocomplete attribute, so Chromium keeps its value in
    the session history and puts it back when the person presses Back.
    """
    with recorded({"/": DESK, "/next": NEXT}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#note")
        page.keyboard.type("PLAIN9999")
        page.click("#next")
        page.wait_for_load_state("domcontentloaded")
        page.go_back()
        restored = page.locator("#note").input_value()
        inside = inside_of(surface, "#note")
        unmasked = region(page.screenshot(), inside)
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)
        picture = surface.observe(VISUAL)

    assert restored == "PLAIN9999"
    assert recording.unprotected == 0
    assert structured.status is ObservationStatus.COMPLETE
    assert "PLAIN9999" not in repr(structured)
    note = next(node for node in structured.nodes if node.name == "Note")
    assert note.secret
    assert note.value is None
    assert picture.image is not None
    masked = region(picture.image, inside)
    assert ImageChops.difference(unmasked, masked).getbbox() is not None
    assert masked.getcolors() == [(masked.width * masked.height, (16, 16, 16))]


def test_a_value_typed_in_an_earlier_takeover_and_brought_back_is_withheld(
    recorded,
) -> None:
    """The first hand-back marked the field. Back restores it in a new field.

    The second takeover typed nothing, so no edit of its own leads the
    surface to the document Back restored.
    """
    with recorded({"/": DESK, "/next": NEXT}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#note")
        page.keyboard.type("PLAIN9999")
        first = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        before = surface.observe(STRUCTURED)
        surface.transfer(Owner.OPERATOR, "iv-2")
        surface.transfer(Owner.HUMAN, "iv-2")
        page.click("#next")
        page.wait_for_load_state("domcontentloaded")
        page.go_back()
        restored = page.locator("#note").input_value()
        second = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-2")
        after = surface.observe(STRUCTURED)

    assert first.unprotected == 0
    assert "PLAIN9999" not in repr(before)
    assert restored == "PLAIN9999"
    assert second.unprotected == 0
    assert after.status is ObservationStatus.COMPLETE
    assert "PLAIN9999" not in repr(after)


def test_the_control_window_beside_a_recorded_session_is_never_recorded(
    recorded,
) -> None:
    """The window is its own context, so the recorder never hears the operator.

    The profile permits new windows, so a window opened in the session's own
    context would be taken under management and recorded rather than closed.
    A click in the session afterwards shows the recorder is still listening.
    """
    with (
        recorded({"/": DESK}, allow_new_windows=True) as (surface, profile),
        ControlWindow.open(surface.browser) as window,
    ):
        sitting = Sitting(surface, [])
        control = Control(
            mode=Mode.DISCOVERY,
            clock=time.monotonic,
            seat=sitting,
            channels=[window],
            worker=False,
        )
        sitting.control = control
        journal = MemoryJournal()
        control.begin(
            profile=profile,
            budget=Budget(profile.budgets, time.monotonic),
            journal=journal,
            context="review the desk",
        )
        panel = window.page
        assert panel is not None
        panel.click("[data-part=note]")
        panel.keyboard.type("NOTE-4242")
        panel.keyboard.press("Enter")
        panel.click("button[data-command=primary]")
        panel.click("button[data-command=terminate]")
        panel.click("button[data-command=terminate]")
        panel.wait_for_function(
            "document.querySelector('[data-field=state]').textContent === 'terminated'"
        )
        surface.idle(0.5)
        heard = surface.activity()
        surface._page.click("#approve")
        surface.idle(0.3)
        session = surface.activity()
        state = control.status().state
        control.finish("terminated")

    commanded = [
        (event.command, event.verdict, event.via)
        for event in journal.events
        if isinstance(event, Commanded)
    ]
    assert heard == ()
    assert [event.kind for event in session] == [ManualKind.CLICK]
    assert state is State.TERMINATED
    assert commanded == [
        (Command.STOP, Verdict.ACCEPTED, Via.PANEL),
        (Command.TERMINATE, Verdict.ACCEPTED, Via.PANEL),
    ]
    assert not any(isinstance(event, ManualAction) for event in journal.events)
    assert "NOTE-4242" not in repr(journal.events)


def test_a_person_on_a_refused_page_keeps_the_session_until_they_leave_it(
    recorded,
) -> None:
    """A refused navigation leaves the browser's error page, outside the profile.

    The hand-back check refuses that page, so the person keeps the session,
    and the run is not blocked by a screen it was never handed. Once the
    person goes back, the resume passes and the run continues.
    """
    refused: list[tuple[str, str]] = []

    def go_back(sitting: Sitting, status: Status) -> bool:
        if status.state is not State.HUMAN_CONTROL or not status.notice:
            return False
        page = sitting.seat._page
        refused.append((page.url, status.notice))
        page.go_back()
        return True

    journal = MemoryJournal()
    with recorded({"/": EXITS}) as (surface, profile):
        sitting = Sitting(
            surface,
            [
                command(State.RUNNING, Command.TAKE_CONTROL),
                acting(State.HUMAN_CONTROL, lambda page: page.click("#away")),
                command(State.HUMAN_CONTROL, Command.RESUME),
                go_back,
                command(State.HUMAN_CONTROL, Command.RESUME),
                command(State.PAUSED, Command.TERMINATE),
            ],
        )
        control = Control(
            mode=Mode.DISCOVERY,
            clock=time.monotonic,
            seat=sitting,
            channels=[Watching()],
            worker=False,
        )
        sitting.control = control
        result = discover(
            "review the desk",
            profile,
            surface=surface,
            decider=Looking(),
            journal=journal,
            clock=time.monotonic,
            control=control,
        )
        final = surface._page.url

    handed = [event for event in journal.events if isinstance(event, HandedBack)]
    manual = [event for event in journal.events if isinstance(event, ManualAction)]
    blocked = [event for event in manual if event.kind is ManualKind.NAVIGATION_REFUSED]
    accepted = journal.events.index(handed[-1])
    assert len(refused) == 1
    assert policy.route_for(profile, refused[0][0]) is None
    assert "outside the profile" in refused[0][1]
    assert [(event.passed, event.failure) for event in handed] == [
        (False, CheckFailure.OFF_ROUTE),
        (True, None),
    ]
    assert len(blocked) == 1
    assert blocked[0].owned
    assert blocked[0].control is Widget.PAGE
    assert not any(isinstance(event, Observed) for event in journal.events[:accepted])
    assert policy.route_for(profile, final) is not None
    assert result.ending is Ending.TERMINATED


def test_a_popup_a_person_opens_where_windows_are_refused_is_closed_and_recorded(
    recorded,
) -> None:
    """The profile allows no new windows, so the popup is closed as it opens.

    It never gets a page id of its own, so the recording says only that a
    page was closed by the profile.
    """

    def open_popup(page: Page) -> None:
        page.click("#pop")
        page.wait_for_timeout(300)

    with recorded({"/": EXITS, "/next": NEXT}) as (surface, profile):
        context = surface._page.context
        with seated(
            surface,
            profile,
            [
                acting(State.HUMAN_CONTROL, open_popup),
                command(State.HUMAN_CONTROL, Command.RESUME),
            ],
        ) as (control, journal):
            send(control, Command.TAKE_CONTROL)
            assert control.halted()
            handoff = control.hold()
            pages = list(context.pages)
            managed = list(surface._pages)
            session = surface._page
            segment = control.segments[-1]

    manual = [
        (event.kind, event.control, event.detail)
        for event in journal.events
        if isinstance(event, ManualAction)
    ]
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert manual == [
        (ManualKind.CLICK, Widget.BUTTON, Detail.NONE),
        (ManualKind.PAGE_CLOSED, Widget.PAGE, Detail.CLOSED_BY_POLICY),
    ]
    assert [step.event.kind for step in segment.steps] == [
        ManualKind.CLICK,
        ManualKind.PAGE_CLOSED,
    ]
    assert pages == [session]
    assert managed == ["page-1"]


# Review regressions: what the recorder hears around a hand-over, and from whom.


def screen_point(surface: BrowserSurface, selector: str) -> ScreenTarget:
    """Capture the screen and return the middle of ``selector`` on it."""
    picture = surface.observe(VISUAL)
    return ScreenTarget(picture.observation_id, centre(surface, selector))


def test_a_person_who_keeps_typing_where_the_automation_typed_is_recorded(
    recorded,
) -> None:
    """A recorder reports one keyed edit per field per focus.

    The automation typed into Note with keys, so the recorder already holds
    a keyed edit for that focus, and that edit is the adapter's own, which a
    hand-back leaves unmarked. The person keeps typing without clicking.
    Only the hand-over telling each recorder to forget that edit makes the
    person's typing a report of its own, and so a field the surface masks.
    """
    with recorded({"/": DESK}) as (surface, _):
        page = surface._page
        clicked = surface.act(Action(ActionKind.CLICK, screen_point(surface, "#note")))
        focus = ScreenTarget(surface.observe(VISUAL).observation_id)
        typed = surface.act(Action(ActionKind.TYPE, focus, "auto "))
        hand_over(surface)
        surface.idle(0.5)
        page.keyboard.type("PERSON42")
        value = page.locator("#note").input_value()
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)

    assert clicked.outcome is Outcome.OK
    assert typed.outcome is Outcome.OK
    assert value == "auto PERSON42"
    assert [(event.kind, event.owned) for event in recording.events] == [
        (ManualKind.EDIT, True)
    ]
    assert recording.events[0].target == AxLocator("textbox", "Note")
    assert recording.unprotected == 0
    assert structured.status is ObservationStatus.COMPLETE
    note = next(node for node in structured.nodes if node.name == "Note")
    assert note.secret
    assert note.value is None
    assert "PERSON42" not in repr(structured)


REOPENED = """<!doctype html><html><body><h1>Statement</h1>
<button id="rewrite" onclick="setTimeout(() => {
  document.open();
  document.write('<h1>Rewritten</h1><input id=late aria-label=Late>');
  document.close();
}, 0)">Rewrite</button>
</body></html>"""


def test_a_page_that_reopens_its_document_is_reported_unwatched(recorded) -> None:
    """``document.open`` erases every listener the recorder added to the window.

    The recorder's own function survives on the window. Without a probe, it
    would still answer for its document. The hand-back would then claim a
    complete recording of a page that heard none of the person's input.
    """
    with recorded({"/": REOPENED}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click("#rewrite")
        page.locator("#late").wait_for()
        page.click("#late")
        page.keyboard.type("UNHEARD77")
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)
        picture = surface.observe(VISUAL)

    assert [event.kind for event in recording.events] == [ManualKind.CLICK]
    assert [(gap.kind, gap.count) for gap in recording.gaps] == [
        (GapKind.FRAME_UNWATCHED, 1)
    ]
    assert "UNHEARD77" not in repr(structured)
    assert "Late" not in [node.name for node in structured.nodes]
    assert picture.image is None


SHADOWED = """<!doctype html><html><body><h1>Member desk</h1>
<div id="host" style="display: inline-block; width: 320px; height: 48px"></div>
<script>
  const root = document.getElementById('host').attachShadow({ mode: 'closed' });
  root.innerHTML = '<input id="inner" aria-label="Inner" ' +
    'style="width: 300px; height: 40px; font-size: 28px">';
</script>
</body></html>"""


def test_typing_inside_a_closed_shadow_root_marks_and_masks_its_host(
    recorded,
) -> None:
    """The input event reaches the recorder retargeted to the shadow host.

    The host is a plain element instead of a text field. The recorder marks it
    because it is not a form control and masks its box in every capture until
    the document is gone.
    """
    with recorded({"/": SHADOWED}) as (surface, _):
        page = surface._page
        inside = inside_of(surface, "#host")
        hand_over(surface)
        page.locator("#host").click()
        page.keyboard.type("SHADOW55")
        unmasked = region(page.screenshot(), inside)
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)
        picture = surface.observe(VISUAL)

    edits = [event for event in recording.events if event.kind is ManualKind.EDIT]
    assert len(edits) == 1
    assert edits[0].owned
    assert recording.unprotected == 0
    assert "SHADOW55" not in repr(recording)
    assert "SHADOW55" not in repr(structured)
    assert picture.image is not None
    masked = region(picture.image, inside)
    assert ImageChops.difference(unmasked, masked).getbbox() is not None
    assert masked.getcolors() == [(masked.width * masked.height, (16, 16, 16))]


APPEARING = """<!doctype html><html><body><h1>Member desk</h1>
<button id="approve" onclick="window.approved = (window.approved || 0) + 1">
  Approve</button>
<script>setTimeout(() => {
  const ready = document.createElement('button');
  ready.textContent = 'Ready';
  document.body.appendChild(ready);
}, 800)</script>
</body></html>"""


def test_a_persons_click_during_a_wait_is_still_a_persons(
    recorded, monkeypatch
) -> None:
    """A wait sends no input, so it is not fenced as the adapter's own.

    The person clicks from inside the wait's own polling, through the
    driver, so their click lands while the action is still running. Fenced,
    it would be taken for the adapter's input and never reach the run.
    """
    with recorded({"/": APPEARING}) as (surface, _):
        page = surface._page
        waited = page.wait_for_timeout
        clicks: list[bool] = []

        def person_clicks(timeout: float) -> None:
            if not clicks:
                clicks.append(True)
                page.click("#approve")
            waited(timeout)

        monkeypatch.setattr(page, "wait_for_timeout", person_clicks)
        result = surface.act(Action(ActionKind.WAIT_FOR, AxLocator("button", "Ready")))
        monkeypatch.undo()
        surface.idle(0.5)
        heard = surface.activity()
        approved = page.evaluate("window.approved")

    assert result.outcome is Outcome.OK
    assert clicks == [True]
    assert approved == 1
    assert [(event.kind, event.target) for event in heard] == [
        (ManualKind.CLICK, AxLocator("button", "Approve"))
    ]
    assert not heard[0].owned


def test_an_edit_the_page_makes_by_itself_is_unprompted_and_still_protected(
    recorded,
) -> None:
    """``execCommand`` fires a trusted input event with no person behind it.

    Nobody touched a key or the pointer, so the edit is reported as
    unprompted: it is not a person's input, and it does not count as a
    change to the session. The field is marked all the same, because
    whatever put the value there, the value can still be private.
    """
    with recorded({"/": DESK}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.evaluate(
            "() => { document.getElementById('note').focus();"
            " document.execCommand('insertText', false, 'SCRIPTED7'); }"
        )
        value = page.locator("#note").input_value()
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)

    assert value == "SCRIPTED7"
    assert [(event.kind, event.detail) for event in recording.events] == [
        (ManualKind.EDIT, Detail.UNPROMPTED)
    ]
    assert not recording.events[0].input
    assert not recording.events[0].changes
    assert recording.unprotected == 0
    assert structured.status is ObservationStatus.COMPLETE
    note = next(node for node in structured.nodes if node.name == "Note")
    assert note.secret
    assert note.value is None
    assert "SCRIPTED7" not in repr(structured)


def test_typing_right_after_a_hand_over_is_the_persons(recorded) -> None:
    """Input after the session went to a person cannot be the automation's.

    The person types at once, into the field the automation just typed
    into. No claim of the automation's covers it, and none could after the
    hand-over, so it is theirs and the field is protected.
    """
    with recorded({"/": DESK}) as (surface, _):
        page = surface._page
        surface.act(Action(ActionKind.CLICK, screen_point(surface, "#note")))
        focus = ScreenTarget(surface.observe(VISUAL).observation_id)
        surface.act(Action(ActionKind.TYPE, focus, "auto "))
        hand_over(surface)
        page.keyboard.type("PERSON42")
        recording = surface.hand_back()
        surface.transfer(Owner.AUTOMATION, "iv-1")
        structured = surface.observe(STRUCTURED)

    assert [event.kind for event in recording.events] == [ManualKind.EDIT]
    note = next(node for node in structured.nodes if node.name == "Note")
    assert note.secret
    assert "PERSON42" not in repr(structured)


# Regressions from the second review. Each fails against commit 89d654a.
#
# Attribution: an event is the automation's only when the recorder finds it
# covered by one of the adapter's claims, by kind and by element, never
# because it arrived soon after the adapter's own input.

FIELDS = """<!doctype html><html><body><h1>Member desk</h1>
<input id="note" aria-label="Note">
<input id="memo" aria-label="Memo">
<button id="approve" onclick="window.approved = (window.approved || 0) + 1">
  Approve</button>
</body></html>"""
NOTE = AxLocator("textbox", "Note")


def named(observation: Observation, name: str) -> AxNode:
    return next(node for node in observation.nodes if node.name == name)


def test_typing_at_once_into_the_field_the_automation_filled_is_a_persons(
    recorded,
) -> None:
    """The person types into the field the fill left focused, a moment later.

    Nobody handed the session over, and no time window counts the typing as
    the automation's. The observation is taken before anything asked the
    surface for activity, and still does not show what the person typed.
    """
    with recorded({"/": FIELDS}) as (surface, _):
        page = surface._page
        filled = surface.act(Action(ActionKind.TYPE, NOTE, "auto "))
        page.keyboard.type("PRIV8812")
        value = page.locator("#note").input_value()
        seen = surface.observe(STRUCTURED)
        heard = surface.activity()

    assert filled.outcome is Outcome.OK
    assert value == "auto PRIV8812"
    assert [(event.kind, event.owned, event.detail) for event in heard] == [
        (ManualKind.EDIT, False, Detail.NONE)
    ]
    assert heard[0].changes
    note = named(seen, "Note")
    assert note.secret
    assert "PRIV8812" not in repr(seen)


def test_typing_at_once_into_another_field_is_a_persons(recorded) -> None:
    with recorded({"/": FIELDS}) as (surface, _):
        page = surface._page
        surface.act(Action(ActionKind.TYPE, NOTE, "auto"))
        page.click("#memo")
        page.keyboard.type("MEMO3141")
        seen = surface.observe(STRUCTURED)
        heard = surface.activity()

    assert [(event.kind, event.target) for event in heard] == [
        (ManualKind.CLICK, AxLocator("textbox", "Memo")),
        (ManualKind.EDIT, AxLocator("textbox", "Memo")),
    ]
    assert named(seen, "Memo").secret
    assert named(seen, "Note").value == "auto"
    assert "MEMO3141" not in repr(seen)


def test_a_click_right_after_the_automations_click_is_a_persons(recorded) -> None:
    with recorded({"/": FIELDS}) as (surface, _):
        page = surface._page
        clicked = surface.act(Action(ActionKind.CLICK, AxLocator("button", "Approve")))
        page.click("#approve")
        approved = page.evaluate("window.approved")
        heard = surface.activity()

    assert clicked.outcome is Outcome.OK
    assert approved == 2
    assert [(event.kind, event.target, event.owned) for event in heard] == [
        (ManualKind.CLICK, AxLocator("button", "Approve"), False)
    ]


def test_input_while_an_operation_waits_stops_it_before_it_sends(
    recorded, monkeypatch
) -> None:
    """The fill waits for its field while the person types elsewhere.

    The surface asks the control at the last moment, and the control takes
    what the recorder heard first, so the fill is never sent and the run
    stops for the person. Their field is marked in the page at once.
    """
    original = ElementHandle.wait_for_element_state
    typed: list[bool] = []

    def person_types(self: ElementHandle, *args: object, **options: object) -> None:
        if not typed:
            typed.append(True)
            frame = self.owner_frame()
            assert frame is not None
            frame.page.click("#memo")
            frame.page.keyboard.type("MEMO2718")
        original(self, *args, **options)  # ty: ignore[invalid-argument-type]

    with (
        bounded(60),
        recorded({"/": FIELDS}) as (surface, profile),
        seated(surface, profile, []) as (control, _),
    ):
        monkeypatch.setattr(ElementHandle, "wait_for_element_state", person_types)
        result = control.guard(surface).act(Action(ActionKind.TYPE, NOTE, "auto"))
        monkeypatch.undo()
        state = control.status().state
        page = surface._page
        filled = page.locator("#note").input_value()
        marked = page.get_attribute("#memo", browser.MARK_ATTRIBUTE)
        token = surface._token

    assert typed == [True]
    assert result.outcome is Outcome.BLOCKED
    assert result.detail == STOPPED
    assert filled == ""
    assert state is State.STOPPING
    assert marked == token


def test_input_during_the_fill_on_its_own_field_is_uncertain_and_protected(
    recorded, monkeypatch
) -> None:
    """A key a person presses while the fill runs lands on the claimed field.

    The claim covers edits to that field, so nothing about the event itself
    says whose it is. The field does not hold the value the fill set, so the
    recorder reports an uncertain edit and marks the field.
    """
    fill = ElementHandle.fill

    def with_a_key(self: ElementHandle, value: str, **options: object) -> None:
        fill(self, value, **options)  # ty: ignore[invalid-argument-type]
        frame = self.owner_frame()
        assert frame is not None
        frame.page.keyboard.type("Z9")

    with recorded({"/": FIELDS}) as (surface, _):
        monkeypatch.setattr(ElementHandle, "fill", with_a_key)
        filled = surface.act(Action(ActionKind.TYPE, NOTE, "auto "))
        monkeypatch.undo()
        value = surface._page.locator("#note").input_value()
        seen = surface.observe(STRUCTURED)
        heard = surface.activity()

    assert filled.outcome is Outcome.OK
    assert value == "auto Z9"
    # Two keys where the fill sends at most one, and a value it did not set.
    assert sorted((event.kind, event.detail) for event in heard) == [
        (ManualKind.EDIT, Detail.UNCERTAIN),
        (ManualKind.KEY, Detail.UNCERTAIN),
    ]
    assert all(event.changes for event in heard)
    assert named(seen, "Note").secret
    assert "Z9" not in repr(seen)


# What counts as a change: no role or tag makes a click harmless, and every
# key is reported as a category.

HANDLERS = """<!doctype html><html><body><h1>Payment</h1>
<p>Amount: <span id="amount">10</span></p>
<div id="bump" onclick="document.getElementById('amount').textContent = '900'">
  Adjust</div>
<table><tr id="row"><td id="cell">Payee 10001</td></tr></table>
<img id="logo" alt="" width="40" height="40"
     src="data:image/gif;base64,R0lGODlhAQABAAAAACw=">
<h2 id="title">Summary</h2>
<span id="deep">Detail</span>
<input id="field" aria-label="Field">
<div id="pane" style="height: 120px; overflow: auto">
  <div style="height: 2000px">Long</div></div>
<div style="height: 3000px"></div>
<script>
const amount = (text) => { document.getElementById('amount').textContent = text; };
document.addEventListener('click', (event) => {
  if (event.target.id === 'deep') { amount('901'); }
  if (event.target.closest('#row')) { amount('902'); }
  if (event.target.id === 'logo') { amount('903'); }
  if (event.target.id === 'title') { amount('904'); }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'F2') { amount('905'); }
  if (event.key === 'ArrowDown' && event.target.id === 'field') {
    event.preventDefault();
    amount('906');
  }
  if (event.key === 'j') { amount('907'); }
  // Neither of these cancels the key, so the browser still scrolls.
  if (event.key === 'ArrowDown' && event.target === document.body) {
    amount('909');
  }
});
document.getElementById('pane').addEventListener('wheel', (event) => {
  event.preventDefault();
  amount('908');
}, { passive: false });
window.addEventListener('wheel', (event) => {
  if (!event.target.closest('#pane')) { amount('910'); }
}, { passive: true });
</script>
</body></html>"""


@pytest.mark.parametrize(
    ("selector", "amount"),
    [
        ("#bump", "900"),
        ("#deep", "901"),
        ("#cell", "902"),
        ("#logo", "903"),
        ("#title", "904"),
    ],
    ids=["generic-element", "delegated-handler", "row", "image", "heading"],
)
def test_a_click_on_any_element_is_a_change(
    recorded, selector: str, amount: str
) -> None:
    with recorded({"/": HANDLERS}) as (surface, _):
        page = surface._page
        hand_over(surface)
        page.click(selector)
        shown = page.locator("#amount").inner_text()
        recording = surface.hand_back()

    assert shown == amount
    assert [event.kind for event in recording.events] == [ManualKind.CLICK]
    assert recording.events[0].changes


def test_clicking_a_generic_element_during_an_approval_voids_it(recorded) -> None:
    """The approval showed 10, but the click changed the amount to 900.

    Before the fix, a click on a ``div`` counted as looking. The approval came
    back unchanged, so the run would have sent the approved click on a screen
    that no longer showed what the person approved.
    """
    request = InterventionRequest(
        trigger=Trigger.RISKY_ACTION,
        goal="pay member 10001",
        profile_id="test",
        step=1,
        route="/**",
        reason="click is declared risky",
        timeout_s=30.0,
        action=Action(ActionKind.CLICK, AxLocator("button", "Send")),
        ask=Ask.APPROVAL,
    )
    with (
        bounded(60),
        recorded({"/": HANDLERS}) as (surface, profile),
        seated(
            surface,
            profile,
            [
                acting(State.AWAITING_APPROVAL, lambda page: page.click("#bump")),
                command(State.AWAITING_APPROVAL, Command.APPROVE),
            ],
        ) as (control, _),
    ):
        handoff = control.intervene(request)
        shown = surface._page.locator("#amount").inner_text()

    assert shown == "900"
    assert handoff.outcome is HandoffOutcome.APPROVED
    assert handoff.changed


@pytest.mark.parametrize(
    ("keys", "focus", "detail", "amount"),
    [
        (["F2"], "", Detail.COMMAND, "905"),
        (["F2"], "#field", Detail.COMMAND, "905"),
        (["ArrowDown"], "#field", Detail.MOVE, "906"),
        (["j", "j", "j"], "", Detail.COMMAND, "907"),
        (["ArrowDown"], "", Detail.MOVE, "909"),
    ],
    ids=[
        "function-key",
        "function-key-in-field",
        "arrow-the-page-cancels",
        "characters-outside-a-field",
        "arrow-the-page-acts-on-without-cancelling",
    ],
)
def test_every_key_is_reported_as_a_category_and_counts_as_a_change(
    recorded, keys: list[str], focus: str, detail: Detail, amount: str
) -> None:
    """A key is a category, never the key, and every key is a change.

    The category is for the journal. Whether the page cancelled a movement
    key says nothing about whether it acted on it, so a movement key the
    browser still scrolled for is a change too. Three characters typed
    outside a field come back as one command, so the recording says nothing
    about how many were typed.
    """
    with recorded({"/": HANDLERS}) as (surface, _):
        page = surface._page
        hand_over(surface)
        if focus:
            page.focus(focus)
        for key in keys:
            page.keyboard.press(key)
        surface.idle(0.3)
        shown = page.locator("#amount").inner_text()
        recording = surface.hand_back()

    assert shown == amount
    assert [(event.kind, event.detail) for event in recording.events] == [
        (ManualKind.KEY, detail)
    ]
    assert recording.events[0].changes
    assert "F2" not in repr(recording)
    assert "'j'" not in repr(recording)


def test_every_wheel_is_a_change_whether_the_page_cancels_it_or_not(
    recorded,
) -> None:
    with recorded({"/": HANDLERS}) as (surface, _):
        page = surface._page
        hand_over(surface)
        box = page.locator("#pane").bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + 10, box["y"] + 10)
        page.mouse.wheel(0, 200)
        surface.idle(0.6)
        cancelled = page.locator("#amount").inner_text()
        page.mouse.move(5, 500)
        page.mouse.wheel(0, 200)
        surface.idle(0.3)
        scrolled = page.evaluate("window.scrollY")
        shown = page.locator("#amount").inner_text()
        recording = surface.hand_back()

    assert cancelled == "908"
    assert shown == "910"
    assert scrolled > 0
    assert [(event.kind, event.detail) for event in recording.events] == [
        (ManualKind.SCROLL, Detail.DOWN),
        (ManualKind.SCROLL, Detail.DOWN),
    ]
    assert all(event.changes for event in recording.events)


def press_arrow_down(page: Page) -> None:
    page.focus("body")
    page.keyboard.press("ArrowDown")


def wheel_down(page: Page) -> None:
    page.mouse.move(5, 500)
    page.mouse.wheel(0, 200)
    page.wait_for_timeout(100)


@pytest.mark.parametrize(
    ("move", "amount"),
    [(press_arrow_down, "909"), (wheel_down, "910")],
    ids=["movement-key", "wheel"],
)
def test_movement_the_page_does_not_cancel_voids_a_pending_approval(
    recorded, move: Callable[[Page], None], amount: str
) -> None:
    """The page changes the amount on a key or a wheel it lets through.

    Before the fix, an uncancelled movement key or wheel counted as looking,
    and the approval came back unchanged for a screen showing a new amount.
    """
    request = InterventionRequest(
        trigger=Trigger.RISKY_ACTION,
        goal="pay member 10001",
        profile_id="test",
        step=1,
        route="/**",
        reason="click is declared risky",
        timeout_s=30.0,
        action=Action(ActionKind.CLICK, AxLocator("button", "Send")),
        ask=Ask.APPROVAL,
    )
    with (
        bounded(60),
        recorded({"/": HANDLERS}) as (surface, profile),
        seated(
            surface,
            profile,
            [
                acting(State.AWAITING_APPROVAL, move),
                command(State.AWAITING_APPROVAL, Command.APPROVE),
            ],
        ) as (control, _),
    ):
        handoff = control.intervene(request)
        shown = surface._page.locator("#amount").inner_text()

    assert shown == amount
    assert handoff.outcome is HandoffOutcome.APPROVED
    assert handoff.changed


PAYING = """<!doctype html><html><body><h1>Payment</h1>
<p>Amount: <output id="amount">10</output></p>
<button id="send" onclick="window.sent = (window.sent || 0) + 1">Send</button>
<div style="height: 3000px"></div>
<script>
document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown') {
    document.getElementById('amount').textContent = '900';
  }
});
</script>
</body></html>"""


@dataclasses.dataclass
class ProposingOnce:
    """A decider that looks, proposes one risky click, then looks and stops."""

    seen: list[Transcript] = dataclasses.field(default_factory=list)

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        """Supply the task for this scripted control test."""
        del goal, notices
        return Task()

    def decide(self, transcript: Transcript) -> Decision:
        self.seen.append(transcript)
        if len(self.seen) == 1:
            return Observe(STRUCTURED, "see the payment")
        if len(self.seen) == 2:
            send = Action(ActionKind.CLICK, AxLocator("button", "Send"))
            return Propose(send, "send the payment")
        if not transcript.observations:
            return Observe(STRUCTURED, "look again")
        return AskHuman(Trigger.NO_PROGRESS, "stop here")


def test_a_voided_approval_never_sends_and_the_run_looks_again(recorded) -> None:
    """Discovery drops the approved click and observes before deciding again.

    The person presses a movement key while the approval waits. The page
    changes the amount without cancelling the key. The approval they then
    give is about a screen that no longer exists, so it sends nothing.
    """
    journal = MemoryJournal()
    decider = ProposingOnce()
    with (
        bounded(90),
        recorded(
            {"/": PAYING},
            actions=with_risky(ActionKind.CLICK),
            escalation={"handoff_timeout_s": 30, "on_timeout": "abort"},
        ) as (surface, profile),
    ):
        sitting = Sitting(
            surface,
            [
                acting(State.AWAITING_APPROVAL, press_arrow_down),
                command(State.AWAITING_APPROVAL, Command.APPROVE),
                command(State.PAUSED, Command.TERMINATE),
            ],
        )
        control = Control(
            mode=Mode.DISCOVERY,
            clock=time.monotonic,
            seat=sitting,
            channels=[Watching()],
            worker=False,
        )
        sitting.control = control
        result = discover(
            "send the payment once",
            profile,
            surface=surface,
            decider=decider,
            journal=journal,
            clock=time.monotonic,
            control=control,
        )
        sent = surface._page.evaluate("window.sent || 0")
        shown = surface._page.locator("#amount").inner_text()

    assert shown == "900"
    assert sent == 0
    assert result.ending is Ending.TERMINATED
    assert not [
        event
        for event in journal.events
        if isinstance(event, Acted) and event.kind is ActionKind.CLICK
    ]
    after = decider.seen[2]
    assert after.observations == ()
    assert "the session changed while the approval was pending; look again" in (
        after.notices
    )
    assert [turn.observations != () for turn in decider.seen[3:4]] == [True]


# Protection: resuming again is another check, never a confirmation.

SWAPPED = """<!doctype html><html><body><h1>Member desk</h1>
<input id="member" aria-label="Member">
<button id="swap" onclick="
  const old = document.getElementById('member');
  const copy = document.createElement('input');
  copy.setAttribute('aria-label', 'Copy');
  copy.value = old.value;
  old.replaceWith(copy);">Swap</button>
</body></html>"""


def resuming(note: str | None = None) -> Callable[[Sitting, Status], bool]:
    """Send Resume, with ``note``, once a person holds the session."""

    def move(sitting: Sitting, status: Status) -> bool:
        if status.state is not State.HUMAN_CONTROL or sitting.control is None:
            return False
        offer = status.offer
        sitting.control.submit(
            Order(
                Command.RESUME,
                status.run,
                status.revision,
                offer.intervention if offer is not None else "",
                note=note,
            )
        )
        return True

    return move


def test_resuming_twice_does_not_let_go_of_a_value_the_page_moved(recorded) -> None:
    """The page copies the person's value into a new field and drops the old one.

    The first Resume fails because the value may have moved. Before the fix,
    a second Resume counted as the person saying the screen is safe, and the
    field the value moved into was read. Now it fails the same way, a note
    saying the screen is safe included, until a reload takes the document.
    """
    with (
        bounded(90),
        recorded(
            {"/": SWAPPED}, escalation={"handoff_timeout_s": 60, "on_timeout": "abort"}
        ) as (surface, profile),
    ):

        def type_and_swap(page: Page) -> None:
            page.click("#member")
            page.keyboard.type("SWAP7788")
            page.click("#swap")

        def reload(page: Page) -> None:
            page.reload()
            page.wait_for_load_state("domcontentloaded")

        with seated(
            surface,
            profile,
            [
                command(State.RUNNING, Command.TAKE_CONTROL),
                acting(State.HUMAN_CONTROL, type_and_swap),
                resuming(),
                resuming("the screen is safe to capture"),
                acting(State.HUMAN_CONTROL, reload),
                resuming(),
            ],
        ) as (control, journal):
            assert control.halted()
            handoff = control.hold()
            seen = control.guard(surface).observe(STRUCTURED)
            events = journal.events

    backs = [event for event in events if isinstance(event, HandedBack)]
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert [(back.passed, back.failure) for back in backs] == [
        (False, CheckFailure.PROTECTION_LOST),
        (False, CheckFailure.PROTECTION_LOST),
        (True, None),
    ]
    assert seen.status is ObservationStatus.COMPLETE
    assert "SWAP7788" not in repr(seen)


# A renamed control and a person's answer to a dialog keep their identity.

SENDING = """<!doctype html><html><body><h1>Payment</h1>
<form id="pay" aria-label="Pay" onsubmit="event.preventDefault();
  window.sent = (window.sent || 0) + 1">
  <input id="amount" aria-label="Amount" value="10">
  <button id="send" type="submit">Send</button>
</form>
<button id="back" onclick="window.back = 1">Back</button>
</body></html>"""


def restricted(
    surface: BrowserSurface, profile: Profile, name: str
) -> tuple[AxNode, Restriction]:
    """Observe the control ``name`` and hold a learned deny on clicking it."""
    seen = surface.observe(STRUCTURED)
    node = named(seen, name)
    route = policy.route_for(profile, surface.location()) or ""
    click = Action(ActionKind.CLICK, AxLocator(node.role, node.name))
    operation = Operation.of(click, route, node=node)
    return node, Restriction(operation, None, Limit.DENY, Source.OPERATOR, 1)


def judge(event: ManualEvent, profile: Profile, deny: Restriction) -> Requirement:
    return classify(event, profile, ask=Ask.PERSON, trigger=None, restrictions=(deny,))


def test_a_person_clicking_a_denied_button_after_it_was_renamed_is_forbidden(
    recorded,
) -> None:
    """The page renames the denied button, but keeps the same element.

    A name-only match let the click through as needing an approval. The
    recorder reports which observed control the click reached, so the deny
    still covers it.
    """
    with recorded({"/": SENDING}) as (surface, profile):
        page = surface._page
        node, deny = restricted(surface, profile, "Send")
        page.evaluate("document.getElementById('send').textContent = 'Continue'")
        hand_over(surface)
        page.click("#send")
        recording = surface.hand_back()

    [click] = [event for event in recording.events if event.kind is ManualKind.CLICK]
    assert click.target == AxLocator("button", "Continue")
    assert click.operation is not None
    assert click.operation.control == node.control
    assert judge(click, profile, deny) is Requirement.FORBIDDEN


def test_an_enter_that_submits_a_denied_form_is_forbidden(recorded) -> None:
    """Enter in the amount field is the same native submission as the click."""
    with recorded({"/": SENDING}) as (surface, profile):
        page = surface._page
        _, deny = restricted(surface, profile, "Send")
        hand_over(surface)
        page.focus("#amount")
        page.keyboard.press("Enter")
        sent = page.evaluate("window.sent")
        recording = surface.hand_back()

    [key] = [event for event in recording.events if event.kind is ManualKind.KEY]
    assert sent == 1
    assert key.detail is Detail.CONFIRM
    assert key.operation is not None
    assert key.operation.submission == deny.operation.submission
    assert judge(key, profile, deny) is Requirement.FORBIDDEN


def test_a_distinct_observed_button_is_not_forbidden_by_the_deny(recorded) -> None:
    with recorded({"/": SENDING}) as (surface, profile):
        page = surface._page
        _, deny = restricted(surface, profile, "Send")
        hand_over(surface)
        page.click("#back")
        recording = surface.hand_back()

    [click] = [event for event in recording.events if event.kind is ManualKind.CLICK]
    assert click.operation is not None
    assert click.operation.control != deny.operation.control
    assert judge(click, profile, deny) is Requirement.APPROVE_EACH_RUN


def test_the_journal_carries_no_operation_identity(recorded) -> None:
    """Keep identity in the live event and closed categories in the journal."""
    with recorded({"/": SENDING}) as (surface, profile):
        page = surface._page
        node, _ = restricted(surface, profile, "Send")
        hand_over(surface)
        page.click("#send")
        recording = surface.hand_back()

    [click] = [event for event in recording.events if event.kind is ManualKind.CLICK]
    entry = ManualAction(
        step=1,
        intervention="iv-1",
        sequence=click.sequence,
        kind=click.kind,
        control=click.control,
        route=click.route,
        frame=click.position,
        secret=click.secret,
        owned=click.owned,
        detail=click.detail,
    )
    assert node.control not in repr(entry)
    assert "Send" not in repr(entry)


CONFIRMING = """<!doctype html><html><body><h1>Queue</h1>
<p id="result">none</p>
</body></html>"""


def undeclared(profile: Profile) -> Profile:
    """Return ``profile`` without either dialog action type granted."""
    actions = {
        kind: risk
        for kind, risk in profile.actions.items()
        if kind not in {ActionKind.ACCEPT_DIALOG, ActionKind.DISMISS_DIALOG}
    }
    return dataclasses.replace(profile, actions=actions)


@pytest.mark.parametrize(
    ("script", "answer", "detail", "result"),
    [
        ("confirm('Post this entry?')", "accept", Detail.ACCEPTED, "true"),
        ("confirm('Post this entry?')", "dismiss", Detail.DISMISSED, "false"),
        ("prompt('Code?', 'DEFAULT42')", "dismiss", Detail.DISMISSED, "null"),
    ],
    ids=["accepted", "dismissed", "prompt-dismissed"],
)
def test_a_person_answering_a_dialog_is_recorded_and_judged_by_its_answer(
    recorded, script: str, answer: str, detail: Detail, result: str
) -> None:
    """The person answers through the dialog itself, not the adapter's action.

    The page's own call returns the answer, so the recorder can say which it
    was. With neither dialog action granted, no answer is permitted work.
    """
    with recorded({"/": CONFIRMING}) as (surface, profile):
        page = surface._page
        hand_over(surface)
        page.evaluate(
            "(script) => setTimeout(() => {"
            " document.getElementById('result').textContent ="
            " String(eval(script)); })",
            script,
        )
        surface.idle(0.3)
        pending = surface._pending["page-1"]
        getattr(pending.dialog, answer)()
        surface.idle(0.3)
        recording = surface.hand_back()
        shown = page.locator("#result").inner_text()

    [event] = [item for item in recording.events if item.kind is ManualKind.DIALOG]
    assert shown == result
    assert event.detail is detail
    assert event.dialog == pending.info.dialog_id
    assert "DEFAULT42" not in repr(recording)
    assert (
        classify(event, undeclared(profile), ask=Ask.PERSON, trigger=None)
        is Requirement.FORBIDDEN
    )
    assert gaps_in(recording.events) == ()


LEAVING = """<!doctype html><html><body><h1>Draft</h1>
<p id="draft">Unsaved</p>
<script>addEventListener('beforeunload', (event) => {
  event.preventDefault();
  event.returnValue = '';
});</script>
</body></html>"""


def test_a_dialog_whose_answer_the_page_cannot_report_stays_unknown(
    recorded,
) -> None:
    """No page function opens the dialog before a page unloads.

    The person accepts it, and the page reloads, but the recorder cannot tell
    which answer they chose. The event keeps the dialog id without an answer,
    and the recording reports the gap. The classifier forbids the step when
    neither answer is granted. When both are granted, the step needs a person
    on every run.
    """
    actions = yaml.safe_load(SITE_PROFILE.read_text())["actions"]
    actions["accept_dialog"] = {"any": "safe"}
    actions["dismiss_dialog"] = {"any": "safe"}
    with recorded({"/": LEAVING}, actions=actions) as (surface, profile):
        page = surface._page
        hand_over(surface)
        page.click("#draft")
        page.evaluate("setTimeout(() => location.reload(), 0)")
        surface.idle(0.3)
        pending = surface._pending["page-1"]
        pending.dialog.accept()
        page.wait_for_load_state("domcontentloaded")
        surface.idle(0.3)
        recording = surface.hand_back()

    [event] = [item for item in recording.events if item.kind is ManualKind.DIALOG]
    assert pending.info.kind == "beforeunload"
    assert event.detail is Detail.NONE
    assert event.dialog == pending.info.dialog_id
    assert Gap(GapKind.DIALOG_ANSWER_UNKNOWN) in gaps_in(recording.events)
    assert (
        classify(event, undeclared(profile), ask=Ask.PERSON, trigger=None)
        is Requirement.FORBIDDEN
    )
    assert (
        classify(event, profile, ask=Ask.PERSON, trigger=None)
        is Requirement.PERSON_EACH_RUN
    )
