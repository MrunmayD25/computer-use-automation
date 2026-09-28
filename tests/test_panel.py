"""Drive the operator control window in headless Chromium.

The window, page, binding, policy, control, and clicks are real. Each test
opens the window in a separate browser context. Playwright clicks it from the
test thread, which acts as the run's owner thread. A ``ScriptedSeat`` replaces
the rest of the run, with no loop, model, or session pages. A test either
drives the control directly or opens a pause or approval while scripted moves
click the window as an operator would.

The first tests need no browser. They check the plain data the window draws
and the strict reading of what the window sends back.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator

import pytest
from fakes import FakeClock, Move, ScriptedSeat
from playwright.sync_api import Browser, Error, Page, expect, sync_playwright

from computeruse.actions import Action, AxLocator
from computeruse.budget import Budget
from computeruse.control import Button, Control, Offer, Order, Status
from computeruse.escalation import (
    Ask,
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
from computeruse.journal import Commanded, MemoryJournal
from computeruse.panel import (
    BINDING,
    TITLE,
    ControlWindow,
    open_control_window,
    order_from,
    payload,
)
from computeruse.profile import ActionKind, Profile

GOAL = "Post payment 4471 for member 12345"
ENTRY = "https://sandbox.example.test/members/12345"
MARKUP = "<img src=x onerror=document.body.dataset.pwned=1>"

PRIMARY = "[data-command=primary]"
APPROVE = "[data-command=approve]"
TERMINATE = "[data-command=terminate]"
MESSAGE = "[data-field=message]"
NOTE = "[data-part=note]"


# What the window draws, without a browser.


def status_with(**changes: object) -> Status:
    base = Status(
        run="run-1a2b3c4d",
        mode=Mode.DISCOVERY,
        revision=4,
        state=State.RUNNING,
        owner=Owner.AUTOMATION,
        step=2,
        primary=Button("Stop", Command.STOP, enabled=True),
        offered=frozenset({Command.TERMINATE, Command.STOP, Command.TAKE_CONTROL}),
        remaining_s=241.26,
        paused_s=12.04,
    )
    return dataclasses.replace(base, **changes)


def offer_with(**changes: object) -> Offer:
    base = Offer(
        intervention="iv-2",
        ask=Ask.APPROVAL,
        trigger=Trigger.RISKY_ACTION,
        reason="posting is declared risky",
        context=GOAL,
        proposal="click button 'Post' (effect: post_payment)",
        route="/members/:id",
        session="the scripted session",
        timeout_s=900.0,
        left_s=899.96,
    )
    return dataclasses.replace(base, **changes)


def test_payload_turns_enums_into_values_and_survives_a_json_round_trip() -> None:
    data = payload(status_with())

    assert data == {
        "run": "run-1a2b3c4d",
        "mode": "discovery",
        "purpose": "",
        "revision": 4,
        "state": "running",
        "owner": "automation",
        "step": 2,
        "primary": {"label": "Stop", "command": "stop", "enabled": True},
        "offered": ["stop", "take_control", "terminate"],
        "remaining_s": 241.3,
        "paused_s": 12.0,
        "notice": "",
        "settling": "",
        "interrupted": "",
        "ending": "",
        "offer": None,
    }
    assert json.loads(json.dumps(data)) == data


def test_payload_describes_the_open_request_and_a_button_with_no_command() -> None:
    status = status_with(
        state=State.STOPPING,
        primary=Button("Stopping...", None, enabled=False),
        offered=frozenset({Command.TERMINATE}),
        offer=offer_with(ask=Ask.PAUSE, trigger=None),
    )

    data = payload(status)

    assert data["primary"] == {
        "label": "Stopping...",
        "command": None,
        "enabled": False,
    }
    assert data["offer"] == {
        "intervention": "iv-2",
        "ask": "pause",
        "trigger": None,
        "reason": "posting is declared risky",
        "context": GOAL,
        "proposal": "click button 'Post' (effect: post_payment)",
        "route": "/members/:id",
        "session": "the scripted session",
        "timeout_s": 900.0,
        "left_s": 900.0,
    }
    assert json.loads(json.dumps(data)) == data


def test_payload_passes_markup_through_unchanged_for_the_window_to_set_as_text() -> (
    None
):
    status = status_with(
        notice=MARKUP, offer=offer_with(reason=MARKUP, proposal=MARKUP, context=MARKUP)
    )

    data = payload(status)
    offer = data["offer"]

    assert data["notice"] == MARKUP
    assert isinstance(offer, dict)
    assert offer["reason"] == offer["proposal"] == offer["context"] == MARKUP


# What the window sends, read strictly.


def message_with(**changes: object) -> dict[str, object]:
    message: dict[str, object] = {
        "command": "approve",
        "run": "run-1a2b3c4d",
        "intervention": "iv-2",
        "revision": 7,
        "note": "amount checked against the invoice",
    }
    message.update(changes)
    return message


def test_order_from_builds_a_panel_order_from_a_well_formed_message() -> None:
    order = order_from(message_with())

    assert order == Order(
        Command.APPROVE,
        "run-1a2b3c4d",
        7,
        intervention="iv-2",
        via=Via.PANEL,
        note="amount checked against the invoice",
    )
    assert order is not None
    assert order.command is Command.APPROVE
    assert order.via is Via.PANEL


@pytest.mark.parametrize("note", ["", "   "])
def test_order_from_reads_a_blank_note_as_no_note(note: str) -> None:
    order = order_from(message_with(note=note))

    assert order is not None
    assert order.note is None


def test_order_from_accepts_a_message_without_a_note() -> None:
    message = message_with(command="stop", intervention="")
    del message["note"]

    order = order_from(message)

    assert order is not None
    assert order.command is Command.STOP
    assert order.note is None


def without(key: str) -> dict[str, object]:
    message = message_with()
    del message[key]
    return message


MALFORMED = {
    "not a mapping": ["approve", "run-1a2b3c4d", "iv-2", 7],
    "nothing": None,
    "a string": "approve",
    "an extra key naming another channel": message_with(via="terminal"),
    "no command": without("command"),
    "no run": without("run"),
    "no intervention": without("intervention"),
    "no revision": without("revision"),
    "an unknown command": message_with(command="approve_all"),
    "a command in capitals": message_with(command="APPROVE"),
    "a command that is not text": message_with(command=3),
    "a run that is not text": message_with(run=12),
    "a run too long": message_with(run="r" * 65),
    "an intervention too long": message_with(intervention="i" * 65),
    "an intervention that is not text": message_with(intervention=None),
    "a revision that is a boolean": message_with(revision=True),
    "a revision that is a fraction": message_with(revision=7.0),
    "a revision that is text": message_with(revision="7"),
    "a negative revision": message_with(revision=-1),
    "a note that is null": message_with(note=None),
    "a note too long": message_with(note="n" * 401),
    "a note with a line break": message_with(note="first\nsecond"),
    "a run with an escape character": message_with(run="run-1" + chr(27) + "[2J"),
}


@pytest.mark.parametrize("message", MALFORMED.values(), ids=list(MALFORMED))
def test_order_from_refuses_a_malformed_message(message: object) -> None:
    assert order_from(message) is None


def test_order_from_accepts_the_longest_ids_and_note_it_allows() -> None:
    order = order_from(
        message_with(run="r" * 64, intervention="i" * 64, note="n" * 400)
    )

    assert order is not None


# The window in a real browser.


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as driver:
        launched = driver.chromium.launch(headless=True)
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def window(browser: Browser) -> Iterator[ControlWindow]:
    with open_control_window(browser) as opened:
        yield opened


def page_of(window: ControlWindow) -> Page:
    page = window.page
    assert page is not None
    return page


@dataclasses.dataclass
class WindowSeat(ScriptedSeat):
    """A scripted seat that also delivers the window's messages while it waits.

    A binding call reaches Python only while the owner thread is inside a
    Playwright call. The browser seat waits inside one. For the same reason,
    this seat waits a few milliseconds on the window page. If the run still
    waits after ``limit`` slices, the test fails instead of hanging.
    """

    window: ControlWindow | None = None
    limit: int = 1000

    def idle(self, seconds: float) -> None:
        super().idle(seconds)
        assert self.slices < self.limit, "the scripted operator never finished"
        page = self.window.page if self.window is not None else None
        if page is not None and not page.is_closed():
            page.wait_for_timeout(5)


@dataclasses.dataclass
class Watched:
    """A channel that passes everything to the window and counts the redraws."""

    window: ControlWindow
    shown: list[Status] = dataclasses.field(default_factory=list)

    def attach(self, control: Control) -> None:
        self.window.attach(control)

    def show(self, status: Status) -> None:
        self.shown.append(status)
        self.window.show(status)

    def listening(self) -> bool:
        return self.window.listening()


def running(
    profile: Profile,
    window: ControlWindow,
    *moves: Move,
    watched: Watched | None = None,
    wait_for_start: bool = False,
) -> tuple[Control, MemoryJournal]:
    """Start a scripted run whose only channel is the window."""
    clock = FakeClock()
    seat = WindowSeat(clock, moves=list(moves), location=ENTRY, window=window)
    control = Control(
        mode=Mode.DISCOVERY,
        clock=clock,
        seat=seat,
        channels=[watched or window],
        wait_for_start=wait_for_start,
        worker=False,
    )
    seat.control = control
    journal = MemoryJournal()
    assert control.begin(
        profile=profile,
        budget=Budget(profile.budgets, clock),
        journal=journal,
        context=GOAL,
    )
    return control, journal


def judged(
    control: Control, journal: MemoryJournal
) -> list[tuple[Command, Verdict, Via]]:
    """Return every command the control judged, once its journal is flushed."""
    control.halted()
    return [
        (event.command, event.verdict, event.via)
        for event in journal.events
        if isinstance(event, Commanded)
    ]


def approval(
    profile: Profile, *, name: str = "Post", reason: str = "posting is risky"
) -> InterventionRequest:
    return InterventionRequest(
        trigger=Trigger.RISKY_ACTION,
        goal=GOAL,
        profile_id=profile.profile_id,
        step=1,
        route="/members/:id",
        reason=reason,
        timeout_s=300.0,
        action=Action(
            ActionKind.CLICK, AxLocator("button", name), effect="post_payment"
        ),
        ask=Ask.APPROVAL,
    )


def buttons(page: Page) -> dict[str, object]:
    """Say what each button reads and whether it can be clicked."""
    primary = page.locator(PRIMARY)
    return {
        "primary": (primary.text_content(), primary.is_enabled()),
        # Approve once exists only while the run asks for an approval.
        "approve": page.locator(APPROVE).is_visible()
        and page.locator(APPROVE).is_enabled(),
        "terminate": page.locator(TERMINATE).is_enabled(),
    }


def fields(page: Page, *names: str) -> dict[str, str | None]:
    return {name: page.text_content(f"[data-field={name}]") for name in names}


def send_through_binding(page: Page, message: object) -> dict[str, object]:
    return page.evaluate(
        "([name, message]) => window[name](message)", [BINDING, message]
    )


def test_start_in_the_window_starts_a_run_that_waits_for_it(profile, window) -> None:
    page = page_of(window)
    seen: dict[str, object] = {}

    def start(seat: ScriptedSeat, status: Status) -> bool:
        del seat
        if status.state is not State.READY:
            return False
        seen.update(buttons(page))
        page.click(PRIMARY)
        return True

    control, journal = running(profile, window, start, wait_for_start=True)

    assert control.status().state is State.RUNNING
    assert seen == {
        "primary": ("Start", True),
        "approve": False,
        "terminate": True,
    }
    assert judged(control, journal) == [(Command.START, Verdict.ACCEPTED, Via.PANEL)]
    expect(page.locator(PRIMARY)).to_have_text("Stop")


def test_a_stop_click_is_accepted_and_the_window_shows_stopping_from_the_receipt_alone(
    profile, window
) -> None:
    watched = Watched(window)
    control, journal = running(profile, window, watched=watched)
    page = page_of(window)
    expect(page.locator(PRIMARY)).to_have_text("Stop")
    redraws = len(watched.shown)

    page.click(PRIMARY)

    expect(page.locator(PRIMARY)).to_have_text("Stopping...")
    expect(page.locator(PRIMARY)).to_be_disabled()
    expect(page.locator(MESSAGE)).to_have_text("accepted")
    assert page.text_content("[data-field=state]") == "stopping"
    assert len(watched.shown) == redraws
    assert control.status().state is State.STOPPING
    assert judged(control, journal) == [(Command.STOP, Verdict.ACCEPTED, Via.PANEL)]


def test_a_double_click_on_stop_sends_one_command(profile, window) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(PRIMARY)).to_have_text("Stop")

    page.dblclick(PRIMARY)

    expect(page.locator(PRIMARY)).to_have_text("Stopping...")
    assert judged(control, journal) == [(Command.STOP, Verdict.ACCEPTED, Via.PANEL)]


def test_a_stop_another_channel_already_applied_is_refused_as_a_duplicate(
    profile, window
) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(PRIMARY)).to_have_text("Stop")
    seen = control.status()
    control.submit(Order(Command.STOP, seen.run, seen.revision, via=Via.TERMINAL))

    page.click(PRIMARY)

    expect(page.locator(MESSAGE)).to_have_text(
        "duplicate: that command was already applied"
    )
    expect(page.locator(PRIMARY)).to_have_text("Stopping...")
    assert judged(control, journal) == [
        (Command.STOP, Verdict.ACCEPTED, Via.TERMINAL),
        (Command.STOP, Verdict.DUPLICATE, Via.PANEL),
    ]


def test_a_click_on_a_screen_the_run_has_moved_past_is_refused_as_stale(
    profile, window
) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(PRIMARY)).to_have_text("Stop")
    seen = control.status()
    control.submit(Order(Command.STOP, seen.run, seen.revision, via=Via.TERMINAL))

    # Terminate, confirmed, from a screen drawn before the stop.
    page.click(TERMINATE)
    page.click(TERMINATE)

    expect(page.locator(MESSAGE)).to_have_text(
        "stale: the run changed since that command was sent"
    )
    expect(page.locator(PRIMARY)).to_have_text("Stopping...")
    assert control.status().state is State.STOPPING
    assert judged(control, journal) == [
        (Command.STOP, Verdict.ACCEPTED, Via.TERMINAL),
        (Command.TERMINATE, Verdict.STALE, Via.PANEL),
    ]


def test_the_window_never_draws_a_status_older_than_one_it_drew(
    profile, window
) -> None:
    control, _ = running(profile, window)
    page = page_of(window)
    before = control.status()
    page.click(PRIMARY)
    expect(page.locator(PRIMARY)).to_have_text("Stopping...")

    window.show(before)

    assert page.text_content(PRIMARY) == "Stopping..."
    assert page.locator(PRIMARY).is_disabled()


def test_approve_once_appears_only_while_an_approval_waits(profile, window) -> None:
    page = page_of(window)
    seen: dict[str, object] = {}

    def approve(seat: ScriptedSeat, status: Status) -> bool:
        del seat
        if status.state is not State.AWAITING_APPROVAL:
            return False
        seen.update(buttons(page))
        seen.update(fields(page, "state", "intervention", "ask", "trigger", "left"))
        page.fill(NOTE, "amount checked against the invoice")
        page.click(APPROVE)
        return True

    control, journal = running(profile, window, approve)
    before = buttons(page)

    handoff = control.intervene(approval(profile))

    assert handoff.outcome is HandoffOutcome.APPROVED
    assert handoff.operator_note == "amount checked against the invoice"
    assert before == {
        "primary": ("Stop", True),
        "approve": False,
        "terminate": True,
    }
    left = seen.pop("left")
    assert left in {"5 min 0 s", "4 min 59 s"}
    assert seen == {
        "primary": ("Resume", True),
        "approve": True,
        "terminate": True,
        "state": "awaiting approval",
        "intervention": "iv-1",
        "ask": "approval",
        "trigger": "risky action",
    }
    expect(page.locator(APPROVE)).to_be_disabled()
    expect(page.locator(PRIMARY)).to_have_text("Stop")
    assert page.input_value(NOTE) == ""
    assert (Command.APPROVE, Verdict.ACCEPTED, Via.PANEL) in judged(control, journal)


def test_resume_sends_the_operator_note_with_the_request_it_answers(
    profile, window
) -> None:
    page = page_of(window)
    seen: dict[str, object] = {}

    def resume(seat: ScriptedSeat, status: Status) -> bool:
        del seat
        if status.state is not State.PAUSED:
            return False
        seen.update(buttons(page))
        seen.update(fields(page, "intervention", "ask", "trigger", "reason"))
        page.fill(NOTE, "moved back to the member page")
        page.click(PRIMARY)
        return True

    control, _ = running(profile, window, resume)
    status = control.status()
    control.submit(Order(Command.STOP, status.run, status.revision, via=Via.TERMINAL))

    handoff = control.hold()

    assert handoff.outcome is HandoffOutcome.RESUMED
    assert handoff.operator_note == "moved back to the member page"
    assert seen == {
        "primary": ("Resume", True),
        "approve": False,
        "terminate": True,
        "intervention": "iv-1",
        "ask": "pause",
        "trigger": "none",
        "reason": "stopped by the operator",
    }


def test_a_proposal_named_with_markup_renders_as_text_and_runs_no_script(
    profile, window
) -> None:
    page = page_of(window)
    seen: dict[str, object] = {}

    def reject(seat: ScriptedSeat, status: Status) -> bool:
        del seat
        if status.state is not State.AWAITING_APPROVAL:
            return False
        seen.update(fields(page, "proposal", "reason", "context"))
        seen["images"] = page.evaluate("() => document.images.length")
        page.click(PRIMARY)
        return True

    control, _ = running(profile, window, reject)
    request = approval(profile, name=MARKUP, reason=f"the page says {MARKUP}")

    handoff = control.intervene(dataclasses.replace(request, goal=MARKUP))

    # Resume declines the proposal. Nothing on the page ran.
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert seen == {
        "proposal": f"click button {MARKUP!r} (effect: post_payment)",
        "reason": f"the page says {MARKUP}",
        "context": MARKUP,
        "images": 0,
    }
    assert page.evaluate("() => document.body.dataset.pwned ?? null") is None
    assert page.title() == TITLE


def test_the_window_policy_blocks_an_inline_handler_that_reached_the_document(
    window,
) -> None:
    page = page_of(window)
    page.evaluate(
        """html => {
          window.violations = [];
          document.addEventListener("securitypolicyviolation", event => {
            window.violations.push(event.effectiveDirective);
          });
          document.body.insertAdjacentHTML("beforeend", html);
        }""",
        MARKUP,
    )

    page.wait_for_function("() => window.violations.includes('script-src-attr')")

    assert page.evaluate("() => document.body.dataset.pwned ?? null") is None


def test_a_page_in_another_context_cannot_see_the_window_binding(
    browser, window
) -> None:
    session = browser.new_context()
    try:
        page = session.new_page()
        page.set_content("<p>Member 12345</p>")
        found = page.evaluate("name => typeof window[name]", BINDING)
        helpers = page.evaluate(
            "() => Object.getOwnPropertyNames(window)"
            ".filter(name => name.startsWith('__playwright'))"
        )
    finally:
        session.close()

    assert found == "undefined"
    assert helpers == []
    assert page_of(window).evaluate("name => typeof window[name]", BINDING) == (
        "function"
    )


def test_a_call_from_a_child_frame_of_the_window_is_refused(profile, window) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    status = control.status()
    message = {
        "command": "terminate",
        "run": status.run,
        "intervention": "",
        "revision": status.revision,
    }

    receipt = page.evaluate(
        """async ([name, message]) => {
          const frame = document.createElement("iframe");
          document.body.append(frame);
          await new Promise(resolve => setTimeout(resolve, 100));
          return await frame.contentWindow[name](message);
        }""",
        [BINDING, message],
    )

    assert receipt["verdict"] == "invalid"
    assert receipt["reason"] == "only the control window can send commands"
    assert control.status().state is State.RUNNING
    assert judged(control, journal) == []


def test_a_malformed_message_from_the_window_never_reaches_the_control(
    profile, window
) -> None:
    control, journal = running(profile, window)
    status = control.status()

    receipt = send_through_binding(
        page_of(window),
        {"command": "terminate", "run": status.run, "revision": status.revision},
    )

    assert receipt["verdict"] == "invalid"
    assert receipt["reason"] == "the window sent a command the run cannot read"
    assert control.status().state is State.RUNNING
    assert judged(control, journal) == []


def test_terminate_needs_a_second_click_within_three_seconds(profile, window) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(TERMINATE)).to_be_enabled()

    page.click(TERMINATE)

    expect(page.locator(TERMINATE)).to_have_text("Confirm terminate")
    assert control.status().state is State.RUNNING

    page.click(TERMINATE)

    expect(page.locator(MESSAGE)).to_have_text("accepted")
    assert control.terminated()
    assert judged(control, journal) == [
        (Command.TERMINATE, Verdict.ACCEPTED, Via.PANEL)
    ]


def test_a_double_click_on_terminate_only_asks_for_confirmation(
    profile, window
) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(TERMINATE)).to_be_enabled()

    page.dblclick(TERMINATE)

    expect(page.locator(TERMINATE)).to_have_text("Confirm terminate")
    expect(page.locator(MESSAGE)).to_have_text(
        "Click Terminate again within 3 seconds to end the run"
    )
    assert control.status().state is State.RUNNING
    assert judged(control, journal) == []


def test_a_clicked_button_keeps_no_focus_for_a_later_enter(profile, window) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(PRIMARY)).to_be_enabled()

    page.click(PRIMARY)
    expect(page.locator(MESSAGE)).to_have_text("accepted")
    page.keyboard.press("Enter")
    page.keyboard.press("Space")

    assert page.evaluate("() => document.activeElement.tagName") == "BODY"
    assert judged(control, journal) == [(Command.STOP, Verdict.ACCEPTED, Via.PANEL)]


def test_a_closed_window_is_opened_again_once_and_then_given_up(
    profile, window
) -> None:
    control, journal = running(profile, window)
    first = page_of(window)
    first.close()
    assert window.listening()

    window.show(control.status())

    second = page_of(window)
    assert second is not first
    assert window.listening()
    expect(second.locator(PRIMARY)).to_have_text("Stop")
    second.click(PRIMARY)
    expect(second.locator(PRIMARY)).to_have_text("Stopping...")

    # Closed again before any draw worked, so the window is given up.
    second.close()
    window.show(control.status())

    assert not window.listening()
    assert window.page is None
    assert second.context.pages == []
    assert judged(control, journal) == [(Command.STOP, Verdict.ACCEPTED, Via.PANEL)]


def test_a_page_the_operator_opens_beside_the_window_is_closed(profile, window) -> None:
    control, _ = running(profile, window)
    page = page_of(window)
    extra = page.context.new_page()

    window.show(control.status())

    assert extra.is_closed()
    assert window.listening()
    assert page.context.pages == [page]


def test_the_window_context_loads_nothing(window, site) -> None:
    page = page_of(window).context.new_page()

    with pytest.raises(Error) as refused:
        page.goto(site)

    assert "ERR_BLOCKED_BY_CLIENT" in str(refused.value)


def test_the_window_shows_the_ending_with_every_button_disabled(
    profile, window
) -> None:
    control, _ = running(profile, window)
    page = page_of(window)

    control.finish("completed")

    expect(page.locator(PRIMARY)).to_have_text("Completed")
    assert buttons(page) == {
        "primary": ("Completed", False),
        "approve": False,
        "terminate": False,
    }
    assert fields(page, "state", "ending") == {
        "state": "completed",
        "ending": "Ended: completed",
    }


def test_a_dialog_held_open_in_a_session_page_does_not_block_the_window(
    browser, profile, window
) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    session = browser.new_context()
    dialogs = []

    def hold_open(dialog) -> None:
        dialogs.append(dialog)

    try:
        held = session.new_page()
        held.on("dialog", hold_open)
        held.set_content("<p>Member 12345</p>")
        with held.expect_event("dialog"):
            held.evaluate("() => { setTimeout(() => alert('Post this payment?'), 0); }")

        window.show(control.status())
        page.click(PRIMARY)

        expect(page.locator(PRIMARY)).to_have_text("Stopping...")
        assert len(dialogs) == 1
        assert judged(control, journal) == [(Command.STOP, Verdict.ACCEPTED, Via.PANEL)]
    finally:
        for dialog in dialogs:
            dialog.dismiss()
        session.close()


def test_a_replay_request_names_its_capability_rather_than_a_goal(window) -> None:
    page = page_of(window)

    window.show(
        status_with(
            mode=Mode.REPLAY,
            state=State.PAUSED,
            offer=offer_with(ask=Ask.PERSON, context="post_payment v3"),
        )
    )

    assert page.text_content("[data-part=context]") == "Capability"
    assert fields(page, "context", "mode", "ask") == {
        "context": "post_payment v3",
        "mode": "replay",
        "ask": "person",
    }


def test_a_terminate_confirmation_lapses_after_three_seconds(profile, window) -> None:
    control, journal = running(profile, window)
    page = page_of(window)
    expect(page.locator(TERMINATE)).to_be_enabled()
    page.click(TERMINATE)
    expect(page.locator(TERMINATE)).to_have_text("Confirm terminate")

    expect(page.locator(TERMINATE)).to_have_text("Terminate", timeout=4500)
    page.click(TERMINATE)

    expect(page.locator(TERMINATE)).to_have_text("Confirm terminate")
    assert control.status().state is State.RUNNING
    assert judged(control, journal) == []


# Findings from the branch review, each written to fail without its fix.


def paced(
    profile: Profile, window: ControlWindow, *moves: Move
) -> tuple[Control, MemoryJournal]:
    """Start a scripted run whose clock moves a quarter second per wait slice.

    The control redraws every channel once a second, so a pause in this run
    offers the window a redraw every four slices.
    """
    clock = FakeClock()
    seat = WindowSeat(
        clock, moves=list(moves), wait_s=0.25, location=ENTRY, window=window
    )
    control = Control(
        mode=Mode.DISCOVERY, clock=clock, seat=seat, channels=[window], worker=False
    )
    seat.control = control
    journal = MemoryJournal()
    assert control.begin(
        profile=profile,
        budget=Budget(profile.budgets, clock),
        journal=journal,
        context=GOAL,
    )
    return control, journal


def leave(page: Page, how: str) -> None:
    """Do what an operator can do to the window's page from the browser."""
    if how == "reload":
        page.reload()
    else:
        page.goto("about:blank")


def drawn(window: ControlWindow, label: str) -> bool:
    """Report, without waiting, whether the window's primary button reads ``label``."""
    page = window.page
    if page is None or page.is_closed():
        return False
    primary = page.locator(PRIMARY)
    return primary.count() == 1 and primary.text_content() == label


@pytest.mark.parametrize("how", ["reload", "blank"])
def test_a_reloaded_window_is_drawn_again_and_a_closed_one_still_listens(
    profile, window, how: str
) -> None:
    left: dict[str, object] = {}
    seen: dict[str, object] = {}

    def reload(seat: ScriptedSeat, status: Status) -> bool:
        del seat
        if status.state is not State.PAUSED or not drawn(window, "Resume"):
            return False
        page = page_of(window)
        leave(page, how)
        left["buttons"] = page.locator("button").count()
        left["listening"] = window.listening()
        return True

    def resume(seat: ScriptedSeat, status: Status) -> bool:
        del seat, status
        if not drawn(window, "Resume"):
            return False
        page = page_of(window)
        seen.update(buttons(page))
        page.click(PRIMARY)
        return True

    control, journal = paced(profile, window, reload, resume)
    status = control.status()
    control.submit(Order(Command.STOP, status.run, status.revision, via=Via.TERMINAL))

    handoff = control.hold()

    assert left == {"buttons": 0, "listening": True}
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert seen == {
        "primary": ("Resume", True),
        "approve": False,
        "terminate": True,
    }
    assert judged(control, journal) == [
        (Command.STOP, Verdict.ACCEPTED, Via.TERMINAL),
        (Command.RESUME, Verdict.ACCEPTED, Via.PANEL),
    ]

    page_of(window).close()

    # The next draw opens the window again, so an operator can still use it.
    assert window.listening()


def test_a_note_holding_a_c1_control_character_is_sent_with_resume_and_accepted(
    profile, window
) -> None:
    page = page_of(window)
    typed: list[str] = []

    def resume(seat: ScriptedSeat, status: Status) -> bool:
        del seat
        if status.state is not State.PAUSED:
            return False
        page.fill(NOTE, "moved back\x85to the member page")
        typed.append(page.input_value(NOTE))
        page.click(PRIMARY)
        return True

    control, journal = running(profile, window, resume)
    status = control.status()
    control.submit(Order(Command.STOP, status.run, status.revision, via=Via.TERMINAL))

    handoff = control.hold()

    # The note field kept the character, so the window had to replace it.
    assert typed == ["moved back\x85to the member page"]
    assert handoff.outcome is HandoffOutcome.RESUMED
    assert handoff.operator_note == "moved back to the member page"
    assert judged(control, journal) == [
        (Command.STOP, Verdict.ACCEPTED, Via.TERMINAL),
        (Command.RESUME, Verdict.ACCEPTED, Via.PANEL),
    ]


def test_a_window_closed_after_a_reopen_and_a_good_draw_is_opened_again(
    profile, window
) -> None:
    control, journal = running(profile, window)
    first = page_of(window)
    first.close()
    window.show(control.status())
    second = page_of(window)
    assert second is not first
    # A draw that works shows the reopened window is usable again.
    window.show(control.status())

    second.close()
    window.show(control.status())

    third = page_of(window)
    assert third is not second
    assert not third.is_closed()
    assert window.listening()
    expect(third.locator(PRIMARY)).to_have_text("Stop")
    third.click(PRIMARY)
    expect(third.locator(PRIMARY)).to_have_text("Stopping...")
    assert judged(control, journal) == [(Command.STOP, Verdict.ACCEPTED, Via.PANEL)]


def test_the_window_offers_only_start_stop_resume_approve_and_terminate(
    profile, window
) -> None:
    running(profile, window)
    page = page_of(window)
    commands = page.eval_on_selector_all(
        "button[data-command]", "nodes => nodes.map(node => node.dataset.command)"
    )
    # Stop pauses and lets a person use the session. Resume declines anything
    # asked. Approve once is drawn only while an approval waits.
    assert commands == ["primary", "approve", "terminate"]
    assert not page.locator(APPROVE).is_visible()
