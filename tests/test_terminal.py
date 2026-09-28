"""Drive the terminal channel through a real pseudo-terminal.

The terminal, its reader thread, and the run control are real. The seat, clock,
and operator are scripted. A test types lines into the pseudo-terminal as a
person would, while a background thread acts as the run's owner thread and
asks the control for an answer. No test opens a browser or calls a model.
"""

import contextlib
import errno
import io
import os
import termios
import threading
import time
from collections.abc import Callable, Iterator

import pytest
from fakes import FakeClock, ScriptedSeat

from computeruse.actions import Action, AxLocator
from computeruse.budget import Budget
from computeruse.control import Control, Order
from computeruse.escalation import (
    Ask,
    Command,
    Handoff,
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
from computeruse.profile import ActionKind, Profile
from computeruse.terminal import TerminalChannel

GOAL = "Pay invoice 4471 for member 12345"


@contextlib.contextmanager
def terminal() -> Iterator[tuple[int, TerminalChannel, io.StringIO]]:
    """Yield the typing end of a pseudo-terminal and a channel reading the other.

    Echo is off so what the test types does not pile up unread, and the
    terminal stays in line mode, the way a person's terminal is.
    """
    master, slave = os.openpty()
    modes = termios.tcgetattr(slave)
    modes[3] = (modes[3] & ~termios.ECHO) | termios.ICANON
    termios.tcsetattr(slave, termios.TCSANOW, modes)
    out = io.StringIO()
    try:
        with os.fdopen(slave) as inp:
            channel = TerminalChannel(out=out, inp=inp)
            try:
                yield master, channel, out
            finally:
                channel.close()
    finally:
        os.close(master)


def typing(master: int, text: str) -> None:
    os.write(master, text.encode())


def printed(out: io.StringIO, text: str, times: int = 1) -> str:
    """Wait until ``text`` has been printed ``times`` times, and return the output."""
    deadline = time.monotonic() + 5
    while out.getvalue().count(text) < times:
        assert time.monotonic() < deadline, f"never printed {text!r}:\n{out.getvalue()}"
        time.sleep(0.005)
    return out.getvalue()


def asked(out: io.StringIO, intervention: str, times: int = 1) -> str:
    """Wait until the terminal has printed a request and the status after it.

    The terminal discards earlier input before it prints that status line,
    so a line typed after it is read.
    """
    return printed(out, f"request {intervention}. Type:", times)


def running(
    channel: TerminalChannel, profile: Profile, *, seated: bool = False
) -> tuple[Control, ScriptedSeat | None]:
    """Return a control whose run has started, with ``channel`` attached."""
    clock = FakeClock()
    seat = ScriptedSeat(clock) if seated else None
    control = Control(
        mode=Mode.DISCOVERY,
        clock=clock,
        seat=seat,
        channels=[channel],
        worker=False,
        interval_s=0.002,
    )
    if seat is not None:
        seat.control = control
    assert control.begin(
        profile=profile,
        budget=Budget(profile.budgets, clock),
        journal=MemoryJournal(),
        context=GOAL,
    )
    return control, seat


def request(
    trigger: Trigger = Trigger.RISKY_ACTION,
    *,
    ask: Ask = Ask.APPROVAL,
    reason: str = "sending a payment is declared risky",
) -> InterventionRequest:
    return InterventionRequest(
        trigger=trigger,
        goal=GOAL,
        profile_id="acme-cu/vendor-core-9.2/sandbox",
        step=3,
        route="/members/:id/payments",
        reason=reason,
        timeout_s=900,
        action=Action(ActionKind.CLICK, AxLocator("button", "Send"), effect="pay"),
        ask=ask,
    )


@contextlib.contextmanager
def asking(
    control: Control, wanted: InterventionRequest
) -> Iterator[Callable[[], Handoff]]:
    """Ask the control on a thread of its own, as the loop's owner thread would.

    Yields a function that waits for the answer. The request is always ended
    before the block exits, so no thread outlives the test.
    """
    answers: list[Handoff] = []
    owner = threading.Thread(
        target=lambda: answers.append(control.intervene(wanted)), daemon=True
    )
    owner.start()

    def answer() -> Handoff:
        owner.join(timeout=5)
        assert answers, "the request was never answered"
        return answers[0]

    try:
        yield answer
    finally:
        if owner.is_alive():
            status = control.status()
            control.submit(Order(Command.TERMINATE, status.run, status.revision))
            owner.join(timeout=5)


def test_a_stop_typed_at_the_terminal_is_shown_as_stopping_at_once(profile):
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)
        printed(out, "[running]")

        typing(master, "stop\n")

        shown = printed(out, "[stopping]")
        # Nothing on this thread delivered the command: the terminal's own
        # thread submitted it and printed the receipt while the run was busy.
        assert "stop: accepted" in shown
        assert control.status().state is State.STOPPING


def test_a_new_request_is_printed_once_with_its_context_and_the_words_to_type(
    profile,
):
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile, seated=True)
        with asking(control, request()) as answer:
            asked(out, "iv-1")
            typing(master, "resume iv-1 wrong invoice\n")
            assert answer().outcome is HandoffOutcome.RESUMED

    shown = out.getvalue()
    assert shown.count("Request iv-1") == 1
    assert f"goal:        {GOAL}" in shown
    assert "reason:      sending a payment is declared risky" in shown
    assert "proposal:    click button 'Send' (effect: pay)" in shown
    assert "route:       /members/:id/payments" in shown
    assert "session:     the scripted session" in shown
    assert "900s left of 900s" in shown
    assert "answers:     resume iv-1 | approve iv-1 | terminate" in shown
    assert "resume: accepted" in shown


def test_an_answer_without_the_request_id_is_refused_before_it_reaches_the_run(
    profile,
):
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)
        with asking(control, request()) as answer:
            asked(out, "iv-1")

            typing(master, "approve\n")

            printed(out, "approve must name the request it answers")
            assert control.status().state is State.AWAITING_APPROVAL
            typing(master, "approve iv-1\n")
            assert answer().outcome is HandoffOutcome.APPROVED


def test_an_answer_naming_an_earlier_request_is_refused_as_stale(profile):
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)
        with asking(control, request()) as answer:
            asked(out, "iv-1")
            typing(master, "resume iv-1\n")
            assert answer().outcome is HandoffOutcome.RESUMED

        with asking(control, request()) as answer:
            asked(out, "iv-2")

            typing(master, "approve iv-1\n")

            shown = printed(out, "approve: refused as stale")
            assert "the open request is iv-2" in shown
            assert control.status().state is State.AWAITING_APPROVAL
            typing(master, "terminate\n")
            assert answer().outcome is HandoffOutcome.TERMINATED


def test_type_ahead_typed_before_a_request_is_printed_does_not_answer_it(profile):
    with terminal() as (master, channel, out):
        control, seat = running(channel, profile, seated=True)
        assert seat is not None
        printed(out, "[running]")
        # A person starts typing while the automation runs, and has not
        # pressed Enter when the run asks about something new.
        typing(master, "stop")

        with asking(control, request()) as answer:
            asked(out, "iv-1")
            typing(master, "\nstatus\n")
            printed(out, "Request iv-1", times=2)

            assert control.status().state is State.AWAITING_APPROVAL
            assert Owner.HUMAN not in seat.transfers
            typing(master, "resume iv-1\n")
            assert answer().outcome is HandoffOutcome.RESUMED


def test_a_command_typed_against_a_status_the_terminal_never_printed_is_stale(
    profile,
):
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)
        printed(out, "[running]")
        # The run moves on without the terminal printing it, as when another
        # channel stops it while the owner thread is inside a browser call.
        status = control.status()
        control.submit(Order(Command.STOP, status.run, status.revision, via=Via.PANEL))

        typing(master, "terminate\n")

        shown = printed(out, "terminate: refused as stale")
        assert "the run changed since that command was sent" in shown
        printed(out, "[stopping]")
        typing(master, "terminate\n")
        printed(out, "terminate: accepted")
        assert control.status().state is State.TERMINATED


def test_approve_is_refused_where_the_request_asks_for_a_person(profile):
    evidence = request(
        Trigger.RECORD_EVIDENCE_REQUIRED,
        ask=Ask.PERSON,
        reason="click here must name its record; please perform the step",
    )
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile, seated=True)
        with asking(control, evidence) as answer:
            shown = asked(out, "iv-1")
            assert "Request iv-1 needs a person (record_evidence_required)" in shown
            assert "Approving is not offered" in shown
            assert "answers:     resume iv-1 | terminate" in shown

            typing(master, "approve iv-1\n")

            shown = printed(out, "approve: refused as invalid")
            assert "this request cannot be approved" in shown
            assert control.status().state is State.PAUSED
            typing(master, "terminate\n")
            assert answer().outcome is HandoffOutcome.TERMINATED


def test_an_unverified_result_explains_that_approving_confirms_what_the_person_sees(
    profile,
):
    confirm = request(
        Trigger.UNVERIFIED_RESULT,
        reason="the result could not be checked against the page",
    )
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)
        with asking(control, confirm) as answer:
            shown = asked(out, "iv-1")
            assert "approve iv-1 confirms the result you see there" in shown
            typing(master, "terminate\n")
            assert answer().outcome is HandoffOutcome.TERMINATED


def test_control_characters_in_live_text_are_printed_as_question_marks(profile):
    out = io.StringIO()
    channel = TerminalChannel(out=out, inp=io.StringIO())
    control, _ = running(channel, profile)
    hostile = request(reason="Balance\x1b[2J\x1b[1;1Hpaid\rin full\x07\nyes")

    control.intervene(hostile)

    shown = out.getvalue()
    assert "reason:      Balance?[2J?[1;1Hpaid?in full??yes" in shown
    for character in ("\x1b", "\r", "\x07"):
        assert character not in shown


def test_a_terminal_that_is_not_interactive_does_not_listen_and_a_request_times_out(
    profile,
):
    out = io.StringIO()
    channel = TerminalChannel(out=out, inp=io.StringIO("approve iv-1\n"))
    control, _ = running(channel, profile)

    assert not channel.listening()
    started = time.monotonic()
    handoff = control.intervene(request())

    assert handoff.outcome is HandoffOutcome.TIMED_OUT
    assert time.monotonic() - started < 1
    assert "this terminal does not read commands" in out.getvalue()
    assert "Type:" not in out.getvalue()


@pytest.mark.parametrize(
    ("line", "reply"),
    [
        ("approve-it", "commands: start | stop | resume"),
        ("stop now", "stop takes nothing after it"),
        ("help", "resume <id> [note]"),
    ],
)
def test_input_that_is_not_a_command_is_answered_here_and_sends_nothing(
    profile, line, reply
):
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)
        before = control.status().revision

        typing(master, f"{line}\n")

        printed(out, reply)
        assert control.status().revision == before
        assert control.status().state is State.RUNNING


def test_a_terminal_whose_input_ends_stops_listening(profile):
    with terminal() as (master, channel, out):
        running(channel, profile)
        assert channel.listening()

        typing(master, "\x04")

        printed(out, "terminal input ended")
        assert not channel.listening()


# Findings from the branch review, each written to fail without its fix.


def journaled(
    channel: TerminalChannel, profile: Profile
) -> tuple[Control, MemoryJournal]:
    """Return a control whose run has started, and the journal it writes."""
    clock = FakeClock()
    control = Control(
        mode=Mode.DISCOVERY,
        clock=clock,
        channels=[channel],
        worker=False,
        interval_s=0.002,
    )
    journal = MemoryJournal()
    assert control.begin(
        profile=profile,
        budget=Budget(profile.budgets, clock),
        journal=journal,
        context=GOAL,
    )
    return control, journal


def test_an_answer_naming_something_other_than_a_request_id_sends_nothing(
    profile,
) -> None:
    with terminal() as (master, channel, out):
        control, journal = journaled(channel, profile)
        with asking(control, request()) as answer:
            asked(out, "iv-1")
            before = control.status().revision

            typing(master, "approve 10001 checked\n")

            printed(out, "approve must name a request id such as iv-2")
            assert control.status().revision == before
            assert control.status().state is State.AWAITING_APPROVAL
            typing(master, "resume iv-1\n")
            assert answer().outcome is HandoffOutcome.RESUMED

    control.halted()
    commanded = [
        (event.command, event.verdict, event.intervention)
        for event in journal.events
        if isinstance(event, Commanded)
    ]
    assert commanded == [(Command.RESUME, Verdict.ACCEPTED, "iv-1")]


def test_a_terminal_held_by_another_job_does_not_listen(profile, monkeypatch) -> None:
    flushed: list[int] = []

    def another_job(fd: int) -> int:
        del fd
        return os.getpgrp() + 1

    def flush(stream: object, queue: int) -> None:
        del stream
        flushed.append(queue)

    monkeypatch.setattr(os, "tcgetpgrp", another_job)
    monkeypatch.setattr(termios, "tcflush", flush)
    with terminal() as (_, channel, out):
        control, seat = running(channel, profile, seated=True)
        assert seat is not None
        # A wait for an answer fails after 50 slices instead of hanging.
        seat.limit = 50

        assert not channel.listening()
        shown = printed(out, "[running]")
        assert "Type:" not in shown

        handoff = control.intervene(request())

    assert handoff.outcome is HandoffOutcome.TIMED_OUT
    assert "this terminal does not read commands" in out.getvalue()
    # Flushing a terminal from a background job would stop the whole process.
    assert flushed == []


def test_a_terminal_that_is_not_the_controlling_one_still_listens(
    profile, monkeypatch
) -> None:
    def not_controlling(fd: int) -> int:
        del fd
        raise OSError(errno.ENOTTY, "not the controlling terminal")

    monkeypatch.setattr(os, "tcgetpgrp", not_controlling)
    with terminal() as (master, channel, out):
        control, _ = running(channel, profile)

        assert channel.listening()
        typing(master, "stop\n")

        printed(out, "stop: accepted")
        assert control.status().state is State.STOPPING


def test_a_later_control_on_the_same_terminal_prints_its_requests_again(profile):
    # Saving opens short controls of its own after the run, each counting its
    # revisions and requests from the start, on the same terminal.
    with terminal() as (master, channel, out):
        first, _ = running(channel, profile, seated=True)
        with asking(first, request()) as answer:
            asked(out, "iv-1")
            typing(master, "resume iv-1\n")
            assert answer().outcome is HandoffOutcome.RESUMED
        before = out.getvalue().count("request iv-1. Type:")
        second, _ = running(channel, profile, seated=True)
        with asking(second, request()) as answer:
            printed(out, "Request iv-1", times=2)
            asked(out, "iv-1", times=before + 1)
            typing(master, "resume iv-1\n")
            assert answer().outcome is HandoffOutcome.RESUMED

    assert out.getvalue().count("Request iv-1") == 2


def test_a_control_after_the_run_ended_still_reads_the_terminal(profile, monkeypatch):
    with terminal() as (master, channel, out):
        entered = threading.Event()
        released = threading.Event()
        reading = threading.Event()
        take = channel._take

        def held(fd):
            if not entered.is_set():
                entered.set()
                assert released.wait(timeout=5)
                return -1, []
            reading.set()
            return take(fd)

        monkeypatch.setattr(channel, "_take", held)
        first, _ = running(channel, profile, seated=True)
        assert entered.wait(timeout=5)
        try:
            first.finish("completed")
            second, _ = running(channel, profile, seated=True)
        finally:
            released.set()
        assert reading.wait(timeout=1), "saving's review has no terminal reader"
        assert channel.listening()
        with asking(second, request()) as answer:
            asked(out, "iv-1")
            typing(master, "resume iv-1\n")
            assert answer().outcome is HandoffOutcome.RESUMED
