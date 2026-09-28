"""An operator channel in the terminal that started the run.

The terminal prints the run's status when it changes and reads commands
typed at it. A reader thread reads each line, submits it to the run's
control, and prints the control's receipt itself. A Stop command entered while
the run is inside a browser call appears immediately, even though the run
acts on it only at its next boundary. The reader thread never touches the
session. It only calls ``Control.submit`` and prints the result.

Every command carries the status revision this terminal last printed, and
an answer carries the request id the person typed. The reader never fills in
the open request or the current revision when it reads a line, because a
line typed before a new request was printed would then answer a request the
person never saw. When the terminal prints a new request, it also discards
input that was typed but not yet read.

Everything printed is live operator data. The goal, the reason, and the
proposal can quote the screen, so this module prints them and writes nothing
to a file. Every unprintable character in printed text becomes ``?``, so a
page cannot move the cursor or rewrite what the terminal already shows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import select
import sys
import termios
import threading
from typing import TYPE_CHECKING, TextIO

from computeruse.control import ANSWERS, Offer, Order, Receipt, Status
from computeruse.escalation import Ask, Command, Mode, Trigger, Verdict, Via

if TYPE_CHECKING:
    from computeruse.control import Control

WORDS = {
    "start": Command.START,
    "stop": Command.STOP,
    "resume": Command.RESUME,
    "approve": Command.APPROVE,
    "terminate": Command.TERMINATE,
}
"""The word an operator types for each command, in the order they are listed.

A person may use the session while the run is paused, so an operator needs no
separate take word, and Resume declines a request it answers, so no reject.
"""

LOCAL = frozenset({"status", "help"})
"""Words the terminal answers itself, without sending the run anything."""

USAGE = (
    "commands: start | stop | resume <id> [note] | approve <id> [note] | "
    "terminate | status | help"
)

HELP = (
    "  start                start a run that is waiting for Start",
    "  stop                 pause at the next step; you may then use the session",
    "  resume <id> [note]   hand the session back; declines anything asked",
    "  approve <id> [note]  let the proposal of request <id> proceed once",
    "  terminate            end the run; an operation already sent settles first",
    "  status               print the status and the open request again",
    "  help                 print this list",
    "  <id> is the request id printed with the request, such as iv-3.",
)

ASKING = {
    Ask.APPROVAL: "needs approval",
    Ask.PERSON: "needs a person",
    Ask.PAUSE: "holds the run for the operator",
}

POLL_S = 0.1
"""How long the reader waits for input before it checks whether to stop."""

READ_BYTES = 4096


@dataclasses.dataclass(frozen=True, slots=True)
class Typed:
    """A parsed operator input that the control has not judged.

    ``command`` is None for a word the terminal answers itself. The
    ``intervention`` is exactly what the person typed; the control decides
    whether it names the open request.
    """

    word: str
    command: Command | None = None
    intervention: str = ""
    note: str | None = None


def parse(line: str) -> Typed | str | None:
    """Parse one typed line, or return why it is not a command.

    Returns None for a blank line. An answer must name the request it
    answers, and the words after the id are a note.

    Examples
    --------
    >>> typed = parse("approve iv-3 amount checked")
    >>> typed.command, typed.intervention, typed.note
    (<Command.APPROVE: 'approve'>, 'iv-3', 'amount checked')
    >>> parse("stop").command
    <Command.STOP: 'stop'>
    >>> parse("resume")
    'resume must name the request it answers: resume <id> [note]'
    >>> parse("stop now")
    'stop takes nothing after it'
    """
    words = line.split()
    if not words:
        return None
    word, rest = words[0], words[1:]
    if word in LOCAL:
        return f"{word} takes nothing after it" if rest else Typed(word)
    command = WORDS.get(word)
    if command is None:
        return USAGE
    if command in ANSWERS:
        if not rest:
            return f"{word} must name the request it answers: {word} <id> [note]"
        named = rest[0]
        number = named.removeprefix("iv-")
        if not named.startswith("iv-") or not number.isdigit() or len(named) > 16:
            return f"{word} must name a request id such as iv-2: {word} <id> [note]"
        return Typed(word, command, named, " ".join(rest[1:]) or None)
    if rest:
        return f"{word} takes nothing after it"
    return Typed(word, command)


def printable(text: str) -> str:
    """Replace every character a terminal would not print as itself with ``?``.

    That covers every control character, including escape, carriage return,
    and newline, and format characters that reorder text.

    Examples
    --------
    >>> printable("Balance" + chr(27) + "[2J" + chr(13) + "wiped")
    'Balance?[2J?wiped'
    """
    return "".join(char if char.isprintable() else "?" for char in text)


def answers(status: Status) -> str:
    """Return the exact words an operator may type now.

    Examples
    --------
    >>> from computeruse.control import Button
    >>> from computeruse.escalation import Owner, State
    >>> offered = frozenset({Command.STOP, Command.TERMINATE})
    >>> answers(Status("run-1", Mode.DISCOVERY, 1, State.RUNNING, Owner.AUTOMATION,
    ...                0, Button("Stop", Command.STOP, enabled=True), offered, 60, 0))
    'stop | terminate'
    """
    offer = status.offer
    named = f" {offer.intervention}" if offer is not None else ""
    return " | ".join(
        f"{word}{named}" if command in ANSWERS else word
        for word, command in WORDS.items()
        if command in status.offered
    )


WRITTEN = frozenset(
    {
        Trigger.RUN_CHECKS,
        Trigger.CAPABILITY_REVIEW,
        Trigger.REPLAY_RESULT,
        Trigger.DISCOVERY_RESULT,
    }
)
"""Requests whose reason the run wrote in several lines: summaries and reports."""


def meaning(offer: Offer, mode: Mode = Mode.DISCOVERY) -> str:
    """Explain request-specific command behavior."""
    named = offer.intervention
    if offer.trigger is Trigger.RECORD_EVIDENCE_REQUIRED:
        return (
            "Approving is not offered: the run cannot prove which record this "
            "step acts on, so a person must do the step in the session."
        )
    if offer.trigger is Trigger.UNCONFIRMED_TEXT:
        return (
            f"Answering approve {named} confirms these texts are the website's "
            f"own and saves the capability. resume {named} NAME=VALUE compares "
            "them with a second test record instead. terminate saves nothing."
        )
    if offer.trigger is Trigger.DELIVERY_UNCERTAIN and offer.ask is Ask.APPROVAL:
        return (
            f"Answering approve {named} says this step went through, and the "
            f"replay goes on. resume {named} reads the page again and stops the "
            "replay if it still cannot tell. The step is never sent again."
        )
    if offer.trigger is Trigger.RUN_CHECKS:
        return (
            f"Answering approve {named} runs these checks. resume {named} or "
            "terminate skips them, and the draft is saved without them."
        )
    if offer.trigger in {Trigger.REPLAY_RESULT, Trigger.DISCOVERY_RESULT}:
        return f"The run has finished. resume {named} closes it."
    if offer.trigger is Trigger.CAPABILITY_REVIEW:
        return (
            f"Answering approve {named} saves an approved copy that replay "
            f"runs without a model. resume {named} or terminate saves only "
            "the draft, which scripts/review.sh can approve later."
        )
    if (
        offer.trigger is Trigger.UNVERIFIED_RESULT
        and offer.ask is Ask.APPROVAL
        and mode is Mode.REPLAY
    ):
        return (
            f"The page cannot show all of the result. Answering approve {named} "
            f"says the result is correct, and the replay goes on. resume {named} "
            "stops the replay."
        )
    if offer.trigger is Trigger.UNVERIFIED_RESULT and offer.ask is Ask.APPROVAL:
        return (
            "The run could not check its result against the page. Answering "
            f"approve {named} confirms the result you see there, and resume "
            f"{named} sends the run back to keep working, with your note."
        )
    if offer.trigger is Trigger.UNVERIFIED_RESULT:
        return "The run could not check its result against the page."
    if offer.ask is Ask.APPROVAL:
        return (
            f"Answering approve {named} lets this one proposal proceed once, if "
            f"the policy still permits it, and resume {named} declines it and "
            "lets the run continue."
        )
    if offer.ask is Ask.PERSON:
        return "Approving is not offered: an approval would not supply what is missing."
    return ""


def _is_tty(stream: TextIO) -> bool:
    """Report whether ``stream`` is a terminal this process may read.

    A run started as a background job still has the terminal as its input,
    but reading it or flushing it would stop the whole process until the job
    is brought to the foreground, so only the foreground job counts.
    """
    try:
        if not stream.isatty():
            return False
        fd = stream.fileno()
    except (OSError, ValueError):
        return False
    try:
        return os.tcgetpgrp(fd) == os.getpgrp()
    except OSError:
        # Not this process's controlling terminal, so no job control applies.
        return True


class TerminalChannel:
    """Show run status and read operator commands in a terminal.

    Parameters
    ----------
    out
        Where the status, requests, and receipts are printed.
    inp
        Where commands are read from. Only a terminal is read, because a
        pipe or a file would answer requests nobody saw.
    """

    def __init__(self, out: TextIO = sys.stderr, inp: TextIO = sys.stdin) -> None:
        self._out = out
        self._inp = inp
        self._lock = threading.Lock()
        self._control: Control | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._closed = False
        # Closed on purpose, or the input ended: reading never starts again.
        self._shut = False
        self._seen = -1
        self._key: tuple[object, ...] = ()
        self._introduced = ""
        self._run = ""
        self._partial = b""
        self._bound = -1

    def attach(self, control: Control) -> None:
        """Start reading commands for ``control`` when the input is a terminal.

        The reader stays available across discovery, checks, and review.
        Closing the channel or reaching the end of input stops it.
        """
        self._control = control
        if self._shut or not _is_tty(self._inp):
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped.clear()
        self._closed = False
        self._thread = threading.Thread(
            target=self._read, name="computeruse-terminal", daemon=True
        )
        self._thread.start()

    def listening(self) -> bool:
        """Report whether commands typed here reach the run."""
        return not self._closed and _is_tty(self._inp)

    def show(self, status: Status) -> None:
        """Print ``status`` if an operator would see something new in it."""
        with self._lock:
            self._render(status)

    def close(self) -> None:
        """Stop reading commands. Safe to call more than once."""
        self._shut = True
        self._closed = True
        self._stopped.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    # Every writing method below requires the print lock.

    def _write(self, *lines: str) -> None:
        for line in lines:
            print(printable(line), file=self._out, flush=True)

    def _render(
        self, status: Status, *, force: bool = False, again: bool = False
    ) -> None:
        """Print the status line and print new request context once.

        ``force`` prints the status line even when nothing in it changed,
        for a receipt. ``again`` prints the open request's context again,
        for the status command. A status older than one already printed is
        not printed, because a receipt can arrive after the run has moved
        on. Input typed before a new request is discarded once the request
        is printed, so nothing typed earlier can answer it, and the status
        line follows.
        """
        if status.run != self._run:
            # A new control counts its revisions and requests from the start.
            self._run, self._seen, self._key, self._introduced = status.run, -1, (), ""
        if status.revision < self._seen:
            return
        self._seen = status.revision
        offer = status.offer
        current = offer.intervention if offer is not None else ""
        fresh = offer is not None and current != self._introduced
        if offer is not None and (fresh or again):
            self._introduce(status, offer)
        if fresh:
            self._introduced = current
            self._discard_input()
        key = (status.state, status.owner, status.step, current, status.notice)
        if force or fresh or key != self._key:
            self._write(self._line(status))
            self._key = key

    def _line(self, status: Status) -> str:
        parts = [
            f"[{status.state.value}] owner {status.owner.value}",
            f"step {status.step}",
            f"{status.remaining_s:.0f}s of run time left",
        ]
        if status.offer is not None:
            parts.append(f"request {status.offer.intervention}")
        if status.settling:
            parts.append(status.settling)
        if status.ending and status.ending != status.state.value:
            parts.append(f"ended {status.ending}")
        line = ", ".join(parts)
        if status.notice:
            line += f". Notice: {status.notice}"
        words = answers(status)
        if words and self.listening():
            line += f". Type: {words}"
        return line

    def _introduce(self, status: Status, offer: Offer) -> None:
        label = "capability:" if status.mode is Mode.REPLAY else "goal:"
        trigger = f" ({offer.trigger.value})" if offer.trigger is not None else ""
        # A summary, a report, or a replay's question keeps its line breaks,
        # each line indented; the run wrote them from its capability and
        # inputs. Any other reason may quote a page, so a break in it stays a
        # question mark and cannot pose as a line this terminal printed.
        reason = offer.reason or "none"
        written = offer.trigger in WRITTEN or status.mode is Mode.REPLAY
        first, *more = reason.split("\n") if written else [reason]
        lines = [
            f"Request {offer.intervention} {ASKING[offer.ask]}{trigger}",
            f"  {label:<13}{offer.context or 'none'}",
            f"  reason:      {first}",
            *(f"               {line}" for line in more),
            f"  proposal:    {offer.proposal or 'none'}",
            f"  route:       {offer.route or 'none'}",
            "  session:     "
            + (offer.session or "nobody can see it, so it cannot be handed over"),
        ]
        if status.interrupted:
            lines.append(f"  interrupted: {status.interrupted}")
        lines.append(
            f"  timeout:     {offer.left_s:.0f}s left of {offer.timeout_s:.0f}s"
        )
        explained = meaning(offer, status.mode)
        if explained:
            lines.append(f"  {explained}")
        if self.listening():
            lines.append(f"  answers:     {answers(status)}")
        else:
            lines.append("  this terminal does not read commands")
        self._write(*lines)

    def _discard_input(self) -> None:
        self._partial = b""
        if not _is_tty(self._inp):
            return
        with contextlib.suppress(termios.error, OSError, ValueError):
            termios.tcflush(self._inp, termios.TCIFLUSH)

    def _receipt(self, word: str, receipt: Receipt) -> None:
        if receipt.verdict is Verdict.ACCEPTED:
            self._write(f"{word}: accepted")
        else:
            self._write(f"{word}: refused as {receipt.verdict.value}; {receipt.reason}")
        self._render(receipt.status, force=True)

    # The methods below run only on the reader thread.

    def _read(self) -> None:
        """Read lines until the channel or its input closes.

        However the thread ends, the channel stops reporting that it
        listens, so a later request does not wait for a reader that is gone.
        """
        try:
            fd = self._inp.fileno()
            while not self._stopped.is_set():
                taken = self._take(fd)
                if taken is None:
                    return
                revision, lines = taken
                for line in lines:
                    self._handle(line, revision)
        except (OSError, ValueError):
            return
        finally:
            self._closed = True

    def _take(self, fd: int) -> tuple[int, list[str]] | None:
        """Wait briefly for input, and return complete lines with their revision.

        The bytes are read under the print lock, so a line is read either
        before a new request is printed, and carries the revision before it,
        or after the request was printed and earlier input was discarded.
        Returns None once the input has ended.
        """
        try:
            ready, _, _ = select.select([fd], [], [], POLL_S)
            if not ready:
                return self._seen, []
            with self._lock:
                ready, _, _ = select.select([fd], [], [], 0)
                if not ready:
                    return self._seen, []
                chunk = os.read(fd, READ_BYTES)
                if not chunk:
                    self._closed = True
                    self._shut = True
                    self._write(
                        "terminal input ended; commands typed here are not read"
                    )
                    return None
                if not self._partial:
                    self._bound = self._seen
                *complete, self._partial = (self._partial + chunk).split(b"\n")
                bound = self._bound
                if complete and self._partial:
                    self._bound = self._seen
        except (OSError, ValueError):
            self._closed = True
            return None
        return bound, [line.decode("utf-8", "replace") for line in complete]

    def _handle(self, line: str, revision: int) -> None:
        control = self._control
        typed = parse(line)
        if typed is None or control is None:
            return
        if isinstance(typed, str):
            with self._lock:
                self._write(typed)
            return
        if typed.command is None:
            with self._lock:
                if typed.word == "help":
                    self._write(*HELP)
                else:
                    self._render(control.status(), force=True, again=True)
            return
        receipt = control.submit(
            Order(
                typed.command,
                control.run,
                revision,
                intervention=typed.intervention,
                via=Via.TERMINAL,
                note=typed.note,
            )
        )
        with self._lock:
            self._receipt(typed.word, receipt)
