"""Exchange status lines and operator commands during evaluation runs.

The harness attaches no person by default. A run that asks for one waits for
its handoff timeout. With this evaluation-only channel, each status change
prints one ``STATUS`` line. A person or an agent reads the live page and
appends a line to ``commands.txt``. The channel parses that line exactly as
the terminal channel does and submits it to the run control. Every command
remains bound to its run and status revision. An approval still covers one
proposal once.

One channel serves a whole batch: each run's control attaches it in turn,
and a command meant for a finished run is refused by that run's control.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

from computeruse.control import Control, Order, Status
from computeruse.escalation import Command, Verdict, Via
from computeruse.evidence import safe_record
from computeruse.terminal import Typed, parse


@dataclasses.dataclass(frozen=True)
class _CommandReceipt:
    command: Command
    intervention: str
    verdict: Verdict


class FileChannel:
    """A channel that reads typed commands from a file and prints each status."""

    def __init__(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.commands = folder / "commands.txt"
        self.commands.touch()
        self.read = len(self.commands.read_text().splitlines())
        self.status: Status | None = None
        self.control: Control | None = None
        self.started = False

    def attach(self, control: Control) -> None:
        """Serve ``control`` and start one command reader for the batch."""
        self.control = control
        self.status = None
        if not self.started:
            self.started = True
            threading.Thread(target=self._reader, daemon=True).start()

    def listening(self) -> bool:
        """Report that a person may be answering, so requests wait for them."""
        return True

    def show(self, status: Status) -> None:
        """Print the status a person answers, including any open request."""
        self.status = status
        offer = status.offer
        line: dict[str, object] = {
            "t": round(time.time()),
            "run": status.run,
            "mode": status.mode.value,
            "state": status.state.value,
            "owner": status.owner.value,
            "step": status.step,
            "revision": status.revision,
            "notice": status.notice,
            "ending": status.ending,
        }
        if offer is not None:
            line["offer"] = {
                "id": offer.intervention,
                "ask": offer.ask.value,
                "trigger": offer.trigger.value if offer.trigger else None,
                "reason": offer.reason,
                "proposal": offer.proposal,
                "route": offer.route,
                "timeout_s": offer.timeout_s,
            }
        print("STATUS " + json.dumps(line), flush=True)

    def _reader(self) -> None:
        while True:
            time.sleep(0.5)
            lines = self.commands.read_text().splitlines()
            for text in lines[self.read :]:
                self.submit(text)
            self.read = len(lines)

    def submit(self, text: str) -> None:
        """Parse one typed line and hand it to the current run's control."""
        typed = parse(text)
        status, control = self.status, self.control
        if status is None or control is None or not isinstance(typed, Typed):
            print("PERSON invalid command", flush=True)
            return
        if typed.command is None:
            return
        receipt = control.submit(
            Order(
                typed.command,
                status.run,
                status.revision,
                typed.intervention or "",
                Via.TERMINAL,
                typed.note,
            )
        )
        print(
            "RECEIPT "
            + json.dumps(
                safe_record(
                    _CommandReceipt(typed.command, typed.intervention, receipt.verdict)
                )
            ),
            flush=True,
        )
