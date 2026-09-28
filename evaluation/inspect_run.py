"""Run fixture evaluation in a visible browser and save structural diagnostics.

Arguments not defined here pass to ``evaluation.integrated``. Diagnostics
retain call counts, response codes, token usage, and control state. Goals,
tool arguments, model text, screenshots, and operator notes are not saved.
A loopback CDP endpoint lets an operator inspect the live replay. This
wrapper never answers an intervention. ``--model`` selects the evaluation
model without changing prompts. ``--success-only`` omits outcome discovery and
non-success replay cases but retains configured fault cases.

Run it with ``uv run --env-file .env python -m evaluation.inspect_run`` and
supply new ``--out`` and ``--operator-file`` folders inside ``.runtime``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import socket
import sys
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx
from playwright.sync_api import Error, sync_playwright

from computeruse import policy
from computeruse.browser import BrowserLimits, BrowserSurface, _session
from computeruse.control import Status
from computeruse.evidence import EvidenceParser, Projection, run_logged, save_event
from computeruse.profile import Profile
from computeruse.surface import SurfaceError
from evaluation import integrated
from evaluation.operator import FileChannel
from evaluation.sites import ROOT

LOG = logging.getLogger(__name__)


def _write(path: Path, event: object) -> None:
    save_event(path, event)


@dataclasses.dataclass(frozen=True)
class _Request:
    call: int
    time: float
    stored: bool | None
    messages: int
    images: int
    tools: int


@dataclasses.dataclass(frozen=True)
class _Response:
    call: int
    time: float
    http_status: int
    metadata_available: bool
    output_items: int
    input_tokens: int | None
    output_tokens: int | None


@dataclasses.dataclass(frozen=True)
class _Browser:
    active: bool
    port: int


@dataclasses.dataclass(frozen=True)
class _TaskConfiguration:
    index: int
    writes: bool
    outcomes: int
    replays: int
    faulted: bool
    omitted_outcomes: int
    omitted_replays: int


@dataclasses.dataclass(frozen=True)
class _Invocation:
    success_only: bool
    cdp_port: int
    argument_count: int
    tasks: tuple[_TaskConfiguration, ...]


class InspectionChannel(FileChannel):
    """Keep operator status beside the existing channel's command file."""

    def __init__(self, folder: Path) -> None:
        super().__init__(folder)
        self._projection = Projection()

    def show(self, status: Status) -> None:
        """Publish live status and retain only its sanitized structure."""
        super().show(status)
        save_event(
            self.commands.parent / "status.json", status, projection=self._projection
        )
        save_event(
            self.commands.parent / "statuses.jsonl",
            status,
            append=True,
            projection=self._projection,
        )


@dataclasses.dataclass
class Inspection:
    """Collect bounded evaluator calls and expose its current browser locally."""

    folder: Path
    port: int
    calls: int = 0

    def request(self, request: httpx.Request) -> None:
        """Count request structure without saving its content or images."""
        if request.url.host != "api.openai.com":
            return
        self.calls += 1
        request.extensions["inspection_call"] = self.calls
        body = _object(request.content)
        inputs = body.get("input")
        messages = inputs if isinstance(inputs, list) else []
        images = sum(
            1
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            for item in message["content"]
            if isinstance(item, dict) and item.get("type") == "input_image"
        )
        stored = body.get("store")
        tools = body.get("tools")
        _write(
            self.folder / f"call-{self.calls:04d}-request.json",
            _Request(
                self.calls,
                time.time(),
                stored if isinstance(stored, bool) else None,
                len(messages),
                images,
                len(tools) if isinstance(tools, list) else 0,
            ),
        )
        LOG.info("Saved request %s", self.calls)

    def response(self, response: httpx.Response) -> None:
        """Save the response status and token counts, never returned content."""
        call = response.request.extensions.get("inspection_call")
        if not isinstance(call, int):
            return
        response.read()
        body = _object(response.content)
        output = body.get("output")
        usage = body.get("usage")
        tokens = usage if isinstance(usage, dict) else {}
        _write(
            self.folder / f"call-{call:04d}-response.json",
            _Response(
                call,
                time.time(),
                response.status_code,
                bool(body),
                len(output) if isinstance(output, list) else 0,
                _count(tokens.get("input_tokens")),
                _count(tokens.get("output_tokens")),
            ),
        )
        LOG.info("Saved response %s with HTTP status %s", call, response.status_code)

    def client(self) -> httpx.Client:
        """Keep the evaluator's usage counter alongside inspection hooks."""
        return httpx.Client(
            event_hooks={
                "request": [self.request],
                "response": [integrated.USAGE.count, self.response],
            }
        )

    @contextlib.contextmanager
    def session(
        self,
        profile: Profile,
        entry_url: str,
        *,
        headless: bool = False,
        limits: BrowserLimits | None = None,
        record: bool = False,
    ) -> Iterator[BrowserSurface]:
        """Use production session setup and cleanup with loopback debugging."""
        if headless:
            raise ValueError("inspection requires a headed browser")
        if policy.route_for(profile, entry_url) is None:
            raise SurfaceError("the entry point is outside the profile")
        endpoint = f"http://127.0.0.1:{self.port}"
        with socket.socket() as available:
            available.bind(("127.0.0.1", self.port))
        with sync_playwright() as driver:
            browser = driver.chromium.launch(
                headless=False,
                args=[
                    "--remote-debugging-address=127.0.0.1",
                    f"--remote-debugging-port={self.port}",
                ],
            )
            try:
                _write(
                    self.folder / "browser.json",
                    _Browser(True, self.port),
                )
                LOG.info("Browser inspection endpoint %s", endpoint)
                yield from _session(
                    browser,
                    profile,
                    entry_url,
                    limits or BrowserLimits(),
                    visible=True,
                    record=record,
                )
            finally:
                with contextlib.suppress(Error):
                    browser.close()
                _write(
                    self.folder / "browser.json",
                    _Browser(False, self.port),
                )


def _object(contents: bytes) -> dict[str, object]:
    try:
        value = json.loads(contents)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _runtime_path(value: str) -> Path:
    path = Path(value).resolve()
    runtime = (ROOT / ".runtime").resolve()
    if path == runtime or not path.is_relative_to(runtime):
        raise argparse.ArgumentTypeError(
            "choose a folder inside the repository's .runtime folder"
        )
    return path


def _task_configuration(
    index: int, original: integrated.Task, selected: integrated.Task
) -> _TaskConfiguration:
    return _TaskConfiguration(
        index,
        selected.writes,
        len(selected.outcomes),
        len(selected.replays),
        selected.faulted is not None,
        len(original.outcomes) - len(selected.outcomes),
        len(original.replays) - len(selected.replays),
    )


def main() -> None:
    """Run integrated evaluation with live inspection and structural evidence."""
    parser = EvidenceParser(
        description=__doc__.split("\n", 1)[0],
        epilog="Other arguments pass through, including --app, --task, and --visual.",
    )
    parser.add_argument("--operator-file", type=_runtime_path, required=True)
    parser.add_argument("--out", type=_runtime_path, required=True)
    parser.add_argument("--cdp-port", type=int, default=4399)
    parser.add_argument("--model", default=integrated.MODEL)
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="omit outcome cases while retaining successful replays and fault cases",
    )
    parser.add_argument("--verbose", action="store_true")
    args, remaining = parser.parse_known_args()
    if not args.model.strip():
        parser.error("--model must name a model")
    if not 1024 <= args.cdp_port <= 65535:
        parser.error("--cdp-port must be between 1024 and 65535")
    if args.operator_file.is_relative_to(args.out):
        parser.error("--operator-file must sit outside --out")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.operator_file.mkdir(parents=True, exist_ok=False)
    inspection = Inspection(args.operator_file, args.cdp_port)
    tasks = (
        tuple(
            dataclasses.replace(
                task,
                outcomes=(),
                replays=tuple(
                    case for case in task.replays if case.expect == "success"
                ),
            )
            for task in integrated.TASKS
        )
        if args.success_only
        else integrated.TASKS
    )
    _write(
        args.operator_file / "invocation.json",
        _Invocation(
            args.success_only,
            args.cdp_port,
            len(remaining),
            tuple(
                _task_configuration(index, original, selected)
                for index, (original, selected) in enumerate(
                    zip(integrated.TASKS, tasks, strict=True), start=1
                )
            ),
        ),
    )
    with (
        patch.object(sys, "argv", [sys.argv[0], *remaining, "--out", str(args.out)]),
        patch.object(integrated, "open_session", inspection.session),
        patch.object(integrated, "model_client", inspection.client),
        patch.object(integrated, "CHANNEL", InspectionChannel(args.operator_file)),
        patch.object(
            integrated, "LunaDecider", partial(integrated.LunaDecider, model=args.model)
        ),
        patch.object(integrated, "MODEL", args.model),
        patch.object(integrated, "TASKS", tasks),
    ):
        integrated.main()


if __name__ == "__main__":
    raise SystemExit(run_logged(main, command="evaluation"))
