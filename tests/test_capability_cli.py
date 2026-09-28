"""Installed artifact review and validation before opening the browser."""

import contextlib
import errno
import json
import os
import select
import signal
import subprocess
import termios
import threading
import time
from pathlib import Path

import pytest

from computeruse.capability import Review, load_capability

EXAMPLE = Path("examples/capabilities/member_transfer.synthetic.json")


def test_review_needs_no_model_and_writes_a_separate_copy(run_cli, tmp_path):
    output = tmp_path / "reviewed.json"
    original = EXAMPLE.read_bytes()
    result = run_cli(
        "review",
        "--capability",
        str(EXAMPLE),
        "--approve",
        "--output",
        str(output),
        OPENAI_API_KEY="",
    )
    assert result.returncode == 0, result.stderr
    assert "<member_id>" in result.stdout
    assert '"command": "review"' in result.stdout
    assert load_capability(output).provenance.review is Review.REVIEWED
    assert EXAMPLE.read_bytes() == original
    again = run_cli(
        "review", "--capability", str(EXAMPLE), "--approve", "--output", str(output)
    )
    assert again.returncode == 2


def test_invalid_replay_opens_no_browser_or_journals(monkeypatch, tmp_path):
    import argparse

    from computeruse import capability_cli, cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid input must not open a browser")

    monkeypatch.setattr(cli, "_browser_run", forbidden)
    args = argparse.Namespace(
        command="replay",
        capability=EXAMPLE,
        profile=Path("examples/profile.yaml"),
        website="https://sandbox.example.test/members",
        input=["undeclared=PRIVATE-INPUT"],
        headed=False,
        journal=tmp_path / "replay.jsonl",
        accept_draft=True,
    )
    assert capability_cli.run(args) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.rule(12)
@pytest.mark.rule(14)
@pytest.mark.parametrize(
    "control", ["uninterrupted", "approve", "terminate", "failed_check"]
)
def test_installed_replay_posts_once_without_a_model_and_keeps_values_private(
    run_cli, tmp_path, control
):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs

    import yaml
    from replay_fakes import _ax, _node, go, limits

    from computeruse.capability import (
        SCHEMA_VERSION,
        Application,
        Bound,
        Capability,
        Destination,
        Field,
        Match,
        Param,
        Present,
        Provenance,
        ProvenanceKind,
        RefKind,
        ResultKind,
        ResultNode,
        Shows,
        SurfaceKind,
        ValueType,
        constant,
        dumps,
        ref,
    )
    from computeruse.profile import ActionKind

    submitted: list[dict[str, list[str]]] = []
    private = "PRIVATE-CLI-MEMO-7462"
    secret = "PRIVATE-PIN-8471"
    account = "PRIVATE-ACCOUNT-2873"
    page_text = "PRIVATE-PAGE-8392"
    controlled = control in {"approve", "terminate"}
    requested, released = threading.Event(), threading.Event()
    form = (
        '<title>Local posting</title><form method="post" action="/work">'
        '<label>Memo<input name="memo"></label>'
        '<label>PIN<input name="pin" type="password"></label>'
        f"<p>{page_text}</p><button>Post</button></form>"
    )
    heading = "Uncertain" if control == "failed_check" else "Posted"
    receipt = (
        f"<title>Local receipt</title><h1>{heading}</h1>"
        f'<label>Account number<input readonly value="{account}"></label>'
        f"<p>{page_text}</p>"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/work?held=1":
                requested.set()
                if not released.wait(timeout=20):
                    self.send_error(503, "test did not release the response")
                    return
            self.reply(form)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            submitted.append(parse_qs(body.decode()))
            self.reply(receipt)

        def reply(self, markup: str) -> None:
            body = markup.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002  inherited HTTP handler signature
            pass

    with contextlib.ExitStack() as stack:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        stack.callback(thread.join, 5)
        stack.callback(server.server_close)
        stack.callback(server.shutdown)
        stack.callback(released.set)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        document = yaml.safe_load(Path("evaluation/profile.yaml").read_text())
        document.update(base_url=base, allow_routes=["/work"], deny_routes=[])
        document["records"] = {"actions": [], "routes": []}
        document["secrets"] = {"session_pin": {"env": "PRIVACY_TEST_PIN"}}
        document["escalation"]["handoff_timeout_s"] = 15 if controlled else 1
        if controlled:
            document["actions"]["click"]["effects"] = {"post_entry": "risky"}
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text(yaml.safe_dump(document))
        check = Shows("receipt", constant("Posted"), Match.EQUALS)
        capability = Capability(
            schema_version=SCHEMA_VERSION,
            capability_id="local_post",
            version=1,
            application=Application(
                document["profile_id"],
                SurfaceKind.BROWSER,
                base,
                "/work",
                (Present("memo"),),
            ),
            provenance=Provenance(
                ProvenanceKind.SYNTHETIC, "", "cli_fixture", Review.REVIEWED
            ),
            inputs=(Field("memo", ValueType.TEXT, True, 1, 80, ()),),
            outputs=(Field("account_number", ValueType.TEXT, True, 1, 80, ()),),
            variables=(),
            secrets=("session_pin",),
            outcomes=(),
            templates=(),
            targets=(
                _ax("memo", "/work", "textbox", "Memo"),
                _ax("pin", "/work", "textbox", "PIN"),
                _ax("account", "/work", "textbox", "Account number"),
                _ax("post", "/work", "button", "Post"),
                _ax("receipt", "/work", "heading", "Posted"),
            ),
            entry="hold" if controlled else "fill",
            nodes=(
                _node(
                    "fill",
                    ActionKind.TYPE,
                    "/work",
                    target="memo",
                    value=ref(RefKind.INPUT, "memo"),
                    transitions=(go("pin"),),
                ),
                _node(
                    "pin",
                    ActionKind.TYPE,
                    "/work",
                    target="pin",
                    value=ref(RefKind.SECRET, "session_pin"),
                    transitions=(go("post"),),
                ),
                _node(
                    "post",
                    ActionKind.CLICK,
                    "/work",
                    target="post",
                    effect="post_entry",
                    mandatory=True,
                    verify=(check,),
                    transitions=(go("read"),),
                ),
                _node(
                    "read",
                    ActionKind.READ,
                    "/work",
                    target="account",
                    into=ref(RefKind.OUTPUT, "account_number"),
                    transitions=(go("done"),),
                ),
                ResultNode(
                    "done",
                    ResultKind.SUCCESS,
                    "",
                    (check, Bound(ref(RefKind.OUTPUT, "account_number"))),
                ),
            ),
            restrictions=(),
            limits=limits(max_wall_clock_s=8, max_help_requests=1),
        )
        if controlled:
            import dataclasses

            capability = dataclasses.replace(
                capability,
                nodes=(
                    _node(
                        "hold",
                        ActionKind.NAVIGATE,
                        "/work",
                        destination=Destination(
                            "/work", (), query=(Param("held", constant("1")),)
                        ),
                        verify=(Present("memo"),),
                        transitions=(go("fill"),),
                    ),
                    *capability.nodes,
                ),
                limits=limits(max_wall_clock_s=30, max_help_requests=3),
            )
        artifact = tmp_path / "capability.json"
        original = dumps(capability)
        artifact.write_text(original)
        journal = tmp_path / "replay.jsonl"
        log = tmp_path / "command.log"
        command = [
            "replay",
            "--capability",
            str(artifact),
            "--profile",
            str(profile_path),
            "--website",
            base + "/work",
            "--input",
            "memo=" + private,
            "--journal",
            str(journal),
            "--log",
            str(log),
        ]
        result = (
            run_cli(*command, OPENAI_API_KEY="", PRIVACY_TEST_PIN=secret)
            if not controlled
            else _controlled_replay(
                command, control, requested, released, submitted, tmp_path, secret
            )
        )
        (tmp_path / "stdout.txt").write_text(result.stdout)
        (tmp_path / "stderr.txt").write_text(result.stderr)
        (tmp_path / "page-effect.json").write_text(json.dumps(submitted, indent=2))
        succeeded = control in {"uninterrupted", "approve"}
        assert result.returncode == (0 if succeeded else 3), (
            result.stdout + result.stderr
        )
        assert submitted == (
            [] if control == "terminate" else [{"memo": [private], "pin": [secret]}]
        )
        assert artifact.read_text() == original
        written = (
            log.read_text()
            + journal.read_text()
            + journal.with_suffix(".human.jsonl").read_text()
            + artifact.read_text()
        )
        if control == "approve":
            assert account in result.stdout
        written += "" if controlled else result.stdout + result.stderr
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        assert events[-1]["event"] == "ReplayEnded"
        assert (events[-1]["status"] == "succeeded") is succeeded
        assert not any(
            marker in written for marker in (private, secret, account, page_text)
        )
        assert "<memo>" in log.read_text()
        if succeeded:
            assert "<account_number>" in log.read_text()
        if control == "failed_check":
            assert any(
                event["event"] == "FailureEvidence" and event["observed"]
                for event in events
            )


class _TerminalReplay:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.transcript = ""
        self.pending = ""

    def send(self, text: str) -> None:
        os.write(self.descriptor, (text + "\n").encode())

    def wait_for(self, text: str) -> str:
        deadline = time.monotonic() + 15
        while True:
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                if text in line:
                    return line.strip()
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"never printed {text!r}:\n{self.transcript}"
            ready, _, _ = select.select([self.descriptor], [], [], remaining)
            assert ready, f"never printed {text!r}:\n{self.transcript}"
            assert self._read(), f"terminal closed before {text!r}:\n{self.transcript}"

    def finish(self, process: subprocess.Popen[bytes]) -> int:
        """Keep reading until ``process`` exits, and return its exit code.

        The replay prints its report after the line a test waits for. A
        terminal nobody reads fills up, and the replay would block writing it.
        """
        deadline = time.monotonic() + 15
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"the replay did not exit:\n{self.transcript}"
            ready, _, _ = select.select([self.descriptor], [], [], min(remaining, 0.2))
            if ready:
                self._read()
        return process.returncode

    def _read(self) -> bool:
        """Read the terminal contents, or return False after it closes."""
        try:
            chunk = os.read(self.descriptor, 4096)
        except OSError as error:
            if error.errno != errno.EIO:
                raise
            chunk = b""
        decoded = chunk.decode(errors="replace")
        self.transcript += decoded
        self.pending += decoded
        return bool(chunk)

    def request(self, state: str) -> str:
        line = self.wait_for(f"[{state}] owner operator")
        assert ". Type:" in line, line
        return line.split("request ", 1)[1].split(".", 1)[0]


def _controlled_replay(
    command: list[str],
    answer: str,
    requested: threading.Event,
    released: threading.Event,
    submitted: list[dict[str, list[str]]],
    folder: Path,
    secret: str,
) -> subprocess.CompletedProcess[str]:
    master, slave = os.openpty()
    modes = termios.tcgetattr(slave)
    modes[3] = (modes[3] & ~termios.ECHO) | termios.ICANON
    termios.tcsetattr(slave, termios.TCSANOW, modes)
    terminal = _TerminalReplay(master)
    root = Path(__file__).resolve().parents[1]
    observed: list[dict[str, object]] = []
    with contextlib.ExitStack() as stack:
        stack.callback(os.close, master)
        stack.callback(os.close, slave)
        process = subprocess.Popen(
            ["computeruse", *command],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={
                **os.environ,
                "OPENAI_API_KEY": "",
                "PRIVACY_TEST_PIN": secret,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "tests"))),
            },
            start_new_session=True,
        )
        try:
            terminal.wait_for("[running] owner automation")
            assert requested.wait(timeout=15), terminal.transcript
            terminal.send("stop")
            terminal.wait_for("stop: accepted")
            terminal.wait_for("[stopping]")
            released.set()
            paused = terminal.request("paused")
            terminal.send("status")
            assert terminal.request("paused") == paused
            observed.append({"state": "paused", "posts": len(submitted)})
            assert submitted == []
            terminal.send(f"resume {paused}")
            first = terminal.request("awaiting_approval")
            observed.append({"state": "approval_after_stop", "posts": len(submitted)})
            assert submitted == []
            terminal.send(f"resume {first}")
            second = terminal.request("awaiting_approval")
            assert second != first
            observed.append({"state": "approval_after_resume", "posts": len(submitted)})
            assert submitted == []
            terminal.send(f"approve {second}" if answer == "approve" else "terminate")
            terminal.wait_for(f"{answer}: accepted")
            terminal.wait_for(
                "replay succeeded" if answer == "approve" else "replay terminated"
            )
            code = terminal.finish(process)
            return subprocess.CompletedProcess(command, code, terminal.transcript, "")
        finally:
            released.set()
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            (folder / "page-effect.json").write_text(json.dumps(submitted, indent=2))
            (folder / "terminal.txt").write_text(terminal.transcript)
            (folder / "ownership-effects.json").write_text(
                json.dumps(observed, indent=2)
            )
