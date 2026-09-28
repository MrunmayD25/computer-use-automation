"""Privacy at command, shell, crash, and evidence-storage boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", [False, True])
def test_documented_shell_logger_never_copies_arguments_or_parser_errors(
    tmp_path, profile_path, failure
):
    log = tmp_path / "command.log"
    marker = "PRIVATE ARGUMENT; $(echo private)-8675"
    command = [
        "bash",
        "-c",
        'source scripts/sites.sh; logged "$@"',
        "privacy-test",
        str(log),
        "computeruse",
        "--goal",
        marker,
        "--website",
        "https://sandbox.example.test/members/8675",
        "--profile",
        str(profile_path),
        *(["--unknown", marker] if failure else []),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == (2 if failure else 0)
    written = log.read_text() + result.stdout + result.stderr
    assert marker not in written
    assert "/members/8675" not in written
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["event"] == "CommandStarted"
    assert records[-1] == {"event": "CommandEnded", "code": result.returncode}
    assert ("invalid_arguments" in written) is failure


@pytest.mark.parametrize("ending", ["success", "error", "cancel", "crash"])
def test_unknown_python_and_native_diagnostics_never_reach_saved_output(
    tmp_path, ending
):
    log = tmp_path / "events.jsonl"
    script = r"""
import os, sys
from computeruse import cli
from computeruse.evidence import FieldSource, fields
from computeruse.journal import RunEnded, JsonlJournal
from computeruse.loop import RunResult, Ending
def operation():
    value = os.environ["PRIVATE_CANARY"]
    print(value)
    print(value, file=sys.stderr)
    os.write(1, value.encode())
    os.write(2, value.encode())
    fields(FieldSource.INPUT, ("member_id", "nickname"))
    with open(os.environ["PRIVATE_JOURNAL"], "x") as stream:
        JsonlJournal(stream).record(RunEnded("failed", 2, value))
    if os.environ["ENDING"] == "error":
        raise RuntimeError(value)
    if os.environ["ENDING"] == "cancel":
        raise KeyboardInterrupt(value)
    if os.environ["ENDING"] == "crash":
        os._exit(9)
    fields(FieldSource.OUTPUT, ("account_number",))
    cli._report_result(RunResult(
        Ending.COMPLETED, 1, value, outputs={"customer_anna_smith": value}
    ))
    return 0
cli._main = operation
raise SystemExit(cli.main())
"""
    marker = "UNDECLARED-PAGE-MODEL-ERROR-9461"
    result = subprocess.run(
        [sys.executable, "-c", script, "--log", str(log)],
        env={
            **os.environ,
            "ENDING": ending,
            "PRIVATE_CANARY": marker,
            "PRIVATE_JOURNAL": str(tmp_path / "journal.jsonl"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    expected = {"success": 0, "error": 2, "cancel": 130, "crash": 9}[ending]
    assert result.returncode == expected
    written = "".join(path.read_text() for path in tmp_path.iterdir())
    assert marker not in written + result.stdout + result.stderr
    assert "customer_anna_smith" not in written + result.stdout + result.stderr
    assert "<member_id>" in written
    assert "<nickname>" in written
    records = [json.loads(line) for line in log.read_text().splitlines()]
    if ending == "crash":
        assert not any(record["event"] == "CommandEnded" for record in records)
    else:
        assert records[-1] == {"event": "CommandEnded", "code": expected}
    if ending == "success":
        assert "<account_number>" in written


def test_log_failure_stops_before_work_and_never_overwrites_prior_evidence(tmp_path):
    log = tmp_path / "existing.log"
    log.write_text("prior evidence\n")
    started = tmp_path / "operation-started"
    script = """
import os
from pathlib import Path
from computeruse.evidence import run_logged
def operation():
    Path(os.environ["OPERATION_STARTED"]).touch()
    return 0
raise SystemExit(run_logged(operation, command="evaluation"))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "--log", str(log)],
        env={**os.environ, "OPERATION_STARTED": str(started)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "log_unavailable" in result.stderr
    assert not started.exists()
    assert log.read_text() == "prior evidence\n"


@pytest.mark.parametrize("failure", ["during_run", "at_end", "close"])
def test_a_log_write_failure_cannot_report_success(
    tmp_path, monkeypatch, capsys, failure
):
    from computeruse.evidence import FieldSource, fields, run_logged

    class FullDisk:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()
            return False

        def write(self, text):
            self.writes += 1
            if self.writes > 1 and failure != "close":
                raise OSError("PRIVATE-DISK-ERROR")
            return self.stream.write(text)

        def flush(self):
            self.stream.flush()

        def close(self):
            self.stream.close()
            if failure == "close":
                raise OSError("PRIVATE-DISK-ERROR")

    original = Path.open
    log = tmp_path / "run.log"
    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *args, **kwargs: (
            FullDisk(original(path, *args, **kwargs))
            if path == log
            else original(path, *args, **kwargs)
        ),
    )
    assert (
        run_logged(
            lambda: (
                fields(FieldSource.INPUT, ("member_id",))
                if failure == "during_run"
                else None
            ),
            command="evaluation",
            log_path=log,
        )
        == 2
    )
    output = capsys.readouterr()
    assert "PRIVATE-DISK-ERROR" not in output.out + output.err
    records = [json.loads(line) for line in output.out.splitlines()]
    assert records[-1] == {"event": "CommandEnded", "code": 2}
    if failure == "close":
        monkeypatch.setattr(Path, "open", original)
        assert json.loads(log.read_text().splitlines()[-1]) == records[-1]
