"""Persist typed metadata while keeping arbitrary diagnostics in a live terminal."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import io
import json
import os
import re
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Protocol, TextIO


class CommandKind(enum.StrEnum):
    """Supported command categories, independent of command arguments."""

    POLICY = "policy"
    DISCOVER = "discover"
    REVIEW = "review"
    REPLAY = "replay"
    EVALUATION = "evaluation"


class FieldSource(enum.StrEnum):
    """The role of named values omitted from evidence."""

    INPUT = "input"
    OUTPUT = "output"
    SECRET = "secret"  # noqa: S105


class ProblemCode(enum.StrEnum):
    """Failures that can be reported without exception or argument text."""

    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_POLICY = "invalid_policy"
    INVALID_INPUTS = "invalid_inputs"
    INVALID_DECLARATIONS = "invalid_declarations"
    MISSING_CREDENTIAL = "missing_credential"
    JOURNAL_UNAVAILABLE = "journal_unavailable"
    INVALID_CAPABILITY = "invalid_capability"
    CANCELLED = "cancelled"
    UNEXPECTED_ERROR = "unexpected_error"
    LOG_UNAVAILABLE = "log_unavailable"


class EvidenceError(OSError):
    """Required evidence failed to persist; optional UI failures cannot hide it."""


@dataclasses.dataclass(frozen=True)
class CommandStarted:
    """The command category without its arguments."""

    command: CommandKind


@dataclasses.dataclass(frozen=True)
class CommandEnded:
    """The actual exit code, written only after command execution ends."""

    code: int


@dataclasses.dataclass(frozen=True)
class Problem:
    """A fixed failure category without its live diagnostic."""

    code: ProblemCode


@dataclasses.dataclass(frozen=True)
class Fields:
    """Declared field names whose values must never reach the writer."""

    source: FieldSource
    names: tuple[str, ...]


_REFERENCES = {
    "profile_id": "profile",
    "capability": "capability",
    "run": "run",
    "node": "node",
    "to": "node",
    "intervention": "request",
    "route": "route",
}
_CATEGORIES = {
    "kind": frozenset(
        {"action", "read", "check", "branch", "human", "result", "confirm_result"}
    ),
    "target_kind": frozenset({"AxLocator", "DomLocator", "ScreenTarget"}),
    "stage": frozenset(
        {
            "observe",
            "decide",
            "act",
            "interpret",
            "verify",
            "resolve",
            "model",
            "surface",
        }
    ),
    "tool": frozenset(
        {"act", "finish", "observe", "remember", "flag", "unknown", "none"}
    ),
    "form": frozenset({"ax", "dom", "visual"}),
    "match": frozenset({"equals", "contains"}),
    "name": frozenset({"constant", "input", "output", "variable", "secret"}),
    "scope": frozenset({"constant", "input", "output", "variable", "secret"}),
    "expected": frozenset(
        {
            "at_route",
            "present",
            "absent",
            "shows",
            "value_is",
            "bound",
            "completed",
            "dialog_open",
            "goal_coverage",
        }
    ),
}


def _field_name(name: str, ordinal: int) -> str:
    return name if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) else f"field_{ordinal}"


class Projection:
    """Convert schema metadata to records with anonymous, consistent references."""

    def __init__(self) -> None:
        self.references: dict[str, dict[str, str]] = {}

    def record(self, event: object) -> dict[str, object]:
        """Project a dataclass event, excluding unsupported payloads."""
        if isinstance(event, Fields):
            names = [_field_name(name, i) for i, name in enumerate(event.names, 1)]
            return {
                "event": "Fields",
                "source": event.source.value,
                "fields": {name: f"<{name}>" for name in names},
            }
        if not dataclasses.is_dataclass(event) or isinstance(event, type):
            return {"event": "Omitted"}
        return {"event": type(event).__name__, **self._fields(event)}

    def _fields(self, event: object) -> dict[str, object]:
        if not dataclasses.is_dataclass(event) or isinstance(event, type):
            return {}
        return {
            field.name: self._value(field.name, getattr(event, field.name))
            for field in dataclasses.fields(event)
        }

    def _value(self, key: str, value: object) -> object:
        if isinstance(value, enum.Enum):
            return value.value
        if value is None or type(value) in {bool, int, float}:
            return value
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._fields(value)
        if isinstance(value, (tuple, list)):
            return [self._value(key, item) for item in value]
        if isinstance(value, frozenset):
            return sorted((self._value(key, item) for item in value), key=str)
        if isinstance(value, str):
            return self._text(key, value)
        return "<redacted>"

    def _text(self, key: str, value: str) -> str:
        if not value:
            return ""
        if key in _REFERENCES:
            category = _REFERENCES[key]
            known = self.references.setdefault(category, {})
            return known.setdefault(value, f"{category}_{len(known) + 1}")
        if key == "secret_name":
            return f"<{_field_name(value, 1)}>"
        if key in {"ending", "reason"}:
            from computeruse.loop import Ending
            from computeruse.replay import Reason

            if value in {*Ending, *Reason}:
                return value
        if key == "role":
            from computeruse.diagnostics import ROLES

            return value if value in ROLES else "other"
        if value in _CATEGORIES.get("expected" if key == "tag" else key, ()):
            return value
        return "<redacted>"


def safe_record(event: object) -> dict[str, object]:
    """Project one event without copying free text or arbitrary mappings."""
    return Projection().record(event)


def write_event(stream: TextIO, event: object) -> None:
    """Write and flush one projected event."""
    EvidenceWriter(stream).record(event)


@contextlib.contextmanager
def open_evidence(path: Path, mode: Literal["x", "w", "a"] = "x") -> Iterator[TextIO]:
    """Open required evidence and propagate storage failures through UI callbacks."""
    try:
        stream = path.open(mode, encoding="utf-8")
    except OSError as error:
        raise EvidenceError("evidence destination unavailable") from error
    try:
        yield stream
    finally:
        try:
            stream.close()
        except OSError as error:
            raise EvidenceError("evidence destination unavailable") from error


def save_event(
    path: Path,
    event: object,
    *,
    append: bool = False,
    projection: Projection | None = None,
) -> None:
    """Save a projected event and distinguish required storage from UI failures."""
    with open_evidence(path, "a" if append else "w") as stream:
        writer = EvidenceWriter(stream)
        if projection is not None:
            writer.projection = projection
        writer.record(event)


class EvidenceWriter:
    """Use one projection and serialization path for an entire evidence stream."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.projection = Projection()

    def record(self, event: object) -> None:
        """Append one safe event and propagate write failures to the caller."""
        try:
            self.stream.write(
                json.dumps(self.projection.record(event), sort_keys=True) + "\n"
            )
            self.stream.flush()
        except OSError as error:
            raise EvidenceError("evidence destination unavailable") from error


_WRITERS: ContextVar[tuple[EvidenceWriter, ...]] = ContextVar(
    "evidence_writers", default=()
)
_PUBLIC: ContextVar[TextIO | None] = ContextVar("evidence_public", default=None)
_ERROR: ContextVar[TextIO | None] = ContextVar("evidence_error", default=None)


def emit(event: object) -> None:
    """Send typed evidence to the current command's saved destinations."""
    for writer in _WRITERS.get():
        writer.record(event)
    if isinstance(event, Problem) and (stream := _ERROR.get()) is not None:
        write_event(stream, event)


def fields(source: FieldSource, names: Iterable[str]) -> None:
    """Log field placeholders without accepting any field values."""
    emit(Fields(source, tuple(names)))


def public_help(text: str) -> None:
    """Write parser-generated help without reopening arbitrary output capture."""
    if stream := _PUBLIC.get():
        stream.write(text)
        stream.flush()
    else:
        print(text, end="")


class _HelpStream(Protocol):
    def write(self, text: str, /) -> object: ...


class EvidenceParser(argparse.ArgumentParser):
    """Keep fixed help visible while rejecting argument echoes from saved output."""

    def print_help(self, file: _HelpStream | None = None) -> None:
        """Send generated help to the public output channel."""
        if file is None:
            public_help(self.format_help())
        else:
            file.write(self.format_help())


@contextlib.contextmanager
def _console(stack: contextlib.ExitStack, stream: TextIO) -> Iterator[TextIO | None]:
    if stream.isatty():
        yield None
        return
    sink = stream
    try:
        descriptor = stream.fileno()
    except (AttributeError, io.UnsupportedOperation):
        yield sink
        return
    stream.flush()
    saved = os.dup(descriptor)
    sink = stack.enter_context(os.fdopen(saved, "w", encoding="utf-8", buffering=1))
    with Path(os.devnull).open("w") as hidden:
        os.dup2(hidden.fileno(), descriptor)
    try:
        yield sink
    finally:
        stream.flush()
        os.dup2(saved, descriptor)


def _log_path() -> Path | None:
    matches = []
    for index, arg in enumerate(sys.argv[1:], 1):
        if arg == "--log":
            if index + 1 >= len(sys.argv):
                raise ValueError("missing log destination")
            matches.append(sys.argv[index + 1])
        elif arg.startswith("--log="):
            matches.append(arg.removeprefix("--log="))
    if len(matches) > 1:
        raise ValueError("duplicate log destination")
    return Path(matches[0]) if matches else None


def run_logged(
    operation: Callable[[], int | None],
    *,
    command: str,
    log_path: Path | None = None,
) -> int:
    """Run with safe redirected output and optional exclusive command evidence.

    Live terminals retain full diagnostics. Redirected Python and native output
    is discarded; only explicit events reach the saved destination. A failed
    evidence destination stops the command and never falls back to raw output.
    """
    try:
        return _run_logged(operation, command, log_path)
    except (OSError, KeyboardInterrupt) as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 2
        with contextlib.suppress(OSError):
            write_event(sys.stdout, CommandEnded(code))
        return code


def _run_logged(
    operation: Callable[[], int | None], command: str, log_path: Path | None
) -> int:
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(_console(stack, sys.stdout))
        err = stack.enter_context(_console(stack, sys.stderr))
        hidden = stack.enter_context(Path(os.devnull).open("w"))
        if out is not None:
            stack.enter_context(contextlib.redirect_stdout(hidden))
        if err is not None:
            stack.enter_context(contextlib.redirect_stderr(hidden))
        public_token = _PUBLIC.set(out)
        stack.callback(_PUBLIC.reset, public_token)
        error_token = _ERROR.set(err)
        stack.callback(_ERROR.reset, error_token)
        return _command(operation, command, log_path, out, err)


def _command(
    operation: Callable[[], int | None],
    command: str,
    log_path: Path | None,
    out: TextIO | None,
    err: TextIO | None,
) -> int:
    storage = contextlib.ExitStack()
    saved_path = None
    token = _WRITERS.set(())
    try:
        path = log_path or _log_path()
        saved = None
        if path is not None:
            saved = EvidenceWriter(storage.enter_context(open_evidence(path)))
            saved_path = path
        public = EvidenceWriter(out) if out is not None else None
        _WRITERS.set(tuple(writer for writer in (saved, public) if writer is not None))
        emit(CommandStarted(CommandKind(command)))
        code = _execute(operation)
        if saved is not None:
            saved.record(CommandEnded(code))
        storage.close()
        if public is not None:
            public.record(CommandEnded(code))
        if code and err is not None:
            write_event(err, CommandEnded(code))
    except (EvidenceError, ValueError, KeyboardInterrupt) as error:
        with contextlib.suppress(OSError):
            storage.close()
        cancelled = isinstance(error, KeyboardInterrupt)
        code = 130 if cancelled else 2
        problem = ProblemCode.CANCELLED if cancelled else ProblemCode.LOG_UNAVAILABLE
        _log_failure(saved_path, out, err, problem, code)
    finally:
        try:
            storage.close()
        finally:
            _WRITERS.reset(token)
    return code


def _log_failure(
    path: Path | None,
    out: TextIO | None,
    err: TextIO | None,
    problem: ProblemCode,
    code: int,
) -> None:
    if path is not None:
        with contextlib.suppress(OSError), open_evidence(path, "a") as stream:
            writer = EvidenceWriter(stream)
            writer.record(Problem(problem))
            writer.record(CommandEnded(code))
    if out is not None:
        with contextlib.suppress(OSError):
            write_event(out, Problem(problem))
            write_event(out, CommandEnded(code))
    with contextlib.suppress(OSError):
        write_event(err or sys.stderr, Problem(problem))


def _execute(operation: Callable[[], int | None]) -> int:
    try:
        return operation() or 0
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 2
        if code:
            emit(Problem(ProblemCode.INVALID_ARGUMENTS))
        return code
    except KeyboardInterrupt:
        emit(Problem(ProblemCode.CANCELLED))
        return 130
    except EvidenceError:
        raise
    except Exception:
        traceback.print_exc()
        emit(Problem(ProblemCode.UNEXPECTED_ERROR))
        return 2
