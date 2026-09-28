"""Commands for reviewing a discovery artifact and replaying it without a model."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
from urllib.parse import urlsplit

from computeruse.capability import (
    ActionNode,
    Capability,
    CapabilityError,
    Condition,
    ResultNode,
    Review,
    accepts_inputs,
    check_profile,
    dumps,
    load_capability,
    validate,
)
from computeruse.evidence import (
    EvidenceError,
    EvidenceWriter,
    FieldSource,
    Problem,
    ProblemCode,
    emit,
    fields,
    open_evidence,
)
from computeruse.journal import JsonlJournal, MemoryJournal
from computeruse.profile import ProfileError, load_profile
from computeruse.replay import (
    REASON_TEXT,
    Actor,
    Delivery,
    ReplayEvent,
    ReplayResult,
    Status,
    replay,
)
from computeruse.surface import SurfaceError

if TYPE_CHECKING:
    from computeruse.control import Channel
    from computeruse.profile import Profile


def parameters(items: list[str]) -> dict[str, str]:
    """Parse explicitly named invocation values without echoing errors."""
    values: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.isidentifier() or name in values or not value:
            raise ValueError("inputs must be distinct NAME=VALUE pairs")
        values[name] = value
    return values


def save(path: Path, capability: Capability) -> None:
    """Write a validated artifact once, refusing to overwrite an earlier one."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(dumps(capability))


def approved(capability: Capability) -> Capability:
    """Return ``capability`` marked reviewed, which replay accepts."""
    reviewed = dataclasses.replace(
        capability,
        provenance=dataclasses.replace(capability.provenance, review=Review.REVIEWED),
    )
    if validate(reviewed):
        raise ValueError("invalid reviewed capability")
    return reviewed


def add_commands(commands: argparse._SubParsersAction) -> None:
    """Register commands that do not need a discovery model or a goal."""
    review = commands.add_parser("review", help="Inspect and approve a saved draft")
    review.add_argument("--capability", type=Path, required=True)
    review.add_argument("--approve", action="store_true")
    review.add_argument("--output", type=Path)
    review.add_argument(
        "--log", type=Path, help="Save structured command evidence without values"
    )
    review.add_argument(
        "--branches",
        type=Path,
        help="Add an explicit outcome or recovery plan before review",
    )
    runner = commands.add_parser("replay", help="Replay a capability without the model")
    runner.add_argument("--capability", type=Path, required=True)
    runner.add_argument("--profile", type=Path, required=True)
    runner.add_argument("--website", required=True)
    runner.add_argument("--input", action="append", default=[])
    runner.add_argument("--headed", action="store_true")
    runner.add_argument("--journal", type=Path)
    runner.add_argument(
        "--log", type=Path, help="Save structured command evidence without values"
    )
    runner.add_argument(
        "--accept-draft",
        action="store_true",
        help="Run an unreviewed draft for testing",
    )


def run(args: argparse.Namespace) -> int:
    """Validate all inputs before opening a session or creating an output."""
    try:
        capability = load_capability(args.capability)
        if args.command == "review":
            return _review(args, capability)
        return _replay(args, capability)
    except EvidenceError:
        raise
    except (CapabilityError, ProfileError, ValueError, OSError, SurfaceError):
        emit(Problem(ProblemCode.INVALID_CAPABILITY))
        print(
            "error: capability, policy, input, or session could not be used",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        emit(Problem(ProblemCode.CANCELLED))
        print("Replay cancelled; the session has been closed.", file=sys.stderr)
        return 130


def _review(args: argparse.Namespace, capability: Capability) -> int:
    if args.branches:
        from computeruse.branching import extend, load_plan

        capability = extend(capability, load_plan(args.branches))
    fields(FieldSource.INPUT, (field.name for field in capability.inputs))
    fields(FieldSource.OUTPUT, (field.name for field in capability.outputs))
    print(
        f"{capability.label}: schema {capability.schema_version}, "
        f"{capability.provenance.review.value}"
    )
    print(
        "Inputs: " + ", ".join(f"{f.name} ({f.type.value})" for f in capability.inputs)
    )
    print("Outputs: " + ", ".join(f.name for f in capability.outputs))
    for node in capability.nodes:
        print(f"  {node.node_id}: {type(node).TAG}")
        introduced = getattr(node, "introduced", "")
        if introduced:
            # Discovery never took this step; a declared binding added it.
            print(
                f"    added by the recorder, not performed in discovery "
                f"({introduced}): accept it only if that input fills this field"
            )
    print(dumps(capability))
    print("Inspect the targets, checks, effects and transitions before approval.")
    if args.approve:
        if args.output is None:
            raise ValueError("approval needs a new output path")
        save(args.output, approved(capability))
        print("Saved the approved copy. Risky actions still ask on every replay.")
    elif args.branches and args.output:
        save(args.output, capability)
        print("Saved the extended draft for review.")
    return 0


class ReplayWriter:
    """Write closed replay events, never input or output values."""

    def __init__(self, stream: TextIO | None) -> None:
        self.writer = EvidenceWriter(stream) if stream is not None else None
        self.count = 0

    def record(self, event: ReplayEvent) -> None:
        """Append one sanitized event."""
        self.count += 1
        if self.writer is not None:
            self.writer.record(event)
        emit(event)


def _replay(args: argparse.Namespace, capability: Capability) -> int:
    from computeruse.cli import _browser_run, _check_entry_point, _describe

    profile = load_profile(args.profile)
    path = _check_entry_point(args.website, profile)
    inputs = parameters(args.input)
    fields(FieldSource.INPUT, inputs)
    fields(FieldSource.SECRET, profile.secrets)
    if check_profile(capability, profile) or not accepts_inputs(capability, inputs):
        raise ValueError("capability and invocation conflict with the profile")
    if capability.provenance.review is Review.DRAFT and not args.accept_draft:
        raise ValueError("review the draft before replay")
    print(_describe(capability.label, urlsplit(args.website).netloc, path, profile))
    with contextlib.ExitStack() as stack:
        stream = (
            stack.enter_context(open_evidence(args.journal)) if args.journal else None
        )
        journal = MemoryJournal()
        human_journal = journal
        if args.journal:
            human_stream = stack.enter_context(
                open_evidence(args.journal.with_suffix(".human.jsonl"))
            )
            human_journal = JsonlJournal(human_stream)
        surface, control, channels = _browser_run(args, profile, stack)
        result = replay(
            capability,
            inputs,
            profile=profile,
            surface=surface,
            control=control,
            journal=human_journal,
            log=ReplayWriter(stream),
            clock=time.monotonic,
            sleep=time.sleep,
            accept_draft=args.accept_draft,
        )
        print(
            f"replay {result.status.value}: {result.reason.value}, "
            f"{result.steps} steps",
            flush=True,
        )
        lines = report(capability, result)
        fields(FieldSource.OUTPUT, result.outputs)
        for line in lines:
            print(f"  {line}", flush=True)
        if args.headed:
            # The window stays open on the result until the person closes it.
            _show_result(channels, profile, capability, lines)
    return 0 if result.status in {Status.SUCCEEDED, Status.OUTCOME} else 3


def _show_result(
    channels: Sequence[Channel],
    profile: Profile,
    capability: Capability | None,
    lines: list[str],
    *,
    discovery: bool = False,
) -> None:
    """Show a finished run's result in the control window until it closes.

    A replay shows its report; a discovery shows why it saved no capability.
    """
    from computeruse.browser import TargetMarker
    from computeruse.budget import Budget
    from computeruse.control import Control
    from computeruse.escalation import Ask, InterventionRequest, Mode, Trigger

    people = [c for c in channels if not isinstance(c, TargetMarker)]
    if not any(channel.listening() for channel in people):
        return
    mode = Mode.DISCOVERY if discovery else Mode.REPLAY
    control = Control(mode=mode, clock=time.monotonic, channels=people)
    if not control.begin(
        profile=profile,
        budget=Budget(profile.budgets, time.monotonic),
        journal=MemoryJournal(),
        context="showing the replay result",
    ):
        return
    control.intervene(
        InterventionRequest(
            trigger=Trigger.DISCOVERY_RESULT if discovery else Trigger.REPLAY_RESULT,
            goal=capability.label if capability is not None else "",
            profile_id=profile.profile_id,
            step=0,
            route="",
            reason="\n".join(lines),
            timeout_s=profile.escalation.handoff_timeout_s,
            ask=Ask.PERSON,
            mode=mode,
        )
    )
    control.finish("completed")


HEADLINES = {
    Status.SUCCEEDED: "Replay succeeded.",
    Status.OUTCOME: "Replay ended with a business outcome the capability declares.",
    Status.STOPPED: "Replay stopped before the first change, as asked.",
    Status.RECOVERY_EXHAUSTED: "Replay stopped: its bounded recovery ran out.",
    Status.NEEDS_HELP: "Replay stopped where a person must decide what happens next.",
    Status.HELP_TIMED_OUT: "Replay stopped: nobody answered in time.",
    Status.FAILED: "Replay failed.",
    Status.TERMINATED: "Replay ended: a person terminated it.",
}
"""The first plain line of a replay's report, for each way it can end."""

REASONS = REASON_TEXT


def report(capability: Capability, result: ReplayResult) -> list[str]:
    """Describe a replay result for the terminal, log, and control window.

    It names the step by what it does, the checks by their controls, and the
    page by its structure. Output values appear, because the caller asked for
    them; input values never do.
    """
    from computeruse.describe import describe_condition, steps

    lines = [HEADLINES.get(result.status, f"Replay {result.status.value}.")]
    if result.outcome:
        lines.append(f"Outcome: {result.outcome.replace('_', ' ')}")
    lines.append(
        f"Why: {REASONS.get(result.reason, result.reason.value.replace('_', ' '))}"
    )
    listed = steps(capability)
    place = {node: index for index, (node, _) in enumerate(listed, 1)}
    if result.status not in {Status.SUCCEEDED, Status.OUTCOME} and result.node:
        if result.node in place:
            index = place[result.node]
            lines.append(
                f"Stopped at: step {index} of {len(listed)}, {listed[index - 1][1]}"
            )
        else:
            lines.append(f"Stopped at: {result.node}")
    diagnostic = result.diagnostic
    if diagnostic is not None:
        conditions = _conditions(capability, diagnostic.node, len(diagnostic.checks))
        for condition, check in zip(conditions, diagnostic.checks, strict=False):
            held = {True: "held", False: "did not hold", None: "not checked"}
            lines.append(
                f"Expected: {describe_condition(condition, capability)}"
                f" ({held[check.held]})"
            )
        state = diagnostic.observed
        if state is not None:
            lines.append(
                f"Page seen: {state.status.value} reading, {state.shown} of "
                f"{state.total} controls visible, "
                + ("a dialog open" if state.dialog else "no dialog")
            )
    done = [a for a in result.attempts if a.delivery is Delivery.COMPLETED]
    uncertain = [a for a in result.attempts if a.delivery is Delivery.UNCERTAIN]
    by_person = sum(1 for a in done if a.actor is Actor.PERSON)
    lines.append(
        f"Steps performed: {len(done)}"
        + (f", {by_person} of them by a person" if by_person else "")
    )
    if uncertain:
        lines.append(
            f"Uncertain changes: {len(uncertain)}, never sent again automatically"
        )
    for entry in result.history:
        where = place.get(entry.node)
        lines.append(
            f"A person was asked{f' at step {where}' if where else ''}: "
            f"{REASONS.get(entry.reason, entry.reason.value)}; "
            f"answer: {entry.answer.value.replace('_', ' ')}"
        )
    lines.extend(f"{name}: {value}" for name, value in result.outputs.items())
    return lines


def _conditions(
    capability: Capability, node_id: str, count: int
) -> tuple[Condition, ...]:
    """Return the checks a failing node evaluated, when their number matches."""
    for node in capability.nodes:
        if node.node_id != node_id:
            continue
        if isinstance(node, ActionNode):
            for group in (node.verify, node.requires):
                if len(group) == count:
                    return group
        if isinstance(node, ResultNode) and len(node.checks) == count:
            return node.checks
    return ()
