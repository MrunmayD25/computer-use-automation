"""Two commands: report what a run would be allowed to do, and run it.

The default command validates a goal, a target, and a profile together, then
prints the resolved policy. It opens and writes nothing. The operator must use
a separate command to start a browser, which keeps policy inspection free of
application effects.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
from urllib.parse import urlsplit

from computeruse.evidence import (
    EvidenceError,
    EvidenceParser,
    FieldSource,
    Problem,
    ProblemCode,
    emit,
    fields,
    open_evidence,
    run_logged,
)
from computeruse.journal import Journal, JsonlJournal, MemoryJournal
from computeruse.profile import (
    ActionKind,
    ObservationMode,
    Profile,
    ProfileError,
    Risk,
    load_profile,
)
from computeruse.urls import parse_http_url

if TYPE_CHECKING:
    from computeruse.capability import Capability
    from computeruse.confirm import Decision
    from computeruse.contract import Contract
    from computeruse.control import Channel, Control
    from computeruse.loop import RunResult
    from computeruse.manual import ManualSegment
    from computeruse.model import LunaDecider
    from computeruse.recorder import Candidate, Recording
    from computeruse.recording import DiscoveryTrace
    from computeruse.surface import Surface

logger = logging.getLogger(__name__)

INPUT_ERROR = 2
RUN_INCOMPLETE = 3


@dataclasses.dataclass(frozen=True)
class PolicyAccepted:
    """Policy categories without origins, record paths, or arbitrary labels."""

    safe_actions: tuple[ActionKind, ...]
    risky_actions: tuple[ActionKind, ...]
    observation_modes: tuple[ObservationMode, ...]
    alternate_observations: int
    max_steps: int
    max_wall_clock_s: float


def _goal(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("goal must not be blank")
    return value


def _website(value: str) -> str:
    try:
        parse_http_url(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "website must be an HTTP(S) URL without embedded credentials"
        ) from error
    return value


def _profile_path(value: str) -> Path:
    path = Path(value).expanduser()
    try:
        readable = path.is_file()
        if readable:
            with path.open("rb"):
                pass
    except OSError:
        readable = False
    if not readable:
        raise argparse.ArgumentTypeError(
            "profile must point to a readable local file"
        ) from None
    return path


def _check_entry_point(website: str, profile: Profile) -> str:
    """Return the entry path after confirming the profile permits it.

    Raises
    ------
    ProfileError
        If the target sits outside the profile's origin or its allowed routes.
    """
    scope = profile.scope
    url = urlsplit(website)
    if not scope.contains(website):
        raise ProfileError("website origin is not the profile's origin")
    path = url.path or "/"
    if scope.route(website) is None:
        raise ProfileError("profile does not permit the entry point")
    return path


def _describe(goal: str, domain: str, entry_path: str, profile: Profile) -> str:
    by_risk = {
        risk: sorted(
            kind.value for kind, declared in profile.actions.items() if declared is risk
        )
        for risk in Risk
    }
    budgets = profile.budgets
    budget_summary = (
        f"{budgets.max_steps} steps, {budgets.max_wall_clock_s}s, "
        f"{budgets.max_retries_per_step} retries/step, "
        f"{budgets.max_navigations} navigations"
    )
    perception = profile.perception
    modes = ", ".join(mode.value for mode in perception.allowed_modes)
    safe_actions = ", ".join(by_risk[Risk.SAFE]) or "none"
    risky_actions = ", ".join(by_risk[Risk.RISKY]) or "none"
    lines = [
        f"goal:    {goal}",
        f"domain:  {domain}",
        f"profile: {profile.profile_id} (schema version {profile.version})",
        f"  environment:    {profile.environment.value}",
        f"  entry point:    {entry_path}",
        f"  safe actions:   {safe_actions}",
        f"  risky actions:  {risky_actions} (always escalate)",
        f"  effect rules:   {_effect_rules(profile)}",
        f"  observe with:   {modes}",
        (
            f"  second look:    {perception.max_alternate_observations_per_step} "
            "per unresolved action"
        ),
        f"  name a record:  {_records(profile)}",
        f"  budgets:        {budget_summary}",
    ]
    scope = profile.scope
    peers = ", ".join(rule.origin for rule in scope.origins) or "none"
    lines[5:5] = [
        f"  allow routes:   {', '.join(scope.allow_routes)}",
        f"  deny routes:    {', '.join(scope.deny_routes) or 'none'}",
        f"  peer origins:   {peers}",
        f"  new windows:    {'allowed' if scope.allow_new_windows else 'denied'}",
        f"  downloads:      {'allowed' if scope.allow_downloads else 'denied'}",
    ]
    for name, variable in sorted(profile.secrets.items()):
        lines.append(f"  secret:         {name} from ${variable} (value not read)")
    lines.append(
        f"  escalation:     handoff timeout {profile.escalation.handoff_timeout_s}s, "
        f"then {profile.escalation.on_timeout}"
    )
    return "\n".join(lines)


def _effect_rules(profile: Profile) -> str:
    rules = [
        f"{kind.value}({effect}) {limit.value}"
        for kind, listed in profile.effects.items()
        for effect, limit in listed.items()
    ]
    return ", ".join(rules) or "none"


def _records(profile: Profile) -> str:
    records = profile.records
    if not records.actions or not records.routes:
        return "no action is required to"
    kinds = ", ".join(kind.value for kind in records.actions)
    return f"{kinds} on {', '.join(records.routes)}"


def _inputs(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--log", type=Path, help="Save structured command evidence without values"
    )
    parser.add_argument(
        "--goal", required=required, type=_goal, help="Natural-language goal"
    )
    parser.add_argument(
        "--website",
        required=required,
        type=_website,
        help="Target HTTP(S) entry point",
    )
    parser.add_argument(
        "--profile",
        required=required,
        type=_profile_path,
        help="Path to the operator's YAML policy profile",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable diagnostic logging"
    )


def _parser() -> argparse.ArgumentParser:
    parser = EvidenceParser(
        prog="computeruse",
        description=(
            "Validate a goal, target, and policy profile, and report them. "
            "No action is taken against the target application unless the "
            "discover or replay command is used."
        ),
        allow_abbrev=False,
    )
    _inputs(parser, required=False)
    commands = parser.add_subparsers(dest="command")
    discover = commands.add_parser(
        "discover",
        help="Drive a real browser toward the goal under the profile",
        allow_abbrev=False,
    )
    _inputs(discover, required=True)
    discover.add_argument(
        "--journal",
        type=Path,
        help="Write one JSON event per line to this file",
    )
    discover.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Show the browser with a control window beside it to start, stop, "
            "take over, and resume the run"
        ),
    )
    discover.add_argument(
        "--visual",
        action="store_true",
        help=(
            "Use screenshots and mouse and keyboard input "
            "without model-facing selectors"
        ),
    )
    discover.add_argument("--model", help="Responses model to use for discovery")
    discover.add_argument(
        "--reviewed-visual-template",
        action="append",
        default=[],
        metavar="STEP=PNG",
        help="Use an operator-reviewed nonsensitive control crop",
    )
    discover.add_argument(
        "--reviewed-visual-result",
        action="append",
        default=[],
        metavar="STEP=PNG",
        help="Use an operator-reviewed nonsensitive result crop",
    )
    discover.add_argument("--save-capability", type=Path)
    discover.add_argument(
        "--no-outcome-checks",
        action="store_true",
        help=(
            "Skip the profile's outcome cases, so saving runs no extra model "
            "discovery; a replay then treats such an answer as unknown"
        ),
    )
    discover.add_argument(
        "--save-approved",
        type=Path,
        help=(
            "After saving the draft, ask for review in the control window, "
            "and write an approved copy here if a person approves it"
        ),
    )
    discover.add_argument("--capability-id", default="discovered_workflow")
    discover.add_argument("--input", action="append", default=[])
    discover.add_argument(
        "--contract", type=Path, help="Typed input and output declarations for export"
    )
    discover.add_argument(
        "--confirm-input",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "A second test record that confirms the website text a capability "
            "saves; replaces the profile's confirmation record for this run"
        ),
    )
    from computeruse.capability_cli import add_commands

    add_commands(commands)
    return parser


def _visual_profile(profile: Profile) -> Profile:
    if ObservationMode.VISUAL not in profile.perception.allowed_modes:
        raise ProfileError("the profile does not permit visual observation")
    return dataclasses.replace(
        profile,
        perception=dataclasses.replace(
            profile.perception, allowed_modes=(ObservationMode.VISUAL,)
        ),
    )


def main() -> int:
    """Validate the inputs, and either report them or run a discovery pass.

    Returns
    -------
    int
        0 when the inputs are reported or the goal is met, 2 for invalid
        input, 3 when a run stopped without meeting the goal.
    """
    command = sys.argv[1] if len(sys.argv) > 1 else "policy"
    if command not in {"discover", "review", "replay"}:
        command = "policy"
    return run_logged(_main, command=command)


def _main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.command in {"review", "replay"}:
        from computeruse.capability_cli import run

        return run(args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if args.command is None and (
        args.goal is None or args.website is None or args.profile is None
    ):
        parser.error(
            "the following arguments are required: --goal, --website, --profile"
        )
    try:
        profile = load_profile(args.profile)
        entry_path = _check_entry_point(args.website, profile)
        domain = urlsplit(args.website).netloc
        if args.command == "discover" and args.visual:
            profile = _visual_profile(profile)
    except ProfileError as error:
        emit(Problem(ProblemCode.INVALID_POLICY))
        print(f"error: {error}", file=sys.stderr)
        return INPUT_ERROR

    logger.debug("Profile accepted; no secret values were resolved.")
    emit(
        PolicyAccepted(
            tuple(kind for kind, risk in profile.actions.items() if risk is Risk.SAFE),
            tuple(kind for kind, risk in profile.actions.items() if risk is Risk.RISKY),
            tuple(profile.perception.allowed_modes),
            profile.perception.max_alternate_observations_per_step,
            profile.budgets.max_steps,
            profile.budgets.max_wall_clock_s,
        )
    )
    fields(FieldSource.SECRET, profile.secrets)
    print(_describe(args.goal, domain, entry_path, profile))
    if args.command is None:
        print("\nNo action was taken. Add the discover command to run the goal.")
        return 0
    try:
        return _discover(args, profile)
    except KeyboardInterrupt:
        emit(Problem(ProblemCode.CANCELLED))
        print("\nRun cancelled; the session has been closed.", file=sys.stderr)
        return 130


def _discover(args: argparse.Namespace, profile: Profile) -> int:
    """Open the authorized surface and report how discovery stopped."""
    import httpx

    from computeruse import loop, model
    from computeruse.capability_cli import parameters
    from computeruse.recording import DiscoveryTrace
    from computeruse.surface import SurfaceError

    trace = DiscoveryTrace() if getattr(args, "save_capability", None) else None
    try:
        inputs = parameters(getattr(args, "input", []))
    except ValueError:
        emit(Problem(ProblemCode.INVALID_INPUTS))
        print("error: inputs must be distinct NAME=VALUE pairs", file=sys.stderr)
        return INPUT_ERROR
    try:
        from computeruse.contract import load_contract

        args.export_contract = (
            load_contract(args.contract) if getattr(args, "contract", None) else None
        )
        args.visual_templates = _visual_files(
            getattr(args, "reviewed_visual_template", [])
        )
        args.visual_results = _visual_files(getattr(args, "reviewed_visual_result", []))
    except (ValueError, OSError):
        emit(Problem(ProblemCode.INVALID_DECLARATIONS))
        print("error: export declarations are invalid or unreadable", file=sys.stderr)
        return INPUT_ERROR
    # Without --input, the model fills in the contract's inputs from the goal.
    if inputs and not _fits_contract(args.export_contract, inputs):
        emit(Problem(ProblemCode.INVALID_INPUTS))
        print("error: inputs do not match the export contract", file=sys.stderr)
        return INPUT_ERROR
    api_key = os.environ.get(model.API_KEY_VARIABLE)
    if not api_key:
        emit(Problem(ProblemCode.MISSING_CREDENTIAL))
        print(
            f"error: {model.API_KEY_VARIABLE} is not set in the environment",
            file=sys.stderr,
        )
        return INPUT_ERROR

    print("\nOpening the authorized session.")
    with contextlib.ExitStack() as stack:
        try:
            journal = _journal(stack, args.journal)
        except OSError:
            emit(Problem(ProblemCode.JOURNAL_UNAVAILABLE))
            print("error: the journal file could not be opened", file=sys.stderr)
            return INPUT_ERROR
        client = stack.enter_context(httpx.Client())
        decider = model.LunaDecider(
            api_key=api_key, client=client, model=args.model or model.MODEL
        )
        control: Control | None = None
        channels: list[Channel] = []

        def prepare(people: Sequence[Channel]) -> None:
            nonlocal inputs
            inputs = _fill_inputs(args, profile, decider, people, inputs)
            fields(FieldSource.INPUT, inputs)

        try:
            surface, control, channels = _browser_run(
                args, profile, stack, prepare=prepare
            )
            result = loop.discover(
                args.goal,
                profile,
                surface=surface,
                decider=decider,
                control=control,
                journal=journal,
                clock=time.monotonic,
                trace=trace,
                inputs=inputs,
                outputs=_declared_outputs(args),
            )
        except _UnfilledError as unfilled:
            emit(Problem(ProblemCode.INVALID_INPUTS))
            print(
                f"error: no value for the input {unfilled.name}; "
                "name it in the goal or answer in the control window",
                file=sys.stderr,
            )
            return INPUT_ERROR
        except EvidenceError:
            raise
        except (SurfaceError, OSError):
            print("error: the session could not continue", file=sys.stderr)
            return INPUT_ERROR
        _report_result(result, declared_outputs=_declared_outputs(args))
        if isinstance(journal, MemoryJournal):
            print(f"  {len(journal.events)} events recorded, none written to disk")
        _report_segments(result.segments)
        if trace is not None:
            # Saving may ask a person, so it runs while their window is open.
            return _save_discovery(
                args, profile, trace, result, inputs, control, channels
            )
    return 0 if result.ending is loop.Ending.COMPLETED else RUN_INCOMPLETE


def _fits_contract(contract: Contract | None, inputs: Mapping[str, str]) -> bool:
    """Report whether explicit inputs name and type exactly the contract's inputs."""
    if contract is None:
        return True
    return set(inputs) == {field.name for field in contract.inputs} and all(
        field.accepts(inputs[field.name]) for field in contract.inputs
    )


class _UnfilledError(Exception):
    """A required input had no value from the goal or from a person."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def _fill_inputs(
    args: argparse.Namespace,
    profile: Profile,
    decider: LunaDecider,
    channels: Sequence[Channel],
    given: dict[str, str],
) -> dict[str, str]:
    """Fill the contract's inputs from the goal before the run's control starts.

    Explicit ``--input`` values, or a run without a contract, are returned
    unchanged. A person is asked for a missing value on the run's channels. A
    value they supply is added to ``args.goal``, so the run and every later
    step work from a goal that names it. Prints input names, never a value.
    """
    from computeruse import browser
    from computeruse.inputs import asking, fill

    if given or args.export_contract is None:
        return given

    people = [c for c in channels if not isinstance(c, browser.TargetMarker)]
    ask = (
        asking(people, profile, args.goal, time.monotonic)
        if any(channel.listening() for channel in people)
        else None
    )
    filled = fill(
        args.goal, args.export_contract.inputs, propose=decider.fill_inputs, ask=ask
    )
    if filled.missing:
        raise _UnfilledError(filled.missing)
    if filled.goal != args.goal:
        print("  inputs the goal did not spell out were added to it")
    args.goal = filled.goal
    print(f"  inputs: {', '.join(filled.inputs)}", flush=True)
    return filled.inputs


def _visual_files(items: list[str]) -> dict[int, bytes]:
    from computeruse.capability import StoragePermit, _template_ok, template

    result: dict[int, bytes] = {}
    for item in items:
        step, separator, name = item.partition("=")
        if (
            not separator
            or not step.isdecimal()
            or int(step) < 1
            or int(step) in result
        ):
            raise ValueError("visual declarations need distinct positive steps")
        path = Path(name)
        if path.stat().st_size > 2_000_000:
            raise ValueError("visual template is too large")
        data = path.read_bytes()
        if not _template_ok(
            template("reviewed", data, StoragePermit.OPERATOR_APPROVED)
        ):
            raise ValueError("visual template is invalid")
        result[int(step)] = data
    return result


def _declared_outputs(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the output names the export contract declares, if there is one."""
    contract = getattr(args, "export_contract", None)
    return tuple(field.name for field in contract.outputs) if contract else ()


def _save_discovery(
    args: argparse.Namespace,
    profile: Profile,
    trace: DiscoveryTrace,
    result: RunResult,
    inputs: dict[str, str],
    control: Control | None,
    channels: Sequence[Channel] = (),
) -> int:
    """Export a completed trace or report why a capability cannot be saved."""
    from computeruse.capability_cli import save
    from computeruse.recording import FactKept
    from computeruse.words import WordsError, add_words, load_words, words_path

    listed = words_path(Path(args.profile))
    excluded = (
        *_private_values(args, profile, inputs, result),
        *(entry.fact.value for entry in trace.entries if isinstance(entry, FactKept)),
    )
    try:
        add_words(listed, (), excluded=excluded)
        known = frozenset(word.text for word in load_words(listed))
    except WordsError:
        print("error: the site's word list is invalid", file=sys.stderr)
        return INPUT_ERROR

    from computeruse.reading import LocalReader

    reader = LocalReader()

    def build(words: frozenset[str], *, collect: bool = False) -> Recording:
        return trace.build(
            result,
            profile=profile,
            inputs=inputs,
            capability_id=args.capability_id,
            run=control.run if control else "discovery",
            safe_text=words,
            contract=args.export_contract,
            visual_templates=args.visual_templates,
            visual_results=args.visual_results,
            collect=collect,
            reader=reader,
            excluded=excluded,
        )

    recorded = build(known, collect=True)
    capability = recorded.capability
    findings: list[str] = []
    checking = capability is not None and _consent(args, profile, recorded, channels)
    failed = False
    if capability is not None and recorded.candidates:
        decision = _confirm_words(
            args,
            profile,
            capability,
            recorded.candidates,
            inputs,
            channels,
            comparing=checking,
        )
        failed = _compared(decision, findings)
        added = add_words(listed, decision.confirmed, excluded=excluded)
        if added:
            print(f"  confirmed {len(added)} website texts; added to {listed.name}")
        for item in decision.unconfirmed:
            print(f"    unconfirmed: {item.text!r}")
        # Rebuild with only confirmed words. A delivered step omits a check with
        # unconfirmed text, while any required unconfirmed text leaves the
        # capability incomplete.
        known = known | {word.text for word in decision.confirmed}
        recorded = build(known)
        capability = recorded.capability
        if capability is None and not decision.complete:
            print(f"  capability not saved: {decision.note or 'texts are unconfirmed'}")
    if capability is not None:
        capability = _learn_outcomes(
            args,
            profile,
            capability,
            inputs,
            channels,
            known,
            findings,
            run=checking and not failed,
        )
    if capability is None:
        lines = _report_incomplete(recorded)
        if args.headed:
            from computeruse.capability_cli import _show_result

            _show_result(channels, profile, None, lines, discovery=True)
        return RUN_INCOMPLETE
    try:
        save(args.save_capability, capability)
    except OSError:
        print("error: capability output could not be created", file=sys.stderr)
        return INPUT_ERROR
    print("  saved a draft capability; inspect it with computeruse review")
    return _review_draft(args, profile, capability, channels, findings)


def _second_record(args: argparse.Namespace, profile: Profile) -> dict[str, str]:
    """Return the second test record: from --confirm-input, or the profile's."""
    from computeruse.capability_cli import parameters

    try:
        return parameters(args.confirm_input) or dict(profile.confirmation)
    except ValueError:
        return {}


def _private_values(
    args: argparse.Namespace,
    profile: Profile,
    inputs: Mapping[str, str],
    result: RunResult,
) -> tuple[str, ...]:
    return tuple(
        filter(
            None,
            (
                *inputs.values(),
                *result.outputs.values(),
                *_second_record(args, profile).values(),
                *(
                    value
                    for case in profile.outcomes.values()
                    for value in case.values()
                ),
                *(
                    os.environ.get(variable, "")
                    for variable in profile.secrets.values()
                ),
                os.environ.get("OPENAI_API_KEY", ""),
            ),
        )
    )


def _outcome_cases(
    args: argparse.Namespace, profile: Profile
) -> Mapping[str, Mapping[str, str]]:
    """Return the profile's outcome cases, or none with --no-outcome-checks."""
    return {} if getattr(args, "no_outcome_checks", False) else profile.outcomes


def _pairs(values: Mapping[str, str]) -> str:
    return ", ".join(f"{name}={value}" for name, value in values.items())


def _consent(
    args: argparse.Namespace,
    profile: Profile,
    recorded: Recording,
    channels: Sequence[Channel],
) -> bool:
    """Ask before saving runs its checks against the application again.

    The comparison and the outcome cases open new windows and act on the
    application, so a person says whether they run. With nobody to ask, the
    checks run; the comparison never makes a change either way.
    """
    from computeruse import browser
    from computeruse.saving import consent

    checks: list[str] = []
    second = _second_record(args, profile)
    if recorded.candidates and second:
        checks.append(
            f"Comparison: repeat the draft in a new window for {_pairs(second)}, "
            f"stopping before any change, to confirm {len(recorded.candidates)} "
            "page texts."
        )
    for outcome, case in _outcome_cases(args, profile).items():
        checks.append(
            f"Outcome check: a short model discovery for {_pairs(case)}, to learn "
            f"the {outcome.replace('_', ' ')} branch."
        )
    people = [c for c in channels if not isinstance(c, browser.TargetMarker)]
    if not checks or not any(channel.listening() for channel in people):
        return True
    print("\nChecks before saving, if you allow them:")
    for line in checks:
        print(f"  {line}")
    allowed = consent(people, profile, args.goal, checks, time.monotonic)
    print("  checks allowed" if allowed else "  checks skipped")
    return allowed


def _compared(decision: Decision, findings: list[str]) -> bool:
    """Note how a comparison ended for the review; True when it failed."""
    if not decision.ended:
        return False
    confirmed = len(decision.confirmed)
    if decision.ended in {"succeeded", "stopped"}:
        findings.append(
            f"Comparison with a second member: {decision.why}; "
            f"{confirmed} texts confirmed."
        )
        return False
    findings.append(
        f"Warning: the comparison with a second member failed: {decision.why}. "
        "Approve only after checking the steps below."
    )
    return True


def _learn_outcomes(
    args: argparse.Namespace,
    profile: Profile,
    capability: Capability,
    inputs: dict[str, str],
    channels: Sequence[Channel],
    known: frozenset[str],
    findings: list[str],
    *,
    run: bool,
) -> Capability:
    """Run the profile's outcome cases when allowed, and note what they taught."""
    for outcome, case in _outcome_cases(args, profile).items():
        named = outcome.replace("_", " ")
        if not run:
            findings.append(f"Outcome check for {named}: not run.")
            continue
        learned = _learn_outcome(
            args, profile, capability, inputs, channels, known, outcome, case
        )
        said = "branch learned" if learned is not capability else "not learned"
        findings.append(f"Outcome check for {named}: {said}.")
        capability = learned
    return capability


def _report_incomplete(recorded: Recording) -> list[str]:
    """Print why no capability was saved, and return it as lines for the window.

    Each gap has its step and what it is; each validation issue names its node.
    """
    codes = ", ".join(sorted({item.gap.value for item in recorded.issues}))
    print(f"  capability incomplete: {codes}")
    lines = ["The discovery finished, but no capability was saved."]
    for item in recorded.issues:
        said = f": {item.detail}" if item.detail else ""
        line = f"step {item.step}, {item.gap.value.replace('_', ' ')}{said}"
        print(f"    {line}")
        lines.append(f"Why: {line}")
    for issue in recorded.artifact_issues:
        node = _issue_node(recorded.rejected, issue.where)
        line = f"{issue.code.value.replace('_', ' ')} at {issue.where}{node}"
        print(f"    {line}")
        lines.append(f"Why: {line}")
    lines.append("Nothing was saved. Run the discovery again to retry.")
    return lines


def _issue_node(capability: Capability | None, where: str) -> str:
    """Describe the node an issue points at by kind, role, page, and effect.

    The draft failed validation and may hold unconfirmed text, so no text
    from its controls is shown.
    """
    from computeruse.capability import ActionNode, HumanNode, StructuralTarget

    head, bracket, rest = where.partition("[")
    number, closed, _ = rest.partition("]")
    if capability is None or head != "nodes" or not bracket or not closed:
        return ""
    if not number.isdecimal() or int(number) >= len(capability.nodes):
        return ""
    node = capability.nodes[int(number)]
    if isinstance(node, HumanNode):
        return f": a person's step on {node.route}"
    if not isinstance(node, ActionNode):
        return f": a {type(node).TAG} node"
    targets = {target.target_id: target for target in capability.targets}
    target = targets.get(node.target or "")
    role = target.role or target.tag if isinstance(target, StructuralTarget) else ""
    parts = [node.kind.value.replace("_", " ")]
    if role:
        parts.append(f"a {role}")
    parts.append(f"on {node.route}")
    if node.effect:
        parts.append(f"with the effect {node.effect}")
    if node.introduced:
        parts.append(f"added by the recorder ({node.introduced})")
    return ": " + " ".join(parts)


def _review_draft(
    args: argparse.Namespace,
    profile: Profile,
    capability: Capability,
    channels: Sequence[Channel],
    findings: Sequence[str] = (),
) -> int:
    """Offer the saved draft for review, and save an approved copy on approval.

    Only with ``--save-approved``, and only when a person can answer. A
    decline, a termination, or no answer leaves the draft for a later review.
    """
    from computeruse import browser
    from computeruse.capability_cli import approved, save
    from computeruse.saving import review, summary

    path = getattr(args, "save_approved", None)
    if path is None:
        return 0
    people = [c for c in channels if not isinstance(c, browser.TargetMarker)]
    if not any(channel.listening() for channel in people):
        print("  nobody can review here; approve the draft with computeruse review")
        return 0
    print("\nReview the capability in the control window:")
    for line in [*findings, *summary(capability).splitlines()]:
        print(f"  {line}")
    if not review(people, profile, args.goal, capability, time.monotonic, findings):
        print("  not approved; the draft waits for scripts/review.sh")
        return 0
    try:
        save(path, approved(capability))
    except (OSError, ValueError):
        print("error: the approved capability could not be created", file=sys.stderr)
        return INPUT_ERROR
    print("  approved in the control window; replay can run it now")
    return 0


def _learn_outcome(
    args: argparse.Namespace,
    profile: Profile,
    capability: Capability,
    inputs: dict[str, str],
    channels: Sequence[Channel],
    known: frozenset[str],
    outcome: str,
    case: Mapping[str, str],
) -> Capability:
    """Run the profile's test case for ``outcome``, and learn what it shows.

    A second, short discovery runs in a fresh session on the same channels,
    with the case's inputs in place of the discovered ones. When the model
    finishes with that outcome and the loop verifies it, the recorded run
    becomes one branch of ``capability`` after the step both runs share.
    Anything else returns ``capability`` unchanged and says why.
    """
    import httpx

    from computeruse import browser, loop, model
    from computeruse.control import Control
    from computeruse.escalation import Mode
    from computeruse.outcomes import adopt, case_goal
    from computeruse.reading import LocalReader
    from computeruse.recording import DiscoveryTrace
    from computeruse.surface import SurfaceError
    from computeruse.words import Confirmed, Word, add_words, words_path

    written = case_goal(args.goal, inputs, case)
    if written is None:
        print(f"  no {outcome} learned: the goal does not name that input")
        return capability
    print(f"\nRunning the profile's test case for {outcome}.")
    trace = DiscoveryTrace()
    tested = {**inputs, **case}
    people = [c for c in channels if not isinstance(c, browser.TargetMarker)]
    try:
        with (
            httpx.Client() as client,
            browser.open_session(
                profile, args.website, headless=not args.headed, record=True
            ) as surface,
        ):
            decider = model.LunaDecider(
                api_key=os.environ.get(model.API_KEY_VARIABLE, ""),
                client=client,
                model=args.model or model.MODEL,
            )
            control = Control(
                mode=Mode.DISCOVERY,
                clock=time.monotonic,
                seat=surface,
                channels=people,
                purpose="outcome_check",
            )
            result = loop.discover(
                written,
                profile,
                surface=surface,
                decider=decider,
                control=control,
                journal=MemoryJournal(),
                clock=time.monotonic,
                trace=trace,
                inputs=tested,
                outputs=_declared_outputs(args),
            )
            recorded = trace.build(
                result,
                profile=profile,
                inputs=tested,
                capability_id=args.capability_id,
                run=control.run,
                safe_text=known,
                contract=args.export_contract,
                collect=True,
                reader=LocalReader(),
                excluded=_private_values(args, profile, tested, result),
            )
            adopted = adopt(
                capability,
                recorded.capability if result.outcome else None,
                outcome,
                decider,
            )
    except EvidenceError:
        raise
    except (SurfaceError, OSError):
        print(f"  no {outcome} learned: the session could not continue")
        return capability
    if adopted.capability is None:
        print(f"  no {outcome} learned: {adopted.note}")
        return capability
    listed = words_path(Path(args.profile))
    add_words(
        listed,
        (Word(text, Confirmed.OUTCOME) for text in adopted.texts),
        excluded=_private_values(args, profile, tested, result),
    )
    print(f"  learned the outcome {outcome}")
    return adopted.capability


def _confirm_words(
    args: argparse.Namespace,
    profile: Profile,
    capability: Capability,
    candidates: tuple[Candidate, ...],
    inputs: dict[str, str],
    channels: Sequence[Channel],
    *,
    comparing: bool = True,
) -> Decision:
    """Confirm the texts a draft would save: by comparison, the model, or a person.

    A second record comes from ``--confirm-input`` or the profile. Without
    one the model may confirm control labels. A person, at the control window
    or the terminal, sees only what is left, and may approve it or name a
    second record in the resume note.
    """
    import httpx

    from computeruse import browser, model
    from computeruse.confirm import compare, decide
    from computeruse.control import Control
    from computeruse.escalation import Mode
    from computeruse.saving import asking

    # The highlight belongs to the discovery's session, so it stays behind.
    people = [c for c in channels if not isinstance(c, browser.TargetMarker)]

    def compare_with(second: Mapping[str, str]) -> Decision:
        # A person may name only the input that changes; the rest stay.
        record = {**inputs, **second}
        with browser.open_session(
            profile, args.website, headless=not args.headed, record=True
        ) as surface:
            # Ask a person to perform any step the capability assigns to one.
            control = Control(
                mode=Mode.REPLAY,
                clock=time.monotonic,
                seat=surface,
                channels=people,
                purpose="comparison",
            )
            return compare(
                capability,
                candidates,
                record,
                inputs,
                profile=profile,
                surface=surface,
                clock=time.monotonic,
                sleep=surface.idle,
                control=control,
            )

    second = _second_record(args, profile) if comparing else {}
    compared = compare_with(second) if second else None
    ask = (
        asking(channels, profile, args.goal, compare_with, time.monotonic)
        if any(channel.listening() for channel in channels)
        else None
    )
    with httpx.Client() as client:
        classifier = model.LunaDecider(
            api_key=os.environ.get(model.API_KEY_VARIABLE, ""),
            client=client,
            model=args.model or model.MODEL,
        )
        return decide(
            candidates,
            compared=compared,
            classifier=None if compared is not None else classifier,
            veto=classifier,
            ask=ask,
        )


def _browser_run(
    args: argparse.Namespace,
    profile: Profile,
    stack: contextlib.ExitStack,
    prepare: Callable[[Sequence[Channel]], None] | None = None,
) -> tuple[Surface, Control, list[Channel]]:
    """Open the recorded browser session and the run's operator channels.

    The terminal channel is always attached, and reads commands only when
    standard input is a terminal. A headed run also gets the control window,
    and waits there for Start. ``prepare`` runs once the channels are open and
    before the run's control attaches to them, so it may ask a person on them.
    """
    from computeruse import browser
    from computeruse.control import Control
    from computeruse.escalation import Mode
    from computeruse.terminal import TerminalChannel

    surface = stack.enter_context(
        browser.open_session(
            profile, args.website, headless=not args.headed, record=True
        )
    )
    terminal = TerminalChannel()
    stack.callback(terminal.close)
    channels: list[Channel] = [terminal]
    if args.headed:
        channels.extend(_control_window(surface, stack))
    if prepare is not None:
        prepare(channels)
    control = Control(
        mode=Mode.REPLAY if args.command == "replay" else Mode.DISCOVERY,
        clock=time.monotonic,
        seat=surface,
        channels=channels,
        wait_for_start=args.headed,
    )
    if args.headed:
        print(
            "Press Start in the control window beside the application, "
            "or type start here."
            if terminal.listening()
            else "Press Start in the control window beside the application.",
            flush=True,
        )
    return surface, control, channels


def _control_window(surface: Surface, stack: contextlib.ExitStack) -> list[Channel]:
    """Open the control window beside the application, and the target highlight.

    The native window stays above every other window. Where it cannot open,
    as with no display for Tk, the run uses the control window inside the
    browser instead. Either way the window gets its own place beside the
    application, so the controls never cover the site or take its clicks.
    """
    from computeruse.browser import BrowserSurface, TargetMarker
    from computeruse.native import WIDTH, NativeWindow, NativeWindowError
    from computeruse.panel import ControlWindow

    if not isinstance(surface, BrowserSurface):
        return []
    marker = TargetMarker(surface)
    try:
        native = stack.enter_context(NativeWindow.open())
    except NativeWindowError:
        window = stack.enter_context(ControlWindow.open(surface.browser))
        if window.page is not None:
            surface.dock(window.page)
        return [window, marker]
    surface.dock_native(native.place, WIDTH)
    return [native, marker]


def _report_result(
    result: RunResult, *, declared_outputs: tuple[str, ...] = ()
) -> None:
    """Print how discovery ended, its outputs, and what verified them."""
    print(f"\nrun {result.ending.value} after {result.steps} steps: {result.detail}")
    fields(
        FieldSource.OUTPUT,
        (
            name if name in declared_outputs else f"output_{index}"
            for index, name in enumerate(result.outputs, 1)
        ),
    )
    for name, value in sorted(result.outputs.items()):
        print(f"  {name}: {value}")
    if result.verification is not None:
        passed = sum(1 for check in result.checks if check.passed)
        print(
            f"  verified by: {result.verification.value}"
            + (f", {passed} of {len(result.checks)} checks" if result.checks else "")
        )


def _report_segments(segments: Sequence[ManualSegment]) -> None:
    """Print what each intervention's recording holds, as counts only."""
    for segment in segments:
        taken = "taken" if segment.taken else "not taken"
        steps = len(segment.steps)
        kinds = ", ".join(gap.kind.value for gap in segment.gaps) or "none"
        print(
            f"  {segment.intervention}: {taken}, {steps} "
            f"{'step' if steps == 1 else 'steps'} recorded, gaps: {kinds}"
        )


def _journal(stack: contextlib.ExitStack, path: Path | None) -> Journal:
    if path is None:
        return MemoryJournal()
    stream: TextIO = stack.enter_context(open_evidence(path))
    return JsonlJournal(stream)
