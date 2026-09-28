"""Measure discovery and model-free replay on local target websites.

The evaluator reads expected statuses from its fixture database. The model
receives only the goal and the application's visible interface. When no
operator is available, the evaluator reports a handoff and never approves it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

import httpx
import yaml
from playwright.sync_api import Request

from computeruse.browser import BrowserSurface, open_session
from computeruse.capability import Capability
from computeruse.capability_cli import ReplayWriter, save
from computeruse.confirm import Decision, compare, decide
from computeruse.contract import Contract
from computeruse.control import Control
from computeruse.escalation import Mode
from computeruse.evidence import (
    EvidenceError,
    EvidenceParser,
    open_evidence,
    run_logged,
    safe_record,
)
from computeruse.journal import JsonlJournal
from computeruse.loop import Ending, RunResult, discover
from computeruse.model import API_KEY_VARIABLE, MODEL, REASONING, LunaDecider
from computeruse.outcomes import adopt, case_goal
from computeruse.profile import ObservationMode, Profile, load_profile
from computeruse.reading import LocalReader
from computeruse.recorder import Candidate, Recording
from computeruse.recording import DiscoveryTrace
from computeruse.replay import Status, replay
from computeruse.words import Confirmed, Word, add_words, load_words, words_path
from evaluation.faults import EXPECTED, Fault, Injection, inject
from evaluation.operator import FileChannel
from evaluation.site_checks import (
    CONTRACT,
    OPEN_ACCOUNT,
    ContextOracle,
    canonical_status,
)
from evaluation.sites import (
    APPS,
    DATABASE,
    ROOT,
    environment,
    fresh,
    server,
    unpack,
)

# A replay that stops here left the missing member for a person to decide.
STOPS_FOR_PERSON = frozenset({Status.NEEDS_HELP, Status.HELP_TIMED_OUT})


@dataclasses.dataclass
class Usage:
    """The model calls and token usage for one application trial.

    Each response supplies its usage counts. The counts contain no request or
    response content, so the evidence may include them.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0

    def reset(self) -> None:
        """Start counting again for the next application."""
        self.calls = self.input_tokens = self.output_tokens = 0
        self.reasoning_tokens = 0
        self.seconds = 0.0

    def count(self, response: httpx.Response) -> None:
        """Add one model response's usage and ignore other responses."""
        if response.request.url.host != "api.openai.com":
            return
        response.read()
        self.calls += 1
        self.seconds += response.elapsed.total_seconds()
        try:
            used = response.json().get("usage") or {}
        except ValueError:
            return
        self.input_tokens += int(used.get("input_tokens") or 0)
        self.output_tokens += int(used.get("output_tokens") or 0)
        details = used.get("output_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)


USAGE = Usage()
"""The current trial's model usage, reset as each application starts."""


def model_client() -> httpx.Client:
    """Return an HTTP client for the model that adds each call to ``USAGE``."""
    return httpx.Client(event_hooks={"response": [USAGE.count]})


@contextlib.contextmanager
def serving(app: str, env: dict[str, str]) -> Iterator[str]:
    """Start a fixture on a fresh copy of the seeded database, then stop it."""
    origin = f"http://127.0.0.1:{APPS[app]}"
    command, where = server(app)
    if shutil.which(command[0], path=env.get("PATH")) is None:
        raise ValueError(f"{command[0]} is required")
    with httpx.Client(timeout=1) as client:
        try:
            ready = client.get(origin).status_code < 500
        except httpx.HTTPError:
            ready = False
        if ready:
            # A server already running holds a database this trial did not
            # reset, so its results would depend on earlier runs.
            raise RuntimeError("stop the running application first")
        fresh()
        process = subprocess.Popen(  # noqa: S603  named local fixture
            command,
            cwd=where,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready and time.monotonic() < deadline:
                try:
                    ready = client.get(origin).status_code < 500
                except httpx.HTTPError:
                    time.sleep(0.2)
            if not ready:
                raise RuntimeError("fixture did not start")
            yield origin
        finally:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


GOALS = ROOT / "evaluation/sites/goals.yaml"
DATASET = "financial-demo-v1:northstar"
INSTITUTION = "northstar"


@dataclasses.dataclass(frozen=True)
class Case:
    """One replay input and what it must end with.

    ``inputs`` replaces some of the discovered inputs. ``expect`` is
    ``success`` or the business outcome the application gives for them.
    """

    inputs: Mapping[str, str]
    expect: str = "success"


@dataclasses.dataclass(frozen=True)
class Task:
    """One workflow the evaluation discovers and replays on every site.

    ``outcomes`` names the profile cases this task learns from. Other outcomes
    do not apply, such as a permission that a lookup never needs.
    ``replays`` contains inputs that discovery never saw. ``faulted`` selects
    the fault-replay case and defaults to the first replay. A write needs a
    separate fault case because repeating the first replay would leave two
    accounts that the capability checks cannot distinguish.
    """

    name: str
    discovered: Mapping[str, str]
    contract: Contract
    outcomes: tuple[str, ...]
    replays: tuple[Case, ...]
    writes: bool = False
    faulted: Case | None = None


LOOKUP = {"member_id": "NM000054", "operator_id": "OP0002"}
TASKS = (
    Task(
        "member_status",
        LOOKUP,
        CONTRACT,
        ("record_not_found",),
        (
            Case({"member_id": "NM000002"}),
            Case({"member_id": "NM000038"}),
            Case({"member_id": "NM999999"}, "record_not_found"),
        ),
    ),
    Task(
        "open_account",
        {
            **LOOKUP,
            "product": "USD savings",
            "nickname": "Holiday fund",
            "statement_delivery": "paper",
        },
        OPEN_ACCOUNT,
        ("record_not_found",),
        (
            Case({"member_id": "NM000002", "nickname": "Emergency fund"}),
            Case({"member_id": "NM999999"}, "record_not_found"),
            Case({"member_id": "NM000042"}, "record_ineligible"),
            # A payments maker, whom both sites offer, lacks the permission to
            # maintain accounts. The profile's own case uses another operator.
            Case({"operator_id": "OP0003"}, "permission_denied"),
        ),
        writes=True,
        faulted=Case({"member_id": "NM000002", "nickname": "Rainy day fund"}),
    ),
)


FAULTS = {
    ("white-label-responsive", "member_status"): (
        Injection(Fault.SLOW, "GET", "/members"),
        Injection(Fault.NEVER_LOADS, "GET", "/members"),
        Injection(Fault.SERVER_ERROR, "GET", "/members"),
        Injection(Fault.SESSION_EXPIRED, "GET", "/members"),
    ),
    ("white-label-responsive", "open_account"): (
        Injection(Fault.LOST_REPLY, "POST", "/commit"),
    ),
    ("web-component-operations", "member_status"): (
        Injection(Fault.SLOW, "GET", "/members"),
        Injection(Fault.SERVER_ERROR, "GET", "/members"),
    ),
    ("web-component-operations", "open_account"): (
        Injection(Fault.LOST_REPLY, "POST", "/commands"),
    ),
    ("canvas-teller", "member_status"): (
        Injection(Fault.SLOW, "POST", "/api/member"),
        Injection(Fault.SERVER_ERROR, "POST", "/api/member"),
    ),
    ("canvas-teller", "open_account"): (
        Injection(Fault.LOST_REPLY, "POST", "/api/open/commit"),
    ),
}
"""Runtime faults to replay each task under, on requests the site really makes.

The responsive portal loads pages and posts forms; the web-component
application loads one page and calls its API, and the canvas teller posts
each screen's request to its own API, so each fault targets the request that
site sends for the same step.
"""


def site_goal(app: str, task: str = "member_status") -> str:
    """Return an operator goal for one task and application.

    Each site presents its institution and sign-on differently. One shared
    sentence could request context that a site never shows. A missing goal is
    an evaluator error and has no default.
    """
    goals = yaml.safe_load(GOALS.read_text())
    tasks = goals.get(task) if isinstance(goals, dict) else None
    goal = tasks.get(app) if isinstance(tasks, dict) else None
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("the application has no declared goal")
    return " ".join(goal.split())


def expected(
    member: str,
    *,
    dataset: str = DATASET,
    institution: str = INSTITUTION,
) -> str:
    """Read fixture truth, outside every model and surface collaborator."""
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as database:
        rows = database.execute(
            "select m.status from members m join datasets d on d.id=m.dataset_id "
            "where m.member_number=? and d.name=? and d.institution_id=?",
            (member, dataset, institution),
        ).fetchall()
    if len(rows) != 1:
        raise ValueError("fixture record is missing or ambiguous")
    return str(rows[0][0])


def accounts(member: str) -> set[tuple[str, str, str, str]]:
    """Read a member's accounts: number, product, nickname, and delivery."""
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as database:
        rows = database.execute(
            "select a.account_number, p.name, a.nickname, a.statement_delivery "
            "from accounts a join members m on m.id=a.member_id "
            "and m.dataset_id=a.dataset_id join products p on p.id=a.product_id "
            "and p.dataset_id=a.dataset_id join datasets d on d.id=a.dataset_id "
            "where m.member_number=? and d.name=? and d.institution_id=?",
            (member, DATASET, INSTITUTION),
        ).fetchall()
    return {(str(a), str(b), str(c), str(e)) for a, b, c, e in rows}


def every_account() -> set[tuple[str, ...]]:
    """Read every account in the database across members and institutions.

    Each keeps its business fields and the complete account row. Left joins
    retain accounts whose member, product, or dataset is missing (rule 13).
    """
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as database:
        rows = database.execute(
            "select a.account_number, m.member_number, d.name, d.institution_id, "
            "p.name, a.nickname, a.statement_delivery, a.* "
            "from accounts a left join members m on m.id=a.member_id "
            "and m.dataset_id=a.dataset_id left join products p on p.id=a.product_id "
            "and p.dataset_id=a.dataset_id left join datasets d on d.id=a.dataset_id"
        ).fetchall()
    found: set[tuple[str, ...]] = {
        (*tuple(str(value) for value in row[:7]), json.dumps(row[7:])) for row in rows
    }
    return found


COMMITS = {
    "white-label-responsive": "/commit",
    "web-component-operations": "/commands",
    "canvas-teller": "/api/open/commit",
}
"""The end of the path each site posts a change to, so sends can be counted.

The web component application reviews at ``/commands/review`` and commits
at ``/commands``; only the commit is counted.
"""


VISUAL = False
"""Whether ``--visual`` limits discovery's observations to screenshots."""


def _screens_only(profile: Profile) -> Profile:
    """Return ``profile`` with discovery limited to screenshot observations."""
    if ObservationMode.VISUAL not in profile.perception.allowed_modes:
        raise ValueError("the profile does not permit visual observation")
    return dataclasses.replace(
        profile,
        perception=dataclasses.replace(
            profile.perception, allowed_modes=(ObservationMode.VISUAL,)
        ),
    )


CHANNEL: FileChannel | None = None
"""The person's file channel, when ``--operator-file`` asks for one."""


def control(surface: BrowserSurface, mode: Mode) -> Control:
    """Use production ownership and recording, with no invented human answers.

    With ``--operator-file``, a person or an agent reading the live page
    answers through the file channel. Without it, nobody answers.
    """
    channels = [CHANNEL] if CHANNEL is not None else []
    return Control(mode=mode, clock=time.monotonic, seat=surface, channels=channels)


@dataclasses.dataclass
class Truth:
    """What the fixture says about one run, read before and after it."""

    task: Task
    inputs: Mapping[str, str]
    before: set[tuple[str, str, str, str]] = dataclasses.field(default_factory=set)
    everything: set[tuple[str, ...]] = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        if self.task.writes:
            self.before = accounts(self.inputs["member_id"])
            self.everything = every_account()

    def result(self, outputs: Mapping[str, str], expect: str) -> bool:
        """Compare the run's effects with the fixture's expected effects.

        A write succeeds only when it adds one account for the requested
        member, dataset, and institution. That account must have the requested
        fields and the number returned by the run. No other account may be
        added, removed, or changed. An outcome changes nothing under rule 13.
        """
        member = self.inputs["member_id"]
        if not self.task.writes:
            if expect != "success":
                return True
            status = canonical_status(outputs.get("membership_status"))
            return status == expected(member)
        added, removed = self.changes()
        if removed:
            return False
        if expect != "success":
            return not added
        number = (outputs.get("account_number") or "").strip()
        return (
            bool(number)
            and len(added) == 1
            and self._intended(next(iter(added)), number)
        )

    def settled(self) -> bool:
        """Report whether the database holds no change, or only the intended one.

        A replay stopped by a fault may have left one commit behind, as a lost
        reply does. It must never leave two commits or touch another account.
        """
        added, removed = self.changes()
        if removed or len(added) > 1:
            return False
        return not added or self._intended(next(iter(added)), "")

    def changes(self) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
        """Return the accounts added, and the ones removed or changed, since before."""
        now = every_account()
        return now - self.everything, self.everything - now

    def _intended(self, row: tuple[str, ...], number: str) -> bool:
        account, member, dataset, institution, product, nickname, delivery = row[:7]
        wanted = (
            self.inputs["member_id"],
            DATASET,
            str(INSTITUTION),
            self.inputs["product"],
            self.inputs["nickname"],
            self.inputs["statement_delivery"],
        )
        held = (member, dataset, institution, product, nickname, delivery)
        return held == wanted and (not number or account == number)


def trial(
    app: str, origin: str, folder: Path, key: str, only: frozenset[str] = frozenset()
) -> dict[str, object]:
    """Run each task on one application, or only those named, each isolated."""
    summary: dict[str, object] = {
        "application": app,
        "model": MODEL,
        "reasoning": REASONING["effort"],
    }
    tasks: dict[str, object] = {}
    for task in TASKS:
        if only and task.name not in only:
            continue
        place = folder / task.name
        place.mkdir()
        try:
            tasks[task.name] = run_task(app, origin, place, key, task)
        except EvidenceError:
            raise
        except Exception as error:  # One failed task must not hide the other.
            tasks[task.name] = {"error": type(error).__name__, "passed": False}
    summary["tasks"] = tasks
    summary["passed"] = all(
        isinstance(item, dict) and item.get("passed") for item in tasks.values()
    )
    return summary


def run_task(
    app: str, origin: str, folder: Path, key: str, task: Task
) -> dict[str, object]:
    """Discover one task, confirm its texts, learn its outcomes, save, and replay."""
    # Resolve every positive case before opening a model or browser session.
    # A nonexistent fixture record is an evaluator error, not an agent failure.
    for case in task.replays:
        if case.expect == "success":
            expected(case.inputs.get("member_id", task.discovered["member_id"]))
    profile = load_profile(profile_path(app))
    goal = site_goal(app, task.name)
    truth = Truth(task, task.discovered)
    # With --visual the model sees only screenshots during discovery, as with
    # ``computeruse discover --visual``. Everything after uses the profile.
    seen = _screens_only(profile) if VISUAL else profile
    result, record, context, rebuild, commits = discover_case(
        app,
        origin,
        folder / "discovery.jsonl",
        seen,
        goal,
        task,
        task.discovered,
        key,
    )
    summary: dict[str, object] = {
        "discovery": result.ending.value,
        "perception": "visual" if VISUAL else "structured",
        "steps": result.steps,
        "verified_by": result.verification,
        "matches_fixture": truth.result(result.outputs, "success"),
        "context": context,
        "commits": commits,
        "effects_allowed": truth.settled() if task.writes else commits == 0,
        "checks": [
            {
                "kind": item.check.kind.value,
                "passed": item.passed,
                "readable": item.shown is not None,
                "match": item.check.match.value,
            }
            for item in result.checks
        ],
        "recorded": record.complete,
        "recording_issues": [safe_record(item) for item in record.issues],
        "artifact_issues": [safe_record(item) for item in record.artifact_issues],
    }
    summary["passed"] = bool(
        result.ending is Ending.COMPLETED
        and all(context.values())
        and summary["matches_fixture"]
        and commits == int(task.writes)
    )
    capability = record.capability
    if capability is None:
        summary["passed"] = False
        return summary
    listed = words_path(profile_path(app))
    if record.candidates:
        decision = confirm_texts(
            profile, origin, capability, record.candidates, key, folder, task
        )
        added = add_words(
            listed,
            decision.confirmed,
            excluded=_private_values(profile, key, task.discovered, result.outputs),
        )
        if decision.unconfirmed:
            # The operator's console, never the evidence folder: a text left
            # here may be a person's name, which only a person can judge.
            texts = [item.text for item in decision.unconfirmed]
            print(f"{app}: unconfirmed texts {json.dumps(texts)}", flush=True)
        summary["texts"] = {
            "candidates": len(record.candidates),
            "confirmed": {
                how.value: sum(
                    1 for word in decision.confirmed if word.confirmed_by is how
                )
                for how in Confirmed
            },
            "added_to_list": len(added),
            "unconfirmed": len(decision.unconfirmed),
            "complete": decision.complete,
        }
        # The draft is built again from the run with only confirmed words. A
        # delivered step leaves out a check whose text read differently. An
        # unconfirmed text the capability needs leaves it incomplete.
        strict = rebuild(frozenset(word.text for word in load_words(listed)))
        summary["recorded"] = strict.complete
        summary["recording_issues"] = [safe_record(item) for item in strict.issues]
        summary["artifact_issues"] = [
            safe_record(item) for item in strict.artifact_issues
        ]
        if strict.capability is None:
            summary["passed"] = False
            return summary
        capability = strict.capability
    learned: dict[str, object] = {}
    for outcome in task.outcomes:
        case = profile.outcomes.get(outcome)
        if case is None:
            learned[outcome] = {"learned": False, "note": "the profile names no case"}
            continue
        capability, learned[outcome] = learn_outcome(
            app, origin, folder, profile, goal, task, capability, outcome, case, key
        )
    summary["outcomes"] = learned
    save(folder / "capability.draft.json", capability)
    replayed = [
        replay_case(app, origin, folder, profile, capability, task, case)
        for case in task.replays
    ]
    summary["replays"] = replayed
    faulted = [
        replay_case(
            app,
            origin,
            folder,
            profile,
            capability,
            task,
            task.faulted or task.replays[0],
            injection,
        )
        for injection in FAULTS.get((app, task.name), ())
    ]
    summary["faults"] = faulted
    summary["passed"] = bool(summary["passed"]) and all(
        item.get("passed") for item in (*replayed, *faulted)
    )
    return summary


def discover_case(
    app: str,
    origin: str,
    journal: Path,
    profile: Profile,
    goal: str,
    task: Task,
    inputs: Mapping[str, str],
    key: str,
) -> tuple[
    RunResult, Recording, dict[str, bool], Callable[[frozenset[str]], Recording], int
]:
    """Run and record discovery while reading context from the fixture.

    The recording collects every unconfirmed text. The returned function
    rebuilds it from the run trace and saves only the supplied words.
    """
    trace = DiscoveryTrace()
    with (
        open_evidence(journal) as stream,
        model_client() as client,
        open_session(profile, origin, record=True) as surface,
    ):
        seat = control(surface, Mode.DISCOVERY)
        oracle = ContextOracle(surface._page, app, inputs["operator_id"])
        sent = _count_commits(surface, app)
        result = discover(
            goal,
            profile,
            surface=surface,
            decider=LunaDecider(api_key=key, client=client),
            control=seat,
            journal=JsonlJournal(stream),
            clock=time.monotonic,
            trace=trace,
            inputs=dict(inputs),
            outputs=tuple(field.name for field in task.contract.outputs),
        )
        context = oracle.check()
    listed = words_path(profile_path(app))
    run = seat.run
    reader = LocalReader()
    excluded = _private_values(profile, key, inputs, result.outputs)

    def build(words: frozenset[str], *, collect: bool = False) -> Recording:
        return trace.build(
            result,
            profile=profile,
            inputs=dict(inputs),
            capability_id=task.name,
            run=run,
            safe_text=words,
            contract=task.contract,
            collect=collect,
            reader=reader,
            excluded=excluded,
        )

    record = build(frozenset(word.text for word in load_words(listed)), collect=True)
    return result, record, context, build, sent[0]


def learn_outcome(
    app: str,
    origin: str,
    folder: Path,
    profile: Profile,
    goal: str,
    task: Task,
    capability: Capability,
    outcome: str,
    case: Mapping[str, str],
    key: str,
) -> tuple[Capability, dict[str, object]]:
    """Run the profile case for ``outcome`` and learn what it shows.

    A short discovery uses the case inputs. If the model finishes with that
    outcome and the loop verifies it, ``adopt`` adds the recorded run as a
    capability branch only after all its texts pass the model's veto where the
    record exists. Otherwise, the capability stays unchanged and the summary
    gives the reason.
    """
    written = case_goal(goal, task.discovered, case)
    if written is None:
        return capability, {"learned": False, "note": "the goal does not name it"}
    inputs = {**task.discovered, **case}
    truth = Truth(task, inputs)
    result, record, context, _, commits = discover_case(
        app, origin, folder / f"{outcome}.jsonl", profile, written, task, inputs, key
    )
    summary: dict[str, object] = {
        "discovery": result.ending.value,
        "steps": result.steps,
        "outcome": outcome if result.outcome == outcome else None,
        "learned": False,
        "context": context,
        "commits": commits,
        "matches_fixture": truth.result(result.outputs, outcome),
    }
    if commits != 0 or not summary["matches_fixture"] or not all(context.values()):
        summary["note"] = "the outcome discovery failed its independent safety checks"
        return capability, summary
    with model_client() as client:
        adopted = adopt(
            capability,
            record.capability if result.outcome else None,
            outcome,
            LunaDecider(api_key=key, client=client),
        )
    if adopted.capability is None:
        summary["adoption_failed"] = True
        return capability, summary
    listed = words_path(profile_path(app))
    added = add_words(
        listed,
        (Word(text, Confirmed.OUTCOME) for text in adopted.texts),
        excluded=_private_values(profile, key, inputs, result.outputs),
    )
    summary.update({"learned": True, "outcome_texts_added": len(added)})
    return adopted.capability, summary


def profile_path(app: str) -> Path:
    """Return where an application's evaluation profile lives."""
    return ROOT / "evaluation/sites/profiles" / (app + ".yaml")


def _private_values(
    profile: Profile, key: str, *fields: Mapping[str, str]
) -> tuple[str, ...]:
    values = (
        key,
        *(value for group in fields for value in group.values()),
        *profile.confirmation.values(),
        *(value for case in profile.outcomes.values() for value in case.values()),
        *(os.environ.get(variable, "") for variable in profile.secrets.values()),
    )
    return tuple(value for value in values if value)


def confirm_texts(
    profile: Profile,
    origin: str,
    capability: Capability,
    candidates: tuple[Candidate, ...],
    key: str,
    folder: Path,
    task: Task,
) -> Decision:
    """Confirm a draft's unconfirmed texts, with no person to approve them.

    The profile's second test record confirms by comparison, on the same run
    control as any replay, so a step left to a person is asked of one. Without
    a second record, the model may confirm interface labels. After a
    comparison the model may take texts away as data. Anything left stays
    unconfirmed, and the draft is built again without it. The comparison's
    replay log is written like any replay's, carrying structure and never
    values.
    """
    compared = None
    if profile.confirmation:
        second = {**task.discovered, **profile.confirmation}
        with (
            open_evidence(folder / "compare.jsonl") as log,
            open_session(profile, origin, record=True) as surface,
        ):
            compared = compare(
                capability,
                candidates,
                second,
                task.discovered,
                profile=profile,
                surface=surface,
                clock=time.monotonic,
                sleep=surface.idle,
                log=ReplayWriter(log),
                # A step left to a person is asked of one, like any replay.
                control=control(surface, Mode.REPLAY),
            )
    with model_client() as client:
        model = LunaDecider(api_key=key, client=client)
        return decide(
            candidates,
            compared=compared,
            classifier=None if compared else model,
            veto=model,
            ask=None,
        )


def replay_case(
    app: str,
    origin: str,
    folder: Path,
    profile: Profile,
    capability: Capability,
    task: Task,
    case: Case,
    injection: Injection | None = None,
) -> dict[str, object]:
    """Replay the saved artifact for one case and compare it with fixture truth.

    A success case must return what the fixture shows, and for a write change
    exactly the one account the inputs describe. An outcome case must end with
    that outcome and, for a write, change nothing. One failed replay never
    hides the others.
    """
    inputs = {**task.discovered, **case.inputs}
    configured = (*task.replays, *((task.faulted,) if task.faulted else ()))
    try:
        number = configured.index(case) + 1
    except ValueError:
        raise ValueError("the replay case is not declared by this task") from None
    name = f"case-{number}"
    if injection is not None:
        name = f"{injection.fault}-{name}"
    fired = None
    try:
        truth = Truth(task, inputs)
        with (
            open_evidence(folder / f"replay-{name}.jsonl") as log,
            open_evidence(folder / f"replay-{name}.human.jsonl") as human,
            open_session(profile, origin, record=True) as surface,
            (
                inject(surface, injection)
                if injection is not None
                else contextlib.nullcontext()
            ) as record,
        ):
            oracle = ContextOracle(surface._page, app, inputs["operator_id"])
            sent = _count_commits(surface, app)
            replayed = replay(
                capability,
                inputs,
                profile=profile,
                surface=surface,
                control=control(surface, Mode.REPLAY),
                journal=JsonlJournal(human),
                log=ReplayWriter(log),
                clock=time.monotonic,
                sleep=surface.idle,
                accept_draft=True,
            )
            context = oracle.check()
            fired = record.fired if record is not None else None
    except EvidenceError:
        raise
    except Exception as error:  # One failed replay must not hide the others.
        return {"input": name, "error": type(error).__name__, "passed": False}
    # A write sends its change at most once, whatever the application does
    # with a second send (rule 13).
    once = not task.writes or sent[0] <= 1
    if injection is not None:
        # The rule, fixed before the runs it judges: a fault case passes when
        # the fault fired, the operator and institution held, nothing was
        # sent twice, and the replay either stopped as the fault allows or
        # finished with the fixture's own answer, by bounded recovery or
        # after a person settled it. A person's answer alone proves nothing.
        succeeded = replayed.status is Status.SUCCEEDED
        if succeeded:
            fixture = truth.result(replayed.outputs, "success")
        else:
            fixture = truth.settled() if task.writes else True
        allowed = EXPECTED[injection.fault] | {Status.SUCCEEDED}
        return {
            "input": name,
            "fault": injection.fault,
            "fired": fired,
            "status": replayed.status,
            "reason": replayed.reason,
            "node": replayed.node,
            "matches_fixture": fixture,
            "context": context,
            "commits": sent[0],
            "passed": bool(fired)
            and replayed.status in allowed
            and fixture
            and once
            and all(context.values()),
        }
    fixture = truth.result(replayed.outputs, case.expect)
    if case.expect == "success":
        ended = replayed.status is Status.SUCCEEDED
        # A write that succeeded sent exactly one change.
        once = once and (not task.writes or sent[0] == 1)
    else:
        ended = replayed.status is Status.OUTCOME and replayed.outcome == case.expect
    return {
        "input": name,
        "expect": case.expect,
        "status": replayed.status,
        "reason": replayed.reason,
        "outcome": replayed.outcome or None,
        "steps": replayed.steps,
        "matches_fixture": fixture,
        "context": context,
        "commits": sent[0],
        "passed": ended and fixture and once and all(context.values()),
    }


def _count_commits(surface: BrowserSurface, app: str) -> list[int]:
    """Count the changes the browser posts to the site's commit path.

    The count is the evaluation's own, from every request the session's
    browser sends, never from what the replay reports about itself.
    """
    sent = [0]
    suffix = COMMITS.get(app)

    def seen(request: Request) -> None:
        path = urlsplit(request.url).path
        if suffix and request.method == "POST" and path.endswith(suffix):
            sent[0] += 1

    surface._page.context.on("request", seen)
    return sent


OPERATOR_VARIABLE = "EVALUATION_OPERATOR"
"""Names who answers a run's requests, for the manifest; unset means nobody."""


def _git(*arguments: str) -> str:
    if not (ROOT / ".git").exists():
        return ""
    return subprocess.run(  # noqa: S603  fixed git argv
        ["git", *arguments],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def uncommitted() -> bool | None:
    """Report whether anything that affects execution differs from the commit.

    That is the product, the evaluation with its profiles and contracts, and
    the locked dependencies (rule 13). It is read before a site's run starts,
    because the run itself adds the words it confirms to the site's list.
    """
    if not _git("rev-parse", "HEAD"):
        return None
    return bool(
        _git(
            "status",
            "--porcelain",
            "--",
            "src",
            "evaluation",
            "pyproject.toml",
            "uv.lock",
        )
    )


class Provenance(TypedDict):
    """Starting source identity, with a live driver path that is never exported."""

    code_revision: str | None
    uncommitted_changes: bool | None
    execution_files: dict[str, str]
    driver: Path | None


def provenance(driver: Path | None = None) -> Provenance:
    """Hash execution inputs before running, including an external Python driver."""
    if driver is None:
        entry = getattr(sys.modules["__main__"], "__file__", None)
        driver = Path(entry).resolve() if entry else None
    paths = {name: ROOT / name for name in ("pyproject.toml", "uv.lock")}
    for folder in (ROOT / "src", ROOT / "evaluation"):
        for directory, children, filenames in os.walk(folder):
            children[:] = [
                name
                for name in children
                if name not in {"__pycache__", "runtime"}
                and not name.startswith(".")
                and not name.endswith(".egg-info")
            ]
            for name in filenames:
                if not name.startswith(".") and not name.endswith((".pyc", ".pyo")):
                    path = Path(directory) / name
                    paths[str(path.relative_to(ROOT))] = path
    if driver is not None and driver.is_file():
        name = (
            str(driver.relative_to(ROOT))
            if driver.is_relative_to(ROOT)
            else f"launcher/{driver.name}"
        )
        paths[name] = driver
    return {
        "code_revision": _git("rev-parse", "HEAD") or None,
        "uncommitted_changes": uncommitted(),
        "execution_files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())
            if path.is_file()
        },
        "driver": driver,
    }


def manifest(
    folder: Path,
    stamp: str,
    dirty: bool | None = None,
    *,
    started: Provenance | None = None,
) -> dict[str, object]:
    """Describe a site's code, model, artifacts, and operator for one run.

    The description contains no goal, page text, or value, so it may accompany
    the evidence. A capability uses its file's SHA-256 as its name, which ties
    the evidence to the artifact replay used. Starting sources remain separate
    from final sources, including any word list confirmed during the run.
    """
    initial = started if started is not None else provenance()
    final = provenance(initial["driver"])
    before, after = initial["execution_files"], final["execution_files"]
    return {
        "code_revision": initial["code_revision"],
        "finished_code_revision": final["code_revision"],
        "uncommitted_changes": initial["uncommitted_changes"]
        if dirty is None
        else dirty,
        "execution_files": before,
        "finished_execution_files": after,
        "changed_execution_files": sorted(
            name
            for name in before.keys() | after.keys()
            if before.get(name) != after.get(name)
        ),
        "python_version": sys.version.split()[0],
        "started_at": stamp,
        "finished_at": _now(),
        "model": MODEL,
        "reasoning": REASONING["effort"],
        "operator_declared": bool(os.environ.get(OPERATOR_VARIABLE)),
        "capabilities": {
            str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("capability.draft.json"))
        },
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    """Save bounded evidence for every requested application."""
    parser = EvidenceParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--app", choices=APPS, action="append")
    parser.add_argument("--node", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--task", choices=[task.name for task in TASKS], action="append", default=[]
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="limit discovery's observations to screenshots",
    )
    parser.add_argument(
        "--operator-file",
        type=Path,
        help="a folder whose commands.txt a person appends answers to",
    )
    args = parser.parse_args()
    global CHANNEL, VISUAL
    VISUAL = args.visual
    if args.operator_file is not None:
        CHANNEL = FileChannel(args.operator_file)
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        parser.error("OPENAI_API_KEY is required for discovery")
    unpack()
    env = environment(args.node)
    args.out.mkdir(parents=True, exist_ok=False)
    # A repeated application runs once because its folder cannot be created twice.
    chosen = args.app or list(APPS)
    results: list[dict[str, object]] = []
    for app in dict.fromkeys(chosen):
        folder = args.out / app
        folder.mkdir()
        print(f"{app}: starting", flush=True)
        summary: dict[str, object]
        started = time.monotonic()
        stamp = _now()
        sources = provenance()
        USAGE.reset()
        try:
            with serving(app, env) as origin:
                summary = trial(app, origin, folder, key, frozenset(args.task))
        except EvidenceError:
            raise
        except Exception as error:  # One failed trial must not hide later websites.
            summary = {
                "application": app,
                "error": type(error).__name__,
                "passed": False,
            }
        summary["usage"] = dataclasses.asdict(USAGE)
        summary["seconds"] = round(time.monotonic() - started, 1)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (folder / "manifest.json").write_text(
            json.dumps(
                manifest(folder, stamp, started=sources)
                | {
                    "tasks": args.task or [task.name for task in TASKS],
                    "visual": VISUAL,
                },
                indent=2,
            )
            + "\n"
        )
        results.append(summary)
        (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(run_logged(main, command="evaluation"))
