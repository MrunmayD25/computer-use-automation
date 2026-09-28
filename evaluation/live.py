"""Run live discovery and compare each result with fixture answers.

Each case defines a goal, an application, the expected result, who verifies
that result, expected requests for a person, and the required reading of the
goal. Only this module contains the fixture answers. A scripted person uses
them as a console operator would use the live page. The model sees the goal
and screen, never the answers. After each run, the checks read the live page
instead of trusting the reported result.

The application is the evaluation site in ``evaluation/site``, served as it
is.

Run one case with the model key in the environment:

    uv run --env-file .env python -m evaluation.live --case eval-balance

``--trials`` repeats each case because one model run does not predict the
next. When provided, ``--out`` receives a summary and sanitized journal for
each run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import yaml

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    RecordEvidence,
    Relation,
)
from computeruse.browser import BrowserSurface, open_session
from computeruse.escalation import (
    Handoff,
    HandoffOutcome,
    InterventionRequest,
    Trigger,
)
from computeruse.evidence import EvidenceParser, Problem, ProblemCode, emit, run_logged
from computeruse.journal import JsonlJournal
from computeruse.loop import Ending, RunResult, Verification, discover
from computeruse.model import API_KEY_VARIABLE, MODEL, REASONING, LunaDecider
from computeruse.profile import ActionKind, Profile, load_profile
from evaluation.serve import serve

ROOT = Path(__file__).resolve().parents[1]


class FixtureProblem(StrEnum):
    """Closed failure categories that do not contain observed or expected values."""

    HANDOFFS_MISMATCH = "handoffs_mismatch"
    RUN_INCOMPLETE = "run_incomplete"
    VERIFICATION_MISMATCH = "verification_mismatch"
    RECORDS_MISSING = "records_missing"
    FIXTURE_MISMATCH = "fixture_mismatch"


@dataclasses.dataclass(frozen=True)
class Case:
    """One live workflow and its expected result.

    ``verification`` names who establishes the result. Use ``executor`` when
    the page shows every requirement. Use ``person`` when a person must confirm
    a relationship that the page cannot show. ``handoffs`` lists the expected
    requests for a person in order. Any other request fails the case.
    ``records`` lists the identifiers required in the goal reading and the
    identifier containing each one, if any.
    """

    app: str
    goal: str
    entry: str
    verification: str
    handoffs: tuple[str, ...] = ()
    records: tuple[tuple[str, str], ...] = ()
    profile: dict[str, Any] = dataclasses.field(default_factory=dict)
    person: dict[str, Callable[[BrowserSurface, InterventionRequest], Handoff]] = (
        dataclasses.field(default_factory=dict)
    )
    check: Callable[[BrowserSurface, RunResult], list[str]] = lambda _p, _r: []


def _read(page: BrowserSurface, target: AxLocator | DomLocator) -> str:
    """Read ``target`` from the live page, or describe why it could not be read."""
    result = page.act(Action(ActionKind.READ, target))
    if result.outcome.value != "ok":
        return f"({result.outcome.value}: {result.detail} at {page.location()})"
    return result.extracted or ""


def _bound(result: RunResult, output: str) -> set[str]:
    """Every record the run's passing checks tied an output's value to."""
    return {
        item.bound.casefold()
        for item in result.checks
        if item.passed and item.check.output == output and item.bound
    }


def _outputs(result: RunResult) -> str:
    return " ".join(result.outputs.values())


# Fixture answers for the evaluation site. They are read again from the page.

LEDGER = ("ledger",)


def _s1001_balance(page: BrowserSurface) -> list[str]:
    cell = DomLocator("td", DomAttribute.TEXT, "$4,212.55", frame=LEDGER)
    account = AxLocator("rowheader", "S-1001", frame=LEDGER)
    proof = RecordEvidence(account, "S-1001", Relation.ROW)
    shown = page.act(Action(ActionKind.READ, cell, evidence=proof)).extracted
    if shown != "$4,212.55":
        return ["the S-1001 balance is not on the final page"]
    return []


def _eval_balance(page: BrowserSurface, result: RunResult) -> list[str]:
    problems = _s1001_balance(page)
    if "$4,212.55" not in _outputs(result):
        problems.append("the reported outputs do not carry $4,212.55")
    if not page.location().endswith("/members/12345"):
        problems.append("the run did not end on member 12345's page")
    return problems


def _eval_balance_bound(page: BrowserSurface, result: RunResult) -> list[str]:
    """Check the balance, and that a passing check tied it to its account row."""
    problems = _eval_balance(page, result)
    if "s-1001" not in _bound(result, "balance"):
        problems.append("no passing check tied the balance to account S-1001")
    return problems


def _eval_popup(page: BrowserSurface, result: RunResult) -> list[str]:
    problems = _eval_balance(page, result)
    listed = page.pages()
    if [item.page_id for item in listed if item.active] != ["page-1"]:
        problems.append("the original page is not the active one")
    return problems


def _eval_memory(page: BrowserSurface, result: RunResult) -> list[str]:
    outputs = _outputs(result)
    problems = [
        f"the reported outputs do not carry {wanted}"
        for wanted in ("Riverside", "$318.00")
        if wanted not in outputs
    ]
    cell = DomLocator("td", DomAttribute.TEXT, "$318.00", frame=LEDGER)
    shown = _read(page, cell)
    if shown != "$318.00":
        problems.append(f"the S-1002 balance is not on the final page: {shown}")
    # Members 12345 and 24680 are both at Riverside, so the value alone says
    # nothing about which member it came from. The branch must have been tied
    # to 24680's own row.
    branch = [name for name, value in result.outputs.items() if value == "Riverside"]
    if not branch or not any("24680" in _bound(result, name) for name in branch):
        problems.append("no passing check tied the branch to member 24680")
    return problems


def _statements_person(page: BrowserSurface, request: InterventionRequest) -> Handoff:
    """Review the statements window as a person would, then close it.

    The person checks the fixture answer, that member 67890 is listed, in
    the new window itself. The run never sees that window.
    """
    opened = [item for item in request.pages if not item.active]
    if len(opened) != 1:
        return Handoff(HandoffOutcome.REJECTED, "expected one new window")
    window = page._pages.get(opened[0].page_id)
    if window is None:
        return Handoff(HandoffOutcome.REJECTED, "the window is gone")
    window.wait_for_load_state("load")
    listed = window.locator("th[scope=row]").all_inner_texts()
    window.close()
    page._page.wait_for_timeout(200)
    if "67890" not in listed:
        return Handoff(HandoffOutcome.REJECTED, "member 67890 is not listed")
    return Handoff(HandoffOutcome.RESUMED, "I reviewed the statements and closed them")


def _confirms(
    fixture: Callable[[BrowserSurface, RunResult], list[str]],
) -> Callable[[BrowserSurface, InterventionRequest], Handoff]:
    """Confirm a result only when the live page shows the fixture answer."""

    def _confirm(page: BrowserSurface, request: InterventionRequest) -> Handoff:
        shown = RunResult(Ending.COMPLETED, request.step, "", outputs=request.outputs)
        problems = [
            problem
            for problem in fixture(page, shown)
            if "tied" not in problem and "active" not in problem
        ]
        if problems:
            return Handoff(HandoffOutcome.REJECTED, "; ".join(problems))
        return Handoff(HandoffOutcome.APPROVED, "the page shows that result")

    return _confirm


def _queue_person(page: BrowserSurface, request: InterventionRequest) -> Handoff:
    """Approve member 10001 as a person would, in the same live session.

    The person here is scripted, and approves with the structured record
    check the run itself could not supply.
    """
    del request
    approved = DomLocator("output", DomAttribute.ID, "approved-member")
    if _read(page, approved) == "10001":
        return Handoff(HandoffOutcome.RESUMED, "member 10001 is already approved")
    number = DomLocator("dd", DomAttribute.ID, "member-number")
    proof = RecordEvidence(number, "10001", Relation.CONTAINER)
    approve = AxLocator("button", "Approve")
    page.act(Action(ActionKind.CLICK, approve, evidence=proof))
    return Handoff(HandoffOutcome.RESUMED, "I approved member 10001")


def _eval_queue(page: BrowserSurface, _: RunResult) -> list[str]:
    approved = _read(page, DomLocator("output", DomAttribute.ID, "approved-member"))
    return [] if approved == "10001" else ["member 10001 was not approved"]


VISUAL_ONLY = {
    "perception": {
        "allowed_modes": ["visual"],
        "max_alternate_observations_per_step": 1,
    }
}
"""Screenshots only, with the evaluation profile's allowance for a second look."""

BALANCE_GOAL = "Report the balance of savings account S-1001 for member 12345."
S1001 = (("12345", ""), ("s-1001", "12345"))
"""The reading of a balance goal: the member, and the account within it."""

UNVERIFIED = Trigger.UNVERIFIED_RESULT.value

CASES = {
    # The ledger is a frame of its own and shows no member number, so no
    # control ties S-1001 to member 12345. A person confirms that tie.
    "eval-balance": Case(
        app="evaluation",
        goal=BALANCE_GOAL,
        entry="/members",
        verification="person",
        handoffs=(UNVERIFIED,),
        records=S1001,
        person={UNVERIFIED: _confirms(_eval_balance)},
        check=_eval_balance_bound,
    ),
    # A screenshot point carries no record evidence at all.
    "eval-balance-visual": Case(
        app="evaluation",
        goal=BALANCE_GOAL,
        entry="/members",
        verification="person",
        handoffs=(UNVERIFIED,),
        records=S1001,
        profile=VISUAL_ONLY,
        person={UNVERIFIED: _confirms(_eval_balance)},
        check=_eval_balance,
    ),
    # Every permitted new window goes to a person, who reviews and closes it.
    "eval-popup": Case(
        app="evaluation",
        goal=(
            "On member 12345's page, open the statements window for the person "
            "reviewing statements, then report the balance of savings account "
            "S-1001 for member 12345."
        ),
        entry="/members/12345",
        verification="person",
        handoffs=(Trigger.NEW_WINDOW.value, UNVERIFIED),
        records=S1001,
        profile={"allow_new_windows": True},
        person={
            Trigger.NEW_WINDOW.value: _statements_person,
            UNVERIFIED: _confirms(_eval_popup),
        },
        check=_eval_popup,
    ),
    "eval-memory": Case(
        app="evaluation",
        goal=(
            "Note the branch of member 24680 on the member search page. Then "
            "open member 12345 and report the balance of savings account "
            "S-1002 together with the branch you noted for member 24680."
        ),
        entry="/members",
        verification="person",
        handoffs=(UNVERIFIED,),
        records=(("24680", ""), ("12345", ""), ("s-1002", "12345")),
        person={UNVERIFIED: _confirms(_eval_memory)},
        check=_eval_memory,
    ),
    "eval-queue-handoff": Case(
        app="evaluation",
        goal="Approve the request of member 10001 in the approval queue.",
        entry="/queue",
        verification="executor",
        handoffs=(Trigger.RECORD_EVIDENCE_REQUIRED.value,),
        records=(("10001", ""),),
        profile=VISUAL_ONLY,
        person={Trigger.RECORD_EVIDENCE_REQUIRED.value: _queue_person},
        check=_eval_queue,
    ),
    # The account page shows the member ID in the same document as the
    # balance, so the executor can tie the two.
}


@dataclasses.dataclass
class Person:
    """The operator channel for a live case: a scripted person, or nobody.

    A request the case has no answer for times out, and counts against the
    case, because the run asked for help it was not expected to need.
    """

    page: BrowserSurface
    acts: dict[str, Callable[[BrowserSurface, InterventionRequest], Handoff]]
    requests: list[InterventionRequest] = dataclasses.field(default_factory=list)

    def request(self, intervention: InterventionRequest) -> Handoff:
        """Record the request, and answer it as the case's person would."""
        self.requests.append(intervention)
        act = self.acts.get(intervention.trigger.value)
        if act is None:
            return Handoff(HandoffOutcome.TIMED_OUT)
        return act(self.page, intervention)


@contextlib.contextmanager
def _evaluation_site() -> Iterator[tuple[str, dict[str, Any]]]:
    document = yaml.safe_load((ROOT / "evaluation" / "profile.yaml").read_text())
    with serve() as base:
        yield base, document


def run_case(name: str, out: Path | None, trial: int = 1) -> dict[str, Any]:
    """Run one case and return its summary, with the fixture checks applied."""
    case = CASES[name]
    with _evaluation_site() as (base, document), httpx.Client() as client:
        document = {**document, **case.profile, "base_url": base}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.yaml"
            path.write_text(yaml.safe_dump(document))
            profile: Profile = load_profile(path)
        stream = io.StringIO()
        started = time.monotonic()
        with open_session(profile, f"{base}{case.entry}") as page:
            person = Person(page, case.person)
            result = discover(
                case.goal,
                profile,
                surface=page,
                decider=LunaDecider(
                    api_key=os.environ[API_KEY_VARIABLE], client=client
                ),
                escalator=person,
                journal=JsonlJournal(stream),
                clock=time.monotonic,
            )
            seconds = round(time.monotonic() - started, 2)
            problems = _expectation(case, result, person)
            fixture_problems = case.check(page, result)
            if fixture_problems:
                problems.append(FixtureProblem.FIXTURE_MISMATCH)
    summary = {
        "case": name,
        "trial": trial,
        "app": case.app,
        "model": MODEL,
        "reasoning": REASONING["effort"],
        "expected": {
            "verification": case.verification,
            "handoffs": list(case.handoffs),
            "record_count": len(case.records),
        },
        "ending": result.ending.value,
        "verification": result.verification.value if result.verification else None,
        "steps": result.steps,
        "seconds": seconds,
        "output_count": len(result.outputs),
        "task": {
            "record_count": len(result.task.records),
            "output_count": len(result.task.outputs),
            "requirement_count": len(result.task.requirements),
            "changes": result.task.changes,
        },
        "checks": [
            {
                "kind": item.check.kind.value,
                "target": type(item.check.target).__name__,
                "record_evidence": item.check.record is not None,
                "record_bound": bool(item.bound),
                "passed": item.passed,
            }
            for item in result.checks
        ],
        "handoffs": [request.trigger.value for request in person.requests],
        "unverified_count": sum(len(request.unverified) for request in person.requests),
        "fixture_problems": problems,
        "fixture_failure_count": len(fixture_problems),
        "fixture_checks_passed": not fixture_problems,
        "passed": not problems,
    }
    if out is not None:
        folder = out / name / f"trial-{trial}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (folder / "journal.jsonl").write_text(stream.getvalue())
    return summary


def _expectation(case: Case, result: RunResult, person: Person) -> list[FixtureProblem]:
    """Compare how the run ended, and how it read the goal, with the case."""
    problems: list[FixtureProblem] = []
    triggers = tuple(request.trigger.value for request in person.requests)
    if triggers != case.handoffs:
        problems.append(FixtureProblem.HANDOFFS_MISMATCH)
    if result.ending is not Ending.COMPLETED:
        problems.append(FixtureProblem.RUN_INCOMPLETE)
    elif result.verification is not Verification(case.verification):
        problems.append(FixtureProblem.VERIFICATION_MISMATCH)
    names = {item.name: item.value for item in result.task.records}
    read = {
        (item.value.casefold(), names.get(item.within, "").casefold())
        for item in result.task.records
    }
    missing = [item for item in case.records if item not in read]
    if missing:
        problems.append(FixtureProblem.RECORDS_MISSING)
    return problems


SHOWN = (
    "case",
    "trial",
    "expected",
    "ending",
    "verification",
    "steps",
    "seconds",
    "handoffs",
    "fixture_problems",
    "passed",
)


def main() -> int:
    """Run the named cases, each for the given number of trials, one line each."""
    parser = EvidenceParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get(API_KEY_VARIABLE):
        emit(Problem(ProblemCode.MISSING_CREDENTIAL))
        print(f"error: {API_KEY_VARIABLE} is not set", file=sys.stderr)
        return 2
    failed = 0
    for name in args.case or sorted(CASES):
        for trial in range(1, args.trials + 1):
            summary = run_case(name, args.out, trial)
            failed += not summary["passed"]
            print(json.dumps({key: summary[key] for key in SHOWN}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_logged(main, command="evaluation"))
