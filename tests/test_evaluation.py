"""Fixture truth, session context, and display semantics stay independent."""

import contextlib
import json
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from playwright.sync_api import Page

from evaluation import integrated
from evaluation.faults import Fault
from evaluation.site_checks import ContextOracle, canonical_status


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Active", "active"),
        ("Inactive", "inactive"),
        ("reactivated", None),
        ("not active", None),
        ("Active account", None),
        (None, None),
    ],
)
def test_status_comparison_requires_one_exact_business_value(value, expected):
    assert canonical_status(value) == expected


def test_ground_truth_is_bound_to_both_institution_and_dataset(tmp_path, monkeypatch):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("create table datasets (id text, name text, institution_id text)")
        db.execute(
            "create table members (dataset_id text, member_number text, status text)"
        )
        db.executemany(
            "insert into datasets values (?, ?, ?)",
            [
                ("one", "world", "bank"),
                ("two", "other", "bank"),
                ("three", "world", "other"),
            ],
        )
        db.executemany(
            "insert into members values (?, ?, ?)",
            [
                ("two", "12345", "active"),
                ("three", "12345", "active"),
                ("one", "12345", "inactive"),
            ],
        )
    monkeypatch.setattr(integrated, "DATABASE", path)
    assert (
        integrated.expected("12345", dataset="world", institution="bank") == "inactive"
    )
    with pytest.raises(ValueError) as error:  # noqa: PT011  exact assertion below
        integrated.expected("99999", dataset="world", institution="bank")
    assert str(error.value) == "fixture record is missing or ambiguous"


def test_selected_but_rejected_signon_is_not_proof_of_context():
    events = {}
    page = SimpleNamespace(
        url="http://127.0.0.1:9999/",
        context=SimpleNamespace(
            on=lambda name, callback: events.update({name: callback})
        ),
    )
    oracle = ContextOracle(cast("Page", page), "white-label-responsive", "OP0002")
    request = SimpleNamespace(
        url=page.url + "signon", method="POST", post_data="binding=0&actorNumber=OP0002"
    )
    events["request"](request)
    assert not oracle.signed_on
    response = SimpleNamespace(request=request, status=403, headers={})
    events["response"](response)
    assert not oracle.signed_on
    response.status = 303
    response.headers = {"location": "/"}
    events["response"](response)
    assert oracle.signed_on
    request.post_data = "binding=1&actorNumber=OP0002"
    events["response"](response)
    assert not oracle.signed_on


@pytest.mark.parametrize("changed", ["operator", "institution"])
def test_context_checks_reject_wrong_operator_or_institution(pages, changed):
    markup = '<ops-app><span class="operator">Operator OP0002</span>'
    markup += '<span class="inst">Institution: Northstar</span></ops-app>'
    classes = {"operator": "operator", "institution": "inst"}
    with pages({"/": markup}) as (surface, _):
        oracle = ContextOracle(surface._page, "web-component-operations", "OP0002")
        assert all(oracle.check().values())
        surface._page.locator(f"ops-app .{classes[changed]}").evaluate(
            "el => el.textContent = 'Another context'"
        )
        assert not oracle.check()[changed]


def test_missing_context_controls_fail_without_waiting_for_a_timeout(pages):
    with pages({"/": "<h1>Unavailable</h1>"}) as (surface, _):
        for app in ("web-component-operations", "an-unknown-application"):
            assert ContextOracle(surface._page, app, "OP0002").check() == {
                "operator": False,
                "institution": False,
            }


@pytest.mark.parametrize("evidence_failure", [False, True])
def test_one_trial_error_does_not_hide_later_sites(
    tmp_path, monkeypatch, evidence_failure
):
    from computeruse.evidence import EvidenceError, run_logged

    folder = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["evaluation", "--out", str(folder)])
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setattr(integrated, "unpack", lambda: None)
    monkeypatch.setattr(integrated, "environment", lambda _: {})
    monkeypatch.setattr(integrated, "APPS", {"first": 1, "second": 2})
    monkeypatch.setattr(
        integrated, "serving", lambda *_: contextlib.nullcontext("local")
    )

    def trial(app, *_):
        if app == "first":
            if evidence_failure:
                raise EvidenceError("private diagnostic text")
            raise KeyError("private diagnostic text")
        return {"application": app, "passed": True}

    monkeypatch.setattr(integrated, "trial", trial)
    code = run_logged(integrated.main, command="evaluation")
    if evidence_failure:
        assert code == 2
        assert not (folder / "summary.json").exists()
        return
    assert code == 0
    written = (folder / "summary.json").read_text()
    results = json.loads(written)
    failed = {key: results[0][key] for key in ("application", "error", "passed")}
    assert failed == {"application": "first", "error": "KeyError", "passed": False}
    # Each site also reports its model usage as counts, which hold no content.
    assert results[0]["usage"]["calls"] == 0
    assert results[1]["passed"]
    assert "private diagnostic text" not in written


def test_a_repeated_site_runs_once(tmp_path, monkeypatch):
    folder = tmp_path / "matrix"
    argv = ["evaluation", "--out", str(folder), "--app", "first", "--app", "first"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setattr(integrated, "unpack", lambda: None)
    monkeypatch.setattr(integrated, "environment", lambda _: {})
    monkeypatch.setattr(
        integrated, "serving", lambda *_: contextlib.nullcontext("local")
    )
    monkeypatch.setattr(
        integrated, "trial", lambda app, *_: {"application": app, "passed": True}
    )
    monkeypatch.setattr(integrated, "APPS", {"first": 1})
    integrated.main()
    assert len(json.loads((folder / "summary.json").read_text())) == 1


def test_every_site_has_its_own_goal_and_a_loadable_profile():
    from computeruse.profile import load_profile
    from evaluation.sites import APPS

    for app in APPS:
        for task in integrated.TASKS:
            goal = integrated.site_goal(app, task.name)
            # A goal names what to do, never the answer it expects.
            assert "NM000054" in goal
            assert not any(
                status in goal.casefold() for status in ("inactive", "deceased")
            )
        profile = load_profile(
            integrated.ROOT / "evaluation/sites/profiles" / f"{app}.yaml"
        )
        # Every outcome a task learns has a test case in the profile.
        wanted = {name for task in integrated.TASKS for name in task.outcomes}
        assert wanted <= set(profile.outcomes)


@pytest.mark.rule(1, 5, 6)
def test_receipt_navigation_is_distinct_from_the_web_component_commit():
    import dataclasses

    from computeruse import operations
    from computeruse.actions import Action, AxLocator, AxNode, Operation
    from computeruse.profile import ActionKind, Limit, load_profile

    profile = load_profile(integrated.profile_path("web-component-operations"))
    action = Action(ActionKind.CLICK, AxLocator("button", "Open receipt"))
    node = AxNode(
        "button", "Open receipt", tag="button", context=("heading:Committed",)
    )
    bound = operations.identify(profile, action, "/", node=node)
    assert bound is not None
    operation = dataclasses.replace(
        Operation.of(action, "/", node=node),
        binding=bound.name,
        business=bound.business,
    )
    assert operations.limit(profile, operation) is None
    assert not operations.required_selections(profile, operation)
    unknown = Operation(ActionKind.CLICK, "/", "screen")
    assert operations.limit(profile, unknown) is Limit.RISKY
    assert operations.required_selections(profile, unknown) == {"open_account"}
    assert (
        operations.identify(
            profile,
            action,
            "/",
            node=dataclasses.replace(node, name="Another operation"),
        )
        is None
    )


@pytest.mark.rule(13)
def test_the_manifest_ties_the_evidence_to_its_code_and_artifacts(
    tmp_path, monkeypatch
):
    import hashlib

    monkeypatch.setattr(integrated, "_git", lambda *_arguments: "")
    saved = tmp_path / "open_account" / "capability.draft.json"
    saved.parent.mkdir()
    saved.write_text('{"capability_id": "open_account"}')
    monkeypatch.setenv(integrated.OPERATOR_VARIABLE, "an agent on the file channel")
    written = integrated.manifest(tmp_path, "2026-09-23T00:00:00Z")
    assert written["capabilities"] == {
        "open_account/capability.draft.json": hashlib.sha256(
            saved.read_bytes()
        ).hexdigest()
    }
    assert written["operator_declared"] is True
    assert "an agent on the file channel" not in json.dumps(written)
    assert written["code_revision"] is None
    assert written["uncommitted_changes"] is None
    assert written["reasoning"] == "xhigh"
    files = written["execution_files"]
    assert isinstance(files, dict)
    assert "evaluation/sites/member-status.contract.json" in files
    assert "evaluation/sites/profiles/canvas-teller.yaml" in files
    assert "src/computeruse/replay.py" in files
    assert "pyproject.toml" in files
    assert "uv.lock" in files
    assert not any("runtime" in Path(name).parts for name in files)
    assert not any("__pycache__" in Path(name).parts for name in files)


@pytest.mark.rule(13)
def test_manifest_keeps_starting_sources_when_the_run_changes_them(
    tmp_path, monkeypatch
):
    source = tmp_path / "evaluation/profile.yaml"
    source.parent.mkdir()
    source.write_text("before")
    driver = tmp_path / "driver.py"
    driver.write_text("# diagnostic launcher\n")
    revision = ["a" * 40]
    monkeypatch.setattr(integrated, "ROOT", tmp_path)
    monkeypatch.setattr(
        integrated,
        "_git",
        lambda *_args: revision[0],
    )
    monkeypatch.setattr(integrated, "uncommitted", lambda: False)
    started = integrated.provenance(driver)
    source.write_text("after")
    driver.write_text("# changed launcher\n")
    revision[0] = "b" * 40
    written = integrated.manifest(tmp_path, "2026-09-24T00:00:00Z", started=started)
    assert written["code_revision"] == "a" * 40
    assert written["finished_code_revision"] == "b" * 40
    assert written["execution_files"] == started["execution_files"]
    assert written["changed_execution_files"] == [
        "driver.py",
        "evaluation/profile.yaml",
    ]
    assert written["finished_execution_files"] != written["execution_files"]


def _write_truth(monkeypatch, before, after):
    """A write task's truth, with the database readings replaced by two sets."""
    readings = iter([before, after, after, after])
    monkeypatch.setattr(integrated, "every_account", lambda: next(readings))
    monkeypatch.setattr(integrated, "accounts", lambda _member: set())
    task = next(task for task in integrated.TASKS if task.writes)
    return integrated.Truth(task, task.discovered)


_DATASET = integrated.DATASET
_HOME = integrated.INSTITUTION
_OLD = ("AC1", "NM000002", _DATASET, _HOME, "USD checking", "Everyday", "paper")


def _new(member="NM000054", nickname="Holiday fund"):
    return ("AC9", member, _DATASET, _HOME, "USD savings", nickname, "paper")


@pytest.mark.rule(13)
def test_a_write_passes_only_with_one_intended_account_and_no_other_change(
    monkeypatch,
):
    truth = _write_truth(monkeypatch, {_OLD}, {_OLD, _new()})
    assert truth.result({"account_number": "AC9"}, "success")
    assert not truth.result({}, "success")
    assert not truth.result({"account_number": "AC8"}, "success")

    two = _write_truth(monkeypatch, {_OLD}, {_OLD, _new(), (*_new()[:1], "x")})
    assert not two.result({"account_number": "AC9"}, "success")

    renamed = (*_OLD[:5], "Renamed", _OLD[6])
    touched = _write_truth(monkeypatch, {_OLD}, {renamed, _new()})
    assert not touched.result({"account_number": "AC9"}, "success")

    elsewhere = _write_truth(monkeypatch, {_OLD}, {_OLD, _new(member="NM000002")})
    assert not elsewhere.result({"account_number": "AC9"}, "success")


@pytest.mark.rule(13)
@pytest.mark.parametrize(
    ("column", "value"), [("status", "closed"), ("currency", "EUR")]
)
def test_account_comparison_includes_fields_outside_the_requested_output(
    tmp_path, monkeypatch, column, value
):
    import sqlite3

    path = tmp_path / "fixture.sqlite"
    monkeypatch.setattr(integrated, "DATABASE", path)
    with sqlite3.connect(path) as database:
        database.executescript(
            "CREATE TABLE accounts (dataset_id, member_id, product_id, account_number, "
            "nickname, statement_delivery, status, currency);"
            "CREATE TABLE members (id, dataset_id, member_number);"
            "CREATE TABLE products (id, dataset_id, name);"
            "CREATE TABLE datasets (id, name, institution_id);"
            "INSERT INTO accounts VALUES "
            "('d', 'm', 'p', 'a', 'n', 'paper', 'open', 'USD');"
            "INSERT INTO members VALUES ('m', 'd', 'member');"
            "INSERT INTO products VALUES ('p', 'd', 'savings');"
            "INSERT INTO datasets VALUES ('d', 'fixture', 'institution');"
        )
    before = integrated.every_account()
    with sqlite3.connect(path) as database:
        # The column comes only from this test's two fixed cases.
        database.execute(f"UPDATE accounts SET {column}=?", (value,))  # noqa: S608
    assert integrated.every_account() != before
    with sqlite3.connect(path) as database:
        database.execute(
            "INSERT INTO accounts VALUES "
            "('missing', 'm', 'p', 'b', 'n', 'paper', 'open', 'USD')"
        )
    assert len(integrated.every_account()) == 2


@pytest.mark.rule(13)
def test_a_stopped_write_may_leave_only_its_one_intended_change(monkeypatch):
    lost = _write_truth(monkeypatch, {_OLD}, {_OLD, _new()})
    assert lost.settled()
    nothing = _write_truth(monkeypatch, {_OLD}, {_OLD})
    assert nothing.settled()
    wrong = _write_truth(monkeypatch, {_OLD}, {_OLD, _new(nickname="Other")})
    assert not wrong.settled()


@pytest.mark.rule(13)
@pytest.mark.parametrize("commits", [0, 1, 2])
def test_discovery_counts_dispatches_even_when_the_application_deduplicates(
    pages, tmp_path, monkeypatch, commits
):
    import dataclasses

    from computeruse.loop import Ending

    truth = _write_truth(monkeypatch, {_OLD}, {_OLD, _new()})
    monkeypatch.setattr(integrated, "Truth", lambda *_: truth)
    task = dataclasses.replace(truth.task, outcomes=(), replays=())
    record = SimpleNamespace(
        complete=True, capability=object(), candidates=(), issues=(), artifact_issues=()
    )
    monkeypatch.setattr(integrated.DiscoveryTrace, "build", lambda *_a, **_k: record)
    monkeypatch.setattr(integrated, "save", lambda *_: None)
    monkeypatch.setattr(integrated, "FAULTS", {})
    monkeypatch.setattr(
        integrated,
        "ContextOracle",
        lambda *_: SimpleNamespace(
            check=lambda: {"operator": True, "institution": True}
        ),
    )
    with pages({"/": "<p>Receipt</p>", "/commit": "<p>Posted once</p>"}) as (
        surface,
        profile,
    ):
        monkeypatch.setattr(integrated, "load_profile", lambda *_: profile)
        monkeypatch.setattr(
            integrated,
            "open_session",
            lambda *_a, **_k: contextlib.nullcontext(surface),
        )

        def agent(*_args, **_kwargs):
            for _ in range(commits):
                surface._page.evaluate(
                    "fetch('/commit', {method:'POST'}).then(r=>r.text())"
                )
            return SimpleNamespace(
                ending=Ending.COMPLETED,
                outputs={"account_number": "AC9"},
                steps=commits,
                verification="loop",
                checks=(),
            )

        monkeypatch.setattr(integrated, "discover", agent)
        result = integrated.run_task(
            "white-label-responsive", surface._page.url, tmp_path, "unused", task
        )
    assert result["commits"] == commits
    assert result["matches_fixture"] is True
    assert result["passed"] is (commits == 1)


@pytest.mark.rule(13)
@pytest.mark.parametrize(
    ("commits", "fixture", "context"),
    [(1, True, True), (0, False, True), (0, True, False)],
)
def test_outcome_discovery_must_pass_independent_checks_before_adoption(
    tmp_path, monkeypatch, commits, fixture, context
):
    from computeruse.loop import Ending

    task = next(task for task in integrated.TASKS if task.writes)
    capability = cast("Any", object())
    result = SimpleNamespace(
        ending=Ending.COMPLETED, steps=1, outcome="record_not_found", outputs={}
    )
    monkeypatch.setattr(integrated, "case_goal", lambda *_: "Look for another record")
    monkeypatch.setattr(
        integrated, "Truth", lambda *_: SimpleNamespace(result=lambda *_: fixture)
    )
    monkeypatch.setattr(
        integrated,
        "discover_case",
        lambda *_: (result, None, {"operator": context}, None, commits),
    )
    monkeypatch.setattr(
        integrated, "adopt", lambda *_: pytest.fail("unsafe outcome adopted")
    )
    returned, summary = integrated.learn_outcome(
        "white-label-responsive",
        "http://localhost",
        tmp_path,
        cast("Any", None),
        "Look for a record",
        task,
        capability,
        "record_not_found",
        {},
        "unused",
    )
    assert returned is capability
    assert summary["learned"] is False
    assert summary["commits"] == commits


@pytest.mark.rule(13)
def test_the_file_channel_submits_a_typed_answer_to_the_run(tmp_path, capsys):
    from computeruse.control import Order
    from computeruse.escalation import Verdict
    from evaluation.operator import FileChannel

    got: list[Order] = []

    class Seen:
        def submit(self, order):
            got.append(order)
            return SimpleNamespace(verdict=Verdict.ACCEPTED, reason="")

    channel = FileChannel(tmp_path)
    channel.control = cast("Any", Seen())
    channel.status = cast("Any", SimpleNamespace(run="run-1", revision=4))
    private = "PRIVATE-OPERATOR-NOTE-7146"
    channel.submit(f"approve iv-2 {private}")
    assert [(o.run, o.revision, o.intervention) for o in got] == [("run-1", 4, "iv-2")]
    assert got[0].note == private
    channel.submit(f"gibberish {private}")
    assert len(got) == 1
    assert private not in capsys.readouterr().out


def test_inspection_saves_metadata_without_model_page_or_operator_content(
    tmp_path, capsys
):
    import base64

    import httpx

    from computeruse.control import Button, Offer, Status
    from computeruse.escalation import Ask, Command, Mode, Owner, State, Trigger
    from evaluation.inspect_run import Inspection, InspectionChannel

    private = "PRIVATE-INSPECTION-CONTENT-7146"
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        json={
            "model": private,
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": private},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,"
                            + base64.b64encode(private.encode()).decode(),
                        },
                    ],
                }
            ],
            "tools": [{"type": "function", "name": private}],
        },
    )
    original = request.content
    inspection = Inspection(tmp_path, 4399)
    inspection.request(request)
    inspection.response(
        httpx.Response(
            200,
            request=request,
            json={
                "id": private,
                "output": [
                    {
                        "type": "function_call",
                        "arguments": json.dumps({private: private}),
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 5},
            },
        )
    )
    status = Status(
        "run-1",
        Mode.DISCOVERY,
        3,
        State.AWAITING_APPROVAL,
        Owner.OPERATOR,
        4,
        Button(private, Command.RESUME, enabled=True),
        frozenset({Command.APPROVE, Command.RESUME}),
        30,
        0,
        notice=private,
        offer=Offer(
            "iv-1",
            Ask.APPROVAL,
            Trigger.RISKY_ACTION,
            private,
            private,
            private,
            private,
            private,
            30,
            30,
        ),
    )
    InspectionChannel(tmp_path).show(status)
    saved = b"".join(path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    assert private.encode() not in saved
    assert not list(tmp_path.glob("*.png"))
    assert request.content == original
    assert private in capsys.readouterr().out
    request_metadata = json.loads((tmp_path / "call-0001-request.json").read_text())
    response_metadata = json.loads((tmp_path / "call-0001-response.json").read_text())
    assert request_metadata["images"] == 1
    assert response_metadata["http_status"] == 200
    assert response_metadata["input_tokens"] == 7


@pytest.mark.parametrize(
    "unavailable",
    [
        "status.json",
        "statuses.jsonl",
        "replay-case-1.jsonl",
        "replay-case-1.human.jsonl",
        "browser-session",
    ],
)
def test_required_evaluation_storage_failure_is_distinct_from_browser_failure(
    tmp_path, monkeypatch, capsys, profile, unavailable
):
    from pathlib import Path

    from computeruse.budget import Budget
    from computeruse.control import Control
    from computeruse.escalation import Mode
    from computeruse.evidence import EvidenceError, run_logged
    from computeruse.journal import MemoryJournal
    from evaluation.inspect_run import InspectionChannel

    private = "PRIVATE-EVIDENCE-WRITE-9461"
    storage_failure = unavailable != "browser-session"
    original = Path.open

    def fail_evidence(path, *args, **kwargs):
        if path.name == unavailable:
            raise OSError(private)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_evidence)
    if not storage_failure:

        def browser_failure(*_args, **_kwargs):
            raise OSError(private)

        monkeypatch.setattr(integrated, "open_session", browser_failure)
    attempts = iter(("direct", "logged"))

    def operation():
        folder = tmp_path / next(attempts)
        folder.mkdir()
        if unavailable.startswith("replay-") or not storage_failure:
            monkeypatch.setattr(integrated, "Truth", lambda *_: object())
            task = integrated.TASKS[0]
            result = integrated.replay_case(
                "white-label-responsive",
                "http://localhost",
                folder,
                profile,
                cast("Any", object()),
                task,
                task.replays[0],
            )
            assert result == {"input": "case-1", "error": "OSError", "passed": False}
            return 0
        channel = InspectionChannel(folder)
        channel.started = True
        control = Control(
            mode=Mode.DISCOVERY, clock=time.monotonic, channels=(channel,)
        )
        assert control.begin(
            profile=profile,
            budget=Budget(profile.budgets, time.monotonic),
            journal=MemoryJournal(),
            context="fixture",
        )
        control.finish("completed")
        return 0

    if storage_failure:
        with pytest.raises(EvidenceError):
            operation()
    else:
        assert operation() == 0
    capsys.readouterr()
    log = tmp_path / "command.jsonl"
    expected = 2 if storage_failure else 0
    assert run_logged(operation, command="evaluation", log_path=log) == expected
    captured = capsys.readouterr()
    written = log.read_text() + captured.out + captured.err
    assert private not in written
    assert ("log_unavailable" in written) is storage_failure
    assert json.loads(log.read_text().splitlines()[-1]) == {
        "event": "CommandEnded",
        "code": expected,
    }


@pytest.mark.parametrize("fixture_matches", [True, False])
def test_live_evaluation_saves_fixture_verdicts_without_runtime_values(
    tmp_path, monkeypatch, profile, fixture_matches
):
    from computeruse.actions import AxLocator
    from computeruse.decider import CheckKind, ResultCheck, Task, TaskOutput, TaskRecord
    from computeruse.escalation import InterventionRequest, Trigger
    from computeruse.loop import CheckResult, Ending, RunResult, Verification
    from evaluation import live

    private = "PRIVATE-LIVE-EVALUATION-7146"
    result = RunResult(
        Ending.COMPLETED,
        2,
        private,
        outputs={private: private},
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.RESULT,
                    AxLocator("textbox", private),
                    private,
                    output=private,
                ),
                private,
                True,
                bound=private,
            ),
        ),
        task=Task(
            records=(TaskRecord(private, private),),
            outputs=(TaskOutput(private, private),),
        ),
    )

    def check(_page, received):
        assert received.outputs == {private: private}
        return [] if fixture_matches else [f"unexpected page value: {private}"]

    case = live.Case(
        app="evaluation",
        goal=private,
        entry="/members",
        verification="executor",
        handoffs=(Trigger.UNVERIFIED_RESULT.value,),
        records=((private.casefold(), ""),),
        check=check,
    )
    monkeypatch.setattr(live, "CASES", {"privacy-check": case})
    monkeypatch.setattr(
        live, "_evaluation_site", lambda: contextlib.nullcontext(("https://local", {}))
    )
    monkeypatch.setattr(live, "load_profile", lambda _: profile)
    monkeypatch.setattr(
        live, "open_session", lambda *_: contextlib.nullcontext(object())
    )
    monkeypatch.setattr(live, "LunaDecider", lambda **_: object())
    monkeypatch.setenv("OPENAI_API_KEY", "unused")

    def discovery(goal, *_args, escalator, **_kwargs):
        assert goal == private
        escalator.requests.append(
            InterventionRequest(
                Trigger.UNVERIFIED_RESULT,
                private,
                private,
                1,
                private,
                private,
                30,
                outputs={private: private},
                unverified=(private,),
            )
        )
        return result

    monkeypatch.setattr(live, "discover", discovery)
    summary = live.run_case("privacy-check", tmp_path)
    folder = tmp_path / "privacy-check" / "trial-1"
    saved = "".join(path.read_text() for path in folder.iterdir())
    assert summary["passed"] is fixture_matches
    assert private not in json.dumps(summary) + saved
    assert summary["output_count"] == 1
    assert summary["task"]["record_count"] == 1
    assert summary["fixture_checks_passed"] is fixture_matches


def test_a_visual_discovery_sees_only_screenshots():
    """--visual narrows discovery's observations, as the CLI's --visual does."""
    from computeruse.profile import ObservationMode, load_profile

    profile = load_profile(
        integrated.ROOT
        / "evaluation"
        / "sites"
        / "profiles"
        / "white-label-responsive.yaml"
    )
    seen = integrated._screens_only(profile)
    assert seen.perception.allowed_modes == (ObservationMode.VISUAL,)
    assert profile.perception.allowed_modes != seen.perception.allowed_modes


def test_every_canvas_fault_names_a_request_the_canvas_server_serves() -> None:
    # The canvas teller is our own site, so its server's routes are known. A
    # fault on a request it never makes would pass without ever firing.
    server = (integrated.ROOT / "evaluation" / "canvas" / "server.mjs").read_text()
    faults = [
        injection
        for (app, _), injections in integrated.FAULTS.items()
        if app == "canvas-teller"
        for injection in injections
    ]
    assert {injection.fault for injection in faults} >= {
        Fault.SLOW,
        Fault.SERVER_ERROR,
        Fault.LOST_REPLY,
    }
    for injection in faults:
        assert f"'{injection.method} {injection.path_ends_with}'" in server


@pytest.mark.rule(13)
def test_checked_in_capabilities_load_under_the_current_schema() -> None:
    """Every capability saved as evidence loads under the current schema."""
    from computeruse.capability import CapabilityError, load_capability

    evidence = integrated.ROOT / "evidence"
    paths = tuple(evidence.rglob("capability*.json")) if evidence.exists() else ()
    refused = []
    for path in paths:
        try:
            load_capability(path)
        except CapabilityError:
            refused.append(path.relative_to(integrated.ROOT))
    assert not refused, refused


def test_evaluation_sources_unpack_without_git_history_or_overwriting_edits(
    tmp_path, monkeypatch
) -> None:
    import zipfile

    from evaluation import sites

    monkeypatch.setattr(sites, "DESTINATION", tmp_path)
    monkeypatch.setattr(sites, "SOURCE", tmp_path / sites.SOURCE.name)
    with zipfile.ZipFile(sites.ARCHIVE) as archive:
        assert all(".git" not in Path(name).parts for name in archive.namelist())
    source = sites.unpack()
    assert not (source / ".git").exists()
    picker = (
        source / "environments/web-component-operations/src/client/components/picker.ts"
    )
    assert picker.is_file()
    picker.write_text("work in progress\n")
    assert sites.unpack() == source
    assert picker.read_text() == "work in progress\n"


def _canvas_services():
    from evaluation import sites

    return sites.SOURCE / "environments/database/financial/dist/services.js"


def _sites_installed() -> bool:
    from evaluation import sites

    return _canvas_services().exists() and sites.SEED.exists()


@pytest.mark.skipif(
    not _sites_installed() or shutil.which("node") is None,
    reason="the evaluation sites are not installed",
)
def test_the_canvas_teller_lists_an_account_as_soon_as_it_is_opened(tmp_path):
    from evaluation import sites

    database = tmp_path / "canvas.sqlite"
    shutil.copyfile(sites.SEED, database)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [
            "node",
            str(integrated.ROOT / "evaluation/canvas/server.mjs"),
            *("--database", str(database), "--services", str(_canvas_services())),
            *("--port", str(port)),
        ],
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            with (
                contextlib.suppress(OSError),
                socket.create_connection(("127.0.0.1", port), timeout=1),
            ):
                break
            assert server.poll() is None, "the canvas server exited"
            assert time.monotonic() < deadline, "the canvas server did not start"
            time.sleep(0.1)
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())

        def ask(path, body=None):
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=None if body is None else json.dumps(body).encode(),
                headers={"content-type": "application/json"},
            )
            with opener.open(request) as response:
                return json.load(response)

        ask("/api/signon", {"operator": "OP0002"})
        before = ask("/api/member", {"number": "NM000054"})
        form = {
            "member": "NM000054",
            "product": "USD savings",
            "nickname": "Holiday fund",
            "delivery": "paper",
        }
        key = ask("/api/open/review", form)["key"]
        opened = ask("/api/open/commit", {**form, "key": key})["account"]

        after = ask("/api/member", {"number": "NM000054"})
        assert after["openAccounts"] == before["openAccounts"] + 1
        assert opened in {account["number"] for account in after["accounts"]}
        assert ask("/api/accounts?query=&page=0")["items"][0]["number"] == opened
        shown = ask(f"/api/account?number={opened}")
        assert (shown["nickname"], shown["delivery"]) == ("Holiday fund", "paper")
    finally:
        server.terminate()
        server.wait(timeout=10)
