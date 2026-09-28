"""Completion must prove the change and context the goal requires."""

from pathlib import Path

import pytest
from fakes import Screen, ScriptedDecider, ScriptedEscalator, ScriptedSurface

from computeruse.actions import (
    ActionKind,
    AxLocator,
    ObservationMode,
    ObservationRequest,
)
from computeruse.decider import (
    CheckKind,
    Finish,
    Match,
    Observe,
    ResultCheck,
    Task,
    TaskRecord,
    TaskRequirement,
)
from computeruse.escalation import Handoff, HandoffOutcome, Trigger
from computeruse.journal import MemoryJournal
from computeruse.loop import (
    CheckResult,
    Ending,
    Verification,
    _compared,
    _uncovered,
    discover,
)
from computeruse.profile import load_profile


def test_reading_an_identifier_does_not_complete_an_approval():
    profile = load_profile(Path("evaluation/profile.yaml"))
    surface = ScriptedSurface(
        [Screen("http://127.0.0.1:8787/members", controls=(("text", "10001"),))],
        extracts=["10001"],
    )
    check = ResultCheck(CheckKind.RECORD, AxLocator("text", "10001"), "10001")
    task = Task(records=(TaskRecord("member", "10001"),))
    result = discover(
        "Approve member 10001.",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [
                Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                Finish({}, checks=(check,)),
            ],
            tasks=[task],
        ),
        journal=MemoryJournal(),
        clock=lambda: 0,
        escalator=ScriptedEscalator([]),
    )
    assert result.ending is not Ending.COMPLETED
    assert not surface.acted


@pytest.mark.parametrize("bound", ["", "10002"])
def test_required_state_cannot_come_from_another_record(bound):
    task = Task(
        records=(TaskRecord("member", "10001"),),
        requirements=(TaskRequirement("approval", "Approved", "member"),),
        changes=True,
    )
    check = ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("status", "Approved"),
        "Approved",
        requirement="approval",
    )
    checked = CheckResult(check, "Approved", True, bound=bound)
    assert "required state approval is not established" in _uncovered(task, (checked,))
    valid = CheckResult(check, "Approved", True, bound="10001")
    assert _uncovered(task, (valid,)) == []


def test_required_operator_context_is_checked_even_without_outputs():
    task = Task(requirements=(TaskRequirement("operator", "OP0002"),))
    assert _uncovered(task, ()) == ["required state operator is not established"]
    check = ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("status", "OP0002"),
        "OP0002",
        requirement="operator",
    )
    assert _uncovered(task, (CheckResult(check, "OP0002", True),)) == []


def test_a_required_state_is_read_only_from_application_values():
    profile = load_profile(Path("evaluation/profile.yaml"))
    surface = ScriptedSurface(
        [Screen("http://127.0.0.1:8787/members", controls=(("text", "OP0002"),))],
        extracts=["OP0002"],
    )
    check = ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("text", "OP0002"),
        "OP0002",
        requirement="operator",
    )
    discover(
        "Sign on as operator OP0002.",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [
                Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                Finish({}, checks=(check,)),
            ],
            tasks=[Task(requirements=(TaskRequirement("operator", "OP0002"),))],
        ),
        journal=MemoryJournal(),
        clock=lambda: 0,
        escalator=ScriptedEscalator([]),
    )
    reads = [
        expect
        for action, expect in zip(surface.acted, surface.expected, strict=True)
        if action.kind is ActionKind.READ
    ]
    # A field the run changed would show what the run typed, so the surface
    # must refuse one that anybody changed in the document showing it.
    assert reads
    assert all(expect is not None and expect.committed for expect in reads)


@pytest.mark.rule(2)
@pytest.mark.parametrize("shown", ["Not Approved", "Approved: No", "No, Approved"])
def test_a_negated_state_does_not_contain_the_required_state(shown):
    check = ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("status", "Approved"),
        "Approved",
        match=Match.CONTAINS,
        requirement="approval",
    )
    assert not _compared(check, shown).passed
    assert _compared(check, "Status: Approved").passed


def _context_task(changes: bool = False) -> Task:
    return Task(
        requirements=(
            TaskRequirement("operator", "OP0002", context=True),
            TaskRequirement("institution", "Acme", context=True),
        ),
        changes=changes,
    )


def _operator_check() -> ResultCheck:
    return ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("status", "OP0002"),
        "OP0002",
        requirement="operator",
    )


def test_context_the_goal_names_is_never_taken_from_the_goal():
    shown = (CheckResult(_operator_check(), "OP0002", True),)
    # The operator was checked. Nothing checked the institution.
    assert _uncovered(_context_task(), shown) == [
        "required state institution is not established"
    ]


def test_a_claim_that_leaves_out_the_operator_check_does_not_cover_it():
    institution = ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("status", "Acme"),
        "Acme",
        requirement="institution",
    )
    # Found in a postback run: another operator was signed on, so the goal's
    # operator appeared nowhere, and a claim without an operator check
    # completed without switching.
    shown = (CheckResult(institution, "Acme", True),)
    assert _uncovered(_context_task(), shown) == [
        "required state operator is not established"
    ]


def test_a_state_the_task_must_reach_is_never_taken_from_the_goal():
    task = Task(
        records=(TaskRecord("member", "10001"),),
        requirements=(TaskRequirement("approval", "Approved", "member"),),
        changes=True,
    )
    assert "required state approval is not established" in _uncovered(task, ())


def test_a_change_needs_a_required_state_besides_its_context():
    from computeruse.loop import _requirement_problem

    problem = _requirement_problem(_context_task(changes=True))
    assert problem == "a task that changes data needs a required end state"


def _unproved_context_run(handoffs: list[Handoff]):
    profile = load_profile(Path("evaluation/profile.yaml"))
    surface = ScriptedSurface(
        [Screen("http://127.0.0.1:8787/members", controls=(("status", "OP0002"),))],
        extracts=["OP0002"] * 8,
    )
    # The model keeps claiming with the operator check alone.
    claims = [Finish({}, checks=(_operator_check(),)) for _ in range(6)]
    escalator = ScriptedEscalator(handoffs)
    result = discover(
        "As operator OP0002 at Acme, confirm you are signed on.",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [Observe(ObservationRequest(ObservationMode.STRUCTURED)), *claims],
            tasks=[_context_task()],
        ),
        journal=MemoryJournal(),
        clock=lambda: 0,
        escalator=escalator,
    )
    return result, escalator


def test_unproved_context_goes_to_a_person_to_confirm():
    result, escalator = _unproved_context_run([Handoff(HandoffOutcome.APPROVED)])

    assert [request.trigger for request in escalator.requests] == [
        Trigger.UNVERIFIED_RESULT
    ]
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.verification is Verification.PERSON


def test_unproved_context_never_completes_on_its_own():
    result, _ = _unproved_context_run([])

    assert result.ending is not Ending.COMPLETED
