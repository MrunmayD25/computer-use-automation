"""A replay's ending is reported in plain words, and shown until it is closed."""

import time
from pathlib import Path

import pytest

from computeruse.actions import ObservationStatus
from computeruse.capability import ActionNode, loads
from computeruse.capability_cli import _show_result, report
from computeruse.control import Control, Order
from computeruse.diagnostics import CheckShape, FailureEvidence, Snapshot
from computeruse.escalation import Ask, Command, Trigger, Via
from computeruse.profile import load_profile
from computeruse.replay import (
    Actor,
    Answer,
    Attempt,
    Delivery,
    HistoryEntry,
    Reason,
    ReplayResult,
    Status,
)
from computeruse.replay import Ask as ReplayAsk

CAPABILITY = loads(
    Path("examples/capabilities/member_transfer.synthetic.json").read_text()
)
ACTIONS = [node for node in CAPABILITY.nodes if isinstance(node, ActionNode)]


def test_a_success_reports_its_outputs_and_the_steps_performed():
    result = ReplayResult(
        Status.SUCCEEDED,
        Reason.COMPLETED,
        "done",
        outputs={"balance": "$10.00"},
        attempts=(
            Attempt(ACTIONS[0].node_id, 1, Delivery.COMPLETED, Actor.AUTOMATION),
        ),
    )
    lines = report(CAPABILITY, result)
    assert lines[0] == "Replay succeeded."
    assert "Why: every step ran and the success checks held" in lines
    assert "Steps performed: 1" in lines
    assert "balance: $10.00" in lines
    assert not any(line.startswith("Stopped at") for line in lines)


def test_a_business_outcome_is_named():
    result = ReplayResult(
        Status.OUTCOME, Reason.BUSINESS_OUTCOME, "absent", outcome="member_not_found"
    )
    lines = report(CAPABILITY, result)
    assert "Outcome: member not found" in lines


@pytest.mark.rule(13)
def test_a_failure_names_the_step_the_checks_the_page_and_the_person():
    node = next(node for node in ACTIONS if node.verify)
    number = [n.node_id for n in ACTIONS].index(node.node_id) + 1
    checks = tuple(CheckShape("present", False) for _ in node.verify)
    result = ReplayResult(
        Status.TERMINATED,
        Reason.OPERATOR_TERMINATED,
        node.node_id,
        attempts=(
            Attempt(ACTIONS[0].node_id, 1, Delivery.COMPLETED, Actor.PERSON),
            Attempt(node.node_id, 1, Delivery.UNCERTAIN, Actor.AUTOMATION),
        ),
        history=(
            HistoryEntry(
                node.node_id,
                "iv-1",
                ReplayAsk.PERSON,
                Reason.VERIFICATION_FAILED,
                Answer.TERMINATED,
                "",
            ),
        ),
        diagnostic=FailureEvidence(
            3,
            node.node_id,
            "operator_terminated",
            (),
            Snapshot(ObservationStatus.COMPLETE, 9, 12, True, False, False, ()),
            checks,
        ),
    )
    lines = report(CAPABILITY, result)
    assert lines[0] == "Replay ended: a person terminated it."
    assert any(
        line.startswith(f"Stopped at: step {number} of {len(ACTIONS)}, ")
        for line in lines
    )
    assert all(
        line.endswith("(did not hold)")
        for line in lines
        if line.startswith("Expected: ")
    )
    assert len([line for line in lines if line.startswith("Expected: ")]) == len(
        node.verify
    )
    assert "Page seen: complete reading, 9 of 12 controls visible, no dialog" in lines
    assert "Steps performed: 1, 1 of them by a person" in lines
    assert "Uncertain changes: 1, never sent again automatically" in lines
    assert any(line.startswith("A person was asked at step") for line in lines)


class Person:
    """A channel where a person answers each request with one command."""

    def __init__(self, command: Command) -> None:
        self.command = command
        self.control: Control | None = None
        self.offers = []

    def attach(self, control) -> None:
        self.control = control

    def listening(self) -> bool:
        return True

    def show(self, status) -> None:
        offer = status.offer
        if offer is None or self.control is None or self.offers:
            return
        self.offers.append(offer)
        self.control.submit(
            Order(
                self.command, status.run, status.revision, offer.intervention, Via.PANEL
            )
        )


def test_the_result_stays_in_the_window_until_the_person_closes_it():
    person = Person(Command.RESUME)
    profile = load_profile(Path("evaluation/profile.yaml"))
    _show_result([person], profile, CAPABILITY, ["Replay failed.", "Why: x"])
    offer = person.offers[0]
    assert offer.trigger is Trigger.REPLAY_RESULT
    assert offer.ask is Ask.PERSON
    assert offer.reason == "Replay failed.\nWhy: x"


def test_nothing_is_shown_when_nobody_can_see_it():
    class Closed(Person):
        def listening(self) -> bool:
            return False

    person = Closed(Command.RESUME)
    profile = load_profile(Path("evaluation/profile.yaml"))
    started = time.monotonic()
    _show_result([person], profile, CAPABILITY, ["Replay failed."])
    assert person.offers == []
    assert time.monotonic() - started < 1


def test_a_discovery_that_saves_nothing_says_why_and_waits_to_be_closed():
    from computeruse.cli import _report_incomplete
    from computeruse.recorder import Gap, Recording, RecordingIssue

    recorded = Recording(
        None,
        (
            RecordingIssue(
                Gap.UNREPRESENTABLE_CHECK, 11, "a shows check: its text is new"
            ),
        ),
    )
    lines = _report_incomplete(recorded)
    assert lines[0] == "The discovery finished, but no capability was saved."
    assert (
        "Why: step 11, unrepresentable check: a shows check: its text is new" in lines
    )

    person = Person(Command.RESUME)
    profile = load_profile(Path("evaluation/profile.yaml"))
    _show_result([person], profile, None, lines, discovery=True)
    offer = person.offers[0]
    assert offer.trigger is Trigger.DISCOVERY_RESULT
    assert offer.reason == "\n".join(lines)
