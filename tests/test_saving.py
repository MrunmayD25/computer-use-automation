"""A person confirms website text after a discovery, through the run's channels."""

import time
from pathlib import Path

import pytest

from computeruse.confirm import Decision, decide
from computeruse.control import Control, Order
from computeruse.escalation import Ask, Command, Trigger, Via
from computeruse.profile import load_profile
from computeruse.recorder import Candidate, Place
from computeruse.saving import asking
from computeruse.words import Confirmed, Word

LABEL = Candidate("Go", frozenset({Place.CONTROL}))
CELL = Candidate("Jordan Smith", frozenset({Place.RECORD}))


class Person:
    """A channel where a person answers the one request with one command."""

    def __init__(self, command: Command, note: str | None = None) -> None:
        self.command = command
        self.note = note
        self.control: Control | None = None
        self.offers = []

    def attach(self, control) -> None:
        self.control = control

    def listening(self) -> bool:
        return True

    def show(self, status) -> None:
        offer = status.offer
        if offer is None or self.offers or self.control is None:
            return
        self.offers.append(offer)
        self.control.submit(
            Order(
                self.command,
                status.run,
                status.revision,
                offer.intervention if self.command is not Command.TERMINATE else "",
                Via.PANEL,
                self.note,
            )
        )


def _ask(person: Person, compared: list[dict[str, str]] | None = None):
    profile = load_profile(Path("evaluation/profile.yaml"))

    def compare_with(second):
        if compared is not None:
            compared.append(dict(second))
        return Decision((Word("Jordan Smith", Confirmed.COMPARISON),), ())

    return asking([person], profile, "look up the member", compare_with, time.monotonic)


def test_approving_confirms_every_listed_text_as_a_persons():
    person = Person(Command.APPROVE)
    words = _ask(person)((LABEL, CELL), "the model did not judge these")
    assert words == (
        Word("Go", Confirmed.PERSON),
        Word("Jordan Smith", Confirmed.PERSON),
    )
    offer = person.offers[0]
    assert offer.trigger is Trigger.UNCONFIRMED_TEXT
    assert offer.ask is Ask.APPROVAL
    assert "'Jordan Smith' (record)" in offer.reason


def test_a_second_record_in_the_resume_note_confirms_by_comparison():
    compared: list[dict[str, str]] = []
    person = Person(Command.RESUME, "compare with member_id=10002")
    words = _ask(person, compared)((CELL,), "unconfirmed")
    assert compared == [{"member_id": "10002"}]
    assert words == (Word("Jordan Smith", Confirmed.COMPARISON),)


@pytest.mark.parametrize(
    ("command", "note"),
    [(Command.RESUME, "looks fine"), (Command.TERMINATE, None)],
)
def test_resuming_without_a_record_or_terminating_confirms_nothing(command, note):
    assert _ask(Person(command, note))((LABEL,), "unconfirmed") == ()


def test_nothing_is_saved_when_the_person_does_not_confirm_it():
    decided = decide(
        (LABEL, CELL),
        compared=None,
        classifier=None,
        veto=None,
        ask=_ask(Person(Command.TERMINATE)),
    )
    assert not decided.complete
    assert decided.unconfirmed == (LABEL, CELL)


SYNTHETIC = Path("examples/capabilities/member_transfer.synthetic.json")


def _draft():
    from computeruse.capability import loads

    return loads(SYNTHETIC.read_text())


@pytest.mark.parametrize(
    ("command", "approved"),
    [(Command.APPROVE, True), (Command.RESUME, False), (Command.TERMINATE, False)],
)
def test_only_an_approval_in_the_window_approves_the_capability(command, approved):
    from computeruse.saving import review, summary

    person = Person(command)
    profile = load_profile(Path("evaluation/profile.yaml"))
    draft = _draft()
    assert review([person], profile, "goal", draft, time.monotonic) is approved
    offer = person.offers[0]
    assert offer.trigger is Trigger.CAPABILITY_REVIEW
    assert offer.ask is Ask.APPROVAL
    assert offer.reason == summary(draft)


@pytest.mark.parametrize("command", [Command.APPROVE, Command.RESUME])
def test_discovery_writes_the_approved_copy_only_on_approval(tmp_path, command):
    import argparse

    from computeruse.capability import Review, load_capability
    from computeruse.cli import _review_draft

    target = tmp_path / "capability.json"
    args = argparse.Namespace(save_approved=target, goal="goal")
    profile = load_profile(Path("evaluation/profile.yaml"))
    assert _review_draft(args, profile, _draft(), [Person(command)]) == 0
    if command is Command.APPROVE:
        assert load_capability(target).provenance.review is Review.REVIEWED
    else:
        assert not target.exists()


def test_without_a_path_nothing_is_asked():
    import argparse

    from computeruse.cli import _review_draft

    profile = load_profile(Path("evaluation/profile.yaml"))
    person = Person(Command.APPROVE)
    none = argparse.Namespace(save_approved=None, goal="goal")
    assert _review_draft(none, profile, _draft(), [person]) == 0
    assert person.offers == []


@pytest.mark.parametrize(
    ("command", "allowed"),
    [(Command.APPROVE, True), (Command.RESUME, False), (Command.TERMINATE, False)],
)
def test_the_checks_run_only_when_a_person_allows_them(command, allowed):
    from computeruse.saving import consent

    person = Person(command)
    profile = load_profile(Path("evaluation/profile.yaml"))
    checks = ["Comparison: repeat the draft", "Outcome check: a missing member"]
    assert consent([person], profile, "goal", checks, time.monotonic) is allowed
    offer = person.offers[0]
    assert offer.trigger is Trigger.RUN_CHECKS
    assert offer.reason == "\n".join(checks)


def test_a_failed_comparison_warns_the_reviewer_and_stops_the_checks():
    from computeruse.cli import _compared

    findings: list[str] = []
    failed = Decision((), (CELL,), "", "failed", "the page did not show it")
    assert _compared(failed, findings) is True
    assert findings[0].startswith("Warning: the comparison with a second member")
    stopped = Decision((Word("Go", Confirmed.COMPARISON),), (), "", "stopped", "x")
    assert _compared(stopped, findings) is False
    assert findings[1] == "Comparison with a second member: x; 1 texts confirmed."
    assert _compared(Decision((), ()), findings) is False
    assert len(findings) == 2


def test_a_write_discovery_can_skip_the_outcome_checks():
    import argparse

    from computeruse.cli import _outcome_cases

    profile = load_profile(
        Path("evaluation/sites/profiles/white-label-responsive.yaml")
    )
    assert _outcome_cases(argparse.Namespace(no_outcome_checks=True), profile) == {}
    assert "record_not_found" in _outcome_cases(argparse.Namespace(), profile)
