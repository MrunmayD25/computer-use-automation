"""A discovery takes its inputs from the goal, or asks a person for them."""

import time
from pathlib import Path

import pytest

from computeruse.capability import Field, ValueType
from computeruse.control import Control, Order
from computeruse.escalation import Ask, Command, Trigger, Via
from computeruse.inputs import asking, fill
from computeruse.profile import load_profile

MEMBER = Field("member_id", ValueType.TEXT, True, 1, 40, ())
NICKNAME = Field("nickname", ValueType.TEXT, True, 1, 40, ())
DELIVERY = Field("delivery", ValueType.CHOICE, True, 1, 20, ("paper", "electronic"))
NOTE = Field("note", ValueType.TEXT, False, 1, 40, ())
GOAL = "Open an account for member 12345 with the nickname Rainy day."


class _Proposing:
    """A model that proposes fixed values, and records what it was asked."""

    def __init__(self, **values: str | None) -> None:
        self.values = values
        self.asked: list[tuple[str, tuple[Field, ...]]] = []

    def __call__(self, goal: str, fields: tuple[Field, ...]) -> dict[str, str | None]:
        self.asked.append((goal, fields))
        return self.values


class _Answering:
    """A person who gives each note in turn, and records each question."""

    def __init__(self, *notes: str | None) -> None:
        self.left = list(notes)
        self.questions: list[str] = []

    def __call__(self, text: str) -> str | None:
        self.questions.append(text)
        return self.left.pop(0)


@pytest.mark.rule(9)
def test_values_the_goal_writes_are_kept_as_the_goal_spells_them():
    propose = _Proposing(member_id="12345", nickname="rainy  DAY", delivery="Paper")
    goal = GOAL + " Send statements on paper."
    filled = fill(goal, (MEMBER, NICKNAME, DELIVERY), propose=propose, ask=None)
    assert filled.inputs == {
        "member_id": "12345",
        "nickname": "Rainy day",
        "delivery": "paper",
    }
    assert filled.goal == goal
    assert not filled.missing
    assert propose.asked == [(goal, (MEMBER, NICKNAME, DELIVERY))]


@pytest.mark.rule(9)
def test_a_value_the_goal_does_not_write_is_asked_of_a_person():
    # The model invents a member the goal never names, and gives no nickname.
    propose = _Proposing(member_id="99999", nickname=None)
    ask = _Answering("12345", "Rainy day")
    filled = fill("Open an account.", (MEMBER, NICKNAME), propose=propose, ask=ask)
    assert filled.inputs == {"member_id": "12345", "nickname": "Rainy day"}
    # Every later check that a value appears in the goal holds for it too.
    assert filled.goal == "Open an account. member_id: 12345. nickname: Rainy day."
    assert ask.questions[0].startswith("The goal gives no value for member_id.")
    assert "Type: text, 1 to 40 characters." in ask.questions[0]


def test_a_choice_outside_the_allowed_values_is_asked_with_the_choices():
    ask = _Answering("Electronic")
    filled = fill(GOAL, (DELIVERY,), propose=_Proposing(delivery="email"), ask=ask)
    assert filled.inputs == {"delivery": "electronic"}
    assert "Allowed values: paper, electronic." in ask.questions[0]


@pytest.mark.rule(9)
def test_a_choice_the_goal_words_differently_is_added_to_the_goal():
    goal = GOAL + " Send statements by post."
    filled = fill(goal, (DELIVERY,), propose=_Proposing(delivery="paper"), ask=None)
    assert filled.inputs == {"delivery": "paper"}
    assert filled.goal == goal + " delivery: paper."


def test_three_answers_that_do_not_fit_end_the_fill_without_repeating_them():
    ask = _Answering("post", "mail", "fax")
    filled = fill(GOAL, (DELIVERY,), propose=_Proposing(), ask=ask)
    assert filled.missing == "delivery"
    assert filled.inputs == {}
    assert len(ask.questions) == 3
    assert ask.questions[1].startswith("That value does not fit delivery.")
    assert not any(
        answer in question
        for question in ask.questions
        for answer in ("post", "mail", "fax")
    )


def test_without_a_person_a_missing_required_input_ends_the_fill():
    propose = _Proposing(member_id="12345")
    filled = fill(GOAL, (MEMBER, NICKNAME), propose=propose, ask=None)
    assert filled.missing == "nickname"


def test_an_optional_input_the_goal_does_not_name_is_left_out():
    ask = _Answering()
    filled = fill(GOAL, (MEMBER, NOTE), propose=_Proposing(member_id="12345"), ask=ask)
    assert filled.inputs == {"member_id": "12345"}
    assert ask.questions == []


class Person:
    """A channel where a person answers each request with one command."""

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
        if offer is None or self.control is None:
            return
        if any(seen.intervention == offer.intervention for seen in self.offers):
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


def _ask(person: Person):
    profile = load_profile(Path("evaluation/profile.yaml"))
    return asking([person], profile, "open an account", time.monotonic)


def test_a_person_answers_in_the_resume_note_of_a_request_for_help():
    person = Person(Command.RESUME, "paper")
    assert _ask(person)("The goal gives no value for delivery.") == "paper"
    offer = person.offers[0]
    assert offer.trigger is Trigger.MISSING_USER_INPUT
    assert offer.ask is Ask.PERSON
    assert offer.reason == "The goal gives no value for delivery."


def test_terminating_a_question_supplies_nothing():
    assert _ask(Person(Command.TERMINATE))("The goal gives no value.") is None
