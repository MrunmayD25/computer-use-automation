"""The invocation values a discovery takes from its goal, or asks a person for.

An export contract names a capability's inputs and types them. Before a
discovery opens anything, the model reads the goal and proposes a value for
each input. This module accepts a proposal only after validating it.

- A choice must be one of the contract's allowed values. The goal may word
  it differently, because mapping words to an allowed value is the one
  judgement the model is asked for.
- Any other value must appear in the goal as whole words. Case and spacing
  may differ, and the value kept is the goal's own spelling, so the recorder
  finds it on the page exactly as the model typed it (rule 9).

A required input without a value that holds is asked of a person in the
control window, one question per input, with its type and allowed values.
The person answers in the resume note. A value the goal does not spell, a
person's answer or a choice the goal words differently, is added to the goal
the run works from, so every later check that a value appears in the goal
holds for it too. A question never repeats an answer, and the
control that asks keeps its journal in memory, so no value is written down.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

from computeruse import matching
from computeruse.budget import Budget, Clock
from computeruse.capability import Field, ValueType
from computeruse.control import Channel, Control
from computeruse.escalation import (
    Ask,
    HandoffOutcome,
    InterventionRequest,
    Mode,
    Trigger,
)
from computeruse.journal import MemoryJournal
from computeruse.profile import Profile

ATTEMPTS = 3
"""How many answers a person may give for one input before the run ends."""

Propose = Callable[[str, tuple[Field, ...]], Mapping[str, str | None]]
"""Return a mapping from each field name to its proposed value or None."""

Answer = Callable[[str], str | None]
"""Ask one question and return the person's note if they resume."""


@dataclasses.dataclass(frozen=True)
class Filled:
    """The values a discovery runs with, and the goal it works from.

    ``missing`` names the first required input nobody supplied; the run
    must not start while it is set.
    """

    inputs: dict[str, str]
    goal: str
    missing: str = ""


def kept(goal: str, field: Field, value: str | None) -> str | None:
    """Return the value to keep for ``field``, or None when it does not hold.

    Examples
    --------
    >>> choice = Field("delivery", ValueType.CHOICE, True, 1, 20, ("paper", "email"))
    >>> kept("send statements by post", choice, "Paper")
    'paper'
    >>> text = Field("nickname", ValueType.TEXT, True, 1, 40, ())
    >>> kept("with the nickname Blue jar", text, "blue JAR")
    'Blue jar'
    >>> kept("with the nickname Blue jar", text, "Sunny day") is None
    True
    """
    if value is None:
        return None
    if field.type is ValueType.CHOICE:
        return _chosen(field, value)
    written = matching.spelled(goal, value)
    return written if written is not None and field.accepts(written) else None


def answered(field: Field, note: str | None) -> str | None:
    """Return a person's answer for ``field`` when it has the field's type."""
    value = (note or "").strip()
    if field.type is ValueType.CHOICE:
        return _chosen(field, value)
    return value if value and field.accepts(value) else None


def _chosen(field: Field, value: str) -> str | None:
    same = [item for item in field.choices if item.casefold() == value.casefold()]
    return same[0] if len(same) == 1 and field.accepts(same[0]) else None


def question(field: Field, *, again: bool = False) -> str:
    """Describe what ``field`` needs, for a person answering in the note.

    Examples
    --------
    >>> delivery = Field("delivery", ValueType.CHOICE, True, 1, 20, ("paper",))
    >>> question(delivery).split(". ")[:3]
    ['The goal gives no value for delivery', 'Type: choice', 'Allowed values: paper']
    """
    lead = (
        f"That value does not fit {field.name}. "
        if again
        else f"The goal gives no value for {field.name}. "
    )
    return f"{lead}{_described(field)} {_instruction(field)}"


def _described(field: Field) -> str:
    match field.type:
        case ValueType.CHOICE:
            return f"Type: choice. Allowed values: {', '.join(field.choices)}."
        case ValueType.DIGITS:
            return "Type: digits only; leading zeros are kept."
        case ValueType.INTEGER:
            return "Type: a whole number, such as 25 or -3."
        case ValueType.DECIMAL:
            return "Type: a number, such as 25.00."
        case ValueType.BOOLEAN:
            return "Type: true or false."
        case _:
            return f"Type: text, {field.min_length} to {field.max_length} characters."


def _instruction(field: Field) -> str:
    if field.type is ValueType.CHOICE:
        return "Type one allowed value in Note, then press Done, continue."
    return "Type the value in Note, then press Done, continue."


def fill(
    goal: str,
    fields: Sequence[Field],
    *,
    propose: Propose,
    ask: Answer | None,
    attempts: int = ATTEMPTS,
) -> Filled:
    """Take each field's value from ``goal``, asking a person for what is missing.

    ``propose`` is asked once, about every field. ``ask`` is None when no
    person can answer, and then a missing required input ends the fill.
    """
    fields = tuple(fields)
    proposed = propose(goal, fields) if fields else {}
    values: dict[str, str] = {}
    supplied: list[str] = []
    for field in fields:
        value = kept(goal, field, proposed.get(field.name))
        if value is None and field.required:
            value = _asked(field, ask, attempts)
            if value is None:
                return Filled(values, goal, missing=field.name)
        if value is None:
            continue
        values[field.name] = value
        if matching.spelled(goal, value) is None:
            supplied.append(f"{field.name}: {value}.")
    worked = " ".join((goal, *supplied)) if supplied else goal
    return Filled(values, worked)


def _asked(field: Field, ask: Answer | None, attempts: int) -> str | None:
    if ask is None:
        return None
    for attempt in range(attempts):
        note = ask(question(field, again=attempt > 0))
        if note is None:
            return None
        value = answered(field, note)
        if value is not None:
            return value
    return None


def asking(
    channels: Sequence[Channel], profile: Profile, goal: str, clock: Clock
) -> Answer:
    """Return how a person answers input questions through the run's channels.

    Each question opens a short control of its own, as saving does, and
    raises one request for help. A resume carries the answer in its note.
    Terminating, or a request nobody answers in time, supplies nothing.
    """

    def ask(text: str) -> str | None:
        control = Control(
            mode=Mode.DISCOVERY, clock=clock, channels=channels, purpose="inputs"
        )
        if not control.begin(
            profile=profile,
            budget=Budget(profile.budgets, clock),
            journal=MemoryJournal(),
            context="filling in the capability's inputs",
        ):
            return None
        handoff = control.intervene(
            InterventionRequest(
                trigger=Trigger.MISSING_USER_INPUT,
                goal=goal,
                profile_id=profile.profile_id,
                step=0,
                route="",
                reason=text,
                timeout_s=profile.escalation.handoff_timeout_s,
                ask=Ask.PERSON,
            )
        )
        resumed = handoff.outcome is HandoffOutcome.RESUMED
        control.finish("completed" if resumed else "terminated")
        return handoff.operator_note if resumed else None

    return ask
