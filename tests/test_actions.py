"""Checks for the vocabulary a run records."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from computeruse.actions import Action, AxLocator, SecretRef
from computeruse.profile import ActionKind

FIELD = AxLocator("textbox", "Member ID")


def test_locator_occurrence_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="occurrence"):
        AxLocator("textbox", "Member ID", occurrence=-1)


def test_locator_allows_an_empty_name_for_unlabelled_controls() -> None:
    assert AxLocator("textbox", "").name == ""


def test_secret_reference_must_name_something() -> None:
    with pytest.raises(ValueError, match="must name a declared secret"):
        SecretRef("")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (ActionKind.CLICK, "click requires target"),
        (ActionKind.TYPE, "type requires target"),
        (ActionKind.NAVIGATE, "navigate requires destination"),
    ],
)
def test_action_requires_the_fields_its_kind_needs(
    kind: ActionKind, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Action(kind)


def test_action_requires_a_value_where_the_kind_takes_one() -> None:
    with pytest.raises(ValueError, match="type requires value"):
        Action(ActionKind.TYPE, FIELD)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Action(ActionKind.OBSERVE, target=FIELD),
            "observe does not take target",
        ),
        (
            lambda: Action(ActionKind.CLICK, target=FIELD, value="x"),
            "click does not take value",
        ),
        (
            lambda: Action(
                ActionKind.NAVIGATE, destination="https://h.test/", value="x"
            ),
            "navigate does not take value",
        ),
    ],
)
def test_action_rejects_fields_its_kind_does_not_use(
    build: Callable[[], Action], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build()
