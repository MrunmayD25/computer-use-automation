"""Discovery and replay decide "this text shows that value" the same way.

Rule 2 in RULES.md. The loop judges a claim with ``_compared``, and replay
judges the saved check with ``text_matches``. Both use ``matching``. These
tests run the same cases through both paths and require one answer.
"""

from __future__ import annotations

import dataclasses

import pytest

from computeruse import matching
from computeruse.actions import AxLocator, AxNode, Point, Scope, ScopeKind, ScreenTarget
from computeruse.capability import (
    LocatorForm,
    Match,
    Purpose,
    ScopeSpec,
    StructuralTarget,
    constant,
    ref,
    text_matches,
)
from computeruse.capability import RefKind as Kind
from computeruse.decider import CheckKind, ResultCheck
from computeruse.decider import Match as CheckMatch
from computeruse.loop import _compared
from computeruse.recording import _saved
from computeruse.retarget import _holds

CASES = [
    # shown, expected, match, purpose, holds
    ("Member 12345 (Riverside)", "12345", "contains", "state", True),
    ("Acct 12345-01", "12345", "contains", "state", False),
    ("A-12345-B", "12345", "contains", "record", False),
    ("Balance -$4.00", "$4.00", "contains", "result", False),
    ("Balance $14.00", "$4.00", "contains", "result", False),
    ("Total 12,345", "12", "contains", "result", False),
    ("Report member 12345's balance.", "12345", "contains", "state", True),
    ("Signed on as Sam Lee (U-7) · Sign out", "U-7", "contains", "requirement", True),
    ("Not Approved", "Approved", "contains", "requirement", False),
    ("Approved: No", "Approved", "contains", "requirement", False),
    ("Not required", "Not required", "equals", "requirement", True),
    ("  Active  ", "Active", "equals", "state", True),
    ("Active", "active", "equals", "state", False),
    ("Member 12345 (Riverside)", "12345", "equals", "record", True),
    ("Member 12345 (Riverside)", "12345", "equals", "state", False),
]


@pytest.mark.rule(2)
@pytest.mark.parametrize(("shown", "expected", "match", "purpose", "holds"), CASES)
def test_discovery_and_replay_give_one_answer(shown, expected, match, purpose, holds):
    check = ResultCheck(
        CheckKind(purpose),
        AxLocator("text", "Anything"),
        expected,
        match=CheckMatch(match),
        requirement="state" if purpose == "requirement" else "",
        output="value" if purpose == "result" else "",
    )
    discovered = _compared(check, shown).passed
    replayed = text_matches(shown, expected, Match(match), purpose=Purpose(purpose))
    assert discovered is replayed is holds


@pytest.mark.rule(2)
def test_replay_contains_refuses_what_discovery_refuses() -> None:
    # Graduated from tests/test_known_gaps.py (finding 1).
    for shown, wanted in (("12345-01", "12345"), ("-$4.00", "$4.00")):
        assert not matching.contains(shown, wanted)
        assert not text_matches(shown, wanted, Match.CONTAINS)


@pytest.mark.rule(2)
def test_a_painted_requirement_refuses_a_negated_state() -> None:
    # Graduated from tests/test_known_gaps.py (finding 3).
    structured = ResultCheck(
        CheckKind.REQUIREMENT,
        AxLocator("status", "Status"),
        "Approved",
        match=CheckMatch.CONTAINS,
        requirement="state",
    )
    painted = dataclasses.replace(
        structured, target=ScreenTarget("capture", Point(10, 10))
    )
    assert not _compared(structured, "Status: Not Approved").passed
    assert not _compared(painted, "Status: Not Approved").passed


@pytest.mark.rule(2)
def test_painted_text_never_merges_two_identifiers() -> None:
    # Recognition drops spaces, so a value with spaces may be found without
    # them. A value without spaces is never found across a word break.
    assert matching.contains("Plan:Goldplus", "Gold plus", painted=True)
    assert not matching.contains("Member NM1 23", "NM123", painted=True)
    assert matching.contains("Status:active", "Active", painted=True)
    assert not matching.contains("Status: 12345-01", "12345", painted=True)


@pytest.mark.rule(3)
def test_a_record_check_is_saved_as_discovery_judged_it() -> None:
    # Finding 8: discovery finds a record's identifier as whole words even
    # when the check says equals, so the saved check says contains.
    check = ResultCheck(CheckKind.RECORD, AxLocator("cell", "M-1"), "M-1")
    assert _saved(check) == (Match.CONTAINS, Purpose.RECORD)
    state = ResultCheck(CheckKind.STATE, AxLocator("cell", "Open"), "Open")
    assert _saved(state) == (Match.EQUALS, Purpose.STATE)


def _row_cell(names: tuple[str, ...], columns: tuple[int, ...]) -> AxNode:
    return AxNode(
        "button",
        "Open",
        scope=Scope(ScopeKind.ROW, names[0]),
        row_names=names,
        row_columns=columns,
    )


@pytest.mark.rule(7)
def test_a_row_alias_is_found_only_in_its_own_column() -> None:
    # Finding 5: a row found by the cell that holds the member is bound to
    # that cell's column, so a note elsewhere that mentions the member is no
    # match.
    target = StructuralTarget(
        "t1",
        "/list",
        LocatorForm.ACCESSIBILITY,
        "button",
        constant("Open"),
        "",
        None,
        None,
        (),
        ScopeSpec(ScopeKind.ROW, ref(Kind.INPUT, "member"), column=2),
    )

    def text(reference):
        return "12345" if reference.kind is Kind.INPUT else reference.value

    own = _row_cell(("E-1", "Jordan Smith 12345"), (1, 2))
    other = _row_cell(("E-2", "Replaces 12345"), (1, 3))
    assert _holds(target, own, text)
    assert not _holds(target, other, text)
