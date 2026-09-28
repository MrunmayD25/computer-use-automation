"""Shared text-matching rules for discovery and replay.

Rule 2 in RULES.md requires the verification loop, recorder, and replay to
interpret text the same way. This module defines that interpretation. It
imports nothing from the rest of the package, so every layer can use it.

A value is found as whole words. The text is split into words at spaces,
and each word loses the punctuation at its edges, such as brackets, commas,
and a closing full stop, and a possessive ending. The value's words must
then appear in a row. Any other character joins a value to its neighbour,
so ``12345`` is not in ``12345-01`` and ``$4.00`` is not in ``-$4.00``.

Painted text, read from a screenshot by recognition, differs in two ways
that the caller asks for by name. Case is ignored, and a colon also
separates words, because recognition drops the space after a label's colon.
A value with spaces may also be found with its spaces removed, because
recognition drops those too; a value without spaces never is, so two
separate identifiers are never merged into one.

A required state is not shown when the text also carries a negating word,
as in ``Not Approved``.

Examples
--------
>>> shows("Member 12345 (Riverside)", "12345", Match.CONTAINS)
True
>>> shows("Acct 12345-01", "12345", Match.CONTAINS)
False
>>> shows("Balance -$4.00", "$4.00", Match.CONTAINS)
False
>>> shows("Not Approved", "Approved", Match.CONTAINS, purpose=Purpose.REQUIREMENT)
False
>>> shows("Status:active", "Active", Match.CONTAINS, painted=True)
True
>>> shows("Plan: Goldplus", "Gold plus", Match.CONTAINS, painted=True)
True
"""

from __future__ import annotations

from enum import StrEnum

EDGE = "()[]{},.;:!?\"'"
"""Punctuation a word may carry at either end without changing the word."""

POSSESSIVES = ("'s", "\N{RIGHT SINGLE QUOTATION MARK}s")
"""Endings that make a possessive, which is not part of the word."""

NEGATIONS = frozenset({"not", "no", "non", "never", "none", "false"})
"""Words that deny a required state shown beside them."""


class Match(StrEnum):
    """How a check compares text: the whole text, or a run of whole words."""

    EQUALS = "equals"
    CONTAINS = "contains"


class Purpose(StrEnum):
    """What a check proves, which decides how strictly it compares.

    A record check shows which record the page is about, so it finds the
    identifier as whole words whatever its match says. A requirement check
    fails on a negated text. A result or a state check compares as its match
    says.
    """

    RESULT = "result"
    RECORD = "record"
    REQUIREMENT = "requirement"
    STATE = "state"


def spaced(text: str) -> str:
    """Return ``text`` with every run of white space made one space, trimmed."""
    return " ".join(text.split())


def same_identifier(first: str, second: str) -> bool:
    """Compare record identities without changing punctuation or internal signs."""
    return spaced(first).casefold() == spaced(second).casefold()


def holds(text: str, value: str) -> bool:
    """Exclude a value appearing anywhere in saved text, ignoring case.

    Privacy exclusion is separate from whole-word evidence matching.

    Examples
    --------
    >>> holds("badge status-active", "Active")
    True
    >>> holds("Search", "")
    False
    """
    return bool(value) and value.casefold() in text.casefold()


def extract(text: str, prefix: str, suffix: str) -> str | None:
    """Read the one whole-word value between exact, unique boundaries.

    The full text uses structured whitespace normalization. Boundaries keep
    their edge spaces, signs, and punctuation. Recognition needs its own
    explicit semantics and cannot use this structured extraction.
    """
    text = spaced(text)
    if not prefix and not suffix:
        return None
    if not text.startswith(prefix) or not text.endswith(suffix):
        return None
    if any(boundary and text.count(boundary) != 1 for boundary in (prefix, suffix)):
        return None
    end = len(text) - len(suffix)
    if end <= len(prefix):
        return None
    value = text[len(prefix) : end]
    if not value or value != value.strip():
        return None
    if positions(text, value) != [(len(prefix), end)]:
        return None
    return value


def word(token: str) -> str:
    """Return ``token`` without edge punctuation or a possessive ending.

    Examples
    --------
    >>> word("(U-7)"), word("12345's"), word("12,345")
    ('U-7', '12345', '12,345')
    """
    stripped = token.strip(EDGE)
    for ending in POSSESSIVES:
        if stripped.endswith(ending) and len(stripped) > len(ending):
            return stripped.removesuffix(ending)
    return stripped


def _words(text: str, *, painted: bool) -> list[tuple[int, int, str]]:
    """Return each word of ``text`` with where it starts and ends in ``text``.

    The positions are those of the word after its edge punctuation is taken
    off, so a caller can cut the text around a value exactly.
    """
    found: list[tuple[int, int, str]] = []
    start = None
    separators = (":",) if painted else ()
    for at, character in enumerate(text + " "):
        if character.isspace() or character in separators:
            if start is not None:
                token = text[start:at]
                bare = word(token)
                if bare:
                    lead = len(token) - len(token.lstrip(EDGE))
                    found.append((start + lead, start + lead + len(bare), bare))
                start = None
        elif start is None:
            start = at
    return found


def positions(text: str, value: str, *, painted: bool = False) -> list[tuple[int, int]]:
    """Return the start and end of each place ``value`` appears as whole words.

    Examples
    --------
    >>> positions("OP2 and OP22, OP2", "OP2")
    [(0, 3), (14, 17)]
    >>> positions("Signed on as Sam Lee (U-7)", "U-7")
    [(22, 25)]
    >>> positions("anything", "")
    []
    """
    fold = str.casefold if painted else str
    words = _words(text, painted=painted)
    wanted = [fold(item) for _, _, item in _words(value, painted=painted)]
    found: list[tuple[int, int]] = []
    if not wanted:
        return found
    size = len(wanted)
    found.extend(
        (words[at][0], words[at + size - 1][1])
        for at in range(len(words) - size + 1)
        if [fold(item) for _, _, item in words[at : at + size]] == wanted
    )
    if painted and not found and size > 1:
        joined = "".join(wanted)
        found = [(start, end) for start, end, item in words if fold(item) == joined]
    return found


def spelled(text: str, value: str) -> str | None:
    """Return how ``text`` writes ``value`` as whole words, ignoring case.

    Spacing inside ``value`` does not matter, because words are compared one
    by one. The answer is the text's own spelling, so a value taken from a
    request keeps the case the request wrote. None when ``value`` is absent,
    or when ``text`` writes it in two different ways.

    Examples
    --------
    >>> spelled("Open it with the nickname Blue jar.", "blue  JAR")
    'Blue jar'
    >>> spelled("member NM54 and member nm54", "NM54") is None
    True
    >>> spelled("member NM540", "NM54") is None
    True
    """
    words = _words(text, painted=False)
    wanted = [item.casefold() for _, _, item in _words(value, painted=False)]
    size = len(wanted)
    found = {
        text[words[at][0] : words[at + size - 1][1]]
        for at in range(len(words) - size + 1)
        if size and [item.casefold() for _, _, item in words[at : at + size]] == wanted
    }
    return found.pop() if len(found) == 1 else None


def contains(
    text: str, value: str, *, painted: bool = False, once: bool = False
) -> bool:
    """Report whether ``value`` appears in ``text`` as whole words.

    With ``once``, it must appear exactly one time.

    Examples
    --------
    >>> contains("Member 12345 (Riverside)", "12345")
    True
    >>> contains("12345 and 12345", "12345", once=True)
    False
    """
    count = len(positions(text, value, painted=painted))
    return count == 1 if once else count > 0


def negated(text: str) -> bool:
    """Report whether ``text`` carries a word that denies a state.

    Examples
    --------
    >>> negated("Not Approved"), negated("Approved: No"), negated("Region: West")
    (True, True, False)
    """
    return any(word(token).casefold() in NEGATIONS for token in text.split())


def shows(
    text: str,
    value: str,
    match: Match | str,
    *,
    purpose: Purpose | str = Purpose.STATE,
    once: bool = False,
    painted: bool = False,
) -> bool:
    """Report whether ``text`` shows ``value`` the way a check of ``purpose`` needs.

    ``EQUALS`` compares the whole text after spacing is made uniform, except
    for a record check, which always finds its identifier as whole words.
    On painted text, ``EQUALS`` also holds when the value is the last words
    of a labelled line, such as ``Status: active``.

    Examples
    --------
    >>> shows("  Active ", "Active", Match.EQUALS)
    True
    >>> shows("Member 12345 (Riverside)", "12345", Match.EQUALS, purpose=Purpose.RECORD)
    True
    >>> shows("Status: active", "active", Match.EQUALS, painted=True)
    True
    >>> shows("Status: inactive", "active", Match.EQUALS, painted=True)
    False
    """
    text, value = spaced(text), spaced(value)
    if not value:
        return False
    if Match(match) is Match.EQUALS and Purpose(purpose) is not Purpose.RECORD:
        if painted:
            found = positions(text, value, painted=True)
            end = len(text.rstrip(EDGE))
            passed = text.casefold() == value.casefold() or bool(
                found and found[-1][1] >= end
            )
        else:
            passed = text == value
    else:
        passed = contains(text, value, painted=painted, once=once)
    if (
        passed
        and Purpose(purpose) is Purpose.REQUIREMENT
        and negated(text)
        and not negated(value)
    ):
        # A required state that is itself negative, such as "Not required",
        # is shown by its own words; any other negation denies it.
        return False
    return passed


def label_before(text: str, value: str) -> str | None:
    """Return the words of a painted line before the value it ends with, or None.

    Recognition may drop a space, so the value is found without spacing or
    case. The label keeps the line's own spelling. A sentence's closing
    punctuation after the value, which ``word`` trims from a word's edge,
    still ends the line with the value.

    Examples
    --------
    >>> label_before("Order number: O-2001", "O-2001")
    'Order number:'
    >>> label_before("Ordernumber:O-2001", "O-2001")
    'Ordernumber:'
    >>> label_before("No order with number O-2001.", "O-2001")
    'No order with number'
    >>> label_before("O-2001", "O-2001") is None
    True
    """
    line = "".join(text.split()).casefold()
    wanted = "".join(value.split()).casefold()
    if wanted and not line.endswith(wanted):
        line = line.rstrip(EDGE)
    if not wanted or not line.endswith(wanted) or line == wanted:
        return None
    keep = len(line) - len(wanted)
    taken = 0
    for index, character in enumerate(text):
        if not character.isspace():
            taken += 1
        if taken == keep:
            label = text[: index + 1].strip()
            return label or None
    return None
