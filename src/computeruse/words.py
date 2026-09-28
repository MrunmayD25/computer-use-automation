"""Website text that capabilities may save, stored in one list per site.

A capability finds controls again by their text, but a saved file must never
contain customer data. Text is saved only after confirmation that it belongs
to the website. The list lives beside the site's profile, as
``<profile>.safe-text.yaml``, and records how each word was confirmed:

- ``operator``: a person wrote it into the file.
- ``comparison``: it read the same for a second test record.
- ``person``: a person approved it at the end of a discovery.
- ``model``: the model judged it an interface label, on a control outside any
  record's row. Only such labels may be confirmed this way.
- ``outcome``: a learned business outcome's check read it. For "not found"
  the record does not exist, so the text cannot name a customer. Any other
  outcome is about a record that exists, so the model must also judge the
  text the website's own before it is added.

Known private values override confirmation and remove matching saved words.
The list is not policy and grants no access: it only decides which texts a
recording may write. An absent file confirms nothing.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

import yaml

from computeruse.matching import holds

MAX_WORD = 200
"""The longest text a list may hold; a longer one is not an interface label."""

HEADER = (
    "# Website text confirmed safe to save in this site's capabilities.\n"
    "# Each word records how it was confirmed. A person may add, correct, or\n"
    "# remove words; the product removes words containing known private values.\n"
)


class Confirmed(StrEnum):
    """How a word came to be on a website's list."""

    OPERATOR = "operator"
    COMPARISON = "comparison"
    PERSON = "person"
    MODEL = "model"
    OUTCOME = "outcome"


@dataclasses.dataclass(frozen=True, slots=True)
class Word:
    """One confirmed text, and how it was confirmed."""

    text: str
    confirmed_by: Confirmed


class WordsError(ValueError):
    """A word list that cannot be read. Its message never quotes the file."""


def words_path(profile: Path) -> Path:
    """Return the word-list path next to ``profile``.

    Examples
    --------
    >>> words_path(Path("sites/profiles/classic.yaml")).as_posix()
    'sites/profiles/classic.safe-text.yaml'
    """
    return profile.with_name(profile.stem + ".safe-text.yaml")


def load_words(path: Path) -> tuple[Word, ...]:
    """Read a website's confirmed words, or return none if no list exists.

    A plain text entry is a word a person wrote. A mapping names the text and
    how it was confirmed. A malformed list is refused rather than read in
    part, so no word is saved on a guess.
    """
    if not path.exists():
        return ()
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        raise WordsError("the word list is not valid YAML") from None
    if not isinstance(document, dict) or set(document) != {"words"}:
        raise WordsError("a word list must declare exactly words")
    entries = document["words"] or []
    if not isinstance(entries, list):
        raise WordsError("words must be a list")
    words: list[Word] = []
    for entry in entries:
        if isinstance(entry, str):
            word = Word(entry, Confirmed.OPERATOR)
        elif isinstance(entry, dict) and set(entry) == {"text", "confirmed_by"}:
            try:
                word = Word(str(entry["text"]), Confirmed(entry["confirmed_by"]))
            except ValueError:
                raise WordsError("a word names an unknown confirmation") from None
        else:
            raise WordsError("each word is a text or a text and its confirmation")
        if not word.text.strip() or len(word.text) > MAX_WORD:
            raise WordsError("each word must be non-empty and short")
        words.append(word)
    if len({word.text for word in words}) != len(words):
        raise WordsError("a word may appear once")
    return tuple(words)


def add_words(
    path: Path, added: Iterable[Word], *, excluded: Iterable[str] = ()
) -> tuple[Word, ...]:
    """Add confirmed words after removing text containing known private values.

    The file is replaced in one step, so a failure leaves the old list whole.
    Returns the safe words that were new. Exclusions override every source of
    confirmation, including words already saved by an operator.
    """
    previous = load_words(path)
    private = tuple(value for value in excluded if value)
    known = tuple(
        word
        for word in previous
        if not any(holds(word.text, value) for value in private)
    )
    held = {word.text for word in known}
    new: list[Word] = []
    for word in added:
        if any(holds(word.text, value) for value in private):
            continue
        if word.text not in held:
            held.add(word.text)
            new.append(word)
    if not new and known == previous:
        return ()
    listed = [
        word.text
        if word.confirmed_by is Confirmed.OPERATOR
        else {"text": word.text, "confirmed_by": word.confirmed_by.value}
        for word in (*known, *new)
    ]
    body = HEADER + yaml.safe_dump(
        {"words": listed}, allow_unicode=True, sort_keys=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(body)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return tuple(new)
