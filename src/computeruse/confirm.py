"""Decide which unconfirmed texts a discovered capability may save.

A recording made in collecting mode holds its draft in memory with the texts
nobody has confirmed yet. Each is confirmed in one of these ways, in order:

1. By comparison. When the profile or the operator names a second test
   record, the draft replays for it without a model. A replay that succeeds
   end to end read every required text again for a different record. Text
   allowed to differ during comparison stays unconfirmed. Rebuilding must
   retain every observed check, so an unconfirmed check prevents export.
   A replay that does not succeed confirms nothing.
   Given the run's control, the replay asks a person for a step the
   capability leaves to one. A person's step saves no page text, only the
   checks around it, and every automated step still reads its texts for the
   second record, so the proof holds. With nobody to answer, a step that
   needs a person ends the comparison unconfirmed.
   Two records can share data by chance, such as a name or a status, so
   the model then sees the compared texts and may take any away as data.
   It can never confirm a text the comparison did not.
2. By the model, only when no second record is named. It may confirm a text
   used solely as an interface control's label outside any record's row;
   every other text, and every text it calls risky, is left for a person.
3. By a person, who sees only what is still unconfirmed.

Every confirmed text is added to the website's word list with how it was
confirmed. The capability is then built again from the run with only
confirmed words. Any required text that stayed unconfirmed leaves the
capability incomplete, including checks on delivered steps.
The recorder has already refused any text that holds an input, an output, or
a remembered value, so none of these ways can let one through.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Protocol

from computeruse.budget import Clock
from computeruse.capability import (
    ActionNode,
    Capability,
    CheckNode,
    HumanNode,
    Purpose,
    RefKind,
    ResultNode,
    Shows,
)
from computeruse.control import Control
from computeruse.profile import Profile
from computeruse.recorder import Candidate
from computeruse.replay import (
    REASON_TEXT,
    MemoryReplayLog,
    ReplayLog,
    Status,
    replay,
)
from computeruse.surface import Surface
from computeruse.words import Confirmed, Word


class Classifier(Protocol):
    """Judges which texts are interface labels rather than a record's data."""

    def classify_words(self, texts: tuple[str, ...]) -> frozenset[str]:
        """Return the texts it judges to be the website's own labels."""
        ...


class Veto(Protocol):
    """Reject compared texts that could contain personal or record data."""

    def keep_words(self, texts: tuple[str, ...]) -> frozenset[str]:
        """Return the texts it judges to be the website's own."""
        ...


type AskPerson = Callable[[tuple[Candidate, ...], str], tuple[Word, ...]]
"""Show unconfirmed texts with a reason and return the confirmed words.

A person may approve texts, confirmed by a person, or name a second test
record whose comparison confirms them, confirmed by comparison.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """Which texts were confirmed, how, and what still blocks saving."""

    confirmed: tuple[Word, ...]
    unconfirmed: tuple[Candidate, ...]
    note: str = ""
    ended: str = ""
    """The comparison replay status, or empty when no comparison ran."""
    why: str = ""
    """A plain-language reason the comparison replay ended."""

    @property
    def complete(self) -> bool:
        """Report whether every candidate text was confirmed."""
        return not self.unconfirmed


def compare(
    capability: Capability,
    candidates: tuple[Candidate, ...],
    second: Mapping[str, str],
    discovered: Mapping[str, str],
    *,
    profile: Profile,
    surface: Surface,
    clock: Clock,
    sleep: Callable[[float], None] | None = None,
    log: ReplayLog | None = None,
    control: Control,
) -> Decision:
    """Replay the draft for a second record and confirm texts it read again.

    The second record must name every input the capability takes and differ
    from the discovered one in every input that names a record, or it proves
    nothing: changing an unrelated option leaves the same customer. Only a
    text the replay met on the page, on a control it resolved and used or in
    a check that held, and that read the same, is confirmed (rule 8). A text
    on a step or branch the replay never reached stays unconfirmed. The
    required control routes a step left to a person through the run's
    channels. An unattended request confirms nothing.
    """
    names = {field.name for field in capability.inputs}
    if set(second) != names:
        return Decision((), candidates, "the second record names other inputs")
    records = _record_inputs(capability) or names
    if any(second[name] == discovered.get(name) for name in records):
        return Decision(
            (), candidates, "the second record repeats an input that names a record"
        )
    result = replay(
        capability,
        second,
        profile=profile,
        surface=surface,
        control=control,
        log=log or MemoryReplayLog(),
        clock=clock,
        sleep=sleep,
        accept_draft=True,
        probing=frozenset(item.text for item in candidates),
        stop_before_change=True,
    )
    ended = result.status.value
    why = REASON_TEXT.get(result.reason, result.reason.value.replace("_", " "))
    if result.status not in {Status.SUCCEEDED, Status.STOPPED}:
        return Decision(
            (),
            candidates,
            f"the replay for the second record ended {ended}",
            ended,
            why,
        )
    words = tuple(
        Word(item.text, Confirmed.COMPARISON)
        for item in candidates
        if item.text in result.seen and item.text not in result.differed
    )
    held = {word.text for word in words}
    left = tuple(item for item in candidates if item.text not in held)
    unseen = [item for item in left if item.text not in result.differed]
    note = (
        "these read differently for the second record"
        if len(unseen) < len(left)
        else ""
    )
    if unseen:
        note = (note + "; " if note else "") + "the replay never met some of these"
    return Decision(words, left, note, ended, why)


def _record_inputs(capability: Capability) -> set[str]:
    """Return the inputs that name a record.

    These are the values record links compare, and the values of record
    checks. A painted screen has no row or container to link, so its record
    is named only by a check whose purpose is the record.
    """
    specs = [
        spec
        for node in capability.nodes
        if isinstance(node, ActionNode)
        for spec in (node.record, node.result_record)
        if spec is not None
    ]
    shows = [
        condition
        for node in capability.nodes
        for condition in _conditions(node)
        if isinstance(condition, Shows)
    ]
    specs.extend(check.record for check in shows if check.record is not None)
    values = [spec.value for spec in specs]
    values.extend(check.value for check in shows if check.purpose is Purpose.RECORD)
    return {value.name for value in values if value.kind is RefKind.INPUT}


def _conditions(node: object) -> tuple[object, ...]:
    """Return every check a node carries, on its steps, edges, and results."""
    if isinstance(node, ActionNode):
        edges = tuple(item for edge in node.transitions for item in edge.when)
        return (*node.requires, *node.verify, *edges)
    if isinstance(node, CheckNode | HumanNode):
        return tuple(item for edge in node.transitions for item in edge.when)
    if isinstance(node, ResultNode):
        return node.checks
    return ()


def judge(candidates: tuple[Candidate, ...], classifier: Classifier) -> Decision:
    """Let the model confirm interface labels and leave every other text.

    Only a text used solely as a control's label outside a record's row is
    shown to the model. What it calls risky, and every text that sits in a
    record's data, stays unconfirmed for a person.
    """
    labels = tuple(item.text for item in candidates if item.interface)
    approved = classifier.classify_words(labels) if labels else frozenset()
    confirmed = tuple(
        Word(item.text, Confirmed.MODEL)
        for item in candidates
        if item.interface and item.text in approved
    )
    held = {word.text for word in confirmed}
    return Decision(confirmed, tuple(i for i in candidates if i.text not in held))


def decide(
    candidates: tuple[Candidate, ...],
    *,
    compared: Decision | None,
    classifier: Classifier | None,
    veto: Veto | None,
    ask: AskPerson | None,
) -> Decision:
    """Combine the comparison, the model, and a person into one decision.

    With a second record, its comparison is the only automatic confirmation:
    a text that did not read the same for it is never given to the model.
    ``veto`` then sees what the comparison confirmed and may take texts away.
    A person may still approve what remains, told why it remains.
    """
    if not candidates:
        return Decision((), ())
    if compared is not None:
        confirmed, left, note = compared.confirmed, compared.unconfirmed, compared.note
        if veto is not None and confirmed:
            kept = veto.keep_words(tuple(word.text for word in confirmed))
            taken = {word.text for word in confirmed if word.text not in kept}
            if taken:
                confirmed = tuple(w for w in confirmed if w.text not in taken)
                left = (*left, *(item for item in candidates if item.text in taken))
                note = (
                    "the model judged that these could be a person's or a record's data"
                )
    elif classifier is not None:
        judged = judge(candidates, classifier)
        confirmed, left, note = judged.confirmed, judged.unconfirmed, ""
        if left:
            note = "the model did not judge these to be interface labels"
    else:
        confirmed, left, note = (), candidates, "no second record and no model"
    if left and ask is not None:
        asked = {item.text for item in left}
        # Only texts that were asked about can come back confirmed.
        added = tuple(word for word in ask(left, note) if word.text in asked)
        held = {word.text for word in added}
        confirmed = (*confirmed, *added)
        left = tuple(item for item in left if item.text not in held)
    # Preserve the comparison outcome for the review.
    ended, why = (compared.ended, compared.why) if compared else ("", "")
    return Decision(tuple(confirmed), tuple(left), note, ended, why)
