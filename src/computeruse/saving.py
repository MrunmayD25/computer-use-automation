"""Ask a person to confirm and review a discovered capability before saving it.

The discovery's control ends with the run. Saving opens a short-lived control
on the same panel and terminal channels. It may raise two requests in order.

- Unconfirmed text. The request lists the texts still unconfirmed. The
  person may approve them, resume with a second test record in the note,
  which compares instead, or terminate. Anything but an approval or a
  successful comparison confirms nothing, and nothing unconfirmed is saved.
- Review. Once everything else is settled, the request summarizes the
  capability: its inputs, outputs, steps, and endings. An approval marks it
  reviewed, so replay runs it. Anything else leaves it a draft.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from computeruse.budget import Budget, Clock
from computeruse.capability import (
    Capability,
    ResultKind,
    ResultNode,
)
from computeruse.confirm import AskPerson, Decision
from computeruse.control import Channel, Control
from computeruse.describe import steps
from computeruse.escalation import (
    Ask,
    HandoffOutcome,
    InterventionRequest,
    Mode,
    Trigger,
)
from computeruse.journal import MemoryJournal
from computeruse.profile import Profile
from computeruse.recorder import Candidate
from computeruse.words import Confirmed, Word

MAX_SHOWN = 20
"""The most texts one request lists; a longer list is summarized by count."""

MAX_STEPS = 12
"""The most steps a review lists; the window cannot scroll."""


def record_in(note: str | None) -> dict[str, str]:
    """Read the ``NAME=VALUE`` pairs a person wrote in a resume note.

    Words without an equals sign are the person's own remarks and are ignored.

    Examples
    --------
    >>> record_in("check with member_id=10002 please")
    {'member_id': '10002'}
    >>> record_in(None)
    {}
    """
    pairs: dict[str, str] = {}
    for word in (note or "").split():
        name, separator, value = word.partition("=")
        if separator and name.isidentifier() and value:
            pairs[name] = value
    return pairs


def listing(candidates: tuple[Candidate, ...]) -> str:
    """List each text with the places where the capability uses it."""
    shown = [
        f"{item.text!r} ({', '.join(sorted(place.value for place in item.places))})"
        for item in candidates[:MAX_SHOWN]
    ]
    more = len(candidates) - len(shown)
    return "; ".join(shown) + (f"; and {more} more" if more > 0 else "")


def asking(
    channels: Sequence[Channel],
    profile: Profile,
    goal: str,
    compare_with: Callable[[Mapping[str, str]], Decision],
    clock: Clock,
) -> AskPerson:
    """Return how a person confirms texts through the run's channels.

    ``compare_with`` replays the draft for a second test record a person
    names, and returns what that comparison confirmed.
    """

    def ask(left: tuple[Candidate, ...], note: str) -> tuple[Word, ...]:
        control = Control(
            mode=Mode.DISCOVERY, clock=clock, channels=channels, purpose="saving"
        )
        if not control.begin(
            profile=profile,
            budget=Budget(profile.budgets, clock),
            journal=MemoryJournal(),
            context="saving the discovered capability",
        ):
            return ()
        reason = (
            f"{note}. Confirm these texts are the website's own, never a "
            f"person's data: {listing(left)}"
        )
        handoff = control.intervene(
            InterventionRequest(
                trigger=Trigger.UNCONFIRMED_TEXT,
                goal=goal,
                profile_id=profile.profile_id,
                step=0,
                route="",
                reason=reason,
                timeout_s=profile.escalation.handoff_timeout_s,
                unverified=tuple(item.text for item in left),
                ask=Ask.APPROVAL,
            )
        )
        control.finish(
            "completed"
            if handoff.outcome in {HandoffOutcome.APPROVED, HandoffOutcome.RESUMED}
            else "terminated"
        )
        if handoff.outcome is HandoffOutcome.APPROVED:
            return tuple(Word(item.text, Confirmed.PERSON) for item in left)
        second = record_in(handoff.operator_note)
        if handoff.outcome is HandoffOutcome.RESUMED and second:
            return compare_with(second).confirmed
        return ()

    return ask


def summary(capability: Capability) -> str:
    """Describe a capability for a person deciding whether to approve it.

    It names types and structure, never a value a run supplied: inputs by
    name, texts only as the capability saved them.

    Examples
    --------
    >>> from pathlib import Path
    >>> from computeruse.capability import loads
    >>> text = Path("examples/capabilities/member_transfer.synthetic.json").read_text()
    >>> print(summary(loads(text)))
    Inputs: member_id (digits), amount (decimal)
    Outputs: balance
    Steps (7):
      1. type member_id into textbox 'Member number'
      2. click button 'Search'
      3. read dd into balance
      4. click link 'Transfer'
      5. type amount into textbox 'Amount'
      6. type teller_pin into textbox 'Teller PIN'
      7. click button 'Submit transfer', with approval on every run
    Ends: outcome member_not_found, success
    """
    listed = [text for _, text in steps(capability)]
    endings = sorted(
        {
            node.result.value
            if node.result is not ResultKind.OUTCOME
            else f"outcome {node.outcome}"
            for node in capability.nodes
            if isinstance(node, ResultNode)
        }
    )
    inputs = ", ".join(f"{f.name} ({f.type.value})" for f in capability.inputs)
    outputs = ", ".join(f.name for f in capability.outputs)
    lines = [
        f"Inputs: {inputs or 'none'}",
        f"Outputs: {outputs or 'none'}",
        f"Steps ({len(listed)}):",
        *(f"  {index}. {text}" for index, text in enumerate(listed[:MAX_STEPS], 1)),
    ]
    if len(listed) > MAX_STEPS:
        lines.append(f"  and {len(listed) - MAX_STEPS} more")
    lines.append("Ends: " + (", ".join(endings) or "none"))
    return "\n".join(lines)


def consent(
    channels: Sequence[Channel],
    profile: Profile,
    goal: str,
    checks: Sequence[str],
    clock: Clock,
) -> bool:
    """Ask a person whether saving may run ``checks``; True only on approval.

    The checks act on the application again, in new windows, so nothing runs
    after a discovery until a person says so.
    """
    control = Control(
        mode=Mode.DISCOVERY, clock=clock, channels=channels, purpose="checks"
    )
    if not control.begin(
        profile=profile,
        budget=Budget(profile.budgets, clock),
        journal=MemoryJournal(),
        context="asking to run the checks",
    ):
        return False
    handoff = control.intervene(
        InterventionRequest(
            trigger=Trigger.RUN_CHECKS,
            goal=goal,
            profile_id=profile.profile_id,
            step=0,
            route="",
            reason="\n".join(checks),
            timeout_s=profile.escalation.handoff_timeout_s,
            ask=Ask.APPROVAL,
        )
    )
    allowed = handoff.outcome is HandoffOutcome.APPROVED
    control.finish("completed" if allowed else "terminated")
    return allowed


def review(
    channels: Sequence[Channel],
    profile: Profile,
    goal: str,
    capability: Capability,
    clock: Clock,
    findings: Sequence[str] = (),
) -> bool:
    """Ask a person to approve ``capability`` for replay; True only on approval.

    ``findings`` say what the checks found, and come before the summary.
    """
    control = Control(
        mode=Mode.DISCOVERY, clock=clock, channels=channels, purpose="review"
    )
    if not control.begin(
        profile=profile,
        budget=Budget(profile.budgets, clock),
        journal=MemoryJournal(),
        context="reviewing the discovered capability",
    ):
        return False
    handoff = control.intervene(
        InterventionRequest(
            trigger=Trigger.CAPABILITY_REVIEW,
            goal=goal,
            profile_id=profile.profile_id,
            step=0,
            route="",
            reason="\n".join([*findings, summary(capability)]),
            timeout_s=profile.escalation.handoff_timeout_s,
            ask=Ask.APPROVAL,
        )
    )
    approved = handoff.outcome is HandoffOutcome.APPROVED
    control.finish("completed" if approved else "terminated")
    return approved
