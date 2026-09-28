"""Bringing a human into a run that cannot safely continue alone.

Two kinds of situation reach a person: an action the operator declared risky,
which always escalates, and a run that cannot proceed on its own. Both pause
the same live session rather than starting a new one, so whoever takes over
sees the state the automation was actually in and hands the same session back.

The reason is a closed enum rather than a sentence. An operator console needs
to route "a person must type a one-time code" differently from "the model
cannot tell two buttons apart", and a free-text reason cannot be routed,
counted, or recorded without carrying whatever the model wrote into it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from computeruse.actions import Action, ObservationMode, PageInfo


class Trigger(StrEnum):
    """The reason the run offers control to a person.

    ``INSUFFICIENT_OBSERVATION`` is the only one the loop will try to answer
    itself, by spending the profile's alternate observation allowance before
    handing over. The rest describe situations another look cannot fix: a
    value nobody supplied, a login, an operator-declared risk, two controls
    that cannot be told apart, or a run that has stopped making progress.

    ``RECORD_EVIDENCE_REQUIRED`` asks a person to perform a step the operator
    requires record evidence for, when the run cannot supply that evidence.
    An approval does not perform it. ``UNVERIFIED_RESULT`` asks a person to
    confirm a result the executor could not check against the page.
    ``NEW_WINDOW`` hands the session to a person because a window or tab
    opened that the run does not operate. Handing the session back is not an
    approval of anything. ``AMBIGUOUS_TASK`` asks a person to confirm how the
    run understood the goal, before the run starts. ``UNCONFIRMED_TEXT`` asks
    a person, after a discovery, to confirm website text a capability would
    save, or to name a second test record that confirms it.
    ``CAPABILITY_REVIEW`` shows a person the saved capability once saving is
    done, and an approval marks it reviewed for replay. ``REPLAY_RESULT``
    shows how a finished replay ended, and waits for the person to close it.
    ``RUN_CHECKS`` asks a person, after a discovery, whether saving may run
    its checks against the application: the comparison and outcome cases.
    ``DISCOVERY_RESULT`` shows why a discovery saved no capability, and waits
    for the person to close it.
    """

    INSUFFICIENT_OBSERVATION = "insufficient_observation"
    AMBIGUOUS_TARGET = "ambiguous_target"
    MISSING_USER_INPUT = "missing_user_input"
    AUTHENTICATION_REQUIRED = "authentication_required"
    RISKY_ACTION = "risky_action"
    NO_PROGRESS = "no_progress"
    DELIVERY_UNCERTAIN = "delivery_uncertain"
    RECORD_EVIDENCE_REQUIRED = "record_evidence_required"
    UNVERIFIED_RESULT = "unverified_result"
    NEW_WINDOW = "new_window"
    AMBIGUOUS_TASK = "ambiguous_task"
    UNCONFIRMED_TEXT = "unconfirmed_text"
    CAPABILITY_REVIEW = "capability_review"
    REPLAY_RESULT = "replay_result"
    RUN_CHECKS = "run_checks"
    DISCOVERY_RESULT = "discovery_result"


PERCEPTION_TRIGGERS = frozenset({Trigger.INSUFFICIENT_OBSERVATION})
"""The triggers another observation could plausibly answer.

The loop does not act on this. A request for a person is honoured as it
stands, even when only one tool has run, because the model is the one that
knows whether looking again would help. It may spend its alternate allowance
first by asking for the other tool; it is never made to. An operator channel
can use this set to tell an operator that a second look is still available.
"""


class HandoffOutcome(StrEnum):
    """How the person answered an intervention offer.

    ``TERMINATED`` means the operator ended the run. Nothing the run does
    afterwards acts on the surface; an operation already dispatched settles and
    is reported, but it is not undone.
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    RESUMED = "resumed"
    TIMED_OUT = "timed_out"
    TERMINATED = "terminated"


class State(StrEnum):
    """Where a run's control stands.

    ``STOPPING`` means an operator asked the automation to stop, and an
    operation it had already dispatched is still settling; nothing new starts.
    ``CHECKING`` means a person handed the session back and the run is
    checking it before automation continues.
    """

    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    HUMAN_CONTROL = "human_control"
    CHECKING = "checking"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class Owner(StrEnum):
    """Who may operate the live session.

    ``OPERATOR`` means the run is held for an operator's command and nobody
    is driving the session: a stop, a pending approval, or a run waiting to
    start. ``HUMAN`` means the person was handed the session itself.
    """

    AUTOMATION = "automation"
    OPERATOR = "operator"
    HUMAN = "human"
    NONE = "none"


class Command(StrEnum):
    """What an operator can tell a run's control."""

    START = "start"
    STOP = "stop"
    TAKE_CONTROL = "take_control"
    RESUME = "resume"
    APPROVE = "approve"
    REJECT = "reject"
    TERMINATE = "terminate"


class Verdict(StrEnum):
    """How the control handled a command."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    INVALID = "invalid"


class Via(StrEnum):
    """Which operator channel a command came through."""

    PANEL = "panel"
    TERMINAL = "terminal"
    ESCALATOR = "escalator"
    SCRIPT = "script"


class CheckFailure(StrEnum):
    """Why a session handed back to the automation was not accepted."""

    SESSION_CLOSED = "session_closed"
    OFF_ROUTE = "off_route"
    DIALOG_OPEN = "dialog_open"
    NOT_READY = "not_ready"
    PROTECTION_LOST = "protection_lost"
    SIGN_IN_REQUIRED = "sign_in_required"
    IN_USE = "in_use"
    STEP_CONDITION = "step_condition"


class Interruption(StrEnum):
    """The state of an operation interrupted by a pause.

    ``PENDING`` means the run did not perform it: a proposal was waiting for
    approval or for the next boundary, a decision was discarded, or the
    surface refused it before sending anything. ``PERFORMED`` means the
    surface reported it done; the run has not confirmed its effect on the
    application. ``UNCERTAIN`` means it may or may not have taken effect,
    so nothing may send it again without checking the screen first.
    """

    NONE = "none"
    PENDING = "pending"
    PERFORMED = "performed"
    UNCERTAIN = "uncertain"


class Mode(StrEnum):
    """Which kind of run is asking: model-driven discovery or replay."""

    DISCOVERY = "discovery"
    REPLAY = "replay"


class Ask(StrEnum):
    """The response type an intervention requests and the answers it permits.

    ``APPROVAL`` asks for a decision about one proposal: approve it once,
    reject it, or take the session instead. ``PERSON`` asks for help the run
    cannot give itself, such as a missing value, a sign-in, or a step that
    must name its record; approving is not offered, because an approval would
    not supply what is missing. ``PAUSE`` is the operator's own stop or
    takeover.
    """

    APPROVAL = "approval"
    PERSON = "person"
    PAUSE = "pause"


@dataclasses.dataclass(frozen=True, slots=True)
class InterventionRequest:
    """Enough context for an operator to act without reading the run's log.

    ``route`` is a template rather than the concrete path. The goal, reason,
    and action may contain member data, so this request belongs to the live
    operator channel and must not be persisted as a journal event.

    ``timeout_s`` is the profile's handoff timeout on its own. It is not
    reduced by the run's remaining execution budget, because the execution
    clock stops while a person holds the session.

    ``observed_modes`` lists the tools this attempt actually ran, including
    the ones that came back unusable. An operator reading "structured, visual"
    knows both have been tried; a list built from what was kept would have
    quietly dropped the failures and made the run look less stuck than it is.

    ``pages`` lists every window the session manages, with its id, route, and
    any dialog waiting in it, so a person handed a new window knows which one
    it is even when its body could not be read. ``outputs`` and
    ``unverified`` are the result a person is asked to confirm and what the
    run could not verify about it. All three are member data for the live
    channel only.
    """

    trigger: Trigger
    goal: str
    profile_id: str
    step: int
    route: str
    reason: str
    timeout_s: float
    action: Action | None = None
    observed_modes: tuple[ObservationMode, ...] = ()
    pages: tuple[PageInfo, ...] = ()
    outputs: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    unverified: tuple[str, ...] = ()
    ask: Ask = Ask.PERSON
    mode: Mode = Mode.DISCOVERY
    intervention: str = ""
    """The id the run's control gave this request, such as ``iv-3``.

    Every answer names it, so an answer to an earlier request cannot be
    spent on this one.
    """
    capability: str = ""
    """What the replay was running, such as a capability name and version."""
    session: str = ""
    """Instructions for locating the run's live session."""


@dataclasses.dataclass(frozen=True, slots=True)
class Handoff:
    """The operator's answer, and anything they want the run to know.

    ``RESUMED`` means they operated the live session themselves; the run
    re-observes before deciding again, because the screen has moved underneath
    it. ``APPROVED`` means the proposed action may proceed as written, once,
    if the gate, the target, and any record evidence still hold.

    ``changed`` says a person used the session or the page moved while the
    request waited, so an approval given now was given about a screen that is
    no longer the one the proposal was made against.
    """

    outcome: HandoffOutcome
    operator_note: str | None = None
    intervention: str = ""
    changed: bool = False


class Escalator(Protocol):
    """Routes an intervention request and waits for an operator's answer."""

    def request(self, intervention: InterventionRequest) -> Handoff:
        """Offer control to a human, returning ``TIMED_OUT`` past the deadline."""
        ...
