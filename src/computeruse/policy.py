"""The gate every action passes through before it reaches a surface.

The caller cannot replace this gate. Other protocols let the loop use a new
surface or model, but an injected policy function could silently permit every
action. The profile varies; the code that interprets it does not.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from enum import StrEnum

from computeruse.actions import ENTER, Action, ObservationRequest, Operation, SecretRef
from computeruse.profile import ActionKind, Limit, Profile, Risk
from computeruse.urls import parse_http_url


class Denial(StrEnum):
    """The reason the profile refused an action."""

    UNDECLARED_ACTION = "undeclared_action"
    UNDECLARED_SECRET = "undeclared_secret"  # noqa: S105  a reason, not a value
    OFF_ORIGIN = "off_origin"
    ROUTE_NOT_PERMITTED = "route_not_permitted"
    MODE_NOT_PERMITTED = "mode_not_permitted"
    ALTERNATE_LIMIT_REACHED = "alternate_limit_reached"
    RECORD_EVIDENCE_MISSING = "record_evidence_missing"
    EFFECT_MISSING = "effect_missing"
    EFFECT_DENIED = "effect_denied"
    SUBMISSION_DENIED = "submission_denied"
    REPEATED_CHANGE = "repeated_change"
    SELECTION_UNCONFIRMED = "selection_unconfirmed"


@dataclasses.dataclass(frozen=True, slots=True)
class Allowed:
    """The action may proceed, at the risk the operator declared for it."""

    risk: Risk
    route: str


@dataclasses.dataclass(frozen=True, slots=True)
class Denied:
    """The action may not proceed."""

    reason: Denial


type Verdict = Allowed | Denied


def route_for(profile: Profile, location: str) -> str | None:
    """Return the allow route admitting ``location``, or None if it is out of policy.

    Parameters
    ----------
    profile
        The operator's policy.
    location
        An absolute HTTP(S) URL naming where the surface currently is.
    """
    return profile.scope.route(location)


def evaluate(action: Action, profile: Profile, location: str) -> Verdict:
    """Decide whether ``action`` may run while the surface sits at ``location``.

    Six questions, in order: is the action type granted at all, does any
    secret it names exist, is the screen it acts on reachable, does an effect
    exception deny or restrict it, for a navigation, is its destination
    reachable, and does the operator require this action on this screen to
    name its record. The record question comes last, so a refusal for missing
    evidence means every other question passed. The loop relies on that when
    it asks a person to perform a step the run cannot prove the record of.
    Risk is reported rather than acted on here, because escalating a risky
    action is the loop's job and is not configurable.

    An action type with effect exceptions must say its effect, or the
    exceptions could not apply to it, so an unlabelled one is refused. The
    label is the model's word, and a label can only make things stricter
    here: an exception is risky or deny, and no label reaches a grant.

    The record question is only whether evidence is present. Whether the
    evidence is true is settled against the observation by the loop and
    against the live page by the surface, neither of which the gate can see.

    Returns
    -------
    Verdict
        ``Allowed`` carrying the declared risk and the route template that
        admitted the action, or ``Denied`` carrying the reason.
    """
    if action.kind not in profile.actions:
        return Denied(Denial.UNDECLARED_ACTION)
    if isinstance(action.value, SecretRef) and action.value.name not in profile.secrets:
        return Denied(Denial.UNDECLARED_SECRET)

    if not profile.scope.contains(location):
        return Denied(Denial.OFF_ORIGIN)
    route = route_for(profile, location)
    if route is None:
        return Denied(Denial.ROUTE_NOT_PERMITTED)
    risk = profile.actions[action.kind]
    if profile.effects.get(action.kind) and action.effect is None:
        return Denied(Denial.EFFECT_MISSING)
    limit = profile.effect_limit(action.kind, action.effect)
    if limit is Limit.DENY:
        return Denied(Denial.EFFECT_DENIED)
    if limit is Limit.RISKY:
        risk = Risk.RISKY

    if action.destination is not None:
        if not profile.scope.contains(action.destination):
            return Denied(Denial.OFF_ORIGIN)
        destination = route_for(profile, action.destination)
        if destination is None:
            return Denied(Denial.ROUTE_NOT_PERMITTED)
        route = destination

    if action.evidence is None and profile.requires_record(
        action.kind, parse_http_url(location).path or "/"
    ):
        return Denied(Denial.RECORD_EVIDENCE_MISSING)
    return Allowed(risk=risk, route=route)


class Source(StrEnum):
    """The source of a restriction held by the run."""

    OPERATOR = "operator"
    PROPOSAL = "proposal"
    FINDING = "finding"


@dataclasses.dataclass(frozen=True, slots=True)
class Restriction:
    """An operation that requires a person or is forbidden.

    The profile is never edited during a run; restrictions live here, beside
    it, for the length of the run. ``OPERATOR`` restrictions bind an
    operation the model labelled with an effect the profile restricts, so
    relabelling that operation later does not escape the profile.
    ``PROPOSAL`` and ``FINDING`` restrictions come from the model, before an
    action and after one. Only ``OPERATOR`` restrictions can deny; the model
    can only add a need for approval.

    ``step`` is the step of the action the restriction was learned from.
    ``reason`` is the model's own words and stays in live working context.
    """

    operation: Operation
    effect: str | None
    limit: Limit
    source: Source
    step: int
    reason: str = ""


LOOKS = frozenset(
    {
        ActionKind.READ,
        ActionKind.ASSERT,
        ActionKind.WAIT,
        ActionKind.WAIT_FOR,
        ActionKind.OBSERVE,
    }
)
"""Action types that send no input, which screen-wide coverage leaves alone."""


def covers(
    held: Operation,
    proposed: Operation,
    lost: frozenset[str] = frozenset(),
    missing: frozenset[str] = frozenset(),
) -> bool:
    """Report whether a restriction on ``held`` applies to ``proposed``.

    Every rule here can only widen a restriction. On the same screen:

    - The same native form submission, whether by a click on the submit
      button or an Enter the browser turns into that click, recognised by
      element within one document and by description after a reload or a
      redraw. While a restricted submission is missing from the screen, any
      submission in its frame.
    - An Enter into a form whose submission is not native, against anything
      restricted in that form, and the reverse against that form's native
      submissions, because the page's own handler might be either.
    - An Enter sent to whatever has focus, against any restricted Enter or
      form submission, because nothing says where it goes.
    - For the same action type, the same control id, which is two locators
      for one element, or the same frame, tag, role, and name, which is the
      same control drawn again.
    - For the same action type and tag, a control first seen after a
      restricted control vanished from the same document with no successor
      of the same name. The two may be one control renamed, and nothing here
      can say they are not.

    Examples
    --------
    >>> send = Operation(ActionKind.CLICK, "/pay", "|button|button|Send payment",
    ...                  "d1:c4", submission="d1:c2>d1:c4", form="d1:c2")
    >>> enter = Operation(ActionKind.PRESS_KEY, "/pay", "|input|textbox|Amount",
    ...                   "d1:c3", "Enter", "d1:c2>d1:c4", "d1:c2")
    >>> covers(send, enter), covers(enter, send)
    (True, True)
    >>> tab = Operation(ActionKind.PRESS_KEY, "/pay", "|input|textbox|Amount",
    ...                 "d1:c3", "Tab", form="d1:c2")
    >>> covers(send, tab)
    False
    """
    if held.business and proposed.business:
        return held.business == proposed.business
    if held.route != proposed.route:
        return False
    if held.business:
        return proposed.kind not in LOOKS
    if "screen" in {held.target, proposed.target}:
        # Pixels cannot prove that another input avoids the restricted
        # operation, so a restriction on screen input covers all input on the
        # screen. A look or a read sends no input and performs nothing, so it
        # is covered only by a restriction learned on a look or a read.
        return proposed.kind not in LOOKS or held.kind in LOOKS
    if _one_submission(held, proposed, missing):
        return True
    if held.kind is not proposed.kind:
        return False
    if held.kind is ActionKind.PRESS_KEY:
        if held.key != proposed.key:
            return False
        if not held.target or not proposed.target:
            return True
    if held.control and held.control == proposed.control:
        return True
    if held.target == proposed.target:
        return True
    return held.control in lost and _successor(held, proposed)


def _one_submission(
    held: Operation, proposed: Operation, missing: frozenset[str]
) -> bool:
    """Report whether the two may submit the same form, whatever their kinds.

    The element ids answer this within one document. The descriptions answer
    it after a reload or a redraw, when the ids are new. A restricted
    submission that the current screen no longer shows under its description
    cannot be told apart from any submission in the same frame, so every one
    of them is treated as it until it is seen again.
    """
    if held.submission and held.submission == proposed.submission:
        return True
    if held.submission_as and held.submission_as == proposed.submission_as:
        return True
    if proposed.uncertain and _same_form(held, proposed):
        return True
    if held.uncertain and _same_form(held, proposed):
        return bool(proposed.submission) or proposed.uncertain
    if held.submission_as in missing and _submits(proposed):
        return _frame(held.submission_as) == _frame(proposed.form_as)
    return _unaimed_enter(proposed) and bool(held.submission or held.uncertain)


def _same_form(held: Operation, proposed: Operation) -> bool:
    if held.form and held.form == proposed.form:
        return True
    return bool(held.form_as) and held.form_as == proposed.form_as


def _submits(operation: Operation) -> bool:
    return bool(operation.submission_as) or operation.uncertain


def submission_limit(profile: Profile, operation: Operation) -> Limit | None:
    """Return the operator's rule for an operation that may submit a form.

    Whether an operation submits comes from the observed control: a native
    form submission, an Enter into a form the page handles itself, or an
    Enter with no target. The model's effect label plays no part, so a
    submission called something harmless is still a submission.

    Examples
    --------
    >>> from computeruse.actions import Action, AxLocator, AxNode
    >>> send = AxNode("button", "Send", tag="button", control="d1:c4",
    ...               form="d1:c2", submits="d1:c2>d1:c4")
    >>> click = Action(ActionKind.CLICK, AxLocator("button", "Send"))
    >>> operation = Operation.of(click, "/pay", node=send)
    >>> from computeruse.profile import Submissions
    >>> class Declared:
    ...     submissions = Submissions(Limit.RISKY, {"/search": None})
    >>> submission_limit(Declared(), operation)
    <Limit.RISKY: 'risky'>
    >>> submission_limit(Declared(), Operation.of(click, "/search", node=send)) is None
    True
    >>> Declared.submissions = Submissions(Limit.RISKY, controls=frozenset({"Send"}))
    >>> submission_limit(Declared(), operation) is None
    True
    """
    if profile.submissions is None:
        return None
    submits = (
        bool(operation.submission) or _submits(operation) or _unaimed_enter(operation)
    )
    if not submits:
        return None
    # Only a click resolved to one observed button can be that button.
    clicked = operation.kind is ActionKind.CLICK and operation.resolved
    name = operation.target.rpartition("|")[2] if clicked else ""
    return profile.submissions.limit(operation.route, name)


def _frame(description: str) -> str:
    return description.partition("#")[0]


def _unaimed_enter(operation: Operation) -> bool:
    return (
        operation.kind is ActionKind.PRESS_KEY
        and operation.key == ENTER
        and not operation.target
    )


def _successor(held: Operation, proposed: Operation) -> bool:
    """Report whether ``proposed`` names a control drawn after ``held`` was lost.

    Examples
    --------
    >>> old = Operation(ActionKind.CLICK, "/p", "|button|button|Send", "d1:c4")
    >>> new = Operation(ActionKind.CLICK, "/p", "|button|button|Pay", "d1:c9")
    >>> _successor(old, new)
    True
    >>> _successor(old, Operation(ActionKind.CLICK, "/p", "|a|link|Pay", "d1:c9"))
    False
    >>> other = Operation(ActionKind.CLICK, "/p", "|button|button|Pay", "d2:c1")
    >>> _successor(old, other)
    False
    """
    held_document, _, held_number = held.control.partition(":c")
    document, _, number = proposed.control.partition(":c")
    if not (held_number.isdigit() and number.isdigit()):
        return False
    if document != held_document or int(number) <= int(held_number):
        return False
    held_frame, held_tag = held.target.split("|")[:2]
    frame, tag = proposed.target.split("|")[:2]
    return (frame, tag) == (held_frame, held_tag)


def learned_limit(
    restrictions: Iterable[Restriction],
    operation: Operation,
    lost: frozenset[str] = frozenset(),
    missing: frozenset[str] = frozenset(),
) -> Limit | None:
    """Return the strictest restriction that applies to this proposal.

    A restriction applies when it covers the operation, whatever the proposal
    calls it and however it names the control. The effect label plays no part
    here: a label cannot lift a restriction, and two unrelated forms are not
    one operation because the model called both of them ``submit``. The
    operator's effect exceptions are applied by label in ``evaluate``, as the
    operator declared them. A deny is returned before anything weaker, so a
    later, weaker restriction never hides it, and a deny matched only because
    identity is uncertain is still a deny, not something a person approves.
    """
    found: Limit | None = None
    for held in restrictions:
        if not covers(held.operation, operation, lost, missing):
            continue
        if held.limit is Limit.DENY:
            return Limit.DENY
        found = Limit.RISKY
    return found


def record_bound(profile: Profile, location: str) -> tuple[ActionKind, ...]:
    """Return the action types that must name their record at ``location``."""
    if route_for(profile, location) is None:
        return ()
    path = parse_http_url(location).path or "/"
    return tuple(
        kind for kind in profile.records.actions if profile.requires_record(kind, path)
    )


def evaluate_observation(
    request: ObservationRequest,
    profile: Profile,
    location: str,
    *,
    held: bool = False,
    alternates_used: int = 0,
) -> Verdict:
    """Decide whether the surface may be looked at, before anything is read.

    Looking is an action. It needs ``observe`` declared in ``actions``, and it
    needs its own tool declared in ``perception.allowed_modes``, because
    granting a screenshot is a different decision from granting an
    accessibility tree: one of them photographs whatever is on the screen.

    This is checked from ``location`` alone, which the adapter reports from
    session metadata, so permission is settled before any page content is
    collected.

    Parameters
    ----------
    request
        The tool being asked for, and why.
    profile
        The operator's policy.
    location
        Where the surface currently is, from session metadata.
    held
        Whether an observation of this screen is already in hand. A second
        look at the same screen is what the alternate allowance bounds.
    alternates_used
        How many alternate looks this unresolved action has already had.

    Returns
    -------
    Verdict
        ``Allowed`` carrying the risk the operator declared for ``observe``,
        or ``Denied`` carrying the reason.
    """
    if ActionKind.OBSERVE not in profile.actions:
        return Denied(Denial.UNDECLARED_ACTION)
    if request.mode not in profile.perception.allowed_modes:
        return Denied(Denial.MODE_NOT_PERMITTED)
    if not profile.scope.contains(location):
        return Denied(Denial.OFF_ORIGIN)
    route = route_for(profile, location)
    if route is None:
        return Denied(Denial.ROUTE_NOT_PERMITTED)
    limit = profile.perception.max_alternate_observations_per_step
    if held and alternates_used >= limit:
        return Denied(Denial.ALTERNATE_LIMIT_REACHED)
    return Allowed(risk=profile.actions[ActionKind.OBSERVE], route=route)
