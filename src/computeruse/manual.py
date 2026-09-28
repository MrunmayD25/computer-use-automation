"""Two representations of what a person did in the live session.

A person working in the session is not the automation, and what they do is
not a proposal the gate saw. It still has to be accounted for: the run has to
know the screen moved, a later capability recorder has to know which steps a
person performed and what each one would need before a replay could repeat
it, and the run's history has to say that someone acted without saying what
they typed or what the screen showed.

Each action a person takes is reported once and kept in two forms. A
``ManualEvent`` is live working context. It holds the page id, the named frame
path, and an accessible-name locator for the control, because a recorder
needs those to find the control again. It is returned to the caller in the
run result and never written down. The journal gets a ``ManualAction`` built
from it with every free-text field removed: the kind, a control category from
a closed list, the route template, frame positions, and flags.

Neither form holds a value. A field edit says a field was edited, never with
what. A key press says which kind of key, never which character. Values are
not captured and dropped later; the page script that reports them never reads
them.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum

from computeruse import policy
from computeruse.actions import Action, AxLocator, Operation
from computeruse.escalation import Ask, Interruption, Mode, Trigger
from computeruse.policy import Restriction
from computeruse.profile import ActionKind, Limit, Profile, Risk


class ManualKind(StrEnum):
    """A category of action taken by a person in the session."""

    CLICK = "click"
    EDIT = "edit"
    SELECT = "select"
    KEY = "key"
    SUBMIT = "submit"
    SCROLL = "scroll"
    NAVIGATION = "navigation"
    NAVIGATION_REFUSED = "navigation_refused"
    PAGE_OPENED = "page_opened"
    PAGE_CLOSED = "page_closed"
    DIALOG = "dialog"


class Widget(StrEnum):
    """A closed category for the control a person used.

    A page sets its own ``role`` attributes and tag names, and either can
    hold any text, including a member's name. The surface maps what the page
    reported onto this list, and anything else becomes ``OTHER``, so a
    journal entry never carries a string the page chose.
    """

    BUTTON = "button"
    LINK = "link"
    TEXTBOX = "textbox"
    SEARCHBOX = "searchbox"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    COMBOBOX = "combobox"
    LISTBOX = "listbox"
    OPTION = "option"
    MENUITEM = "menuitem"
    TAB = "tab"
    SWITCH = "switch"
    SLIDER = "slider"
    SPINBUTTON = "spinbutton"
    CANVAS = "canvas"
    FORM = "form"
    DIALOG = "dialog"
    PAGE = "page"
    CELL = "cell"
    ROW = "row"
    HEADING = "heading"
    IMAGE = "image"
    OTHER = "other"

    @classmethod
    def of(cls, role: str, tag: str = "") -> Widget:
        """Map a reported role, or failing that a tag, onto the closed list.

        Examples
        --------
        >>> Widget.of("button")
        <Widget.BUTTON: 'button'>
        >>> Widget.of("S-1001", "x-member")
        <Widget.OTHER: 'other'>
        >>> Widget.of("", "canvas")
        <Widget.CANVAS: 'canvas'>
        """
        known = {item.value for item in cls}
        for word in (role, tag):
            if word in known:
                return cls(word)
        return cls.OTHER


class Detail(StrEnum):
    """Additional journal detail about a manual action.

    A key is reported only as one of these categories, never as the key. A
    scroll is reported only as a direction, and a dialog only as its answer.
    A category describes the input for the journal. It is never evidence
    that the input left the application unchanged.
    """

    NONE = ""
    CONFIRM = "confirm"
    CANCEL = "cancel"
    MOVE = "move"
    SHORTCUT = "shortcut"
    COMMAND = "command"
    """Any other key that is not movement: a function key, a character typed
    outside a text field, or a key inside one that changed nothing in it."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    CLOSED_BY_POLICY = "closed_by_policy"
    UNCERTAIN = "uncertain"
    """Input on the control the automation was sending to, while it sent, that
    the automation's own input does not account for. It may be a person's,
    and a field it reached is protected."""
    UNPROMPTED = "unprompted"
    """An edit with no key or pointer input just before it, which the page's
    own script can cause. It is protected like any edit, and it does not
    count as a person changing the session."""


class Requirement(StrEnum):
    """What a capability needs to repeat a manual step.

    ``VALIDATE_AND_BIND`` is an ordinary step. A recorder may turn it into a
    reusable step only after its target resolves to exactly one control on a
    fresh observation and, for an edit, the value is bound to a declared
    input, because the value itself was never read.

    ``APPROVE_EACH_RUN`` is a step taken where the run needed an approval, or
    one of a kind the operator declared risky or restricted by effect. Every
    replay needs a fresh approval, or a person performs it. An approval given
    in this run is not carried into the capability.

    ``PERSON_EACH_RUN`` is a step only a person can take: credential or
    one-time-code entry, a step that must name its record when the run cannot,
    a native dialog whose answer the recording could not establish, when
    either answer is permitted, or a control the recorder could not identify.
    A replay stops for a person at that point every time.

    ``CONTEXT`` is movement that is not a step, such as scrolling, moving
    focus, or the navigation a recorded click caused.

    ``FORBIDDEN`` is a step the profile or a learned deny does not permit at
    all: an action type the profile does not grant, a route it does not
    permit, the control a learned deny holds or one that cannot be told apart
    from it, or a dialog answer that may have been one of those. It stays in
    the audit record of what happened. It is never a step of a capability,
    for automation or for a person, and a person doing it once does not make
    it permitted. ``PERSON_EACH_RUN`` is only for work the profile permits.
    ``ManualSegment`` says what a capability recorder owes a forbidden step.
    """

    VALIDATE_AND_BIND = "validate_and_bind"
    APPROVE_EACH_RUN = "approve_each_run"
    PERSON_EACH_RUN = "person_each_run"
    CONTEXT = "context"
    FORBIDDEN = "forbidden"


class GapKind(StrEnum):
    """A gap the recording reports because it could not capture the action."""

    NOT_RECORDED = "not_recorded"
    """The operator channel or surface cannot record input at all."""
    EVENTS_DROPPED = "events_dropped"
    """Events past the per-segment limit were counted, not kept."""
    NOT_DRAINED = "not_drained"
    """Events a page reported sending never arrived before the hand-back."""
    FRAME_UNWATCHED = "frame_unwatched"
    """A permitted frame had no recorder, so input there is unknown."""
    UNIDENTIFIED_TARGET = "unidentified_target"
    """A control with no role and no accessible name was used."""
    DIALOG_ANSWER_UNKNOWN = "dialog_answer_unknown"
    """A person closed a native dialog, and the recording could not establish
    whether they accepted or dismissed it."""
    UNATTRIBUTED_NAVIGATION = "unattributed_navigation"
    """A page changed with no recorded click or key before it, such as the
    browser's own address bar, back button, or reload."""


@dataclasses.dataclass(frozen=True, slots=True)
class ManualEvent:
    """One action by a person as reported by the surface. Live context only.

    ``sequence`` is the order the surface assigned when the event arrived,
    after sorting by the time the page stamped on it. ``owned`` says the
    person had taken control; an event from a person who used the session
    while the run was only paused is recorded with ``owned`` false.
    """

    sequence: int
    at: float
    kind: ManualKind
    owned: bool
    page: str = ""
    route: str = ""
    frame: tuple[str, ...] = ()
    position: tuple[int, ...] = ()
    control: Widget = Widget.OTHER
    target: AxLocator | None = None
    secret: bool = False
    detail: Detail = Detail.NONE
    location: str = ""
    """The address of the frame the event came from. Live context only."""
    operation: Operation | None = None
    """The physical operation, in the format used by the policy matcher:
    the observed control's id,
    its frame, tag, role, and name, and the native submission it could
    perform. None when the surface could not establish it. Live context
    only; the journal never carries it."""
    present: frozenset[str] | None = None
    """The ids of observed controls still in the event's document when the
    surface identified it, so a restricted control that left can be told
    apart from one that is still there. None when unknown."""
    dialog: str = ""
    """The id the surface reported a native dialog under, for a dialog a
    person answered. Its answer is ``detail``: accepted, dismissed, or none
    when the surface could not establish it. Live context only."""

    @property
    def changes(self) -> bool:
        """Report whether this could have changed what the run was looking at.

        This decides whether automation stops and whether a pending approval
        still holds, and it does not read the journal category to decide.
        Every click counts, whatever it landed on, because a page can handle a
        click on any element from a listener on it or on any ancestor. Every
        key counts, a movement key included, and so does every scroll: a page
        can act on either without cancelling it, and nothing the recorder sees
        tells whether it did. A person can still scroll to read a record, and
        a later approval covers the screen as they left it. Only a navigation
        the profile refused, which never happened, and an edit the page's own
        script made are left out.
        """
        match self.kind:
            case ManualKind.NAVIGATION_REFUSED:
                return False
            case ManualKind.EDIT:
                return self.detail is not Detail.UNPROMPTED
            case _:
                return True

    @property
    def input(self) -> bool:
        """Report whether a person's own device produced this change.

        Navigations, pages opening and closing, and dialogs closing are
        noticed by the surface, and a page's own script can cause any of
        them. A click, a key, an edit, a selection, a submission, or a scroll
        is input.
        """
        return self.changes and self.kind not in {
            ManualKind.NAVIGATION,
            ManualKind.PAGE_OPENED,
            ManualKind.PAGE_CLOSED,
            ManualKind.DIALOG,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Gap:
    """One recording gap and how many times it occurred."""

    kind: GapKind
    count: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class ManualStep:
    """A person's action with what a replay would need to repeat it."""

    event: ManualEvent
    requirement: Requirement


@dataclasses.dataclass(frozen=True, slots=True)
class ManualSegment:
    """The actions a person took during one intervention.

    The segment is the integration point for the capability recorder. It is
    live context, returned in the run result and never
    journaled, because steps name controls by their accessible names. A
    recorder writing it into a capability owes these obligations:

    - Keep every requirement as it is. A step may be made stricter, never
      looser.
    - Never write an approval, because an approval is a decision about one
      execution.
    - Keep a ``FORBIDDEN`` step in the audit record of what happened, and
      never export it as a step, for automation or for a person.
    - Never join the steps around a forbidden step as though it did not
      happen. If the workflow depends on it, reject the export or report the
      capability incomplete.
    - Treat ``complete``, a recording with no gaps, as saying only that the
      recording is whole. It says nothing about whether the workflow is
      permitted or can be exported.
    - Keep a person's activity during a replay in that replay's history. It
      never rewrites the saved capability.
    """

    run: str
    intervention: str
    mode: Mode
    ask: Ask
    trigger: Trigger | None
    taken: bool
    steps: tuple[ManualStep, ...]
    gaps: tuple[Gap, ...]
    interrupted: Interruption
    proposal: Action | None = None

    @property
    def complete(self) -> bool:
        """Report whether the recording has no gaps at all.

        A complete recording can still hold forbidden steps, so this never
        means the workflow is permitted or can be exported.
        """
        return not self.gaps


NAVIGATION_CAUSE_S = 5.0
"""How long after a click or key a page change still counts as caused by it."""

PERSON_TRIGGERS = frozenset(
    {Trigger.AUTHENTICATION_REQUIRED, Trigger.RECORD_EVIDENCE_REQUIRED}
)
"""Triggers whose manual work only a person can do on every run."""

_KINDS = {
    ManualKind.CLICK: ActionKind.CLICK,
    ManualKind.EDIT: ActionKind.TYPE,
    ManualKind.SELECT: ActionKind.SELECT,
    ManualKind.KEY: ActionKind.PRESS_KEY,
}


def classify(
    event: ManualEvent,
    profile: Profile,
    *,
    ask: Ask,
    trigger: Trigger | None,
    restrictions: tuple[Restriction, ...] = (),
) -> Requirement:
    """Return what a replay would need to repeat ``event``.

    The step is judged the way the gate would judge the same action, and
    what the profile forbids is settled first. An action type the profile
    does not grant, or a route it does not permit, makes the step
    ``FORBIDDEN``. So does a learned deny on the same control, or on a
    control this event cannot be told apart from. Only a permitted step can
    need a person: a secret, a sign-in, a control with no identity, or a
    step the profile requires record evidence for, because a manual step
    carries no evidence. An action type the operator declared risky, or one
    with effect rules, needs an approval each run. A manual step has no
    effect label, so a recorder must label it, and the gate judges the
    labelled step on every replay; an effect the profile denies is refused
    there. One denied effect does not forbid every step of its action type.

    The rules only ever make a step stricter. Everything from
    ``VALIDATE_AND_BIND`` to ``PERSON_EACH_RUN`` also needs its target
    validated and any value bound before a recorder may write it as a step.
    """
    if event.kind in {
        ManualKind.SCROLL,
        ManualKind.SUBMIT,
        ManualKind.NAVIGATION,
        ManualKind.NAVIGATION_REFUSED,
        ManualKind.PAGE_OPENED,
        ManualKind.PAGE_CLOSED,
    } or (event.kind is ManualKind.KEY and event.detail is Detail.MOVE):
        return Requirement.CONTEXT
    if policy.route_for(profile, event.location) is None:
        return Requirement.FORBIDDEN
    if event.kind is ManualKind.DIALOG:
        return _answer(event, profile, ask, trigger, restrictions)
    kind = _KINDS.get(event.kind)
    if kind is None:
        return Requirement.PERSON_EACH_RUN
    return _judged(event, kind, profile, ask, trigger, restrictions, control=True)


def _judged(
    event: ManualEvent,
    kind: ActionKind,
    profile: Profile,
    ask: Ask,
    trigger: Trigger | None,
    restrictions: tuple[Restriction, ...],
    *,
    control: bool,
) -> Requirement:
    """Judge ``event`` as the action ``kind``, the way the gate would judge it.

    ``control`` says the action names a control, so an event with no control
    identity needs a person. A dialog answer names the dialog instead.
    """
    if kind not in profile.actions:
        return Requirement.FORBIDDEN
    if _denied(event, kind, restrictions):
        return Requirement.FORBIDDEN
    if (
        event.secret
        or trigger in PERSON_TRIGGERS
        or (control and event.target is None)
        or kind in policy.record_bound(profile, event.location)
    ):
        return Requirement.PERSON_EACH_RUN
    learned = [
        held.limit
        for held in restrictions
        if held.operation.kind is kind and held.operation.route == event.route
    ]
    effects = profile.effects.get(kind, {})
    if ask is Ask.APPROVAL or learned or profile.actions[kind] is Risk.RISKY or effects:
        return Requirement.APPROVE_EACH_RUN
    return Requirement.VALIDATE_AND_BIND


_ANSWERS = {
    Detail.ACCEPTED: (ActionKind.ACCEPT_DIALOG,),
    Detail.DISMISSED: (ActionKind.DISMISS_DIALOG,),
}
"""The action a known answer to a native dialog is."""


def _answer(
    event: ManualEvent,
    profile: Profile,
    ask: Ask,
    trigger: Trigger | None,
    restrictions: tuple[Restriction, ...],
) -> Requirement:
    """Judge a person's answer to a native dialog as the action it was.

    A known answer is ``accept_dialog`` or ``dismiss_dialog`` and is judged
    as that action type, grants, effect rules, and learned restrictions
    included. An unknown answer may have been either, so each is judged. If
    either could be forbidden, the event is forbidden, because nothing shows
    the person gave the permitted one. If both are permitted it still needs a
    person every run, because a replay cannot repeat an answer nobody knows.
    """
    kinds = _ANSWERS.get(
        event.detail, (ActionKind.ACCEPT_DIALOG, ActionKind.DISMISS_DIALOG)
    )
    judged = [
        _judged(event, kind, profile, ask, trigger, restrictions, control=False)
        for kind in kinds
    ]
    if len(judged) == 1:
        return judged[0]
    if Requirement.FORBIDDEN in judged:
        return Requirement.FORBIDDEN
    return Requirement.PERSON_EACH_RUN


_UNIDENTIFIED = ("locator|", "painted|")
"""How an operation's target starts when no observed control identified it."""


def _denied(
    event: ManualEvent, kind: ActionKind, restrictions: tuple[Restriction, ...]
) -> bool:
    """Report whether a learned deny may cover what ``event`` did.

    The event's operation is matched by ``policy.covers``, the rule the loop
    applies to the automation's own proposals, so a control renamed since the
    deny, or an Enter that performs a restricted native submission, is still
    covered. What the policy matcher learns from observations is taken from
    the event instead: a restricted control whose id the event's document no
    longer holds is lost, and a restricted submission whose controls it no
    longer holds is missing. A key known only as a category may be any key.
    An event the surface could not identify cannot be told apart from any
    operation of its action type on its route.

    Examples
    --------
    >>> from computeruse.policy import Source
    >>> send = Operation(ActionKind.CLICK, "/pay", "|button|button|Send", "d1:c4")
    >>> deny = Restriction(send, None, Limit.DENY, Source.OPERATOR, 1)
    >>> renamed = dataclasses.replace(send, target="|button|button|Continue")
    >>> click = ManualEvent(1, 0.0, ManualKind.CLICK, owned=True, route="/pay",
    ...                     operation=renamed, present=frozenset({"d1:c4"}))
    >>> _denied(click, ActionKind.CLICK, (deny,))
    True
    >>> other = dataclasses.replace(renamed, control="d1:c5")
    >>> _denied(dataclasses.replace(click, operation=other), ActionKind.CLICK,
    ...         (deny,))
    False
    """
    present = event.present
    lost = frozenset(
        held.operation.control
        for held in restrictions
        if held.operation.control
        and (present is None or held.operation.control not in present)
    )
    missing = frozenset(
        held.operation.submission_as
        for held in restrictions
        if held.operation.submission_as
        and (
            present is None
            or not all(
                part in present for part in held.operation.submission.split(">") if part
            )
        )
    )
    operation = event.operation
    for held in restrictions:
        if held.limit is not Limit.DENY or held.operation.route != event.route:
            continue
        restricted = held.operation
        if operation is None:
            # Use the same scope that ``policy.covers`` assigns to an unresolved
            # target. A screen point covers all input. A submission covers a
            # click or key.
            either = kind in {ActionKind.CLICK, ActionKind.PRESS_KEY}
            if (
                restricted.kind is kind
                or restricted.target == "screen"
                or (either and bool(restricted.submission or restricted.submission_as))
            ):
                return True
            continue
        proposed = operation
        if kind is ActionKind.PRESS_KEY and not operation.key:
            proposed = dataclasses.replace(operation, key=restricted.key)
        if policy.covers(restricted, proposed, lost, missing):
            return True
        if restricted.kind is kind and restricted.target.startswith(_UNIDENTIFIED):
            # The restricted operation was never tied to one control, so no
            # control can be shown not to be it.
            return True
        if (
            not operation.control
            and restricted.kind is kind
            and (not restricted.control or restricted.control in lost)
            and restricted.target.split("|")[:2] == operation.target.split("|")[:2]
        ):
            # A control this surface never observed may be the restricted one
            # drawn again, and nothing here can say it is not.
            return True
    return False


def gaps_in(events: tuple[ManualEvent, ...]) -> tuple[Gap, ...]:
    """Return the gaps the events themselves show.

    A control with no identity, and a page change with no click, key, or
    submission on that page shortly before it, are both things the recording
    saw happen without being able to say how.

    Examples
    --------
    >>> moved = ManualEvent(1, 10.0, ManualKind.NAVIGATION, owned=True, page="page-1")
    >>> gaps_in((moved,))
    (Gap(kind=<GapKind.UNATTRIBUTED_NAVIGATION: 'unattributed_navigation'>, count=1),)
    >>> click = ManualEvent(1, 9.5, ManualKind.CLICK, owned=True, page="page-1",
    ...                     target=AxLocator("link", "Savings"))
    >>> gaps_in((click, moved))
    ()
    """
    unidentified = sum(
        1
        for event in events
        if event.kind in {ManualKind.CLICK, ManualKind.EDIT, ManualKind.SELECT}
        and event.target is None
    )
    causes = {ManualKind.CLICK, ManualKind.KEY, ManualKind.SUBMIT, ManualKind.EDIT}
    unattributed = 0
    for index, event in enumerate(events):
        if event.kind is not ManualKind.NAVIGATION:
            continue
        earlier = events[:index]
        if not any(
            item.kind in causes
            and item.page == event.page
            and event.at - item.at <= NAVIGATION_CAUSE_S
            for item in earlier
        ):
            unattributed += 1
    unanswered = sum(
        1
        for event in events
        if event.kind is ManualKind.DIALOG and event.detail is Detail.NONE
    )
    found = []
    if unidentified:
        found.append(Gap(GapKind.UNIDENTIFIED_TARGET, unidentified))
    if unanswered:
        found.append(Gap(GapKind.DIALOG_ANSWER_UNKNOWN, unanswered))
    if unattributed:
        found.append(Gap(GapKind.UNATTRIBUTED_NAVIGATION, unattributed))
    return tuple(found)


def merge(gaps: tuple[Gap, ...]) -> tuple[Gap, ...]:
    """Add up gaps of the same kind, keeping first-seen order.

    Examples
    --------
    >>> merge((Gap(GapKind.EVENTS_DROPPED, 2), Gap(GapKind.EVENTS_DROPPED, 3)))
    (Gap(kind=<GapKind.EVENTS_DROPPED: 'events_dropped'>, count=5),)
    """
    totals: dict[GapKind, int] = {}
    for gap in gaps:
        totals[gap.kind] = totals.get(gap.kind, 0) + gap.count
    return tuple(Gap(kind, count) for kind, count in totals.items() if count)


def summary(steps: tuple[ManualStep, ...], limit: int = 10) -> tuple[str, ...]:
    """Describe what a person did, briefly, for the model's next decision.

    Live context only: it names controls by their accessible names, which the
    model has already been shown on the page. It never includes a value.

    Examples
    --------
    >>> approve = AxLocator("button", "Approve")
    >>> click = ManualEvent(1, 0.0, ManualKind.CLICK, owned=True, route="/queue",
    ...                     target=approve, control=Widget.BUTTON)
    >>> summary((ManualStep(click, Requirement.VALIDATE_AND_BIND),))
    ("clicked button 'Approve' on /queue",)
    """
    lines = []
    for step in steps[:limit]:
        event = step.event
        where = f" on {event.route}" if event.route else ""
        named = (
            f"{event.target.role} {event.target.name!r}"
            if event.target is not None
            else f"a {event.control} with no accessible name"
        )
        match event.kind:
            case ManualKind.CLICK:
                lines.append(f"clicked {named}{where}")
            case ManualKind.EDIT:
                lines.append(f"edited {named}{where}; its value is withheld")
            case ManualKind.SELECT:
                lines.append(f"changed the selection of {named}{where}")
            case ManualKind.KEY:
                lines.append(f"pressed a {event.detail or 'special'} key{where}")
            case ManualKind.DIALOG if event.detail:
                lines.append(f"{event.detail.value} a dialog{where}")
            case ManualKind.DIALOG:
                lines.append(f"closed a dialog{where}; its answer is unknown")
            case ManualKind.NAVIGATION:
                lines.append(f"moved to {event.route or 'a page outside the profile'}")
            case ManualKind.NAVIGATION_REFUSED:
                lines.append("tried to open a page the profile does not permit")
            case ManualKind.PAGE_OPENED | ManualKind.PAGE_CLOSED:
                lines.append(f"{event.kind.value.replace('_', ' ')} {event.page}")
            case _:
                continue
    if len(steps) > limit:
        lines.append(f"and {len(steps) - limit} more")
    return tuple(lines)


def single_click(segments: tuple[ManualSegment, ...]) -> ManualEvent | None:
    """Return the one click a person made, when it is all they did that counts.

    Scrolling and other movement do not count. The click must name its
    control, and the rules above must judge it repeatable, with or without an
    approval each run. Anything else, or any gap in the record, gives None,
    and the person's work stays a person's step.
    """
    if any(segment.gaps for segment in segments):
        return None
    steps = [
        step
        for segment in segments
        for step in segment.steps
        if step.requirement is not Requirement.CONTEXT
    ]
    if len(steps) != 1:
        return None
    step = steps[0]
    if (
        step.event.kind is not ManualKind.CLICK
        or step.event.target is None
        or step.requirement
        not in {Requirement.VALIDATE_AND_BIND, Requirement.APPROVE_EACH_RUN}
    ):
        return None
    return step.event
