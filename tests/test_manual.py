"""How a person's manual step is classified for a later capability.

The classifier first applies profile restrictions and learned denials. It
never records forbidden work as a step that a person may take on every run.
"""

from __future__ import annotations

import dataclasses

import pytest

from computeruse.actions import AxLocator, Operation
from computeruse.escalation import Ask, Trigger
from computeruse.manual import (
    Detail,
    Gap,
    GapKind,
    ManualEvent,
    ManualKind,
    ManualStep,
    Requirement,
    classify,
    gaps_in,
    summary,
)
from computeruse.policy import Restriction, Source
from computeruse.profile import ActionKind, Limit, Profile

MEMBER = "https://sandbox.example.test/members/12345"
SAVINGS = "https://sandbox.example.test/members/12345/savings"
WIRE = "https://sandbox.example.test/members/12345/savings/wire"
SEND = AxLocator("button", "Send")

CLICK = ManualEvent(
    1,
    0.0,
    ManualKind.CLICK,
    owned=True,
    route="/members/:id",
    target=SEND,
    location=MEMBER,
)


def held(
    limit: Limit,
    target: str = "|button|button|Send",
    kind: ActionKind = ActionKind.CLICK,
    submission: str = "",
) -> Restriction:
    operation = Operation(kind, "/members/:id", target, "d1:c4", submission=submission)
    return Restriction(operation, None, limit, Source.OPERATOR, 1)


def judged(
    profile: Profile,
    event: ManualEvent = CLICK,
    *,
    ask: Ask = Ask.PERSON,
    trigger: Trigger | None = None,
    restrictions: tuple[Restriction, ...] = (),
) -> Requirement:
    return classify(event, profile, ask=ask, trigger=trigger, restrictions=restrictions)


def test_an_ordinary_permitted_click_is_validated_and_bound(edited_profile) -> None:
    profile = edited_profile(
        actions={"observe": {"any": "safe"}, "click": {"any": "safe"}}
    )

    assert judged(profile) is Requirement.VALIDATE_AND_BIND


def test_a_click_on_a_denied_route_is_forbidden_even_for_a_secret(profile) -> None:
    on_wire = dataclasses.replace(
        CLICK, route="/members/:id/savings/wire", location=WIRE
    )
    typed = dataclasses.replace(on_wire, kind=ManualKind.EDIT, secret=True)

    assert judged(profile, on_wire) is Requirement.FORBIDDEN
    assert judged(profile, typed) is Requirement.FORBIDDEN
    assert (
        judged(profile, on_wire, trigger=Trigger.AUTHENTICATION_REQUIRED)
        is Requirement.FORBIDDEN
    )


def test_an_action_type_the_profile_does_not_grant_is_forbidden(
    edited_profile,
) -> None:
    profile = edited_profile(actions={"observe": {"any": "safe"}})

    assert judged(profile) is Requirement.FORBIDDEN
    assert judged(profile, trigger=Trigger.AUTHENTICATION_REQUIRED) is (
        Requirement.FORBIDDEN
    )


def test_a_permitted_step_without_record_evidence_needs_a_person(profile) -> None:
    on_savings = dataclasses.replace(
        CLICK, route="/members/:id/savings/**", location=SAVINGS
    )

    assert judged(profile, on_savings) is Requirement.PERSON_EACH_RUN


SEND_OP = Operation(
    ActionKind.CLICK,
    "/members/:id",
    "|button|button|Send",
    "d1:c4",
    submission="d1:c2>d1:c4",
    form="d1:c2",
    submission_as="#form|pay||||>button|Send",
    form_as="#form|pay||||",
)
PRESENT = frozenset({"d1:c2", "d1:c3", "d1:c4", "d1:c5"})
BACK = Operation(ActionKind.CLICK, "/members/:id", "|button|button|Back", "d1:c5")
DENY = Restriction(SEND_OP, None, Limit.DENY, Source.OPERATOR, 1)


def operated(
    event: ManualEvent,
    operation: Operation,
    present: frozenset[str] = PRESENT,
) -> ManualEvent:
    return dataclasses.replace(event, operation=operation, present=present)


def test_a_learned_deny_still_forbids_its_control_under_a_new_name(
    profile,
) -> None:
    """The page renamed the denied button, but its element id did not change.

    A name-only comparison let the renamed button's click through as a step
    that only needs an approval. ``policy.covers`` matches the element id.
    """
    renamed = dataclasses.replace(SEND_OP, target="|button|button|Continue")
    click = operated(
        dataclasses.replace(CLICK, target=AxLocator("button", "Continue")), renamed
    )

    assert judged(profile, click, restrictions=(DENY,)) is Requirement.FORBIDDEN


def test_an_enter_that_performs_a_denied_native_submission_is_forbidden(
    profile,
) -> None:
    """A click on the submit button and an Enter in its form are one submission."""
    enter = Operation(
        ActionKind.PRESS_KEY,
        "/members/:id",
        "|input|textbox|Amount",
        "d1:c3",
        "Enter",
        "d1:c2>d1:c4",
        "d1:c2",
        "#form|pay||||>button|Send",
        "#form|pay||||",
    )
    key = operated(
        dataclasses.replace(
            CLICK,
            kind=ManualKind.KEY,
            detail=Detail.CONFIRM,
            target=AxLocator("textbox", "Amount"),
        ),
        enter,
    )

    assert judged(profile, key, restrictions=(DENY,)) is Requirement.FORBIDDEN


def test_a_denied_enter_forbids_a_click_on_its_submit_button(profile) -> None:
    enter = Operation(
        ActionKind.PRESS_KEY,
        "/members/:id",
        "|input|textbox|Amount",
        "d1:c3",
        "Enter",
        "d1:c2>d1:c4",
        "d1:c2",
    )
    deny = Restriction(enter, None, Limit.DENY, Source.OPERATOR, 1)

    assert judged(profile, operated(CLICK, SEND_OP), restrictions=(deny,)) is (
        Requirement.FORBIDDEN
    )


def test_a_denied_submission_that_left_the_page_covers_every_submission(
    profile,
) -> None:
    """The restricted button's controls are gone, so any submission may be it."""
    save = Operation(
        ActionKind.CLICK,
        "/members/:id",
        "|button|button|Save",
        "d1:c9",
        submission="d1:c8>d1:c9",
        form="d1:c8",
        submission_as="#form|save||||>button|Save",
        form_as="#form|save||||",
    )
    click = operated(
        dataclasses.replace(CLICK, target=AxLocator("button", "Save")),
        save,
        frozenset({"d1:c8", "d1:c9"}),
    )

    assert judged(profile, click, restrictions=(DENY,)) is Requirement.FORBIDDEN


def test_an_input_nobody_could_identify_cannot_escape_a_deny(profile) -> None:
    """With no operation, nothing shows the input was not the denied one."""
    back = dataclasses.replace(CLICK, target=AxLocator("button", "Back"))

    assert back.operation is None
    assert judged(profile, back, restrictions=(DENY,)) is Requirement.FORBIDDEN


def test_a_control_never_observed_may_be_a_denied_one_drawn_again(profile) -> None:
    fresh = Operation(ActionKind.CLICK, "/members/:id", "|button|button|Pay")
    gone = operated(CLICK, fresh, frozenset({"d1:c5"}))
    shown = operated(CLICK, fresh)

    assert judged(profile, gone, restrictions=(DENY,)) is Requirement.FORBIDDEN
    # With the denied button still on the page, a new button is another one.
    assert judged(profile, shown, restrictions=(DENY,)) is (
        Requirement.APPROVE_EACH_RUN
    )


def test_a_distinct_control_the_deny_does_not_cover_is_not_forbidden(
    profile,
) -> None:
    """Another observed button on the same route, outside the denied form.

    It needs an approval, as any step on a route with a learned restriction
    does, and it is not forbidden.
    """
    click = operated(
        dataclasses.replace(CLICK, target=AxLocator("button", "Back")), BACK
    )

    assert judged(profile, click, restrictions=(DENY,)) is Requirement.APPROVE_EACH_RUN


@pytest.mark.parametrize(
    "restriction",
    [
        held(Limit.DENY, target="screen", kind=ActionKind.PRESS_KEY),
        held(Limit.DENY, target="locator|button|Send"),
    ],
    ids=["screen-point", "unresolved-control"],
)
def test_a_learned_deny_that_cannot_be_told_apart_forbids_the_step(
    profile, restriction: Restriction
) -> None:
    other = operated(
        dataclasses.replace(CLICK, target=AxLocator("button", "Back")), BACK
    )

    assert judged(profile, other, restrictions=(restriction,)) is (
        Requirement.FORBIDDEN
    )


def test_a_learned_deny_on_another_route_or_type_does_not_apply(profile) -> None:
    elsewhere = dataclasses.replace(SEND_OP, route="/members")
    typed = dataclasses.replace(SEND_OP, kind=ActionKind.TYPE, submission="")
    restrictions = (
        Restriction(elsewhere, None, Limit.DENY, Source.OPERATOR, 1),
        Restriction(typed, None, Limit.DENY, Source.OPERATOR, 1),
    )

    assert judged(profile, operated(CLICK, BACK), restrictions=restrictions) is (
        Requirement.APPROVE_EACH_RUN
    )


def test_one_denied_effect_does_not_forbid_every_click(profile) -> None:
    """The example profile denies the click effect ``close_account``.

    A manual click has no effect label, so it needs an approval each run,
    and the gate judges the step a recorder labels. It is not forbidden.
    """
    assert judged(profile) is Requirement.APPROVE_EACH_RUN


def test_a_key_that_is_not_movement_is_a_step(profile) -> None:
    key = dataclasses.replace(CLICK, kind=ManualKind.KEY, detail=Detail.COMMAND)

    assert judged(profile, key) is Requirement.APPROVE_EACH_RUN


# Classify a person's native-dialog answer as the action they took.

ANSWERED = dataclasses.replace(
    CLICK, kind=ManualKind.DIALOG, target=None, dialog="dialog-1"
)
DIALOG_GRANTS = {
    "both": {"accept_dialog": {"any": "safe"}, "dismiss_dialog": {"any": "safe"}},
    "accept-only": {"accept_dialog": {"any": "safe"}},
    "neither": {},
}


def granting(edited_profile, grants: str) -> Profile:
    return edited_profile(actions={"observe": {"any": "safe"}, **DIALOG_GRANTS[grants]})


@pytest.mark.parametrize(
    ("detail", "grants", "requirement"),
    [
        (Detail.ACCEPTED, "both", Requirement.VALIDATE_AND_BIND),
        (Detail.DISMISSED, "both", Requirement.VALIDATE_AND_BIND),
        (Detail.ACCEPTED, "accept-only", Requirement.VALIDATE_AND_BIND),
        (Detail.DISMISSED, "accept-only", Requirement.FORBIDDEN),
        (Detail.ACCEPTED, "neither", Requirement.FORBIDDEN),
        (Detail.NONE, "both", Requirement.PERSON_EACH_RUN),
        (Detail.NONE, "accept-only", Requirement.FORBIDDEN),
        (Detail.NONE, "neither", Requirement.FORBIDDEN),
    ],
    ids=[
        "accepted-permitted",
        "dismissed-permitted",
        "accepted-where-only-accepting-is-granted",
        "dismissed-undeclared",
        "accepted-undeclared",
        "unknown-either-permitted",
        "unknown-one-undeclared",
        "unknown-neither-permitted",
    ],
)
def test_a_dialog_answer_is_judged_against_its_own_action_type(
    edited_profile, detail: Detail, grants: str, requirement: Requirement
) -> None:
    """Doing it by hand never makes an undeclared answer permitted.

    An answer that nobody could identify may have been either response. If
    either response is forbidden, the unknown answer is forbidden. Otherwise,
    it needs a person on every run.
    """
    event = dataclasses.replace(ANSWERED, detail=detail)

    assert judged(granting(edited_profile, grants), event) is requirement


def test_a_risky_or_learned_denied_dialog_answer_is_judged_as_such(profile) -> None:
    """The example profile declares accepting a dialog risky.

    A deny the run learned on dismissing dialogs on this route forbids a
    dismissal, and an unknown answer with it.
    """
    accepted = dataclasses.replace(ANSWERED, detail=Detail.ACCEPTED)
    dismissed = dataclasses.replace(ANSWERED, detail=Detail.DISMISSED)
    unknown = dataclasses.replace(ANSWERED, detail=Detail.NONE)
    dismissing = Operation(ActionKind.DISMISS_DIALOG, "/members/:id")
    deny = Restriction(dismissing, None, Limit.DENY, Source.OPERATOR, 1)

    assert judged(profile, accepted) is Requirement.APPROVE_EACH_RUN
    assert judged(profile, dismissed) is Requirement.VALIDATE_AND_BIND
    assert judged(profile, dismissed, restrictions=(deny,)) is Requirement.FORBIDDEN
    assert judged(profile, unknown, restrictions=(deny,)) is Requirement.FORBIDDEN
    assert judged(profile, accepted, restrictions=(deny,)) is (
        Requirement.APPROVE_EACH_RUN
    )


def test_an_unknown_dialog_answer_is_a_recording_gap() -> None:
    unknown = dataclasses.replace(ANSWERED, detail=Detail.NONE)
    accepted = dataclasses.replace(ANSWERED, detail=Detail.ACCEPTED)

    assert gaps_in((unknown,)) == (Gap(GapKind.DIALOG_ANSWER_UNKNOWN),)
    assert gaps_in((accepted,)) == ()
    assert summary((ManualStep(unknown, Requirement.PERSON_EACH_RUN),)) == (
        "closed a dialog on /members/:id; its answer is unknown",
    )
