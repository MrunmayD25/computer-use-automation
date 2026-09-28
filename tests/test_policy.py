"""Checks for the gate every action passes before it reaches a surface."""

from __future__ import annotations

import dataclasses

import pytest
from fakes import grants

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    Operation,
    RecordEvidence,
    Relation,
    SecretRef,
)
from computeruse.policy import (
    Allowed,
    Denial,
    Denied,
    Restriction,
    Source,
    evaluate,
    learned_limit,
    record_bound,
    route_for,
)
from computeruse.profile import ActionKind, Limit, Profile, Risk

MEMBER = "https://sandbox.example.test/members/12345"
SAVINGS = "https://sandbox.example.test/members/12345/savings"
WIRE = "https://sandbox.example.test/members/12345/savings/wire"
FIELD = AxLocator("textbox", "Member ID")


def test_safe_action_is_allowed_with_the_route_that_admitted_it(
    profile: Profile,
) -> None:
    verdict = evaluate(Action(ActionKind.READ, FIELD), profile, MEMBER)

    assert verdict == Allowed(risk=Risk.SAFE, route="/members/:id")


def test_risk_is_reported_as_the_operator_declared_it(profile: Profile) -> None:
    verdict = evaluate(
        Action(ActionKind.CLICK, FIELD, effect="submit_payment"), profile, MEMBER
    )

    assert isinstance(verdict, Allowed)
    assert verdict.risk is Risk.RISKY


def test_action_type_absent_from_the_profile_is_denied(edited_profile) -> None:
    profile = edited_profile(actions=grants({"observe": "safe", "read": "safe"}))

    verdict = evaluate(Action(ActionKind.CLICK, FIELD), profile, MEMBER)

    assert verdict == Denied(Denial.UNDECLARED_ACTION)


def test_secret_not_declared_in_the_profile_is_denied(profile: Profile) -> None:
    action = Action(ActionKind.TYPE, FIELD, SecretRef("unlisted"))

    assert evaluate(action, profile, MEMBER) == Denied(Denial.UNDECLARED_SECRET)


def test_declared_secret_is_allowed(profile: Profile) -> None:
    action = Action(ActionKind.TYPE, FIELD, SecretRef("login_password"))

    assert isinstance(evaluate(action, profile, MEMBER), Allowed)


@pytest.mark.parametrize(
    "location",
    [
        "https://elsewhere.test/members/12345",
        "http://sandbox.example.test/members/12345",
        "https://sandbox.example.test:8443/members/12345",
        "not a url",
    ],
)
def test_action_on_another_origin_is_denied(profile: Profile, location: str) -> None:
    verdict = evaluate(Action(ActionKind.CLICK, FIELD), profile, location)

    assert verdict == Denied(Denial.OFF_ORIGIN)


def test_action_on_an_unreachable_route_is_denied(profile: Profile) -> None:
    verdict = evaluate(
        Action(ActionKind.CLICK, FIELD), profile, "https://sandbox.example.test/admin"
    )

    assert verdict == Denied(Denial.ROUTE_NOT_PERMITTED)


def test_navigation_is_judged_by_its_destination(profile: Profile) -> None:
    verdict = evaluate(
        Action(ActionKind.NAVIGATE, destination=SAVINGS), profile, MEMBER
    )

    assert verdict == Allowed(risk=Risk.SAFE, route="/members/:id/savings/**")


def test_navigation_into_a_deny_route_is_denied(profile: Profile) -> None:
    verdict = evaluate(Action(ActionKind.NAVIGATE, destination=WIRE), profile, MEMBER)

    assert verdict == Denied(Denial.ROUTE_NOT_PERMITTED)


def test_navigation_off_origin_is_denied(profile: Profile) -> None:
    verdict = evaluate(
        Action(ActionKind.NAVIGATE, destination="https://elsewhere.test/members"),
        profile,
        MEMBER,
    )

    assert verdict == Denied(Denial.OFF_ORIGIN)


def test_route_for_names_the_template_not_the_concrete_path(
    profile: Profile,
) -> None:
    assert route_for(profile, MEMBER) == "/members/:id"
    assert route_for(profile, WIRE) is None
    assert route_for(profile, "https://elsewhere.test/members") is None


@pytest.mark.rule(7)
def test_a_record_bound_action_without_evidence_is_denied(profile) -> None:
    """The example profile binds clicks on every savings screen to a record."""
    here = "https://sandbox.example.test/members/12345/savings/close"
    cell = DomLocator("dd", DomAttribute.ID, "account-number")
    evidence = RecordEvidence(cell, "S-1001", Relation.CONTAINER)

    click = Action(ActionKind.CLICK, FIELD, effect="open_account")
    bare = evaluate(click, profile, here)
    proved = evaluate(dataclasses.replace(click, evidence=evidence), profile, here)
    elsewhere = evaluate(click, profile, MEMBER)
    reading = evaluate(Action(ActionKind.READ, FIELD), profile, here)

    assert bare == Denied(Denial.RECORD_EVIDENCE_MISSING)
    assert isinstance(proved, Allowed)
    assert isinstance(elsewhere, Allowed)
    assert isinstance(reading, Allowed)
    assert record_bound(profile, here) == (ActionKind.CLICK, ActionKind.PRESS_KEY)
    assert record_bound(profile, MEMBER) == ()


@pytest.mark.rule(1)
def test_an_exception_restricts_a_wildcard_grant_and_never_widens_it(
    profile,
) -> None:
    here = MEMBER
    button = AxLocator("button", "Send payment")

    plain = evaluate(Action(ActionKind.CLICK, button, effect="open"), profile, here)
    risky = evaluate(
        Action(ActionKind.CLICK, button, effect="submit_payment"), profile, here
    )
    denied = evaluate(
        Action(ActionKind.CLICK, button, effect="close_account"), profile, here
    )
    unlabelled = evaluate(Action(ActionKind.CLICK, button), profile, here)

    assert plain == Allowed(risk=Risk.SAFE, route="/members/:id")
    assert risky == Allowed(risk=Risk.RISKY, route="/members/:id")
    assert denied == Denied(Denial.EFFECT_DENIED)
    assert unlabelled == Denied(Denial.EFFECT_MISSING)


def test_a_learned_restriction_follows_the_operation_not_the_label() -> None:
    send = Operation(
        ActionKind.CLICK,
        "/payments",
        "|button|button|Send payment",
        "d1:c4",
        submission="d1:c2>d1:c4",
        form="d1:c2",
    )
    held = [Restriction(send, "submit_payment", Limit.RISKY, Source.FINDING, 3)]
    by_type = dataclasses.replace(send, target="|button|button|Pay")
    enter = Operation(
        ActionKind.PRESS_KEY,
        "/payments",
        "|input|textbox|Amount",
        "d1:c3",
        "Enter",
        "d1:c2>d1:c4",
        "d1:c2",
    )
    tab = dataclasses.replace(enter, key="Tab", submission="")
    draft = dataclasses.replace(
        send,
        target="|input|button|Save draft",
        control="d1:c5",
        submission="d1:c2>d1:c5",
    )
    search = Operation(
        ActionKind.PRESS_KEY,
        "/payments",
        "|input|textbox|Find a payee",
        "d1:c8",
        "Enter",
        "d1:c7>",
        "d1:c7",
    )
    unaimed = Operation(ActionKind.PRESS_KEY, "/payments", key="Enter")
    elsewhere = dataclasses.replace(send, route="/queue")

    assert learned_limit(held, by_type) is Limit.RISKY
    assert learned_limit(held, enter) is Limit.RISKY
    assert learned_limit(held, unaimed) is Limit.RISKY
    assert learned_limit(held, tab) is None
    assert learned_limit(held, draft) is None
    assert learned_limit(held, search) is None
    assert learned_limit(held, elsewhere) is None


@pytest.mark.rule(1)
def test_a_deny_is_never_hidden_by_a_weaker_restriction() -> None:
    send = Operation(ActionKind.CLICK, "/payments", "|button|button|Send", "d1:c4")
    held = [
        Restriction(send, "submit_payment", Limit.DENY, Source.OPERATOR, 2),
        Restriction(send, "confirm", Limit.RISKY, Source.FINDING, 5),
    ]

    assert learned_limit(held, send) is Limit.DENY
    assert learned_limit(list(reversed(held)), send) is Limit.DENY


def test_a_submission_is_recognised_by_description_after_a_reload() -> None:
    """New element ids, same form and button as the page shows them."""
    paying = "#form|payment|||||input:Amount,input:Memo"
    click = Operation(
        ActionKind.CLICK,
        "/payments",
        "|button|button|Send payment",
        "d1:c4",
        submission="d1:c2>d1:c4",
        form="d1:c2",
        submission_as=f"{paying}>button|button|Send payment",
        form_as=paying,
    )
    held = [Restriction(click, "submit_payment", Limit.DENY, Source.OPERATOR, 3)]
    enter_after = Operation(
        ActionKind.PRESS_KEY,
        "/payments",
        "|input|textbox|Amount",
        "d2:c31",
        "Enter",
        "d2:c29>d2:c33",
        "d2:c29",
        submission_as=f"{paying}>button|button|Send payment",
        form_as=paying,
    )
    draft = dataclasses.replace(
        enter_after,
        kind=ActionKind.CLICK,
        key="",
        submission_as=f"{paying}>input|button|Save draft",
    )
    renamed = dataclasses.replace(
        draft, submission_as="#form|payment-v2|||||input:Amount>button|button|Pay"
    )
    missing = frozenset({click.submission_as})

    assert learned_limit(held, enter_after) is Limit.DENY
    assert learned_limit(held, draft) is None
    assert learned_limit(held, renamed) is None
    assert learned_limit(held, renamed, missing=missing) is Limit.DENY


@pytest.mark.rule(6)
def test_a_restriction_on_screen_input_covers_input_but_not_reads() -> None:
    """Findings 39 and 43: a read sends no input and cannot commit anything."""
    from computeruse.policy import covers

    screen = Operation(ActionKind.CLICK, "/", "screen")
    read = Operation(ActionKind.READ, "/", "screen")
    click = Operation(ActionKind.CLICK, "/", "|button|button|Search", "d1:c2")
    look = Operation(ActionKind.READ, "/", "|cell|cell|Status", "d1:c3")
    assert covers(screen, click)
    assert covers(screen, dataclasses.replace(click, kind=ActionKind.TYPE))
    assert not covers(screen, look)
    assert not covers(screen, read)
    # A restriction learned on a read still covers reads.
    assert covers(read, look)
