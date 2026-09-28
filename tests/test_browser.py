"""Check the adapter against Chromium and the local evaluation pages.

These tests start the browser, serve the pages, and assert what the adapter
does to a live page. The loop tests cover control flow. These tests cover the
browser implementation beneath it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    Expectation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    PageState,
    Scope,
    ScopeKind,
    SecretRef,
    VisualAnchor,
)
from computeruse.browser import BrowserSurface, open_session
from computeruse.profile import ActionKind, Profile

STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)
VISUAL = ObservationRequest(ObservationMode.VISUAL)

OPEN = AxLocator("button", "Open")
LOCK = DomLocator("span", DomAttribute.CSS_CLASS, "posting-lock")
LOCK_STATE = DomLocator("output", DomAttribute.ID, "lock-state")
APPROVAL_STATE = DomLocator("output", DomAttribute.ID, "approval-state")
PASSCODE = AxLocator("textbox", "Approver passcode")
MOVE = DomLocator("button", DomAttribute.ID, "move-control")
CLEAR = DomLocator("button", DomAttribute.ID, "clear-pad")


@pytest.fixture
def surface(site: str, site_profile) -> Callable[..., Iterator[BrowserSurface]]:
    """Return a factory that opens a session at one of the evaluation pages."""

    def _open(path: str = "/members", **edits: object):
        profile: Profile = site_profile(**edits)
        return open_session(profile, f"{site}{path}")

    return _open


def names(observation, role: str) -> list[str]:
    return [node.name for node in observation.nodes if node.role == role]


# Structured observation.


def test_structured_observation_reports_roles_names_and_row_context(
    surface, site_profile
) -> None:
    with surface("/members") as page:
        observation = page.observe(
            ObservationRequest(site_profile().perception.allowed_modes[0])
        )

    assert observation.mode is ObservationMode.STRUCTURED
    assert observation.status is ObservationStatus.COMPLETE
    assert "Search" in names(observation, "button")
    assert "Member number" in names(observation, "textbox")
    opens = [
        node
        for node in observation.nodes
        if node.name == "Open" and node.role == "button"
    ]
    assert len(opens) == 3
    assert {node.scope.name for node in opens if node.scope} == {
        "12345",
        "67890",
        "24680",
    }
    assert all(node.scope.kind is ScopeKind.ROW for node in opens if node.scope)


def test_structured_observation_reports_supported_interactions(surface) -> None:
    with surface("/members") as page:
        observation = page.observe(STRUCTURED)

    search = next(
        node
        for node in observation.nodes
        if node.name == "Search" and node.tag == "button"
    )
    field = next(
        node
        for node in observation.nodes
        if node.name == "Member number" and node.tag == "input"
    )
    assert ActionKind.CLICK in search.interactions
    assert ActionKind.PRESS_KEY in field.interactions
    assert ActionKind.TYPE in field.interactions


def test_structured_observation_reports_dom_attributes(surface) -> None:
    with surface("/members/12345") as page:
        observation = page.observe(STRUCTURED)

    lock = next(node for node in observation.nodes if "posting-lock" in node.classes)
    assert lock.name == "Posting lock"
    assert ("title", "Posting lock") in lock.attributes
    assert lock.tag == "span"


def test_a_permitted_frame_is_read_and_its_controls_carry_its_name(surface) -> None:
    with surface("/members/12345") as page:
        observation = page.observe(STRUCTURED)

    inside = [node for node in observation.nodes if node.frame == ("ledger",)]
    assert inside
    assert any("$4,212.55" in (node.name or "") for node in inside)
    assert observation.page_state.signature


def test_the_page_signature_ignores_a_second_capture_of_the_same_page(
    surface,
) -> None:
    with surface("/members/12345") as page:
        first = page.observe(STRUCTURED)
        second = page.observe(STRUCTURED)

    assert first.observation_id != second.observation_id
    assert not first.page_state.differs_from(second.page_state)


# Frames outside the profile.


def test_a_frame_outside_the_profile_is_not_read(surface) -> None:
    narrow = ["/members", "/members/:id", "/static/**"]

    with surface("/members/12345", allow_routes=narrow) as page:
        observation = page.observe(STRUCTURED)

    assert all(node.frame == () for node in observation.nodes)
    assert observation.status is ObservationStatus.PARTIAL
    assert any("outside the profile" in note for note in observation.notes)


def test_a_screenshot_masks_a_frame_outside_the_profile(surface) -> None:
    narrow = ["/members", "/members/:id", "/static/**"]

    with surface("/members/12345", allow_routes=narrow) as page:
        picture = page.observe(VISUAL)

    assert picture.image is not None
    assert picture.visual is not None
    assert "a frame outside the profile" in picture.visual.masked


# Secrets.


@pytest.mark.rule(18)
def test_a_screenshot_masks_password_fields(surface, site_profile) -> None:
    with surface("/members/12345") as page:
        picture = page.observe(
            ObservationRequest(site_profile().perception.allowed_modes[1])
        )

    assert picture.mode is ObservationMode.VISUAL
    assert picture.visual is not None
    assert "credential fields" in picture.visual.masked
    assert picture.image is not None
    assert picture.image.startswith(b"\x89PNG")


def test_a_password_value_is_never_reported(surface) -> None:
    with surface("/members/12345") as page:
        page.act(Action(ActionKind.TYPE, PASSCODE, "not-a-real-passcode"))
        observation = page.observe(STRUCTURED)

    field = next(
        node
        for node in observation.nodes
        if node.name == "Approver passcode" and node.tag == "input"
    )
    assert field.secret is True
    assert field.value is None
    assert "not-a-real-passcode" not in repr(observation)


def test_a_declared_secret_is_read_from_the_environment_or_refused(
    surface, monkeypatch
) -> None:
    monkeypatch.delenv("APPROVER_PASSCODE", raising=False)
    typing = Action(ActionKind.TYPE, PASSCODE, SecretRef("approver_passcode"))

    with surface("/members/12345") as page:
        refused = page.act(typing)
        monkeypatch.setenv("APPROVER_PASSCODE", "synthetic-value")
        accepted = page.act(typing)
        observation = page.observe(STRUCTURED)

    assert refused.outcome is Outcome.NOT_ACTIONABLE
    assert accepted.outcome is Outcome.OK
    assert "synthetic-value" not in repr(observation)


# Target resolution.


def test_a_repeated_label_without_its_row_is_ambiguous(surface) -> None:
    with surface("/members") as page:
        before = page.location()
        result = page.act(Action(ActionKind.CLICK, OPEN))
        after = page.location()

    assert result.outcome is Outcome.AMBIGUOUS
    assert before == after


def test_a_row_scope_resolves_one_of_three_identical_buttons(surface) -> None:
    target = AxLocator("button", "Open", scope=Scope(ScopeKind.ROW, "12345"))

    with surface("/members") as page:
        result = page.act(Action(ActionKind.CLICK, target))
        landed = page.location()

    assert result.outcome is Outcome.OK
    assert landed.endswith("/members/12345")


def test_a_scope_that_matches_nothing_is_refused(surface) -> None:
    target = AxLocator("button", "Open", scope=Scope(ScopeKind.ROW, "00000"))

    with surface("/members") as page:
        result = page.act(Action(ActionKind.CLICK, target))

    assert result.outcome is Outcome.NOT_FOUND


def test_a_control_with_no_accessible_name_is_reached_through_the_dom(
    surface,
) -> None:
    with surface("/members/12345") as page:
        clicked = page.act(Action(ActionKind.CLICK, LOCK))
        state = page.act(Action(ActionKind.READ, LOCK_STATE))

    assert clicked.outcome is Outcome.OK
    assert state.extracted == "unlocked"


def test_reading_a_cell_inside_a_frame_returns_its_text(surface) -> None:
    balance = DomLocator("td", DomAttribute.TEXT, "$4,212.55", frame=("ledger",))

    with surface("/members/12345") as page:
        result = page.act(Action(ActionKind.READ, balance))

    assert result.outcome is Outcome.OK
    assert result.extracted == "$4,212.55"


def test_a_target_that_matches_nothing_is_refused(surface) -> None:
    with surface("/members") as page:
        result = page.act(Action(ActionKind.CLICK, AxLocator("button", "Wire funds")))

    assert result.outcome is Outcome.NOT_FOUND


@pytest.mark.rule(6)
def test_a_decision_made_against_another_page_is_refused_as_stale(surface) -> None:
    with surface("/members") as page:
        result = page.act(
            Action(ActionKind.CLICK, AxLocator("button", "Search")),
            expect=Expectation(PageState("http://127.0.0.1:1/members")),
        )

    assert result.outcome is Outcome.STALE


def test_an_unlisted_key_is_refused(surface) -> None:
    with surface("/members") as page:
        result = page.act(Action(ActionKind.PRESS_KEY, value="a"))

    assert result.outcome is Outcome.NOT_ACTIONABLE


# Painted controls.


def test_a_painted_control_is_found_again_after_it_moves(surface) -> None:
    """The recorded target is an anchor name. The position is found again."""
    with surface("/members/12345") as page:
        picture = page.observe(VISUAL)
        anchors = [region for region in picture.regions if region.frame == ()]
        assert anchors, "the canvas produced no candidate region"
        anchor = VisualAnchor(anchors[0].anchor_id)
        moved = page.act(Action(ActionKind.CLICK, MOVE))
        assert moved.outcome is Outcome.OK
        result = page.act(Action(ActionKind.CLICK, anchor))
        state = page.act(Action(ActionKind.READ, APPROVAL_STATE))

    assert result.outcome is Outcome.OK
    assert state.extracted == "posted"


def test_a_painted_region_that_is_gone_is_reported_not_found(surface) -> None:
    with surface("/members/12345") as page:
        picture = page.observe(VISUAL)
        anchor = VisualAnchor(picture.regions[0].anchor_id)
        page.act(Action(ActionKind.CLICK, CLEAR))
        result = page.act(Action(ActionKind.CLICK, anchor))

    assert result.outcome is Outcome.NOT_FOUND


def test_a_painted_region_cannot_be_typed_into(surface) -> None:
    with surface("/members/12345") as page:
        picture = page.observe(VISUAL)
        anchor = VisualAnchor(picture.regions[0].anchor_id)
        result = page.act(Action(ActionKind.TYPE, anchor, "text"))

    assert result.outcome is Outcome.NOT_ACTIONABLE


# Session boundaries.


def test_a_navigation_to_an_unlisted_route_never_leaves_the_page(surface) -> None:
    link = AxLocator("link", "Administration")

    with surface("/members/12345") as page:
        result = page.act(Action(ActionKind.CLICK, link))
        landed = page.location()

    assert landed.endswith("/members/12345")
    assert result.outcome is Outcome.BLOCKED
    assert "a navigation outside the profile was blocked" in result.side_effects


def test_a_navigation_into_a_deny_route_never_leaves_the_page(surface) -> None:
    link = AxLocator("link", "Wire funds")

    with surface("/members/12345") as page:
        result = page.act(Action(ActionKind.CLICK, link))
        landed = page.location()

    assert landed.endswith("/members/12345")
    assert result.outcome is Outcome.BLOCKED


def test_a_new_window_is_closed_when_the_profile_denies_one(surface) -> None:
    button = AxLocator("button", "Open statements in a new window")

    with surface("/members/12345") as page:
        clicked = page.act(Action(ActionKind.CLICK, button))
        after = page.act(Action(ActionKind.READ, LOCK_STATE))
        windows = len(page._page.context.pages)

    # Depending on the machine, the popup event can arrive before or after the
    # click returns. One result or the next therefore reports the side effect.
    reported = clicked.side_effects + after.side_effects
    assert reported.count("a new window was closed") == 1
    assert windows == 1


def test_the_session_refuses_an_entry_point_outside_the_profile(
    site: str, site_profile
) -> None:
    from computeruse.surface import SurfaceError

    profile = site_profile()
    with (
        pytest.raises(SurfaceError, match="outside the profile"),
        open_session(profile, f"{site}/admin"),
    ):
        pass
