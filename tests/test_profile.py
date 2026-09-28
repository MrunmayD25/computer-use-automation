"""Check that the profile loader validates input and supplies no defaults."""

import dataclasses
from pathlib import Path

import pytest

from computeruse.profile import (
    ActionKind,
    BrowserScope,
    Environment,
    Limit,
    ObservationMode,
    ProfileError,
    Risk,
    load_profile,
)

TOP_LEVEL_KEYS = (
    "version",
    "profile_id",
    "environment",
    "base_url",
    "allow_routes",
    "deny_routes",
    "allow_new_windows",
    "allow_downloads",
    "actions",
    "perception",
    "records",
    "budgets",
    "secrets",
    "escalation",
    "origins",
    "submissions",
)


def test_example_profile_loads(profile_path: Path) -> None:
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)
    assert profile.scope.origin == "https://sandbox.example.test"
    assert profile.actions[ActionKind.ACCEPT_DIALOG] is Risk.RISKY
    assert profile.actions[ActionKind.CLICK] is Risk.SAFE
    assert profile.secrets["login_password"] == "APP_PASSWORD"


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
def test_every_declaration_is_required(write_profile, key: str) -> None:
    """No key may fall back to a default, because a default would grant access."""
    with pytest.raises(ProfileError, match=key):
        load_profile(write_profile(**{key: None}))


@pytest.mark.rule(1)
def test_unknown_action_type_is_rejected(write_profile) -> None:
    with pytest.raises(ProfileError, match="unknown action type"):
        load_profile(
            write_profile(
                actions={"click": {"any": "safe"}, "wire_funds": {"any": "safe"}}
            )
        )


def test_unknown_risk_label_is_rejected(write_profile) -> None:
    with pytest.raises(ProfileError, match="must be declared safe or risky"):
        load_profile(write_profile(actions={"click": {"any": "probably fine"}}))


def test_unknown_setting_is_rejected(write_profile) -> None:
    with pytest.raises(ProfileError, match="unknown settings"):
        load_profile(write_profile(allow_all=True))


def test_empty_allow_routes_is_rejected(write_profile) -> None:
    with pytest.raises(ProfileError, match="at least one reachable route"):
        load_profile(write_profile(allow_routes=[]))


def test_secret_values_may_not_be_inlined(write_profile) -> None:
    with pytest.raises(ProfileError, match="exactly one env source"):
        load_profile(write_profile(secrets={"login_password": "hunter2"}))


def test_profile_is_immutable(profile_path: Path) -> None:
    """Policy cannot be widened after loading, at runtime as well as statically.

    The type checker also rejects both assignments. The suppressions are
    deliberate because this test covers the runtime rejection.
    """
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.scope.base_url = "https://elsewhere.example.test"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        profile.actions[ActionKind.ACCEPT_DIALOG] = Risk.SAFE  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("path", "permitted"),
    [
        ("/members", True),
        ("/members/12345", True),
        ("/members/12345/savings", True),
        ("/members/12345/savings/detail", True),
        ("/members/12345/savings/wire", False),
        ("/members/12345/extra", False),
        ("/admin", False),
        ("/", False),
    ],
)
def test_route_policy(profile_path: Path, path: str, permitted: bool) -> None:
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)
    assert profile.scope.permits_route(path) is permitted


@pytest.mark.parametrize(
    "path",
    [
        "/members/12345/savings/%77ire",
        "/members/12345/savings/%2577ire",
        "/members/12345/savings/../../../../admin",
        "/members/12345/savings/%2e%2e/%2e%2e/admin",
        "/members/12345/savings/..\\..\\admin",
        "/members/12345/savings/%2fwire",
        "/members//12345/savings/detail",
        "members/12345",
    ],
)
def test_ambiguous_or_encoded_denied_routes_are_refused(
    profile_path: Path, path: str
) -> None:
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)
    assert not profile.scope.permits_route(path)


@pytest.mark.parametrize("key", ["allow_routes", "deny_routes"])
@pytest.mark.parametrize(
    "route",
    [
        "/members/**/only",
        "/members/:",
        "/members/*",
        "/members/../admin",
        "/members//detail",
        "/members?admin",
        "/members/%69d",
    ],
)
def test_invalid_route_templates_are_rejected(
    write_profile, key: str, route: str
) -> None:
    with pytest.raises(ProfileError):
        load_profile(write_profile(**{key: [route]}))


def test_duplicate_yaml_key_cannot_erase_a_deny_list(profile_path: Path) -> None:
    with profile_path.open("a") as stream:
        stream.write("\ndeny_routes: []\n")
    with pytest.raises(ProfileError):
        load_profile(profile_path)


@pytest.mark.parametrize(
    ("section", "declaration"),
    [
        ("actions", "click: risky"),
        ("budgets", "max_steps: 1"),
        ("escalation", "handoff_timeout_s: 1"),
    ],
)
def test_nested_duplicate_keys_are_rejected(
    profile_path: Path, section: str, declaration: str
) -> None:
    content = profile_path.read_text()
    content = content.replace(f"{section}:\n", f"{section}:\n  {declaration}\n")
    profile_path.write_text(content)
    with pytest.raises(ProfileError):
        load_profile(profile_path)


@pytest.mark.parametrize(
    "base_url",
    ["https://[", "https://host:bad", "https://host\\evil", "https://host\x00"],
)
def test_malformed_base_urls_are_profile_errors(write_profile, base_url: str) -> None:
    with pytest.raises(ProfileError):
        load_profile(write_profile(base_url=base_url))


def test_non_string_yaml_keys_are_profile_errors(profile_path: Path) -> None:
    with profile_path.open("a") as stream:
        stream.write("\n123: true\n")
    with pytest.raises(ProfileError):
        load_profile(profile_path)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/members/12345/",
        "/members/%31%32%33%34%35",
        "/members/12345/savings/detail",
    ],
)
def test_canonical_and_unambiguously_encoded_paths_are_allowed(
    write_profile, path: str
) -> None:
    profile = load_profile(
        write_profile(allow_routes=["/", "/members/:id", "/members/:id/savings/**"])
    )
    assert isinstance(profile.scope, BrowserScope)
    assert profile.scope.permits_route(path)


def test_recursive_yaml_merge_is_rejected(profile_path: Path) -> None:
    profile_path.write_text("&policy {<<: *policy}")
    with pytest.raises(ProfileError):
        load_profile(profile_path)


def test_matched_route_returns_the_template_that_admitted_the_path(
    profile_path: Path,
) -> None:
    """A run records the template, so a member identifier stays out of evidence."""
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)

    assert profile.scope.matched_route("/members/12345") == "/members/:id"
    assert (
        profile.scope.matched_route("/members/12345/savings")
        == "/members/:id/savings/**"
    )


def test_matched_route_reports_nothing_for_an_unreachable_path(
    profile_path: Path,
) -> None:
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)

    assert profile.scope.matched_route("/members/12345/savings/wire") is None
    assert profile.scope.matched_route("/admin") is None


# Perception: the schema change that grants an observation tool.


def test_perception_is_loaded_as_declared(profile_path: Path) -> None:
    perception = load_profile(profile_path).perception

    assert perception.allowed_modes == (
        ObservationMode.STRUCTURED,
        ObservationMode.VISUAL,
    )
    assert perception.max_alternate_observations_per_step == 1
    assert ObservationMode.VISUAL in perception.allowed_modes


@pytest.mark.parametrize("version", [1, 2, 3])
def test_a_profile_from_a_previous_schema_is_refused(write_profile, version) -> None:
    """An older profile granted each action type by a bare risk string."""
    with pytest.raises(ProfileError, match="version must be 4"):
        load_profile(write_profile(version=version))


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ({"max_alternate_observations_per_step": 1}, "allowed_modes"),
        ({"allowed_modes": ["structured"]}, "max_alternate_observations_per_step"),
        (
            {
                "allowed_modes": ["structured", "xray"],
                "max_alternate_observations_per_step": 1,
            },
            "unknown observation mode",
        ),
        (
            {
                "allowed_modes": ["visual", "visual"],
                "max_alternate_observations_per_step": 1,
            },
            "must not repeat",
        ),
        (
            {"allowed_modes": [], "max_alternate_observations_per_step": 1},
            "at least one mode",
        ),
        (
            {
                "allowed_modes": ["structured"],
                "max_alternate_observations_per_step": -1,
            },
            "between 0 and",
        ),
        (
            {
                "allowed_modes": ["structured"],
                "max_alternate_observations_per_step": 99,
            },
            "between 0 and",
        ),
        (
            {
                "allowed_modes": ["structured"],
                "max_alternate_observations_per_step": 11,
            },
            "between 0 and 10",
        ),
        (
            {
                "allowed_modes": ["structured"],
                "max_alternate_observations_per_step": True,
            },
            "must be int",
        ),
        (
            {
                "allowed_modes": ["structured"],
                "max_alternate_observations_per_step": 1,
                "screenshot_everything": True,
            },
            "unknown settings",
        ),
    ],
)
def test_perception_declarations_are_validated(
    write_profile, declared: dict, message: str
) -> None:
    with pytest.raises(ProfileError, match=message):
        load_profile(write_profile(perception=declared))


def test_no_alternate_observations_is_a_valid_declaration(write_profile) -> None:
    profile = load_profile(
        write_profile(
            perception={
                "allowed_modes": ["structured"],
                "max_alternate_observations_per_step": 0,
            }
        )
    )

    assert profile.perception.max_alternate_observations_per_step == 0
    assert ObservationMode.VISUAL not in profile.perception.allowed_modes


# Records: which actions must name the record they act on.


def test_records_are_loaded_as_declared(profile_path: Path) -> None:
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)

    assert profile.records.actions == (ActionKind.CLICK, ActionKind.PRESS_KEY)
    assert profile.requires_record(ActionKind.CLICK, "/members/12345/savings/close")
    assert not profile.requires_record(ActionKind.CLICK, "/members/12345")
    assert not profile.requires_record(ActionKind.READ, "/members/12345/savings/x")


def test_an_empty_records_declaration_requires_nothing(write_profile) -> None:
    profile = load_profile(write_profile(records={"actions": [], "routes": []}))

    assert not profile.requires_record(ActionKind.CLICK, "/members/12345")


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ({"routes": ["/members/:id"]}, "actions"),
        ({"actions": ["click"]}, "routes"),
        ({"actions": ["approve"], "routes": []}, "unknown action type"),
        ({"actions": ["navigate"], "routes": []}, "name a control"),
        ({"actions": ["click", "click"], "routes": []}, "must not repeat"),
        ({"actions": ["click"], "routes": ["members"]}, "canonical paths"),
        (
            {"actions": ["click"], "routes": [], "every_click": True},
            "unknown settings",
        ),
    ],
)
def test_records_declarations_are_validated(
    write_profile, declared: dict, message: str
) -> None:
    with pytest.raises(ProfileError, match=message):
        load_profile(write_profile(records=declared))


# Grants and effect exceptions.


def test_record_rules_can_cover_pointed_scroll_input(write_profile) -> None:
    profile = load_profile(
        write_profile(records={"actions": ["scroll"], "routes": ["/members/:id"]})
    )
    assert ActionKind.SCROLL in profile.records.actions


def test_effect_exceptions_are_loaded_as_declared(profile_path: Path) -> None:
    profile = load_profile(profile_path)
    assert isinstance(profile.scope, BrowserScope)

    assert profile.environment is Environment.SANDBOX
    assert profile.effect_limit(ActionKind.CLICK, "submit_payment") is Limit.RISKY
    assert profile.effect_limit(ActionKind.CLICK, "close_account") is Limit.DENY
    assert profile.effect_limit(ActionKind.CLICK, "open_member") is None
    assert profile.effect_limit(ActionKind.READ, "submit_payment") is None


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ({"click": "safe"}, "must grant any"),
        ({"click": {"effects": {"submit_payment": "risky"}}}, "must grant any"),
        (
            {"click": {"any": "safe", "effects": {"submit_payment": "safe"}}},
            "risky or deny",
        ),
        (
            {"click": {"any": "safe", "effects": {"Submit Payment": "risky"}}},
            "lowercase",
        ),
        ({"click": {"any": "safe", "effects": {"any": "risky"}}}, "lowercase"),
        ({"click": {"any": "safe", "effects": ["submit_payment"]}}, "mapping"),
        ({"click": {"any": "safe", "everything": True}}, "unknown settings"),
        ({"observe": {"any": "safe", "effects": {"peek": "risky"}}}, "observe"),
        ({"submit": {"any": "risky"}}, "unknown action type"),
    ],
)
def test_grants_and_exceptions_are_validated(write_profile, declared, message) -> None:
    with pytest.raises(ProfileError, match=message):
        load_profile(write_profile(actions=declared))


@pytest.mark.parametrize("environment", [None, "staging", True])
def test_the_environment_must_be_declared_as_one_of_two(
    write_profile, environment
) -> None:
    with pytest.raises(ProfileError, match="environment"):
        load_profile(write_profile(environment=environment))


def test_a_profile_may_grant_up_to_ten_further_looks_between_actions(write_profile):
    # A page that changes after an action needs several looks to follow.
    perception = {
        "allowed_modes": ["structured"],
        "max_alternate_observations_per_step": 10,
    }
    profile = load_profile(write_profile(perception=perception))
    assert profile.perception.max_alternate_observations_per_step == 10


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("risky", Limit.RISKY), ("deny", Limit.DENY), ("by_effect", None)],
)
def test_a_version_6_profile_says_how_submissions_are_treated(
    edited_profile, declared: str, expected: Limit | None
) -> None:
    rule = edited_profile(submissions=declared).submissions
    assert rule is not None
    assert rule.limit("/members") is expected


def test_an_operator_names_the_screens_whose_submissions_change_nothing(
    edited_profile, write_profile
) -> None:
    rule = edited_profile(
        submissions={"any": "risky", "routes": {"/members": "by_effect"}}
    ).submissions
    assert rule is not None
    assert rule.limit("/members") is None
    assert rule.limit("/members/:id") is Limit.RISKY
    named = edited_profile(
        submissions={"any": "risky", "controls": ["Review"]}
    ).submissions
    assert named is not None
    assert named.limit("/members/:id", "Review") is None
    assert named.limit("/members/:id", "Commit") is Limit.RISKY
    # An exception must name a screen the profile already allows.
    with pytest.raises(ProfileError, match="allow routes"):
        load_profile(
            write_profile(
                submissions={"any": "risky", "routes": {"/other": "by_effect"}}
            )
        )


def test_submissions_must_be_one_of_the_named_rules(write_profile) -> None:
    with pytest.raises(ProfileError, match="submissions"):
        load_profile(write_profile(submissions="safe"))


def test_an_older_profile_cannot_declare_submissions(write_profile) -> None:
    # Versions 4 and 5 judge a submission by its effect, as they always did.
    older = load_profile(write_profile(version=5, submissions=None))
    assert older.submissions is None
    with pytest.raises(ProfileError, match="version 6"):
        load_profile(write_profile(version=5))
