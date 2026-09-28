"""General contracts for outcomes, value provenance, and peer origins."""

import dataclasses
import json
from pathlib import Path

import pytest
import yaml
from replay_fakes import INPUTS, Clock, Page, People, bank, status
from test_recorder import record, transfer_events
from test_replay import run

from computeruse.branching import Branch, Plan, Recovery, extend
from computeruse.capability import (
    LocatorForm,
    Present,
    StructuralTarget,
    constant,
    dumps,
)
from computeruse.escalation import HandoffOutcome
from computeruse.profile import ActionKind, ProfileError, Risk, load_profile
from computeruse.replay import Status


@pytest.fixture
def profile(tmp_path):
    from replay_fakes import PROFILE

    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


def test_discovered_linear_workflow_accepts_an_explicit_business_branch(profile):
    capability = record(profile, transfer_events()).capability
    assert capability is not None
    before = dumps(capability)
    target = StructuralTarget(
        "missing",
        "/members",
        LocatorForm.ACCESSIBILITY,
        "status",
        constant("No member found"),
        "",
        None,
        None,
        (),
        None,
    )
    plan = Plan(
        1,
        (target,),
        (Branch("s2", (Present("missing"),), Recovery.OUTCOME, "member_missing", 1),),
    )
    extended = extend(capability, plan)
    app = bank()
    from computeruse.profile import ActionKind

    app.rules[("search", ActionKind.CLICK, "Search")] = "not_found"
    from computeruse.capability import Review

    extended = dataclasses.replace(
        extended,
        provenance=dataclasses.replace(extended.provenance, review=Review.REVIEWED),
    )
    result, _ = run(
        extended,
        app,
        People([]),
        profile,
        inputs={**INPUTS, "member_id": "99999"},
        clock=Clock(),
    )
    assert result.status is Status.OUTCOME
    assert result.outcome == "member_missing"
    assert dumps(capability) == before


def test_review_cli_adds_a_branch_to_a_separate_draft(profile, run_cli, tmp_path):
    from computeruse.capability import _encode, load_capability

    capability = record(profile, transfer_events()).capability
    assert capability is not None
    source = tmp_path / "source.json"
    source.write_text(dumps(capability))
    before = source.read_bytes()
    target = StructuralTarget(
        "missing",
        "/members",
        LocatorForm.ACCESSIBILITY,
        "status",
        constant("No member found"),
        "",
        None,
        None,
        (),
        None,
    )
    plan = Plan(
        1,
        (target,),
        (Branch("s2", (Present("missing"),), Recovery.OUTCOME, "member_missing", 1),),
    )
    authored = tmp_path / "branches.json"
    authored.write_text(json.dumps(_encode(plan)))
    output = tmp_path / "extended.json"
    result = run_cli(
        "review",
        "--capability",
        str(source),
        "--branches",
        str(authored),
        "--output",
        str(output),
        OPENAI_API_KEY="",
    )
    assert result.returncode == 0, result.stderr
    assert "member_missing" in load_capability(output).outcomes
    assert source.read_bytes() == before


def test_peer_origins_require_explicit_routes_and_keep_deny_precedence(tmp_path):
    document = yaml.safe_load(Path("evaluation/profile.yaml").read_text())
    document["version"] = 5
    document["origins"] = [
        {
            "origin": "http://127.0.0.1:9001",
            "allow_routes": ["/desk/**"],
            "deny_routes": ["/desk/delete"],
        }
    ]
    path = tmp_path / "multi.yaml"
    path.write_text(yaml.safe_dump(document))
    profile = load_profile(path)
    assert (
        profile.scope.route("http://127.0.0.1:9001/desk/member")
        == "http://127.0.0.1:9001/desk/**"
    )
    assert profile.scope.route("http://127.0.0.1:9001/desk/delete") is None
    assert profile.scope.route("http://127.0.0.1:9002/desk/member") is None
    del document["origins"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ProfileError):
        load_profile(path)


def test_startup_navigation_rechecks_policy_before_testing_entry_markers(profile):
    from replay_fakes import transfer_capability

    app = bank()

    def ready(surface):
        if surface.looks == 1:
            surface.screens["search"].path = "/members#ready"

    app.on_observe = ready
    result, _ = run(
        transfer_capability(), app, People([HandoffOutcome.APPROVED]), profile
    )
    assert result.status is Status.SUCCEEDED
    assert app.looks >= 2


@pytest.mark.parametrize("settles", [False, True])
def test_authored_wait_recovery_never_repeats_the_original_operation(profile, settles):
    capability = record(profile, transfer_events()).capability
    assert capability is not None
    target = StructuralTarget(
        "loading",
        "/members",
        LocatorForm.ACCESSIBILITY,
        "status",
        constant("Loading"),
        "",
        None,
        None,
        (),
        None,
    )
    extended = extend(
        capability,
        Plan(
            1, (target,), (Branch("s2", (Present("loading"),), Recovery.WAIT, "", 2),)
        ),
    )
    app = bank()
    app.screens["loading"] = Page("/members", (status("Loading"),))
    app.rules[("search", ActionKind.CLICK, "Search")] = "loading"
    if settles:

        def ready(surface):
            if surface.state == "loading" and surface.looks >= 5:
                surface.state = "member"

        app.on_observe = ready
    from computeruse.capability import Review

    extended = dataclasses.replace(
        extended,
        provenance=dataclasses.replace(extended.provenance, review=Review.REVIEWED),
    )
    profile = dataclasses.replace(
        profile, actions={**profile.actions, ActionKind.WAIT: Risk.SAFE}
    )
    result, _ = run(
        extended,
        app,
        People([HandoffOutcome.APPROVED, HandoffOutcome.APPROVED]),
        profile,
        inputs=INPUTS,
        clock=Clock(),
    )
    assert app.acted_on("Search") == 1
    assert result.status is (Status.SUCCEEDED if settles else Status.RECOVERY_EXHAUSTED)


def test_invalid_peer_address_has_a_safe_profile_error(tmp_path):
    document = yaml.safe_load(Path("evaluation/profile.yaml").read_text())
    document.update(
        version=5,
        origins=[
            {
                "origin": "https://user:private-value@example.test",
                "allow_routes": ["/**"],
                "deny_routes": [],
            }
        ],
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ProfileError) as error:
        load_profile(path)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("restored", [False, True])
def test_authored_human_recovery_requires_a_checked_continuation(profile, restored):
    capability = record(profile, transfer_events()).capability
    assert capability is not None
    target = StructuralTarget(
        "loading",
        "/members",
        LocatorForm.ACCESSIBILITY,
        "status",
        constant("Loading"),
        "",
        None,
        None,
        (),
        None,
    )
    extended = extend(
        capability,
        Plan(
            1, (target,), (Branch("s2", (Present("loading"),), Recovery.HUMAN, "", 2),)
        ),
    )
    from computeruse.capability import Review

    extended = dataclasses.replace(
        extended,
        provenance=dataclasses.replace(extended.provenance, review=Review.REVIEWED),
    )
    app = bank()
    app.screens["loading"] = Page("/members", (status("Loading"),))
    app.rules[("search", ActionKind.CLICK, "Search")] = "loading"

    def resume(_request):
        if restored:
            app.state = "member"
        return HandoffOutcome.RESUMED

    people = People([resume, HandoffOutcome.APPROVED, HandoffOutcome.APPROVED])
    result, _ = run(extended, app, people, profile, clock=Clock())
    assert app.acted_on("Search") == 1
    assert (result.status is Status.SUCCEEDED) is restored
    if not restored:
        assert app.acted_on("Transfer funds") == 0
