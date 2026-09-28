"""Exercise the installed CLI's input contract and policy reporting."""

import json
from pathlib import Path

import pytest

ENTRY = "https://sandbox.example.test/members/12345"


def test_help(run_cli) -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    for option in ("--goal", "--website", "--profile", "--verbose"):
        assert option in result.stdout


def test_discover_exposes_visual_mode_and_model_choice(run_cli):
    result = run_cli("discover", "--help")
    assert result.returncode == 0
    assert "--visual" in result.stdout
    assert "--model" in result.stdout


@pytest.mark.parametrize("option", ["--contract", "--reviewed-visual-template"])
def test_invalid_export_declarations_fail_before_model_or_browser(
    run_cli, profile_path, tmp_path, option
):
    invalid = tmp_path / "declaration"
    invalid.write_text("private-invalid-data")
    value = (
        "2=" + str(invalid) if option == "--reviewed-visual-template" else str(invalid)
    )
    result = run_cli(
        "discover",
        "--goal",
        "Task",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        option,
        value,
        OPENAI_API_KEY="",
    )
    assert result.returncode == 2
    assert "invalid_declarations" in result.stderr
    assert "private-invalid-data" not in result.stderr
    assert "OPENAI_API_KEY" not in result.stderr


def test_visual_flag_cannot_grant_a_mode_the_profile_denies(run_cli, write_profile):
    profile = write_profile(
        perception={
            "allowed_modes": ["structured"],
            "max_alternate_observations_per_step": 1,
        }
    )
    result = run_cli(
        "discover",
        "--goal",
        "test",
        "--website",
        ENTRY,
        "--profile",
        str(profile),
        "--visual",
        OPENAI_API_KEY="",
    )
    assert result.returncode == 2
    assert "invalid_policy" in result.stderr


@pytest.mark.parametrize("missing", ["--goal", "--website", "--profile"])
def test_required_inputs(run_cli, profile_path: Path, missing: str) -> None:
    inputs = {"--goal": "Task", "--website": ENTRY, "--profile": str(profile_path)}
    args = [
        item for key, value in inputs.items() if key != missing for item in (key, value)
    ]
    result = run_cli(*args)
    assert result.returncode == 2
    assert "invalid_arguments" in result.stderr


def test_reports_policy_without_persisting_goal_or_record_url(
    run_cli, profile_path: Path
) -> None:
    result = run_cli(
        "--goal",
        "Read the savings balance PRIVATE-GOAL-7391",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
    )
    assert result.returncode == 0
    assert "PRIVATE-GOAL-7391" not in result.stdout + result.stderr
    assert "/members/12345" not in result.stdout + result.stderr
    assert "PolicyAccepted" in result.stdout


def test_reports_declared_risk_per_action(run_cli, profile_path: Path) -> None:
    result = run_cli(
        "--goal", "Task", "--website", ENTRY, "--profile", str(profile_path)
    )
    accepted = next(
        json.loads(line)
        for line in result.stdout.splitlines()
        if json.loads(line)["event"] == "PolicyAccepted"
    )
    assert "click" in accepted["safe_actions"]
    assert "accept_dialog" not in accepted["safe_actions"]
    assert "accept_dialog" in accepted["risky_actions"]


def test_secret_values_are_never_read(run_cli, profile_path: Path) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        "--verbose",
        APP_PASSWORD="correct-horse-battery",
    )
    assert result.returncode == 0
    assert "<login_password>" in result.stdout
    assert "correct-horse-battery" not in result.stdout + result.stderr


def test_entry_point_outside_allow_routes_is_refused(
    run_cli, profile_path: Path
) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        "https://sandbox.example.test/admin",
        "--profile",
        str(profile_path),
    )
    assert result.returncode == 2
    assert "invalid_policy" in result.stderr
    assert "/admin" not in result.stderr


def test_deny_route_wins_over_allow_route(run_cli, profile_path: Path) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        "https://sandbox.example.test/members/12345/savings/wire",
        "--profile",
        str(profile_path),
    )
    assert result.returncode == 2
    assert "invalid_policy" in result.stderr


def test_website_outside_profile_origin_is_refused(run_cli, profile_path: Path) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        "https://elsewhere.example.test/members/12345",
        "--profile",
        str(profile_path),
    )
    assert result.returncode == 2
    assert "invalid_policy" in result.stderr


@pytest.mark.parametrize(
    "website",
    [
        "example.com",
        "file:///tmp/x",
        "https://user:secret@sandbox.example.test",
        "https://sandbox.example.test:bad",
        "https://sandbox.example.test:0",
        "https://sandbox.example.test\\evil",
        "https://sandbox.example.test\x01",
    ],
)
def test_invalid_website(run_cli, profile_path: Path, website: str) -> None:
    result = run_cli(
        "--goal", "Task", "--website", website, "--profile", str(profile_path)
    )
    assert result.returncode == 2
    assert "invalid_arguments" in result.stderr


def test_missing_profile_file(run_cli, tmp_path: Path) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        ENTRY,
        "--profile",
        str(tmp_path / "absent.yaml"),
    )
    assert result.returncode == 2
    assert "invalid_arguments" in result.stderr


@pytest.mark.parametrize(
    "contents", [b"\xff", b"version: [secret-marker\n", b"base_url: https://[\n"]
)
def test_malformed_profile_has_safe_diagnostics(
    run_cli, profile_path: Path, contents: bytes
) -> None:
    profile_path.write_bytes(contents)
    result = run_cli(
        "--goal",
        "private-goal",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        "--verbose",
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "secret-marker" not in result.stderr
    assert "private-goal" not in result.stderr


@pytest.mark.parametrize(
    "path",
    ["/members/12345/savings/%77ire", "/members/12345/savings/../../../../admin"],
)
def test_noncanonical_entry_cannot_bypass_policy(
    run_cli, profile_path: Path, path: str
) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        f"https://sandbox.example.test{path}",
        "--profile",
        str(profile_path),
    )
    assert result.returncode == 2


def test_equivalent_origin_is_accepted(run_cli, profile_path: Path) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        "https://SANDBOX.EXAMPLE.TEST:443/members/12345",
        "--profile",
        str(profile_path),
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    "edits",
    [
        {"actions": {"secret-marker": "safe"}},
        {"actions": {"click": "secret-marker"}},
        {"secret-marker": True},
        {"secrets": {"secret-marker": "inline-value"}},
    ],
)
def test_profile_validation_does_not_echo_contents(
    run_cli, write_profile, edits: dict[str, object]
) -> None:
    result = run_cli(
        "--goal",
        "Task",
        "--website",
        ENTRY,
        "--profile",
        str(write_profile(**edits)),
        "--verbose",
    )
    assert result.returncode == 2
    assert "secret-marker" not in result.stderr
    assert "inline-value" not in result.stderr
    assert "PolicyAccepted" not in result.stdout


def test_oversized_profile_is_refused(run_cli, profile_path: Path) -> None:
    profile_path.write_bytes(b"#" * (1024 * 1024 + 1))
    result = run_cli(
        "--goal", "Task", "--website", ENTRY, "--profile", str(profile_path)
    )
    assert result.returncode == 2
    assert "invalid_policy" in result.stderr


def test_fifo_profile_is_refused_without_opening(run_cli, tmp_path: Path) -> None:
    import os

    path = tmp_path / "profile.fifo"
    os.mkfifo(path)
    result = run_cli("--goal", "Task", "--website", ENTRY, "--profile", str(path))
    assert result.returncode == 2
    assert "invalid_arguments" in result.stderr


def test_reports_the_permitted_observation_tools(run_cli, profile_path: Path) -> None:
    result = run_cli(
        "--goal", "Task", "--website", ENTRY, "--profile", str(profile_path)
    )

    accepted = next(
        json.loads(line)
        for line in result.stdout.splitlines()
        if json.loads(line)["event"] == "PolicyAccepted"
    )
    assert accepted["observation_modes"] == ["structured", "visual"]
    assert accepted["alternate_observations"] == 1


def test_discover_is_a_separate_command(run_cli) -> None:
    result = run_cli("--help")

    assert "discover" in result.stdout
    assert "Drive a real browser" in result.stdout


def test_validating_inputs_does_not_start_a_browser(
    run_cli, profile_path: Path
) -> None:
    """The default command reports policy and opens nothing."""
    result = run_cli(
        "--goal", "Task", "--website", ENTRY, "--profile", str(profile_path)
    )

    assert result.returncode == 0
    assert "PolicyAccepted" in result.stdout
    assert "Opening a browser" not in result.stdout


def test_discover_refuses_to_start_without_a_credential(
    run_cli, profile_path: Path
) -> None:
    result = run_cli(
        "discover",
        "--goal",
        "Task",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        OPENAI_API_KEY="",
    )

    assert result.returncode == 2
    assert "missing_credential" in result.stderr
    assert "Opening a browser" not in result.stdout


CONTRACT = Path("evaluation/sites/member-status.contract.json").resolve()


@pytest.mark.rule(9)
def test_explicit_inputs_must_name_every_contract_input(
    run_cli, profile_path: Path
) -> None:
    # Without --input, the model fills values from the goal. Supplying one
    # --input requires all inputs.
    result = run_cli(
        "discover",
        "--goal",
        "Look up member 12345.",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        "--contract",
        str(CONTRACT),
        "--input",
        "member_id=12345",
        OPENAI_API_KEY="",
    )

    assert result.returncode == 2
    assert "invalid_inputs" in result.stderr
    assert "12345" not in result.stderr


def test_a_contract_without_inputs_goes_on_to_fill_them_from_the_goal(
    run_cli, profile_path: Path
) -> None:
    result = run_cli(
        "discover",
        "--goal",
        "Look up member 12345.",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        "--contract",
        str(CONTRACT),
        OPENAI_API_KEY="",
    )

    # The contract check passes. The run stops only for the missing key.
    assert result.returncode == 2
    assert "missing_credential" in result.stderr


def test_discover_refuses_an_entry_point_outside_the_profile(
    run_cli, profile_path: Path
) -> None:
    result = run_cli(
        "discover",
        "--goal",
        "Send a wire",
        "--website",
        "https://sandbox.example.test/members/12345/savings/wire",
        "--profile",
        str(profile_path),
        OPENAI_API_KEY="unused-by-this-path",
    )

    assert result.returncode == 2
    assert "invalid_policy" in result.stderr


def test_discover_refuses_a_journal_path_it_cannot_open(
    run_cli, profile_path: Path, tmp_path: Path
) -> None:
    result = run_cli(
        "discover",
        "--goal",
        "Task",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        "--journal",
        str(tmp_path / "no-such-directory" / "run.jsonl"),
        OPENAI_API_KEY="unused-by-this-path",
    )

    assert result.returncode == 2
    assert "journal_unavailable" in result.stderr


def test_discover_help_says_a_headed_run_opens_the_control_window(run_cli):
    result = run_cli("discover", "--help")

    assert result.returncode == 0
    assert "control window beside it" in " ".join(result.stdout.split())


def _discover_without_a_terminal(*args: str):
    """Run discover with standard input closed off, so no terminal can answer."""
    import os
    import subprocess

    return subprocess.run(
        ["computeruse", "discover", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={**os.environ, "OPENAI_API_KEY": ""},
    )


@pytest.mark.parametrize("headed", [False, True])
def test_a_run_without_a_terminal_reports_inputs_then_stops_for_a_missing_key(
    profile_path: Path, headed: bool
) -> None:
    result = _discover_without_a_terminal(
        "--goal",
        "Read the savings balance",
        "--website",
        ENTRY,
        "--profile",
        str(profile_path),
        *(["--headed"] if headed else []),
    )

    assert result.returncode == 2
    assert "Read the savings balance" not in result.stdout
    assert "PolicyAccepted" in result.stdout
    assert "missing_credential" in result.stderr
    assert "Opening the authorized session" not in result.stdout
    assert "Press Start" not in result.stdout
    assert "Traceback" not in result.stderr


def test_saved_text_is_confirmed_by_a_second_record_not_listed_per_run(run_cli):
    result = run_cli("discover", "--help")

    shown = " ".join(result.stdout.split())
    assert "--confirm-input" in shown
    assert "second test record" in shown
    assert "--safe-text" not in shown
