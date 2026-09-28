"""Shared fixtures for the CLI's contract checks and the browser tests."""

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from computeruse.profile import Profile, load_profile

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROFILE = ROOT / "examples" / "profile.yaml"
SITE_PROFILE = ROOT / "evaluation" / "profile.yaml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker that ties a test to a rule in RULES.md."""
    config.addinivalue_line(
        "markers", "rule(number): the RULES.md rule this test verifies"
    )


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    """Return a disposable copy of the shipped example profile."""
    destination = tmp_path / "profile.yaml"
    shutil.copy(EXAMPLE_PROFILE, destination)
    return destination


@pytest.fixture
def write_profile(tmp_path: Path):
    """Return a factory that writes a profile with the given edits applied."""

    def _write(**edits: object) -> Path:
        document = yaml.safe_load(EXAMPLE_PROFILE.read_text())
        for key, value in edits.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value
        destination = tmp_path / "edited.yaml"
        destination.write_text(yaml.safe_dump(document))
        return destination

    return _write


@pytest.fixture
def run_cli():
    """Return a helper that invokes the installed command."""

    def _run(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["computeruse", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={**os.environ, **env},
        )

    return _run


@pytest.fixture
def profile() -> Profile:
    """Return the shipped example profile, loaded and validated."""
    return load_profile(EXAMPLE_PROFILE)


@pytest.fixture
def edited_profile(write_profile):
    """Return a factory that loads the example profile with edits applied."""

    def _load(**edits: object) -> Profile:
        return load_profile(write_profile(**edits))

    return _load


@pytest.fixture(scope="session")
def site() -> Iterator[str]:
    """Serve the evaluation pages on a port the operating system picks."""
    from evaluation.serve import serve

    with serve() as base_url:
        yield base_url


@pytest.fixture
def site_profile(site: str, tmp_path: Path):
    """Return a factory for the evaluation profile, bound to the live port."""

    def _load(**edits: object) -> Profile:
        document = yaml.safe_load(SITE_PROFILE.read_text())
        document["base_url"] = site
        for key, value in edits.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value
        destination = tmp_path / "site.yaml"
        destination.write_text(yaml.safe_dump(document))
        return load_profile(destination)

    return _load


@pytest.fixture
def pages(tmp_path: Path):
    """Return a helper that serves written pages and opens a session on one.

    The profile is the evaluation profile with the base URL moved to the
    written pages, every path allowed, and any edits applied.
    """
    from pages import serve_pages

    from computeruse.browser import open_session

    @contextlib.contextmanager
    def _open(written: dict[str, str], path: str = "/", limits=None, **edits: object):
        with serve_pages(written) as base:
            document = yaml.safe_load(SITE_PROFILE.read_text())
            document["base_url"] = base
            document["allow_routes"] = ["/**"]
            document["deny_routes"] = ["/forbidden/**"]
            for key, value in edits.items():
                document[key] = value
            destination = tmp_path / "pages.yaml"
            destination.write_text(yaml.safe_dump(document))
            profile = load_profile(destination)
            with open_session(profile, f"{base}{path}", limits=limits) as surface:
                yield surface, profile

    return _open
