"""Unpack and run the three target applications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evaluation/sites/full-v1.zip"
DESTINATION = ROOT / "evaluation/runtime"
SOURCE = DESTINATION / "demo-sources"
DATABASE = ROOT / ".runtime/evaluation.sqlite"
SEED = ROOT / ".runtime/evaluation.seed.sqlite"
"""The seeded database, never served. Each evaluation trial serves a fresh
copy of it at ``DATABASE``, so records a previous run opened are gone."""
APPS = {
    "web-component-operations": 4360,
    "white-label-responsive": 4363,
    "canvas-teller": 4390,
}
"""The three target applications and their ports. Two are served from the
archive. The canvas teller is in ``evaluation/canvas``, drawn on one canvas
over the same financial service."""


def server(app: str) -> tuple[list[str], Path]:
    """Return the command that serves an application, and where it runs."""
    port = str(APPS[app])
    if app == "canvas-teller":
        services = SOURCE / "environments/database/financial/dist/services.js"
        return (
            [
                "node",
                str(ROOT / "evaluation/canvas/server.mjs"),
                "--database",
                str(DATABASE),
                "--services",
                str(services),
                "--port",
                port,
            ],
            ROOT,
        )
    command = ["npm", "start", "-w", app, "--", "--database", str(DATABASE)]
    return [*command, "--port", port], SOURCE / "environments"


def unpack() -> Path:
    """Validate and extract the ready-to-run application sources."""
    manifest = json.loads((ARCHIVE.parent / "manifest.json").read_text())
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("evaluation archive checksum mismatch")
    if not SOURCE.exists():
        extract()
    return SOURCE


def extract() -> None:
    """Extract the archive after refusing unsafe or corrupt entries."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("unsafe archive entry")
        if archive.testzip() is not None:
            raise ValueError("evaluation archive is corrupt")
        archive.extractall(DESTINATION)


def environment(node: Path | None) -> dict[str, str]:
    """Require the runtime the financial service declares."""
    executable = str(node) if node else shutil.which("node")
    if executable is None:
        raise ValueError("Node.js 25.9 or newer is required")
    version = subprocess.check_output(  # noqa: S603  operator-selected runtime
        [executable, "--version"], text=True
    ).strip()
    major, minor, *_ = [int(part) for part in version.removeprefix("v").split(".")]
    if (major, minor) < (25, 9):
        raise ValueError("Node.js 25.9 or newer is required. Pass --node PATH")
    return {
        **os.environ,
        "PATH": str(Path(executable).resolve().parent)
        + os.pathsep
        + os.environ["PATH"],
    }


def fresh() -> None:
    """Replace the served database with a copy of the seeded one."""
    if not SEED.exists():
        raise ValueError("seed the evaluation database first: sites.py seed")
    for suffix in ("-wal", "-shm"):
        DATABASE.with_name(DATABASE.name + suffix).unlink(missing_ok=True)
    shutil.copyfile(SEED, DATABASE)


def execute(command: list[str], env: dict[str, str], cwd: Path = SOURCE) -> None:
    """Run a fixture command with argument boundaries preserved."""
    subprocess.run(command, cwd=cwd, env=env, check=True)  # noqa: S603  local fixture tools


def main() -> None:
    """List, install, seed or start one of the target applications."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("list", "unpack", "install", "seed", "start")
    )
    parser.add_argument("--app", choices=APPS)
    parser.add_argument("--node", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        for app, port in APPS.items():
            print(f"{app}: http://127.0.0.1:{port}/")
        return
    unpack()
    if args.command == "unpack":
        print(SOURCE)
        return
    env = environment(args.node)
    if args.command == "install":
        execute(["npm", "ci", "--prefix", "environments/database/financial"], env)
        execute(["npm", "ci", "--prefix", "environments"], env)
        execute(["npm", "--prefix", "environments", "run", "build"], env)
    elif args.command == "seed":
        execute(
            [
                "node",
                "environments/database/financial/dist/cli.js",
                "seed",
                "--path",
                str(SEED),
                "--scale",
                "small",
                "--verbose",
            ],
            env,
        )
        fresh()
    elif args.app:
        command, where = server(args.app)
        execute(command, env, where)
    else:
        parser.error("start requires --app")


if __name__ == "__main__":
    main()
