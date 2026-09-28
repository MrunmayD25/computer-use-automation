"""Verify retained evidence files against their manifest and six-case index.

Structured command logs anonymize capability identifiers. Matching journal and
log events establish a consistent recorded run, not the concrete artifact's
identity. Manifest hashes establish file integrity, while independent result
checks remain necessary to establish business correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from computeruse.capability import Review, load_capability, validate

CASES = frozenset(
    f"{site}-{task}"
    for site in ("responsive", "components", "canvas")
    for task in ("lookup", "write")
)


def verify(folder: Path) -> tuple[int, int]:
    """Check file integrity and require artifacts for successful replay claims.

    Parameters
    ----------
    folder
        Evidence directory containing manifest.json and summary.json.

    Returns
    -------
    tuple[int, int]
        Number of retained files and successful replay journals checked.

    Raises
    ------
    ValueError
        A file changed, a case is missing, or a success lacks its artifact.
    """
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if (
        summary["status"] != "complete"
        or len(summary["cases"]) != len(CASES)
        or {case["id"] for case in summary["cases"]} != CASES
    ):
        raise ValueError("the index must cover all three sites and both tasks")
    for name, expected in manifest["files"].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"file is outside the evidence folder: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"file checksum differs: {name}")
    successes = 0
    for case in summary["cases"]:
        place = folder / case["id"]
        _check_discovery(place, case)
        for journal in place.rglob("replay-[0-9]*.jsonl"):
            if journal.name.endswith(".human.jsonl"):
                continue
            successes += _check_replay(journal)
    return len(manifest["files"]), successes


def _check_discovery(place: Path, case: dict[str, object]) -> None:
    for required in ("discovery.log", "discovery.jsonl"):
        if not (place / required).is_file():
            raise ValueError(f"missing {case['id']}/{required}")
    events = [
        json.loads(line)
        for line in (place / "discovery.jsonl").read_text().splitlines()
    ]
    endings = [event for event in events if event.get("event") == "RunEnded"]
    if not endings or endings[-1] != case["discovery"]:
        raise ValueError(f"discovery result differs from the index: {case['id']}")
    for filename, claimed in (
        ("capability.draft.json", case["capability_exported"]),
        ("capability.json", case["approved"]),
    ):
        path = place / filename
        if path.is_file() != claimed:
            raise ValueError(f"artifact differs from the index: {path}")
        if claimed:
            capability = load_capability(path)
            if validate(capability):
                raise ValueError(f"invalid artifact: {path}")
            if (
                filename == "capability.json"
                and capability.provenance.review is not Review.REVIEWED
            ):
                raise ValueError(f"artifact is not approved: {path}")


def _check_replay(journal: Path) -> bool:
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    endings = [event for event in events if event.get("event") == "ReplayEnded"]
    if not endings:
        raise ValueError(f"replay has no final result: {journal}")
    if endings[-1]["status"] != "succeeded":
        return False
    capability = load_capability(journal.parent / "capability.json")
    if validate(capability) or capability.provenance.review is not Review.REVIEWED:
        raise ValueError(f"invalid capability for {journal}")
    started = events[0]
    if (
        started.get("event") != "ReplayStarted"
        or started.get("version") != capability.version
    ):
        raise ValueError(f"replay has no matching capability version: {journal}")
    log = journal.with_suffix(".log").read_text()
    if log.lstrip().startswith("{"):
        if started.get("capability") not in {
            capability.capability_id,
            "capability_1",
        }:
            raise ValueError(f"unexpected replay capability identifier: {journal}")
        _check_command_log(journal, log, started, endings[-1])
    elif (
        started.get("capability") != capability.capability_id
        or "replay succeeded:" not in log
    ):
        raise ValueError(f"success has no matching terminal log: {journal}")
    return True


def _check_command_log(
    journal: Path,
    log: str,
    started: dict[str, object],
    ended: dict[str, object],
) -> None:
    records = [json.loads(line) for line in log.splitlines()]
    commands = [
        record
        for record in records
        if record.get("event") in {"CommandStarted", "CommandEnded"}
    ]
    expected = [
        {"command": "replay", "event": "CommandStarted"},
        {"code": 0, "event": "CommandEnded"},
    ]
    if commands != expected or [records[0], records[-1]] != expected:
        raise ValueError(f"replay command did not complete successfully: {journal}")
    boundaries = [
        record
        for record in records
        if record.get("event") in {"ReplayStarted", "ReplayEnded"}
    ]
    if boundaries != [started, ended]:
        raise ValueError(f"replay result differs from the command log: {journal}")


def main() -> None:
    """Check an evidence folder without a browser, model key, or server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    files, successes = verify(args.folder)
    print(
        f"Verified {files} files, six attempts, and "
        f"{successes} replay journals reporting success."
    )
    if args.verbose:
        print("Files are intact. Business results are in summary.json.")


if __name__ == "__main__":
    main()
