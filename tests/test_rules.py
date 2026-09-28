"""Keep RULES.md synchronized with the tests that verify it.

Every rule needs a test marked ``@pytest.mark.rule(N)``. An enforced rule
needs at least one marked test that is expected to pass. A marked expected
failure must be strict. A fix that makes it pass then exposes a stale rule
status. The checker reads markers from the test sources with ``ast``, so a
test does not have to run to be counted.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "RULES.md"
STATUSES = frozenset({"enforced", "partial", "planned"})


@dataclasses.dataclass(frozen=True)
class Marked:
    """One test that names a rule, and whether it is a strict expected failure."""

    where: str
    rule: int
    expected_failure: bool
    strict: bool


def rules() -> dict[int, str]:
    """Return each rule's number and the first word of its status."""
    found: dict[int, str] = {}
    number: int | None = None
    for line in RULES.read_text().splitlines():
        if line.startswith("## Rule "):
            number = int(line.removeprefix("## Rule ").split(".", 1)[0])
            found[number] = ""
        elif line.startswith("Status: ") and number is not None:
            found[number] = line.removeprefix("Status: ").split()[0].strip(".,")
            number = None
    return found


def _mark(decorator: ast.expr, name: str) -> ast.Call | None:
    """Return ``decorator`` when it is a call of ``pytest.mark.<name>``."""
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == name
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mark"
    ):
        return decorator
    return None


def marked() -> list[Marked]:
    """Read every ``pytest.mark.rule`` in the test sources."""
    found: list[Marked] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            failure = next(
                (
                    call
                    for decorator in node.decorator_list
                    if (call := _mark(decorator, "xfail")) is not None
                ),
                None,
            )
            strict = failure is not None and any(
                keyword.arg == "strict"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in failure.keywords
            )
            for decorator in node.decorator_list:
                call = _mark(decorator, "rule")
                if call is None:
                    continue
                if not call.args:
                    raise TypeError(f"{path.name}:{node.lineno} names no rule")
                for argument in call.args:
                    if not (
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, int)
                        and not isinstance(argument.value, bool)
                    ):
                        raise TypeError(f"{path.name}:{node.lineno} names no rule")
                    found.append(
                        Marked(
                            f"{path.name}::{node.name}",
                            argument.value,
                            failure is not None,
                            strict,
                        )
                    )
    return found


def test_every_number_in_a_rule_marker_is_tracked(tmp_path, monkeypatch):
    import sys

    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "test_example.py").write_text(
        "@pytest.mark.rule(7, 17)\n"
        "@pytest.mark.xfail(strict=True, raises=AssertionError)\n"
        "def test_example():\n"
        "    assert False\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    found = marked()
    assert {item.rule for item in found} == {7, 17}
    assert all(item.expected_failure and item.strict for item in found)


@pytest.mark.parametrize("arguments", ["", "7, 'invalid'", "7, True"])
def test_an_invalid_rule_number_cannot_hide_after_a_valid_one(
    tmp_path, monkeypatch, arguments
):
    import sys

    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "test_example.py").write_text(
        f"@pytest.mark.rule({arguments})\ndef test_example():\n    pass\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(TypeError):
        marked()


def test_every_rule_has_a_known_status() -> None:
    found = rules()
    assert found, "RULES.md holds no rules"
    assert sorted(found) == list(range(1, len(found) + 1))
    unknown = {
        number: status for number, status in found.items() if status not in STATUSES
    }
    assert not unknown, unknown


def test_every_marker_names_a_rule() -> None:
    known = rules()
    unknown = [item.where for item in marked() if item.rule not in known]
    assert not unknown, unknown


def test_every_rule_has_a_test() -> None:
    covered = {item.rule for item in marked()}
    missing = [number for number in rules() if number not in covered]
    assert not missing, f"rules without a marked test: {missing}"


def test_an_enforced_rule_has_a_test_expected_to_pass() -> None:
    passing = {item.rule for item in marked() if not item.expected_failure}
    only_failures = [
        number
        for number, status in rules().items()
        if status == "enforced" and number not in passing
    ]
    assert not only_failures, only_failures


def test_every_expected_failure_is_strict() -> None:
    loose = [
        item.where for item in marked() if item.expected_failure and not item.strict
    ]
    assert not loose, loose


@pytest.mark.rule(20)
def test_no_evaluation_value_appears_in_product_source() -> None:
    inventory = ROOT / "evaluation" / "sites" / "fixture-values.yaml"
    values = yaml.safe_load(inventory.read_text())["values"]
    assert values
    source = {
        path: path.read_text() for path in (ROOT / "src" / "computeruse").rglob("*.py")
    }
    found = [
        f"{path.relative_to(ROOT)}: {value}"
        for path, text in source.items()
        for value in values
        if value in text
    ]
    assert not found, found


def test_every_inventory_value_is_still_used_by_the_evaluation() -> None:
    inventory = ROOT / "evaluation" / "sites" / "fixture-values.yaml"
    values = yaml.safe_load(inventory.read_text())["values"]
    runtime = ROOT / "evaluation" / "runtime"
    texts = [
        path.read_text(errors="replace")
        for pattern in ("*.py", "*.yaml", "*.json")
        for path in (ROOT / "evaluation").rglob(pattern)
        if path != inventory and runtime not in path.parents
    ]
    unused = [value for value in values if not any(value in text for text in texts)]
    assert not unused, unused
