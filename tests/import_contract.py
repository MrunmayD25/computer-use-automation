"""Shared dependency checks for protected product modules."""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True, slots=True)
class ImportContract:
    """Allowed dependency edges for one protected module."""

    first_party: frozenset[str]
    external_roots: frozenset[str] = frozenset()


_CONTRACTS = {
    "computeruse.loop": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.budget",
                "computeruse.control",
                "computeruse.coverage",
                "computeruse.decider",
                "computeruse.diagnostics",
                "computeruse.escalation",
                "computeruse.journal",
                "computeruse.manual",
                "computeruse.matching",
                "computeruse.operations",
                "computeruse.policy",
                "computeruse.profile",
                "computeruse.recording",
                "computeruse.surface",
            }
        )
    ),
    "computeruse.control": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.budget",
                "computeruse.escalation",
                "computeruse.journal",
                "computeruse.manual",
                "computeruse.operations",
                "computeruse.policy",
                "computeruse.profile",
                "computeruse.surface",
            }
        )
    ),
    "computeruse.manual": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.escalation",
                "computeruse.policy",
                "computeruse.profile",
            }
        )
    ),
    "computeruse.capability": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.matching",
                "computeruse.operations",
                "computeruse.policy",
                "computeruse.profile",
                "computeruse.urls",
            }
        ),
        frozenset({"PIL"}),
    ),
    "computeruse.recorder": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.capability",
                "computeruse.matching",
                "computeruse.operations",
                "computeruse.policy",
                "computeruse.profile",
            }
        )
    ),
    "computeruse.replay": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.budget",
                "computeruse.capability",
                "computeruse.control",
                "computeruse.coverage",
                "computeruse.describe",
                "computeruse.diagnostics",
                "computeruse.escalation",
                "computeruse.journal",
                "computeruse.manual",
                "computeruse.matching",
                "computeruse.operations",
                "computeruse.policy",
                "computeruse.profile",
                "computeruse.reading",
                "computeruse.replay_control",
                "computeruse.retarget",
                "computeruse.surface",
            }
        )
    ),
    "computeruse.replay_control": ImportContract(
        frozenset(
            {
                "computeruse.control",
                "computeruse.escalation",
                "computeruse.profile",
                "computeruse.replay",
            }
        )
    ),
    "computeruse.retarget": ImportContract(
        frozenset(
            {
                "computeruse",
                "computeruse.actions",
                "computeruse.capability",
                "computeruse.matching",
                "computeruse.reading",
                "computeruse.surface",
                "computeruse.visual",
            }
        )
    ),
}

_ADAPTER_OWNERS = {
    "computeruse.browser": "Use Surface; browser.py owns browser sessions.",
    "computeruse.model": "Use Decider; model.py owns provider calls.",
    "computeruse.native": "Use Control; native.py is a command channel.",
    "computeruse.panel": "Use Control; panel.py is a command channel.",
    "computeruse.terminal": "Use Control; terminal.py is a command channel.",
}

_BOUNDARY_OWNERS = {
    "computeruse.loop": "Use the Surface, Decider, Escalator, and Control protocols.",
    "computeruse.control": "Keep ownership in Control and input behind Surface.",
    "computeruse.manual": "Keep evidence in manual.py and ownership in Control.",
    "computeruse.capability": "Keep saved data in Capability and effects in Surface.",
    "computeruse.recorder": "Keep recording over Action and Capability types.",
    "computeruse.replay": "Use Surface for input and Control for ownership.",
    "computeruse.replay_control": "Keep requests here and ownership in Control.",
    "computeruse.retarget": "Resolve saved targets through Surface and retarget.py.",
}


def imported_modules(source: str, *, module: str) -> frozenset[str]:
    """Resolve module imports from Python source.

    Parameters
    ----------
    source
        Python source to parse.
    module
        Fully qualified name of the source module.

    Returns
    -------
    frozenset[str]
        Absolute module names imported by the source.
    """
    package = module.rpartition(".")[0]
    top_level_package = module.partition(".")[0]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = node.module or ""
        if node.level:
            relative = f"{'.' * node.level}{base}"
            try:
                base = importlib.util.resolve_name(relative, package)
            except ImportError as error:
                raise AssertionError(
                    f"{module} contains an import outside its package: {relative}"
                ) from error
        if base == top_level_package:
            imported.update(
                base if alias.name == "*" else f"{base}.{alias.name}"
                for alias in node.names
            )
        elif base:
            imported.add(base)
    return frozenset(imported)


def _owner_for(dependency: str, module: str) -> str:
    for adapter, owner in _ADAPTER_OWNERS.items():
        if dependency == adapter or dependency.startswith(f"{adapter}."):
            return owner
    return _BOUNDARY_OWNERS[module]


def assert_source_import_contract(source: str, *, module: str) -> None:
    """Assert that source imports only permitted dependency edges.

    Parameters
    ----------
    source
        Python source for the protected module.
    module
        Fully qualified name used to select its import contract.
    """
    contract = _CONTRACTS[module]
    violations: list[str] = []
    for dependency in sorted(imported_modules(source, module=module)):
        root = dependency.partition(".")[0]
        if root == "computeruse":
            if dependency not in contract.first_party:
                violations.append(
                    f"{module} may not import {dependency}. "
                    f"{_owner_for(dependency, module)}"
                )
        elif (
            root not in sys.stdlib_module_names and root not in contract.external_roots
        ):
            violations.append(
                f"{module} may not import external dependency {dependency}. "
                f"{_BOUNDARY_OWNERS[module]}"
            )
    assert not violations, "\n".join(violations)


def assert_module_import_contract(module: ModuleType) -> None:
    """Assert that an imported module follows its dependency contract.

    Parameters
    ----------
    module
        Imported module whose source and qualified name select the contract.
    """
    path = module.__file__
    assert path is not None, f"{module.__name__} has no source file"
    assert_source_import_contract(
        Path(path).read_text(encoding="utf-8"), module=module.__name__
    )
