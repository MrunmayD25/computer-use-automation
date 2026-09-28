"""Checks for the shared import contract resolver."""

import pytest
from import_contract import assert_source_import_contract, imported_modules


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("import computeruse.browser as driver", {"computeruse.browser"}),
        ("from computeruse import browser as driver", {"computeruse.browser"}),
        ("from .browser import open_session", {"computeruse.browser"}),
        ("from . import browser", {"computeruse.browser"}),
    ],
)
def test_imports_resolve_absolute_relative_and_package_aliases(statement, expected):
    assert imported_modules(statement, module="computeruse.loop") == expected


@pytest.mark.parametrize(
    ("module", "statement", "owner"),
    [
        (
            "computeruse.loop",
            "from computeruse.browser import open_session",
            "Surface",
        ),
        (
            "computeruse.control",
            "from computeruse import browser as driver",
            "Surface",
        ),
        (
            "computeruse.replay",
            "from .browser import open_session",
            "Surface",
        ),
        (
            "computeruse.loop",
            "from computeruse.model import ResponsesDecider",
            "Decider",
        ),
        (
            "computeruse.control",
            "from computeruse.native import NativeWindow",
            "Control",
        ),
        (
            "computeruse.replay",
            "from computeruse.terminal import TerminalChannel",
            "Control",
        ),
    ],
)
def test_adapter_shortcuts_are_rejected_with_the_owner_to_use(module, statement, owner):
    with pytest.raises(AssertionError) as raised:
        assert_source_import_contract(statement, module=module)

    assert "may not import" in str(raised.value)
    assert owner in str(raised.value)


def test_external_dependencies_remain_forbidden():
    with pytest.raises(AssertionError) as raised:
        assert_source_import_contract("import playwright", module="computeruse.loop")

    assert "external dependency playwright" in str(raised.value)


def test_allowed_module_edges_do_not_inventory_imported_identifiers():
    assert_source_import_contract(
        "from computeruse.actions import RenamedAction", module="computeruse.loop"
    )
