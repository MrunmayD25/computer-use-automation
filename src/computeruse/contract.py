"""Explicit artifact types and storage declarations supplied by the operator."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import cast

from computeruse.capability import (
    Field,
    Issue,
    _decode,
    _field_ok,
    _unique_keys,
    identifier,
)


@dataclasses.dataclass(frozen=True)
class Binding:
    """An input that fills a form field, as the operator declares it.

    A form can open with a field already set, such as the operator selected on
    a sign-on page. Discovery submits it unchanged, so the recorder may add a
    step that sets it from the input. It does so only for a field named by a
    binding. Equal sample values alone never tie a field to an input (rule 4).
    ``field`` is the field's accessible name.
    """

    input: str
    field: str


@dataclasses.dataclass(frozen=True)
class Contract:
    """Names and types have meanings independent of discovery's sample values.

    Version 2 adds ``bindings``, the inputs that fill prefilled form fields.
    """

    version: int
    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]
    bindings: tuple[Binding, ...] = ()


def load_contract(path: Path) -> Contract:
    """Read declarations without returning values in error messages."""
    if path.stat().st_size > 1_000_000:
        raise ValueError("export contract is too large")
    issues: list[Issue] = []
    document = json.loads(path.read_text(), object_pairs_hook=_unique_keys)
    result = cast("Contract", _decode(Contract, document, "contract", issues))
    if issues or result.version != 2:
        raise ValueError("export contract is invalid")
    names = {field.name for field in result.inputs}
    if not all(
        binding.input in names and binding.field.strip() for binding in result.bindings
    ):
        raise ValueError("export bindings are invalid")
    for group in (result.inputs, result.outputs):
        if len({field.name for field in group}) != len(group) or not all(
            identifier(field.name) and _field_ok(field) for field in group
        ):
            raise ValueError("export fields are invalid")
    return result
