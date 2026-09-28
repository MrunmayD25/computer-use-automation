"""Collect bounded structured pages through the caller's observation gate.

Two identical traversals establish coverage. The observation remains
incomplete if the page changes, the tree is uncounted, any frame is refused or
fails, or the run reaches a budget boundary. This module does not grant
permission, scroll the application, or read a surface directly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from computeruse.actions import (
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Window,
)

MAX_NODES = 5_000
MAX_PAGES = 32


def gather(
    first: Observation,
    request: ObservationRequest,
    read: Callable[[ObservationRequest], Observation | None],
) -> Observation:
    """Complete one paged observation without treating missing controls as absent."""
    window = first.window
    if (
        request.mode is not ObservationMode.STRUCTURED
        or request.narrowed
        or window is None
        or not window.rest
        or not window.counted
        or window.total > MAX_NODES
    ):
        return first
    collected = _pass(first, request, read)
    if collected is None:
        return first
    again = read(request)
    checked = None if again is None else _pass(again, request, read)
    if checked is None or checked.nodes != collected.nodes:
        return dataclasses.replace(first, status=ObservationStatus.PARTIAL)
    if checked.page_state.differs_from(collected.page_state):
        return dataclasses.replace(first, status=ObservationStatus.PARTIAL)
    return checked


def _pass(
    first: Observation,
    request: ObservationRequest,
    read: Callable[[ObservationRequest], Observation | None],
) -> Observation | None:
    window = first.window
    if (
        window is None
        or not window.counted
        or not window.collected
        or window.total > MAX_NODES
    ):
        return None
    nodes = list(first.nodes)
    part = first
    for _ in range(MAX_PAGES):
        if len(nodes) >= window.total:
            if part.status is not ObservationStatus.COMPLETE:
                return None
            return dataclasses.replace(
                first,
                status=ObservationStatus.COMPLETE,
                nodes=tuple(nodes),
                notes=(),
                window=Window(0, len(nodes), window.total),
            )
        part = read(dataclasses.replace(request, start=len(nodes)))
        if part is None or not _compatible(first, part, len(nodes)):
            return None
        nodes.extend(part.nodes)
    return None


def _compatible(first: Observation, part: Observation | None, start: int) -> bool:
    return bool(
        part is not None
        and part.usable
        and part.nodes
        and part.dialog is None
        and not first.page_state.differs_from(part.page_state)
        and part.pages == first.pages
        and part.window is not None
        and first.window is not None
        and part.window.start == start
        and part.window.counted
        and part.window.collected
        and part.window.total == first.window.total
    )
