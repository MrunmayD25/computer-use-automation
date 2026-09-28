"""The evaluation's injected faults do what they say, in a real Chromium."""

import time

from computeruse.actions import (
    Action,
    AxLocator,
    ObservationMode,
    ObservationRequest,
)
from computeruse.profile import ActionKind
from evaluation.faults import Fault, Injection, inject

LOOK = ObservationRequest(ObservationMode.STRUCTURED)
PAGES = {
    "/": '<h1>Start</h1><a href="/next">Next</a>',
    "/next": "<h1>Next page</h1>",
}
NEXT = Action(ActionKind.CLICK, AxLocator("link", "Next"))


def _headings(surface) -> set[str]:
    return {node.name for node in surface.observe(LOOK).nodes if node.role == "heading"}


def test_a_server_error_replaces_the_page(pages):
    with pages(PAGES) as (surface, _):
        with inject(surface, Injection(Fault.SERVER_ERROR, "GET", "/next")) as record:
            surface.act(NEXT)
        assert record.fired
        assert "Internal server error" in _headings(surface)


def test_a_slow_page_arrives_after_its_delay(pages):
    with pages(PAGES) as (surface, _):
        started = time.monotonic()
        with inject(surface, Injection(Fault.SLOW, "GET", "/next", delay_s=1.0)):
            surface.act(NEXT)
        assert time.monotonic() - started >= 1.0
        assert "Next page" in _headings(surface)


def test_a_page_that_never_loads_leaves_the_next_one_unseen(pages):
    with (
        pages(PAGES) as (surface, _),
        inject(surface, Injection(Fault.NEVER_LOADS, "GET", "/next")) as record,
    ):
        surface.act(NEXT)
        assert record.fired
        assert "Next page" not in {node.name for node in surface.observe(LOOK).nodes}


def test_an_expired_session_loses_its_cookie_before_the_request(pages):
    with pages(PAGES) as (surface, _):
        surface._page.evaluate("document.cookie = 'session=abc; path=/'")
        with inject(surface, Injection(Fault.SESSION_EXPIRED, "GET", "/next")):
            surface.act(NEXT)
        assert surface._context.cookies() == []


def test_a_request_the_fault_does_not_match_passes_through(pages):
    with pages(PAGES) as (surface, _):
        with inject(surface, Injection(Fault.SERVER_ERROR, "POST", "/next")) as record:
            surface.act(NEXT)
        assert not record.fired
        assert "Next page" in _headings(surface)
