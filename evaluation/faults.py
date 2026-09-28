"""Inject one runtime fault into an evaluation replay, never a target website.

Back-office applications can respond slowly, fail requests, lose replies after
saving, and end idle sessions. The target applications cannot produce these
conditions on demand. The evaluation therefore intercepts the first matching
request between the browser and the application, then checks how the replay
classifies the result.

The product does not import this module. The evaluation installs each fault
after the product screens requests. Unmatched requests still pass through
that screening, and every matched request is one the profile permits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from collections.abc import Iterator
from enum import StrEnum
from urllib.parse import urlsplit

from playwright.sync_api import BrowserContext, Error, Route
from playwright.sync_api import Request as DriverRequest

from computeruse.browser import BrowserSurface
from computeruse.replay import Status


class Fault(StrEnum):
    """One kind of runtime trouble, applied once."""

    SLOW = "slow"
    """The response arrives after a delay the replay can wait out."""

    NEVER_LOADS = "never_loads"
    """The request times out, and no response arrives.

    The request fails at once instead of staying open. An open request prevents
    the browser session from closing, while an immediate timeout still shows
    the replay a page that did not load.
    """

    SERVER_ERROR = "server_error"
    """The application answers with an error page instead."""

    LOST_REPLY = "lost_reply"
    """The application saves the request, and its reply never arrives."""

    SESSION_EXPIRED = "session_expired"
    """The session cookie is gone before the request is sent."""


EXPECTED = {
    Fault.SLOW: frozenset({Status.SUCCEEDED}),
    Fault.NEVER_LOADS: frozenset(
        {Status.RECOVERY_EXHAUSTED, Status.NEEDS_HELP, Status.FAILED}
    ),
    Fault.SERVER_ERROR: frozenset({Status.FAILED, Status.NEEDS_HELP}),
    Fault.LOST_REPLY: frozenset({Status.SUCCEEDED, Status.NEEDS_HELP}),
    Fault.SESSION_EXPIRED: frozenset({Status.SUCCEEDED, Status.NEEDS_HELP}),
}
"""The allowed replay statuses for each fault.

A slow page is recovered from. A page that never loads and an error page end
the replay with a named step, or hand it to a person. A lost reply and an
expired session go to a person, who may settle them so the replay finishes.

The harness also accepts the fixture's expected answer after bounded recovery
or a person's intervention. It never accepts the person's answer alone. The
fixture, operator, institution, and commit count decide the result. A lost
reply must leave exactly one record, never two. Rule 13 fixed this criterion
before the evaluation runs.
"""


@dataclasses.dataclass(frozen=True)
class Injection:
    """The request method and path suffix where one fault applies."""

    fault: Fault
    method: str
    path_ends_with: str
    delay_s: float = 3.0

    def matches(self, request: DriverRequest) -> bool:
        """Report whether ``request`` is the one this fault is for."""
        return request.method == self.method and urlsplit(request.url).path.endswith(
            self.path_ends_with
        )


@dataclasses.dataclass
class Record:
    """Whether the fault fired, for the summary."""

    fired: bool = False


@contextlib.contextmanager
def inject(surface: BrowserSurface, injection: Injection) -> Iterator[Record]:
    """Apply ``injection`` once to the session behind ``surface``.

    Every request this does not match falls back to the product's own
    screening. The fault fires once, on the first match, and then steps aside.
    """
    context: BrowserContext = surface._context
    record = Record()
    lock = threading.Lock()

    def handle(route: Route, request: DriverRequest) -> None:
        with lock:
            first = not record.fired and injection.matches(request)
            if first:
                record.fired = True
        if not first:
            route.fallback()
            return
        _apply(injection, route, context)

    context.route("**/*", handle)
    try:
        yield record
    finally:
        with contextlib.suppress(Error):
            context.unroute("**/*", handle)


def _apply(injection: Injection, route: Route, context: BrowserContext) -> None:
    fault = injection.fault
    if fault is Fault.SLOW:
        # A short wait on the driver's thread, as a slow network would cause.
        time.sleep(injection.delay_s)
        route.fallback()
    elif fault is Fault.NEVER_LOADS:
        route.abort("timedout")
    elif fault is Fault.SERVER_ERROR:
        route.fulfill(
            status=500,
            content_type="text/html",
            body="<h1>Internal server error</h1><p>The request failed.</p>",
        )
    elif fault is Fault.LOST_REPLY:
        # The application receives and saves the request. The reply is lost.
        route.fetch()
        route.abort("connectionreset")
    elif fault is Fault.SESSION_EXPIRED:
        context.clear_cookies()
        route.fallback()
