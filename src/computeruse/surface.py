"""The protocol for observing an authorized session and sending live input.

The browser enforces HTTP routes and provides structured and screenshot
observations through this protocol. The loop and policy rules are shared.
Replay needs a separate target contract because live coordinates do not
identify a control across runs.

The protocol has five methods. Policy requirements determine the split
between the first three. ``location`` reports session metadata and
reads no page content, so the gate can settle permission before anything is
collected. ``observe`` collects content and is therefore only ever called
after that gate. ``act`` returns no observation at all, because reading the
page after an action is another observation and has to pass the gate too.
``capabilities`` lists the action types and target forms the adapter supports,
so the decider does not receive operations the adapter cannot perform.
``pages`` lists the windows the session manages, using only metadata, so the
loop can hand a new one to a person before sending more input.

An observation tool does not act. It does not scroll a control into view, and
it does not photograph one element at a time in a way that would. Revealing
more of a page is a ``scroll`` action, declared by the operator like any
other, so a run cannot move the page while claiming to be looking at it.
"""

from typing import Protocol

from computeruse.actions import (
    Action,
    ActionResult,
    AxNode,
    Capabilities,
    Expectation,
    Observation,
    ObservationRequest,
    PageInfo,
    ScreenTarget,
)
from computeruse.operations import BoundOperation


class SurfaceError(RuntimeError):
    """A session-wide failure that prevents the surface from continuing.

    An adapter reports a missing control, an ambiguous one, or a refused side
    effect through ``Outcome``. This is for the case where the session itself
    is gone: the browser crashed, the driver died, or the page context closed.
    The loop ends the run because no live session remains for another decision.
    """


class Surface(Protocol):
    """A live session that supports observation and input.

    Implementations own their own entry point, viewport, window policy, and
    download policy; the loop never opens a session, so it cannot open one
    outside the profile.
    """

    def location(self) -> str:
        """Return the current scope location from metadata, reading no content."""
        ...

    def capabilities(self) -> Capabilities:
        """Return the action types this adapter performs, with their target forms.

        The profile defines what is permitted. This method reports what the
        adapter can do. A decision is offered only what both allow. The loop
        refuses a proposal outside that set before the gate, with a notice to
        correct it.
        """
        ...

    def pages(self) -> tuple[PageInfo, ...]:
        """Return every window the session manages, reading no page content.

        A surface with one window and no notion of others returns an empty
        tuple, and the loop then compares nothing.
        """
        ...

    def observe(self, request: ObservationRequest) -> Observation:
        """Collect one observation with the requested tool, without acting.

        An observation tool does not click, type, scroll, or submit. Scrolling
        is a separate proposed action so that it passes the gate like any
        other change to the page.
        """
        ...

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        """Resolve ``action``'s target against the live surface and perform it.

        A locator is resolved again immediately before acting. Screen input
        checks its capture, viewport, focus or target pixels, and frame scope.
        ``expect`` describes what the decision was made against: the page the
        run was on and the context path the control sat under. A surface that
        has moved on, or a control that now belongs to a different record, reports
        ``Outcome.STALE`` instead of acting on something else.
        """
        ...


def control_at(surface: object, target: ScreenTarget) -> str:
    """Return the id of the observed control under a screenshot point, or "".

    A screenshot action on ordinary page content is authorized as the
    operation of the element under its point, so a learned restriction on it
    binds to that control rather than to the whole screen (rule 6). A
    surface that cannot tell, a point on a canvas or on nothing, and an
    element no observation described all give "", and the action keeps the
    conservative route-wide identity. This is an optional query: a surface
    offers it by defining ``control_at``.
    """
    find = getattr(surface, "control_at", None)
    if not callable(find):
        return ""
    found = find(target)
    return found if isinstance(found, str) else ""


def operation_binding(
    surface: object,
    action: Action,
    observation: Observation | None,
    node: AxNode | None = None,
) -> BoundOperation | None:
    """Ask the adapter to attest a binding from already observed evidence."""
    method = getattr(surface, "operation_binding", None)
    return method(action, observation, node) if callable(method) else None
