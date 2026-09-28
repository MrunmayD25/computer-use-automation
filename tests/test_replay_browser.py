"""Replay against a real Chromium, at the adapter boundary only.

These cases test replay at the ``BrowserSurface`` boundary. One covers an
approval while the page changes. Another covers a native-dialog answer, which
the adapter refuses unless the expectation names that dialog. The tests serve
local pages on a port selected by the operating system. The capabilities are
synthetic fixtures. These tests do not connect replay to discovery, the CLI,
or an operator console.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from replay_fakes import People, go, limits

from computeruse.actions import (
    Action,
    AxLocator,
    Expectation,
    ObservationMode,
    ObservationRequest,
    PageState,
)
from computeruse.browser import BrowserLimits, BrowserSurface, open_session
from computeruse.capability import (
    SCHEMA_VERSION,
    ActionNode,
    Application,
    Approval,
    Capability,
    DialogOpen,
    LocatorForm,
    Present,
    Provenance,
    ProvenanceKind,
    ResultKind,
    ResultNode,
    Review,
    StructuralTarget,
    SurfaceKind,
    constant,
)
from computeruse.escalation import Ask, HandoffOutcome, InterventionRequest
from computeruse.profile import ActionKind, load_profile
from computeruse.replay import MemoryReplayLog, Reason, Status, replay

WORK = """<!doctype html>
<html><head><title>Entries</title></head><body><main>
<h1>Entries</h1>
<h2 id="state">Account Ready</h2>
<button id="freeze" type="button">Freeze</button>
<button id="post" type="button">Post entry</button>
<h2 id="count">Posts: 0</h2>
<div id="result"></div>
<script>
let posts = 0;
document.getElementById("freeze").onclick = () => {
  document.getElementById("state").textContent = "Account Frozen";
};
document.getElementById("post").onclick = () => {
  posts += 1;
  document.getElementById("count").textContent = "Posts: " + posts;
  document.getElementById("result").innerHTML = "<h2>Entry posted</h2>";
};
</script>
</main></body></html>
"""

DIALOG = """<!doctype html>
<html><head><title>Posting</title></head><body><main>
<h1>Posting</h1>
<button id="post" type="button">Post</button>
<div id="result"></div>
<script>
function show(text) {
  document.getElementById("result").innerHTML = "<h2>" + text + "</h2>";
}
document.getElementById("post").onclick = () => {
  show(confirm("Post this entry?") ? "Entry posted" : "Entry kept");
};
</script>
</main></body></html>
"""

PAGES = {"/work": WORK, "/dialog": DIALOG}

PROFILE = """\
version: 4
profile_id: local/replay-browser/sandbox
environment: sandbox
base_url: {base}
allow_routes: [/work, /dialog]
deny_routes: []
allow_new_windows: false
allow_downloads: false
actions:
  observe: {{any: safe}}
  read: {{any: safe}}
  click: {{any: safe}}
  navigate: {{any: safe}}
  accept_dialog: {{any: safe}}
  dismiss_dialog: {{any: safe}}
perception:
  allowed_modes: [structured]
  max_alternate_observations_per_step: 1
records:
  actions: []
  routes: []
budgets:
  max_steps: 40
  max_wall_clock_s: 120
  max_retries_per_step: 3
  max_navigations: 4
secrets: {{}}
escalation:
  handoff_timeout_s: 60
  on_timeout: abort
"""


@contextlib.contextmanager
def serve(pages: Mapping[str, str]) -> Iterator[str]:
    """Serve ``pages`` by exact path on a free port, and yield the base URL."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            """Answer a written page, or 404."""
            body = pages.get(self.path.split("?")[0])
            if body is None:
                self.send_error(404, "no such page")
                return
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Stay quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    with serve(PAGES) as url:
        yield url


@pytest.fixture
def profile(base, tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE.format(base=base))
    return load_profile(path)


@contextlib.contextmanager
def browser(profile, base: str, path: str) -> Iterator[BrowserSurface]:
    with open_session(
        profile, base + path, limits=BrowserLimits(operation_ms=2_000)
    ) as surface:
        yield surface


def heading(target_id: str, route: str, name: str) -> StructuralTarget:
    return StructuralTarget(
        target_id,
        route,
        LocatorForm.ACCESSIBILITY,
        "heading",
        constant(name),
        "",
        None,
        None,
        (),
        None,
    )


def button(target_id: str, route: str, name: str) -> StructuralTarget:
    return dataclasses.replace(heading(target_id, route, name), role="button")


def capability(base: str, route: str, nodes, targets, result: str) -> Capability:
    return Capability(
        schema_version=SCHEMA_VERSION,
        capability_id="post_entry",
        version=1,
        application=Application(
            "local/replay-browser/sandbox",
            SurfaceKind.BROWSER,
            base,
            route,
            (Present("post"),),
        ),
        provenance=Provenance(
            ProvenanceKind.SYNTHETIC, "", "hand_written_fixture", Review.REVIEWED
        ),
        inputs=(),
        outputs=(),
        variables=(),
        secrets=(),
        outcomes=(),
        templates=(),
        targets=targets,
        entry=nodes[0].node_id,
        nodes=(
            *nodes,
            ResultNode("done", ResultKind.SUCCESS, "", (Present(result),)),
        ),
        restrictions=(),
        limits=limits(max_settle_observations=5, max_help_requests=4),
    )


def work_capability(base: str) -> Capability:
    """Post an entry, approved each run, while the account shows Ready."""
    post = ActionNode(
        "post",
        ActionKind.CLICK,
        "/work",
        "post",
        None,
        None,
        "post_entry",
        None,
        None,
        None,
        Approval.EACH_RUN,
        True,
        (Present("ready"),),
        (Present("posted"),),
        (go("done"),),
    )
    return capability(
        base,
        "/work",
        (post,),
        (
            button("post", "/work", "Post entry"),
            heading("ready", "/work", "Account Ready"),
            heading("posted", "/work", "Entry posted"),
        ),
        "posted",
    )


def dialog_capability(base: str, route: str, kind: ActionKind, result: str):
    """Click Post, which opens a confirmation, then answer it, approved each run."""
    post = ActionNode(
        "post",
        ActionKind.CLICK,
        route,
        "post",
        None,
        None,
        "open_posting",
        None,
        None,
        None,
        Approval.NONE,
        False,
        (),
        (),
        (go("answer", DialogOpen("confirm", constant("Post this entry?"))),),
    )
    answer = ActionNode(
        "answer",
        kind,
        route,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Approval.EACH_RUN,
        True,
        (DialogOpen("confirm", constant("Post this entry?")),),
        (Present("result"),),
        (go("done"),),
    )
    return capability(
        base,
        route,
        (post, answer),
        (button("post", route, "Post"), heading("result", route, result)),
        "result",
    )


def run(cap, surface, people, profile):
    return replay(
        cap,
        {},
        profile=profile,
        surface=surface,
        control=people.control(time.monotonic),
        log=MemoryReplayLog(),
        clock=time.monotonic,
        sleep=time.sleep,
    )


def headings(surface: BrowserSurface) -> set[str]:
    seen = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
    return {node.name for node in seen.nodes if node.role == "heading"}


def click(surface: BrowserSurface, name: str) -> None:
    here = PageState(surface.location())
    surface.act(
        Action(ActionKind.CLICK, AxLocator("button", name)), expect=Expectation(here)
    )


def test_a_precondition_changed_during_approval_stops_the_click(profile, base) -> None:
    with browser(profile, base, "/work") as surface:

        def freeze(request: InterventionRequest) -> HandoffOutcome:
            assert request.ask is Ask.APPROVAL
            click(surface, "Freeze")
            return HandoffOutcome.APPROVED

        people = People(
            [
                freeze,
                HandoffOutcome.RESUMED,
                HandoffOutcome.RESUMED,
                HandoffOutcome.RESUMED,
            ]
        )

        result = run(work_capability(base), surface, people, profile)

        assert result.status is Status.NEEDS_HELP
        assert "Posts: 0" in headings(surface)
        assert [r.reason for r in result.history][1:] == [Reason.PRECONDITION_UNMET] * 3


def test_an_entry_posted_during_approval_is_not_posted_again(profile, base) -> None:
    with browser(profile, base, "/work") as surface:

        def post_it(_request: InterventionRequest) -> HandoffOutcome:
            click(surface, "Post entry")
            return HandoffOutcome.APPROVED

        result = run(work_capability(base), surface, People([post_it]), profile)

        assert result.status is Status.SUCCEEDED
        assert "Posts: 1" in headings(surface)


def test_an_approved_click_runs_once_in_a_real_browser(profile, base) -> None:
    with browser(profile, base, "/work") as surface:
        people = People([HandoffOutcome.APPROVED])

        result = run(work_capability(base), surface, people, profile)

        assert result.status is Status.SUCCEEDED
        assert "Posts: 1" in headings(surface)
        assert len(people.requests) == 1


@pytest.mark.parametrize(
    ("kind", "result"),
    [
        (ActionKind.ACCEPT_DIALOG, "Entry posted"),
        (ActionKind.DISMISS_DIALOG, "Entry kept"),
    ],
)
def test_a_native_dialog_is_answered_by_its_id(profile, base, kind, result) -> None:
    """Regression: the answer named no dialog, and Chromium's adapter refused it."""
    with browser(profile, base, "/dialog") as surface:
        people = People([HandoffOutcome.APPROVED])

        replayed = run(
            dialog_capability(base, "/dialog", kind, result), surface, people, profile
        )

        assert replayed.status is Status.SUCCEEDED, replayed
        assert result in headings(surface)
        assert [r.ask for r in people.requests] == [Ask.APPROVAL]


def test_a_dialog_replaced_during_approval_is_approved_again(profile, base) -> None:
    """A person answers the first dialog and opens another while approving."""
    with browser(profile, base, "/dialog") as surface:
        answered: list[str] = []

        def someone_answers_and_posts_again(
            _request: InterventionRequest,
        ) -> HandoffOutcome:
            observed = surface.observe(ObservationRequest(ObservationMode.STRUCTURED))
            assert observed.dialog is not None
            answered.append(observed.dialog.dialog_id)
            surface.act(
                Action(ActionKind.DISMISS_DIALOG),
                expect=Expectation(
                    PageState(surface.location()), dialog=observed.dialog.dialog_id
                ),
            )
            click(surface, "Post")
            return HandoffOutcome.APPROVED

        people = People([someone_answers_and_posts_again, HandoffOutcome.APPROVED])
        cap = dialog_capability(
            base, "/dialog", ActionKind.ACCEPT_DIALOG, "Entry posted"
        )

        result = run(cap, surface, people, profile)

        assert result.status is Status.SUCCEEDED, result
        assert [r.reason for r in result.history] == [
            Reason.APPROVAL_REQUIRED,
            Reason.APPROVAL_INVALIDATED,
        ]
        assert answered == ["dialog-1"]
        assert "Entry posted" in headings(surface)


class Watched:
    """Passes every call to a real surface, and notes where each look happened."""

    def __init__(self, surface: BrowserSurface) -> None:
        self.surface = surface
        self.looked_at: list[str] = []

    def location(self) -> str:
        return self.surface.location()

    def pages(self):
        return self.surface.pages()

    def capabilities(self):
        return self.surface.capabilities()

    def observe(self, request: ObservationRequest):
        self.looked_at.append(self.surface.location())
        return self.surface.observe(request)

    def act(self, action: Action, *, expect: Expectation | None = None):
        return self.surface.act(action, expect=expect)


def test_a_risky_look_approved_for_one_page_is_not_taken_on_another(
    base, tmp_path
) -> None:
    path = tmp_path / "risky.yaml"
    path.write_text(
        PROFILE.format(base=base).replace(
            "observe: {any: safe}", "observe: {any: risky}"
        )
    )
    risky = load_profile(path)
    with browser(risky, base, "/work") as surface:
        watched = Watched(surface)
        looked_when_asked: list[int] = []

        def navigate_away(_request: InterventionRequest) -> HandoffOutcome:
            looked_when_asked.append(len(watched.looked_at))
            if len(looked_when_asked) == 1:
                surface.act(
                    Action(ActionKind.NAVIGATE, destination=base + "/dialog"),
                    expect=Expectation(PageState(surface.location())),
                )
            return HandoffOutcome.APPROVED

        people = People([navigate_away, navigate_away])

        result = run(work_capability(base), watched, people, risky)

        assert [r.reason for r in result.history] == [
            Reason.APPROVAL_REQUIRED,
            Reason.APPROVAL_INVALIDATED,
        ]
        assert looked_when_asked == [0, 0]
        assert watched.looked_at == [base + "/dialog"]
        assert result.reason is Reason.INCOMPATIBLE_APPLICATION
