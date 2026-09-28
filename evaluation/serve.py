"""Serve the static evaluation pages locally.

The pages contain invented data. The server has no application logic. It gives
a real browser an origin to visit and lets tests request an available port.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(__file__).resolve().parent / "site"

DEMO_PORT = 8787
"""The port ``evaluation/profile.yaml`` declares, for a run started by hand."""

ROUTES = {
    "/members": "members.html",
    "/members/12345": "member.html",
    "/members/12345/savings": "savings.html",
    "/members/12345/savings/wire": "wire.html",
    "/admin": "admin.html",
    "/queue": "queue.html",
    "/queue/bare": "queue-bare.html",
    "/queue/shared": "shared.html",
    "/payments": "payments.html",
    "/static/core.css": "core.css",
    "/static/pad.js": "pad.js",
    "/static/extras.js": "extras.js",
    "/static/queue.js": "queue.js",
    "/static/shared.js": "shared.js",
    "/static/payments.js": "payments.js",
}
"""Every path this server answers. Anything else is a 404, including a guess."""

TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        """Answer one of the declared routes, or 404."""
        path = self.path.split("?")[0].rstrip("/") or "/members"
        name = ROUTES.get(path)
        if name is None:
            self.send_error(404, "no such page")
            return
        body = (SITE / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES[Path(name).suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress request logs because the journal records run evidence."""


@contextlib.contextmanager
def serve(port: int = 0) -> Iterator[str]:
    """Run the evaluation site and yield its base URL.

    Parameters
    ----------
    port
        A fixed port, or 0 to let the operating system pick a free one.
    """
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(_Handler))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    with serve(DEMO_PORT) as address:
        print(f"Serving the evaluation site at {address}. Press Control-C to stop.")
        with contextlib.suppress(KeyboardInterrupt):
            threading.Event().wait()
