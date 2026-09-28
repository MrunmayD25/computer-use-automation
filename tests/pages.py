"""A disposable local server for pages a single test writes.

The evaluation site stays unchanged. A case writes any markup the site does
not have, such as two frames with one name or a popup with its own dialog.
This module serves that markup on a port selected by the operating system.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextlib.contextmanager
def serve_pages(pages: Mapping[str, str]) -> Iterator[str]:
    """Serve ``pages`` by exact path and yield the base URL.

    Parameters
    ----------
    pages
        HTML bodies keyed by path. Any other path is a 404.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            """Answer one of the written pages, or 404."""
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
