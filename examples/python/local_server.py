"""A tiny HTTP server for the examples, so that they run without a network.

It is not part of the library: the standard library's ``http.server`` on a background thread,
answering every GET with what ``handler(path, headers)`` returns.
"""

import contextlib
import threading
from collections.abc import Callable, Iterator
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# handler(path, headers) -> (status, body); ``path`` includes the query string.
Handler = Callable[[str, Message], tuple[int, bytes]]


@contextlib.contextmanager
def serve(handler: Handler) -> Iterator[str]:
    """Serve on 127.0.0.1, on a free port, until the ``with`` block ends; yields the base URL."""

    class Answer(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep connections alive between requests

        def do_GET(self) -> None:
            status, body = handler(self.path, self.headers)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass  # no line per request on the console

    server = ThreadingHTTPServer(("127.0.0.1", 0), Answer)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
