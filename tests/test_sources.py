"""Source: items fetched by Rust from memory, files or HTTP, with metadata.

The HTTP tests run a local http.server in a thread. Encoded images and expected
values are the golden fixtures of the Rust tests (tests/fixtures/image).
"""

import ssl
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from dataloader_rs import ImageCollator, PyDataset, Source
from dataloader_rs import PyDataloader as DataLoader

FIXTURES = Path(__file__).parent / "fixtures" / "image"
# localhost / 127.0.0.1, signed by a test CA that no trust store holds.
TLS = Path(__file__).parent / "fixtures" / "tls"
RGB_PNG = (FIXTURES / "rgb.png").read_bytes()
# The 64x48 scene as HWC uint8, as Pillow decodes it.
RGB_U8 = np.frombuffer((FIXTURES / "rgb.u8").read_bytes(), dtype=np.uint8).reshape(48, 64, 3)

# Retries without waiting.
QUICK = {"backoff": 0.001, "max_backoff": 0.005}


def hwc_u8() -> ImageCollator:
    return ImageCollator([], dtype="uint8", layout="nhwc", input_size=(64, 48))


def run(source: Source, **kwargs) -> tuple[np.ndarray, list]:
    """Every batch of an epoch, concatenated, and the metadata in order."""
    kwargs.setdefault("batch_size", 4)
    values, meta = [], []
    for batch in DataLoader(source, collate_fn=hwc_u8(), **kwargs):
        values.append(np.array(batch))
        meta.extend(batch.meta)
    return np.concatenate(values), meta


class Server(ThreadingHTTPServer):
    # The default listen backlog of 5 drops concurrent connections.
    request_queue_size = 128
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # Clients that time out close their connection: not an error here.
        pass


Handler = Callable[[str, int], tuple[int, dict[str, str], bytes]]

# The headers of every request the test servers received.
HEADERS: list[dict[str, str]] = []


@contextmanager
def http_server(handler: Handler) -> Iterator[tuple[str, list[str]]]:
    """A server answering `handler(path, n)` (`n` counts requests from 1); yields the base URL
    and the list of requested paths."""
    seen: list[str] = []
    lock = threading.Lock()

    class Handle(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            with lock:
                seen.append(self.path)
                HEADERS.append({k.lower(): v for k, v in self.headers.items()})
                number = len(seen)
            status, headers, body = handler(self.path, number)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = Server(("127.0.0.1", 0), Handle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def png(path: str, number: int) -> tuple[int, dict[str, str], bytes]:
    return 200, {}, RGB_PNG


# ── Memory and files ─────────────────────────────────────────────────────────


def test_from_bytes_values_and_meta():
    source = Source.from_bytes([RGB_PNG] * 6, meta=[{"id": i} for i in range(6)])
    assert len(source) == 6
    assert repr(source) == "Source(bytes, len=6)"
    values, meta = run(source, num_workers=2)
    assert values.shape == (6, 48, 64, 3)
    assert (values == RGB_U8).all()
    assert meta == [{"id": i} for i in range(6)]


def test_getitem_and_plain_dataset():
    source = Source.from_bytes([b"a", bytearray(b"bc")], meta=["x", "y"])
    assert isinstance(source, PyDataset)
    assert source[1] == (b"bc", "y")
    with pytest.raises(IndexError):
        source[2]
    # Without ImageCollator it is a plain PyDataset of (bytes, meta) items.
    batches = list(DataLoader(source, batch_size=2))
    assert len(batches) == 1
    assert [tuple(item) for item in batches[0]] == [(b"a", "x"), (b"bc", "y")]
    assert Source.from_bytes([b"a"])[0] == (b"a", None)


def test_from_files(tmp_path):
    paths = []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        path.write_bytes(RGB_PNG)
        paths.append(path if i % 2 else str(path))
    source = Source.from_files(paths, meta=list(range(5)))
    assert repr(source) == "Source(files, len=5)"
    values, meta = run(source, batch_size=2, num_workers=2)
    assert (values == RGB_U8).all()
    assert meta == list(range(5))


def test_file_errors_fail_their_batch(tmp_path):
    good = tmp_path / "good.png"
    good.write_bytes(RGB_PNG)
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (len(RGB_PNG) + 1))
    missing = tmp_path / "missing.png"
    source = Source.from_files([good, missing, good, big, good], max_bytes=len(RGB_PNG))
    with pytest.raises(FileNotFoundError, match="item 1 .*missing.png"):
        source[1]
    with pytest.raises(OSError, match="item 3 .*above the limit"):
        source[3]
    for workers in (0, 2):
        results = []
        iterator = iter(DataLoader(source, batch_size=1, num_workers=workers, collate_fn=hwc_u8()))
        for _ in range(5):
            try:
                batch = next(iterator)
                results.append(len(batch))
                del batch
            except OSError as error:
                results.append(type(error))
        assert results == [1, FileNotFoundError, 1, OSError, 1]


def test_invalid_settings():
    with pytest.raises(ValueError, match="2 items but the metadata list has 1"):
        Source.from_bytes([b"a", b"b"], meta=[1])
    with pytest.raises(ValueError, match="URL 1"):
        Source.from_urls(["http://h/a", "not a url"])
    with pytest.raises(ValueError, match="header name"):
        Source.from_urls(["http://h/a"], headers={"Bad Name": "x"})
    with pytest.raises(ValueError) as info:
        Source.from_urls(["http://h/a"], headers={"X-Key": "s3cret\n"})
    assert "s3cret" not in str(info.value)
    with pytest.raises(ValueError, match="timeout"):
        Source.from_urls(["http://h/a"], timeout=-1)
    with pytest.raises(ValueError, match="max_bytes"):
        Source.from_files(["a"], max_bytes=0)
    with pytest.raises(ValueError, match='"b" is not in the template'):
        Source.from_template("http://h/{a}", [{"b": 1}])


# ── HTTP ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("workers", "in_parallel"), [(0, 1), (0, 4), (2, 1), (3, 2)])
def test_from_urls_through_the_loader(workers, in_parallel):
    with http_server(png) as (base, seen):
        urls = [f"{base}/img/{i}.png" for i in range(10)]
        source = Source.from_urls(urls, meta=[f"label-{i}" for i in range(10)])
        values, meta = run(source, num_workers=workers, items_in_parallel=in_parallel)
    assert values.shape == (10, 48, 64, 3)
    assert (values == RGB_U8).all()
    assert meta == [f"label-{i}" for i in range(10)]
    assert sorted(seen) == sorted(f"/img/{i}.png" for i in range(10))


def test_from_template_encodes_and_sends_headers():
    with http_server(lambda path, n: (200, {}, path.encode())) as (base, seen):
        source = Source.from_template(
            base + "/{slide}/{x}_{y}.png",
            [{"slide": "a b", "x": 0, "y": 1}, {"slide": "../etc", "x": 2, "y": "?q"}],
            headers={"Authorization": "Bearer s3cret"},
        )
        assert source[0][0] == b"/a%20b/0_1.png"
        assert source[1][0] == b"/..%2Fetc/2_%3Fq.png"
    assert seen == ["/a%20b/0_1.png", "/..%2Fetc/2_%3Fq.png"]
    assert HEADERS[-1].get("authorization") == "Bearer s3cret"


def test_retries_transient_statuses():
    def flaky(path, number):
        return (503, {}, b"") if number <= 2 else (200, {}, RGB_PNG)

    with http_server(flaky) as (base, seen):
        source = Source.from_urls([f"{base}/a.png"], **QUICK)
        assert source[0][0] == RGB_PNG
    assert len(seen) == 3


def test_gives_up_and_hides_the_query():
    with http_server(lambda path, n: (502, {}, b"")) as (base, seen):
        source = Source.from_urls([f"{base}/a.png?token=secret"], max_retries=2, **QUICK)
        with pytest.raises(OSError, match=r"HTTP status 502 after 3 attempt") as info:
            source[0]
    assert "secret" not in str(info.value)
    assert len(seen) == 3


def test_permanent_status_fails_its_batch_only():
    def handler(path, number):
        return (404, {}, b"") if path == "/3.png" else (200, {}, RGB_PNG)

    with http_server(handler) as (base, seen):
        source = Source.from_urls([f"{base}/{i}.png" for i in range(8)], **QUICK)
        outcomes = []
        iterator = iter(DataLoader(source, batch_size=2, num_workers=2, collate_fn=hwc_u8()))
        for _ in range(4):
            try:
                batch = next(iterator)
                outcomes.append(len(batch))
                del batch
            except OSError as error:
                outcomes.append(str(error))
    assert outcomes[0] == outcomes[2] == outcomes[3] == 2
    assert "item 3" in outcomes[1] and "404" in outcomes[1]
    assert seen.count("/3.png") == 1  # 404 is not retried


def test_timeout_raises_timeout_error():
    def slow(path, number):
        time.sleep(2)
        return 200, {}, RGB_PNG

    with http_server(slow) as (base, _):
        source = Source.from_urls([f"{base}/a.png"], timeout=0.2, max_retries=0)
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            list(DataLoader(source, batch_size=1, collate_fn=hwc_u8()))
        assert time.monotonic() - start < 1.5


def test_response_size_cap():
    with http_server(lambda path, n: (200, {}, b"\0" * 4096)) as (base, _):
        source = Source.from_urls([f"{base}/a"], max_bytes=1024)
        with pytest.raises(OSError, match="4096 bytes, above the limit of 1024"):
            source[0]


def test_requests_overlap_without_the_gil():
    # 8 requests of 0.2 s each with 8 in flight: about 0.2 s, not 1.6 s.
    def slow(path, number):
        time.sleep(0.2)
        return 200, {}, RGB_PNG

    with http_server(slow) as (base, _):
        source = Source.from_urls([f"{base}/{i}.png" for i in range(8)])
        start = time.monotonic()
        values, _ = run(source, batch_size=8, num_workers=0, items_in_parallel=8)
        elapsed = time.monotonic() - start
    assert values.shape[0] == 8
    assert elapsed < 1.0, elapsed


# ── HTTPS ────────────────────────────────────────────────────────────────────


def test_https_url_is_accepted():
    source = Source.from_urls(["https://example.com/a.png", "http://example.com/b.png"])
    assert len(source) == 2


def test_https_untrusted_certificate_fails_without_retry():
    """A self-signed / private-CA server fails its item with OSError at the first attempt.
    The positive case needs a trusted root: it is a Rust test (sources::http::tests::tls)."""
    connections = []

    class TlsServer(Server):
        def get_request(self):
            connections.append(1)
            return super().get_request()

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(TLS / "localhost.pem", TLS / "localhost.key")
    server = TlsServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"https://127.0.0.1:{server.server_address[1]}/a.png"
        source = Source.from_urls([url + "?token=secret"], max_retries=3, **QUICK)
        with pytest.raises(OSError, match="invalid peer certificate") as info:
            source[0]
        message = str(info.value)
        assert type(info.value) is OSError
        assert message.startswith(f"item 0 ({url}): request failed after 1 attempt(s)")
        assert "secret" not in message
        assert len(connections) == 1
    finally:
        server.shutdown()
        server.server_close()
