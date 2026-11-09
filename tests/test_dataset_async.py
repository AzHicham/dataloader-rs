"""Async datasets: ``PyAsyncDataset`` and the Rust datasets (``FileDataset``,
``HttpDataset``, ``ObjectStoreDataset``) in ``PyDataloader``, and how the
loader checks a dataset against its base class."""

import asyncio
import collections
import gc
import logging
import signal
import subprocess
import sys
import threading
import time
import warnings
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dataloader_rs import (
    FileDataset,
    HttpDataset,
    ObjectStoreDataset,
    PyAsyncDataset,
    PyDataset,
    RustDataset,
)
from dataloader_rs import PyDataloader as DataLoader

# ── PyAsyncDataset ────────────────────────────────────────────────────────────


class AsyncRange(PyAsyncDataset):
    """Item ``i`` is ``i`` after a short sleep that varies per item, so that
    items finish out of order; records the peak of coroutines running at once."""

    def __init__(self, n, delay=0.002, fail_at=None, jitter=True):
        super().__init__()
        self.n = n
        self.delay = delay
        self.jitter = jitter
        self.fail_at = fail_at
        self.running = 0
        self.peak = 0
        self.cancelled = 0
        self.lock = threading.Lock()

    def __len__(self):
        return self.n

    async def __getitem__(self, index):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(self.delay * (index % 3 if self.jitter else 1))
            if index == self.fail_at:
                raise LookupError(f"no item {index}")
            return index
        except asyncio.CancelledError:
            with self.lock:
                self.cancelled += 1
            raise
        finally:
            with self.lock:
                self.running -= 1


@pytest.mark.parametrize(("num_workers", "item_concurrency"), [(0, 1), (2, 4), (3, 1)])
def test_async_batches_in_order(num_workers, item_concurrency):
    loader = DataLoader(
        AsyncRange(30),
        batch_size=4,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
    )
    assert len(loader) == 8
    for _ in range(2):
        batches = list(loader)
        assert [i for b in batches for i in b] == list(range(30))
        assert batches[-1] == [28, 29]


def test_async_shuffle_and_collate_fn():
    loader = DataLoader(
        AsyncRange(20, delay=0),
        batch_size=5,
        shuffle=True,
        seed=3,
        num_workers=2,
        item_concurrency=5,
        collate_fn=sorted,
    )
    batches = list(loader)
    assert all(b == sorted(b) for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(20))


def test_in_flight_is_bounded_by_workers_times_item_concurrency():
    ds = AsyncRange(96, delay=0.01, jitter=False)
    loader = DataLoader(ds, batch_size=8, num_workers=3, item_concurrency=4)
    assert sum(len(b) for b in loader) == 96
    assert ds.peak == 12


def test_one_runtime_thread_drives_64_coroutines():
    ds = AsyncRange(256, delay=0.01, jitter=False)
    loader = DataLoader(ds, batch_size=32, num_workers=2, item_concurrency=32, runtime_threads=1)
    assert sum(len(b) for b in loader) == 256
    assert ds.peak == 64


@pytest.mark.parametrize("threads", [0, 1025])
def test_runtime_threads_out_of_range_is_refused(threads):
    with pytest.raises(ValueError, match="runtime_threads must be between 1 and 1024"):
        DataLoader(AsyncRange(4), runtime_threads=threads)


class SyncRange(PyDataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index


def test_runtime_threads_with_a_sync_dataset_is_refused():
    with pytest.raises(ValueError, match="runtime_threads only applies to async datasets"):
        DataLoader(SyncRange(), runtime_threads=2)
    assert list(DataLoader(SyncRange(), batch_size=4, runtime_threads=None)) == [[0, 1, 2, 3]]


class AsyncBatched(PyAsyncDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.batch_calls = 0

    def __len__(self):
        return self.n

    async def __getitem__(self, index):
        return index

    async def __getitems__(self, indices):
        self.batch_calls += 1
        await asyncio.sleep(0)
        return [i * 10 for i in indices]


def test_async_getitems_is_used_when_items_are_not_concurrent():
    ds = AsyncBatched(10)
    assert list(DataLoader(ds, batch_size=4, num_workers=2)) == [
        [0, 10, 20, 30],
        [40, 50, 60, 70],
        [80, 90],
    ]
    assert ds.batch_calls == 3
    # With item_concurrency > 1, each item is its own __getitem__ call.
    assert list(DataLoader(ds, batch_size=4, item_concurrency=2))[0] == [0, 1, 2, 3]
    assert ds.batch_calls == 3


def test_async_exception_keeps_its_type_and_fails_its_batch_only():
    loader = DataLoader(AsyncRange(12, fail_at=5), batch_size=4, num_workers=2, item_concurrency=4)
    it = iter(loader)
    assert next(it) == [0, 1, 2, 3]
    with pytest.raises(LookupError, match="no item 5"):
        next(it)
    assert next(it) == [8, 9, 10, 11]
    with pytest.raises(StopIteration):
        next(it)


def test_dropping_the_iterator_cancels_coroutines_without_waiting():
    ds = AsyncRange(1000, delay=3600, jitter=False)
    loader = DataLoader(ds, batch_size=8, num_workers=2, item_concurrency=8)
    it = iter(loader)
    deadline = time.monotonic() + 5
    while ds.running < 16:
        assert time.monotonic() < deadline, "coroutines never started"
        time.sleep(0.005)
    start = time.monotonic()
    del it
    assert time.monotonic() - start < 0.5
    while ds.running:
        assert time.monotonic() < deadline, "coroutines still running"
        time.sleep(0.005)
    assert ds.cancelled >= 16
    # The loader still works.
    ds.delay = 0
    assert sum(len(b) for b in loader) == 1000


def test_dropping_loaders_mid_epoch_awaits_every_coroutine(caplog):
    """The stopped event loop cancels the coroutines still pending and waits for them before
    it closes: none is left never awaited, no task is destroyed pending."""
    threads_before = set(threading.enumerate())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(100):
            ds = AsyncRange(1000, delay=0.0005 * (i % 4), jitter=False)
            loader = DataLoader(ds, batch_size=4, num_workers=i % 3 + 1, item_concurrency=i % 5 + 1)
            it = iter(loader)
            next(it)
            del it, loader
            gc.collect()
        deadline = time.monotonic() + 10
        while any(
            t.name == "dataloader-asyncio" and t not in threads_before
            for t in threading.enumerate()
        ):
            assert time.monotonic() < deadline, "event loop threads still running"
            time.sleep(0.01)
        gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]
    assert "destroyed but it is pending" not in caplog.text


_SLOW_ASYNC = """
import asyncio
from dataloader_rs import PyDataloader, PyAsyncDataset

class Slow(PyAsyncDataset):
    def __len__(self):
        return 4

    async def __getitem__(self, index):
        await asyncio.sleep(30)
        return index

it = iter(PyDataloader(Slow(), num_workers=1))
print("waiting", flush=True)
try:
    next(it)
except KeyboardInterrupt:
    print("interrupted", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ctrl_c_interrupts_a_wait_on_an_async_dataset():
    proc = subprocess.Popen(
        [sys.executable, "-c", _SLOW_ASYNC],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "waiting"
        time.sleep(0.3)
        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=10)
    finally:
        proc.kill()
    assert out.strip() == "interrupted", err
    assert time.monotonic() - start < 2


# ── base class checks ─────────────────────────────────────────────────────────


def test_pydataset_with_async_getitem_is_refused():
    class Wrong(PyDataset):
        def __len__(self):
            return 2

        async def __getitem__(self, index):
            return index

    with pytest.raises(TypeError, match="PyAsyncDataset"):
        DataLoader(Wrong())


def test_pydataset_with_async_getitems_is_refused():
    class Wrong(PyDataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

        async def __getitems__(self, indices):
            return indices

    with pytest.raises(TypeError, match="must not be async"):
        DataLoader(Wrong())


def test_pyasyncdataset_with_sync_getitem_is_refused():
    class Wrong(PyAsyncDataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    with pytest.raises(TypeError, match="must be async def"):
        DataLoader(Wrong())


def test_pyasyncdataset_without_getitem_is_refused():
    class Empty(PyAsyncDataset):
        def __len__(self):
            return 2

    with pytest.raises(TypeError, match="async def __getitem__"):
        DataLoader(Empty())


def test_pyasyncdataset_with_sync_aclose_is_refused():
    class Wrong(AsyncRange):
        def aclose(self):
            pass

    with pytest.raises(TypeError, match="aclose must be async def"):
        DataLoader(Wrong(2))


@pytest.mark.parametrize("base", [PyDataset, PyAsyncDataset])
def test_missing_len_is_refused(base):
    class NoLen(base):
        async def __getitem__(self, index):
            return index

    with pytest.raises(TypeError, match="__len__"):
        DataLoader(NoLen())


def test_both_bases_are_refused():
    try:

        class Both(PyDataset, PyAsyncDataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return index

    except TypeError:
        return  # Python itself refuses the two native bases.
    with pytest.raises(TypeError, match="not both"):
        DataLoader(Both())


def test_other_objects_are_refused():
    with pytest.raises(TypeError, match="must be a PyDataset"):
        DataLoader([1, 2, 3])


class ReturnsCoroutine(PyDataset):
    """A sync __getitem__ calling an async helper without awaiting it."""

    def __len__(self):
        return 8

    async def _load(self, index):
        return index

    def __getitem__(self, index):
        return self._load(index)


@pytest.mark.parametrize(("num_workers", "item_concurrency"), [(0, 1), (2, 1), (0, 2)])
def test_sync_getitem_returning_a_coroutine_fails_on_the_first_item(
    num_workers, item_concurrency, recwarn
):
    loader = DataLoader(
        ReturnsCoroutine(), batch_size=4, num_workers=num_workers, item_concurrency=item_concurrency
    )
    with pytest.raises(TypeError, match="returned an awaitable"):
        next(iter(loader))
    del loader
    assert not [w for w in recwarn if "never awaited" in str(w.message)]


# ── Rust datasets ─────────────────────────────────────────────────────────────


@pytest.fixture
def files(tmp_path):
    paths = []
    for i in range(10):
        path = tmp_path / f"{i}.txt"
        path.write_bytes(f"file-{i}".encode())
        paths.append(path)
    return paths


def test_file_dataset_bytes_and_transform(files):
    ds = FileDataset(files)
    assert len(ds) == 10
    assert isinstance(ds, RustDataset)
    assert not hasattr(ds, "map")
    batches = list(DataLoader(ds, batch_size=4, num_workers=2, item_concurrency=2))
    assert batches[0] == [b"file-0", b"file-1", b"file-2", b"file-3"]
    loader = DataLoader(
        ds, batch_size=10, item_concurrency=10, runtime_threads=2, transform=lambda i, b: b.decode()
    )
    assert next(iter(loader))[9] == "file-9"


def test_file_dataset_errors(files, tmp_path):
    it = iter(DataLoader(FileDataset([tmp_path / "missing"]), batch_size=1))
    with pytest.raises(OSError, match="missing"):
        next(it)
    it = iter(DataLoader(FileDataset(files, max_bytes=3), batch_size=1))
    with pytest.raises(ValueError, match="limit of 3"):
        next(it)


def test_file_dataset_retry_and_timeout_settings(files, tmp_path):
    ds = FileDataset(files, timeout=5.0, retries=2, backoff=0.01, max_backoff=0.1)
    assert next(iter(DataLoader(ds, batch_size=2))) == [b"file-0", b"file-1"]
    # A missing file is not transient: one attempt, at once.
    ds = FileDataset([tmp_path / "missing"], retries=3, backoff=10.0, max_backoff=10.0)
    start = time.monotonic()
    with pytest.raises(OSError, match="after 1 attempt"):
        next(iter(DataLoader(ds)))
    assert time.monotonic() - start < 5
    with pytest.raises(ValueError, match="timeout"):
        FileDataset(files, timeout=-1)
    with pytest.raises(ValueError, match="at most 32"):
        FileDataset(files, retries=33)


def test_file_dataset_from_template(files, tmp_path):
    for slide, level in [("a b", 0), ("a b", 1), ("c", 0)]:
        (tmp_path / slide).mkdir(exist_ok=True)
        (tmp_path / slide / f"{level}.bin").write_bytes(f"{slide}-{level}".encode())
    ds = FileDataset.from_template(
        f"{tmp_path}/{{slide}}/{{level}}.bin",
        {"slide": ["a b", "a b", "c"], "level": [0, 1, 0]},
        retries=1,
        timeout=5.0,
    )
    assert len(ds) == 3
    assert list(DataLoader(ds, batch_size=3)) == [[b"a b-0", b"a b-1", b"c-0"]]
    ds = FileDataset.from_template(f"{tmp_path}/{{index}}.txt", length=10)
    assert [i for b in DataLoader(ds, batch_size=4) for i in b][9] == b"file-9"


@pytest.mark.parametrize("value", ["..", ".", "../etc", "a/b", "a\\b", "a\0b"])
def test_file_dataset_template_values_cannot_leave_the_directory(tmp_path, value):
    with pytest.raises(ValueError, match="item 1"):
        FileDataset.from_template(f"{tmp_path}/{{name}}.bin", {"name": ["ok", value]})


class _Handler(BaseHTTPRequestHandler):
    hits = collections.Counter()

    def do_GET(self):
        _Handler.hits[self.path] += 1
        parts = self.path.strip("/").split("/")
        if parts[0] == "ok":
            self._send(200, f"item-{parts[1]}".encode())
        elif parts[0] == "status":
            self._send(int(parts[1]), b"no")
        elif parts[0] == "flaky" and _Handler.hits[self.path] <= 2:
            self._send(503, b"busy")
        elif parts[0] == "flaky":
            self._send(200, b"ok")
        elif parts[0] == "echo":
            self._send(200, self.path.encode())
        elif parts[0] == "header":
            self._send(200, self.headers.get("X-Token", "").encode())
        else:
            self._send(404, b"not found")

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_http_dataset_in_order(server):
    urls = [f"{server}/ok/{i}" for i in range(40)]
    loader = DataLoader(
        HttpDataset(urls),
        batch_size=8,
        num_workers=2,
        item_concurrency=8,
        transform=lambda i, b: b.decode(),
    )
    items = [i for b in loader for i in b]
    assert items == [f"item-{i}" for i in range(40)]


def test_http_dataset_headers_retries_and_errors(server):
    ds = HttpDataset([f"{server}/header"], headers={"X-Token": "secret"})
    assert next(iter(DataLoader(ds))) == [b"secret"]

    ds = HttpDataset([f"{server}/flaky/a"], retries=3, backoff=0.001)
    assert next(iter(DataLoader(ds))) == [b"ok"]
    assert _Handler.hits["/flaky/a"] == 3

    it = iter(DataLoader(HttpDataset([f"{server}/status/404"], retries=3)))
    with pytest.raises(OSError, match="404"):
        next(it)
    assert _Handler.hits["/status/404"] == 1


def test_http_dataset_rejects_bad_settings():
    with pytest.raises(ValueError):
        HttpDataset(["not a url"])
    with pytest.raises(ValueError, match="timeout"):
        HttpDataset(["http://x"], timeout=-1)
    with pytest.raises(ValueError):
        HttpDataset(["http://x"], headers={"bad header": "x"})


def test_http_dataset_from_template_with_numpy_columns(server):
    np = pytest.importorskip("numpy")

    cols, rows = np.meshgrid(np.arange(5, dtype=np.int32), np.arange(4, dtype=np.uint16))
    xs, ys = (cols.ravel() * 224).astype(np.int64), (rows.ravel() * 224).astype(np.uint64)
    ds = HttpDataset.from_template(
        server + "/echo/{slide}/tile?x={x}&y={y}&w=224&i={index}&b={{}}",
        {"x": xs, "y": ys, "slide": ["a b/c"] * len(xs)},
    )
    assert len(ds) == 20
    loader = DataLoader(
        ds, batch_size=8, num_workers=2, item_concurrency=4, transform=lambda i, b: b.decode()
    )
    items = [i for b in loader for i in b]
    assert items == [
        f"/echo/a%20b%2Fc/tile?x={x}&y={y}&w=224&i={i}&b={{}}"
        for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist(), strict=True))
    ]

    ds = HttpDataset.from_template(server + "/ok/{index}", length=3, retries=0)
    assert list(DataLoader(ds, batch_size=3)) == [[b"item-0", b"item-1", b"item-2"]]


@pytest.mark.parametrize(
    ("template", "columns", "length", "error", "match"),
    [
        ("", None, 1, ValueError, "empty"),
        ("http://h/{x", {"x": [1]}, None, ValueError, "unclosed"),
        ("http://h/x}", None, 1, ValueError, "unbalanced"),
        ("http://h/{y}", {"x": [1]}, None, ValueError, "unknown placeholder"),
        ("http://h/{x}", {"x": [1], "z": [1]}, None, ValueError, "not used"),
        ("http://h/{x}/{y}", {"x": [1, 2], "y": [1]}, None, ValueError, "has 1 values"),
        ("http://h/{x}", {"x": [1, 2]}, 3, ValueError, "the length has 3"),
        ("http://h/{index}", None, None, ValueError, "needs a length"),
        ("{x}", {"x": [1]}, None, ValueError, "invalid URL"),
        ("http://h/{x}", {"x": [".."]}, None, ValueError, "path segment"),
        ("http://h/{x}", {"x": "abc"}, None, TypeError, "sequence of ints"),
        ("http://h/{x}", {"x": [1.5]}, None, TypeError, "sequence of ints"),
        ("http://h/{x}", {"x": [1, "a"]}, None, TypeError, "sequence of ints"),
        ("http://h/{x}", {"x": [2**64]}, None, TypeError, "sequence of ints"),
        ("http://h/{x}", {1: [1]}, None, TypeError, "names must be strs"),
    ],
)
def test_http_dataset_from_template_errors(template, columns, length, error, match):
    with pytest.raises(error, match=match):
        HttpDataset.from_template(template, columns, length=length)


def test_object_store_dataset_on_an_http_store(server):
    ds = ObjectStoreDataset(f"{server}/ok", ["1", "2"], options={"allow_http": "true"})
    assert list(DataLoader(ds, batch_size=2, item_concurrency=2)) == [[b"item-1", b"item-2"]]
    with pytest.raises(ValueError, match="no_such_option"):
        ObjectStoreDataset(f"{server}/ok", ["1"], options={"no_such_option": "1"})
    with pytest.raises(ValueError, match="timeout"):
        ObjectStoreDataset(f"{server}/ok", ["1"], options={"allow_http": "true", "timeout": "5s"})
    with pytest.raises(ValueError, match="timeout"):
        ObjectStoreDataset(f"{server}/ok", ["1"], options={"allow_http": "true"}, timeout=-1)


def test_object_store_dataset_retries_with_our_settings(server):
    options = {"allow_http": "true"}
    ds = ObjectStoreDataset(
        f"{server}/flaky", ["os-a"], options=options, retries=3, backoff=0.001, timeout=5.0
    )
    assert next(iter(DataLoader(ds))) == [b"ok"]
    assert _Handler.hits["/flaky/os-a"] == 3
    ds = ObjectStoreDataset(f"{server}/flaky", ["os-b"], options=options, retries=0)
    with pytest.raises(OSError, match="after up to 1 attempt"):
        next(iter(DataLoader(ds)))
    assert _Handler.hits["/flaky/os-b"] == 1


def test_secrets_stay_out_of_reprs_and_errors():
    http = HttpDataset(["http://x/a"], headers={"Authorization": "Bearer s3cret"})
    store = ObjectStoreDataset(
        "s3://bucket/prefix",
        ["k"],
        options={
            "aws_region": "eu-west-3",
            "aws_access_key_id": "AKIDEXAMPLE",
            "aws_secret_access_key": "s3cret",
        },
    )
    for ds in (http, store):
        assert "s3cret" not in repr(ds)
    with pytest.raises(ValueError, match="password") as info:
        ObjectStoreDataset("https://user:s3cret@host/prefix?sig=s1gnature", ["k"])
    assert "s3cret" not in str(info.value) and "s1gnature" not in str(info.value)


def test_object_store_dataset_from_template(server):
    options = {"allow_http": "true"}
    ds = ObjectStoreDataset.from_template(
        f"{server}/ok", "{name}", {"name": ["a b", "c"]}, options=options, retries=1
    )
    assert len(ds) == 2
    assert list(DataLoader(ds, batch_size=2)) == [[b"item-a%20b", b"item-c"]]
    ds = ObjectStoreDataset.from_template(f"{server}/ok", "{index}", length=3, options=options)
    assert list(DataLoader(ds, batch_size=3)) == [[b"item-0", b"item-1", b"item-2"]]
    for value in ["", ".", "..", "a/b"]:
        with pytest.raises(ValueError, match="item 1"):
            ObjectStoreDataset.from_template(
                f"{server}/ok", "{k}", {"k": ["a", value]}, options=options
            )


# ── aclose ────────────────────────────────────────────────────────────────────


class Session:
    """An aiohttp-like session: bound to the loop it was opened on."""

    def __init__(self):
        self.loop = None
        self.closed_on = []

    async def close(self):
        self.closed_on.append(asyncio.get_running_loop())


class WithSession(AsyncRange):
    def __init__(self, n):
        super().__init__(n)
        self.session = Session()

    async def __getitem__(self, index):
        self.session.loop = asyncio.get_running_loop()
        return await super().__getitem__(index)

    async def aclose(self):
        await self.session.close()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_aclose_is_awaited_once_on_the_loop_when_the_loader_is_dropped(num_workers):
    dataset = WithSession(8)
    loader = DataLoader(dataset, batch_size=4, num_workers=num_workers)
    for _ in range(2):
        assert [i for b in loader for i in b] == list(range(8))
    assert dataset.session.closed_on == []
    del loader
    gc.collect()
    assert dataset.session.closed_on == [dataset.session.loop]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_async_loader_in_a_reference_cycle_is_collected(num_workers):
    """An async loader and its iterator kept by an object that its collate_fn is bound to are
    freed by gc, after aclose: dropping the iterator waits for its batch tasks and pool jobs,
    dropping the loader awaits aclose (see test_loader_in_a_reference_cycle_is_collected)."""

    class Pipeline:
        def __init__(self):
            self.dataset = WithSession(8)
            self.loader = DataLoader(
                self.dataset, batch_size=4, num_workers=num_workers, collate_fn=self.collate
            )

        def collate(self, items):
            return items

    pipeline = Pipeline()
    session = pipeline.dataset.session
    assert list(pipeline.loader) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    pipeline.it = iter(pipeline.loader)
    assert [next(pipeline.it), next(pipeline.it)] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    ref = weakref.ref(pipeline)
    del pipeline
    gc.collect()
    assert ref() is None
    assert session.closed_on == [session.loop]


def test_aclose_is_awaited_when_no_item_was_loaded():
    dataset = WithSession(8)
    DataLoader(dataset)
    gc.collect()
    assert len(dataset.session.closed_on) == 1


def test_dataset_without_aclose_is_fine():
    loader = DataLoader(AsyncRange(4), batch_size=2)
    assert list(loader) == [[0, 1], [2, 3]]
    del loader
    gc.collect()


def test_aclose_raising_is_logged(caplog):
    class Failing(AsyncRange):
        async def aclose(self):
            raise OSError("session already gone")

    loader = DataLoader(Failing(4), batch_size=2)
    assert list(loader) == [[0, 1], [2, 3]]
    with caplog.at_level(logging.ERROR, logger="dataloader_rs"):
        del loader
        gc.collect()
    assert "aclose() failed: OSError('session already gone')" in caplog.text


def test_hanging_aclose_is_bounded(caplog):
    class Hanging(AsyncRange):
        async def aclose(self):
            await asyncio.sleep(3600)

    loader = DataLoader(Hanging(4), batch_size=2)
    assert list(loader) == [[0, 1], [2, 3]]
    start = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="dataloader_rs"):
        del loader
        gc.collect()
    assert time.monotonic() - start < 7
    assert "did not finish in 5.0 s" in caplog.text
