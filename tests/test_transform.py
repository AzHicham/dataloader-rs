"""``PyDataloader(transform=f)``: the per-item transform the loader applies to
every kind of dataset, called as ``f(index, item)``."""

import asyncio
import gc
import threading
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import dataloader_rs
from dataloader_rs import (
    FileDataset,
    HttpDataset,
    ObjectStoreDataset,
    PyAsyncDataset,
    PyDataset,
    RustDataset,
)
from dataloader_rs import PyDataloader as DataLoader

# (num_workers, item_concurrency): direct, item lanes, workers, workers with lanes.
MODES = [(0, 1), (0, 4), (2, 1), (2, 4)]


class Numbers(PyDataset):
    """Item ``i`` is ``i``; ``__getitem__`` raises ``LookupError`` at ``fail_at``."""

    def __init__(self, n, fail_at=None):
        super().__init__()
        self.n = n
        self.fail_at = fail_at

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if index == self.fail_at:
            raise LookupError(f"no item {index}")
        return index


class AsyncNumbers(PyAsyncDataset):
    """Item ``i`` is ``i``, after a short sleep."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    async def __getitem__(self, index):
        await asyncio.sleep(0.001 * (index % 3))
        return index


class _Handler(BaseHTTPRequestHandler):
    """``GET /<anything>/<i>`` answers ``<i>``."""

    def do_GET(self):
        body = self.path.rsplit("/", 1)[-1].encode()
        self.send_response(200)
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


@pytest.fixture
def files(tmp_path):
    paths = []
    for i in range(20):
        path = tmp_path / f"{i}.txt"
        path.write_bytes(str(i).encode())
        paths.append(path)
    return paths


def as_int(item):
    """Items of the Rust datasets are ``bytes``; the others are ints already."""
    return int(item) if isinstance(item, bytes) else item


def make_dataset(kind, server, files):
    """A dataset of 20 items whose item ``i`` is ``i`` (or ``b"i"``)."""
    if kind == "py":
        return Numbers(20)
    if kind == "async":
        return AsyncNumbers(20)
    if kind == "file":
        return FileDataset(files)
    if kind == "http":
        return HttpDataset([f"{server}/n/{i}" for i in range(20)])
    if kind == "template":
        return HttpDataset.from_template(server + "/n/{index}", length=20)
    if kind == "object":
        keys = [str(i) for i in range(20)]
        return ObjectStoreDataset(f"{server}/n", keys, options={"allow_http": "true"})
    raise AssertionError(kind)


KINDS = ["py", "async", "file", "http", "template", "object"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_transform_applies_to_every_dataset_kind(
    kind, num_workers, item_concurrency, server, files
):
    loader = DataLoader(
        make_dataset(kind, server, files),
        batch_size=6,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
        transform=lambda index, item: (index, as_int(item) * 10),
    )
    for _ in range(2):
        assert [i for b in loader for i in b] == [(i, i * 10) for i in range(20)]


@pytest.mark.parametrize("kind", KINDS)
def test_rust_items_are_bytes_without_transform(kind, server, files):
    batch = next(iter(DataLoader(make_dataset(kind, server, files), batch_size=3)))
    if kind in ("py", "async"):
        assert batch == [0, 1, 2]
    else:
        assert batch == [b"0", b"1", b"2"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_transform_exception_keeps_its_type(kind, num_workers, item_concurrency, server, files):
    def pick(index, item):
        return {0: "zero", 1: "one", 2: "two"}[as_int(item)]

    it = iter(
        DataLoader(
            make_dataset(kind, server, files),
            batch_size=3,
            num_workers=num_workers,
            item_concurrency=item_concurrency,
            transform=pick,
        )
    )
    assert next(it) == ["zero", "one", "two"]
    with pytest.raises(KeyError):
        next(it)


@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_transform_feeds_collate_fn(num_workers, item_concurrency):
    loader = DataLoader(
        Numbers(6),
        batch_size=3,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
        transform=lambda index, item: str(item),
        collate_fn="-".join,
    )
    assert list(loader) == ["0-1-2", "3-4-5"]


class Compose:
    """torchvision's ``Compose``: a callable object applying ``transforms`` in turn."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def test_torchvision_style_compose_takes_the_item_only():
    compose = Compose([lambda x: x + 1, str])
    loader = DataLoader(Numbers(4), batch_size=2, num_workers=2, transform=lambda i, x: compose(x))
    assert list(loader) == [["1", "2"], ["3", "4"]]


def test_torchvision_compose_on_tensors():
    torch = pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")

    class Images(PyDataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return torch.full((3, 4, 4), float(index))

    compose = transforms.Compose([transforms.Normalize([1.0] * 3, [2.0] * 3)])
    loader = DataLoader(Images(), batch_size=4, num_workers=2, transform=lambda i, x: compose(x))
    batch = next(iter(loader))
    assert [float(t[0, 0, 0]) for t in batch] == [-0.5, 0.0, 0.5, 1.0]


def test_transform_must_be_callable():
    with pytest.raises(TypeError, match="transform must be callable"):
        DataLoader(Numbers(4), transform=3)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_transform_gets_the_sampled_index(kind, num_workers, item_concurrency, server, files):
    loader = DataLoader(
        make_dataset(kind, server, files),
        batch_size=6,
        shuffle=True,
        seed=5,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
        transform=lambda index, item: (index, as_int(item)),
    )
    seen = []
    for batch in loader:
        assert all(index == item for index, item in batch)
        seen.extend(index for index, _ in batch)
    assert seen != list(range(20))
    assert sorted(seen) == list(range(20))


def test_transform_picks_coordinates_by_index():
    np = pytest.importorskip("numpy")
    coords = np.arange(40).reshape(20, 2)
    loader = DataLoader(
        Numbers(20),
        batch_size=8,
        shuffle=True,
        seed=1,
        transform=lambda index, item: (item, coords[index]),
    )
    for batch in loader:
        assert [c.tolist() for _, c in batch] == [[2 * i, 2 * i + 1] for i, _ in batch]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_with_a_transform_in_a_reference_cycle_is_collected(num_workers):
    """A loader kept by an object that its transform is bound to is freed by gc."""

    class Pipeline:
        def __init__(self):
            self.loader = DataLoader(
                Numbers(8), batch_size=4, num_workers=num_workers, transform=self.double
            )

        def double(self, index, item):
            return item * 2

    pipeline = Pipeline()
    assert list(pipeline.loader) == [[0, 2, 4, 6], [8, 10, 12, 14]]
    pipeline.it = iter(pipeline.loader)
    next(pipeline.it)
    ref = weakref.ref(pipeline)
    del pipeline
    gc.collect()
    assert ref() is None


def test_removed_map_api_is_gone():
    assert not hasattr(RustDataset, "map")
    stub = Path(dataloader_rs.__file__).with_name("__init__.pyi").read_text()
    assert "def map" not in stub
    assert ".map(" not in stub
