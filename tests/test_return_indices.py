"""``PyDataloader(return_indices=True)``: each batch comes as ``(batch, indices)``, the
dataset indices of its items in the order the sampler drew them, for every kind of dataset
and collator; and the Rust-only path never attaches to the interpreter per item."""

import asyncio
import gc
import struct
import zlib

import numpy as np
import pytest
from dataloader_rs.dataloader_rs import _attach_count

from dataloader_rs import (
    FileDataset,
    ImageBatch,
    ImageCollator,
    PyAsyncDataset,
    PyDataset,
)
from dataloader_rs import PyDataloader as DataLoader
from dataloader_rs import transforms as T

N = 22
# (num_workers, item_concurrency): direct, item lanes, workers with lanes.
MODES = [(0, 1), (0, 4), (2, 4)]


def png(value: int) -> bytes:
    """A 4 x 3 RGB PNG whose every byte is ``value``."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    rows = b"".join(b"\x00" + bytes([value]) * 12 for _ in range(3))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 3, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


PNGS = [png(i) for i in range(N)]


class Pngs(PyDataset):
    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at

    def __len__(self):
        return N

    def __getitem__(self, index):
        if index == self.fail_at:
            raise KeyError(index)
        return PNGS[index]


class AsyncPngs(PyAsyncDataset):
    def __len__(self):
        return N

    async def __getitem__(self, index):
        await asyncio.sleep(0.001 * (index % 3))
        return PNGS[index]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pngs")
    paths = [directory / f"{i}.png" for i in range(N)]
    for path, data in zip(paths, PNGS, strict=True):
        path.write_bytes(data)
    return paths


def dataset(kind, files):
    return {"py": Pngs, "async": AsyncPngs, "file": lambda: FileDataset(files)}[kind]()


def check_indices(indices, batch_size=4):
    assert type(indices) is list and all(type(i) is int for i in indices)
    assert 0 < len(indices) <= batch_size


@pytest.mark.parametrize("kind", ["py", "async", "file"])
@pytest.mark.parametrize("collate", ["none", "function", "images"])
@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_batches_come_with_their_indices(files, kind, collate, num_workers, item_concurrency):
    options = {
        "none": {},
        "function": {"collate_fn": tuple},
        "images": {
            "transform": T.Compose([T.Decode(), T.ToRgb()]),
            "collate_fn": ImageCollator(layout="NHWC"),
        },
    }[collate]
    loader = DataLoader(
        dataset(kind, files),
        batch_size=4,
        shuffle=True,
        seed=7,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
        return_indices=True,
        **options,
    )
    for _epoch in range(2):
        seen = []
        for batch, indices in loader:
            check_indices(indices)
            if collate == "images":
                assert isinstance(batch, ImageBatch)
                pixels = np.asarray(batch)
                assert pixels.shape == (len(indices), 3, 4, 3)
                assert [int(p.min()) for p in pixels] == indices
                assert [int(p.max()) for p in pixels] == indices
            else:
                assert type(batch) is (tuple if collate == "function" else list)
                assert [bytes(item) for item in batch] == [PNGS[i] for i in indices]
            seen += indices
        assert seen != list(range(N)), "the sampler shuffles"
        assert sorted(seen) == list(range(N))


def test_numpy_takes_the_indices_as_they_are():
    coords = np.arange(2 * N).reshape(N, 2)
    loader = DataLoader(Pngs(), batch_size=5, shuffle=True, return_indices=True)
    for _batch, indices in loader:
        np.testing.assert_array_equal(coords[indices], [(2 * i, 2 * i + 1) for i in indices])


def test_drop_last_drops_the_indices_too(files):
    loader = DataLoader(
        FileDataset(files), batch_size=4, shuffle=True, drop_last=True, return_indices=True
    )
    batches = list(loader)
    assert len(batches) == len(loader) == N // 4
    for batch, indices in batches:
        assert len(batch) == len(indices) == 4
        assert [bytes(item) for item in batch] == [PNGS[i] for i in indices]


def test_a_python_sampler_gives_the_indices():
    order = [5, 3, 21, 0, 9, 2, 2]
    loader = DataLoader(Pngs(), batch_size=3, sampler=order, return_indices=True)
    assert [indices for _batch, indices in loader] == [[5, 3, 21], [0, 9, 2], [2]]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_failed_batch_keeps_the_next_ones_aligned(num_workers):
    loader = DataLoader(Pngs(fail_at=4), batch_size=3, num_workers=num_workers, return_indices=True)
    iterator = iter(loader)
    assert next(iterator)[1] == [0, 1, 2]
    with pytest.raises(KeyError):
        next(iterator)
    batch, indices = next(iterator)
    assert indices == [6, 7, 8]
    assert list(batch) == PNGS[6:9]


def test_without_return_indices_batches_are_unchanged():
    loader = DataLoader(Pngs(), batch_size=4, collate_fn=tuple)
    batches = list(loader)
    assert all(type(batch) is tuple for batch in batches)
    assert [item for batch in batches for item in batch] == PNGS


@pytest.mark.parametrize("item_concurrency", [1, 4])
def test_the_rust_path_never_attaches_per_item(files, item_concurrency):
    """A Rust dataset, the Rust transforms and an ``ImageCollator``: items are fetched,
    decoded and collated without the interpreter; only ``__next__`` builds Python objects,
    once per batch. A Python transform on the same loader attaches for every item."""
    rust = T.Compose([T.Decode(), T.ToRgb(), T.ToFloat(), T.ToChw()])

    def loader(transform):
        return DataLoader(
            FileDataset(files),
            batch_size=4,
            shuffle=True,
            num_workers=2,
            item_concurrency=item_concurrency,
            transform=transform,
            collate_fn=ImageCollator(),
            return_indices=True,
        )

    native = loader(rust)
    gc.collect()  # no epoch of an earlier test is left running
    before = _attach_count()
    seen = []
    for batch, indices in native:
        assert isinstance(batch, ImageBatch)
        assert batch.shape == (len(indices), 3, 3, 4)
        np.testing.assert_allclose(
            np.asarray(batch).reshape(len(indices), -1).max(axis=1), np.array(indices) / 255
        )
        seen += indices
    assert _attach_count() == before
    assert sorted(seen) == list(range(N))

    python = loader(lambda _index, data: rust(data))
    before = _attach_count()
    assert sum(len(indices) for _batch, indices in python) == N
    assert _attach_count() - before >= N
