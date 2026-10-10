"""Async datasets: ``async def __getitem__`` / ``async def __getitems__``.

Batches are awaited concurrently on one persistent event loop thread, so
network-bound datasets overlap their requests without a thread per request.
"""

import asyncio
import threading
import time

import pytest

from dataloader_rs import PyDataloader as DataLoader
from dataloader_rs import PyDataset


class AsyncItemDs(PyDataset):
    """async __getitem__ that 'waits on the network' and tracks overlap."""

    def __init__(self, n, latency=0.0):
        super().__init__()
        self.n = n
        self.latency = latency
        self.in_flight = 0
        self.peak = 0

    def __len__(self):
        return self.n

    async def __getitem__(self, index):
        self.in_flight += 1  # single event loop thread: no lock needed
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(self.latency)
        self.in_flight -= 1
        return index


class AsyncBatchDs(PyDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.calls = []

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        raise AssertionError("__getitems__ must be used")

    async def __getitems__(self, indices):
        self.calls.append(indices)
        await asyncio.sleep(0)
        return [i * 10 for i in indices]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_async_getitem_yields_batches_in_order(num_workers):
    loader = DataLoader(AsyncItemDs(10), batch_size=4, num_workers=num_workers)
    assert list(loader) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_requests_overlap_within_and_across_batches():
    # 64 samples x 50 ms: sequential fetching would take 3.2 s.
    ds = AsyncItemDs(64, latency=0.05)
    loader = DataLoader(ds, batch_size=8, num_workers=2, prefetch_depth=2)
    start = time.perf_counter()
    out = [x for batch in loader for x in batch]
    elapsed = time.perf_counter() - start
    assert out == list(range(64))
    assert elapsed < 1.0, f"took {elapsed:.2f}s"
    assert 8 < ds.peak <= (2 + 2) * 8, f"peak in flight {ds.peak}"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_async_getitems_called_once_per_batch(num_workers):
    ds = AsyncBatchDs(6)
    loader = DataLoader(ds, batch_size=3, num_workers=num_workers)
    assert list(loader) == [[0, 10, 20], [30, 40, 50]]
    assert ds.calls == [[0, 1, 2], [3, 4, 5]]


def test_async_with_collate_fn_and_sampler():
    loader = DataLoader(AsyncItemDs(6), batch_size=3, sampler=[5, 4, 3, 2, 1, 0], collate_fn=sum)
    assert list(loader) == [12, 3]


def test_exception_keeps_its_type_and_epoch_continues():
    class FailingDs(AsyncItemDs):
        async def __getitem__(self, index):
            if index == 5:
                raise KeyError(f"missing {index}")
            return index

    it = iter(DataLoader(FailingDs(12), batch_size=4))
    assert next(it) == [0, 1, 2, 3]
    with pytest.raises(KeyError, match="missing 5"):
        next(it)
    assert next(it) == [8, 9, 10, 11]


def test_getitems_wrong_length_raises():
    class ShortDs(AsyncBatchDs):
        async def __getitems__(self, indices):
            return indices[:-1]

    with pytest.raises(ValueError, match="__getitems__ returned 1 samples for 2 indices"):
        next(iter(DataLoader(ShortDs(4), batch_size=2)))


def test_event_loop_persists_across_epochs():
    """Loop-bound objects created lazily (like an aiohttp session) keep working."""

    class SessionDs(AsyncItemDs):
        def __init__(self, n):
            super().__init__(n)
            self.limit = None
            self.loops = set()

        async def __getitem__(self, index):
            if self.limit is None:
                self.limit = asyncio.Semaphore(2)  # binds to the running loop
            self.loops.add(id(asyncio.get_running_loop()))
            async with self.limit:
                await asyncio.sleep(0)
            return index

    ds = SessionDs(8)
    loader = DataLoader(ds, batch_size=4)
    for _ in range(3):
        assert len(list(loader)) == 2
    assert len(ds.loops) == 1


def test_one_loop_thread_per_loader():
    loader = DataLoader(AsyncItemDs(8), batch_size=2)
    for _ in range(3):
        list(loader)
    loop_threads = [t for t in threading.enumerate() if t.name == "dataloader-asyncio"]
    assert len(loop_threads) >= 1
    del loader


def test_early_break_cancels_pending_and_next_epoch_works():
    ds = AsyncItemDs(40, latency=0.01)
    loader = DataLoader(ds, batch_size=4, prefetch_depth=4)
    for i, _ in enumerate(loader):
        if i == 1:
            break
    assert [x for batch in loader for x in batch] == list(range(40))


def test_len_of_async_iterator():
    it = iter(DataLoader(AsyncItemDs(10), batch_size=4))
    assert len(it) == 3
    next(it)
    assert len(it) == 2


def test_sync_datasets_are_unaffected():
    class SyncDs(PyDataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index

    assert list(DataLoader(SyncDs(), batch_size=2)) == [[0, 1], [2]]


# ── max_concurrency ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_max_concurrency_caps_getitem_calls_in_flight(limit):
    ds = AsyncItemDs(40, latency=0.005)
    loader = DataLoader(ds, batch_size=8, prefetch_depth=4, max_concurrency=limit)
    assert [x for batch in loader for x in batch] == list(range(40))
    assert ds.peak == limit, f"peak {ds.peak} with max_concurrency={limit}"


def test_max_concurrency_counts_getitems_calls():
    class CountingBatchDs(AsyncBatchDs):
        def __init__(self, n):
            super().__init__(n)
            self.in_flight = 0
            self.peak = 0

        async def __getitems__(self, indices):
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(0.005)
            self.in_flight -= 1
            return indices

    ds = CountingBatchDs(40)
    loader = DataLoader(ds, batch_size=4, prefetch_depth=8, max_concurrency=2)
    assert [x for batch in loader for x in batch] == list(range(40))
    assert ds.peak == 2


def test_max_concurrency_is_shared_across_epochs():
    ds = AsyncItemDs(16, latency=0.002)
    loader = DataLoader(ds, batch_size=4, max_concurrency=2)
    for _ in range(3):
        assert len(list(loader)) == 4
    assert ds.peak == 2


def test_max_concurrency_requires_async_dataset():
    class SyncDs(PyDataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    with pytest.raises(ValueError, match="requires an async def"):
        DataLoader(SyncDs(), max_concurrency=4)


def test_max_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="max_concurrency must be > 0"):
        DataLoader(AsyncItemDs(2), max_concurrency=0)
