"""Item-level parallelism: ``parallel_items=True`` and ``num_threads``.

The items of each batch are fetched concurrently on the loader's thread pool,
each with its own ``__getitem__`` call — for I/O-bound datasets (network,
object storage) where one batch would otherwise wait for its items one by one.
"""

import threading
import time

import pytest

from dataloader_rs import PyDataloader as DataLoader
from dataloader_rs import PyDataset
from tests.py_dataloader_test_utils import ListDataset


class SlowDs(PyDataset):
    """Sleeps per item (releases the GIL) and records how many calls overlap."""

    def __init__(self, n, delay=0.02):
        super().__init__()
        self.n = n
        self.delay = delay
        self._lock = threading.Lock()
        self._running = 0
        self.max_running = 0

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        with self._lock:
            self._running += 1
            self.max_running = max(self.max_running, self._running)
        time.sleep(self.delay)
        with self._lock:
            self._running -= 1
        return index


@pytest.mark.parametrize("num_workers", [0, 2])
def test_parallel_items_same_batches_as_sequential(num_workers):
    expected = list(DataLoader(ListDataset(range(50)), batch_size=8))
    loader = DataLoader(
        ListDataset(range(50)),
        batch_size=8,
        num_workers=num_workers,
        parallel_items=True,
        num_threads=4,
    )
    assert list(loader) == expected
    assert list(loader) == expected, "a second epoch must match too"


@pytest.mark.parametrize("num_workers", [0, 1])
def test_items_of_one_batch_run_concurrently(num_workers):
    ds = SlowDs(16)
    loader = DataLoader(
        ds, batch_size=16, num_workers=num_workers, parallel_items=True, num_threads=8
    )
    assert list(loader) == [list(range(16))]
    assert ds.max_running > 1, "items of a batch were fetched one at a time"


def test_without_parallel_items_a_batch_is_fetched_sequentially():
    ds = SlowDs(8)
    assert list(DataLoader(ds, batch_size=8, num_workers=1)) == [list(range(8))]
    assert ds.max_running == 1


@pytest.mark.parametrize("num_workers", [0, 2])
def test_parallel_items_keeps_exception_types(num_workers):
    class KeyErrorDs(ListDataset):
        def __getitem__(self, idx):
            if idx == 5:
                raise KeyError(f"missing {idx}")
            return super().__getitem__(idx)

    loader = DataLoader(
        KeyErrorDs(range(8)), batch_size=4, num_workers=num_workers, parallel_items=True
    )
    with pytest.raises(KeyError, match="missing 5"):
        list(loader)


def test_parallel_items_fetches_with_getitem_even_if_getitems_exists():
    class BatchedDs(ListDataset):
        def __getitems__(self, indices):
            raise AssertionError("__getitems__ must not be used with parallel_items")

    loader = DataLoader(BatchedDs(range(6)), batch_size=3, parallel_items=True, num_threads=2)
    assert list(loader) == [[0, 1, 2], [3, 4, 5]]


def test_rejects_zero_num_threads():
    with pytest.raises(ValueError, match="num_threads must be > 0"):
        DataLoader(ListDataset(range(4)), parallel_items=True, num_threads=0)
