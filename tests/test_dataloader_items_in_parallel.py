"""Item-level parallelism: ``items_in_parallel``.

With ``items_in_parallel=n > 1``, up to ``n`` items of each batch are fetched
concurrently, each with its own ``__getitem__`` call — for I/O-bound datasets
(network, object storage) where a batch would otherwise wait for its items one
by one.
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
def test_items_in_parallel_same_batches_as_sequential(num_workers):
    expected = list(DataLoader(ListDataset(range(50)), batch_size=8))
    loader = DataLoader(
        ListDataset(range(50)),
        batch_size=8,
        num_workers=num_workers,
        items_in_parallel=4,
    )
    assert list(loader) == expected
    assert list(loader) == expected, "a second epoch must match too"


@pytest.mark.parametrize("num_workers", [0, 1])
def test_items_of_one_batch_run_concurrently(num_workers):
    ds = SlowDs(16)
    loader = DataLoader(ds, batch_size=16, num_workers=num_workers, items_in_parallel=4)
    assert list(loader) == [list(range(16))]
    assert ds.max_running == 4, f"expected 4 items in flight, saw {ds.max_running}"


def test_by_default_a_batch_is_fetched_sequentially():
    ds = SlowDs(8)
    assert list(DataLoader(ds, batch_size=8, num_workers=1)) == [list(range(8))]
    assert ds.max_running == 1


@pytest.mark.parametrize("num_workers", [0, 2])
def test_items_in_parallel_keeps_exception_types(num_workers):
    class KeyErrorDs(ListDataset):
        def __getitem__(self, idx):
            if idx == 5:
                raise KeyError(f"missing {idx}")
            return super().__getitem__(idx)

    loader = DataLoader(
        KeyErrorDs(range(8)), batch_size=4, num_workers=num_workers, items_in_parallel=2
    )
    with pytest.raises(KeyError, match="missing 5"):
        list(loader)


def test_items_in_parallel_fetches_with_getitem_even_if_getitems_exists():
    class BatchedDs(ListDataset):
        def __getitems__(self, indices):
            raise AssertionError("__getitems__ must not be used with items_in_parallel > 1")

    # The last batch holds a single item, which needs no lane.
    loader = DataLoader(BatchedDs(range(7)), batch_size=3, items_in_parallel=2)
    assert list(loader) == [[0, 1, 2], [3, 4, 5], [6]]


def test_rejects_zero_items_in_parallel():
    with pytest.raises(ValueError, match="items_in_parallel must be > 0"):
        DataLoader(ListDataset(range(4)), items_in_parallel=0)
