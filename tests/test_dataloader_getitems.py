"""Batched fetching via ``__getitems__`` (PyTorch's batched-fetch protocol).

A dataset that defines ``__getitems__(indices)`` is called once per batch
with the list of indices instead of calling ``__getitem__`` once per index.
"""

import threading

import pytest

from dataloader_rs import PyDataloader as DataLoader
from dataloader_rs import PyDataset


class BatchedDs(PyDataset):
    """Records every __getitems__ call; __getitem__ must never be used."""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.calls = []
        self._lock = threading.Lock()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        raise AssertionError("__getitem__ must not be called when __getitems__ exists")

    def __getitems__(self, indices):
        with self._lock:
            self.calls.append(indices)
        return [i * 10 for i in indices]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_getitems_called_once_per_batch(num_workers):
    ds = BatchedDs(10)
    loader = DataLoader(ds, batch_size=4, num_workers=num_workers)
    assert list(loader) == [[0, 10, 20, 30], [40, 50, 60, 70], [80, 90]]
    assert sorted(ds.calls) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert all(isinstance(call, list) for call in ds.calls)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_getitems_output_goes_through_collate_fn(num_workers):
    loader = DataLoader(BatchedDs(6), batch_size=3, num_workers=num_workers, collate_fn=sum)
    assert list(loader) == [30, 120]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_getitems_receives_sampler_order(num_workers):
    ds = BatchedDs(6)
    loader = DataLoader(ds, batch_size=3, num_workers=num_workers, sampler=[5, 3, 1, 0, 2, 4])
    assert list(loader) == [[50, 30, 10], [0, 20, 40]]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_getitems_may_return_any_iterable(num_workers):
    class TupleDs(BatchedDs):
        def __getitems__(self, indices):
            return tuple(i + 1 for i in indices)

    loader = DataLoader(TupleDs(4), batch_size=2, num_workers=num_workers)
    assert list(loader) == [[1, 2], [3, 4]]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_getitems_wrong_length_raises(num_workers):
    class ShortDs(BatchedDs):
        def __getitems__(self, indices):
            return indices[:-1]

    loader = DataLoader(ShortDs(4), batch_size=2, num_workers=num_workers)
    with pytest.raises(Exception, match="__getitems__ returned 1 samples for 2 indices"):
        next(iter(loader))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_getitems_exception_propagates(num_workers):
    class FailingDs(BatchedDs):
        def __getitems__(self, indices):
            raise RuntimeError("batched read failed")

    loader = DataLoader(FailingDs(4), batch_size=2, num_workers=num_workers)
    with pytest.raises(RuntimeError, match="batched read failed"):
        next(iter(loader))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_datasets_without_getitems_still_use_getitem(num_workers):
    class PlainDs(PyDataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index

    loader = DataLoader(PlainDs(), batch_size=2, num_workers=num_workers)
    assert list(loader) == [[0, 1], [2]]
