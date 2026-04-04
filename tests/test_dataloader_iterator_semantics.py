"""Iterator protocol and lifetime semantics tests.

Tests verify:
  - ExactSizeIterator-style __len__ on the iterator object
  - Loader __len__ before and after drop_last
  - Exhaustion after a full epoch
  - Multiple independent iterators from the same loader
  - __iter__(it) returns the iterator itself (protocol requirement)
"""

import pytest

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset

# ── __len__ on the iterator ───────────────────────────────────────────────────


def test_iter_len_decrements_as_consumed():
    """Iterator __len__ must start at ceil(N/bs) and decrement with each next()."""
    loader = DataLoader(ListDataset(range(7)), batch_size=3, drop_last=False)
    it = iter(loader)
    assert len(it) == 3  # ceil(7/3)
    next(it)
    assert len(it) == 2
    next(it)
    assert len(it) == 1
    next(it)
    assert len(it) == 0


def test_parallel_iter_len_decrements():
    """Same __len__ decrement guarantee must hold in the parallel path."""
    loader = DataLoader(ListDataset(range(8)), batch_size=2, num_workers=2)
    it = iter(loader)
    expected = 4  # 8/2
    assert len(it) == expected
    consumed = 0
    while True:
        try:
            next(it)
            consumed += 1
            assert len(it) == expected - consumed
        except StopIteration:
            break
    assert consumed == expected


# ── loader __len__ ────────────────────────────────────────────────────────────


def test_loader_len_after_drop_last():
    """DataLoader.__len__ must reflect drop_last semantics."""
    loader_keep = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=False)
    assert len(loader_keep) == 4  # ceil(10/3)

    loader_drop = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=True)
    assert len(loader_drop) == 3  # floor(10/3)


# ── Exhaustion ────────────────────────────────────────────────────────────────


def test_iter_is_exhausted_only_once():
    """After a full epoch, calling next() on the iterator must raise StopIteration."""
    loader = DataLoader(ListDataset(range(6)), batch_size=3)
    it = iter(loader)
    next(it)
    next(it)
    with pytest.raises(StopIteration):
        next(it)  # exhausted — must raise, not loop
    # A second StopIteration call must also raise (not crash, not loop).
    with pytest.raises(StopIteration):
        next(it)


# ── Reusability ───────────────────────────────────────────────────────────────


def test_loader_reusable_across_epochs():
    """Calling iter() a second time on the same loader must restart the epoch."""
    loader = DataLoader(ListDataset(range(8)), batch_size=3)
    epoch1 = list(loader)
    epoch2 = list(loader)
    assert epoch1 == epoch2


# ── iter(it) returns self ─────────────────────────────────────────────────────


def test_iter_returns_self():
    """iter(it) must return the iterator object itself (Python iterator protocol)."""
    loader = DataLoader(ListDataset(range(4)), batch_size=2)
    it = iter(loader)
    assert iter(it) is it


# ── Multiple independent iterators ───────────────────────────────────────────


def test_multiple_iters_independent():
    """Two iterators obtained from the same loader represent independent epochs.

    Advancing iter1 must not affect iter2's position, and vice-versa.
    Each iterator must independently cover all N items.
    """
    n = 8
    bs = 2
    loader = DataLoader(ListDataset(range(n)), batch_size=bs)

    it1 = iter(loader)
    it2 = iter(loader)

    # Consume iter1 fully.
    items1 = []
    for batch in it1:
        items1.extend(batch)

    # iter2 should still yield all items independently.
    items2 = []
    for batch in it2:
        items2.extend(batch)

    assert sorted(items1) == list(range(n)), "iter1 must cover all items"
    assert sorted(items2) == list(range(n)), "iter2 must cover all items independently"
