"""Sequential DataLoader tests (num_workers=0, the default).

These tests exercise every edge-case in the single-threaded code path:
batch count arithmetic, drop_last semantics, ordering guarantees,
multi-epoch reuse, and return types.
"""
import math

import pytest

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset, ToyDataset, all_items


# ── Coverage / correctness ────────────────────────────────────────────────────

def test_seq_all_items_covered():
    """Every index in [0, N) must appear exactly once across all batches."""
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=3)
    assert all_items(loader) == list(range(n))


def test_seq_empty_dataset():
    """N=0 → iterator is immediately exhausted; no batches are yielded."""
    # edge case: empty dataset
    loader = DataLoader(ListDataset([]), batch_size=4)
    batches = list(loader)
    assert batches == [], "empty dataset must produce no batches"


def test_seq_single_item():
    """N=1, bs=1 → exactly one batch containing the single item."""
    loader = DataLoader(ListDataset([42]), batch_size=1)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0] == [42]


def test_seq_bs_equals_n():
    """N=5, bs=5 → exactly one batch containing all 5 items."""
    # edge case: bs == N
    loader = DataLoader(ListDataset(range(5)), batch_size=5)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0] == [0, 1, 2, 3, 4]


def test_seq_bs_larger_than_n_no_drop():
    """N=3, bs=10, drop_last=False → one partial batch with all 3 items."""
    # edge case: bs > N, no drop
    loader = DataLoader(ListDataset(range(3)), batch_size=10, drop_last=False)
    batches = list(loader)
    assert len(batches) == 1, "should get exactly one (partial) batch"
    assert batches[0] == [0, 1, 2]


def test_seq_bs_larger_than_n_drop_last():
    """N=3, bs=10, drop_last=True → 0 batches (the only batch is partial)."""
    # edge case: bs > N with drop_last — nothing to yield
    loader = DataLoader(ListDataset(range(3)), batch_size=10, drop_last=True)
    batches = list(loader)
    assert batches == [], "drop_last=True with bs>N must yield no batches"


def test_seq_batch_size_one():
    """N=5, bs=1 → 5 batches, each of length 1."""
    loader = DataLoader(ListDataset(range(5)), batch_size=1)
    batches = list(loader)
    assert len(batches) == 5
    for i, batch in enumerate(batches):
        assert len(batch) == 1
        assert batch[0] == i


def test_seq_prime_batch_size():
    """N=100, bs=7 → ceil(100/7) = 15 batches."""
    n, bs = 100, 7
    loader = DataLoader(ListDataset(range(n)), batch_size=bs)
    batches = list(loader)
    expected = math.ceil(n / bs)
    assert len(batches) == expected
    # All items must still be present.
    items = sorted(x for b in batches for x in b)
    assert items == list(range(n))


def test_seq_multi_epoch_consistent():
    """The same loader must yield identical batches across 5 epochs."""
    loader = DataLoader(ListDataset(range(12)), batch_size=4)
    first_epoch = list(loader)
    for _ in range(4):
        assert list(loader) == first_epoch, "every epoch must be identical"


def test_seq_items_are_integers():
    """Return type of each item must be int when the dataset returns ints."""
    loader = DataLoader(ListDataset(range(6)), batch_size=3)
    for batch in loader:
        for item in batch:
            assert isinstance(item, int), f"expected int, got {type(item)}"


def test_seq_order_preserved():
    """Sequential sampler → batches arrive in ascending [0,1,2,...] order."""
    n = 9
    loader = DataLoader(ListDataset(range(n)), batch_size=3)
    batches = list(loader)
    flat = [x for b in batches for x in b]
    assert flat == list(range(n)), "sequential sampler must preserve item order"


def test_seq_drop_last_exact_divisor():
    """N=9, bs=3, drop_last=True → 3 full batches (nothing is dropped)."""
    # edge case: N perfectly divisible by bs — drop_last has no effect
    loader = DataLoader(ListDataset(range(9)), batch_size=3, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        assert len(b) == 3


def test_seq_drop_last_partial():
    """N=10, bs=3, drop_last=True → 3 full batches (last 1-item batch dropped)."""
    # floor(10/3) = 3 complete batches; the partial batch of 1 is discarded.
    loader = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        assert len(b) == 3, "every remaining batch must be full"
