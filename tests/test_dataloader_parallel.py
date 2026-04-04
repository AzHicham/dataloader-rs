"""Parallel DataLoader tests (num_workers > 0).

Tests verify:
  - item coverage (same items as sequential path)
  - drop_last semantics in the parallel path
  - loader reuse across epochs
  - safe early-drop / iterator cancellation
  - custom collate_fn applied correctly in parallel mode
  - large prefetch_depth does not corrupt results
"""

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import (
    ListDataset,
    all_items,
)

# ── Coverage / correctness ────────────────────────────────────────────────────


def test_par_all_items_covered_2w():
    """N=20, bs=4, num_workers=2 → all 20 items present, none duplicated."""
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=2)
    assert all_items(loader) == list(range(n))


def test_par_all_items_covered_4w():
    """N=20, bs=4, num_workers=4 → all 20 items present, none duplicated."""
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4)
    assert all_items(loader) == list(range(n))


def test_par_order_matches_sequential():
    """Sequential sampler (default) → parallel output order matches sequential.

    When ordering is deterministic (no shuffle), the parallel path must
    deliver batches in the same order as the sequential path.
    """
    n = 16
    bs = 4
    seq = DataLoader(ListDataset(range(n)), batch_size=bs, num_workers=0)
    par = DataLoader(ListDataset(range(n)), batch_size=bs, num_workers=4)

    seq_batches = list(seq)
    par_batches = list(par)
    assert seq_batches == par_batches, "parallel order must match sequential for sequential sampler"


def test_par_more_workers_than_batches():
    """N=8, bs=2 → 4 batches; num_workers=20 (most workers sit idle).

    edge case: num_workers > number of batches — must still yield all items.
    """
    n = 8
    loader = DataLoader(ListDataset(range(n)), batch_size=2, num_workers=20)
    assert all_items(loader) == list(range(n))


def test_par_empty_dataset():
    """N=0, num_workers=4 → 0 batches, no hang."""
    # edge case: empty dataset with parallel workers
    loader = DataLoader(ListDataset([]), batch_size=4, num_workers=4)
    assert list(loader) == []


def test_par_single_batch():
    """N=4, bs=4, num_workers=4 → exactly 1 batch with all 4 items."""
    loader = DataLoader(ListDataset(range(4)), batch_size=4, num_workers=4)
    batches = list(loader)
    assert len(batches) == 1
    assert sorted(batches[0]) == [0, 1, 2, 3]


def test_par_drop_last():
    """N=11, bs=4, num_workers=4, drop_last=True → floor(11/4)=2 full batches."""
    loader = DataLoader(ListDataset(range(11)), batch_size=4, num_workers=4, drop_last=True)
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert len(b) == 4, "every batch must be full when drop_last=True"


def test_par_multi_epoch_consistent():
    """A parallel loader used over 5 epochs must cover the same items each time."""
    n = 16
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=2)
    first = all_items(loader)
    for _ in range(4):
        assert all_items(loader) == first, "items must be identical across epochs"


# ── Cancellation / early drop ─────────────────────────────────────────────────


def test_par_early_drop_no_hang():
    """Consume 1 batch then discard the iterator; the next full epoch must work.

    If there is a deadlock in the worker shutdown path this test hangs — a
    clear signal of the bug.
    """
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=2)

    # Consume just one batch.
    it = iter(loader)
    next(it)
    del it  # drop iterator with workers potentially still running

    # A new full epoch must complete correctly.
    assert all_items(loader) == list(range(n))


def test_par_early_drop_zero_consumed():
    """Discard the iterator immediately without consuming anything.

    edge case: zero batches consumed before drop.
    """
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4)

    it = iter(loader)
    del it  # dropped before any next() calls

    assert all_items(loader) == list(range(n))


# ── Collate function ──────────────────────────────────────────────────────────


def test_par_with_collate_fn():
    """Custom collate_fn must be applied correctly in the parallel path."""

    def collate_fn(items):
        return sum(items)  # reduce the batch to a single integer

    n = 12
    bs = 4
    loader = DataLoader(ListDataset(range(n)), batch_size=bs, num_workers=2, collate_fn=collate_fn)
    sums = list(loader)
    # batch 0: 0+1+2+3=6, batch 1: 4+5+6+7=22, batch 2: 8+9+10+11=38
    assert sums == [6, 22, 38]


# ── Prefetch depth ────────────────────────────────────────────────────────────


def test_par_large_prefetch():
    """prefetch_depth=100 >> number of batches; results must still be correct."""
    # edge case: extremely large prefetch_depth
    n = 8
    loader = DataLoader(ListDataset(range(n)), batch_size=2, num_workers=2, prefetch_depth=100)
    assert all_items(loader) == list(range(n))
