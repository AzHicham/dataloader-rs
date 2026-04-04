"""Safety / stress tests for early drop, GC, and concurrent cancellation.

These tests hammer the shutdown path to ensure there are no hangs, deadlocks,
or use-after-free bugs when iterators are dropped mid-epoch.
"""
import gc
import hashlib

import pytest

from dataloader_rs import PyDataloader as DataLoader, PyDataset
from tests.py_dataloader_test_utils import ListDataset, SlowDs, all_items


# ── Helpers ───────────────────────────────────────────────────────────────────

def _full_epoch(loader):
    """Consume all batches and return sorted flat items."""
    items = []
    for batch in loader:
        items.extend(batch)
    return sorted(items)


# ── Single early-drop then full recovery ──────────────────────────────────────

def test_early_drop_then_full_epoch():
    """Drop after 1 batch; subsequent full epoch yields all items."""
    n = 32
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4, prefetch_depth=8)
    it = iter(loader)
    next(it)
    del it
    assert _full_epoch(loader) == list(range(n))


def test_early_drop_zero_consumed():
    """Drop iterator immediately without consuming any batch."""
    n = 32
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4, prefetch_depth=8)
    it = iter(loader)
    del it
    assert _full_epoch(loader) == list(range(n))


# ── Repeated early drop / recover cycles ─────────────────────────────────────

def test_repeated_early_drop_with_slow_ds():
    """20 epochs: every 3rd epoch drop after first batch; full epoch otherwise.

    Uses SlowDs so workers are actually running (not just queued) when the
    iterator is dropped.  Verifies:
      - no hang on shutdown
      - no data corruption on subsequent full epochs
    """
    n = 32
    ds = SlowDs(n)
    loader = DataLoader(ds, batch_size=4, num_workers=4, prefetch_depth=4)

    expected = list(range(n))

    for i in range(20):
        if i % 3 == 0:
            it = iter(loader)
            next(it)
            del it
        else:
            assert _full_epoch(loader) == expected


def test_zero_consume_drop_20_cycles():
    """20 consecutive zero-consume drops followed by a clean full epoch."""
    n = 32
    ds = SlowDs(n)
    loader = DataLoader(ds, batch_size=4, num_workers=4, prefetch_depth=4)

    for _ in range(20):
        it = iter(loader)
        del it

    assert _full_epoch(loader) == list(range(n))


# ── GC-triggered drop ─────────────────────────────────────────────────────────

def test_gc_drop_does_not_hang():
    """Create an iterator without keeping a reference; gc.collect() triggers drop.

    Simulates the case where Python's cyclic GC finalizes the iterator.
    """
    n = 40
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4, prefetch_depth=8)

    # Create iterator, consume one batch, then release all references.
    it = iter(loader)
    next(it)
    it = None  # noqa: F841
    gc.collect()

    assert _full_epoch(loader) == list(range(n))


def test_gc_drop_zero_consumed():
    """Iterator created but never consumed; GC finalizes it."""
    n = 40
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4, prefetch_depth=8)

    iter(loader)   # no reference kept
    gc.collect()

    assert _full_epoch(loader) == list(range(n))


# ── Ordering invariant under early drop ───────────────────────────────────────

def test_order_preserved_after_partial_epoch():
    """After dropping mid-epoch, the next full epoch must be in sequential order."""
    n = 24
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4)

    # Drop after 2 batches.
    it = iter(loader)
    next(it)
    next(it)
    del it

    # Collect full epoch and verify order.
    batches = list(loader)
    flat = [x for b in batches for x in b]
    assert flat == list(range(n))


# ── Channel-full scenario (prefetch_depth=1) ──────────────────────────────────

def test_early_drop_channel_full_prefetch_1():
    """prefetch_depth=1 means result channel fills immediately.

    Workers will block on send(); dropping the iterator must unblock them
    (by dropping the receiver) and allow join() to complete.
    """
    n = 64
    ds = SlowDs(n)
    loader = DataLoader(ds, batch_size=4, num_workers=4, prefetch_depth=1)

    it = iter(loader)
    next(it)
    del it  # receiver dropped → workers unblock → threads join → no hang

    assert _full_epoch(loader) == list(range(n))


# ── Workers > batches edge case under early drop ──────────────────────────────

def test_early_drop_more_workers_than_batches():
    """More workers than batches; idle workers must still shut down cleanly."""
    n = 8
    loader = DataLoader(ListDataset(range(n)), batch_size=2, num_workers=20, prefetch_depth=8)

    it = iter(loader)
    next(it)
    del it

    assert _full_epoch(loader) == list(range(n))
