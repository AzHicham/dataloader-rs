"""Error propagation tests for PyDataloader.

Covers:
  - dataset __getitem__ raising in sequential mode
  - dataset __getitem__ raising in parallel mode
  - collate_fn raising in sequential mode
  - collate_fn raising in parallel mode
  - partial failures (some batches ok, one fails)
  - loader reuse after an epoch that raised errors
"""

import pytest

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import FailingDs, ListDataset

# ── Sequential error propagation ─────────────────────────────────────────────


def test_dataset_error_surfaces_in_iteration():
    """dataset.__getitem__ raising must propagate as RuntimeError from next()."""
    loader = DataLoader(FailingDs(8, fail_index=3), batch_size=4)
    it = iter(loader)
    with pytest.raises(RuntimeError, match="boom at index 3"):
        next(it)


def test_collate_error_surfaces_in_iteration():
    """A collate_fn that raises must propagate as RuntimeError from next()."""

    def bad_collate(_items):
        raise RuntimeError("collate exploded")

    loader = DataLoader(ListDataset(range(8)), batch_size=4, collate_fn=bad_collate)
    it = iter(loader)
    with pytest.raises(RuntimeError, match="collate exploded"):
        next(it)


def test_dataset_error_after_success():
    """fail_index=8, bs=4: first 2 batches succeed, third fails.

    Batches cover indices [0-3], [4-7], [8-11].  The third batch contains
    index 8 so it must raise RuntimeError.
    """
    # First two batches succeed; index 8 falls in the third batch.
    loader = DataLoader(FailingDs(12, fail_index=8), batch_size=4)
    it = iter(loader)

    batch0 = next(it)  # indices 0-3, all ok
    assert batch0 == [0, 1, 2, 3], "first batch must succeed"

    batch1 = next(it)  # indices 4-7, all ok
    assert batch1 == [4, 5, 6, 7], "second batch must succeed"

    with pytest.raises(RuntimeError, match="boom at index 8"):
        next(it)  # indices 8-11, index 8 fails


# ── Parallel error propagation ────────────────────────────────────────────────


def test_dataset_error_parallel():
    """num_workers=4: a dataset error must surface as RuntimeError.

    We cannot guarantee which batch is delivered first in a parallel setting,
    so we iterate all batches and assert that at least one raised.
    """
    loader = DataLoader(FailingDs(16, fail_index=5), batch_size=4, num_workers=4)
    raised = False
    it = iter(loader)
    for _ in range(4):  # 16/4 = 4 batches
        try:
            next(it)
        except RuntimeError:
            raised = True
            break
    assert raised, "a dataset error must surface as RuntimeError in parallel mode"


def test_collate_error_parallel():
    """collate_fn raising in parallel mode must propagate as RuntimeError."""

    def bad_collate(_items):
        raise RuntimeError("parallel collate exploded")

    loader = DataLoader(ListDataset(range(8)), batch_size=4, num_workers=2, collate_fn=bad_collate)
    it = iter(loader)
    with pytest.raises(RuntimeError, match="parallel collate exploded"):
        next(it)


# ── Reuse after error ─────────────────────────────────────────────────────────


def test_error_then_next_epoch_works():
    """After an epoch that raised, the loader must work correctly in the next."""
    fail_index = 3
    loader = DataLoader(FailingDs(8, fail_index=fail_index), batch_size=4)

    # Epoch 1: consume up to the error, ignore it.
    try:
        for _ in loader:
            pass
    except RuntimeError:
        pass

    # Epoch 2: iterate fully — the loader must not be in a broken state.
    # Indices 0-3 contain fail_index=3 so the first batch still fails.
    it = iter(loader)
    with pytest.raises(RuntimeError, match=f"boom at index {fail_index}"):
        next(it)
    # But the second batch (indices 4-7) should still succeed.
    second = next(it)
    assert second == [4, 5, 6, 7]
