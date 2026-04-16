"""Validation / construction tests for PyDataloader.

Tests verify that:
  - invalid arguments are rejected with the appropriate error type and message
  - valid but unusual arguments are accepted without error
  - __len__ on the loader reflects the dataset size and batch settings
  - collate_fn receives the correct argument type
"""

import pytest

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset

# ── Required rejections ───────────────────────────────────────────────────────


def test_rejects_zero_batch_size():
    """batch_size=0 is invalid and must raise ValueError."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        DataLoader(ListDataset([1, 2, 3]), batch_size=0)


def test_rejects_zero_prefetch_depth():
    """prefetch_depth=0 is invalid and must raise ValueError."""
    with pytest.raises(ValueError, match="prefetch_depth must be > 0"):
        DataLoader(ListDataset([1, 2, 3]), prefetch_depth=0)


def test_rejects_shuffle_and_sampler_together():
    """shuffle=True and a custom sampler are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        DataLoader(ListDataset([1, 2, 3]), shuffle=True, sampler=[0, 1, 2])


def test_rejects_generator_without_shuffle():
    """generator= without shuffle=True must raise ValueError."""
    with pytest.raises(ValueError, match="generator requires shuffle=True"):
        DataLoader(ListDataset([1, 2, 3]), generator=42)


def test_rejects_dataset_not_subclassing_pydataset():
    """A dataset that does not subclass PyDataset must raise TypeError."""

    class PlainDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    with pytest.raises(TypeError, match="PyDatasetBase"):
        DataLoader(PlainDataset(), batch_size=1)


# ── Required acceptances ──────────────────────────────────────────────────────


def test_accepts_zero_num_workers():
    """num_workers=0 (sequential mode) must be accepted without error."""
    loader = DataLoader(ListDataset(range(10)), batch_size=3, num_workers=0)
    assert len(loader) == 4  # ceil(10/3)


def test_accepts_large_num_workers():
    """num_workers=16 must be accepted without error."""
    loader = DataLoader(ListDataset(range(10)), batch_size=3, num_workers=16)
    assert len(loader) == 4


def test_accepts_large_prefetch_depth():
    """prefetch_depth=1000 must be accepted without error or hang."""
    # edge case: extremely large prefetch_depth
    loader = DataLoader(ListDataset(range(10)), batch_size=3, prefetch_depth=1000)
    assert len(loader) == 4


def test_batch_size_equals_dataset_len():
    """bs == N is a valid configuration and must not be rejected."""
    n = 7
    loader = DataLoader(ListDataset(range(n)), batch_size=n)
    batches = list(loader)
    assert len(batches) == 1
    assert sorted(batches[0]) == list(range(n))


# ── __len__ correctness ───────────────────────────────────────────────────────


def test_len_matches_drop_last_false():
    """len(loader) == ceil(N / bs) when drop_last=False."""
    loader = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=False)
    assert len(loader) == 4


def test_len_matches_drop_last_true():
    """len(loader) == floor(N / bs) when drop_last=True."""
    loader = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=True)
    assert len(loader) == 3


def test_dataset_len_respected():
    """The loader must honour dataset.__len__ for batch count computation."""

    class SizedDs(ListDataset):
        def __len__(self):
            # Report 6 even though there are 10 items — loader must trust __len__.
            return 6

    loader = DataLoader(SizedDs(range(10)), batch_size=3)
    # ceil(6/3) = 2
    assert len(loader) == 2


def test_accepts_num_workers_argument():
    """Providing num_workers must not raise and must produce the correct length."""
    loader = DataLoader(ListDataset(range(10)), batch_size=3, num_workers=0)
    assert len(loader) == 4


def test_accepts_prefetch_depth_argument():
    """Providing prefetch_depth must not raise and must produce the correct length."""
    loader = DataLoader(ListDataset(range(10)), batch_size=3, prefetch_depth=4)
    assert len(loader) == 4


# ── Collate function interface ────────────────────────────────────────────────


def test_collate_fn_receives_list():
    """collate_fn must receive a plain Python list of items, not some other type."""
    received_type = []

    def inspecting_collate(items):
        received_type.append(type(items))
        return items  # pass-through

    loader = DataLoader(ListDataset(range(4)), batch_size=4, collate_fn=inspecting_collate)
    list(loader)  # trigger one call to collate_fn
    assert received_type, "collate_fn must have been called at least once"
    assert received_type[0] is list, f"collate_fn must receive a list, got {received_type[0]}"
