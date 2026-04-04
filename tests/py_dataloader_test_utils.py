"""Shared test fixtures and helpers for the PyDataloader test suite.

All test files import from this module so that dataset definitions and
utility functions stay in one place.
"""
import hashlib

from dataloader_rs import PyDataloader as DataLoader, PyDataset


# ── Dataset implementations ───────────────────────────────────────────────────

class ToyDataset(PyDataset):
    """Returns dicts {x: i, y: i*2}. Useful for collation tests."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"x": index, "y": index * 2}


class ListDataset(PyDataset):
    """Wraps a list; returns items by index. General-purpose fixture."""

    def __init__(self, values):
        super().__init__()
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]


class CpuBoundDs(PyDataset):
    """Performs a short SHA-256 computation per item (~0.1 ms) so that
    parallel tests have real CPU work to overlap without using sleep."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # Lightweight but non-trivial CPU work: hash the index 50 times.
        h = hashlib.sha256(str(index).encode())
        for _ in range(50):
            h = hashlib.sha256(h.digest())
        return index


class SlowDs(PyDataset):
    """Performs enough SHA-256 work to make inter-item overlap detectable.

    NOTE: We use CPU work rather than time.sleep() to keep tests fast and
    avoid flakiness from scheduler jitter.  The sha256 loop is calibrated
    to take ~0.5 ms so concurrency is detectable without being slow.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        h = hashlib.sha256(str(index).encode())
        for _ in range(200):
            h = hashlib.sha256(h.digest())
        return index


class FailingDs(PyDataset):
    """Raises RuntimeError exactly at `fail_index`; succeeds everywhere else."""

    def __init__(self, n: int, fail_index: int):
        super().__init__()
        self.n = n
        self.fail_index = fail_index

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if index == self.fail_index:
            raise RuntimeError(f"boom at index {index}")
        return index


class CountingDs(PyDataset):
    """Thread-safe call counter: appending to a list is atomic in CPython.

    Use `ds.calls` to inspect the list of indices that were accessed.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.calls: list = []  # list.append is GIL-atomic in CPython

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self.calls.append(index)
        return index


# ── Helpers ───────────────────────────────────────────────────────────────────

def materialize(loader: DataLoader):
    """Return list(loader). Collects all batches from one epoch."""
    return list(loader)


def all_items(loader: DataLoader) -> list:
    """Flatten and sort all items across all batches in one epoch.

    Useful for coverage assertions that don't care about batch order.
    """
    items = []
    for batch in loader:
        items.extend(batch)
    items.sort()
    return items
