#!/usr/bin/env python3
"""Shared datasets, collate functions and consumer work for the pytest-benchmark guards.

The comparison with PyTorch (compare_torch.py) has its own workloads in real_world.py.
"""

from __future__ import annotations

import hashlib
import time

from dataloader_rs import PyDataset

# 1MB payload — pre-allocated once at import time.
# __getitem__ never allocates a large buffer; only the 32-byte digest is created.
_PAYLOAD = bytes(range(256)) * 4096  # 1 048 576 bytes


# ---------------------------------------------------------------------------
# dataloader_rs datasets
# ---------------------------------------------------------------------------


class InMemoryDs(PyDataset):
    """Trivial dataset returning the index. Isolates loader/sampler overhead."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


class InMemoryBatchedDs(InMemoryDs):
    """InMemoryDs with ``__getitems__``: one Python call per batch, not per item."""

    def __getitems__(self, indices: list[int]) -> list[int]:
        return indices


class CpuBoundDs(PyDataset):
    """SHA-256 dataset: ≈ 0.5 ms per item, GIL-releasing C extension, minimal allocation.

    Each call hashes 1MB of data via hashlib (C extension that releases the GIL)
    and returns only a 32-byte digest — ideal for parallelism benchmarks.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> bytes:
        h = hashlib.sha256()
        h.update(index.to_bytes(4, "little"))
        h.update(_PAYLOAD)
        return h.digest()  # 32 bytes


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def sum_collate(items: list[int]) -> int:
    """Collate a list of ints by summing them."""
    return sum(items)


def cat_collate(items: list[bytes]) -> bytes:
    """Collate a list of byte strings by concatenating them."""
    return b"".join(items)


# ---------------------------------------------------------------------------
# Consumer work
# ---------------------------------------------------------------------------


def training_step(seconds: float) -> None:
    """Stand-in for a training step: waits without holding the GIL, like a GPU step."""
    time.sleep(seconds)
