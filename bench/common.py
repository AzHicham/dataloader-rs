#!/usr/bin/env python3
"""
Shared utilities for dataloader_rs benchmarks.

Provides datasets, collate functions, timing helpers, and result formatting
used by all focused benchmark files in this directory.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

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


class CpuBoundDs(PyDataset):
    """SHA-256 dataset: ~1ms per item, GIL-releasing C extension, minimal allocation.

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
# Torch equivalents (defined lazily to avoid hard dependency)
# ---------------------------------------------------------------------------


def _make_torch_datasets():
    """Return (TorchInMemoryDs, TorchCpuBoundDs) classes using torch.utils.data.Dataset."""
    from torch.utils.data import Dataset  # type: ignore[import]

    class TorchInMemoryDs(Dataset):
        """Trivial torch Dataset returning the index."""

        def __init__(self, n: int):
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, index: int) -> int:
            return index

    class TorchCpuBoundDs(Dataset):
        """SHA-256 torch Dataset: ~1ms per item, GIL-releasing."""

        def __init__(self, n: int):
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, index: int) -> bytes:
            h = hashlib.sha256()
            h.update(index.to_bytes(4, "little"))
            h.update(_PAYLOAD)
            return h.digest()

    return TorchInMemoryDs, TorchCpuBoundDs


# Expose at module level so bench files can do `from common import TorchInMemoryDs`
# without triggering an import error when torch is absent.
try:
    TorchInMemoryDs, TorchCpuBoundDs = _make_torch_datasets()
except ImportError:
    TorchInMemoryDs = None  # type: ignore[assignment,misc]
    TorchCpuBoundDs = None  # type: ignore[assignment,misc]


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
# Result type
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    group: str
    name: str  # e.g. "ours" / "torch"
    param: str  # the swept parameter value, as a string
    us_per_item: float
    items_per_s: float


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_epoch(loader, repeats: int) -> list[float]:
    """Run one full epoch `repeats` times and return a list of wall-clock durations (seconds)."""
    durations: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _batch in loader:
            pass
        durations.append(time.perf_counter() - t0)
    return durations


def run_case(
    name: str,
    param: str,
    n_items: int,
    fn: Callable[[], None],
    warmup: int,
    repeats: int,
    group: str = "",
) -> BenchResult:
    """Time a callable `fn` (one epoch), return a BenchResult.

    Args:
        name:    Library label, e.g. "ours" or "torch".
        param:   The swept parameter value as a string.
        n_items: Total number of items per epoch (for throughput calculation).
        fn:      Zero-argument callable that runs one epoch.
        warmup:  Number of un-timed warm-up calls.
        repeats: Number of timed calls; median is reported.
        group:   Optional group label for the BenchResult.
    """
    for _ in range(warmup):
        fn()
    durations: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    median_s = statistics.median(durations)
    items_per_s = n_items / median_s if median_s > 0 else 0.0
    us_per_item = (median_s / n_items) * 1e6 if n_items > 0 else 0.0
    return BenchResult(
        group=group,
        name=name,
        param=param,
        us_per_item=us_per_item,
        items_per_s=items_per_s,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def fmt_results(results: list[BenchResult]) -> None:
    """Print results as CSV (group,library,param,us_per_item,items_per_s) then a summary table."""
    print("group,library,param,us_per_item,items_per_s")
    for r in results:
        print(f"{r.group},{r.name},{r.param},{r.us_per_item:.2f},{r.items_per_s:.0f}")

    # Human-readable summary
    if not results:
        return
    print()
    print(f"{'group':<30} {'library':<8} {'param':<10} {'us/item':>10} {'items/s':>12}")
    print("-" * 76)
    for r in results:
        print(
            f"{r.group:<30} {r.name:<8} {r.param:<10} {r.us_per_item:>10.2f} {r.items_per_s:>12.0f}"
        )
