"""Non-regression benchmarks: per-batch overhead across batch sizes.

Uses InMemoryDs (O(1) per item) so that per-batch fixed costs (collation,
channel send/recv, Python list construction) dominate.  Tracks both the
direct path (num_workers=0) and the threaded path (num_workers=4), with
per-item ``__getitem__`` and batched ``__getitems__`` fetching.
"""

from __future__ import annotations

import pytest
from common import InMemoryBatchedDs, InMemoryDs, sum_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 4_096
NUM_WORKERS_PAR = 4
PREFETCH_DEPTH_PAR = 16


@pytest.mark.parametrize("batch_size", [1, 64, 4_096])
def test_batch_size_sequential(benchmark, batch_size):
    """Direct path: per-batch overhead amortises as batch_size grows."""
    loader = PyDataloader(InMemoryDs(N), batch_size=batch_size, collate_fn=sum_collate)
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)


@pytest.mark.parametrize("batch_size", [1, 64, 4_096])
def test_batch_size_parallel(benchmark, batch_size):
    """Threaded path: channel pressure drops as batch_size grows."""
    loader = PyDataloader(
        InMemoryDs(N),
        batch_size=batch_size,
        collate_fn=sum_collate,
        num_workers=NUM_WORKERS_PAR,
        prefetch_depth=PREFETCH_DEPTH_PAR,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)


@pytest.mark.parametrize("batch_size", [64, 4_096])
def test_batch_size_sequential_getitems(benchmark, batch_size):
    """Direct path with __getitems__: one Python call per batch."""
    loader = PyDataloader(InMemoryBatchedDs(N), batch_size=batch_size, collate_fn=sum_collate)
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)


@pytest.mark.parametrize("batch_size", [64, 4_096])
def test_batch_size_parallel_getitems(benchmark, batch_size):
    """Threaded path with __getitems__: one Python call per batch."""
    loader = PyDataloader(
        InMemoryBatchedDs(N),
        batch_size=batch_size,
        collate_fn=sum_collate,
        num_workers=NUM_WORKERS_PAR,
        prefetch_depth=PREFETCH_DEPTH_PAR,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)
