"""Non-regression benchmarks: prefetch depth impact on pipeline throughput.

Uses InMemoryDs with num_workers=4 so the channel is the bottleneck at
depth=1 and pipeline saturation occurs at larger depths.  Regressions here
point to changes in channel capacity, waker logic, or batch dispatch.
"""

from __future__ import annotations

import pytest
from common import InMemoryDs, sum_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 4_096
BATCH_SIZE = 64
NUM_WORKERS = 4


@pytest.mark.parametrize("prefetch_depth", [1, 4, 16])
def test_prefetch_depth(benchmark, prefetch_depth):
    """Pipeline throughput vs prefetch buffer depth (workers=4, InMemoryDs)."""
    loader = PyDataloader(
        InMemoryDs(N),
        batch_size=BATCH_SIZE,
        collate_fn=sum_collate,
        num_workers=NUM_WORKERS,
        prefetch_depth=prefetch_depth,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)
