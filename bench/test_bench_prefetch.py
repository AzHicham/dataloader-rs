"""Non-regression benchmarks: prefetch depth impact on pipeline throughput.

Uses CpuBoundDs (SHA-256 per item, ~1 ms, GIL-releasing) with num_workers=4
so batches take real time to produce and the prefetch buffer can actually
saturate.  With InMemoryDs the pipeline never fills up and depth has no
observable effect.  Regressions here point to changes in channel capacity or
batch dispatch.
"""

from __future__ import annotations

import pytest
from common import CpuBoundDs, cat_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 256
BATCH_SIZE = 32
NUM_WORKERS = 4


@pytest.mark.parametrize("prefetch_depth", [1, 4, 16])
def test_prefetch_depth(benchmark, prefetch_depth):
    """Pipeline throughput vs prefetch buffer depth (workers=4, CpuBoundDs)."""
    loader = PyDataloader(
        CpuBoundDs(N),
        batch_size=BATCH_SIZE,
        collate_fn=cat_collate,
        num_workers=NUM_WORKERS,
        prefetch_depth=prefetch_depth,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)
