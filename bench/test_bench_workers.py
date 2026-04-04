"""Non-regression benchmarks: worker-count scaling.

Uses InMemoryDs (O(1) per item) to measure the threading overhead and channel
pipeline efficiency in isolation from dataset I/O cost.  Regressions here
indicate changes in worker spawn, channel, or GIL-management code.
"""

from __future__ import annotations

import pytest
from common import InMemoryDs, sum_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 4_096
BATCH_SIZE = 64
PREFETCH_DEPTH = 16


@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_num_workers(benchmark, num_workers):
    """Throughput vs worker count; 0 = direct path, 1+ = threaded path."""
    loader = PyDataloader(
        InMemoryDs(N),
        batch_size=BATCH_SIZE,
        collate_fn=sum_collate,
        num_workers=num_workers,
        prefetch_depth=PREFETCH_DEPTH,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)
