"""Non-regression benchmarks: worker-count scaling.

Uses CpuBoundDs (SHA-256 per item, ~1 ms, GIL-releasing) so that parallelism
is actually measurable.  InMemoryDs would make workers appear *slower* than
the direct path because thread overhead > dataset cost.  Regressions here
indicate changes in worker spawn, channel, or GIL-management code.
"""

from __future__ import annotations

import pytest
from common import CpuBoundDs, cat_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 256
BATCH_SIZE = 32
PREFETCH_DEPTH = 8


@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_num_workers(benchmark, num_workers):
    """Throughput vs worker count; 0 = direct path, 1+ = threaded path."""
    loader = PyDataloader(
        CpuBoundDs(N),
        batch_size=BATCH_SIZE,
        collate_fn=cat_collate,
        num_workers=num_workers,
        prefetch_depth=PREFETCH_DEPTH,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)
