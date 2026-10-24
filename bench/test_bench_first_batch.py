"""Non-regression benchmarks: latency to the first batch.

An epoch of exactly one batch of 32 CpuBoundDs items (≈ 0.5 ms each, releases the
GIL), so a round is `iter()` until that batch is ready: what a training loop
waits for before its first step. Calibrated for a 4-vCPU runner.
"""

from __future__ import annotations

import pytest
from common import CpuBoundDs, cat_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

BATCH_SIZE = 32


@pytest.mark.parametrize(
    ("num_workers", "items_in_parallel"), [(0, 1), (1, 1), (0, 4), (1, 4)], ids=str
)
def test_first_batch(benchmark, num_workers, items_in_parallel):
    loader = PyDataloader(
        CpuBoundDs(BATCH_SIZE),
        batch_size=BATCH_SIZE,
        collate_fn=cat_collate,
        num_workers=num_workers,
        items_in_parallel=items_in_parallel,
    )
    benchmark.pedantic(lambda: next(iter(loader)), warmup_rounds=1, rounds=10)
