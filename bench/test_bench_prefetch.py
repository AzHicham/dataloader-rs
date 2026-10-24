"""Non-regression benchmarks: loading overlapped with the consumer's work.

The consumer takes a training step per batch (a wait that releases the GIL,
like a GPU step), so a loader that prefetches well hides its loading behind
those steps. CpuBoundDs (SHA-256, ~1 ms per item, releases the GIL) with
2 workers produces a batch about as fast as the consumer takes one.

  sequential (num_workers=0): load and step one after the other
  prefetch_depth 1, 2, 4:     loading overlapped with the steps
"""

from __future__ import annotations

import pytest
from common import CpuBoundDs, cat_collate, training_step

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 128
BATCH_SIZE = 8  # 16 batches, ~8 ms each to load on one thread
STEP_S = 0.004  # the consumer's step: as long as 2 workers take per batch


def epoch(loader):
    for _batch in loader:
        training_step(STEP_S)


def test_overlap_sequential(benchmark):
    loader = PyDataloader(CpuBoundDs(N), batch_size=BATCH_SIZE, collate_fn=cat_collate)
    benchmark.pedantic(lambda: epoch(loader), warmup_rounds=1, rounds=5)


@pytest.mark.parametrize("prefetch_depth", [1, 2, 4])
def test_overlap_prefetch_depth(benchmark, prefetch_depth):
    loader = PyDataloader(
        CpuBoundDs(N),
        batch_size=BATCH_SIZE,
        collate_fn=cat_collate,
        num_workers=2,
        prefetch_depth=prefetch_depth,
    )
    benchmark.pedantic(lambda: epoch(loader), warmup_rounds=1, rounds=5)
