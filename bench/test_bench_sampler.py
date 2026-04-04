"""Non-regression benchmarks: sampler throughput (sequential vs shuffle).

Uses InMemoryDs (O(1) per item) to isolate the sampler's index-generation
cost from dataset I/O.  num_workers=0 so the benchmark measures the direct
path only — worker overhead is covered by test_bench_workers.py.
"""

from __future__ import annotations

import pytest
from common import InMemoryDs, sum_collate

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

N = 10_000
BATCH_SIZE = 64


@pytest.fixture
def sequential_loader():
    return PyDataloader(InMemoryDs(N), batch_size=BATCH_SIZE, collate_fn=sum_collate)


@pytest.fixture
def shuffle_loader():
    return PyDataloader(InMemoryDs(N), batch_size=BATCH_SIZE, collate_fn=sum_collate, shuffle=True)


def test_sampler_sequential(benchmark, sequential_loader):
    """Sequential sampler: baseline throughput on the direct path."""
    benchmark.pedantic(lambda: list(sequential_loader), warmup_rounds=1, rounds=5)


def test_sampler_shuffle(benchmark, shuffle_loader):
    """Shuffle sampler: inside-out Fisher-Yates via fastrand (wyrand)."""
    benchmark.pedantic(lambda: list(shuffle_loader), warmup_rounds=1, rounds=5)
