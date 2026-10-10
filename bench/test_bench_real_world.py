"""Non-regression benchmarks: real-world workloads (dataloader_rs only).

Small versions of bench_real_world.py; the full comparison against
torch.utils.data.DataLoader lives there.
"""

from __future__ import annotations

import pytest
from real_world import (
    OursAsyncRemote,
    OursImageAugment,
    OursLocalFiles,
    OursRemote,
    make_file_tree,
    stack_collate,
)

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

BATCH_SIZE = 32


@pytest.mark.parametrize("num_workers", [0, 4])
def test_local_files(benchmark, num_workers):
    root = make_file_tree(256)
    loader = PyDataloader(
        OursLocalFiles(root, 256),
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        prefetch_depth=8,
        collate_fn=stack_collate,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_image_augment(benchmark, num_workers):
    loader = PyDataloader(
        OursImageAugment(128),
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        prefetch_depth=8,
        collate_fn=stack_collate,
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=5)


def test_remote_threads(benchmark):
    loader = PyDataloader(OursRemote(128), batch_size=BATCH_SIZE, num_workers=4, prefetch_depth=4)
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=3)


def test_remote_async(benchmark):
    loader = PyDataloader(
        OursAsyncRemote(128), batch_size=BATCH_SIZE, prefetch_depth=4, max_concurrency=64
    )
    benchmark.pedantic(lambda: list(loader), warmup_rounds=1, rounds=3)
