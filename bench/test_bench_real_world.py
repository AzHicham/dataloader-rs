"""Non-regression benchmarks: real-world workloads (dataloader_rs only).

Small versions of the compare_torch.py workloads; the comparison with
torch.utils.data.DataLoader lives there. Each round drops its batches as it
goes, like a training loop, instead of keeping the epoch in a list.
"""

from __future__ import annotations

import pytest
from real_world import (
    OursImageAugment,
    OursLocalFiles,
    OursRemote,
    make_file_tree,
    stack_collate,
)

from dataloader_rs import PyDataloader

pytestmark = pytest.mark.bench

BATCH_SIZE = 32


def _drain(loader) -> None:
    for _ in loader:
        pass


@pytest.mark.parametrize("num_workers", [0, 4])
def test_local_files(benchmark, num_workers):
    root = make_file_tree(1024)
    loader = PyDataloader(
        OursLocalFiles(root, 1024),
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        prefetch_depth=8,
        collate_fn=stack_collate,
    )
    benchmark.pedantic(lambda: _drain(loader), warmup_rounds=1, rounds=5)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_image_augment(benchmark, num_workers):
    loader = PyDataloader(
        OursImageAugment(512),
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        prefetch_depth=8,
        collate_fn=stack_collate,
    )
    benchmark.pedantic(lambda: _drain(loader), warmup_rounds=1, rounds=5)


def test_remote_threads(benchmark):
    loader = PyDataloader(OursRemote(128), batch_size=BATCH_SIZE, num_workers=4, prefetch_depth=4)
    benchmark.pedantic(lambda: _drain(loader), warmup_rounds=1, rounds=3)


def test_remote_items_in_parallel(benchmark):
    loader = PyDataloader(
        OursRemote(128),
        batch_size=BATCH_SIZE,
        num_workers=4,
        prefetch_depth=4,
        items_in_parallel=8,
    )
    benchmark.pedantic(lambda: _drain(loader), warmup_rounds=1, rounds=3)
