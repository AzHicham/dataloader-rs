#!/usr/bin/env python3
"""
Benchmark: prefetch_depth sweep.

Tests how throughput changes as the prefetch buffer depth grows, using a
CPU-bound dataset (SHA-256 ~1ms/item) with a fixed 4 workers.

Expected behaviour:
- depth=1: consumer blocks waiting for the next batch — workers are starved.
- depth=2..8: throughput grows as the pipeline fills; workers stay busy.
- depth >= 2×workers (=8): plateau — prefetch is deep enough to never stall.

Note on torch prefetch_factor:
  torch uses a *per-worker* prefetch_factor, so the total in-flight batches is
  prefetch_factor × num_workers. The table shows the torch factor directly; the
  effective total is noted in the summary.

Run:
  uv run --python 3.13t python -X gil=0 bench/bench_prefetch_depth.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    BenchResult,
    CpuBoundDs,
    cat_collate,
    fmt_results,
    run_case,
)

from dataloader_rs import PyDataloader

DEPTH_SWEEP = [1, 2, 4, 8, 16, 32]
TORCH_FACTOR_SWEEP = [1, 2, 4, 8, 16]  # per-worker; total = factor × 4

N_ITEMS = 128
BATCH_SIZE = 16  # 8 batches per epoch
NUM_WORKERS = 4  # fixed — workers can outpace consumer, so depth matters


def bench_ours(warmup: int, repeats: int) -> list[BenchResult]:
    results = []
    for depth in DEPTH_SWEEP:
        loader = PyDataloader(
            CpuBoundDs(N_ITEMS),
            batch_size=BATCH_SIZE,
            collate_fn=cat_collate,
            num_workers=NUM_WORKERS,
            prefetch_depth=depth,
        )
        results.append(
            run_case(
                name="ours",
                param=str(depth),
                n_items=N_ITEMS,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="prefetch_depth",
            )
        )
    return results


def bench_torch(warmup: int, repeats: int) -> list[BenchResult]:
    try:
        from common import TorchCpuBoundDs
        from torch.utils.data import DataLoader  # type: ignore[import]
    except ImportError:
        print("# torch not available — skipping torch benchmarks", flush=True)
        return []

    results = []
    for factor in TORCH_FACTOR_SWEEP:
        loader = DataLoader(
            TorchCpuBoundDs(N_ITEMS),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=cat_collate,
            prefetch_factor=factor,
            persistent_workers=True,
        )
        results.append(
            run_case(
                name="torch",
                # Show factor as param; total in-flight = factor × NUM_WORKERS
                param=str(factor),
                n_items=N_ITEMS,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="prefetch_depth",
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep prefetch_depth (ours) / prefetch_factor (torch) with 4 workers."
    )
    parser.add_argument("--warmup", type=int, default=2, help="warmup epochs per case (default: 2)")
    parser.add_argument(
        "--repeats", type=int, default=10, help="timed epochs per case (default: 10)"
    )
    args = parser.parse_args()

    results: list[BenchResult] = []
    results.extend(bench_ours(args.warmup, args.repeats))
    results.extend(bench_torch(args.warmup, args.repeats))

    fmt_results(results)

    print()
    print("# Note: torch param is per-worker prefetch_factor; total in-flight = factor × 4 workers")
    print()

    # Side-by-side: ours vs torch across their own sweep ranges
    ours_res = [r for r in results if r.name == "ours"]
    torch_res = [r for r in results if r.name == "torch"]

    if ours_res:
        print("# dataloader_rs  prefetch_depth → items/s")
        print(f"  {'depth':<8}", end="")
        for r in ours_res:
            print(f"  {r.param:>6}", end="")
        print()
        print(f"  {'items/s':<8}", end="")
        for r in ours_res:
            print(f"  {r.items_per_s:>6.0f}", end="")
        print()

    if torch_res:
        print()
        print("# torch  prefetch_factor (per-worker) → items/s")
        print(f"  {'factor':<8}", end="")
        for r in torch_res:
            print(f"  {r.param:>6}", end="")
        print()
        print(f"  {'items/s':<8}", end="")
        for r in torch_res:
            print(f"  {r.items_per_s:>6.0f}", end="")
        print()


if __name__ == "__main__":
    main()
