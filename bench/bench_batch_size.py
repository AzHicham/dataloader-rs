#!/usr/bin/env python3
"""
Benchmark: batch_size sweep.

Tests how throughput scales as batch_size grows using a trivial InMemoryDs
(returns the index, ~0 cost per item). This isolates per-batch overhead
(collation, channel send/recv, Python object creation) from per-item cost.

Two scenarios are measured for each library:
  a) Sequential  (num_workers=0): single-threaded, shows overhead amortization.
  b) Parallel    (num_workers=4, prefetch_depth=16): batches overlap with the
     consumer; larger batches reduce channel pressure.

Expected behaviour:
- Small batch_size (1, 8): per-batch overhead dominates → low items/s.
- Large batch_size (512, 1024, 4096): overhead amortized → items/s plateaus
  near the raw iteration ceiling.

Run:
  uv run --python 3.13t python -X gil=0 bench/bench_batch_size.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    BenchResult,
    InMemoryDs,
    fmt_results,
    run_case,
    sum_collate,
)

from dataloader_rs import PyDataloader

BS_SWEEP = [1, 8, 32, 128, 512, 1024, 4096]

N_ITEMS = 4096
NUM_WORKERS_PAR = 4
PREFETCH_DEPTH_PAR = 16


def bench_ours(warmup: int, repeats: int) -> list[BenchResult]:
    results = []

    # a) Sequential
    for bs in BS_SWEEP:
        loader = PyDataloader(
            InMemoryDs(N_ITEMS),
            batch_size=bs,
            collate_fn=sum_collate,
            num_workers=0,
            prefetch_depth=1,
        )
        results.append(
            run_case(
                name="ours",
                param=str(bs),
                n_items=N_ITEMS,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="batch_size/sequential",
            )
        )

    # b) Parallel
    for bs in BS_SWEEP:
        loader = PyDataloader(
            InMemoryDs(N_ITEMS),
            batch_size=bs,
            collate_fn=sum_collate,
            num_workers=NUM_WORKERS_PAR,
            prefetch_depth=PREFETCH_DEPTH_PAR,
        )
        results.append(
            run_case(
                name="ours",
                param=str(bs),
                n_items=N_ITEMS,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="batch_size/parallel",
            )
        )

    return results


def bench_torch(warmup: int, repeats: int) -> list[BenchResult]:
    try:
        from common import TorchInMemoryDs
        from torch.utils.data import DataLoader  # type: ignore[import]
    except ImportError:
        print("# torch not available — skipping torch benchmarks", flush=True)
        return []

    results = []

    # a) Sequential
    for bs in BS_SWEEP:
        loader = DataLoader(
            TorchInMemoryDs(N_ITEMS),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=sum_collate,
        )
        results.append(
            run_case(
                name="torch",
                param=str(bs),
                n_items=N_ITEMS,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="batch_size/sequential",
            )
        )

    # b) Parallel
    for bs in BS_SWEEP:
        loader = DataLoader(
            TorchInMemoryDs(N_ITEMS),
            batch_size=bs,
            shuffle=False,
            num_workers=NUM_WORKERS_PAR,
            collate_fn=sum_collate,
            prefetch_factor=max(1, PREFETCH_DEPTH_PAR // NUM_WORKERS_PAR),
            persistent_workers=True,
        )
        results.append(
            run_case(
                name="torch",
                param=str(bs),
                n_items=N_ITEMS,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="batch_size/parallel",
            )
        )

    return results


def _print_scenario_table(results: list[BenchResult], group: str, label: str) -> None:
    subset = [r for r in results if r.group == group]
    if not subset:
        return
    libs = list(dict.fromkeys(r.name for r in subset))  # preserve insertion order
    print(f"# {label}  (items/s)")
    header = f"  {'batch_size':<12}"
    for lib in libs:
        header += f"  {lib:>10}"
    print(header)
    for bs in BS_SWEEP:
        row = f"  {bs:<12}"
        for lib in libs:
            match = next((r for r in subset if r.name == lib and r.param == str(bs)), None)
            row += f"  {match.items_per_s:>10.0f}" if match else f"  {'n/a':>10}"
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep batch_size with InMemoryDs to measure per-batch overhead."
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
    _print_scenario_table(results, "batch_size/sequential", "Sequential (num_workers=0)")
    _print_scenario_table(
        results,
        "batch_size/parallel",
        f"Parallel   (num_workers={NUM_WORKERS_PAR}, prefetch_depth={PREFETCH_DEPTH_PAR})",
    )


if __name__ == "__main__":
    main()
