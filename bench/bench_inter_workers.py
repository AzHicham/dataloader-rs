#!/usr/bin/env python3
"""
Benchmark: inter-batch worker parallelism (num_workers sweep).

Tests how throughput scales as more workers are added using a CPU-bound dataset
(SHA-256 ~1ms/item). With the GIL released inside hashlib, workers run truly in
parallel under Python 3.13t (free-threaded) or via OS threads otherwise.

Expected behaviour:
- num_workers=0: serial, bottlenecked by single-threaded GIL.
- num_workers=1..N: throughput grows roughly linearly up to the number of
  physical cores, then plateaus.
- torch with persistent_workers=True avoids per-epoch process spawn overhead.

Run:
  uv run --python 3.13t python -X gil=0 bench/bench_inter_workers.py
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CpuBoundDs,
    cat_collate,
    run_case,
    fmt_results,
    BenchResult,
)

from dataloader_rs import PyDataloader

WORKER_SWEEP = [0, 1, 2, 4, 8]
BS_SWEEP = [1, 4, 16, 64]

N_ITEMS = 256
PREFETCH_DEPTH = 16     # 2× max workers — never starves the workers


def bench_ours(warmup: int, repeats: int) -> list[BenchResult]:
    results = []
    for bs in BS_SWEEP:
        for nw in WORKER_SWEEP:
            loader = PyDataloader(
                CpuBoundDs(N_ITEMS),
                batch_size=bs,
                collate_fn=cat_collate,
                num_workers=nw,
                prefetch_depth=PREFETCH_DEPTH,
            )
            results.append(
                run_case(
                    name="ours",
                    param=f"bs={bs},w={nw}",
                    n_items=N_ITEMS,
                    fn=lambda l=loader: [_ for _ in l],
                    warmup=warmup,
                    repeats=repeats,
                    group=f"inter_workers/bs={bs}",
                )
            )
    return results


def bench_torch(warmup: int, repeats: int) -> list[BenchResult]:
    try:
        from torch.utils.data import DataLoader  # type: ignore[import]
        from common import TorchCpuBoundDs
    except ImportError:
        print("# torch not available — skipping torch benchmarks", flush=True)
        return []

    results = []
    for bs in BS_SWEEP:
        for nw in WORKER_SWEEP:
            kwargs: dict = {}
            if nw > 0:
                kwargs["persistent_workers"] = True
                kwargs["prefetch_factor"] = max(1, PREFETCH_DEPTH // max(nw, 1))
            loader = DataLoader(
                TorchCpuBoundDs(N_ITEMS),
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                collate_fn=cat_collate,
                **kwargs,
            )
            results.append(
                run_case(
                    name="torch",
                    param=f"bs={bs},w={nw}",
                    n_items=N_ITEMS,
                    fn=lambda l=loader: [_ for _ in l],
                    warmup=warmup,
                    repeats=repeats,
                    group=f"inter_workers/bs={bs}",
                )
            )
    return results


def _print_bs_table(results: list[BenchResult], bs: int) -> None:
    group = f"inter_workers/bs={bs}"
    print(f"# batch_size={bs}  (items/s)")
    print(f"  {'num_workers':<14}", end="")
    for nw in WORKER_SWEEP:
        print(f"  {nw:>7}", end="")
    print()
    for lib in ("ours", "torch"):
        subset = [r for r in results if r.name == lib and r.group == group]
        if not subset:
            continue
        print(f"  {lib:<14}", end="")
        for nw in WORKER_SWEEP:
            match = next((r for r in subset if r.param == f"bs={bs},w={nw}"), None)
            print(f"  {match.items_per_s:>7.0f}" if match else f"  {'n/a':>7}", end="")
        print()
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep num_workers × batch_size for CPU-bound dataset."
    )
    parser.add_argument("--warmup", type=int, default=2, help="warmup epochs per case (default: 2)")
    parser.add_argument("--repeats", type=int, default=10, help="timed epochs per case (default: 10)")
    args = parser.parse_args()

    results: list[BenchResult] = []
    results.extend(bench_ours(args.warmup, args.repeats))
    results.extend(bench_torch(args.warmup, args.repeats))

    fmt_results(results)

    print()
    for bs in BS_SWEEP:
        _print_bs_table(results, bs)


if __name__ == "__main__":
    main()
