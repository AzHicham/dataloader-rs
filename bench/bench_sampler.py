#!/usr/bin/env python3
"""
Benchmark: sampler type and dataset size sweep on a CPU-bound dataset.

Uses CpuBoundDs (SHA-256 loop, ~0.5 ms/item) with num_workers=4 to reflect
real-world usage: shuffle overhead amortised over actual CPU work.

Compares sequential vs shuffle for ours and torch across N=[1k, 10k, 100k].

Run:
  uv run --python 3.13t python -X gil=0 bench/bench_sampler.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    BenchResult,
    CpuBoundDs,
    TorchCpuBoundDs,
    cat_collate,
    fmt_results,
    run_case,
)

from dataloader_rs import PyDataloader

N_SWEEP = [1_000, 10_000, 100_000]
BATCH_SIZE = 64
NUM_WORKERS = 4
PREFETCH_DEPTH = 16


class RandomIndexSampler:
    """Simple Python shuffle sampler compatible with PyDataloader."""

    def __init__(self, n: int, seed: int = 42):
        self.n = n
        self.rng = random.Random(seed)

    def __iter__(self):
        indices = list(range(self.n))
        self.rng.shuffle(indices)
        return iter(indices)


def bench_ours(warmup: int, repeats: int) -> list[BenchResult]:
    results = []
    for n in N_SWEEP:
        # Sequential
        loader = PyDataloader(
            CpuBoundDs(n),
            batch_size=BATCH_SIZE,
            collate_fn=cat_collate,
            num_workers=NUM_WORKERS,
            prefetch_depth=PREFETCH_DEPTH,
        )
        results.append(
            run_case(
                name="ours",
                param=f"N={n},seq",
                n_items=n,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="sampler/sequential",
            )
        )

        # Shuffle
        loader_shuf = PyDataloader(
            CpuBoundDs(n),
            batch_size=BATCH_SIZE,
            collate_fn=cat_collate,
            sampler=RandomIndexSampler(n),
            num_workers=NUM_WORKERS,
            prefetch_depth=PREFETCH_DEPTH,
        )
        results.append(
            run_case(
                name="ours",
                param=f"N={n},shuf",
                n_items=n,
                fn=lambda l=loader_shuf: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="sampler/shuffle",
            )
        )
    return results


def bench_torch(warmup: int, repeats: int) -> list[BenchResult]:
    try:
        import torch  # type: ignore[import]
        from torch.utils.data import DataLoader  # type: ignore[import]
    except ImportError:
        print("# torch not available — skipping torch benchmarks", flush=True)
        return []

    results = []
    for n in N_SWEEP:
        # Sequential
        loader = DataLoader(
            TorchCpuBoundDs(n),
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SequentialSampler(range(n)),
            num_workers=NUM_WORKERS,
            collate_fn=cat_collate,
            persistent_workers=True,
            prefetch_factor=max(1, PREFETCH_DEPTH // NUM_WORKERS),
        )
        results.append(
            run_case(
                name="torch",
                param=f"N={n},seq",
                n_items=n,
                fn=lambda l=loader: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="sampler/sequential",
            )
        )

        # Shuffle
        gen = torch.Generator().manual_seed(42)
        loader_shuf = DataLoader(
            TorchCpuBoundDs(n),
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.RandomSampler(range(n), generator=gen),
            num_workers=NUM_WORKERS,
            collate_fn=cat_collate,
            persistent_workers=True,
            prefetch_factor=max(1, PREFETCH_DEPTH // NUM_WORKERS),
        )
        results.append(
            run_case(
                name="torch",
                param=f"N={n},shuf",
                n_items=n,
                fn=lambda l=loader_shuf: [_ for _ in l],
                warmup=warmup,
                repeats=repeats,
                group="sampler/shuffle",
            )
        )
    return results


def _print_overhead_table(results: list[BenchResult]) -> None:
    """Show sequential vs shuffle items/s side by side for each N and library."""
    libs = list(dict.fromkeys(r.name for r in results))
    print("# Sequential vs Shuffle (items/s) — larger gap = more shuffle overhead")
    header = f"  {'N':<10}  {'sampler':<10}"
    for lib in libs:
        header += f"  {lib:>10}"
    print(header)
    for n in N_SWEEP:
        for sampler_tag, group in (
            ("sequential", "sampler/sequential"),
            ("shuffle", "sampler/shuffle"),
        ):
            row = f"  {n:<10}  {sampler_tag:<10}"
            param_key = f"N={n},{sampler_tag[:4]}"
            for lib in libs:
                match = next(
                    (
                        r
                        for r in results
                        if r.name == lib and r.group == group and r.param == param_key
                    ),
                    None,
                )
                row += f"  {match.items_per_s:>10.0f}" if match else f"  {'n/a':>10}"
            print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep dataset size N for sequential vs shuffle sampler (num_workers=0)."
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
    _print_overhead_table(results)


if __name__ == "__main__":
    main()
