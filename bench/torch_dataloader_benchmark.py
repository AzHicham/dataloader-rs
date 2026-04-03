#!/usr/bin/env python3
"""
Torch DataLoader benchmark suite mirroring `python_dataloader_benchmark.py`.

Run:
  uv add torch
  uv run --python 3.13 python bench/torch_dataloader_benchmark.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable


def require_torch():
    try:
        import torch  # noqa: PLC0415
        from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torch is not installed in this environment.\n"
            "Install it first (example): uv add torch"
        ) from exc
    return torch, DataLoader, Dataset


@dataclass
class BenchCase:
    group: str
    name: str
    param: str
    elements: int
    run_once: Callable[[], None]


def time_case(case: BenchCase, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        case.run_once()
    durations: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        case.run_once()
        durations.append(time.perf_counter() - t0)
    median_s = statistics.median(durations)
    items_per_s = case.elements / median_s if median_s > 0 else 0.0
    return median_s, items_per_s


def print_results(rows: Iterable[tuple[str, str, str, float, float]]) -> None:
    print("group,name,param,median_s,items_per_s")
    for group, name, param, median_s, items_per_s in rows:
        print(f"{group},{name},{param},{median_s:.6f},{items_per_s:.2f}")


def build_cases() -> list[BenchCase]:
    torch, DataLoader, Dataset = require_torch()

    class InMemoryDs(Dataset):
        def __init__(self, n: int):
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, index: int) -> int:
            return index

    class HeavyCpuDs(Dataset):
        def __init__(self, n: int):
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, index: int) -> list[int]:
            acc = int(index)
            for i in range(10_000):
                acc = (acc * 6364136223846793005 + i) & 0xFFFFFFFFFFFFFFFF
            return [acc & 0xFF] * 256

    def sum_collate(items: list[int]) -> int:
        return sum(items)

    def cat_collate(items: list[list[int]]) -> list[int]:
        merged: list[int] = []
        for part in items:
            merged.extend(part)
        return merged

    def run_epoch(loader) -> None:
        for batch in loader:
            _ = batch

    cases: list[BenchCase] = []

    # 1) throughput/inmemory
    n = 1_000
    for bs in (32, 128, 512):
        seq_loader = DataLoader(
            InMemoryDs(n),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=sum_collate,
        )
        cases.append(
            BenchCase(
                group="throughput/inmemory",
                name="sequential",
                param=f"bs={bs}",
                elements=n,
                run_once=lambda loader=seq_loader: run_epoch(loader),
            )
        )

        par_loader = DataLoader(
            InMemoryDs(n),
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            collate_fn=sum_collate,
            prefetch_factor=4,
            persistent_workers=True,
        )
        cases.append(
            BenchCase(
                group="throughput/inmemory",
                name="parallel_4w",
                param=f"bs={bs}",
                elements=n,
                run_once=lambda loader=par_loader: run_epoch(loader),
            )
        )

    # 2) throughput/cpu_bound
    n = 128
    bs = 16
    for workers in (0, 1, 2, 4):
        kwargs = {}
        if workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["persistent_workers"] = True
        loader = DataLoader(
            HeavyCpuDs(n),
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            collate_fn=cat_collate,
            **kwargs,
        )
        cases.append(
            BenchCase(
                group="throughput/cpu_bound",
                name="workers",
                param=f"{workers}",
                elements=n,
                run_once=lambda loader=loader: run_epoch(loader),
            )
        )

    # 3) prefetch_depth (Torch equivalent: prefetch_factor)
    n = 128
    bs = 16
    for depth in (1, 2, 4, 8):
        loader = DataLoader(
            HeavyCpuDs(n),
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            collate_fn=cat_collate,
            prefetch_factor=depth,
            persistent_workers=True,
        )
        cases.append(
            BenchCase(
                group="prefetch_depth",
                name="depth",
                param=f"{depth}",
                elements=n,
                run_once=lambda loader=loader: run_epoch(loader),
            )
        )

    # 4) batch_size
    n = 512
    for bs in (1, 8, 32, 128, 512):
        loader = DataLoader(
            InMemoryDs(n),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=sum_collate,
        )
        cases.append(
            BenchCase(
                group="batch_size",
                name="bs",
                param=f"{bs}",
                elements=n,
                run_once=lambda loader=loader: run_epoch(loader),
            )
        )

    # 5) sampler_overhead
    n = 10_000
    bs = 128
    seq_loader = DataLoader(
        InMemoryDs(n),
        batch_size=bs,
        sampler=torch.utils.data.SequentialSampler(range(n)),
        num_workers=0,
        collate_fn=sum_collate,
    )
    cases.append(
        BenchCase(
            group="sampler_overhead",
            name="sequential",
            param="-",
            elements=n,
            run_once=lambda loader=seq_loader: run_epoch(loader),
        )
    )

    gen = torch.Generator().manual_seed(42)
    rnd_loader = DataLoader(
        InMemoryDs(n),
        batch_size=bs,
        sampler=torch.utils.data.RandomSampler(range(n), generator=gen),
        num_workers=0,
        collate_fn=sum_collate,
    )
    cases.append(
        BenchCase(
            group="sampler_overhead",
            name="random",
            param="-",
            elements=n,
            run_once=lambda loader=rnd_loader: run_epoch(loader),
        )
    )

    # 6) early_drop (kept for parity but can be filtered out)
    n = 1_000
    bs = 32
    seq_loader = DataLoader(InMemoryDs(n), batch_size=bs, num_workers=0)
    par_loader = DataLoader(
        InMemoryDs(n),
        batch_size=bs,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )

    def early_drop(loader) -> None:
        it = iter(loader)
        _ = next(it)
        del it

    cases.append(
        BenchCase(
            group="early_drop",
            name="sequential",
            param="-",
            elements=bs,
            run_once=lambda loader=seq_loader: early_drop(loader),
        )
    )
    cases.append(
        BenchCase(
            group="early_drop",
            name="parallel_4w",
            param="-",
            elements=bs,
            run_once=lambda loader=par_loader: early_drop(loader),
        )
    )

    # Ensure random module is referenced to avoid lints if logic changes.
    _ = random
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch DataLoader microbenchmarks")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations per case")
    parser.add_argument("--repeats", type=int, default=5, help="timed iterations per case")
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="substring filter on `group/name` (example: throughput/inmemory)",
    )
    args = parser.parse_args()

    rows = []
    for case in build_cases():
        key = f"{case.group}/{case.name}"
        if args.filter and args.filter not in key:
            continue
        median_s, items_per_s = time_case(case, warmup=args.warmup, repeats=args.repeats)
        rows.append((case.group, case.name, case.param, median_s, items_per_s))

    print_results(rows)


if __name__ == "__main__":
    main()
