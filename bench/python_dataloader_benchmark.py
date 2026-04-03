#!/usr/bin/env python3
"""
Python benchmark suite aligned with `benches/dataloader.rs`.

Run:
  maturin develop --uv -m ./Cargo.toml
  uv run --python 3.13 python bench/python_dataloader_benchmark.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from dataloader_rs import PyDataloader, PyDataset, bench_dataset_get_dispatch


class InMemoryDs(PyDataset):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


class LightIoDs(PyDataset):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        time.sleep(50e-6)
        return index


class HeavyCpuDs(PyDataset):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> list[int]:
        acc = int(index)
        for i in range(10_000):
            acc = (acc * 6364136223846793005 + i) & 0xFFFFFFFFFFFFFFFF
        return [acc & 0xFF] * 256


class RandomIndexSampler:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.rng = random.Random(seed)

    def __iter__(self):
        indices = list(range(self.n))
        self.rng.shuffle(indices)
        return iter(indices)


def sum_collate(items: list[int]) -> int:
    return sum(items)


def cat_collate(items: list[list[int]]) -> list[int]:
    merged: list[int] = []
    for part in items:
        merged.extend(part)
    return merged


@dataclass
class BenchCase:
    group: str
    name: str
    param: str
    elements: int
    run_once: Callable[[], None]


def run_epoch(loader: PyDataloader) -> None:
    for batch in loader:
        _ = batch


def pure_python_get_dispatch(dataset: PyDataset, iters: int) -> float:
    if iters <= 0:
        raise ValueError("iters must be > 0")
    n = len(dataset)
    if n <= 0:
        raise ValueError("dataset length must be > 0")
    getitem = dataset.__getitem__
    start = time.perf_counter()
    for i in range(iters):
        _ = getitem(i % n)
    return time.perf_counter() - start


def time_case(case: BenchCase, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        case.run_once()
    durations: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        case.run_once()
        durations.append(time.perf_counter() - t0)
    median_s = statistics.median(durations)
    items_per_s = case.elements / median_s if median_s > 0 else 0.0
    ns_per_iter = (median_s / case.elements) * 1e9 if case.elements > 0 else 0.0
    return median_s, items_per_s, ns_per_iter


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    # 0) dataset_get_dispatch (Rust->Python call boundary only)
    n = 1_000
    ds_dispatch = InMemoryDs(n)
    dispatch_iters = 200_000
    cases.append(
        BenchCase(
            group="dataset_dispatch",
            name="rust_to_py_getitem_bridge",
            param=f"iters={dispatch_iters}",
            elements=dispatch_iters,
            run_once=lambda ds=ds_dispatch: bench_dataset_get_dispatch(ds, dispatch_iters),
        )
    )
    cases.append(
        BenchCase(
            group="dataset_dispatch",
            name="pure_python_getitem_call",
            param=f"iters={dispatch_iters}",
            elements=dispatch_iters,
            run_once=lambda ds=ds_dispatch: pure_python_get_dispatch(ds, dispatch_iters),
        )
    )

    # 1) throughput/inmemory
    n = 1_000
    for bs in (32, 128, 512):
        seq_loader = PyDataloader(
            InMemoryDs(n),
            batch_size=bs,
            collate_fn=sum_collate,
            num_workers=0,
            prefetch_depth=4,
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
        par_loader = PyDataloader(
            InMemoryDs(n),
            batch_size=bs,
            collate_fn=sum_collate,
            num_workers=4,
            prefetch_depth=4,
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
        loader = PyDataloader(
            HeavyCpuDs(n),
            batch_size=bs,
            collate_fn=cat_collate,
            num_workers=workers,
            prefetch_depth=4,
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

    # 3) prefetch_depth (CPU-bound workload)
    n = 128
    bs = 16
    for depth in (1, 2, 4, 8):
        loader = PyDataloader(
            HeavyCpuDs(n),
            batch_size=bs,
            collate_fn=cat_collate,
            num_workers=4,
            prefetch_depth=depth,
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
        loader = PyDataloader(
            InMemoryDs(n),
            batch_size=bs,
            collate_fn=sum_collate,
            num_workers=0,
            prefetch_depth=4,
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
    seq_loader = PyDataloader(
        InMemoryDs(n),
        batch_size=bs,
        collate_fn=sum_collate,
        num_workers=0,
        prefetch_depth=4,
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
    rnd_loader = PyDataloader(
        InMemoryDs(n),
        batch_size=bs,
        sampler=RandomIndexSampler(n, seed=42),
        collate_fn=sum_collate,
        num_workers=0,
        prefetch_depth=4,
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

    # 6) early_drop
    n = 1_000
    bs = 32

    def early_drop(loader: PyDataloader) -> None:
        it = iter(loader)
        _ = next(it)
        del it

    seq_loader = PyDataloader(InMemoryDs(n), batch_size=bs, num_workers=0, prefetch_depth=4)
    cases.append(
        BenchCase(
            group="early_drop",
            name="sequential",
            param="-",
            elements=bs,
            run_once=lambda loader=seq_loader: early_drop(loader),
        )
    )
    par_loader = PyDataloader(InMemoryDs(n), batch_size=bs, num_workers=4, prefetch_depth=4)
    cases.append(
        BenchCase(
            group="early_drop",
            name="parallel_4w",
            param="-",
            elements=bs,
            run_once=lambda loader=par_loader: early_drop(loader),
        )
    )

    return cases


def print_results(rows: Iterable[tuple[str, str, str, float, float, float]]) -> None:
    print("group,name,param,ns_per_iter,median_s,items_per_s")
    for group, name, param, ns_per_iter, median_s, items_per_s in rows:
        print(f"{group},{name},{param},{ns_per_iter:.2f},{median_s:.6f},{items_per_s:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Python DataLoader microbenchmarks")
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
        median_s, items_per_s, ns_per_iter = time_case(case, warmup=args.warmup, repeats=args.repeats)
        rows.append((case.group, case.name, case.param, ns_per_iter, median_s, items_per_s))

    print_results(rows)


if __name__ == "__main__":
    main()
