#!/usr/bin/env python3
"""
Benchmark: real-world workloads, dataloader_rs vs torch.utils.data.DataLoader.

The per-sample work lives in the dataset and is identical for both loaders
(see real_world.py), so the numbers compare the loaders themselves:

  local_files    read a 128 KB file + numpy normalisation      (I/O fan-out)
  image_augment  crop / flip / normalise a decoded 3x256x256    (numpy, GIL-free)
  remote         10 ms simulated network latency per sample    (I/O-bound)
  python_cpu     pure-Python tokenisation, holds the GIL       (torch's home turf)
  python_cpu_local  the same with per-thread tables       (the free-threaded pattern)

Every case times whole epochs including iterator creation, which is what a
training loop pays. torch runs both with fresh workers each epoch (its
default) and with persistent_workers=True.

Run:
  python bench/bench_real_world.py [--workers 4] [--repeats 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))
from common import BenchResult, run_case
from real_world import (
    OursImageAugment,
    OursLocalFiles,
    OursPythonCpu,
    OursPythonCpuLocal,
    OursRemote,
    cpu_count,
    make_file_tree,
    stack_collate,
    torch_datasets,
)

from dataloader_rs import PyDataloader

BATCH_SIZE = 32
# Requests in flight for the network workload's item-level case.
REMOTE_IN_FLIGHT = 64


def ours_dataset(name, n, root):
    return {
        "local_files": lambda: OursLocalFiles(root, n),
        "image_augment": lambda: OursImageAugment(n),
        "remote": lambda: OursRemote(n),
        "python_cpu": lambda: OursPythonCpu(n),
        "python_cpu_local": lambda: OursPythonCpuLocal(n),
    }[name]()


def torch_dataset(name, n, root):
    cls = torch_datasets()[name]
    return cls(root, n) if name == "local_files" else cls(n)


def epoch(loader):
    for _ in loader:
        pass


def bench(args) -> list[BenchResult]:
    results: list[BenchResult] = []
    scale = args.scale
    n_files = int(2048 * scale)
    root = make_file_tree(n_files)
    sizes = {
        "local_files": n_files,
        "image_augment": int(1024 * scale),
        "remote": int(512 * scale),
        "python_cpu": int(2048 * scale),
        "python_cpu_local": int(2048 * scale),
    }
    w = args.workers
    # I/O-bound work benefits from more threads than cores.
    io_workers = 4 * w

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        DataLoader = None
        print("# torch not available — benchmarking dataloader_rs only", flush=True)

    for name in args.workloads:
        n = sizes[name]
        workers = io_workers if name == "remote" else w

        def add(label, make_loader, param=None):
            loader = make_loader()
            results.append(
                run_case(
                    name=label,
                    param=param or f"w={workers if label != 'ours w=0' else 0}",
                    n_items=n,
                    fn=lambda: epoch(loader),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    group=name,
                )
            )
            print(f"  {name:<17} {label:<24} {results[-1].items_per_s:>10.0f} items/s", flush=True)

        add(
            "ours w=0",
            lambda: PyDataloader(
                ours_dataset(name, n, root), batch_size=BATCH_SIZE, collate_fn=stack_collate
            ),
        )
        add(
            "ours",
            lambda: PyDataloader(
                ours_dataset(name, n, root),
                batch_size=BATCH_SIZE,
                num_workers=workers,
                prefetch_depth=2 * workers,
                collate_fn=stack_collate,
            ),
        )
        if name == "remote":
            # Item-level concurrency: REMOTE_IN_FLIGHT requests, spread as
            # `workers` batches x `items` items of each batch.
            items = max(1, REMOTE_IN_FLIGHT // workers)
            add(
                "ours items_in_parallel",
                lambda: PyDataloader(
                    ours_dataset(name, n, root),
                    batch_size=BATCH_SIZE,
                    num_workers=workers,
                    prefetch_depth=2 * workers,
                    items_in_parallel=items,
                    collate_fn=stack_collate,
                ),
                param=f"{workers}x{items}",
            )
        if DataLoader is None:
            continue
        add(
            "torch",
            lambda: DataLoader(
                torch_dataset(name, n, root),
                batch_size=BATCH_SIZE,
                num_workers=workers,
                collate_fn=stack_collate,
            ),
        )
        add(
            "torch persistent",
            lambda: DataLoader(
                torch_dataset(name, n, root),
                batch_size=BATCH_SIZE,
                num_workers=workers,
                persistent_workers=True,
                collate_fn=stack_collate,
            ),
        )
        if name == "remote":
            # The same number of requests in flight for torch: one process each.
            add(
                "torch persistent",
                lambda: DataLoader(
                    torch_dataset(name, n, root),
                    batch_size=BATCH_SIZE,
                    num_workers=REMOTE_IN_FLIGHT,
                    persistent_workers=True,
                    collate_fn=stack_collate,
                ),
                param=f"w={REMOTE_IN_FLIGHT}",
            )
    return results


def print_table(results: list[BenchResult]) -> None:
    print()
    print(f"{'workload':<17} {'loader':<22} {'workers':<8} {'items/s':>10} {'vs torch':>9}")
    print("-" * 68)
    for group in dict.fromkeys(r.group for r in results):
        rows = [r for r in results if r.group == group]
        torch_rate = next((r.items_per_s for r in rows if r.name == "torch"), None)
        for r in rows:
            ratio = f"{r.items_per_s / torch_rate:.2f}x" if torch_rate else ""
            print(f"{group:<17} {r.name:<22} {r.param:<8} {r.items_per_s:>10.0f} {ratio:>9}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--workers", type=int, default=cpu_count(), help="CPU workers (default: cores)"
    )
    parser.add_argument("--warmup", type=int, default=1, help="warm-up epochs per case")
    parser.add_argument("--repeats", type=int, default=5, help="timed epochs per case (median)")
    parser.add_argument("--scale", type=float, default=1.0, help="dataset size multiplier")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["local_files", "image_augment", "remote", "python_cpu", "python_cpu_local"],
        choices=["local_files", "image_augment", "remote", "python_cpu", "python_cpu_local"],
    )
    args = parser.parse_args()
    # More workers than cores is deliberate for the I/O-bound workload.
    warnings.filterwarnings("ignore", message="This DataLoader will create")
    print(f"# cores={cpu_count()} workers={args.workers} batch_size={BATCH_SIZE}", flush=True)
    print_table(bench(args))


if __name__ == "__main__":
    main()
