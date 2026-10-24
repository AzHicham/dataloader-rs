#!/usr/bin/env python3
"""
Compare dataloader_rs with torch.utils.data.DataLoader on the same workloads.

Each workload's per-sample work lives in real_world.py and is shared by both
loaders, so only the loading differs. Each loader collates the way it is meant
to be used: dataloader_rs with `np.stack`, torch with its default collation
for array samples (a tensor built in shared memory by its workers, so a batch
is not pickled between processes); other samples stay a list in both. For
every loader configuration this measures, over whole epochs:

  throughput      items per second, iterator creation included   (median)
  first batch     time from iter(loader) to the first batch       (median)

Workloads:
  in_memory          returns the index                      (loader overhead only)
  local_files        read a 128 KB file + normalise         (I/O fan-out)
  image_augment      crop / flip / normalise an image       (NumPy, releases the GIL)
  remote             10 ms blocking network round trip      (latency-bound)
  python_cpu         pure-Python tokenisation, shared vocab (GIL-bound; contended on 3.14t)
  python_cpu_local   the same with per-thread tables        (the free-threaded pattern)

Defaults are calibrated for a 4-vCPU machine (a GitHub-hosted runner); pin a
bigger machine to 4 cores to compare (`taskset -c 0,2,4,6 ...`).

Run:
  python bench/compare_torch.py [--workers 4] [--repeats 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(__file__))
from real_world import (
    OursImageAugment,
    OursInMemory,
    OursLocalFiles,
    OursPythonCpu,
    OursPythonCpuLocal,
    OursRemote,
    make_file_tree,
    stack_collate,
    torch_datasets,
)

from dataloader_rs import PyDataloader

BATCH_SIZE = 32
# Requests in flight for the latency-bound workload's item-level and torch cases.
REMOTE_IN_FLIGHT = 64
# Workloads with array samples, which torch collates itself (see the docstring).
ARRAY_WORKLOADS = {"local_files", "image_augment"}
WORKLOADS = [
    "in_memory",
    "local_files",
    "image_augment",
    "remote",
    "python_cpu",
    "python_cpu_local",
]
SIZES = {
    "in_memory": 16384,
    "local_files": 2048,
    "image_augment": 1024,
    "remote": 512,
    "python_cpu": 2048,
    "python_cpu_local": 2048,
}


@dataclass
class Result:
    workload: str
    loader: str  # "dataloader_rs" or "torch"
    config: str
    items_per_s: float
    first_batch_ms: float


def ours_dataset(name, n, root):
    return {
        "in_memory": lambda: OursInMemory(n),
        "local_files": lambda: OursLocalFiles(root, n),
        "image_augment": lambda: OursImageAugment(n),
        "remote": lambda: OursRemote(n),
        "python_cpu": lambda: OursPythonCpu(n),
        "python_cpu_local": lambda: OursPythonCpuLocal(n),
    }[name]()


def torch_dataset(name, n, root):
    cls = torch_datasets()[name]
    return cls(root, n) if name == "local_files" else cls(n)


def measure(loader, n_items: int, warmup: int, repeats: int) -> tuple[float, float]:
    """Median items/s and median first-batch latency (ms) over `repeats` epochs."""
    durations, latencies = [], []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        it = iter(loader)
        next(it)
        t1 = time.perf_counter()
        for _ in it:
            pass
        t2 = time.perf_counter()
        if i >= warmup:
            latencies.append(t1 - t0)
            durations.append(t2 - t0)
    return n_items / statistics.median(durations), statistics.median(latencies) * 1e3


def configs(name: str, n: int, root, workers: int, torch_loader):
    """(loader, config label, factory) for every configuration of a workload."""
    w = workers

    def ours(**kw):
        return lambda: PyDataloader(
            ours_dataset(name, n, root), batch_size=BATCH_SIZE, collate_fn=stack_collate, **kw
        )

    yield "dataloader_rs", "w=0", ours()
    yield "dataloader_rs", f"w={w}", ours(num_workers=w)
    if name == "remote":
        k = max(1, REMOTE_IN_FLIGHT // w)
        yield "dataloader_rs", f"w={w} items={k}", ours(num_workers=w, items_in_parallel=k)
    if torch_loader is None:
        return

    torch_collate = None if name in ARRAY_WORKLOADS else stack_collate

    def torch(**kw):
        return lambda: torch_loader(
            torch_dataset(name, n, root), batch_size=BATCH_SIZE, collate_fn=torch_collate, **kw
        )

    yield "torch", "w=0", torch()
    yield "torch", f"w={w}", torch(num_workers=w)
    yield "torch", f"w={w} persistent", torch(num_workers=w, persistent_workers=True)
    if name == "remote":
        # The same number of requests in flight for torch: one process each.
        yield (
            "torch",
            f"w={REMOTE_IN_FLIGHT} persistent",
            torch(num_workers=REMOTE_IN_FLIGHT, persistent_workers=True),
        )


def run(args) -> list[Result]:
    try:
        from torch.utils.data import DataLoader as TorchDataLoader
    except ImportError:
        TorchDataLoader = None
        print("# torch not available: dataloader_rs only", flush=True)

    root = make_file_tree(SIZES["local_files"])
    results: list[Result] = []
    for name in args.workloads:
        n = SIZES[name]
        for loader, config, factory in configs(name, n, root, args.workers, TorchDataLoader):
            items_per_s, first_ms = measure(factory(), n, args.warmup, args.repeats)
            results.append(Result(name, loader, config, items_per_s, first_ms))
            print(
                f"  {name:<17} {loader:<14} {config:<18} {items_per_s:>10.0f} items/s"
                f"  first batch {first_ms:>8.2f} ms",
                flush=True,
            )
    return results


def print_table(results: list[Result]) -> None:
    print()
    header = f"{'workload':<17} {'loader':<14} {'config':<18} {'items/s':>10} {'first ms':>9}"
    print(header + f" {'vs best torch':>14}")
    print("-" * (len(header) + 15))
    for workload in dict.fromkeys(r.workload for r in results):
        rows = [r for r in results if r.workload == workload]
        best_torch = max((r.items_per_s for r in rows if r.loader == "torch"), default=None)
        for r in rows:
            ratio = f"{r.items_per_s / best_torch:.2f}x" if best_torch else ""
            print(
                f"{workload:<17} {r.loader:<14} {r.config:<18} {r.items_per_s:>10.0f}"
                f" {r.first_batch_ms:>9.2f} {ratio:>14}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--workers", type=int, default=4, help="workers (default 4: the CI runner)")
    parser.add_argument("--warmup", type=int, default=1, help="warm-up epochs per configuration")
    parser.add_argument("--repeats", type=int, default=5, help="timed epochs per configuration")
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS, choices=WORKLOADS)
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args()
    # More workers than cores is deliberate for the latency-bound workload.
    warnings.filterwarnings("ignore", message="This DataLoader will create")
    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    meta = {
        "python": sys.version.split()[0],
        "gil": gil,
        "cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "workers": args.workers,
        "batch_size": BATCH_SIZE,
    }
    print("# " + " ".join(f"{k}={v}" for k, v in meta.items()), flush=True)
    results = run(args)
    print_table(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": [asdict(r) for r in results]}, f, indent=1)


if __name__ == "__main__":
    main()
