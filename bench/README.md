# `bench/`: Python benchmarks

Two kinds of scripts, sharing their datasets:

- **Regression guards** (`test_bench_*.py`): pytest-benchmark cases on `dataloader_rs` alone, run
  by CI on every pull request (free-threaded 3.14t) and tracked per commit.
- **Comparison with PyTorch** (`compare_torch.py`, `compare_async.py`, `compare_image.py`): the
  same workloads through `dataloader_rs.PyDataloader` and `torch.utils.data.DataLoader`, run on
  demand.

Both are sized for a 4-vCPU machine, the CI runner: at most 4 workers, except where waiting on
the network is the point. Both run with the allocator settings of `real_world.PINNED_ALLOCATOR`
(glibc's thresholds, mimalloc's purge delay), so they measure the loaders rather than the
allocators' defaults: `compare_torch.py` re-executes itself with them, and CI sets them for the
guards ([Allocator settings](#allocator-settings)). Results: [BENCH.md](../BENCH.md).

## Regression guards (pytest-benchmark)

Marked `bench`, so a plain `pytest` deselects them.

| File | Question | Cases | Dataset |
|---|---|---|---|
| `test_bench_batch_size.py` | loader overhead per batch | `batch_size` 1, 64, 4096; no workers and 4; with `__getitems__` | `InMemoryDs(4096)`, `InMemoryBatchedDs` |
| `test_bench_workers.py` | scaling with workers | `num_workers` 0, 1, 2, 4 | `CpuBoundDs(256)`, `batch_size` 32 |
| `test_bench_prefetch.py` | does loading overlap a training step? | sequential; `prefetch_depth` 1, 2, 4 with 2 workers | `CpuBoundDs(128)`, `batch_size` 8, a 4 ms step per batch |
| `test_bench_first_batch.py` | latency to the first batch | (`num_workers`, `item_concurrency`) (0, 1), (1, 1), (0, 4), (1, 4) | one batch of 32 `CpuBoundDs` items |
| `test_bench_sampler.py` | sampler cost | sequential vs `shuffle=True` | `InMemoryDs(10000)`, `batch_size` 64 |
| `test_bench_real_world.py` | real per-item work | local files and image augmentation (0 and 4 workers); blocking network (4 workers, with and without 8 items in parallel) | `real_world.py` workloads |

```bash
uv run maturin develop --release --features python
GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 \
MIMALLOC_PURGE_DELAY=-1 \
uv run python -X gil=0 -m pytest bench/ -m bench --benchmark-enable --benchmark-warmup=on \
  --benchmark-min-rounds=5
```

pytest's header says whether the allocator is pinned.

## Comparison with PyTorch

`compare_torch.py` runs six workloads, each through both loaders: no workers, 4 workers, torch's
workers fresh and persistent, and for `remote` 64 requests in flight (`item_concurrency` for
`dataloader_rs`, 64 processes for torch). For every configuration it reports the median
throughput over whole epochs, iterator creation included, and the median latency to the first
batch.

| Workload | Per item | Stands for |
|---|---|---|
| `in_memory` | returns the index | the loader's own cost |
| `local_files` | read a 16 KB file, normalise to float32 | file I/O fan-out |
| `image_augment` | random 112 crop, flip, normalise a 3×128×128 image | NumPy augmentation |
| `remote` | 10 ms blocking sleep | an object-store or HTTP GET |
| `python_cpu` | pure-Python tokenisation, shared vocabulary | Python preprocessing |
| `python_cpu_local` | the same with per-thread tables | the same, written for free-threaded Python |

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
taskset -c 0,2,4,6 uv run python -X gil=0 bench/compare_torch.py --workers 4 --repeats 5 --json out.json
uv run --python 3.13 python bench/compare_torch.py --workloads remote   # GIL build, one workload
```

`--help` lists the options; `--default-allocator` keeps the allocators' defaults; without torch
installed only `dataloader_rs` runs. Array samples stay small (64 and 147 KB, batches of 2 and
4.6 MiB): from 16 MiB, mimalloc resets a block freed by another thread whatever its settings.

## Async datasets against PyTorch

`compare_async.py` fetches 2 048 tiles of 64 KiB over HTTP (a mock tile server, `aiohttp.web`
answering after 10 ms with keep-alive, in its own processes on `--server-cpus`) in batches of 64,
through torch with 8, 16, 32 and 64 worker processes and through `dataloader_rs` with a sync `PyDataset`
on threads (both with a `requests.Session` per thread or process, as a real client would), a
`PyAsyncDataset` (an `aiohttp.ClientSession` closed in `aclose()`) and the Rust `HttpDataset`.
Each measurement builds a fresh loader, then reports the median epoch time, first-batch time and
CPU time over rounds run in a rotated order. Results: [BENCH.md](../BENCH.md).

```bash
uv run maturin develop --release
uv run python -X gil=0 bench/compare_async.py --rounds 7
uv run --python 3.13 python bench/compare_async.py --rounds 7   # GIL build
```

## Shared code

| File | What |
|---|---|
| `common.py` | guard datasets (`InMemoryDs`, `InMemoryBatchedDs`, `CpuBoundDs`), collate functions, `training_step` (a consumer step that releases the GIL) |
| `real_world.py` | the workloads as mixins (`InMemoryWork`, `LocalFilesWork`, `ImageAugmentWork`, `RemoteWork`, `PythonCpuWork`, `PythonCpuLocalWork`) combined with `PyDataset` and, at module level so they pickle, with `torch.utils.data.Dataset`; `make_file_tree` writes the test files under the temporary directory |
| `conftest.py` | puts `bench/` on `sys.path` for the pytest cases |

torch is only needed for the comparison; the guards and `real_world.py` import it lazily.

## Image tiles: `compare_image.py`

A tile feature-extraction pass on local PNG tiles: torch's `DataLoader` with Pillow and torchvision
(4 and 8 worker processes) against `FileDataset` with the same Pillow code as a Python
`transform`, and with `dataloader_rs.transforms` + `ImageCollator` (with and without reused
buffers). Batches become torch tensors (`torch.from_dlpack` for `dataloader_rs`); the first
batch of each `dataloader_rs` loader is checked bit for bit against torch's.

```bash
taskset -c 0,2,4,6 python bench/compare_image.py --dir ~/tiles --rounds 5   # Pillow + torchvision for torch
```

## Allocator settings

With their default settings, the allocators behind NumPy give the memory of every freed batch
back to the system, and the next batch faults it in again. With worker threads, those page faults
serialise the workers: with 4 workers on 3.14t, `local_files` and `image_augment` run about half
as fast as with the settings below. The setting belongs to the process, so it is the
application's choice; the benchmarks pin it for both loaders, and it is what we recommend:

```bash
# CPython 3.13 and 3.14t: NumPy allocates with glibc's malloc. Set both thresholds:
# setting one alone turns off glibc's adjustment of the other and is slower than the defaults.
GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 python train.py
# CPython 3.15t: NumPy's buffers come from CPython's mimalloc
MIMALLOC_PURGE_DELAY=-1 python train.py
```

The cost is memory kept for reuse, about 100 MB in these benchmarks. A Rust program can call
`mallopt` with the same thresholds (`benches/bench_real_world.rs` does), or use another global
allocator. `compare_torch.py --default-allocator` measures the defaults instead.
