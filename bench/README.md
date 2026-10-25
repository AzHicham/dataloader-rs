# `bench/`: Python benchmarks

Two kinds of scripts, sharing their datasets:

- **Regression guards** (`test_bench_*.py`): pytest-benchmark cases on `dataloader_rs` alone, run
  by CI on every pull request (free-threaded 3.14t) and tracked per commit.
- **Comparison with PyTorch** (`compare_torch.py`): the same workloads through
  `dataloader_rs.PyDataloader` and `torch.utils.data.DataLoader`, run on demand.

Both are sized for a 4-vCPU machine, the CI runner: at most 4 workers, except where waiting on
the network is the point. Both run with the allocator settings of `real_world.PINNED_ALLOCATOR`
(glibc's thresholds, mimalloc's purge delay), so they measure the loaders rather than the
allocators' defaults: `compare_torch.py` re-executes itself with them, and CI sets them for the
guards. Method, results and caveats: [BENCH.md](../BENCH.md).

## Regression guards (pytest-benchmark)

Marked `bench`, so a plain `pytest` deselects them.

| File | Question | Cases | Dataset |
|---|---|---|---|
| `test_bench_batch_size.py` | loader overhead per batch | `batch_size` 1, 64, 4096; no workers and 4; with `__getitems__` | `InMemoryDs(4096)`, `InMemoryBatchedDs` |
| `test_bench_workers.py` | scaling with workers | `num_workers` 0, 1, 2, 4 | `CpuBoundDs(256)`, `batch_size` 32 |
| `test_bench_prefetch.py` | does loading overlap a training step? | sequential; `prefetch_depth` 1, 2, 4 with 2 workers | `CpuBoundDs(128)`, `batch_size` 8, a 4 ms step per batch |
| `test_bench_first_batch.py` | latency to the first batch | (`num_workers`, `items_in_parallel`) (0, 1), (1, 1), (0, 4), (1, 4) | one batch of 32 `CpuBoundDs` items |
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
workers fresh and persistent, and for `remote` 64 requests in flight (`items_in_parallel` for
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

## Shared code

| File | What |
|---|---|
| `common.py` | guard datasets (`InMemoryDs`, `InMemoryBatchedDs`, `CpuBoundDs`), collate functions, `training_step` (a consumer step that releases the GIL) |
| `real_world.py` | the workloads as mixins (`InMemoryWork`, `LocalFilesWork`, `ImageAugmentWork`, `RemoteWork`, `PythonCpuWork`, `PythonCpuLocalWork`) combined with `PyDataset` and, at module level so they pickle, with `torch.utils.data.Dataset`; `make_file_tree` writes the test files under the temporary directory |
| `conftest.py` | puts `bench/` on `sys.path` for the pytest cases |

torch is only needed for the comparison; the guards and `real_world.py` import it lazily.
