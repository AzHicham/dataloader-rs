# `bench/`: Python benchmarks

Two kinds of scripts, sharing their datasets and harness:

- **Regression guards** (`test_bench_*.py`): pytest-benchmark cases on `dataloader_rs` alone, run
  by CI on every pull request and tracked per commit.
- **Comparisons with PyTorch** (`bench_*.py`): stand-alone sweeps that run the same datasets
  through `dataloader_rs.PyDataloader` and `torch.utils.data.DataLoader`, run on demand.

Method, results and caveats: [BENCH.md](../BENCH.md).

## Regression guards (pytest-benchmark)

Marked `bench`, so a plain `pytest` deselects them.

| File | Cases | Dataset |
|---|---|---|
| `test_bench_batch_size.py` | `batch_size` 1, 64, 4096; sequential and 4 workers; with `__getitems__` | `InMemoryDs(4096)`, `InMemoryBatchedDs` |
| `test_bench_workers.py` | `num_workers` 0, 1, 4 | `CpuBoundDs(256)`, `batch_size` 32 |
| `test_bench_prefetch.py` | `prefetch_depth` 1, 4, 16 | `CpuBoundDs(256)`, 4 workers |
| `test_bench_sampler.py` | sequential vs `shuffle=True` | `InMemoryDs(10000)`, `batch_size` 64 |
| `test_bench_real_world.py` | local files, image augmentation (0 and 4 workers), blocking network | `real_world.py` workloads |

```bash
uv run maturin develop --release --features python
uv run pytest bench/ -m bench --benchmark-enable --benchmark-warmup=on --benchmark-min-rounds=10
```

## Comparisons with PyTorch

| Script | Sweep | Dataset |
|---|---|---|
| `bench_batch_size.py` | `batch_size` 1 … 4096, sequential and 4 workers | `InMemoryDs` |
| `bench_inter_workers.py` | `num_workers` 0 … 8 | `CpuBoundDs` (SHA-256 of 1 MB) |
| `bench_prefetch_depth.py` | `prefetch_depth` (torch: `prefetch_factor`) | `CpuBoundDs`, 4 workers |
| `bench_sampler.py` | sequential vs shuffle, N 1k … 100k | `CpuBoundDs`, 4 workers |
| `bench_real_world.py` | four real workloads, `w=0`, threads, torch fresh and persistent | `real_world.py` |

```bash
uv run python bench/bench_real_world.py --workers 4 --repeats 5
uv run python bench/bench_batch_size.py --warmup 2 --repeats 10
uv run --python 3.14t python -X gil=0 bench/bench_inter_workers.py   # free-threaded
```

Each script prints a table and a CSV; `--help` lists its options.

## Shared code

| File | What |
|---|---|
| `common.py` | datasets (`InMemoryDs`, `InMemoryBatchedDs`, `CpuBoundDs`, their torch twins), collate functions, `run_case` / `time_epoch` (median of timed epochs), result tables |
| `real_world.py` | the real-world workloads as mixins (`LocalFilesWork`, `ImageAugmentWork`, `RemoteWork`, `PythonCpuWork`) combined with `PyDataset` and with `torch.utils.data.Dataset`, so both loaders run identical work; `make_file_tree` writes the test files under the temp directory |
| `conftest.py` | puts `bench/` on `sys.path` for the pytest cases |

torch is only needed for the comparisons; the regression guards and the real-world module import
it lazily.
