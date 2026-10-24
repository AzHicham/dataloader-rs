# Benchmarks

What `dataloader-rs` is measured on, why, how, and how it compares with
`torch.utils.data.DataLoader`. Every number below comes from one session on one laptop pinned to
4 cores, the size of the CI runners (section 2). The CI re-runs the regression suites on every
pull request and plots them per commit on the dashboards for
[Rust](https://azhicham.github.io/dataloader-rs/benchmarks/rust/) and
[Python 3.14t](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| Reference run | |
|---|---|
| Code | `perf/single-rayon-pool` ([#47](https://github.com/AzHicham/dataloader-rs/pull/47)), 24 October 2026 |
| Machine | Intel Core i9-13900H laptop, pinned to 4 performance cores |
| Python | CPython 3.13.3 (GIL), 3.14.9t and 3.15.1t (free-threaded, `-X gil=0`); torch 2.14.1+cpu, NumPy 2.5.4 |
| Rust | 1.98.1, Criterion 0.8 |

Contents:

1. [Summary](#1-summary)
2. [Setup and calibration](#2-setup-and-calibration)
3. [Method: what is measured and why](#3-method-what-is-measured-and-why)
4. [Results: Rust](#4-results-rust)
5. [Results: Python against PyTorch](#5-results-python-against-pytorch)
6. [Results: Python regression guards](#6-results-python-regression-guards)
7. [Reading the results: choosing settings](#7-reading-the-results-choosing-settings)
8. [Caveats and known bottlenecks](#8-caveats-and-known-bottlenecks)
9. [Running it again](#9-running-it-again)

## 1. Summary

`dataloader_rs.PyDataloader` against `torch.utils.data.DataLoader`, the best configuration of each
loader per workload: no workers or 4, plus 64 requests in flight for `remote`. 4 cores,
`batch_size` 32, the same dataset code for both. Ratio of items per second; above 1,
`dataloader_rs` is faster.

| workload | 3.13 (GIL) | 3.14t | 3.15t |
|---|---:|---:|---:|
| `in_memory` | **5.88×** | **4.90×** | **5.90×** |
| `local_files` | 0.94× | **1.22×** | **1.63×** |
| `image_augment` | 0.99× | 1.01× | **1.07×** |
| `remote` | **3.12×** | **2.99×** | **2.96×** |
| `python_cpu` | 0.42× | 0.31× | 0.38× |
| `python_cpu_local` | 0.44× | 0.79× | 0.94× |

- **Loader overhead** (`in_memory`, items that cost nothing): **4.9–5.9× faster**. Without
  workers the whole loop runs in Rust; torch's runs in Python.
- **Network-bound items** (`remote`, 10 ms per item): **≈ 3× faster**. With
  `items_in_parallel=16`, 4 workers keep 64 requests in flight. torch parallelises whole batches,
  one process each: this 16-batch epoch keeps at most 16 of its 64 processes busy, and its first
  batch always waits for `batch_size` × 10 ms.
- **Array samples** (`local_files`, `image_augment`): **parity to 1.6×**, and neither loader
  gains much from 4 workers. Both pay glibc's `malloc` for returning the samples' memory to the
  system after every batch and faulting it back in (section 8). With glibc's trim and mmap
  thresholds raised, `dataloader_rs` with 4 workers reaches **1.5× (`local_files`) and 1.8×
  (`image_augment`)** torch's best on 3.14t; torch's processes barely gain.
- **Pure-Python CPU work** (`python_cpu*`): **torch is faster**, 1.1–3.2×. Its processes share
  nothing. Threads are serialised by the GIL on 3.13, and contend on the reference counts of the
  shared vocabulary on 3.14t and 3.15t. With per-thread tables, 3.15t comes within 6 %.
- **First batch**: `dataloader_rs` starts an epoch in milliseconds. torch's fresh workers take
  12–334 ms: process start-up, then a whole batch per worker.

Latency to the first batch on 3.14t, workers already started for persistent torch (median of 5
epochs):

| workload | dataloader_rs, 4 workers | torch, 4 fresh workers | torch, 4 persistent workers |
|---|---:|---:|---:|
| `in_memory` | 0.2 ms | 12.1 ms | 0.3 ms |
| `local_files` | 9.2 ms | 55.4 ms | 27.3 ms |
| `image_augment` | 11.6 ms | 123 ms | 22.6 ms |
| `remote` | 22.9 ms (items=16) | 334 ms | 323 ms |
| `python_cpu` | 8.2 ms | 14.4 ms | 2.0 ms |
| `python_cpu_local` | 1.9 ms | 14.6 ms | 2.9 ms |

Rust, without Python (section 4): up to 1 G items/s through the sequential path, about 0.5 µs of
scheduling per batch with workers, 2.8× with 4 workers and 3.0–3.2× with 4 item lanes on
CPU-bound items, and loading fully hidden behind a consumer step from `prefetch_depth = 1`.

## 2. Setup and calibration

The CI runs every suite on GitHub-hosted runners: for public repositories, Linux machines with
**4 vCPUs and 16 GB of RAM** (x64 and arm64). Every suite is sized for them:

- worker sweeps stop at 4 threads: beyond the core count they measure oversubscription, not the
  loader;
- the per-item costs keep an epoch between milliseconds and a few hundred milliseconds, so a suite
  runs in about a minute;
- the exception is waiting on the network, where threads that sleep are the point: 64 requests in
  flight.

Local runs emulate the runner by pinning every process to 4 performance cores, one hardware
thread each:

```bash
taskset -c 0,2,4,6 cargo bench
taskset -c 0,2,4,6 python -X gil=0 bench/compare_torch.py
```

| | |
|---|---|
| CPU | Intel Core i9-13900H: 6 performance cores (2 threads each), 8 efficiency cores. CPUs 0, 2, 4, 6 are 4 distinct performance cores |
| Memory | 32 GB |
| System | Ubuntu, Linux 6.14, glibc 2.41 with default `malloc` settings, on AC power |
| Rust | 1.98.1, bench profile (`opt-level = 3`, thin LTO, one codegen unit) |
| Python | CPython 3.13.3, 3.14.9t, 3.15.1t; torch 2.14.1+cpu; NumPy 2.5.4; pytest-benchmark 5.3 |
| Extension | `maturin develop --release --features python`, once per interpreter |
| Files | the `local_files` trees are in the temporary directory, a RAM-backed tmpfs here: reads measure the file API and the copy, not a disk |

## 3. Method: what is measured and why

### 3.1 One question per benchmark

| Question | Measured as | Rust (`benches/`) | Python (`bench/`) |
|---|---|---|---|
| What does the loader itself cost? | throughput on items that cost nothing | `bench_batch_size` | `test_bench_batch_size.py`; `in_memory` |
| Do more workers help on CPU-bound items? | throughput, `num_workers` 0–4 | `bench_num_workers` | `test_bench_workers.py` |
| Item-level or batch-level parallelism? | throughput at equal thread counts | `bench_items_in_parallel` | — |
| Does loading overlap the training step? | epoch time with a step per batch | `bench_prefetch_depth` | `test_bench_prefetch.py` |
| How long until the first batch? | `iter()` to the first batch | `bench_first_batch` | `test_bench_first_batch.py`; `compare_torch.py` |
| What does the sampler cost? | index generation | `bench_sampler` | `test_bench_sampler.py` |
| How does real per-item work spread? | file reads, augmentation, network latency, Python code | `bench_real_world` | `test_bench_real_world.py`; `compare_torch.py` |
| How do we compare with PyTorch? | throughput and first batch, same dataset code | — | `compare_torch.py` |

Throughput is what a long training run sees. First-batch latency is what every epoch start, a
validation pass or an interactive session waits for, and where torch's worker processes cost the
most. Overlap is what a training loop sees when loading is not the bottleneck: loading should
hide behind the step.

### 3.2 Datasets and workloads

| Name | Per-item work | Cost here | Stands for |
|---|---|---|---|
| `InMemoryDs` (Rust, Python), `in_memory` | returns the index | ≈ 0 | the loader's own cost |
| `HeavyCpuDs` (Rust) | 100 000 steps of a xorshift-multiply chain the compiler cannot shorten | ≈ 110 µs | decoding or augmenting a small sample |
| `CpuBoundDs` (Python) | SHA-256 of 1 MB, releases the GIL | ≈ 0.5 ms | C-extension work |
| `local_files` | read a file (128 KB in Python, 64 KB in Rust), normalise to float32 | ≈ 140 µs | file I/O fan-out |
| `image_augment` | random 224 crop, flip, normalise a 3×256×256 image with NumPy | ≈ 180 µs | image augmentation |
| `remote` | blocking sleep: 10 ms in Python, 2 ms in Rust | latency | an object-store or HTTP GET |
| `python_cpu` | pure-Python tokenisation over a module-level vocabulary | ≈ 40 µs | Python preprocessing |
| `python_cpu_local` | the same with per-thread tables (`threading.local`) | ≈ 40 µs | the same, written for free-threaded Python |

### 3.3 How a number is produced

- **Rust**: Criterion; one iteration is a **whole epoch**, iterator creation included; the loader
  is built once, outside the timed loop. Tables report Criterion's point estimate.
- **Python against PyTorch** (`bench/compare_torch.py`): `time.perf_counter` around whole epochs,
  1 warm-up and 5 timed epochs per configuration, **median**. Throughput includes creating the
  iterator, which is where torch starts fresh workers; the first batch is `iter()` plus the first
  `next()`.
- **Same work on both sides**: each workload is written once (`bench/real_world.py`) and combined
  with `PyDataset` and with `torch.utils.data.Dataset`.
- **Each loader collates its own way**: `dataloader_rs` with `np.stack`; torch with its default
  collation for array samples, which builds the batch tensor in shared memory inside the worker.
  Handing torch the NumPy `collate_fn` instead would pickle each 16 MB batch through a pipe and
  make it 3.6–3.9× slower on these workloads (measured), an unfair comparison. Other samples stay
  a `list` in both.
- **torch fresh and persistent**: fresh worker processes per epoch (the default) and
  `persistent_workers=True`. For `remote`, torch also runs 64 persistent processes, the same
  number of requests in flight as `dataloader_rs` with 4 workers × 16 items.
- **Regression guards**: pytest-benchmark, at least 5 rounds after a warm-up, median.
- **Tracking in CI**: Criterion's bencher output and pytest-benchmark's JSON go to
  [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark), which
  stores one point per commit on `dev` and fails the job, with a comment on the pull request, when
  a benchmark is more than 1.5× slower than the last `dev` result.

## 4. Results: Rust

Criterion suites in `benches/`, without Python ([benches/README.md](benches/README.md) lists
their parameters).

### 4.1 Loader overhead: `bench_batch_size`

4 096 `InMemoryDs` items summed per batch: the loader's own cost. Sequential is `num_workers = 0`;
parallel is 4 workers with `prefetch_depth = 16`. Time per epoch:

| batch_size | sequential | 4 workers |
|---:|---:|---:|
| 1 | 119.7 µs | 2.00 ms |
| 8 | 27.4 µs | 200.6 µs |
| 32 | 14.2 µs | 108.4 µs |
| 128 | 10.9 µs | 26.1 µs |
| 512 | 6.0 µs | 15.6 µs |
| 1024 | 5.0 µs | 12.5 µs |
| 4096 | 3.9 µs | 13.9 µs |

- Sequential: 29 ns per batch at `batch_size` 1, and 0.7–1 G items/s from `batch_size` 512 up.
- Parallel: about 0.5 µs of scheduling per batch (claim, hand-off, reorder), so workers only pay
  off when a batch costs more than a few microseconds. Nearly free batches are faster without
  workers.

### 4.2 Batch-level parallelism: `bench_num_workers`

256 `HeavyCpuDs` items, `batch_size` 16, `prefetch_depth` 4:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| num_workers = 0 | 24.13 ms | 10,611 | 1.00× |
| num_workers = 1 | 24.69 ms | 10,367 | 0.98× |
| num_workers = 2 | 12.70 ms | 20,161 | 1.90× |
| num_workers = 4 | 8.54 ms | 29,972 | 2.82× |

1.9× with 2 workers, 2.8× with 4. One worker matches the sequential path: its only extra cost is
the hand-off. Most of the gap to 4× is the laptop's clock: items take ≈ 127 µs each when 4 cores
are busy, ≈ 96 µs when one is (compare the first and third rows of 4.3).

### 4.3 Item-level parallelism: `bench_items_in_parallel`

128 `HeavyCpuDs` items in 2 batches of 64: few batches, large ones.

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| workers 1, items 1 | 12.33 ms | 10,383 | 1.00× |
| workers 1, items 2 | 6.48 ms | 19,762 | 1.90× |
| workers 1, items 4 | 4.05 ms | 31,573 | 3.04× |
| workers 2, items 2 | 3.90 ms | 32,813 | 3.16× |
| workers 4, items 1 | 6.45 ms | 19,837 | 1.91× |

With 2 batches per epoch, 4 batch workers can only use 2 threads (1.9×). Spreading each batch's
items over the threads uses all 4: 3.0–3.2×.

### 4.4 Overlap with the consumer: `bench_prefetch_depth`

16 batches of 4 `HeavyCpuDs` items (≈ 0.45 ms to load each); the consumer works ≈ 0.22 ms per
batch, like a training step; 2 workers load a batch every ≈ 0.22 ms.

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| sequential (no workers) | 9.02 ms | 7,095 | 1.00× |
| prefetch_depth = 1 | 3.76 ms | 17,002 | 2.40× |
| prefetch_depth = 2 | 3.81 ms | 16,807 | 2.37× |
| prefetch_depth = 4 | 3.84 ms | 16,669 | 2.35× |
| prefetch_depth = 8 | 3.87 ms | 16,535 | 2.33× |

Sequential, loading and the step add up: ≈ 0.56 ms per batch. With workers, loading hides behind
the step: 3.8 ms for 16 steps of ≈ 0.22 ms plus the first batch. `prefetch_depth = 1` already
overlaps fully; a deeper prefetch only holds more memory.

### 4.5 Latency to the first batch: `bench_first_batch`

An epoch of one batch of 32 `HeavyCpuDs` items: one iteration is `iter()` until that batch is
ready.

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| workers 0, items 1 | 3.11 ms | 10,273 | 1.00× |
| workers 1, items 1 | 3.20 ms | 9,990 | 0.97× |
| workers 0, items 2 | 1.70 ms | 18,841 | 1.83× |
| workers 0, items 4 | 979.9 µs | 32,657 | 3.18× |
| workers 1, items 4 | 988.5 µs | 32,372 | 3.15× |

A worker adds ≈ 0.1 ms of hand-off. `items_in_parallel = 4` cuts the wait 3.2×: the first batch is
fetched by 4 threads instead of one.

### 4.6 Sampler: `bench_sampler`

Index generation only, `batch_size` 128, no workers:

| N | sequential | random |
|---:|---:|---:|
| 1,000 | 2.0 µs | 3.5 µs |
| 10,000 | 25.2 µs | 39.3 µs |
| 100,000 | 261.3 µs | 424.8 µs |

2.6 ns per index sequential, 4.2 ns random (a Fisher–Yates permutation per epoch): 1.6×.

### 4.7 Real per-item work: `bench_real_world`

1 024 files of 64 KB, read and normalised to `f32`, `batch_size` 32:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| num_workers = 0 | 75.59 ms | 13,546 | 1.00× |
| num_workers = 1 | 19.79 ms | 51,741 | 3.82× |
| num_workers = 2 | 11.46 ms | 89,335 | 6.59× |
| num_workers = 4 | 21.93 ms | 46,694 | 3.45× |

The allocator decides this one. Each item allocates a 256 KB `Vec<f32>`. With default settings,
glibc gives the memory of every dropped batch back to the system and the next batch faults it in
again. Without workers that costs 3.9×. With `GLIBC_TUNABLES=glibc.malloc.trim_threshold=1073741824:glibc.malloc.mmap_threshold=33554432`
the same epoch takes 19.5, 20.2, 11.3 and 15.5 ms with 0, 1, 2 and 4 workers. Beyond 2 workers it
stops scaling even then: the epoch moves ≈ 450 MB through memory, and 4 workers are slower than 2
(not investigated yet).

512 items with a 2 ms blocking round trip, `batch_size` 32, 4 workers:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| 4 requests in flight (4 workers × 1 items) | 264.28 ms | 1,937 | 1.00× |
| 16 requests in flight (4 workers × 4 items) | 79.64 ms | 6,429 | 3.32× |
| 64 requests in flight (4 workers × 16 items) | 21.09 ms | 24,273 | 12.53× |

Requests in flight are `num_workers × items_in_parallel`, and throughput follows them: 264 ms
for an ideal 256 ms at 4 in flight, 21 ms for an ideal 16 ms at 64.

## 5. Results: Python against PyTorch

`bench/compare_torch.py`, 4 workers, `batch_size` 32. `w` is `num_workers`, `items` is
`items_in_parallel`; the last column divides by the best torch configuration of the workload.

### 5.1 CPython 3.14t (free-threaded)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 24,033,398 | 0.02 ms | 4.90× |
| `in_memory` | **dataloader_rs** | w=4 | 820,095 | 0.17 ms | 0.17× |
| `in_memory` | torch | w=0 | 4,903,192 | 0.03 ms | 1.00× |
| `in_memory` | torch | w=4 | 413,309 | 12.14 ms | 0.08× |
| `in_memory` | torch | w=4 persistent | 1,228,449 | 0.32 ms | 0.25× |
| `local_files` | **dataloader_rs** | w=0 | 7,342 | 4.45 ms | 1.22× |
| `local_files` | **dataloader_rs** | w=4 | 6,813 | 9.25 ms | 1.13× |
| `local_files` | torch | w=0 | 6,030 | 4.19 ms | 1.00× |
| `local_files` | torch | w=4 | 5,152 | 55.38 ms | 0.85× |
| `local_files` | torch | w=4 persistent | 5,457 | 27.34 ms | 0.90× |
| `image_augment` | **dataloader_rs** | w=0 | 5,507 | 6.20 ms | 0.99× |
| `image_augment` | **dataloader_rs** | w=4 | 5,583 | 11.59 ms | 1.01× |
| `image_augment` | torch | w=0 | 4,335 | 7.68 ms | 0.78× |
| `image_augment` | torch | w=4 | 3,284 | 123 ms | 0.59× |
| `image_augment` | torch | w=4 persistent | 5,545 | 22.58 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 322 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 397 | 323 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 4,664 | 22.89 ms | 2.99× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 389 | 334 ms | 0.25× |
| `remote` | torch | w=4 persistent | 396 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,562 | 327 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 23,774 | 1.33 ms | 0.31× |
| `python_cpu` | **dataloader_rs** | w=4 | 13,119 | 8.16 ms | 0.17× |
| `python_cpu` | torch | w=0 | 26,904 | 1.30 ms | 0.36× |
| `python_cpu` | torch | w=4 | 39,735 | 14.39 ms | 0.52× |
| `python_cpu` | torch | w=4 persistent | 75,711 | 1.98 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 24,090 | 1.34 ms | 0.34× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 55,430 | 1.90 ms | 0.79× |
| `python_cpu_local` | torch | w=0 | 25,130 | 1.34 ms | 0.36× |
| `python_cpu_local` | torch | w=4 | 40,454 | 14.56 ms | 0.58× |
| `python_cpu_local` | torch | w=4 persistent | 70,255 | 2.92 ms | 1.00× |

### 5.2 CPython 3.13 (GIL)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 26,971,813 | 0.01 ms | 5.88× |
| `in_memory` | **dataloader_rs** | w=4 | 2,438,613 | 0.09 ms | 0.53× |
| `in_memory` | torch | w=0 | 4,587,084 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 473,194 | 8.50 ms | 0.10× |
| `in_memory` | torch | w=4 persistent | 909,926 | 0.36 ms | 0.20× |
| `local_files` | **dataloader_rs** | w=0 | 7,561 | 4.38 ms | 0.94× |
| `local_files` | **dataloader_rs** | w=4 | 5,751 | 16.19 ms | 0.72× |
| `local_files` | torch | w=0 | 8,020 | 4.12 ms | 1.00× |
| `local_files` | torch | w=4 | 6,175 | 40.21 ms | 0.77× |
| `local_files` | torch | w=4 persistent | 6,378 | 21.67 ms | 0.80× |
| `image_augment` | **dataloader_rs** | w=0 | 5,457 | 6.22 ms | 0.99× |
| `image_augment` | **dataloader_rs** | w=4 | 4,932 | 32.97 ms | 0.89× |
| `image_augment` | torch | w=0 | 4,541 | 6.84 ms | 0.82× |
| `image_augment` | torch | w=4 | 4,835 | 50.90 ms | 0.88× |
| `image_augment` | torch | w=4 persistent | 5,518 | 26.35 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 324 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 396 | 323 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 4,909 | 21.53 ms | 3.12× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 392 | 334 ms | 0.25× |
| `remote` | torch | w=4 persistent | 397 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,571 | 326 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 29,683 | 1.07 ms | 0.42× |
| `python_cpu` | **dataloader_rs** | w=4 | 29,739 | 5.84 ms | 0.42× |
| `python_cpu` | torch | w=0 | 26,821 | 1.25 ms | 0.38× |
| `python_cpu` | torch | w=4 | 42,844 | 13.88 ms | 0.61× |
| `python_cpu` | torch | w=4 persistent | 70,196 | 2.85 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 29,909 | 1.08 ms | 0.42× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 31,285 | 5.70 ms | 0.44× |
| `python_cpu_local` | torch | w=0 | 29,548 | 1.16 ms | 0.41× |
| `python_cpu_local` | torch | w=4 | 45,090 | 12.96 ms | 0.63× |
| `python_cpu_local` | torch | w=4 persistent | 71,448 | 2.68 ms | 1.00× |

### 5.3 CPython 3.15t (free-threaded)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 28,053,161 | 0.02 ms | 5.90× |
| `in_memory` | **dataloader_rs** | w=4 | 1,489,004 | 0.07 ms | 0.31× |
| `in_memory` | torch | w=0 | 4,753,919 | 0.03 ms | 1.00× |
| `in_memory` | torch | w=4 | 396,686 | 13.97 ms | 0.08× |
| `in_memory` | torch | w=4 persistent | 1,373,676 | 0.24 ms | 0.29× |
| `local_files` | **dataloader_rs** | w=0 | 7,050 | 4.49 ms | 0.95× |
| `local_files` | **dataloader_rs** | w=4 | 12,083 | 10.70 ms | 1.63× |
| `local_files` | torch | w=0 | 7,396 | 4.48 ms | 1.00× |
| `local_files` | torch | w=4 | 6,123 | 47.10 ms | 0.83× |
| `local_files` | torch | w=4 persistent | 6,789 | 23.34 ms | 0.92× |
| `image_augment` | **dataloader_rs** | w=0 | 5,463 | 5.82 ms | 0.93× |
| `image_augment` | **dataloader_rs** | w=4 | 6,256 | 13.60 ms | 1.07× |
| `image_augment` | torch | w=0 | 5,843 | 5.28 ms | 1.00× |
| `image_augment` | torch | w=4 | 3,601 | 117 ms | 0.62× |
| `image_augment` | torch | w=4 persistent | 5,485 | 24.10 ms | 0.94× |
| `remote` | **dataloader_rs** | w=0 | 99 | 323 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 396 | 323 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 4,629 | 22.80 ms | 2.96× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 390 | 335 ms | 0.25× |
| `remote` | torch | w=4 persistent | 397 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,563 | 327 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 25,417 | 1.24 ms | 0.36× |
| `python_cpu` | **dataloader_rs** | w=4 | 27,297 | 3.85 ms | 0.38× |
| `python_cpu` | torch | w=0 | 28,195 | 1.24 ms | 0.39× |
| `python_cpu` | torch | w=4 | 32,681 | 18.31 ms | 0.46× |
| `python_cpu` | torch | w=4 persistent | 71,472 | 1.84 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 26,802 | 1.19 ms | 0.38× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 66,213 | 1.71 ms | 0.94× |
| `python_cpu_local` | torch | w=0 | 28,795 | 1.19 ms | 0.41× |
| `python_cpu_local` | torch | w=4 | 39,469 | 14.68 ms | 0.56× |
| `python_cpu_local` | torch | w=4 persistent | 70,684 | 3.09 ms | 1.00× |

### 5.4 Reading the tables

- **`in_memory`**: without workers `dataloader_rs` streams 24–28 M items/s against torch's
  4.6–4.9 M, because the loop runs in Rust. With 4 workers both loaders are bound by handing
  batches over. `dataloader_rs` is ahead on 3.13 and 3.15t, behind torch's persistent workers on
  3.14t: 25–45 µs per batch through Python against 0.5 µs in Rust (section 8).
- **`local_files`, `image_augment`**: every configuration lands between 3.3 k and 8 k items/s,
  except `dataloader_rs` with 4 workers on 3.15t for `local_files` (12 k). Workers, threads or
  processes, mostly wait on the allocator's page faults, not on the loader (section 8).
- **`remote`**: requests in flight decide throughput, and only `items_in_parallel` raises them
  without more processes: 4 workers × 16 items reach 4.6–4.9 k items/s. torch's first batch is
  always `batch_size` × 10 ms = 320 ms, because one process fetches a whole batch.
- **`python_cpu`**: threads lose. With the GIL they run one at a time. On 3.14t and 3.15t they run
  in parallel but contend on the reference counts of the shared vocabulary dict and word list:
  4 workers are slower than none on 3.14t.
- **`python_cpu_local`**: per-thread tables remove that contention. 4 workers give 2.3× on 3.14t
  and 2.5× on 3.15t, close to torch's persistent processes (0.79× and 0.94×).

## 6. Results: Python regression guards

`bench/test_bench_*.py` on 3.14t, the CI's interpreter, pinned to 4 cores; median per round.
These cases run on every pull request; their history is on the
[dashboard](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| file | case | median |
|---|---|---:|
| `test_bench_batch_size.py` | `test_batch_size_sequential[1]` | 0.54 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential[64]` | 0.13 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential[4096]` | 0.14 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[1]` | 185.63 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[64]` | 1.52 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[4096]` | 0.63 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential_getitems[64]` | 0.08 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential_getitems[4096]` | 0.07 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel_getitems[64]` | 1.44 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel_getitems[4096]` | 0.47 ms |
| `test_bench_first_batch.py` | `test_first_batch[0-1]` | 15.85 ms |
| `test_bench_first_batch.py` | `test_first_batch[1-1]` | 16.14 ms |
| `test_bench_first_batch.py` | `test_first_batch[0-4]` | 5.54 ms |
| `test_bench_first_batch.py` | `test_first_batch[1-4]` | 6.29 ms |
| `test_bench_prefetch.py` | `test_overlap_sequential` | 135.15 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[1]` | 69.26 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[2]` | 70.37 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[4]` | 69.90 ms |
| `test_bench_real_world.py` | `test_local_files[0]` | 80.00 ms |
| `test_bench_real_world.py` | `test_local_files[4]` | 49.16 ms |
| `test_bench_real_world.py` | `test_image_augment[0]` | 42.22 ms |
| `test_bench_real_world.py` | `test_image_augment[4]` | 25.64 ms |
| `test_bench_real_world.py` | `test_remote_threads` | 324.05 ms |
| `test_bench_real_world.py` | `test_remote_items_in_parallel` | 41.36 ms |
| `test_bench_sampler.py` | `test_sampler_sequential` | 0.27 ms |
| `test_bench_sampler.py` | `test_sampler_shuffle` | 0.30 ms |
| `test_bench_workers.py` | `test_num_workers[0]` | 125.52 ms |
| `test_bench_workers.py` | `test_num_workers[1]` | 129.87 ms |
| `test_bench_workers.py` | `test_num_workers[2]` | 67.67 ms |
| `test_bench_workers.py` | `test_num_workers[4]` | 42.05 ms |

The first-batch and overlap guards mirror the Rust suites: 4 item lanes cut the first batch from
16 to 5.5 ms; with 2 workers, 16 batches overlap with 4 ms steps (70 ms, against 135 ms
sequentially, 64 ms of steps). `test_batch_size_parallel[1]` shows the Python path's per-batch
cost: 186 ms for 4 096 single-item batches, 45 µs each (section 8).

## 7. Reading the results: choosing settings

- **`num_workers`**: 0 when a batch costs about as much as handing it over: a few µs in Rust,
  tens of µs through Python. Otherwise up to the number of cores for CPU-bound items; beyond that
  threads only compete for the cores.
- **`items_in_parallel`**: for datasets that wait (object stores, HTTP, databases): requests in
  flight are `num_workers × items_in_parallel`, and the pool has that many threads. Also for the
  first batch, and for epochs with fewer batches than workers.
- **`prefetch_depth`**: the default 2 is enough when loading keeps up with the step; a deeper
  prefetch only absorbs uneven batch times, at the cost of memory.
- **Python**: on the GIL build, threads help when the dataset releases the GIL (I/O, NumPy, most C
  extensions). For pure-Python preprocessing use free-threaded 3.14t or 3.15t and keep hot lookup
  tables per thread (`threading.local`). If that is not possible, torch's processes are faster.
- **Array samples on Linux**: raise glibc's `malloc` trim and mmap thresholds, or use jemalloc or
  mimalloc (section 8).

## 8. Caveats and known bottlenecks

- **glibc `malloc` and array samples.** NumPy allocates sample and batch buffers with `malloc`.
  With default settings glibc returns freed memory to the system and faults it back in on the next
  allocation. A plain Python loop over `local_files` takes ≈ 147 page faults per item by default
  and none with raised thresholds. With worker threads, those faults and the system calls behind
  them serialise the workers. On 3.14t with
  `GLIBC_TUNABLES=glibc.malloc.trim_threshold=1073741824:glibc.malloc.mmap_threshold=33554432`
  set for both loaders:

  | workload | `dataloader_rs`, 4 workers | best torch |
  |---|---:|---:|
  | `local_files` | 5 743 → 12 779 items/s | 8 510 (`w=0`) |
  | `image_augment` | 4 433 → 10 461 items/s | 5 623 (`w=0`) |

  Not set by default here; the next round of optimisation addresses it in the loader.
- **Per-batch cost of the Python parallel path.** 25–45 µs per batch on 3.14t, against 0.5 µs in
  Rust. Only nearly free batches feel it (`in_memory`, `batch_size` 1), but it caps tiny batches.
  This is the other target of the next round.
- **One machine, one session.** A laptop with turbo clocks: ratios measured in the same session
  are more reliable than absolute numbers, and a 4-vCPU runner is slower per core.
- **Workload sizes favour neither loader**, with one exception: the `remote` epoch has 16 batches,
  so torch keeps only 16 of its 64 processes busy. With more batches than processes it would
  approach the same 64-in-flight ceiling, with 64 Python processes.
- **Page cache.** `local_files` reads files that sit in memory: the file API and the copy, not a
  disk.
- **CI numbers are noisy.** Shared runners vary by 1.5× and more between runs; the 1.5× alert
  catches large regressions only. Measure a change with an interleaved local A/B (section 9.4).

## 9. Running it again

### 9.1 Rust (Criterion)

```bash
taskset -c 0,2,4,6 cargo bench                            # every suite, pinned like the runner
cargo bench --bench bench_num_workers                     # one suite
cargo bench --bench bench_real_world -- local_files       # filter inside a suite
cargo bench --bench bench_first_batch -- --output-format bencher   # CI's output format
```

HTML reports land in `target/criterion/`.

### 9.2 Python regression guards (as in CI)

```bash
uv sync --group dev
uv run maturin develop --release --features python
uv run python -X gil=0 -m pytest bench/ -m bench --benchmark-enable --benchmark-warmup=on \
  --benchmark-min-rounds=5
```

### 9.3 Python against PyTorch

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
taskset -c 0,2,4,6 uv run python -X gil=0 bench/compare_torch.py --workers 4 --repeats 5 --json out.json
uv run --python 3.13 python bench/compare_torch.py     # with the GIL
```

`--workloads` runs a subset; without torch installed, only `dataloader_rs` runs.

### 9.4 Comparing two versions

Run both builds alternately in the same session (A, B, A, B, …) and compare medians: the machine
drifts more between sessions than between two interleaved runs. For Criterion, delete
`target/criterion` between builds so it does not compare against the other build's baseline.
