# Benchmarks

What `dataloader-rs` is measured on, why, how, and how it compares with
`torch.utils.data.DataLoader`. Every number below comes from one session on one laptop pinned to
4 cores, the size of the CI runners (section 2). The CI re-runs the regression suites on every
pull request and plots them per commit on the dashboards for
[Rust](https://azhicham.github.io/dataloader-rs/benchmarks/rust/) and
[Python 3.14t](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| Reference run | |
|---|---|
| Code | `perf/hot-path` ([#53](https://github.com/AzHicham/dataloader-rs/pull/53)), on top of [#47](https://github.com/AzHicham/dataloader-rs/pull/47), 25 October 2026 |
| Machine | Intel Core i9-13900H laptop, pinned to 4 performance cores |
| Python | CPython 3.13.3 (GIL), 3.14.9t and 3.15.1t (free-threaded, `-X gil=0`); torch 2.14.1+cpu, NumPy 2.5.4 |
| Rust | 1.98.1, Criterion 0.8 |
| Allocator | settings pinned for both loaders by the harnesses (section 8) |

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
`batch_size` 32, the same dataset code and the same allocator settings for both. Ratio of items
per second; above 1, `dataloader_rs` is faster.

| workload | 3.13 (GIL) | 3.14t | 3.15t |
|---|---:|---:|---:|
| `in_memory` | **4.92×** | **6.02×** | **6.61×** |
| `local_files` | 1.00× | **3.52×** | **3.41×** |
| `image_augment` | **1.52×** | **2.61×** | **2.48×** |
| `remote` | **3.87×** | **3.19×** | **3.17×** |
| `python_cpu` | 0.41× | 0.34× | 0.40× |
| `python_cpu_local` | 0.39× | **1.16×** | **1.19×** |

> **Where PyTorch is faster, and why.** The workers are threads in one process. On regular CPython
> (the GIL build, 3.13 here), only one thread runs Python code at a time, so pure-Python work in
> `__getitem__` does not run in parallel, while PyTorch's worker processes each run their own
> interpreter: PyTorch is 2.4–2.6× faster on `python_cpu` and `python_cpu_local`. On free-threaded
> 3.14t / 3.15t the threads do run Python in parallel, but threads that all read one shared dict
> or list contend on its reference count: PyTorch is 2.5–2.9× faster on `python_cpu`, and
> `dataloader_rs` 1.16–1.19× faster once each thread has its own tables (`python_cpu_local`).
> Work that releases the GIL (I/O, NumPy, image decoding) is not affected.

- **Loader overhead** (`in_memory`, items that cost nothing): **4.9–6.6× faster**. Without
  workers the whole loop runs in Rust; torch's runs in Python. With 4 workers, `dataloader_rs`
  hands over 5.6–12 M items/s against 1.0–1.4 M for torch's persistent workers.
- **Network-bound items** (`remote`, 10 ms per item): **3.2–3.9× faster**. With
  `items_in_parallel=16`, 4 workers keep 64 requests in flight. torch parallelises whole batches,
  one process each: this 16-batch epoch keeps at most 16 of its 64 processes busy, and its first
  batch always waits for `batch_size` × 10 ms.
- **Array samples** (`local_files`, `image_augment`): **2.5–3.5× faster** on 3.14t and 3.15t,
  where 4 workers reach 180–193 k and 57–59 k items/s. On 3.13 the GIL serialises the small
  NumPy calls: 1.0–1.5×. torch's best configuration is no workers or persistent ones: handing a
  batch between processes costs more than these small samples. With the allocators' default
  settings, 4 workers lose up to half of that on 3.14t (section 8).
- **Pure-Python CPU work** (`python_cpu*`): with a shared vocabulary, **torch is faster**
  (2.4–2.9×): threads are serialised by the GIL on 3.13, and contend on the reference counts of
  the shared objects on 3.14t and 3.15t. With per-thread tables (`python_cpu_local`),
  `dataloader_rs` is **1.16–1.19× faster** on 3.14t and 3.15t; on 3.13 torch stays 2.6× ahead.
- **First batch**: `dataloader_rs` starts an epoch in milliseconds. torch's fresh workers take
  11–334 ms: process start-up, then a whole batch per worker.

Latency to the first batch on 3.14t, workers already started for persistent torch (median of 5
epochs):

| workload | dataloader_rs, 4 workers | torch, 4 fresh workers | torch, 4 persistent workers |
|---|---:|---:|---:|
| `in_memory` | 0.03 ms | 11.1 ms | 0.24 ms |
| `local_files` | 0.73 ms | 23.3 ms | 5.7 ms |
| `image_augment` | 2.2 ms | 36.2 ms | 10.3 ms |
| `remote` | 20.8 ms (items=16) | 334 ms | 323 ms |
| `python_cpu` | 9.0 ms | 13.0 ms | 2.9 ms |
| `python_cpu_local` | 1.3 ms | 13.6 ms | 1.9 ms |

Rust, without Python (section 4): up to 1 G items/s through the sequential path, about 0.5 µs of
scheduling per batch with workers, 3.1× with 4 workers and 3.4–3.5× with 4 item lanes on
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
| System | Ubuntu, Linux 6.14, glibc 2.41, on AC power; allocator settings pinned by the harnesses (section 8) |
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
| `local_files` | read a 16 KB file, normalise to float32 | ≈ 18 µs (Python), 6 µs (Rust) | file I/O fan-out |
| `image_augment` | random 112 crop, flip, normalise a 3×128×128 image with NumPy | ≈ 42 µs | image augmentation |
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
  Handing torch the NumPy `collate_fn` instead would pickle every batch through a pipe: 3.6–3.9×
  slower with 16 MB batches (measured), an unfair comparison. Other samples stay a `list` in both.
- **Allocator settings pinned, for both loaders**: `GLIBC_TUNABLES` (glibc's mmap and trim
  thresholds, what NumPy allocates with on 3.13 and 3.14t) and `MIMALLOC_PURGE_DELAY=-1`
  (CPython's mimalloc, what NumPy allocates with on 3.15t). With the defaults, the allocator
  returns the memory of every batch to the system and the numbers measure that instead of the
  loader (section 8). `compare_torch.py` re-executes itself with them, CI sets them for the
  guards, and the Rust real-world suite calls `mallopt`. Array samples also stay small (64 and
  147 KB, batches of 2 and 4.6 MiB): from 16 MiB, mimalloc resets a block freed by another thread
  whatever its settings.
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
| 1 | 112.1 µs | 1.85 ms |
| 8 | 27.3 µs | 186.4 µs |
| 32 | 15.1 µs | 101.9 µs |
| 128 | 9.1 µs | 24.0 µs |
| 512 | 5.5 µs | 14.9 µs |
| 1024 | 4.6 µs | 19.9 µs |
| 4096 | 3.9 µs | 13.0 µs |

- Sequential: 27 ns per batch at `batch_size` 1, and 0.7–1 G items/s from `batch_size` 512 up.
- Parallel: about 0.5 µs of scheduling per batch, so workers only pay off when a batch costs more
  than a few microseconds; nearly free batches are faster without workers. Most of it is a batch
  task exiting when the window is full and being spawned again as the consumer advances: with a
  window that never fills, the whole path (claim, fetch, hand-off, reorder) costs ≈ 0.16 µs.

### 4.2 Batch-level parallelism: `bench_num_workers`

256 `HeavyCpuDs` items, `batch_size` 16, `prefetch_depth` 4:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| num_workers = 0 | 23.94 ms | 10,692 | 1.00× |
| num_workers = 1 | 24.21 ms | 10,576 | 0.99× |
| num_workers = 2 | 12.31 ms | 20,802 | 1.95× |
| num_workers = 4 | 7.68 ms | 33,329 | 3.12× |

1.95× with 2 workers, 3.1× with 4. One worker matches the sequential path: its only extra cost is
the hand-off. Most of the gap to 4× is the laptop's clock: items take ≈ 111 µs each when 4 cores
are busy, ≈ 95 µs when one is (compare the first and third rows of 4.3).

### 4.3 Item-level parallelism: `bench_items_in_parallel`

128 `HeavyCpuDs` items in 2 batches of 64: few batches, large ones.

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| workers 1, items 1 | 12.16 ms | 10,529 | 1.00× |
| workers 1, items 2 | 6.26 ms | 20,441 | 1.94× |
| workers 1, items 4 | 3.56 ms | 35,960 | 3.42× |
| workers 2, items 2 | 3.48 ms | 36,740 | 3.49× |
| workers 4, items 1 | 6.15 ms | 20,816 | 1.98× |

With 2 batches per epoch, 4 batch workers can only use 2 threads (2.0×). Spreading each batch's
items over the threads uses all 4: 3.4–3.5×.

### 4.4 Overlap with the consumer: `bench_prefetch_depth`

16 batches of 4 `HeavyCpuDs` items (≈ 0.45 ms to load each); the consumer works ≈ 0.22 ms per
batch, like a training step; 2 workers load a batch every ≈ 0.22 ms.

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| sequential (no workers) | 8.97 ms | 7,135 | 1.00× |
| prefetch_depth = 1 | 3.69 ms | 17,352 | 2.43× |
| prefetch_depth = 2 | 3.73 ms | 17,172 | 2.41× |
| prefetch_depth = 4 | 3.73 ms | 17,180 | 2.41× |
| prefetch_depth = 8 | 3.72 ms | 17,186 | 2.41× |

Sequential, loading and the step add up: ≈ 0.56 ms per batch. With workers, loading hides behind
the step: 3.7 ms for 16 steps of ≈ 0.22 ms plus the first batch. `prefetch_depth = 1` already
overlaps fully; a deeper prefetch only holds more memory.

### 4.5 Latency to the first batch: `bench_first_batch`

An epoch of one batch of 32 `HeavyCpuDs` items: one iteration is `iter()` until that batch is
ready.

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| workers 0, items 1 | 3.05 ms | 10,506 | 1.00× |
| workers 1, items 1 | 3.12 ms | 10,259 | 0.98× |
| workers 0, items 2 | 1.63 ms | 19,592 | 1.86× |
| workers 0, items 4 | 832.8 µs | 38,422 | 3.66× |
| workers 1, items 4 | 882.9 µs | 36,245 | 3.45× |

A worker adds ≈ 0.07 ms of hand-off. `items_in_parallel = 4` cuts the wait 3.7×: the first batch is
fetched by 4 threads instead of one.

### 4.6 Sampler: `bench_sampler`

Index generation only, `batch_size` 128, no workers:

| N | sequential | random |
|---:|---:|---:|
| 1,000 | 1.8 µs | 3.4 µs |
| 10,000 | 21.8 µs | 33.8 µs |
| 100,000 | 236.0 µs | 366.7 µs |

2.4 ns per index sequential, 3.7 ns random (a Fisher–Yates permutation per epoch): 1.6×.

### 4.7 Real per-item work: `bench_real_world`

4 096 files of 16 KB, read and normalised to `f32`, `batch_size` 32, with glibc's malloc
thresholds pinned by the suite (section 8):

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| num_workers = 0 | 23.98 ms | 170,778 | 1.00× |
| num_workers = 1 | 24.97 ms | 164,026 | 0.96× |
| num_workers = 2 | 13.41 ms | 305,479 | 1.79× |
| num_workers = 4 | 12.01 ms | 341,143 | 2.00× |

1.8× with 2 workers, 2.0× with 4. Two things decide this suite besides the loader:

- **The allocator.** Each item allocates a 64 KB `Vec<f32>`. With glibc's defaults, the memory of
  every dropped batch goes back to the system and the next batch faults it in again: the same
  epoch is 3.4× slower without workers (15 page faults per item). The suite therefore calls
  `mallopt` with the thresholds of section 8 before it starts; a Rust program can do the same, or
  pick another global allocator.
- **Memory bandwidth.** This laptop copies 22.5, 27.5 and 32 GB/s with 1, 2 and 4 threads. With
  64 KB files, about 0.7 MB of memory traffic per item, 2 workers already filled it and 4 were
  slower than 2: each item cost twice the CPU time, in memory stalls. 16 KB files keep the suite
  within the bandwidth (4 KB files scale 2.6× with 4 workers).

512 items with a 2 ms blocking round trip, `batch_size` 32, 4 workers:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| 4 requests in flight (4 workers × 1 items) | 264.35 ms | 1,937 | 1.00× |
| 16 requests in flight (4 workers × 4 items) | 78.64 ms | 6,511 | 3.36× |
| 64 requests in flight (4 workers × 16 items) | 20.99 ms | 24,395 | 12.60× |

Requests in flight are `num_workers × items_in_parallel`, and throughput follows them: 264 ms
for an ideal 256 ms at 4 in flight, 20.8 ms for an ideal 16 ms at 64.

## 5. Results: Python against PyTorch

`bench/compare_torch.py`, 4 workers, `batch_size` 32. `w` is `num_workers`, `items` is
`items_in_parallel`; the last column divides by the best torch configuration of the workload.

### 5.1 CPython 3.14t (free-threaded, the default)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 31,730,292 | 0.01 ms | 6.02× |
| `in_memory` | **dataloader_rs** | w=4 | 10,937,206 | 0.03 ms | 2.07× |
| `in_memory` | torch | w=0 | 5,274,323 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 470,457 | 11.13 ms | 0.09× |
| `in_memory` | torch | w=4 persistent | 1,432,027 | 0.24 ms | 0.27× |
| `local_files` | **dataloader_rs** | w=0 | 58,495 | 0.57 ms | 1.14× |
| `local_files` | **dataloader_rs** | w=4 | 180,448 | 0.73 ms | 3.52× |
| `local_files` | torch | w=0 | 51,307 | 0.71 ms | 1.00× |
| `local_files` | torch | w=4 | 33,007 | 23.27 ms | 0.64× |
| `local_files` | torch | w=4 persistent | 43,244 | 5.73 ms | 0.84× |
| `image_augment` | **dataloader_rs** | w=0 | 23,576 | 1.43 ms | 1.05× |
| `image_augment` | **dataloader_rs** | w=4 | 58,845 | 2.22 ms | 2.61× |
| `image_augment` | torch | w=0 | 21,087 | 1.58 ms | 0.94× |
| `image_augment` | torch | w=4 | 15,675 | 36.22 ms | 0.70× |
| `image_augment` | torch | w=4 persistent | 22,546 | 10.29 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 323 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 396 | 323 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 5,006 | 20.79 ms | 3.19× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 389 | 334 ms | 0.25× |
| `remote` | torch | w=4 persistent | 396 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,568 | 326 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 26,372 | 1.22 ms | 0.34× |
| `python_cpu` | **dataloader_rs** | w=4 | 12,568 | 8.97 ms | 0.16× |
| `python_cpu` | torch | w=0 | 27,429 | 1.20 ms | 0.35× |
| `python_cpu` | torch | w=4 | 43,298 | 13.02 ms | 0.55× |
| `python_cpu` | torch | w=4 persistent | 78,242 | 2.89 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 27,115 | 1.19 ms | 0.34× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 92,757 | 1.33 ms | 1.16× |
| `python_cpu_local` | torch | w=0 | 25,202 | 1.30 ms | 0.32× |
| `python_cpu_local` | torch | w=4 | 42,108 | 13.60 ms | 0.53× |
| `python_cpu_local` | torch | w=4 persistent | 79,833 | 1.87 ms | 1.00× |

### 5.2 CPython 3.15t (free-threaded)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 29,052,788 | 0.01 ms | 6.61× |
| `in_memory` | **dataloader_rs** | w=4 | 11,596,870 | 0.04 ms | 2.64× |
| `in_memory` | torch | w=0 | 4,398,395 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 453,801 | 13.07 ms | 0.10× |
| `in_memory` | torch | w=4 persistent | 1,408,675 | 0.24 ms | 0.32× |
| `local_files` | **dataloader_rs** | w=0 | 56,083 | 0.58 ms | 0.99× |
| `local_files` | **dataloader_rs** | w=4 | 192,613 | 0.74 ms | 3.41× |
| `local_files` | torch | w=0 | 56,486 | 0.62 ms | 1.00× |
| `local_files` | torch | w=4 | 32,916 | 24.59 ms | 0.58× |
| `local_files` | torch | w=4 persistent | 50,065 | 4.34 ms | 0.89× |
| `image_augment` | **dataloader_rs** | w=0 | 24,017 | 1.35 ms | 1.05× |
| `image_augment` | **dataloader_rs** | w=4 | 56,971 | 2.10 ms | 2.48× |
| `image_augment` | torch | w=0 | 22,930 | 1.47 ms | 1.00× |
| `image_augment` | torch | w=4 | 15,591 | 38.76 ms | 0.68× |
| `image_augment` | torch | w=4 persistent | 22,918 | 10.32 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 323 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 395 | 322 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 4,978 | 20.80 ms | 3.17× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 389 | 335 ms | 0.25× |
| `remote` | torch | w=4 persistent | 395 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,569 | 326 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 25,838 | 1.23 ms | 0.35× |
| `python_cpu` | **dataloader_rs** | w=4 | 30,018 | 3.10 ms | 0.40× |
| `python_cpu` | torch | w=0 | 28,075 | 1.18 ms | 0.38× |
| `python_cpu` | torch | w=4 | 41,963 | 13.10 ms | 0.56× |
| `python_cpu` | torch | w=4 persistent | 74,317 | 2.28 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 28,637 | 1.14 ms | 0.36× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 95,361 | 1.72 ms | 1.19× |
| `python_cpu_local` | torch | w=0 | 28,037 | 1.19 ms | 0.35× |
| `python_cpu_local` | torch | w=4 | 46,104 | 13.79 ms | 0.58× |
| `python_cpu_local` | torch | w=4 persistent | 80,088 | 2.98 ms | 1.00× |

### 5.3 CPython 3.13 (GIL)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 29,271,935 | 0.02 ms | 4.92× |
| `in_memory` | **dataloader_rs** | w=4 | 5,638,138 | 0.05 ms | 0.95× |
| `in_memory` | torch | w=0 | 5,951,350 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 574,458 | 8.21 ms | 0.10× |
| `in_memory` | torch | w=4 persistent | 1,029,458 | 0.34 ms | 0.17× |
| `local_files` | **dataloader_rs** | w=0 | 54,258 | 0.63 ms | 1.00× |
| `local_files` | **dataloader_rs** | w=4 | 39,863 | 3.32 ms | 0.73× |
| `local_files` | torch | w=0 | 54,291 | 0.63 ms | 1.00× |
| `local_files` | torch | w=4 | 38,634 | 15.19 ms | 0.71× |
| `local_files` | torch | w=4 persistent | 47,633 | 4.58 ms | 0.88× |
| `image_augment` | **dataloader_rs** | w=0 | 23,703 | 1.42 ms | 1.08× |
| `image_augment` | **dataloader_rs** | w=4 | 33,493 | 4.16 ms | 1.52× |
| `image_augment` | torch | w=0 | 21,525 | 1.57 ms | 0.98× |
| `image_augment` | torch | w=4 | 17,578 | 23.71 ms | 0.80× |
| `image_augment` | torch | w=4 persistent | 21,970 | 10.45 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 322 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 396 | 322 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 6,088 | 21.42 ms | 3.87× |
| `remote` | torch | w=0 | 99 | 324 ms | 0.06× |
| `remote` | torch | w=4 | 390 | 334 ms | 0.25× |
| `remote` | torch | w=4 persistent | 395 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,573 | 325 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 29,043 | 1.15 ms | 0.36× |
| `python_cpu` | **dataloader_rs** | w=4 | 32,573 | 4.69 ms | 0.41× |
| `python_cpu` | torch | w=0 | 30,675 | 1.08 ms | 0.38× |
| `python_cpu` | torch | w=4 | 46,916 | 12.75 ms | 0.58× |
| `python_cpu` | torch | w=4 persistent | 80,246 | 2.77 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 29,015 | 1.12 ms | 0.35× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 32,014 | 5.17 ms | 0.39× |
| `python_cpu_local` | torch | w=0 | 28,627 | 1.14 ms | 0.35× |
| `python_cpu_local` | torch | w=4 | 46,957 | 12.66 ms | 0.57× |
| `python_cpu_local` | torch | w=4 persistent | 81,810 | 2.83 ms | 1.00× |

### 5.4 Reading the tables

- **`in_memory`**: without workers `dataloader_rs` streams 29–32 M items/s against torch's
  4.4–6.0 M, because the loop runs in Rust. With 4 workers both loaders are bound by handing
  batches over: `dataloader_rs` moves 5.6–12 M items/s, 5.4–8.2× torch's persistent workers
  (1.0–1.4 M).
- **`local_files`, `image_augment`**: on 3.14t and 3.15t, 4 workers reach 180–193 k and
  57–59 k items/s, 3.1–3.4× and 2.4–2.5× the sequential path. On 3.13 the GIL serialises these
  small NumPy calls: 4 workers are slower than none on `local_files` (40 k against 54 k) and
  1.4× faster on `image_augment`. torch's best configuration is no workers or persistent ones
  (51–57 k and 22–23 k): moving a batch between processes costs more than these small samples.
- **`remote`**: requests in flight decide throughput, and only `items_in_parallel` raises them
  without more processes: 4 workers × 16 items reach 5.0–6.1 k items/s, for an ideal 6.4 k.
  torch's first batch is always `batch_size` × 10 ms = 320 ms, because one process fetches a whole
  batch.
- **`python_cpu`**: threads lose. With the GIL they run one at a time. On 3.14t and 3.15t they run
  in parallel but contend on the reference counts of the shared vocabulary dict and word list:
  4 workers are slower than none on 3.14t.
- **`python_cpu_local`**: per-thread tables remove that contention: 4 workers give 3.3–3.4× on
  3.14t and 3.15t, 1.16–1.19× torch's persistent processes; that ratio spreads from about 0.94×
  to 1.23× between runs (section 8). On 3.13 the GIL serialises them.

## 6. Results: Python regression guards

`bench/test_bench_*.py` on 3.14t, the CI's interpreter, pinned to 4 cores; median per round.
These cases run on every pull request; their history is on the
[dashboard](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| file | case | median |
|---|---|---:|
| `test_bench_batch_size.py` | `test_batch_size_sequential[1]` | 0.55 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential[64]` | 0.14 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential[4096]` | 0.13 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[1]` | 1.11 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[64]` | 0.27 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[4096]` | 0.14 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential_getitems[64]` | 0.07 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential_getitems[4096]` | 0.06 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel_getitems[64]` | 0.05 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel_getitems[4096]` | 0.06 ms |
| `test_bench_first_batch.py` | `test_first_batch[0-1]` | 14.27 ms |
| `test_bench_first_batch.py` | `test_first_batch[1-1]` | 13.89 ms |
| `test_bench_first_batch.py` | `test_first_batch[0-4]` | 3.71 ms |
| `test_bench_first_batch.py` | `test_first_batch[1-4]` | 3.94 ms |
| `test_bench_prefetch.py` | `test_overlap_sequential` | 118.31 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[1]` | 68.72 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[2]` | 68.70 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[4]` | 68.62 ms |
| `test_bench_real_world.py` | `test_local_files[0]` | 17.09 ms |
| `test_bench_real_world.py` | `test_local_files[4]` | 6.16 ms |
| `test_bench_real_world.py` | `test_image_augment[0]` | 20.89 ms |
| `test_bench_real_world.py` | `test_image_augment[4]` | 10.33 ms |
| `test_bench_real_world.py` | `test_remote_threads` | 325.30 ms |
| `test_bench_real_world.py` | `test_remote_items_in_parallel` | 40.68 ms |
| `test_bench_sampler.py` | `test_sampler_sequential` | 0.28 ms |
| `test_bench_sampler.py` | `test_sampler_shuffle` | 0.29 ms |
| `test_bench_workers.py` | `test_num_workers[0]` | 118.20 ms |
| `test_bench_workers.py` | `test_num_workers[1]` | 113.91 ms |
| `test_bench_workers.py` | `test_num_workers[2]` | 58.27 ms |
| `test_bench_workers.py` | `test_num_workers[4]` | 37.55 ms |

The guards run with the allocator pinned, as in CI. The first-batch and overlap guards mirror
the Rust suites: 4 item lanes cut the first batch from 14 to 3.7 ms; with 2 workers, 16 batches
overlap with 4 ms steps (69 ms, against 118 ms sequentially, 64 ms of steps).
`test_batch_size_parallel[1]` shows the Python path's per-batch cost: 1.1 ms for 4 096
single-item batches, 0.3 µs each (about 45 µs before
[#53](https://github.com/AzHicham/dataloader-rs/pull/53), section 8).

## 7. Reading the results: choosing settings

- **`num_workers`**: 0 when a batch costs about as much as handing it over: well under a
  microsecond in Rust, a few microseconds through Python. Otherwise up to the number of cores for
  CPU-bound items; beyond that threads only compete for the cores.
- **`items_in_parallel`**: for datasets that wait (object stores, HTTP, databases): requests in
  flight are `num_workers × items_in_parallel`, and the pool has that many threads. Also for the
  first batch, and for epochs with fewer batches than workers.
- **`prefetch_depth`**: the default 2 is enough when loading keeps up with the step; a deeper
  prefetch only absorbs uneven batch times, at the cost of memory.
- **Python**: on the GIL build, threads help when the dataset releases the GIL (I/O, NumPy, most C
  extensions). For pure-Python preprocessing use free-threaded 3.14t or 3.15t and keep hot lookup
  tables per thread (`threading.local`). If that is not possible, torch's processes are faster.
- **Array samples on Linux**: start the process with the allocator settings of section 8, the
  ones the benchmarks run with: `GLIBC_TUNABLES` on 3.13 and 3.14t, `MIMALLOC_PURGE_DELAY=-1` on
  3.15t. A Rust program can use the same glibc setting, or another global allocator.

## 8. Caveats and known bottlenecks

- **The allocator and array samples.** With default settings, the allocator behind NumPy returns
  the memory of freed batches to the system and faults it back in for the next ones; with worker
  threads, those page faults and the system calls behind them serialise the workers. It is a
  process-wide setting, so it belongs to the application, not to the loader. The benchmarks pin
  it for both loaders (section 3.3), and it is what we recommend:

  ```bash
  # CPython 3.13 and 3.14t: NumPy allocates with glibc's malloc
  GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 python train.py
  # CPython 3.15t: NumPy's buffers come from CPython's mimalloc
  MIMALLOC_PURGE_DELAY=-1 python train.py
  ```

  What the defaults cost on the workloads of this page (`compare_torch.py --default-allocator`
  against the pinned run; 3.15t is not affected at these sizes, and its differences are noise):

  | workload | interpreter | `dataloader_rs`, 4 workers | best torch |
  |---|---|---:|---:|
  | `local_files` | 3.13 | 41.2 k → 39.9 k items/s | 47.0 k → 54.3 k |
  | `local_files` | 3.14t | 104.5 k → 180.4 k items/s | 38.8 k → 51.3 k |
  | `local_files` | 3.15t | 196.5 k → 192.6 k items/s | 53.5 k → 56.5 k |
  | `image_augment` | 3.13 | 25.9 k → 33.5 k items/s | 21.5 k → 22.0 k |
  | `image_augment` | 3.14t | 28.7 k → 58.8 k items/s | 21.0 k → 22.5 k |
  | `image_augment` | 3.15t | 61.0 k → 57.0 k items/s | 20.6 k → 22.9 k |

  *glibc (3.13, 3.14t).* Both thresholds have to move: setting either one turns off glibc's
  dynamic adjustment of the other, and one alone is slower than the defaults. Measured with the
  earlier, larger samples (512–588 KB, batches of 16–18 MiB) on 3.14t with 4 workers:

  | `malloc` settings | `local_files` | `image_augment` | RSS after the epochs |
  |---|---:|---:|---:|
  | glibc defaults | 9.8–10.8 k items/s | 5.2–6.8 k | 416–425 MB |
  | trim threshold 1 GiB only | 3.7–3.8 k | 2.8–2.9 k | 262 MB |
  | mmap threshold 32 MiB only | 5.3–6.6 k | 4.7–4.8 k | 338–355 MB |
  | both, trim 64 MiB (recommended) | 14.3–14.4 k | 11.7 k | 503–517 MB |
  | both, trim 1 GiB | 14.0–14.5 k | 11.8 k | 508–523 MB |
  | jemalloc (`LD_PRELOAD`) | 8.0 k | 5.9 k | 340 MB |

  *mimalloc (3.15t).* On 3.15t a NumPy array does not come from glibc at all (a 16 MB array leaves
  glibc's heap untouched), so `GLIBC_TUNABLES` has no effect there. The consumer thread frees
  each batch a worker allocated: for blocks of 16 MiB and more, mimalloc resets the whole block
  on such a cross-thread free (`MADV_FREE`), and it purges freed segments 10 ms later
  (`MADV_DONTNEED`). With the earlier 18.4 MiB batches of `image_augment`, purges returned 1.3 GB
  per run to the system against 0.5 GB with 16.1 MiB batches, and 4 workers dropped from 11–12 k
  to 8.5–9.8 k items/s at that boundary; `MIMALLOC_PURGE_DELAY=-1` brought them back to
  10.6–11.1 k. The reset itself has no option, so the benchmarks keep batches far below 16 MiB.

  The cost is memory kept for reuse: about 100 MB more with glibc, 80–90 MB with mimalloc, with
  the earlier samples. Pure-Python work is unaffected: its objects come from CPython's own
  allocator either way.
- **Per-batch cost of the Python parallel path.** About 0.3–4 µs per batch on 3.14t, against
  25–45 µs before [#53](https://github.com/AzHicham/dataloader-rs/pull/53): each worker thread
  keeps one Python thread state, where every attach used to create and destroy one. What remains
  is the hand-off itself and, without `collate_fn`, building the batch `list` on the consumer
  thread, whose objects belong to the workers: a few µs per 32 items on free-threaded builds.
- **One machine, one session.** A laptop with turbo clocks: ratios measured in the same session
  are more reliable than absolute numbers, and a 4-vCPU runner is slower per core. Some
  workloads also spread from run to run: with default settings, `python_cpu_local` with 4
  workers measured 71–95 k items/s for `dataloader_rs` and 68–85 k for torch's persistent workers
  across runs of the same code, a 0.94–1.23× ratio.
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
GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 \
MIMALLOC_PURGE_DELAY=-1 \
uv run python -X gil=0 -m pytest bench/ -m bench --benchmark-enable --benchmark-warmup=on \
  --benchmark-min-rounds=5
```

The variables pin the allocator as CI does; pytest's header says whether it is pinned.

### 9.3 Python against PyTorch

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
taskset -c 0,2,4,6 uv run python -X gil=0 bench/compare_torch.py --workers 4 --repeats 5 --json out.json
uv run --python 3.13 python bench/compare_torch.py     # with the GIL
```

`--workloads` runs a subset; without torch installed, only `dataloader_rs` runs. The script
pins the allocator settings of section 8 for both loaders by re-executing itself with them;
`--default-allocator` measures the defaults instead (section 8's table).

### 9.4 Comparing two versions

Run both builds alternately in the same session (A, B, A, B, …) and compare medians: the machine
drifts more between sessions than between two interleaved runs. For Criterion, delete
`target/criterion` between builds so it does not compare against the other build's baseline.
