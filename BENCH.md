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
| `in_memory` | **6.20×** | **5.99×** | **5.66×** |
| `local_files` | **1.16×** | **1.23×** | **1.77×** |
| `image_augment` | **1.09×** | 1.02× | **1.50×** |
| `remote` | **3.16×** | **3.91×** | **3.91×** |
| `python_cpu` | 0.43× | 0.35× | 0.42× |
| `python_cpu_local` | 0.43× | 1.04× | 0.94× |

- **Loader overhead** (`in_memory`, items that cost nothing): **5.7–6.2× faster**. Without
  workers the whole loop runs in Rust; torch's runs in Python. With 4 workers, `dataloader_rs`
  hands over 5.5–12 M items/s against 1.0–1.3 M for torch's persistent workers.
- **Network-bound items** (`remote`, 10 ms per item): **3.2–3.9× faster**. With
  `items_in_parallel=16`, 4 workers keep 64 requests in flight. torch parallelises whole batches,
  one process each: this 16-batch epoch keeps at most 16 of its 64 processes busy, and its first
  batch always waits for `batch_size` × 10 ms.
- **Array samples** (`local_files`, `image_augment`): **1.0–1.8× faster** with glibc's default
  `malloc` settings, where glibc returns the samples' memory to the system after every batch and
  faults it back in, which serialises worker threads. With the two thresholds raised for the
  process (`GLIBC_TUNABLES`, section 8), both loaders measured with it: **1.5–1.9× faster** on
  every interpreter (table below). torch's worker processes do not scale on these workloads
  either way.
- **Pure-Python CPU work** (`python_cpu*`): with a shared vocabulary, **torch is faster**
  (2.3–2.9×): threads are serialised by the GIL on 3.13, and contend on the reference counts of
  the shared objects on 3.14t and 3.15t. With per-thread tables (`python_cpu_local`),
  `dataloader_rs` is on par to 1.2× faster on 3.14t and 3.15t (0.94–1.04× in this run; medians of
  three interleaved runs: 1.18× and 1.10×); on 3.13 torch stays 2.3× ahead.
- **First batch**: `dataloader_rs` starts an epoch in milliseconds. torch's fresh workers take
  13–335 ms: process start-up, then a whole batch per worker.

Array samples with `GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432`
set for the whole run, so for both loaders (ratio, then `dataloader_rs` / torch items per second):

| workload | 3.13 (GIL) | 3.14t | 3.15t |
|---|---:|---:|---:|
| `local_files` | **1.48×** (13.0 k / 8.8 k) | **1.52×** (13.5 k / 8.9 k) | **1.45×** (13.9 k / 9.6 k) |
| `image_augment` | **1.65×** (10.3 k / 6.2 k) | **1.87×** (11.2 k / 6.0 k) | **1.59×** (10.5 k / 6.6 k) |

Latency to the first batch on 3.14t, workers already started for persistent torch (median of 5
epochs):

| workload | dataloader_rs, 4 workers | torch, 4 fresh workers | torch, 4 persistent workers |
|---|---:|---:|---:|
| `in_memory` | 0.02 ms | 12.8 ms | 0.31 ms |
| `local_files` | 10.9 ms | 53.3 ms | 24.7 ms |
| `image_augment` | 10.4 ms | 114 ms | 21.3 ms |
| `remote` | 20.8 ms (items=16) | 335 ms | 323 ms |
| `python_cpu` | 16.8 ms | 14.0 ms | 2.2 ms |
| `python_cpu_local` | 1.5 ms | 13.7 ms | 3.2 ms |

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
| System | Ubuntu, Linux 6.14, glibc 2.41 with default `malloc` settings unless a table says otherwise (section 8), on AC power |
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

1 024 files of 64 KB, read and normalised to `f32`, `batch_size` 32:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| num_workers = 0 | 65.05 ms | 15,741 | 1.00× |
| num_workers = 1 | 18.62 ms | 54,984 | 3.49× |
| num_workers = 2 | 16.47 ms | 62,174 | 3.95× |
| num_workers = 4 | 18.71 ms | 54,743 | 3.48× |

The allocator decides this one. Each item allocates a 256 KB `Vec<f32>`. With default settings,
glibc gives the memory of every dropped batch back to the system and the next batch faults it in
again; without workers that costs 3.6×. With the thresholds section 8 recommends,
`GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432`, the
same epoch takes 18.2, 18.2, 10.2 and 12.6 ms with 0, 1, 2 and 4 workers; a Rust program can
also pick another global allocator. Beyond 2 workers the epoch
stops scaling even then: it moves ≈ 450 MB through memory, and 4 workers are slower than 2 (not
investigated yet).

512 items with a 2 ms blocking round trip, `batch_size` 32, 4 workers:

| case | time / epoch | items / s | vs first |
|---|---:|---:|---:|
| 4 requests in flight (4 workers × 1 items) | 264.30 ms | 1,937 | 1.00× |
| 16 requests in flight (4 workers × 4 items) | 79.62 ms | 6,431 | 3.32× |
| 64 requests in flight (4 workers × 16 items) | 20.84 ms | 24,573 | 12.69× |

Requests in flight are `num_workers × items_in_parallel`, and throughput follows them: 264 ms
for an ideal 256 ms at 4 in flight, 20.8 ms for an ideal 16 ms at 64.

## 5. Results: Python against PyTorch

`bench/compare_torch.py`, 4 workers, `batch_size` 32. `w` is `num_workers`, `items` is
`items_in_parallel`; the last column divides by the best torch configuration of the workload.

### 5.1 CPython 3.14t (free-threaded)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 29,115,826 | 0.01 ms | 5.99× |
| `in_memory` | **dataloader_rs** | w=4 | 11,323,738 | 0.02 ms | 2.33× |
| `in_memory` | torch | w=0 | 4,860,433 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 427,069 | 12.82 ms | 0.09× |
| `in_memory` | torch | w=4 persistent | 1,262,378 | 0.31 ms | 0.26× |
| `local_files` | **dataloader_rs** | w=0 | 7,645 | 4.24 ms | 1.23× |
| `local_files` | **dataloader_rs** | w=4 | 7,240 | 10.91 ms | 1.16× |
| `local_files` | torch | w=0 | 6,223 | 4.28 ms | 1.00× |
| `local_files` | torch | w=4 | 5,201 | 53.27 ms | 0.84× |
| `local_files` | torch | w=4 persistent | 5,550 | 24.66 ms | 0.89× |
| `image_augment` | **dataloader_rs** | w=0 | 5,636 | 6.20 ms | 1.02× |
| `image_augment` | **dataloader_rs** | w=4 | 5,129 | 10.38 ms | 0.92× |
| `image_augment` | torch | w=0 | 5,485 | 5.75 ms | 0.99× |
| `image_augment` | torch | w=4 | 3,768 | 114 ms | 0.68× |
| `image_augment` | torch | w=4 persistent | 5,551 | 21.27 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 323 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 397 | 322 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 6,116 | 20.84 ms | 3.91× |
| `remote` | torch | w=0 | 99 | 322 ms | 0.06× |
| `remote` | torch | w=4 | 390 | 335 ms | 0.25× |
| `remote` | torch | w=4 persistent | 396 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,566 | 327 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 25,242 | 1.27 ms | 0.35× |
| `python_cpu` | **dataloader_rs** | w=4 | 11,712 | 16.75 ms | 0.16× |
| `python_cpu` | torch | w=0 | 24,290 | 1.40 ms | 0.34× |
| `python_cpu` | torch | w=4 | 39,833 | 14.03 ms | 0.55× |
| `python_cpu` | torch | w=4 persistent | 72,052 | 2.15 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 23,980 | 1.35 ms | 0.35× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 71,273 | 1.50 ms | 1.04× |
| `python_cpu_local` | torch | w=0 | 26,685 | 1.32 ms | 0.39× |
| `python_cpu_local` | torch | w=4 | 41,629 | 13.73 ms | 0.61× |
| `python_cpu_local` | torch | w=4 persistent | 68,603 | 3.25 ms | 1.00× |

### 5.2 CPython 3.13 (GIL)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 25,449,016 | 0.02 ms | 6.20× |
| `in_memory` | **dataloader_rs** | w=4 | 5,483,903 | 0.05 ms | 1.34× |
| `in_memory` | torch | w=0 | 4,105,343 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 527,901 | 8.05 ms | 0.13× |
| `in_memory` | torch | w=4 persistent | 991,236 | 0.30 ms | 0.24× |
| `local_files` | **dataloader_rs** | w=0 | 7,591 | 4.27 ms | 1.16× |
| `local_files` | **dataloader_rs** | w=4 | 5,561 | 25.75 ms | 0.85× |
| `local_files` | torch | w=0 | 6,134 | 4.04 ms | 0.94× |
| `local_files` | torch | w=4 | 6,244 | 47.46 ms | 0.96× |
| `local_files` | torch | w=4 persistent | 6,529 | 29.12 ms | 1.00× |
| `image_augment` | **dataloader_rs** | w=0 | 6,015 | 5.39 ms | 1.09× |
| `image_augment` | **dataloader_rs** | w=4 | 4,761 | 11.83 ms | 0.86× |
| `image_augment` | torch | w=0 | 4,767 | 6.51 ms | 0.86× |
| `image_augment` | torch | w=4 | 4,762 | 48.95 ms | 0.86× |
| `image_augment` | torch | w=4 persistent | 5,535 | 24.61 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 322 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 397 | 323 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 4,971 | 21.68 ms | 3.16× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 390 | 337 ms | 0.25× |
| `remote` | torch | w=4 persistent | 397 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,572 | 325 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 28,044 | 1.16 ms | 0.37× |
| `python_cpu` | **dataloader_rs** | w=4 | 32,370 | 6.52 ms | 0.43× |
| `python_cpu` | torch | w=0 | 27,359 | 1.23 ms | 0.37× |
| `python_cpu` | torch | w=4 | 43,334 | 15.60 ms | 0.58× |
| `python_cpu` | torch | w=4 persistent | 74,891 | 2.73 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 30,350 | 1.07 ms | 0.41× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 31,572 | 5.78 ms | 0.43× |
| `python_cpu_local` | torch | w=0 | 27,162 | 1.31 ms | 0.37× |
| `python_cpu_local` | torch | w=4 | 40,614 | 15.42 ms | 0.55× |
| `python_cpu_local` | torch | w=4 persistent | 73,194 | 2.67 ms | 1.00× |

### 5.3 CPython 3.15t (free-threaded)

| workload | loader | config | items / s | first batch | throughput vs best torch |
|---|---|---|---:|---:|---:|
| `in_memory` | **dataloader_rs** | w=0 | 28,770,814 | 0.01 ms | 5.66× |
| `in_memory` | **dataloader_rs** | w=4 | 11,784,481 | 0.03 ms | 2.32× |
| `in_memory` | torch | w=0 | 5,079,617 | 0.02 ms | 1.00× |
| `in_memory` | torch | w=4 | 404,810 | 12.67 ms | 0.08× |
| `in_memory` | torch | w=4 persistent | 1,335,706 | 0.33 ms | 0.26× |
| `local_files` | **dataloader_rs** | w=0 | 7,229 | 4.45 ms | 1.05× |
| `local_files` | **dataloader_rs** | w=4 | 12,178 | 8.79 ms | 1.77× |
| `local_files` | torch | w=0 | 6,864 | 3.75 ms | 1.00× |
| `local_files` | torch | w=4 | 6,074 | 56.95 ms | 0.88× |
| `local_files` | torch | w=4 persistent | 6,570 | 24.82 ms | 0.96× |
| `image_augment` | **dataloader_rs** | w=0 | 5,990 | 5.63 ms | 1.07× |
| `image_augment` | **dataloader_rs** | w=4 | 8,416 | 17.55 ms | 1.50× |
| `image_augment` | torch | w=0 | 5,084 | 6.31 ms | 0.90× |
| `image_augment` | torch | w=4 | 3,688 | 113 ms | 0.66× |
| `image_augment` | torch | w=4 persistent | 5,619 | 24.89 ms | 1.00× |
| `remote` | **dataloader_rs** | w=0 | 99 | 322 ms | 0.06× |
| `remote` | **dataloader_rs** | w=4 | 397 | 322 ms | 0.25× |
| `remote` | **dataloader_rs** | w=4 items=16 | 6,116 | 20.75 ms | 3.91× |
| `remote` | torch | w=0 | 99 | 323 ms | 0.06× |
| `remote` | torch | w=4 | 390 | 336 ms | 0.25× |
| `remote` | torch | w=4 persistent | 396 | 323 ms | 0.25× |
| `remote` | torch | w=64 persistent | 1,563 | 327 ms | 1.00× |
| `python_cpu` | **dataloader_rs** | w=0 | 26,150 | 1.25 ms | 0.42× |
| `python_cpu` | **dataloader_rs** | w=4 | 22,969 | 6.91 ms | 0.37× |
| `python_cpu` | torch | w=0 | 23,409 | 1.43 ms | 0.37× |
| `python_cpu` | torch | w=4 | 31,351 | 18.77 ms | 0.50× |
| `python_cpu` | torch | w=4 persistent | 62,857 | 3.21 ms | 1.00× |
| `python_cpu_local` | **dataloader_rs** | w=0 | 27,353 | 1.21 ms | 0.34× |
| `python_cpu_local` | **dataloader_rs** | w=4 | 75,090 | 1.43 ms | 0.94× |
| `python_cpu_local` | torch | w=0 | 23,965 | 1.43 ms | 0.30× |
| `python_cpu_local` | torch | w=4 | 41,355 | 13.57 ms | 0.52× |
| `python_cpu_local` | torch | w=4 persistent | 79,889 | 2.70 ms | 1.00× |

### 5.4 Reading the tables

- **`in_memory`**: without workers `dataloader_rs` streams 25–29 M items/s against torch's
  4.1–5.1 M, because the loop runs in Rust. With 4 workers both loaders are bound by handing
  batches over: `dataloader_rs` moves 5.5–12 M items/s, 5.6–9× torch's persistent workers
  (1.0–1.3 M).
- **`local_files`, `image_augment`**: with glibc's defaults, 4 workers reach 5.6–7.2 k and
  4.8–5.1 k items/s on 3.13 and 3.14t, less than no workers (7.6 k and 5.6–6.0 k), and 12.2 k
  and 8.4 k on 3.15t.
  With the thresholds raised (summary), 13–14 k and 10–11 k on all three. torch's best
  configuration is no workers at all (6.2–9.6 k and 5.5–6.6 k): its worker processes do not
  scale on these workloads.
- **`remote`**: requests in flight decide throughput, and only `items_in_parallel` raises them
  without more processes: 4 workers × 16 items reach 5.0–6.1 k items/s, for an ideal 6.4 k.
  torch's first batch is always `batch_size` × 10 ms = 320 ms, because one process fetches a whole
  batch.
- **`python_cpu`**: threads lose. With the GIL they run one at a time. On 3.14t and 3.15t they run
  in parallel but contend on the reference counts of the shared vocabulary dict and word list:
  4 workers are slower than none on 3.14t.
- **`python_cpu_local`**: per-thread tables remove that contention: 4 workers give 2.7–3.0× on
  3.14t and 3.15t in this run, on par with torch's persistent processes, and 1.10–1.18× ahead of
  them over three interleaved runs (section 8). On 3.13 the GIL serialises them.

## 6. Results: Python regression guards

`bench/test_bench_*.py` on 3.14t, the CI's interpreter, pinned to 4 cores; median per round.
These cases run on every pull request; their history is on the
[dashboard](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| file | case | median |
|---|---|---:|
| `test_bench_batch_size.py` | `test_batch_size_sequential[1]` | 0.57 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential[64]` | 0.15 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential[4096]` | 0.13 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[1]` | 1.14 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[64]` | 0.27 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel[4096]` | 0.16 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential_getitems[64]` | 0.07 ms |
| `test_bench_batch_size.py` | `test_batch_size_sequential_getitems[4096]` | 0.05 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel_getitems[64]` | 0.08 ms |
| `test_bench_batch_size.py` | `test_batch_size_parallel_getitems[4096]` | 0.07 ms |
| `test_bench_first_batch.py` | `test_first_batch[0-1]` | 14.76 ms |
| `test_bench_first_batch.py` | `test_first_batch[1-1]` | 14.77 ms |
| `test_bench_first_batch.py` | `test_first_batch[0-4]` | 4.03 ms |
| `test_bench_first_batch.py` | `test_first_batch[1-4]` | 4.11 ms |
| `test_bench_prefetch.py` | `test_overlap_sequential` | 119.67 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[1]` | 68.74 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[2]` | 68.94 ms |
| `test_bench_prefetch.py` | `test_overlap_prefetch_depth[4]` | 68.59 ms |
| `test_bench_real_world.py` | `test_local_files[0]` | 87.86 ms |
| `test_bench_real_world.py` | `test_local_files[4]` | 62.46 ms |
| `test_bench_real_world.py` | `test_image_augment[0]` | 57.78 ms |
| `test_bench_real_world.py` | `test_image_augment[4]` | 32.89 ms |
| `test_bench_real_world.py` | `test_remote_threads` | 322.49 ms |
| `test_bench_real_world.py` | `test_remote_items_in_parallel` | 40.81 ms |
| `test_bench_sampler.py` | `test_sampler_sequential` | 0.29 ms |
| `test_bench_sampler.py` | `test_sampler_shuffle` | 0.29 ms |
| `test_bench_workers.py` | `test_num_workers[0]` | 119.48 ms |
| `test_bench_workers.py` | `test_num_workers[1]` | 120.25 ms |
| `test_bench_workers.py` | `test_num_workers[2]` | 65.52 ms |
| `test_bench_workers.py` | `test_num_workers[4]` | 37.28 ms |

The first-batch and overlap guards mirror the Rust suites: 4 item lanes cut the first batch from
15 to 4 ms; with 2 workers, 16 batches overlap with 4 ms steps (69 ms, against 120 ms
sequentially, 64 ms of steps). `test_batch_size_parallel[1]` shows the Python path's per-batch
cost: 1.1 ms for 4 096 single-item batches, 0.3 µs each (45 µs before
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
- **Array samples on Linux**: start the process with glibc's `malloc` thresholds raised,
  `GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432`
  (section 8). A Rust program can use the same setting, or another global allocator.

## 8. Caveats and known bottlenecks

- **glibc `malloc` and array samples.** NumPy allocates sample and batch buffers with `malloc`.
  With default settings glibc returns freed memory to the system and faults it back in on the next
  allocation: a plain Python loop over `local_files` takes ≈ 147 page faults per item, and none
  with raised thresholds. With worker threads those faults and the system calls behind them
  serialise the workers. The fix is a process-wide setting, so it belongs to the application, not
  to the loader; start the process with:

  ```bash
  GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 python train.py
  ```

  Both thresholds have to move: setting either one turns off glibc's dynamic adjustment of the
  other, and one alone is slower than the defaults. On 3.14t with 4 workers:

  | `malloc` settings | `local_files` | `image_augment` | RSS after the epochs |
  |---|---:|---:|---:|
  | glibc defaults | 9.8–10.8 k items/s | 5.2–6.8 k | 416–425 MB |
  | trim threshold 1 GiB only | 3.7–3.8 k | 2.8–2.9 k | 262 MB |
  | mmap threshold 32 MiB only | 5.3–6.6 k | 4.7–4.8 k | 338–355 MB |
  | both, trim 64 MiB (recommended) | 14.3–14.4 k | 11.7 k | 503–517 MB |
  | both, trim 1 GiB | 14.0–14.5 k | 11.8 k | 508–523 MB |
  | jemalloc (`LD_PRELOAD`) | 8.0 k | 5.9 k | 340 MB |

  The cost is memory kept for reuse, about 100 MB more here. Pure-Python work is unaffected: its
  objects come from CPython's own allocator.
- **Per-batch cost of the Python parallel path.** About 0.3–4 µs per batch on 3.14t, against
  25–45 µs before [#53](https://github.com/AzHicham/dataloader-rs/pull/53): each worker thread
  keeps one Python thread state, where every attach used to create and destroy one. What remains
  is the hand-off itself and, without `collate_fn`, building the batch `list` on the consumer
  thread, whose objects belong to the workers: a few µs per 32 items on free-threaded builds.
- **`image_augment` on 3.15t** does not scale like on 3.14t (7.2 k against 12.3 k items/s with 4
  workers). Not investigated yet.
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
uv run python -X gil=0 -m pytest bench/ -m bench --benchmark-enable --benchmark-warmup=on \
  --benchmark-min-rounds=5
```

### 9.3 Python against PyTorch

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
taskset -c 0,2,4,6 uv run python -X gil=0 bench/compare_torch.py --workers 4 --repeats 5 --json out.json
uv run --python 3.13 python bench/compare_torch.py     # with the GIL
```

`--workloads` runs a subset; without torch installed, only `dataloader_rs` runs. The summary's
second table sets the recommended `malloc` thresholds for the whole run:

```bash
GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 \
  taskset -c 0,2,4,6 uv run python -X gil=0 bench/compare_torch.py --workloads local_files image_augment
```

### 9.4 Comparing two versions

Run both builds alternately in the same session (A, B, A, B, …) and compare medians: the machine
drifts more between sessions than between two interleaved runs. For Criterion, delete
`target/criterion` between builds so it does not compare against the other build's baseline.
