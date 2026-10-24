# Benchmarks

What `dataloader-rs` is measured on, how, and how it compares with
`torch.utils.data.DataLoader`. Every number below was measured on one laptop (section 2); the
CI re-runs the regression suites on every pull request and plots them per commit on the
dashboards for [Rust](https://azhicham.github.io/dataloader-rs/benchmarks/rust/) and
[Python 3.14t](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

Two reference runs:

| Run | Code | Python | Sections |
|---|---|---|---|
| Synthetic suites | v0.1.0 (April 2026) | CPython 3.13.3 (GIL) and 3.13.3t (`-X gil=0`), torch 2.11.0+cpu | 4, 5 |
| Real-world workloads | `dev` (October 2026) | CPython 3.13.3 (GIL), torch 2.14.1+cpu, numpy 2.5.4 | 6 |

Contents:

1. [Summary](#1-summary)
2. [Setup](#2-setup)
3. [Method: what is measured and how](#3-method-what-is-measured-and-how)
4. [Results: Rust](#4-results-rust)
5. [Results: Python against PyTorch, synthetic](#5-results-python-against-pytorch-synthetic)
6. [Results: real-world workloads](#6-results-real-world-workloads)
7. [Reading the results](#7-reading-the-results)
8. [Caveats](#8-caveats)
9. [Running it again](#9-running-it-again)

## 1. Summary

Python, `dataloader_rs.PyDataloader` against `torch.utils.data.DataLoader`:

| Workload | Result |
|---|---|
| Array samples (read + normalise a file, augment an image), 4 workers | **≈ 7× faster** (6.8–7.0×) |
| Sequential path (`num_workers=0`), any batch size | **1.5–61× faster** |
| Parallel, dataset that releases the GIL (SHA-256 of 1 MB) | 2–6 % faster |
| Network-bound (10 ms per sample, blocking), 16 workers | 1.2× faster; both capped by workers |
| Pure-Python CPU work holding the GIL, regular CPython | **torch 1.6× faster** (persistent processes) |
| Parallel, trivial `__getitem__`, medium batch sizes | mixed: torch ahead at `batch_size` 128–1024 on GIL CPython |

Rust (Criterion): the direct path streams **≈ 1 G items/s** at `batch_size` 4096 (pure loader
overhead); 8 worker threads give 3.1× on a CPU-bound dataset and item-level parallelism 2.9×
with 8 rayon threads; the random sampler costs 1.6–2× the sequential one.

The pattern: threads win wherever samples are large or the dataset releases the GIL, because
nothing is pickled between processes; processes win on GIL-bound pure-Python work on regular
CPython.

## 2. Setup

| | |
|---|---|
| Machine | laptop, Intel Core i9-13900H (14 cores / 20 threads), Linux x86_64, on AC power |
| Rust | stable, `opt-level = 3`, `lto = "thin"`, `codegen-units = 1` (bench profile) |
| Python | see the reference runs above; the extension built with `maturin develop --release` |
| PyTorch | CPU wheels; `num_workers` worker processes, default start method |

## 3. Method: what is measured and how

### 3.1 Three families of benchmarks

| Family | Files | Question | Where it runs |
|---|---|---|---|
| Rust loader overhead | `benches/*.rs` (Criterion) | what each setting costs or buys, without Python | locally; CI on every PR (regression tracking) |
| Python regression guards | `bench/test_bench_*.py` (pytest-benchmark) | did the Python path get slower? | CI on every PR, free-threaded 3.14t |
| Python against PyTorch | `bench/bench_*.py` (own harness) | how do we compare with `torch.utils.data.DataLoader`? | locally, on demand |

Every file is listed with its parameters in [benches/README.md](benches/README.md) and
[bench/README.md](bench/README.md).

### 3.2 Datasets

The synthetic datasets isolate one cost at a time; the real-world ones run the per-sample work of
actual datasets.

| Dataset | Per-sample work | Isolates |
|---|---|---|
| `InMemoryDs` (Rust, Python) | returns the index | loader, sampler and collation overhead |
| `LightCpuDs` (Rust) | 1 000 LCG iterations (µs) | pipeline fill: prefetch depth |
| `HeavyCpuDs` (Rust) | 50 000 LCG iterations (≈ 1 ms), 256 bytes out | parallel scaling |
| `CpuBoundDs` (Python) | SHA-256 of 1 MB (≈ 1 ms), releases the GIL | parallel scaling through Python |
| `local_files` | read a 128 KB file (64 KB in Rust), normalise to float32 | I/O fan-out |
| `image_augment` | random crop 224, flip, normalise a decoded 3×256×256 image (NumPy) | GIL-releasing CPU work |
| `remote` | 10 ms (2 ms in Rust) blocking sleep | network latency |
| `python_cpu` | pure-Python tokenisation | GIL-bound CPU work |

### 3.3 What a measurement is

- **Rust**: Criterion's default statistics (warm-up, then samples until the estimate is stable;
  the tables report the point estimate). One iteration is a **whole epoch**, iterator creation
  included; the loader is built once, outside the timed loop.
- **Python against PyTorch**: `time.perf_counter` around a whole epoch, iterator creation
  included (that is where torch starts its worker processes), after warm-up epochs; the tables
  report the **median** (2 warm-up + 10 timed epochs for the synthetic sweeps, 1 + 5 for the
  real-world ones) as items per second.
- **Same work on both sides**: the real-world workloads are written once as mixins
  (`bench/real_world.py`) combined with `PyDataset` and with `torch.utils.data.Dataset`, so only
  the loading differs. Both loaders get the same `batch_size`, `num_workers` and `collate_fn`.
- **torch, fresh and persistent**: torch is run with its default fresh worker processes per epoch
  and with `persistent_workers=True`; tables say which.
- **Regression tracking (CI)**: Criterion's bencher output and pytest-benchmark's JSON go to
  [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark), which
  stores one point per commit on `dev` and comments on a pull request when a benchmark is more
  than **1.5×** slower than the last `dev` result.

## 4. Results: Rust

Reference run: synthetic suites, v0.1.0. Rust measures the loader alone, without Python.

### 4.1 `batch_size` — sequential path (InMemoryDs, 4 096 items, num_workers=0)

The sequential path processes batches inline in `Iterator::next` with no threads.
`InMemoryDs::get` is a trivial `Ok(index)` so this measures pure iterator + sampler
overhead.


| batch_size | time / epoch | items / s   |
| ---------- | ------------ | ----------- |
| 1          | 107 µs       | 38 M/s      |
| 8          | 26 µs        | 157 M/s     |
| 32         | 14.6 µs      | 281 M/s     |
| 128        | 9.4 µs       | 434 M/s     |
| 512        | 6.2 µs       | 661 M/s     |
| 1 024      | 4.6 µs       | 896 M/s     |
| 4 096      | 4.1 µs       | **1.0 G/s** |


Per-batch overhead (Vec allocation + sampler slice) is amortized at larger batch
sizes. At `bs=4096` the entire epoch is one batch — cost is dominated by a single
`Vec::with_capacity` call.

### 4.2 `batch_size` — parallel path (InMemoryDs, 4 096 items, num_workers=4, prefetch=16)


| batch_size | time / epoch |
| ---------- | ------------ |
| 1          | 948 µs       |
| 8          | 181 µs       |
| 32         | 105 µs       |
| 128        | 76.8 µs      |
| 512        | 69.4 µs      |
| 1 024      | 67.7 µs      |
| 4 096      | 68.5 µs      |


The parallel path plateaus at ~68–77 µs — this is the crossbeam channel roundtrip
floor (send + reorder buffer lookup). With `InMemoryDs` items are free so the
channel is the bottleneck.

### 4.3 `inter_workers` — CPU-bound dataset (HeavyCpuDs, N=256, bs=16, prefetch=32)


| num_workers | time / epoch | speedup vs 0     |
| ----------- | ------------ | ---------------- |
| 0           | 1.53 ms      | 1×               |
| 1           | 2.82 ms      | **0.54× slower** |
| 2           | 1.53 ms      | 1×               |
| 4           | 832 µs       | 1.8×             |
| 8           | 494 µs       | 3.1×             |


**1 worker is slower than sequential**: channel + thread overhead with no
parallelism gain. Meaningful speedup starts at 2 workers. Scaling to 8 workers
gives 3.1× on a 20-thread machine — sub-linear because `HeavyCpuDs` is CPU-bound
and the scheduler saturates physical cores.

### 4.4 `intra_workers` — rayon intra-batch parallelism (HeavyCpuDs, N=128, bs=64, inter=2)


| intra_workers     | time / epoch | speedup vs 0     |
| ----------------- | ------------ | ---------------- |
| 0                 | 823 µs       | 1×               |
| 1                 | 1.00 ms      | **0.82× slower** |
| 2                 | 600 µs       | 1.4×             |
| 4                 | 405 µs       | 2.0×             |
| 8                 | 281 µs       | 2.9×             |
| inter=4 + intra=4 | 410 µs       | —                |


Rayon intra-batch scales well. `inter=4 + intra=4` matches `intra=4` alone — the
cores are saturated and adding both dimensions adds context-switch pressure without
further throughput gain.

### 4.5 `prefetch_depth` (N=128, bs=8, 4 workers)


| depth | time / epoch |
| ----- | ------------ |
| 1     | 94.9 µs      |
| 2     | 95.1 µs      |
| 4     | 89.1 µs      |
| 8     | 89.2 µs      |
| 16    | 89.5 µs      |


Negligible effect for CPU-bound datasets — workers drain the channel faster than
the consumer blocks. Depth 4 is sufficient; going higher wastes memory without
throughput gain. **Recommended: 4–8.**

### 4.6 `sampler` (InMemoryDs, various N)


| N       | sequential | random  | ratio |
| ------- | ---------- | ------- | ----- |
| 1 000   | 1.81 µs    | 3.67 µs | 2.0×  |
| 10 000  | 25.5 µs    | 39.7 µs | 1.6×  |
| 100 000 | 237 µs     | 382 µs  | 1.6×  |


The random sampler (inside-out Fisher-Yates with `fastrand` wyrand) is consistently
~1.6–2× slower than the sequential sampler. Both scale O(N).

## 5. Results: Python against PyTorch, synthetic

Reference run: synthetic suites, v0.1.0, CPython 3.13 and free-threaded 3.13t, torch 2.11.
Free-threaded 3.13t is no longer supported (PyO3 requires 3.14t); the 3.13t columns show what
removing the GIL does, and 3.14t behaves the same way.

### 5.1 Why the GIL matters here

- **Sequential path (`num_workers=0`):** We call `__getitem__` directly in the
calling thread. Minimal overhead in both modes.
- **Parallel path, regular CPython:** Our worker *threads* serialize through the
GIL when calling Python. Torch spawns separate *processes* that bypass the GIL.
Torch wins on pure-Python datasets.
- **Parallel path, GIL-releasing `__getitem__`** (hashlib, NumPy, PIL, file I/O
via C extensions): threads run truly in parallel in both modes. Near parity.
- **Free-threaded Python (3.13t, `-X gil=0`):** No GIL. Our threads scale freely
— the process-overhead comparison disappears.

### 5.2 Sequential path — `batch_size` sweep (InMemoryDs, N=4 096, num_workers=0)

Our direct path calls `__getitem__` with a cached bound method — no queue, no
thread, no C++ machinery.


| batch_size | ours GIL | ours 3.13t   | torch GIL | torch 3.13t |
| ---------- | -------- | ------------ | --------- | ----------- |
| 1          | 3 604 k  | **7 473 k**  | 59 k      | 229 k       |
| 8          | 10 311 k | **20 872 k** | 445 k     | 1 643 k     |
| 32         | 13 154 k | **23 776 k** | 1 545 k   | 5 159 k     |
| 128        | 15 148 k | **26 471 k** | 4 052 k   | 11 454 k    |
| 512        | 15 696 k | **28 098 k** | 7 572 k   | 16 797 k    |
| 1 024      | 15 420 k | **30 165 k** | 8 610 k   | 17 496 k    |
| 4 096      | 15 740 k | **31 369 k** | 10 066 k  | 18 665 k    |


We win in both modes. On 3.13t our sequential path is 2× faster than on GIL Python
because even in the direct path each `__getitem__` call carries some GIL management
cost. Advantage over torch: **1.5–33× (GIL) / 1.7–33× (3.13t)**.

---

### 5.3 Parallel path — `batch_size` sweep (InMemoryDs, N=4 096, 4 workers)

`InMemoryDs.__getitem__` is a trivial `return index` — purely Python, no GIL
release. This is the stress test for threading overhead.


| batch_size | ours GIL     | ours 3.13t  | torch GIL   | torch 3.13t  |
| ---------- | ------------ | ----------- | ----------- | ------------ |
| 1          | 34 k         | 36 k        | 13 k        | **77 k**     |
| 8          | 227 k        | 406 k       | 158 k       | **463 k**    |
| 32         | 752 k        | 1 614 k     | 493 k       | **2 046 k**  |
| 128        | 1 002 k      | 4 731 k     | **1 653 k** | **5 388 k**  |
| 512        | 1 360 k      | 7 092 k     | **4 078 k** | **7 679 k**  |
| 1 024      | 1 469 k      | 6 103 k     | **5 030 k** | **10 415 k** |
| 4 096      | **10 666 k** | **6 651 k** | 6 985 k     | 4 528 k      |


**GIL Python:** torch wins at bs≥128 because multiprocessing bypasses the GIL
entirely. We win only at small batch sizes (fewer items to serialize) and bs=4096
(single batch, process overhead dominates).

**3.13t:** Our parallel path improves **4–7×** vs GIL mode (threads now actually
parallel). Torch also improves but less so (processes already bypassed GIL).
Torch still edges ahead at bs=8–1024 because crossbeam channel + reorder buffer
overhead is non-negligible for a trivial O(1) dataset. At bs=4096 (one batch) we
win clearly — process spawn cost exceeds batch value.

**The takeaway:** `num_workers > 0` + trivial `__getitem__` is not a realistic
workload. Nobody parallelizes a dict lookup. The meaningful comparison is below.

---

### 5.4 Parallel path — worker scaling (CpuBoundDs, SHA-256 of 1 MB per item, ~1 ms/item)

`hashlib.sha256` releases the GIL during the C computation. This reflects
real-world I/O- or compute-heavy datasets.


| workers | ours GIL   | ours 3.13t | torch GIL | torch 3.13t |
| ------- | ---------- | ---------- | --------- | ----------- |
| 0       | 2 105      | 2 434      | 2 069     | 2 394       |
| 1       | 2 122      | 2 363      | 2 059     | 2 350       |
| 2       | 4 430      | **4 606**  | 3 975     | 4 613       |
| 4       | 8 353      | **8 965**  | 8 300     | 8 976       |
| 8       | **16 159** | **17 036** | 15 635    | 16 939      |


All four columns scale linearly with workers. GIL vs no-GIL barely matters here
because `hashlib` releases the GIL regardless. We hold a **2–6% edge** at w≥2
across both Python modes — in-process threads have lower coordination overhead
than inter-process queues.

---

### 5.5 Prefetch depth (CpuBoundDs, N=128, bs=16, 4 workers)


| depth | ours GIL | ours 3.13t | torch GIL | torch 3.13t |
| ----- | -------- | ---------- | --------- | ----------- |
| 1     | 8 181    | **8 865**  | 7 710     | 8 675       |
| 4     | 8 052    | **8 970**  | 7 657     | 8 787       |
| 16    | 7 992    | **8 916**  | 7 498     | 8 687       |


Flat across all depths in both modes — workers always have work. We run **~2–6%**
ahead. The 3.13t runs are faster overall because Python object allocation
(batch list construction) is cheaper without GIL contention.

---

### 5.6 Sampler (CpuBoundDs, N=1k–100k, 4 workers)


| N       | ours seq | ours shuf | torch seq | torch shuf |
| ------- | -------- | --------- | --------- | ---------- |
| 1 000   | 8 219    | 8 569     | 7 324     | 7 221      |
| 10 000  | 8 355    | 8 081     | 7 507     | 7 424      |
| 100 000 | 7 827    | 7 751     | **8 117** | **8 378**  |


Near parity at all sizes. We edge ahead at small N; torch's `randperm` (C++)
catches up at N=100k for shuffle. The gap is tiny relative to dataset fetch time.

## 6. Results: real-world workloads

Reference run: real-world workloads, `dev`, October 2026.

The benchmarks above isolate loader overhead with synthetic datasets. These run
the per-sample work real datasets do — reading files, augmenting images,
waiting on the network, running Python code. The work is written once
(`bench/real_world.py`) and used by both loaders, so only the loading differs:
scheduling, parallelism, prefetching, collation, and moving samples to the
consumer. Each case times whole epochs, iterator creation included, which is
what a training loop pays.

**Machine:** laptop, Intel Core i9-13900H (20 threads), Linux x86_64
**Python:** CPython 3.13.3 (GIL), torch 2.14.1+cpu, numpy 2.5.4
**Run:** `python bench/bench_real_world.py --workers 4` — 1 warm-up + 5 timed
epochs, median, `batch_size=32`, `collate_fn=np.stack` for array samples.

| workload | per-sample work | items | workers |
|---|---|---|---|
| `local_files` | read a 128 KB file, normalise to float32 | 2 048 | 4 |
| `image_augment` | random crop 224, flip, normalise a decoded 3×256×256 image | 1 024 | 4 |
| `remote` | 10 ms network round trip (`time.sleep`) | 512 | 16 |
| `python_cpu` | pure-Python tokenisation (holds the GIL) | 2 048 | 4 |

### 6.1 Python — dataloader_rs vs torch.utils.data.DataLoader (items/s)

| workload | ours `w=0` | ours | torch | torch persistent | best ours vs torch |
|---|---:|---:|---:|---:|---:|
| `local_files` | 7 439 | **11 041** | 1 581 | 1 656 | **7.0×** |
| `image_augment` | 4 837 | **8 682** | 1 274 | 1 357 | **6.8×** |
| `remote` | 99 | **1 573** | 1 352 | 1 563 | 1.2× |
| `python_cpu` | 26 265 | 25 924 | 41 694 | **65 036** | 0.6× |

What the numbers say:

- **Array samples (`local_files`, `image_augment`): about 7× faster.** torch's
  worker processes must serialise every sample (a 0.5 MB array here) back to
  the main process; threads hand over a reference. For large samples that
  transfer, not the work, dominates torch's epoch. Even `w=0` beats torch's
  4 worker processes.
- **Threads help as far as the dataset releases the GIL.** File reads and
  numpy release it: 4 workers are 1.5× (`local_files`) and 1.8×
  (`image_augment`) faster than `w=0`; the GIL-held Python parts cap the rest.
- **Network-bound (`remote`): both loaders are capped by their workers.** With
  blocking calls each worker has one request in flight (16 / 10 ms ≈ 1.6 K/s),
  so throughput follows `num_workers`, not the loader.
- **Pure-Python CPU (`python_cpu`): torch wins.** Code that holds the GIL cannot
  run in parallel on threads; torch's processes can (1.6× with persistent
  workers, 2.5× ours). Use free-threaded Python (3.14t) for this case.

### 6.2 Rust (Criterion, `benches/bench_real_world.rs`)

`cargo bench --bench bench_real_world`

**`local_files`** — read a 64 KB file + normalise to `Vec<f32>`, 1 024 items:

| num_workers | default glibc malloc | malloc trimming disabled¹ |
|---:|---:|---:|
| 0 | 75.7 ms | 18.2 ms |
| 2 | 30.3 ms | 14.9 ms |
| 4 | 23.8 ms | 15.3 ms |
| 8 | 22.7 ms | **12.3 ms** |

¹ `GLIBC_TUNABLES=glibc.malloc.trim_threshold=4294967295:glibc.malloc.mmap_threshold=33554432`.
With default settings, each 256 KB sample buffer allocated and freed on the
main thread makes glibc return memory to the OS and fault it back in every
batch, which inflates `num_workers=0` 4× (worker threads use separate arenas
that churn less). This is an allocator effect, not the loader: datasets that
produce large per-sample buffers should use mimalloc/jemalloc or these
tunables. With it removed, the work is light enough that reading files and
allocating, not the loader, limit scaling (1.5× at 8 workers).

**`remote_2ms`** — 2 ms network round trip per sample, 512 items:

| threads (requests in flight) | epoch | throughput |
|---:|---:|---:|
| 4 | 266 ms | 1.9 K/s |
| 16 | 66.8 ms | 7.7 K/s |
| 64 | 17.2 ms | 29.8 K/s |

Blocking I/O needs one thread per request in flight; throughput scales
linearly with the thread count.

## 7. Reading the results

### 7.1 Overview

| Scenario                            | GIL Python                      | 3.13t (no GIL)                         |
| ----------------------------------- | ------------------------------- | -------------------------------------- |
| **Sequential, any batch size**      | **1.5–61× faster**              | **1.7–33× faster**                     |
| **Parallel, trivial `__getitem__`** | mixed (torch wins at medium bs) | mixed (torch still edges at medium bs) |
| **Parallel, GIL-releasing dataset** | **~2–10% faster**               | **~2–6% faster**                       |
| **Prefetch depth sensitivity**      | flat, ~6% ahead                 | flat, ~2% ahead                        |
| **Shuffle sampler throughput**      | parity–7% ahead                 | parity                                 |

### 7.2 When to use this library

- **Sequential path (`num_workers=0`)**: always faster — 2–60× depending on batch
size. Zero configuration needed.
- **GIL-releasing `__getitem__`** (NumPy, PIL, HDF5, file I/O, any C extension
that releases the GIL): the parallel path gives 2–10% more throughput than torch
at no cost.
- **Free-threaded Python (3.14t)**: the intended long-term target. Threads
scale without GIL tax and our parallel path is consistently competitive.

### 7.3 When PyTorch is better

- **Pure-Python `__getitem__` with medium batch sizes on regular CPython**: torch
multiprocessing bypasses the GIL. Prefer `num_workers=0` (where we win) or
switch to 3.14t.
- **Very large shuffle samplers (N≥100k)**: `torch.randperm` is marginally faster.
Irrelevant relative to actual dataset I/O cost.

## 8. Caveats

- **One machine.** A laptop with performance and efficiency cores and frequency scaling. Ratios
  between loaders measured in the same session are more reliable than absolute numbers.
- **Two dates.** Sections 4–5 come from v0.1.0 (April 2026); the loader's code has changed since
  (errors, panics, `__getitems__`, samplers). Section 6 was measured on `dev` in October 2026.
  Rerun before quoting a number (section 9).
- **Synthetic datasets are extremes.** `InMemoryDs` makes the loader itself the whole cost; real
  datasets add I/O and compute that usually dominate — section 6 is closer to practice.
- **Page cache.** `local_files` reads files that are in the page cache after the first epoch: it
  measures the file API and the copy, not a disk.
- **Allocator.** The Rust `local_files` numbers depend on glibc's trimming (section 6): measure
  your own dataset with the allocator you ship.
- **CI numbers are noisy.** Shared GitHub runners vary by 1.5× and more between runs; the CI
  alert threshold (1.5×) catches large regressions, not small ones. Use an interleaved local A/B
  (section 9.4) to measure a change.

## 9. Running it again

### 9.1 Rust (Criterion)

```bash
cargo bench                                  # every suite
cargo bench --bench bench_inter_workers      # one suite
cargo bench -- num_workers/4                 # filter inside a suite
cargo bench --bench bench_real_world         # file reads + network latency
```

HTML reports land in `target/criterion/`.

### 9.2 Python regression guards (as in CI)

```bash
uv sync --group dev
uv run maturin develop --release --features python
uv run pytest bench/ -m bench --benchmark-enable --benchmark-warmup=on --benchmark-min-rounds=10
```

### 9.3 Python against PyTorch

```bash
uv run python bench/bench_batch_size.py     --warmup 2 --repeats 10
uv run python bench/bench_inter_workers.py  --warmup 2 --repeats 10
uv run python bench/bench_prefetch_depth.py --warmup 2 --repeats 10
uv run python bench/bench_sampler.py        --warmup 2 --repeats 10
uv run python bench/bench_real_world.py     --workers 4
```

These run on free-threaded CPython 3.14t, the repository's default (`.python-version`). For the
GIL comparison: `uv run --python 3.13 python bench/...`.

### 9.4 Comparing two versions

Run both builds alternately in the same session (A, B, A, B, …) and compare medians: machine
state drifts more between sessions than between two interleaved runs. For Criterion, delete
`target/criterion` between builds so it does not compare against the other build's baseline.
