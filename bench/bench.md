# Benchmark Results

Machine: Linux x86_64, Python 3.13 (GIL), CPU-only.  
Workloads: `InMemoryDs` (O(1) return index), `HeavyCpuDs` (50k LCG iterations), `LightCpuDs` (1k LCG), `CpuBoundDs` (sha256 ×200, ~0.5 ms/item).

---

## Rust Core (Criterion)

### `batch_size` — sequential path (InMemoryDs, 4096 items)

| batch_size | time/epoch |
|-----------|-----------|
| 1         | 115 µs    |
| 8         | 28 µs     |
| 32        | 13 µs     |
| 128       | 8.9 µs    |
| 512       | 5.2 µs    |
| 1024      | 4.4 µs    |
| 4096      | 3.8 µs    |

At small batch sizes, per-batch iterator overhead (Vec allocation, sampler call) dominates. At bs=4096 the epoch completes in 3.8 µs — sampler + allocation fully amortized.

### `batch_size` — parallel path (InMemoryDs, 4 workers, prefetch=16)

| batch_size | time/epoch |
|-----------|-----------|
| 1         | 808 µs    |
| 8         | 152 µs    |
| 32        | 91 µs     |
| 128       | 67 µs     |
| 512–4096  | ~58 µs    |

Parallel path plateaus at ~58 µs for large batches — this is the thread synchronization floor (crossbeam channel roundtrip × N workers × prefetch depth).

### `inter_workers` (HeavyCpuDs, bs=16, prefetch=32)

| workers | time    | speedup vs seq |
|---------|---------|---------------|
| 0       | 1.30 ms | 1×            |
| 1       | 2.27 ms | **0.57× — slower** |
| 2       | 1.26 ms | 1.03×         |
| 4       | 720 µs  | 1.8×          |
| 8       | 389 µs  | 3.3×          |

**1 worker is slower than sequential**: thread + channel overhead with no parallelism gain. Meaningful speedup starts at 2 workers, scales well to 8.

### `intra_workers` — rayon intra-batch (HeavyCpuDs, inter=2, bs=64)

| intra | time    | speedup vs 0 |
|-------|---------|-------------|
| 0     | 698 µs  | 1×           |
| 1     | 751 µs  | 0.93×        |
| 2     | 471 µs  | 1.5×         |
| 4     | 270 µs  | 2.6×         |
| 8     | 198 µs  | 3.5×         |
| inter=4 + intra=4 | 271 µs | — |

Rayon intra-batch parallelism scales well on its own. Combining inter=4 + intra=4 gives the same result as intra=4 alone — the CPU is saturated and adding both dimensions adds context-switch pressure without further gain.

### `prefetch_depth` (LightCpuDs, 4 workers, bs=8)

| depth | time   |
|-------|--------|
| 1     | 78.9 µs |
| 2     | 78.2 µs |
| 4     | 73.6 µs |
| 8     | 74.1 µs |
| 16    | 73.0 µs |

Negligible effect beyond depth=4. For CPU-bound work, workers drain the channel faster than the consumer can block — the channel is not the bottleneck. **Recommended: 4–8.**

### `early_drop` (InMemoryDs, bs=10, 1000 items)

| scenario            | workers | time    |
|---------------------|---------|---------|
| consume_1_then_drop | 0       | 2.07 µs |
| drop_immediately    | 0       | 2.15 µs |
| consume_1_then_drop | 1       | 34.1 µs |
| drop_immediately    | 1       | 31.4 µs |
| consume_1_then_drop | 2       | 43.7 µs |
| drop_immediately    | 2       | 43.6 µs |
| consume_1_then_drop | 4       | 67.0 µs |
| drop_immediately    | 4       | 66.7 µs |

Early drop cost scales linearly with worker count (~17 µs/worker for thread join). No hang, no memory leak. Consuming 1 batch vs. 0 makes no measurable difference — the cancel flag + receiver drop is the dominant shutdown path.

### `sampler` (InMemoryDs)

| N      | sequential | random  | ratio |
|--------|-----------|---------|-------|
| 1 000  | 1.87 µs   | 6.57 µs | 3.5×  |
| 10 000 | 22.0 µs   | 71.2 µs | 3.2×  |
| 100 000| 220 µs    | 772 µs  | 3.5×  |

Random sampler (Fisher-Yates + SmallRng) is consistently 3.5× slower than sequential. Scales O(N) as expected.

---

## Python vs PyTorch

### `inter_workers` — CPU-bound (sha256 ×200 per item, N=128, bs=16)

| workers | ours (items/s) | torch (items/s) | ratio     |
|---------|---------------|-----------------|-----------|
| 0       | 2 131         | 2 135           | **parity**|
| 1       | 2 351         | 2 116           | **+11%**  |
| 2       | 4 501         | 4 091           | **+10%**  |
| 4       | 7 989         | 7 716           | **+3.5%** |
| 8       | 15 178        | 14 126          | **+7.4%** |

We win across all worker counts for CPU-bound work. No multiprocessing overhead (no pickle, no fork/spawn), and the Rust crossbeam channel is lighter than `multiprocessing.Queue`.

### `batch_size` — in-memory dataset, sequential path (N=4096, num_workers=0)

| batch_size | ours (items/s) | torch (items/s) | ratio      |
|-----------|---------------|-----------------|------------|
| 1         | 7 812 k        | 223 k           | **35×**    |
| 8         | 23 077 k       | 1 676 k         | **14×**    |
| 32        | 29 853 k       | 5 389 k         | **5.5×**   |
| 128       | 33 387 k       | 12 568 k        | **2.7×**   |
| 512       | 35 290 k       | 20 519 k        | **1.7×**   |
| 4096      | 37 573 k       | 25 575 k        | **1.5×**   |

Our sequential path calls `__getitem__` directly in the calling thread with a cached bound method — no GIL release/reacquire, no queue, no C++ machinery. The advantage is dramatic at small batch sizes and converges to ~1.5× at large ones.

### `batch_size` — in-memory dataset, parallel path (N=4096, num_workers=4, prefetch=16)

| batch_size | ours (items/s) | torch (items/s) | ratio   |
|-----------|---------------|-----------------|---------|
| 1         | 90 k           | 32 k            | **2.8×**|
| 8         | 534 k          | 224 k           | **2.4×**|
| 32        | 1 867 k        | 820 k           | **2.3×**|
| 128       | 5 129 k        | 2 264 k         | **2.3×**|
| 512       | 9 676 k        | 5 239 k         | **1.8×**|
| 4096      | 14 227 k       | 7 114 k         | **2.0×**|

Consistently **2–2.8× faster** in parallel for in-memory data. We use raw pointer sharing — no IPC, no pickle, direct shared-memory access. PyTorch must serialize items across process boundaries.

### `prefetch_depth` (CpuBoundDs, 4 workers, N=64, bs=8)

| depth | ours (items/s) | torch (items/s) |
|-------|---------------|-----------------|
| 1     | 8 115         | 7 423           |
| 2     | 8 366         | 7 529           |
| 4     | 8 137         | 7 710           |
| 8     | 7 908         | 7 423           |
| 16    | 7 935         | 7 446           |

Both libraries are flat — prefetch depth is irrelevant when CPU is the bottleneck. We're ~8–11% ahead at small depths, converge at larger ones.

### `sampler` — sequential vs shuffle (InMemoryDs, num_workers=0)

| N       | ours seq  | ours shuf | torch seq | torch shuf |
|---------|----------|----------|----------|-----------|
| 1 000   | 32 380 k  | 5 711 k   | 10 040 k  | 6 560 k    |
| 10 000  | 32 044 k  | 5 521 k   | 12 219 k  | 8 973 k    |
| 100 000 | 29 900 k  | 5 254 k   | 12 438 k  | 7 958 k    |

- **Sequential**: we're **2.5–3× faster** than PyTorch (pure Rust range vs. Python range).
- **Shuffle**: PyTorch is **1.2–1.6× faster** than us. PyTorch uses `torch.randperm()` (C++, highly optimized). Our shuffle is `rand::SmallRng` Fisher-Yates — fast but not at that level. This is the only benchmark where we lose, and the gap is purely in the sampler.

---

## Summary

| Scenario                        | vs PyTorch         |
|---------------------------------|--------------------|
| Sequential, in-memory (any bs)  | **1.5–35× faster** |
| Parallel, in-memory             | **2–2.8× faster**  |
| Parallel, CPU-bound             | **~5–10% faster**  |
| Prefetch depth sensitivity      | comparable         |
| Shuffle sampler                 | **20–60% slower**  |

The shuffle gap is the only weakness and is isolated to the sampler. Everything else is at parity or better.
