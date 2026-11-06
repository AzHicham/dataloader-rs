# Collators

A collator turns the samples of one batch, `Vec<Item>`, into the batch value the iterator yields.
It runs on the thread that built the batch (a worker thread with `num_workers > 0`). Choose one
with `DataLoaderBuilder::collator`; the default is `VecCollator`.

| Collator | Batch | Feature |
|---|---|---|
| `VecCollator` | `Vec<Item>`, unchanged | |
| `DefaultCollator` | column-wise, like PyTorch's `default_collate` (below) | `ndarray` / `torch-rs` for arrays and tensors |
| `TorchPinnedCollator<C>` | `C`'s `tch::Tensor` batch, pinned in host memory when CUDA or MPS is available | `torch-rs` |
| `PooledCollator<C, A>` | a `PooledBatch`: `C` (a `CollateInto`) writes the batch into a reused buffer from allocator `A` | |

## The `Collator` trait

```rust
use dataloader_rs::{Collator, error::Result};

/// Sum of the batch's samples.
struct Sum;

impl Collator<u64> for Sum {
    type Batch = u64;

    fn collate(&self, items: Vec<u64>) -> Result<u64> {
        Ok(items.into_iter().sum())
    }
}

assert_eq!(Sum.collate(vec![1, 2, 3]).unwrap(), 6);
```

`Collator<Item>: Send + Sync + 'static` and `Batch: Send + 'static`: one collator is shared by
every worker, and batches cross threads. A `collate` error is reported for its batch, like a
dataset error.

## `DefaultCollator`

Collates by structure, recursively, through the `DefaultCollate` trait:

| Sample type | Batch type | How |
|---|---|---|
| `bool`, `char`, integers, `f32`, `f64`, `String` | `Vec<T>` | collected |
| `(A, B, …)`, up to 6 elements | `(A::Batch, B::Batch, …)` | per position |
| `[T; N]` | `[T::Batch; N]` | per position |
| `Vec<T>` | `Vec<T::Batch>` | per position; every sample must have the same length |
| `HashMap<K, V>`, `BTreeMap<K, V>` | the same map of `V::Batch` | per key; every sample must have the same keys |
| `ndarray::Array<A, D>` (`ndarray`) | `Array<A, D::Larger>` | stacked on a new axis 0 |
| `tch::Tensor` (`torch-rs`) | `Tensor` | `Tensor::stack` on dimension 0 |

A sample that breaks a rule (different lengths, different keys, arrays of different shapes)
makes `collate` return an error for that batch.

```rust
use dataloader_rs::{DataLoader, Dataset, collator::DefaultCollator, error::Result};

/// (label, features)
struct Points;

impl Dataset for Points {
    type Item = (u32, [f32; 2]);

    fn get(&self, index: usize) -> Result<(u32, [f32; 2])> {
        let x = index as f32;
        Ok((index as u32 % 2, [x, x * 10.0]))
    }

    fn len(&self) -> usize {
        4
    }
}

let mut loader = DataLoader::builder(Points)
    .batch_size(2)
    .collator(DefaultCollator)
    .build();
let (labels, [xs, ys]) = loader.iter().next().unwrap().unwrap();
assert_eq!(labels, vec![0, 1]);
assert_eq!(xs, vec![0.0, 1.0]);
assert_eq!(ys, vec![0.0, 10.0]);
```

Implement `DefaultCollate` for your own sample type to make it collatable the same way.

## `TorchPinnedCollator`

`DataLoaderBuilder::pin_memory(true)` (feature `torch-rs`) wraps the current collator, whose
batch must be a `tch::Tensor`, and pins each batch in page-locked host memory, which speeds up
the copy to a GPU. Pinning is skipped when neither CUDA nor MPS is available.

## Reusable batch buffers

A collator allocates every batch it returns. For large batches (64 × 3 × 224 × 224 `f32` is
38.5 MB) the allocation, its page faults and its free cost milliseconds per batch, and the
memory cannot be page-locked for a fast, asynchronous copy to the GPU.
`DataLoaderBuilder::collate_into(collator, allocator)` instead writes each batch into a buffer
taken from a pool owned by the loader, and the batch goes back to the pool when it is dropped.

| Piece | Role |
|---|---|
| `CollateInto<Item>` | `collate_into(items, &mut buffer) -> Result<Output>`: write the samples into the buffer; `Output` is per-batch metadata (`()` if none) |
| `BufferAllocator` | `allocate() -> Result<Buffer>`: one buffer for a full batch; any `Fn() -> Result<B>` closure is one |
| `HeapAllocator<T>` | `Vec<T>` buffers of a fixed length (`for_shape(&[64, 3, 224, 224])`), sizes checked |
| `TorchPinnedAllocator` | `tch::Tensor` buffers, pinned when CUDA or MPS is available (`torch-rs`) |
| `ConcatCollator<T, B>` | sample `i` (`AsRef<[T]>`, `item_len` elements) copied to `i * item_len` of an `AsMut<[T]>` buffer |
| `DefaultCollator` | also a `CollateInto` for `tch::Tensor` (`torch-rs`) and `ndarray::Array` (`ndarray`) samples: rows `0..n` of the buffer, in place |
| `PooledBatch<B, O>` | the batch: derefs to the buffer; `num_samples()`, `output()`, `detach()` |
| `BufferPool<A>` | the pool, for diagnostics: `capacity()`, `allocated()`, `available()` (`loader.collator().pool()`) |

```rust
use dataloader_rs::{
    DataLoader, Dataset,
    collator::{ConcatCollator, HeapAllocator},
    error::Result,
};

/// One CHW image of 3 × 4 × 4 per sample.
struct Images;

impl Dataset for Images {
    type Item = Vec<f32>;

    fn get(&self, index: usize) -> Result<Vec<f32>> {
        Ok(vec![index as f32; 3 * 4 * 4])
    }

    fn len(&self) -> usize {
        10
    }
}

let mut loader = DataLoader::builder(Images)
    .batch_size(4)
    .num_workers(2)
    .collate_into(
        ConcatCollator::new(3 * 4 * 4),
        HeapAllocator::<f32>::for_shape(&[4, 3, 4, 4])?,
    )
    .build();

for batch in &mut loader {
    let batch = batch?;
    let samples = &batch[..batch.num_samples() * 3 * 4 * 4]; // NCHW, the last batch is shorter
    assert_eq!(samples.len() % (3 * 4 * 4), 0);
} // each batch returns its buffer to the pool here
assert!(loader.collator().pool().allocated() <= 2 + 2 + 1);
# Ok::<(), dataloader_rs::Error>(())
```

Measured with `benches/bench_collate_into.rs` (64 × 3 × 224 × 224, i9-13900H E-cores, medians
of 12 interleaved rounds): one `f32` batch (38.5 MB) collates in 4.3 ms instead of 18.9 ms, an
epoch of 8 batches takes 34 ms instead of 145 ms without workers and 24 ms instead of 78 ms with
2. A 16-bit batch (19 MB) gains nothing: glibc's `malloc` serves blocks below its 32 MB mmap
threshold from memory it keeps, so only batches above it pay page faults on every allocation.
Reuse pays for large batches, and for pinned memory, which is too slow to allocate per batch.

**Bounded memory.** The pool allocates buffers on demand, up to `num_workers + prefetch_depth`
(`1` with `num_workers = 0`) plus `max_held_batches` (default `1`), and never more. When every
buffer is taken, the worker about to collate waits for the consumer to drop a batch: this is
backpressure, and a consumer that hands batches to another thread (a GPU stream, say) simply slows
the workers down. Keeping more than `max_held_batches` batches alive **on the thread that
iterates** (`collect()`, a growing `Vec`) deadlocks: the batch it waits for needs a buffer it
holds. Raise `max_held_batches` to keep several (e.g. `2` to process one batch while the
previous one is still being copied to the GPU), or `detach()` a batch to keep its buffer for good.

**Contents.** A buffer comes back with the content of an earlier batch. `collate_into` must write
every sample it is given; a last, partial batch fills only the first `num_samples()` samples,
and the rest of the buffer is stale. Errors and panics in `collate_into` return the buffer to the
pool, and are reported like any collator's.

**Pinned memory.** `TorchPinnedAllocator` pins safely through libtorch. For memory from another
library, pass a closure: for instance ONNX Runtime's `CUDA_PINNED` allocator, with the batch
buffer an `ort` tensor, or `cudaHostAlloc` through cudarc. Allocation that needs `unsafe` lives
in that closure, in the caller's crate; this crate stays `#![forbid(unsafe_code)]` and has no
CUDA dependency.

## From Python

`collate_fn=` takes any callable `list[sample] -> batch`; without it a batch is a `list`. It runs
on the worker thread that fetched the batch ([python/README.md](../../python/README.md)).
