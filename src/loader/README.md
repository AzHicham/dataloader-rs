# The loader

`DataLoader<Src, S, C, T>` owns a dataset wrapped in a source `Src`, a sampler `S`, a collator
`C` and a transform `T` (`Identity` by default). Each call to `iter()` runs one epoch and yields
one `Result<C::Batch>` per batch, in epoch order. The source is a `SyncSource<D>`, made by
`DataLoader::builder` from a `Dataset`, or an `AsyncSource<A>`, made by
`DataLoader::builder_async` from an `AsyncDataset` and its tokio runtime; `dataset()` returns the
`D` or the `A`.

| File | What |
|---|---|
| `builder.rs` | `DataLoaderBuilder`: the settings, `sampler`, `transform` and `collator` (which change the loader's type), and `build`. |
| `dataloader.rs` | `DataLoader`: `iter`, `iter_with_indices`, `len`, `batch_len` and accessors. `&mut DataLoader` implements `IntoIterator`. |
| `source/` | The sealed `LoaderSource` trait (`loader_source.rs`, `sealed.rs`), and its two implementations, `SyncSource` (`sync.rs`) and `AsyncSource` (`asynchronous.rs`). Each starts its own kind of epoch and sizes the thread pool (`threads.rs`). `EpochSettings` (`settings.rs`) carries the settings into an epoch. |
| `iter.rs` | `DataLoaderIter`, the iterator over the epoch of either source, and `IndexedIter`, the same with the indices of each batch. |
| `pool.rs` | `LoaderPool`: the loader's rayon pool, rebuilt after `fork()`. |
| `sync/epoch.rs` | `SyncEpoch`: the direct path, or the batch tasks on the pool. |
| `sync/worker.rs` | `process_batch`, which fetches, transforms and collates one batch, and `Epoch`: the batch tasks, the claim window and the item lanes. |
| `sync/reorder.rs` | `ParallelCore`: the reorder ring, the check for `fork()`, and what happens when an iterator is dropped. |
| `asynchronous/epoch.rs` | Feature `async`. `AsyncEpoch`: the driver task, the batch tasks and the bounded channel ([ARCHITECTURE.md, 2.6](../../ARCHITECTURE.md#26-async-dataset-dataloaderbuilder_async)) |
| `asynchronous/runtime.rs` | Feature `async`. The tokio runtime of the loader: its own (sized by `runtime_threads`) or the caller's (`runtime`). |

## Building and iterating

```rust
use dataloader_rs::{DataLoader, Dataset, RandomSampler, error::Result};

struct Numbers(usize);

impl Dataset for Numbers {
    type Item = u32;
    fn get(&self, index: usize) -> Result<u32> {
        Ok(index as u32)
    }
    fn len(&self) -> usize {
        self.0
    }
}

let mut loader = DataLoader::builder(Numbers(100))
    .batch_size(16)
    .drop_last(true)
    .sampler(RandomSampler::new(7))
    .num_workers(2)
    .prefetch_depth(4)
    .build();

assert_eq!(loader.len(), 100);     // samples
assert_eq!(loader.batch_len(), 6); // batches per epoch (100 / 16, drop_last)

for _epoch in 0..2 {
    let mut seen = 0;
    for batch in &mut loader {
        seen += batch.unwrap().len();
    }
    assert_eq!(seen, 96);
}
```

`DataLoader::builder(dataset)` starts with `SequentialSampler`, `Identity` and `VecCollator`.
`.sampler(…)`, `.transform(…)` and `.collator(…)` replace them and keep every other setting.

```rust
use dataloader_rs::{DataLoader, Dataset, RandomSampler, error::Result};

struct Numbers;

impl Dataset for Numbers {
    type Item = u32;
    fn get(&self, index: usize) -> Result<u32> {
        Ok(index as u32)
    }
    fn len(&self) -> usize {
        10
    }
}

// The transform gets each item with its index; the collator gets the transformed items.
let mut loader = DataLoader::builder(Numbers)
    .batch_size(4)
    .sampler(RandomSampler::new(1))
    .transform(|index: usize, x: u32| -> Result<(usize, u32)> { Ok((index, x * 10)) })
    .build();
for batch in &mut loader {
    // Item `i` is `i`: the transform sees the index the sampler drew.
    assert!(batch.unwrap().iter().all(|&(i, x)| x == i as u32 * 10));
}
```

| Setting | Default | Meaning |
|---|---|---|
| `batch_size` | `1` | The number of samples per batch; at least 1. |
| `drop_last` | `false` | Drops the last batch of an epoch when it is smaller than `batch_size`. |
| `num_workers` | `0` | The number of batches processed in parallel. With `0`, batches are processed inside `next()`. |
| `item_concurrency` | `1` | The number of items of one batch fetched in parallel; at least 1. With `1`, each batch is one `get_batch` call. |
| `prefetch_depth` | `2` | How far batches may run ahead: at most `num_workers + prefetch_depth` are in flight or waiting. At least 1. |
| `runtime_threads` | the CPUs available | Async datasets only: the threads of the loader's tokio runtime, from 1 to `MAX_RUNTIME_THREADS` (1024). It cannot be combined with `runtime(handle)`. |
| `pin_memory` | off | Pins `tch::Tensor` batches in page-locked memory (feature `torch-rs`). |

For a sync dataset, the loader's pool has `max(num_workers, 1) × item_concurrency` threads, or
none when both settings are at their defaults, so at most that many fetches run at once. An
async loader's pool only transforms and collates, and is sized as described in ARCHITECTURE.md.

## The two paths

- **`num_workers = 0`**: `next()` fetches, transforms and collates the next batch on the calling
  thread, with no other thread and no channel involved. This is the fastest path when samples
  are cheap.
- **`num_workers > 0`**: up to `num_workers` batch tasks run on the loader's pool, which is built
  once and reused by every epoch. They claim batches in epoch order, never more than
  `num_workers + prefetch_depth` ahead of the consumer, and the iterator returns the batches in
  order.

With `item_concurrency = k > 1`, either path fetches each batch with `k` lanes on the pool.
Each lane calls `Dataset::get` for the batch's next item and transforms it, so at most `k` items
of a batch are in flight, and they are transformed in parallel.

The threads, the bounded prefetch and the reorder ring are described in
[ARCHITECTURE.md, section 2](../../ARCHITECTURE.md#2-life-of-an-epoch).

## Errors, panics, early exit

| Event | Result |
|---|---|
| `get`, `get_batch`, the transform or `collate` returns `Err` | The iterator yields `Some(Err(e))` for that batch and continues with the next one. |
| `get_batch` returns more or fewer items than indices | The batch fails. |
| The dataset, the transform or the collator panics | The panic reaches the caller of `next()`, with or without workers. |
| The iterator is dropped in the middle of an epoch | The tasks stop claiming batches. `drop` does not wait; a batch already running finishes in the background. |
| The process forks (`fork()`) | The next `iter()` in the child process builds a new pool. |
| The loader is dropped while an iterator is alive | This is allowed: the iterator holds its own `Arc` clones of the dataset, the transform and the collator. |

## Choosing the settings

| Situation | Settings |
|---|---|
| Samples cost microseconds (in memory, or a tiny decode) | `num_workers(0)`: handing batches between threads would cost more than the work. |
| Each sample costs about a millisecond or more (I/O, decoding, augmentation) | `num_workers` up to the number of cores, and the default `prefetch_depth` of 2; more only if batch times vary a lot. |
| Few, large batches of CPU-bound items | `item_concurrency` above 1, so that one batch is spread over several cores. |
| Blocking I/O (network, remote storage) | `num_workers × item_concurrency` is the number of requests in flight: `num_workers(4).item_concurrency(16)` keeps 64 in flight. For many more, use an async dataset. |
| Many nearly free batches (8 in-memory items or fewer per batch) | `num_workers(0)`, or a larger `batch_size`: scheduling a batch costs about half a microsecond. |

Measurements behind these rules: [benches/README.md](../../benches/README.md#reference-results).
