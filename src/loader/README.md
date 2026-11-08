# The loader

`DataLoader<Src, S, C, T>` owns a dataset in a source `Src`, a sampler `S`, a collator `C` and
a transform `T` (`Identity` by default); each call to `iter()` runs one epoch and yields
`Result<C::Batch>` in epoch order. The source is a `SyncSource<D>` (`builder`, a `Dataset`) or
an `AsyncSource<A>` (`builder_async`, an `AsyncDataset` and its tokio runtime); `dataset()`
returns `D` or `A`.

| File | What |
|---|---|
| `builder.rs` | `DataLoaderBuilder`: settings, `sampler` / `transform` / `collator` (which change the loader's type), `build` |
| `core.rs` | `DataLoader`: `iter`, `len`, `batch_len`, accessors; `&mut DataLoader` is `IntoIterator` |
| `source.rs` | the sealed `LoaderSource` trait, `SyncSource` and `AsyncSource`: each starts its own kind of epoch and sizes the pool |
| `iter.rs` | `DataLoaderIter`; `SyncEpoch`: the direct path, and the parallel core (reorder buffer, drop) |
| `worker.rs` | `process_batch` (fetch + transform + collate one batch) and `Epoch`: batch tasks, the claim window |
| `core.rs` (`LoaderPool`) | the loader's rayon pool, rebuilt after `fork()` |
| `async_epoch.rs` | feature `async`: the loader's tokio runtime and `AsyncEpoch`: the driver task, batch tasks, the bounded channel ([ARCHITECTURE.md, 3.7](../../ARCHITECTURE.md#37-async-datasets-builder_async)) |

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

`DataLoader::builder(dataset)` starts with `SequentialSampler`, `Identity` and `VecCollator`;
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
| `batch_size` | `1` | samples per batch (must be > 0) |
| `drop_last` | `false` | drop a final batch smaller than `batch_size` |
| `num_workers` | `0` | batches processed in parallel; `0` processes batches inside `next()` |
| `item_concurrency` | `1` | items of one batch fetched in parallel (must be > 0); `1` = one `get_batch` call per batch |
| `prefetch_depth` | `2` | how far batches may run ahead: at most `num_workers + prefetch_depth` in flight or waiting (must be > 0) |
| `runtime_threads` | CPUs available | async only: worker threads of the loader's tokio runtime (1 to `MAX_RUNTIME_THREADS` = 1024; exclusive with `runtime(handle)`) |
| `pin_memory` | — | pin `tch::Tensor` batches (feature `torch-rs`) |

The loader's pool has `max(num_workers, 1) × item_concurrency` threads (none with both at their
defaults), so that many fetches run at once at most.

## The two paths

- **`num_workers = 0`**: `next()` fetches, transforms and collates the next batch on the calling
  thread. No
  thread, no channel. Fastest when samples are cheap.
- **`num_workers > 0`**: up to `num_workers` batch tasks run on the loader's pool (built once,
  reused by every epoch); they claim batches in epoch order inside a window of
  `num_workers + prefetch_depth`, and the iterator returns them in order.

With `item_concurrency = k > 1`, either path fetches each batch with `k` lanes on the pool,
each calling `Dataset::get` for the batch's next item and transforming it: at most `k` items of
a batch in flight, transformed in parallel.

The threads, the backpressure and the reorder buffer are described in
[ARCHITECTURE.md, section 3](../../ARCHITECTURE.md#3-life-of-an-epoch).

## Errors, panics, early exit

| Event | Result |
|---|---|
| `get`, `get_batch`, the transform or `collate` returns `Err` | the iterator yields `Some(Err(e))` for that batch and continues |
| `get_batch` returns more or fewer items than indices | the batch fails |
| the dataset, transform or collator panics | the panic reaches the caller of `next()`, with or without workers |
| the iterator is dropped mid-epoch | tasks stop claiming batches; `drop` does not wait, a batch already running finishes in the background |
| the process forks (`fork()`) | the next `iter()` in the child builds a new pool |
| the loader is dropped while an iterator lives | allowed: the iterator holds its own `Arc`s of the dataset, transform and collator |

## Choosing the settings

| Situation | Settings |
|---|---|
| samples cost microseconds (in memory, tiny decode) | `num_workers(0)`: the hand-off between threads would cost more than the work |
| each sample costs about a millisecond or more (I/O, decode, augment) | `num_workers` up to the number of cores; the default `prefetch_depth` 2, more only if batch times vary a lot |
| few, large batches of CPU-bound items | `item_concurrency` > 1, so one batch is spread over several cores |
| blocking I/O (network, remote storage) | requests in flight = `num_workers × item_concurrency`: `num_workers(4).item_concurrency(16)` keeps 64 in flight |
| many nearly free batches (`batch_size` ≤ 8, in-memory items) | `num_workers(0)`, or a larger `batch_size`: scheduling a batch costs about half a microsecond |

Measurements behind these rules: [BENCH.md](../../BENCH.md#4-results-rust).
