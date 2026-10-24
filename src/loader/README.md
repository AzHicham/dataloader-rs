# The loader

`DataLoader<D, S, C>` owns a dataset `D`, a sampler `S` and a collator `C`; each call to `iter()`
runs one epoch and yields `Result<C::Batch>` in epoch order.

| File | What |
|---|---|
| `builder.rs` | `DataLoaderBuilder`: settings, `sampler` / `collator` (which change the loader's type), `build` |
| `core.rs` | `DataLoader`: `iter`, `len`, `batch_len`, accessors; `&mut DataLoader` is `IntoIterator` |
| `iter.rs` | `DataLoaderIter`: the direct path, and the parallel core (threads, reorder buffer, drop) |
| `worker.rs` | `process_batch` (fetch + collate one batch) and the worker loop |

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

`DataLoader::builder(dataset)` starts with `SequentialSampler` and `VecCollator`; `.sampler(…)`
and `.collator(…)` replace them and keep every other setting.

| Setting | Default | Meaning |
|---|---|---|
| `batch_size` | `1` | samples per batch (must be > 0) |
| `drop_last` | `false` | drop a final batch smaller than `batch_size` |
| `num_workers` | `0` | threads processing whole batches; `0` processes batches inside `next()` |
| `intra_workers` | `0` | threads of a rayon pool fetching the items of a batch in parallel |
| `prefetch_depth` | `1` | finished batches that may wait for the consumer (must be > 0) |
| `pin_memory` | — | pin `tch::Tensor` batches (feature `torch-rs`) |

## The two paths

- **`num_workers = 0`**: `next()` fetches and collates the next batch on the calling thread. No
  thread, no channel. Fastest when samples are cheap.
- **`num_workers > 0`**: `iter()` spawns `num_workers` threads for the epoch; they take batches
  from a queue and send results through a channel of capacity `prefetch_depth`; the iterator
  returns them in order and joins the threads when it is dropped.

With `intra_workers > 0`, either path fetches each batch's items with `Dataset::get` in parallel
on a rayon pool of `intra_workers` threads, built once with the loader.

The threads, the backpressure and the reorder buffer are described in
[ARCHITECTURE.md, section 3](../../ARCHITECTURE.md#3-life-of-an-epoch).

## Errors, panics, early exit

| Event | Result |
|---|---|
| `get`, `get_batch` or `collate` returns `Err` | the iterator yields `Some(Err(e))` for that batch and continues |
| the dataset or collator panics | the panic reaches the caller of `next()`, with or without workers |
| the iterator is dropped mid-epoch | workers stop taking batches; `drop` waits for batches already running |
| the loader is dropped while an iterator lives | allowed: the iterator holds its own `Arc`s of the dataset and collator |

## Choosing the settings

| Situation | Settings |
|---|---|
| samples cost microseconds (in memory, tiny decode) | `num_workers(0)`: the hand-off between threads would cost more than the work |
| each sample costs about a millisecond or more (I/O, decode, augment) | `num_workers` up to the number of cores; `prefetch_depth` 2–8 |
| few, large batches of CPU-bound items | `intra_workers` > 0, so one batch is spread over several cores |
| blocking I/O (network, remote storage) | more threads than cores: requests in flight ≈ `num_workers` (or `intra_workers`) |

Measurements behind these rules: [BENCH.md](../../BENCH.md#4-results-rust).
