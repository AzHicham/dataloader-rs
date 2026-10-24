# `src/`: the crate

| Path | What | Guide |
|---|---|---|
| `lib.rs` | crate docs and the top-level re-exports | |
| `dataset.rs` | the `Dataset` trait (map-style) and `IterableDataset` | below |
| `error.rs` | `Error` (`Box<dyn Error + Send + Sync>`) and `Result<T>` | below |
| `sampler/` | `Sampler`, `SequentialSampler`, `RandomSampler`, `DistributedSampler`, `BatchSampler` | [sampler/README.md](sampler/README.md) |
| `collator/` | `Collator`, `VecCollator`, `DefaultCollator` / `DefaultCollate`, `TorchPinnedCollator` | [collator/README.md](collator/README.md) |
| `loader/` | `DataLoader`, `DataLoaderBuilder`, `DataLoaderIter`, the worker threads | [loader/README.md](loader/README.md) |
| `python/` | the PyO3 module behind the Python package (feature `python`) | [python/README.md](python/README.md) |

Top-level re-exports: `Dataset`, `IterableDataset`, `Sampler`, `SequentialSampler`,
`RandomSampler`, `DistributedSampler`, `BatchSampler`, `Collator`, `VecCollator`, `DataLoader`,
`DataLoaderBuilder`, `DataLoaderIter`, `Error`, `Result`.

## The `Dataset` contract

```rust
use dataloader_rs::{Dataset, error::Result};

struct Squares(usize);

impl Dataset for Squares {
    type Item = u64;

    fn get(&self, index: usize) -> Result<u64> {
        Ok((index as u64).pow(2))
    }

    fn len(&self) -> usize {
        self.0
    }
}

assert_eq!(Squares(10).get(3).unwrap(), 9);
```

| Method | Required | Called |
|---|---|---|
| `get(&self, index)` | yes | per item: by `get_batch`'s default, and by the item-parallel path (`parallel_items`) on several threads at once |
| `len(&self)` | yes | at every `iter()`, to size the epoch |
| `get_batch(&self, indices)` | no, defaults to `get` per index | once per batch when items are not fetched in parallel; override it to amortise per-batch setup (one lock, one query, one file open) |
| `is_empty(&self)` | no | convenience |

- **Thread safety**: `Dataset: Send + Sync + 'static`, and `get` takes `&self`: it may run on
  several threads at the same time. Keep state immutable, or behind your own synchronisation.
- **Items**: `Item: Send + 'static`, since items cross from worker threads to the consumer.
- **Indices** come from the sampler; a custom sampler can produce any index, so `get` should
  return an error, not panic, on an index it cannot serve.

`IterableDataset` (streaming sources without random access) is defined for forward
compatibility; `DataLoader` does not drive it yet.

## Errors

`Error` is `Box<dyn std::error::Error + Send + Sync + 'static>`, so `?` converts any error type,
and a `&str` or `String` converts with `.into()`. An error is reported for its batch only: the
iterator yields `Err` for that batch and continues with the next one.
