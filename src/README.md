# `src/`: the crate

| Path | What | Guide |
|---|---|---|
| `lib.rs` | crate docs and the top-level re-exports | |
| `dataset/` | the sources: `sync.rs`, the `Dataset` trait (map-style) and `IterableDataset`; `asynchronous.rs` (feature `async`), `AsyncDataset`; `io/`, the built-in async datasets: `FileDataset`, `FetchError`, and what they share, `Retry` (`retry.rs`), `Column` and the template (`template/`), the list / template / function of locations (`location.rs`); `http`: `HttpDataset`; `object-store`: `ObjectStoreDataset` | below |
| `transform.rs` | `Transform`: per-item CPU work the loader applies, with the item's index; `Identity`; closures are transforms, `then` chains them | |
| `error.rs` | `Error` (`Box<dyn Error + Send + Sync>`) and `Result<T>` | below |
| `sampler/` | `Sampler`, `SequentialSampler`, `RandomSampler`, `DistributedSampler`, `BatchSampler` | [sampler/README.md](sampler/README.md) |
| `collator/` | `Collator`, `VecCollator`, `DefaultCollator` / `DefaultCollate`, `TorchPinnedCollator` | [collator/README.md](collator/README.md) |
| `loader/` | `DataLoader`, `DataLoaderBuilder`, `DataLoaderIter`, `SyncSource` / `AsyncSource`, the worker threads | [loader/README.md](loader/README.md) |
| `python/` | the PyO3 module behind the Python package (feature `python`) | [python/README.md](python/README.md) |

Top-level re-exports: `Dataset`, `IterableDataset`, `Sampler`, `SequentialSampler`,
`RandomSampler`, `DistributedSampler`, `BatchSampler`, `Collator`, `VecCollator`, `DataLoader`,
`DataLoaderBuilder`, `DataLoaderIter`, `SyncSource`, `Transform`, `Identity`, `Error`, `Result`;
feature `async`: `AsyncDataset`, `AsyncSource`, `MAX_RUNTIME_THREADS`.

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
| `get(&self, index)` | yes | per item: by `get_batch`'s default, and by the item lanes (`item_concurrency > 1`) on several threads at once |
| `len(&self)` | yes | at every `iter()`, to size the epoch |
| `get_batch(&self, indices)` | no, defaults to `get` per index | once per batch when items are not fetched in parallel; override it to amortise per-batch setup (one lock, one query, one file open) |
| `is_empty(&self)` | no | convenience |

- **Thread safety**: `Dataset: Send + Sync + 'static`, and `get` takes `&self`: it may run on
  several threads at the same time. Keep state immutable, or behind your own synchronisation.
- **Items**: `Item: Send + 'static`, since items cross from worker threads to the consumer.
- **Indices** come from the sampler; a custom sampler can produce any index, so `get` should
  return an error, not panic, on an index it cannot serve.

## The `AsyncDataset` contract (feature `async`)

The same, with `get` (and `get_batch`) returning a future, for items that wait:

```rust,ignore
impl AsyncDataset for Tiles {
    type Item = bytes::Bytes;
    async fn get(&self, index: usize) -> Result<bytes::Bytes> { /* await a request */ }
    fn len(&self) -> usize { self.urls.len() }
}
let loader = DataLoader::builder_async(Tiles { .. }).item_concurrency(32).build();
```

- `get` is awaited `item_concurrency` at a time per batch, `num_workers` batches at once;
  `get_batch` (default: `get` per index in turn) is used when `item_concurrency` is 1.
- Dropping a future cancels its fetch (an iterator dropped mid-epoch drops them all).
- CPU work belongs in the loader's `transform`, which runs on its rayon pool, not in `get`.
- Built in, all returning `Bytes` and capped at `max_bytes` (64 MiB by default): `HttpDataset`
  (headers; rustls with bundled roots), `FileDataset`, `ObjectStoreDataset` (S3, GCS, Azure,
  HTTP stores). Each builder takes the locations as a list (`urls`, `paths`, `keys` or
  `ranges`), a function of the index (`url_fn`, `path_fn`, `key_fn`) or a template
  (`url_template`, `path_template`, `key_template`; `_len` variants for `{index}` only): parsed
  once, with one `Column` of values per placeholder, each location made when its item is
  fetched. Template strings are percent-encoded in URLs; in paths and keys they are written as
  they are, but cannot hold `/` (nor `\` or NUL in a path) or be `.` / `..`.
- One `Retry` (`max_retries`, `initial_backoff`, `max_backoff`) for all three; each dataset
  decides what is transient: HTTP connection errors, timeouts, 408/429/5xx and `Retry-After`
  (3 retries by default); file timeouts and `TimedOut` / `Interrupted` / `WouldBlock` I/O
  errors (none by default; `timeout` per attempt for network file systems); object stores retry
  by themselves (`object_store`'s `RetryConfig`, set from `Retry`; `timeout` per request).
  Errors are a `FetchError`.

`IterableDataset` (streaming sources without random access) is defined for forward
compatibility; `DataLoader` does not drive it yet.

## Errors

`Error` is `Box<dyn std::error::Error + Send + Sync + 'static>`, so `?` converts any error type,
and a `&str` or `String` converts with `.into()`. An error is reported for its batch only: the
iterator yields `Err` for that batch and continues with the next one.
