# `src/`: the crate

| Path | What | Guide |
|---|---|---|
| `lib.rs` | The crate's documentation and the top-level re-exports. | |
| `dataset/` | The sources. `sync.rs` holds the `Dataset` trait (random access by index) and `IterableDataset`; `asynchronous.rs` holds `AsyncDataset` (feature `async`). `io/` holds the built-in async datasets (`FileDataset`, `HttpDataset` with the `http` feature, `ObjectStoreDataset` with the `object-store` feature), their error type `FetchError`, and what they share: `Retry` (`retry.rs`), `Column` and the templates (`template/`), and the list, template or function of locations (`location.rs`). | below |
| `transform.rs` | `Transform`, the per-item CPU work the loader applies, which receives the item's index; `Identity`, the default. Closures are transforms, and `then` chains two. | |
| `error.rs` | `Error` (a `Box<dyn Error + Send + Sync>`) and `Result<T>`. | below |
| `sampler/` | `Sampler`, `SequentialSampler`, `RandomSampler`, `DistributedSampler` and `BatchSampler`. | [sampler/README.md](sampler/README.md) |
| `collator/` | `Collator`, `VecCollator`, `DefaultCollator` and its `DefaultCollate` trait, `TorchPinnedCollator`, and `ImageCollator` with its `ImageBatch` (feature `image`). | [collator/README.md](collator/README.md) |
| `image/` | Feature `image`: the `Image<T, L>` type, `Decode`, and the colour, crop, pad, resize, flip, rotate, float conversion, normalisation and layout transforms; `ImageError`. | [image/README.md](image/README.md) |
| `loader/` | `DataLoader`, `DataLoaderBuilder`, `DataLoaderIter`, `IndexedIter`, `SyncSource` and `AsyncSource`, and the worker threads. | [loader/README.md](loader/README.md) |
| `python/` | The PyO3 module behind the Python package (feature `python`). | [python/README.md](python/README.md) |

Top-level re-exports: `Dataset`, `IterableDataset`, `Sampler`, `SequentialSampler`,
`RandomSampler`, `DistributedSampler`, `BatchSampler`, `Collator`, `VecCollator`, `DataLoader`,
`DataLoaderBuilder`, `DataLoaderIter`, `IndexedIter`, `SyncSource`, `Transform`, `Identity`,
`Error`, `Result`; feature `async`: `AsyncDataset`, `AsyncSource`, `MAX_RUNTIME_THREADS`.

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
| `get(&self, index)` | yes | Once per item: by the default `get_batch`, and by the item lanes (`item_concurrency > 1`), on several threads at once. |
| `len(&self)` | yes | At every `iter()`, to size the epoch. |
| `get_batch(&self, indices)` | no; by default it calls `get` for each index | Once per batch, when the items are not fetched in parallel. Override it to share setup across a batch (one lock, one query, one opened file). |
| `is_empty(&self)` | no | Never by the loader; a convenience. |

- **Thread safety**: `Dataset: Send + Sync + 'static`, and `get` takes `&self`: it may run on
  several threads at the same time. Keep state immutable, or behind your own synchronisation.
- **Items**: `Item: Send + 'static`, since items move from the worker threads to the consumer.
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

- The loader awaits `get` for up to `item_concurrency` items of each batch at a time, and for
  `num_workers` batches at once. When `item_concurrency` is 1, it calls `get_batch` instead,
  which by default awaits `get` for each index in turn.
- Dropping a future cancels its fetch; an iterator dropped in the middle of an epoch drops them
  all.
- CPU work belongs in the loader's `transform`, which runs on the loader's rayon pool, not in
  `get`, which runs on the async runtime.
- Three datasets are built in, all returning `Bytes` capped at `max_bytes` (64 MiB by default):
  `HttpDataset` (with headers; TLS by rustls with the bundled Mozilla roots), `FileDataset`, and
  `ObjectStoreDataset` (S3, Google Cloud Storage, Azure and HTTP stores).
- Each builder takes the locations as a list (`urls`, `paths`, `keys` or `ranges`), as a
  function of the index (`url_fn`, `path_fn`, `key_fn`), or as a template (`url_template`,
  `path_template`, `key_template`, and their `_len` variants for templates that use only
  `{index}`). A template is parsed once, with one `Column` of values per placeholder, and each
  location is made only when its item is fetched. String values are percent-encoded in URLs;
  in paths and keys they are written as they are, but cannot contain `/` (nor `\` or NUL in a
  path) or be `.` or `..`.
- One `Retry` type (`max_retries`, `initial_backoff`, `max_backoff`) configures all three, and
  each dataset decides which failures are transient. `HttpDataset` retries connection errors,
  time-outs, the statuses 408, 429 and 5xx, and honours `Retry-After` (3 retries by default).
  `FileDataset` retries time-outs and the I/O errors `TimedOut`, `Interrupted` and `WouldBlock`
  (no retries by default; set a `timeout` per attempt for network file systems). Object stores
  retry by themselves, with the `object_store` crate's `RetryConfig` set from `Retry` and a
  `timeout` per request. Failures are reported as a `FetchError`.

`IterableDataset` (a streaming source without random access) is defined for forward
compatibility; `DataLoader` cannot load from it yet.

## Errors

`Error` is `Box<dyn std::error::Error + Send + Sync + 'static>`, so `?` converts any error type,
and a `&str` or `String` converts with `.into()`. An error is reported for its batch only: the
iterator yields `Err` for that batch and continues with the next one.
