# Data sources (feature `sources`)

Where the bytes of each item come from, as a `Dataset`: memory, files, or HTTP. A `Source` maps
an index to bytes; `SourceDataset` adds per-item metadata (a label, an id, coordinates) and
yields `(bytes, meta)` items. `ImageCollator` (feature `image`) takes these as they are and
returns the metadata with the batch.

```rust,no_run
use dataloader_rs::{
    DataLoader,
    sources::{HttpSource, SourceDataset},
};

let urls: Vec<String> = (0..1024).map(|i| format!("http://images.internal/{i}.jpeg")).collect();
let labels: Vec<u32> = (0..1024).map(|i| i % 10).collect();
let dataset = SourceDataset::with_meta(HttpSource::urls(urls)?, labels)?;

let mut loader = DataLoader::builder(dataset)
    .batch_size(64)
    .num_workers(4)
    .items_in_parallel(8) // 32 requests in flight
    .build();
for batch in &mut loader {
    let batch: Vec<(Vec<u8>, u32)> = batch?; // (encoded image, label) per item
}
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

With the `image` feature, `.collate_into(ImageCollator::<f32>::new(transform)?, allocator)`
decodes and transforms the bytes into a pooled batch buffer instead; each item's label is then
in `batch.output()[i].meta`.

## Sources

| Source | Item | Bytes |
|---|---|---|
| `MemorySource<B>` | an element of a `Vec<B>`, cloned (`Arc<[u8]>` by default) | `B` |
| `FileSource` | a path; the whole file is read | `Vec<u8>` |
| `HttpSource` | a URL, from a list or a template; one `GET` | `Vec<u8>` |
| your own | `impl Source` | any `AsRef<[u8]>` |

A source is called from the worker threads, one item per call. A batch fetches its items one
after the other unless the loader's `items_in_parallel` is above 1: for HTTP, set it so that
`num_workers × items_in_parallel` requests are in flight.

## `HttpSource`

```rust,no_run
use std::time::Duration;
use dataloader_rs::sources::{HttpSource, Retry};

let source = HttpSource::builder()
    .header("Authorization", "Bearer <token>")
    .timeout(Duration::from_secs(10))       // per attempt, connect to last byte; default 30 s
    .connect_timeout(Duration::from_secs(2)) // default 10 s
    .retry(Retry { max_retries: 5, ..Retry::default() })
    .max_bytes(16 << 20)                    // per response; default 64 MiB
    .max_idle_connections(64)               // kept alive for reuse
    .template(
        "http://tiles.internal/{slide}/{x}_{y}.jpeg",
        [[("slide", "a1".to_string()), ("x", "0".into()), ("y", "0".into())]],
    )?;
# Ok::<(), dataloader_rs::sources::SourceError>(())
```

- **Client**: [`ureq`] 3, blocking, HTTP/1.1. Connections are kept alive and reused across items
  and batches. Redirects are followed (5 by default). Proxies come from the usual environment
  variables. Responses are not decompressed.
- **URLs**: a list, or a template whose `{name}`s are replaced per item. Template values are
  percent-encoded except `A-Z a-z 0-9 - . _ ~`, so a value cannot add a path segment or a query.
  Every URL is parsed when the source is built.
- **Retries**: on connection errors, timeouts, truncated bodies, and statuses 408, 429, 500,
  502, 503, 504, with exponential backoff and jitter, honouring a `Retry-After` in seconds up to
  `max_backoff`. Other statuses and malformed responses fail at once.
- **Size cap**: a `Content-Length` above `max_bytes` fails before the body is read; a body
  without one is read up to `max_bytes + 1` bytes, then fails. Memory is reserved fallibly.
- **Secrets**: header values are never shown; errors show URLs without user info, query and
  fragment.

### TLS

`sources` fetches plain `http://` only: its dependencies are pure Rust, with no C. The
`sources-tls` feature adds `https://` through rustls with webpki root certificates; rustls'
crypto provider (ring) contains C and assembly. An `https://` URL without the feature is an
error when the source is built.

## `FileSource`

`FileSource::new(paths).max_bytes(n)` reads each file whole. A file above `max_bytes` (default
64 MiB) fails, checked against its length before reading and while reading. Paths are used as
given.

## Errors

`SourceError` names the item and where it came from:
`item 3 (http://host/a.png): HTTP status 503 after 4 attempt(s)`. In a loader, a failed item
fails its batch only, and the next batch loads. Downcast the loader's error to `SourceError` to
inspect it.

[`ureq`]: https://docs.rs/ureq
