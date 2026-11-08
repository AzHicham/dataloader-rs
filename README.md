# dataloader-rs

[![CI](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/dataloader-rs)](https://crates.io/crates/dataloader-rs)
[![docs.rs](https://img.shields.io/docsrs/dataloader-rs)](https://docs.rs/dataloader-rs)
[![PyPI](https://img.shields.io/pypi/v/dataloader-rs)](https://pypi.org/project/dataloader-rs/)
[![MSRV](https://img.shields.io/badge/rustc-1.95%2B-orange.svg)](#minimum-supported-rust-version)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PyTorch's `DataLoader` model — dataset, sampler, transform, collator, loader — as a typed Rust
library, with a Python binding that runs the same pipeline on threads instead of worker processes.

- **Familiar model, typed end to end**: `DataLoader<Dataset, Sampler, Collator, Transform>` is
  generic and monomorphized; batches come out in epoch order as `Result<Batch>`.
- **A dataset is a source, the transform is the loader's**: a dataset, sync or async, only
  fetches; the per-item transform (decode, augment) is a loader setting, run on the loader's
  threads with the item's index, in parallel across the items of a batch.
- **Threads, not processes**: workers share the dataset through `Arc`, so samples are handed
  over by reference instead of being pickled across processes. Against
  `torch.utils.data.DataLoader` on 4 cores, the Python binding is **4.9–6.6× faster on the
  loader's own overhead, 3.2–3.9× on network-bound datasets, 2.5–3.5× on array samples** with
  free-threaded Python, and starts an epoch in milliseconds
  ([BENCH.md](BENCH.md)).
- **Two explicit levels of parallelism**: `num_workers` batches in parallel and
  `item_concurrency` items of each batch — on one thread pool built with the loader and reused
  by every epoch.
- **Async datasets**: an `AsyncDataset` is fetched by futures on a tokio runtime,
  `num_workers × item_concurrency` at once, and collated on the loader's threads; built in:
  `HttpDataset`, `FileDataset`, `ObjectStoreDataset` (S3, GCS, Azure), each over a list, a
  template with per-item values or a function of the index, with retries, timeouts and a size
  cap. In Python, `PyAsyncDataset` takes `async def __getitem__`, and the Rust datasets fetch
  without the GIL: 2 048 HTTP tiles in 0.38 s against 1.6–3.0 s for torch's worker processes.
- **Bounded, ordered prefetch**: at most `num_workers + prefetch_depth` batches are in flight or
  waiting, even behind one slow batch; they come out in epoch order.
- **Failures surface**: a dataset or collator error is returned for its batch, a panic is
  re-raised on the consumer, and Python exceptions keep their type.
- **Python binding**: `PyDataloader` mirrors the PyTorch arguments (`batch_size`, `shuffle`,
  `sampler`, `collate_fn`, `drop_last`, `seed`, `__getitems__`), plus `transform`, called as
  `f(index, item)`; wheels for CPython 3.11 to 3.15
  and free-threaded 3.14t / 3.15t on Linux (x86_64, arm64) and macOS (arm64).
- **No unsafe code**: `#![forbid(unsafe_code)]`.

> **When PyTorch is faster.** The workers are threads in one process, not processes. On regular
> CPython (the GIL build, such as 3.13), only one thread runs Python code at a time: a
> `__getitem__` that spends its time in pure Python (tokenising text, parsing records, Python
> loops) gets no faster with more workers, while PyTorch's worker processes each run their own
> interpreter and are 2.4–2.6× faster on such datasets. Threads win when the work releases the
> GIL (file and network I/O, NumPy, image decoding), and on free-threaded CPython 3.14t / 3.15t.
> Even there, threads that all read one shared Python dict or list contend on its reference
> count, and PyTorch is 2.5–2.9× faster: give each thread its own copy (`threading.local`).
> Measurements: [BENCH.md](BENCH.md#1-summary).

## Contents

- [Quick start: Rust](#quick-start-rust)
- [Quick start: Python](#quick-start-python)
- [Async datasets](#async-datasets)
- [Concepts](#concepts)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Feature flags](#feature-flags)
- [Performance](#performance)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Building and testing](#building-and-testing)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [License](#license)

## Quick start: Rust

```toml
[dependencies]
dataloader-rs = "0.1"
```

```rust
use dataloader_rs::{DataLoader, Dataset, RandomSampler, error::Result};

/// One sample per file; `get` is called from worker threads.
struct Files(Vec<std::path::PathBuf>);

impl Dataset for Files {
    type Item = Vec<u8>;

    fn get(&self, index: usize) -> Result<Vec<u8>> {
        Ok(std::fs::read(&self.0[index])?)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn main() -> Result<()> {
    let paths = std::fs::read_dir("data")?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;

    let mut loader = DataLoader::builder(Files(paths))
        .batch_size(32)
        .sampler(RandomSampler::new(42)) // reproducible shuffle
        .num_workers(4)                  // 4 batches in flight
        .prefetch_depth(8)               // run up to 8 batches ahead
        .drop_last(true)
        .build();

    for epoch in 0..3 {
        for batch in &mut loader {
            let batch: Vec<Vec<u8>> = batch?; // VecCollator: a Vec of samples
            println!("epoch {epoch}: {} samples", batch.len());
        }
    }
    Ok(())
}
```

`cargo run --example basic` runs a complete example.

## Quick start: Python

```bash
pip install dataloader-rs
```

```python
from dataloader_rs import PyDataloader, PyDataset


class Files(PyDataset):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> bytes:
        with open(self.paths[index], "rb") as f:
            return f.read()


loader = PyDataloader(Files(paths), batch_size=32, shuffle=True, seed=0, num_workers=4)
for batch in loader:  # a list of 32 `bytes`, or whatever collate_fn returns
    ...
```

The full Python guide — arguments, samplers, `collate_fn`, `__getitems__`, async datasets,
errors, threads and the GIL — is [python/README.md](python/README.md).

## Async datasets

For items that wait (network, object storage), an `AsyncDataset` returns a future from `get`,
and `DataLoader::builder_async` takes the same options as `builder`: `num_workers` batches (at
least one) are fetched at once, each with `item_concurrency` futures in flight, on the loader's
own tokio runtime (one thread per available CPU, or `.runtime_threads(n)`; or yours,
`.runtime(handle)`); items are transformed and batches collated on the loader's threads, and
come out in epoch order through the same iterator. Features `async`, `http`, `object-store`.

```rust
use dataloader_rs::{DataLoader, dataset::{Column, HttpDataset}, error::Result};

fn main() -> Result<()> {
    let (xs, ys): (Vec<i64>, Vec<i64>) = (0..64 * 64).map(|i| (i % 64 * 224, i / 64 * 224)).unzip();
    let tiles = HttpDataset::builder()
        .header("Authorization", "Bearer <token>")
        .url_template(
            "https://imgsrv.example/slides/slide-1/tile?x={x}&y={y}&w=224&h=224",
            [("x", Column::from(xs.clone())), ("y", Column::from(ys.clone()))],
        )?;

    let mut loader = DataLoader::builder_async(tiles)
        .batch_size(64)
        .num_workers(2)
        .item_concurrency(32) // 64 requests in flight
        // CPU work, on the loader's pool as each tile arrives (decode it here); the index
        // picks the tile's coordinates
        .transform(move |i: usize, png: bytes::Bytes| -> Result<(usize, (i64, i64))> {
            Ok((png.len(), (xs[i], ys[i])))
        })
        .build();
    for batch in &mut loader {
        for (size, (x, y)) in batch? {
            println!("tile ({x}, {y}): {size} bytes");
        }
    }
    Ok(())
}
```

A dataset, sync or async, only fetches. The loader's `Transform` (`.transform(t)`; closures
`Fn(usize, In) -> Result<Out>` taking the item's index, chained with `then`) applies per item:
right after `get` on the worker for a `Dataset`, on the loader's pool for an `AsyncDataset`,
never on the runtime's threads. The index is the item's dataset index, whatever the sampler.
One failed item fails its batch, a panic is re-raised by `next()`, and
dropping the iterator aborts every future in flight without waiting.

The built-in datasets return `Bytes`, capped at `max_bytes` (64 MiB by default), and share how
they find item `i` and when they try it again:

| Dataset | Item `i` | List | Template | Function | Retried (default retries) | Timeout (default) |
|---|---|---|---|---|---|---|
| `HttpDataset` | body of a GET | `urls` | `url_template` | `url_fn` | connection errors, timeouts, 408, 429, 5xx; `Retry-After` honoured (3) | per attempt (30 s), connect (10 s) |
| `FileDataset` | a file | `paths` | `path_template` | `path_fn` | timeouts, I/O errors `TimedOut`, `Interrupted`, `WouldBlock` (0) | per attempt (none) |
| `ObjectStoreDataset` | an object, or a byte range (`ranges`) | `keys` | `key_template` | `key_fn` | by `object_store` itself: 5xx, 429, 408, connection errors, timeouts (3) | per request (30 s) |

Each builder ends with one of these. A template (`{name}`: the item's value in column `name`,
a `Column` of integers or strings; `{index}`; `{{` and `}}`) is parsed once, and each location
made when its item is fetched. Strings are percent-encoded in a URL, written as they are in a
path or a key, where a value cannot hold `/` nor be `.` or `..`: it cannot leave the template's
directory. One `Retry` (`max_retries`, `initial_backoff`, `max_backoff`; doubling, with jitter)
configures all three; `ObjectStoreDataset` hands it to the store it opens, which retries itself.

```rust
use std::time::Duration;

use dataloader_rs::{dataset::{Column, FileDataset, ObjectStoreDataset, Retry}, error::Result};

fn datasets() -> Result<(FileDataset, ObjectStoreDataset)> {
    let slides = Column::from(vec!["a.svs", "a.svs", "b.svs"]);
    // /mnt/tiles/a.svs/0.png, /mnt/tiles/a.svs/1.png, /mnt/tiles/b.svs/2.png
    let files = FileDataset::builder()
        .timeout(Duration::from_secs(10)) // a hung network file system fails the attempt
        .retry(Retry { max_retries: 2, ..Retry::default() })
        .path_template("/mnt/tiles/{slide}/{index}.png", [("slide", slides.clone())])?;
    // s3://bucket/tiles/a.svs/0.png, ...
    let objects = ObjectStoreDataset::from_url("s3://bucket/tiles", [("aws_region", "eu-west-3")])
        .retry(Retry { max_retries: 5, ..Retry::default() })
        .key_template("{slide}/{index}.png", [("slide", slides)])?;
    Ok((files, objects))
}
```

## Concepts

| Piece | Trait / type | Provided | Guide |
|---|---|---|---|
| Dataset | `Dataset`: `get(index)`, `len()`, optional `get_batch(indices)` | — (yours) | [src/README.md](src/README.md) |
| Async dataset | `AsyncDataset`: `async get(index)`, `len()` (feature `async`) | `HttpDataset`, `FileDataset`, `ObjectStoreDataset` | [src/README.md](src/README.md) |
| Transform | `Transform<In>`: per-item CPU work `apply(index, item)`, set on the loader; closures; `then` | `Identity` (default) | [src/transform.rs](src/transform.rs) |
| Sampler | `Sampler`: the index order of one epoch | `SequentialSampler`, `RandomSampler`, `DistributedSampler` | [src/sampler](src/sampler/README.md) |
| Collator | `Collator<Item>`: merge a batch's samples | `VecCollator` (default), `DefaultCollator`, `TorchPinnedCollator` | [src/collator](src/collator/README.md) |
| Loader | `DataLoader` / `DataLoaderBuilder` / `DataLoaderIter`; the dataset held as a `SyncSource` or an `AsyncSource` | — | [src/loader](src/loader/README.md) |

One epoch: the sampler produces indices, `BatchSampler` cuts them into batches of `batch_size`,
each batch is fetched (`Dataset::get_batch`, or `get` per item in parallel), transformed item by
item and collated, and the iterator yields the batches in order. How the threads, the bounded prefetch and the reorder
buffer fit together is in [ARCHITECTURE.md](ARCHITECTURE.md).

## How it works

Every loader has the same front: the sampler gives the epoch's indices, `iter()` cuts them into
batches, and the consumer reads batches **in epoch order** with `next()`. What runs where depends
on the dataset.

### Sync dataset (`DataLoader::builder`)

`num_workers = 0`: `next()` fetches, transforms and collates the batch on the consumer's thread
(no pool, no hand-off; with `item_concurrency > 1` its items are fetched on a small pool).

`num_workers > 0` (here `num_workers = 2`, `item_concurrency = 3`):

```text
 sampler ──▶ batches  b0  b1  b2  b3  b4 ...    claimed in order, at most
                      │   │                      num_workers + prefetch_depth ahead of next()
                      ▼   ▼
 ┌─ loader's rayon pool: max(num_workers, 1) × item_concurrency threads ────────────┐
 │                                                                                  │
 │  batch task (b0)                          batch task (b1)                        │
 │    lane: get(i) → transform(i, item) ┐      lane: get → transform ┐              │
 │    lane: get(i) → transform(i, item) ├─▶    lane: get → transform ├─▶ collate    │
 │    lane: get(i) → transform(i, item) ┘ │    lane: get → transform ┘     │        │
 │                                     collate                             │        │
 └──────────────────────────────────────┼──────────────────────────────────┼────────┘
                                        ▼  batches finish in any order     ▼
                         reorder ring (num_workers + prefetch_depth slots)
                                        ▼  in epoch order
                         consumer thread: for batch in loader → next()
```

- A batch task claims the next batch, fetches its items in `item_concurrency` lanes (each lane:
  `get`, then the transform, on the same thread), collates, and claims again; when nothing is
  claimable it exits instead of blocking a thread.
- With `item_concurrency = 1` a batch is one `get_batch` call, then transforms, then collate.
- Memory is bounded: at most `num_workers + prefetch_depth` batches are in flight or waiting.

### Async dataset (`DataLoader::builder_async`)

Fetches are futures on tokio, CPU work is on rayon, and the two never mix:

```text
 ┌─ loader's tokio runtime: runtime_threads threads (or yours: runtime(handle)) ──────────┐
 │                                                                                        │
 │  driver task ── batches in epoch order, num_workers batch tasks at once                │
 │     batch task (b0): get(i).await  ×  item_concurrency in flight ──┐                   │
 │     batch task (b1): get(i).await  ×  item_concurrency in flight ──┤                   │
 └────────────────────────────────────────────────────────────────────┼───────────────────┘
                              each item as it arrives                 │ ▲ result back
                                                                      ▼ │ (oneshot)
 ┌─ loader's rayon pool: transforms and collation ────────────────────────────────────────┐
 │   transform(i, item) per item   ·   collate once the batch's items are all in          │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
             driver: in order, waits for room (prefetch_depth) ──▶ channel ──▶ next()
```

- `num_workers × item_concurrency` = requests in flight; each costs a future, not a thread.
- A batch's transforms run while its other fetches are still in flight.
- `runtime_threads` is CPU for the network (receive, TLS, body copy); the rayon pool is CPU for
  your transform and collate: `min(num_workers × item_concurrency, CPUs)` threads with a
  transform, `max(num_workers, 1)` without.

### Python

The same two paths run under the Python `PyDataloader`; what changes is who holds the GIL.

```text
 PyDataset            pool thread ── __getitem__ (GIL) ── transform (GIL) ── collate_fn (GIL)

 HttpDataset,         tokio ── fetch in Rust, no GIL ──▶ pool thread ── to bytes + transform (GIL,
 FileDataset, ...                                                       only for these calls)

 PyAsyncDataset       tokio batch task needs item i
                        └─▶ rayon global pool thread: coro = __getitem__(i) (GIL),
                            run_coroutine_threadsafe(coro, loop)
                              └─▶ "dataloader-asyncio" thread: the event loop awaits coro
                                    └─ done ──▶ oneshot ──▶ batch task ──▶ transform on pool
```

- The consumer waits for a batch **without** the GIL, so training code keeps running.
- On a free-threaded build (3.14t) Python code in workers runs in parallel; under the GIL, only
  Rust work (fetching for the Rust datasets, waiting) and I/O release it.

### Threads at a glance

| Thread(s) | Created | Count | Runs |
|---|---|---|---|
| loader's rayon pool `dataloader-N` | `build()`, one per loader | see above | batch tasks, `get`, transforms, collation |
| loader's tokio runtime `dataloader-io` | `build()` of an async loader, unless `runtime(handle)` | `runtime_threads` (default: CPUs) | fetch futures, driver and batch tasks |
| rayon global pool `dataloader-map-N` | first loader with a `PyAsyncDataset` (Python) | CPUs, shared by the process | creates and submits coroutines |
| `dataloader-asyncio` | each loader with a `PyAsyncDataset` (Python) | 1 | the asyncio event loop; `aclose()` on drop |

Details (claim window, reorder ring, failures, `fork()`) are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration

Every setting is a `DataLoaderBuilder` method; the Python argument of the same name maps to it.

| Builder method | Default | Effect |
|---|---|---|
| `batch_size(n)` | `1` | samples per batch |
| `drop_last(bool)` | `false` | drop a final batch smaller than `batch_size` |
| `sampler(s)` | `SequentialSampler` | index order of each epoch |
| `transform(t)` | `Identity` | per-item work `t.apply(index, item)` before collation |
| `collator(c)` | `VecCollator` | how a batch's samples are merged |
| `num_workers(n)` | `0` | batches processed in parallel; `0` processes batches inside `next()` |
| `item_concurrency(k)` | `1` | items of one batch fetched in parallel (`1`: one `get_batch` call per batch); formerly `items_in_parallel`, still accepted (deprecated) |
| `prefetch_depth(n)` | `2` | how far batches may run ahead: at most `num_workers + n` in flight or waiting |
| `runtime_threads(n)` | CPUs available | async only: worker threads of the loader's tokio runtime, 1 to `MAX_RUNTIME_THREADS` (1024) |
| `pin_memory(bool)` | — | pin `tch::Tensor` batches in host memory (feature `torch-rs`) |

At most `max(num_workers, 1) × item_concurrency` fetches run at once, on that many threads, or
as that many futures for an async dataset (`builder_async`, whose pool, for transforms and
collation, has as many threads up to the CPUs available with a transform, `num_workers`
without). `runtime(handle)` (async only) runs the futures on your tokio runtime; it excludes
`runtime_threads`. An invalid setting (`0`, above the maximum, both together) panics
in the builder, like `batch_size(0)`.

Sizing an async loader:

| Setting | Sizes | Rule |
|---|---|---|
| `num_workers × item_concurrency` | requests in flight | latency × wanted items/s; far more than cores |
| `runtime_threads` | CPU for network receive, TLS, body copy | default (CPUs available) is enough; lower it to leave cores to the rest |
| the loader's rayon pool | transforms and collation | the CPU-bound part |

Rules of thumb: `num_workers = 0` is the fastest path for cheap samples (no thread hand-off);
use workers once a batch costs well more than its hand-off, about 0.5 µs in Rust (a few
microseconds through Python); for blocking I/O, `item_concurrency` so that
`num_workers × item_concurrency` requests are in flight, far more than cores; the default
`prefetch_depth` 2 is enough when loading keeps up, beyond that it only holds memory
([BENCH.md](BENCH.md#4-results-rust)).

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `ndarray` | off | `DefaultCollator` for `ndarray::Array` (stacked on a new axis 0) |
| `torch-rs` | off | `DefaultCollator` for `tch::Tensor` (stacked), `TorchPinnedCollator`, `pin_memory` |
| `async` | off | `AsyncDataset`, `DataLoader::builder_async`, `FileDataset` (tokio, futures-util) |
| `http` | off | `HttpDataset`: reqwest over rustls with the *ring* provider and the bundled Mozilla roots |
| `object-store` | off | `ObjectStoreDataset`: S3, GCS, Azure and HTTP stores of the `object_store` crate, same TLS |
| `python` | off | the PyO3 module behind the `dataloader-rs` Python package; enables `async`, `http`, `object-store` |

## Performance

Measured on an Intel Core i9-13900H pinned to 4 cores, the size of the CI runners, on CPython
3.13, 3.14t and 3.15t; method, every table and the caveats are in [BENCH.md](BENCH.md).
Continuous results per commit: dashboards for [Rust](https://azhicham.github.io/dataloader-rs/benchmarks/rust/) and [Python 3.14t](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| Python workload, against `torch.utils.data.DataLoader` (best configuration of each) | Result |
|---|---|
| Loader overhead (in-memory samples) | **4.9–6.6× faster** |
| Network-bound samples (10 ms each), `item_concurrency` | **3.2–3.9× faster**; first batch 21 ms against 320+ ms |
| 2 048 HTTP tiles of 64 KiB (10 ms each), `HttpDataset` against torch with 8 / 16 workers | **4.3–7.8× faster** (0.38 s against 1.64–2.99 s), on 3.13 and 3.14t |
| First batch of an epoch | **milliseconds**, against 11–334 ms for torch's fresh worker processes |
| Array samples: read + normalise a file, augment an image (4 workers) | **2.5–3.5× faster** on 3.14t / 3.15t, 1.0–1.5× on 3.13; with the allocator settings of [BENCH.md, section 8](BENCH.md#8-caveats-and-known-bottlenecks) |
| Pure-Python CPU work, per-thread tables, free-threaded 3.14t / 3.15t | **≈ 1.2× faster** |
| Pure-Python CPU work on shared objects, or under the GIL | torch 2.4–2.9× faster (processes share nothing) |

## Documentation

| Document | Content |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | execution paths, threads, backpressure, ordering, errors, panics, the Python binding |
| [BENCH.md](BENCH.md) | what is measured and how, results, comparison with PyTorch, caveats, how to rerun |
| [python/README.md](python/README.md) | the Python package (also its PyPI page) |
| [src/README.md](src/README.md) | code map, and the `Dataset` contract |
| [src/loader](src/loader/README.md), [src/sampler](src/sampler/README.md), [src/collator](src/collator/README.md), [src/python](src/python/README.md) | one guide per module (also on docs.rs) |
| [bench/](bench/README.md), [benches/](benches/README.md), [tests/](tests/README.md), [examples/](examples/README.md) | what each script, suite and test file covers |
| [CHANGELOG.md](CHANGELOG.md) | release notes (generated by semantic-release) |

## Project layout

| Path | What |
|---|---|
| [`src/`](src/README.md) | the crate: `dataset`, `sampler`, `collator`, `loader`, `python` |
| [`dataloader_rs/`](dataloader_rs/__init__.pyi) | Python package: re-exports and type stubs |
| [`python/`](python/README.md) | Python package README (PyPI) |
| [`tests/`](tests/README.md) | Rust integration tests and the pytest suite |
| [`benches/`](benches/README.md) | Criterion benchmarks (Rust) |
| [`bench/`](bench/README.md) | Python benchmarks: pytest-benchmark guards and comparisons with PyTorch |
| [`examples/`](examples/README.md) | runnable Rust examples |

## Building and testing

Prerequisites: Rust ≥ 1.95, [uv](https://docs.astral.sh/uv/). The development environment uses
free-threaded CPython 3.14t (`.python-version`); for a GIL build prefix commands with
`uv run --python 3.13`.

```bash
# Rust
cargo build
cargo test --features ndarray,python          # add torch-rs to test tch (downloads libtorch)

# Python
uv sync --group dev                            # maturin, pytest, numpy, torch
uv run maturin develop --features python       # build the extension into the venv
uv run pytest                                  # tests/ (benchmarks are deselected)

# Lint, as CI does
CI=1 prek run --all-files                      # ruff, ty, cargo fmt, clippy -D warnings
```

Benchmarks: [BENCH.md, section 9](BENCH.md#9-running-it-again).

Commits follow [Conventional Commits](https://www.conventionalcommits.org/); releases, the
changelog, the crate and the wheels are published by semantic-release from `main`.

## Minimum supported Rust version

Rust **1.95** (`rust-version` in `Cargo.toml`), checked in CI by the `Rust tests (MSRV)` job.

## License

MIT — see [LICENSE](LICENSE).
