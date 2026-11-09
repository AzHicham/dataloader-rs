# dataloader-rs

[![CI](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/dataloader-rs)](https://crates.io/crates/dataloader-rs)
[![docs.rs](https://img.shields.io/docsrs/dataloader-rs)](https://docs.rs/dataloader-rs)
[![PyPI](https://img.shields.io/pypi/v/dataloader-rs)](https://pypi.org/project/dataloader-rs/)
[![MSRV](https://img.shields.io/badge/rustc-1.95%2B-orange.svg)](#minimum-supported-rust-version)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`dataloader-rs` loads training and inference data in batches, the way PyTorch's
`torch.utils.data.DataLoader` does: a dataset gives items by index, a sampler chooses their
order, a transform prepares each item, and a collator turns the items of a batch into one value.
It is a Rust library, and a Python package built on it. Unlike PyTorch, its workers are threads
in one process instead of separate processes, items can be fetched asynchronously (many network
requests in flight without a thread each), and image decoding and transforms can run in Rust
without holding Python's global interpreter lock (the GIL).

## Contents

- [Why use it](#why-use-it)
- [Installation](#installation)
- [Quick start: Rust](#quick-start-rust)
- [Quick start: Python](#quick-start-python)
- [Concepts](#concepts)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Built-in datasets](#built-in-datasets)
- [Image operations](#image-operations)
- [Performance](#performance)
- [Examples](#examples)
- [Documentation](#documentation)
- [Building and testing](#building-and-testing)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [License](#license)

## Why use it

- **Threads instead of processes.** The workers share the dataset in memory, so nothing is
  copied or pickled between processes, and an epoch starts in milliseconds instead of waiting
  for worker processes to start.
- **Asynchronous fetching.** A dataset can return futures (Rust) or coroutines (Python). The
  loader keeps many requests in flight, from a few threads, which suits object stores, HTTP
  APIs and network file systems. Ready-made datasets read files, objects and HTTP URLs.
- **Rust work without the GIL.** From Python, the built-in datasets fetch, and the image
  transforms decode and normalise, in Rust while the interpreter is free, and the batches reach
  NumPy and PyTorch without a copy.
- **The same model as PyTorch.** The pieces and the arguments (`batch_size`, `shuffle`,
  `sampler`, `num_workers`, `collate_fn`, `drop_last`) are the ones you know. Batches come out in
  order, an error fails only its own batch, and a Python exception keeps its type.
- **Bounded memory.** The number of batches being loaded or waiting is capped by two settings,
  even when one batch is much slower than the others.
- **Typed and safe Rust.** The loader is generic over its dataset, transform and collator, so
  the compiler checks how they fit together, and the crate contains no `unsafe` code.

Against PyTorch's `DataLoader` on 4 cores, the Python package is 4.9 to 6.6 times faster on the
loader's own overhead, 3.2 to 3.9 times faster on network-bound datasets, and 2.5 to 3.5 times
faster on array samples with free-threaded Python (see [Performance](#performance)).

> **When PyTorch is the better choice.** Because the workers are threads, they share one Python
> interpreter. On regular CPython (the builds with a GIL, such as 3.13), only one thread runs
> Python code at a time. A `__getitem__` that spends its time in pure Python code (tokenising
> text, parsing records, Python loops) therefore gets no faster with more workers, while
> PyTorch's worker processes each have their own interpreter and are 2.4 to 2.6 times faster on
> such datasets. Threads do well when the work releases the GIL (file and network I/O, NumPy,
> image decoding) and on the free-threaded builds of CPython (3.14t, 3.15t). Even there, threads
> that all read the same Python dict or list slow each other down on its reference count, and
> PyTorch is 2.5 to 2.9 times faster unless each thread gets its own copy (with
> `threading.local`). The measurements are in [BENCH.md](BENCH.md#1-summary).

## Installation

**Rust.** Add the crate, with the features you need:

```bash
cargo add dataloader-rs                          # the loader, samplers and collators
cargo add dataloader-rs --features http,image    # plus HttpDataset and the image operations
```

| Feature | Adds |
|---|---|
| `async` | `AsyncDataset`, `DataLoader::builder_async` and `FileDataset` (uses tokio). |
| `http` | `HttpDataset`, an HTTP client (reqwest over rustls, with the bundled Mozilla root certificates). Enables `async`. |
| `object-store` | `ObjectStoreDataset`, for S3, Google Cloud Storage, Azure and HTTP stores (the `object_store` crate). Enables `async`. |
| `image` | The `image` module (decoding and image transforms, in pure Rust) and `ImageCollator`. |
| `ndarray` | `DefaultCollator` stacks `ndarray::Array` items into one array. |
| `torch-rs` | `DefaultCollator` stacks `tch::Tensor` items; `pin_memory` and `TorchPinnedCollator`. |
| `python` | The PyO3 module behind the Python package. Enables `async`, `http`, `object-store` and `image`. |

No feature is enabled by default.

**Python.** Install the package from PyPI:

```bash
pip install dataloader-rs
```

| Wheels for | Platforms |
|---|---|
| CPython 3.11, 3.12, 3.13, 3.14 and 3.15, and the free-threaded 3.14t and 3.15t | Linux x86_64 (a CPU with AVX2: Intel from 2013, AMD from 2017) and arm64 (glibc 2.28 or later); macOS arm64 (11.0 or later) |

On other platforms, pip builds the package from source, which needs a Rust toolchain.

## Quick start: Rust

A dataset implements two methods: `get`, which returns one item, and `len`. The loader does the
rest.

```rust
use dataloader_rs::{DataLoader, Dataset, RandomSampler, error::Result};

/// One item per file: its bytes. `get` is called from the loader's worker threads.
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
        .sampler(RandomSampler::new(42)) // a new order every epoch, the same on every run
        .num_workers(4)                  // load 4 batches in parallel
        .prefetch_depth(8)               // and keep up to 8 more ready
        .drop_last(true)                 // skip a last batch smaller than 32
        .build();

    for epoch in 0..3 {
        for batch in &mut loader {
            let batch: Vec<Vec<u8>> = batch?; // by default, a batch is a Vec of its items
            println!("epoch {epoch}: {} files", batch.len());
        }
    }
    Ok(())
}
```

`cargo run --example basic` runs a complete version, and [examples/](examples/README.md) has
more.

## Quick start: Python

A dataset subclasses `PyDataset` and implements `__len__` and `__getitem__`, as in PyTorch.

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
for epoch in range(3):
    for batch in loader:  # a list of 32 `bytes` objects, or what collate_fn returns
        ...
```

The [Python guide](python/README.md) covers every argument, `collate_fn`, `__getitems__`,
transforms, async datasets, the built-in datasets, errors, and how threads and the GIL interact.

## Concepts

| Piece | What it does | Provided |
|---|---|---|
| Dataset | Returns the item at an index, and the number of items. A `Dataset` returns it directly; an `AsyncDataset` returns a future. A dataset only fetches: it does not decode or augment. | Your own type; `HttpDataset`, `FileDataset` and `ObjectStoreDataset` (async) |
| Sampler | Chooses the order of the indices in each epoch. | `SequentialSampler` (the default), `RandomSampler`, `DistributedSampler` |
| Transform | Turns each item into what the collator needs (decoding, augmentation). It receives the item and its index, and runs on the loader's threads. | `Identity` (the default), any closure `Fn(usize, Item) -> Result<Output>`, the image operations |
| Collator | Turns the items of a batch into one batch value. | `VecCollator` (the default: a `Vec`), `DefaultCollator` (like PyTorch's `default_collate`), `ImageCollator`, `TorchPinnedCollator` |
| Loader | Holds the pieces and the settings. Each `iter()` runs one epoch and yields the batches in order, each as a `Result`. | `DataLoader`, built with `DataLoader::builder` or `DataLoader::builder_async` |

In one epoch, the sampler produces the indices, the loader cuts them into batches of
`batch_size`, and each batch is fetched, transformed item by item, and collated. The iterator
returns the batches in the sampler's order, whatever order they finish in. The transform
receives each item's dataset index, so it can look up per-item information, such as a label or
a position, that the dataset does not return.

## How it works

Every loader has the same front: the sampler gives the epoch's indices, `iter()` cuts them into
batches, and your loop reads the batches in order with `next()`. What runs where depends on the
kind of dataset and on two settings: `num_workers`, the number of batches loaded in parallel,
and `item_concurrency`, the number of items of one batch fetched in parallel.

### A sync dataset, without workers

With `num_workers = 0` (the default), there is no background work: each call to `next()`
fetches the next batch, transforms its items and collates them on your own thread. For cheap
items this is the fastest setting, because nothing is handed between threads. If
`item_concurrency` is above 1, the items of that batch are fetched on a small thread pool, and
`next()` waits for them.

### A sync dataset, with workers

With `num_workers > 0`, the loader builds a thread pool once, when the loader is built, and
reuses it for every epoch. Here with `num_workers = 2` and `item_concurrency = 3`:

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

Up to `num_workers` **batch tasks** run on the pool. Each one claims the next batch that nobody
has claimed yet, loads it, and claims another. Inside a batch, `item_concurrency` **lanes**
share the work: each lane takes the batch's next item, calls `get`, applies the transform on the
same thread, and takes another item until none is left. Once all its items are ready, the batch
is collated and handed over. With `item_concurrency = 1`, there is a single lane, which fetches
the whole batch with one `get_batch` call.

Batches finish in any order. A finished batch waits in a **reorder ring** until all the batches
before it have been returned, so your loop always receives them in the sampler's order.

Memory stays bounded because a task may only claim a batch that is less than
`num_workers + prefetch_depth` batches ahead of the one your loop is waiting for. If one batch is
slow, the others do not run ahead without limit: the tasks stop claiming, and exit instead of
blocking a thread, until your loop catches up.

### An async dataset

An `AsyncDataset` (built with `DataLoader::builder_async`) returns a future for each item.
Fetching runs on a tokio runtime, and all CPU work runs on the loader's thread pool, so neither
slows the other down:

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

A **driver task** starts one **batch task** per batch, in order, and keeps `num_workers` of
them running (at least one). Each batch task keeps up to `item_concurrency` item futures in
flight, so `num_workers × item_concurrency` requests run at once. A request in flight costs a
future, not a thread: a few runtime threads can drive thousands of them.

When an item arrives, its transform is sent to the thread pool and runs while the batch's other
requests are still in flight; the result comes back to the batch task through a one-shot
channel. When all its items are in, the batch is collated on the pool too. The driver then
passes the batches, in order, to a channel that your loop reads. At most `prefetch_depth`
finished batches wait in that channel, so a slow consumer does not let finished batches pile up.

### From Python

The Python `PyDataloader` runs the same two paths. What changes is when the GIL is held.

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

- A **`PyDataset`** is loaded like a sync Rust dataset. The worker threads hold the GIL while
  they call `__getitem__`, `transform` and `collate_fn`, and release it in between.
- The **built-in datasets** (`HttpDataset`, `FileDataset`, `ObjectStoreDataset`) fetch in Rust,
  without the GIL. The GIL is taken only to turn an item into `bytes` and to call your
  `transform`. With a `dataloader_rs.transforms` transform and `collate_fn=ImageCollator()`, the
  GIL is not taken at all until the batch is handed to your loop: the items stay Rust values
  from fetch to batch.
- A **`PyAsyncDataset`** has `async def __getitem__`. The loader runs the coroutines on an
  asyncio event loop of its own, on a background thread, and waits for them from the async path
  described above.

Your loop waits for the next batch **without** holding the GIL, so the workers can run while it
waits. On a free-threaded build of CPython (3.14t, 3.15t), the workers also run Python code in
parallel; with the GIL, only the work that releases it (Rust code, I/O, NumPy) runs in
parallel.

### Threads at a glance

| Threads | Created | How many | What they run |
|---|---|---|---|
| The loader's thread pool, `dataloader-N` | When the loader is built, one pool per loader, reused by every epoch | Sync dataset: `max(num_workers, 1) × item_concurrency`, or none when both settings are at their defaults. Async dataset: the same, but at most the number of CPUs, with a transform; `max(num_workers, 1)` without one. | Batch tasks, `get`, transforms, collation |
| The loader's tokio runtime, `dataloader-io` | When an async loader is built, unless you pass your own runtime with `runtime(handle)` | `runtime_threads`, by default the number of CPUs available | The futures that fetch items, the driver and the batch tasks |
| Rayon's global pool, `dataloader-map-N` | Python only, with the first loader over a `PyAsyncDataset` | The number of CPUs, shared by the whole process | Creating the coroutines and submitting them to the event loop |
| `dataloader-asyncio` | Python only, one per loader over a `PyAsyncDataset` | 1 | The asyncio event loop; it awaits `aclose()` when the loader is dropped |

[ARCHITECTURE.md](ARCHITECTURE.md) explains these mechanisms in more detail, along with errors,
panics, early exits and `fork()`.

## Configuration

Every setting is a method of the builder; the Python argument of the same name does the same.

| Builder method | Default | What it does |
|---|---|---|
| `batch_size(n)` | `1` | Sets the number of items per batch. |
| `drop_last(bool)` | `false` | Drops the last batch of an epoch when it has fewer than `batch_size` items. |
| `sampler(s)` | `SequentialSampler` | Sets the order of the indices in each epoch. |
| `transform(t)` | `Identity` | Sets the per-item work, called as `t.apply(index, item)` before collation. |
| `collator(c)` | `VecCollator` | Sets how the items of a batch become one batch value. |
| `num_workers(n)` | `0` | Sets how many batches are loaded in parallel. With `0`, each batch is loaded inside `next()`, on your thread. |
| `item_concurrency(k)` | `1` | Sets how many items of one batch are fetched in parallel. With `1`, a batch is fetched with one `get_batch` call. (`items_in_parallel` is its former, deprecated name.) |
| `prefetch_depth(n)` | `2` | Sets how far loading may run ahead of your loop: at most `num_workers + n` batches are being loaded or waiting. |
| `runtime_threads(n)` | the number of CPUs | Async datasets only. Sets the number of threads of the loader's tokio runtime, from 1 to 1024. |
| `runtime(handle)` | none | Async datasets only. Runs the fetches on your own multi-threaded tokio runtime instead of the loader's. It cannot be combined with `runtime_threads`. |
| `pin_memory(bool)` | none | With the `torch-rs` feature, pins `tch::Tensor` batches in page-locked memory for faster copies to a GPU. |

An invalid value (a `batch_size`, `item_concurrency` or `prefetch_depth` of 0, too many runtime
threads, or both runtime settings) makes the builder panic.

Some rules of thumb, all measured in [BENCH.md](BENCH.md#4-results-rust):

- **Cheap items** (in memory, a few microseconds each): keep `num_workers = 0`. Handing a batch
  between threads costs about half a microsecond in Rust, and a few microseconds through Python,
  which is more than the work itself.
- **CPU-bound items** (decoding, augmentation): set `num_workers` up to the number of cores. For
  a few large batches, `item_concurrency` spreads the items of one batch over several cores.
- **Waiting items** (network, remote storage): `num_workers × item_concurrency` is the number of
  requests in flight. Size it from the latency and the throughput you need (64 requests of 10 ms
  each give at most 6,400 items per second); it can be far larger than the number of cores. An
  async dataset handles large numbers best, because a request in flight costs a future rather
  than a thread.
- **`prefetch_depth`**: the default of 2 is enough while loading keeps up with your loop. A
  larger value only helps when batch times vary a lot, and it costs memory.

## Built-in datasets

With the `async`, `http` and `object-store` features, three ready-made async datasets return
each item as bytes (`bytes::Bytes` in Rust, `bytes` in Python), capped at 64 MiB by default.

| Dataset | Item `i` | Retried by default |
|---|---|---|
| `HttpDataset` | The body of a GET request. All requests share one client, which keeps connections open; headers are sent with every request. | 3 times, after connection errors, time-outs and the statuses 408, 429, 500, 502, 503 and 504. A `Retry-After` header is honoured. |
| `FileDataset` | The content of a local file. | Not at all; you can set a time limit per attempt and retries, for network file systems. |
| `ObjectStoreDataset` | An object (or a byte range of one) in S3, Google Cloud Storage, Azure or an HTTP store. | 3 times, by the `object_store` crate itself, after 5xx statuses, 429, 408, connection errors and time-outs. |

Each dataset takes the locations of its items in one of three ways: a list, a function of the
index, or a **template** with columns of values, such as
`https://tiles.example/tile?x={x}&y={y}` with one column of `x` values and one of `y` values. A
template is parsed once, and each location is made only when its item is fetched, so a dataset
of millions of items does not keep millions of strings. In a URL, string values are
percent-encoded; in a path or an object key, a value cannot contain `/` or be `..`, so it cannot
point outside the template's directory.

The Rust API is in [src/README.md](src/README.md#the-asyncdataset-contract-feature-async) and on
docs.rs; the Python API is in the [Python guide](python/README.md#async-datasets).

## Image operations

With the `image` feature, image decoding and the usual torchvision transforms are available as
transforms of one item, run in parallel on the loader's threads. In Rust, a chain of them is
typed: an operation that needs float pixels (`Normalize`) placed before the conversion to floats
(`ToFloat`) does not compile. `ImageCollator` writes the images of a batch into one buffer.

| Rust (`dataloader_rs::image`) and Python (`dataloader_rs.transforms`) | Equivalent in Pillow, torchvision or NumPy | Same values |
|---|---|---|
| `Decode` (PNG, JPEG, WebP, TIFF, BMP; with limits against decompression bombs) | `Image.open` | Exact; JPEG within 3 levels |
| `ToRgb`, `ToRgba`, `ToGray` | `convert("RGB")`, `convert("RGBA")`, `convert("L")` | Exact |
| `Crop`, `CenterCrop` | `F.crop`, `CenterCrop` | Exact |
| `Pad` (constant, edge, reflect, symmetric) | `Pad`, `numpy.pad` | Exact |
| `PadTo` (pads up to a fixed size, never crops) | `F.pad` on the right and bottom edges | Exact |
| `Resize` (nearest, bilinear, bicubic, Lanczos3; antialiased) | `Resize` | Nearest exact; bilinear within 1 level |
| `HFlip`, `VFlip`, `Rotate90` | `F.hflip`, `F.vflip`, `numpy.rot90` | Exact |
| `ToFloat` | The values of `ToTensor` | Exact |
| `Normalize` | `Normalize` | Exact |
| `ToChw`, `ToHwc` | The axis order of `ToTensor` | Exact |

```rust
use dataloader_rs::{DataLoader, Transform, collator::ImageCollator, dataset::FileDataset};
use dataloader_rs::image::{Decode, Normalize, PadTo, ToChw, ToFloat, ToRgb};

let mut loader = DataLoader::builder_async(FileDataset::new(paths))
    .batch_size(64)
    .num_workers(4)
    .item_concurrency(4)
    .transform(
        Decode::new()                      // bytes to an 8-bit image, height × width × channels
            .then(ToRgb)                   // 3 channels
            .then(PadTo::new(224, 224, 0.0))
            .then(ToFloat::default())      // floats in [0, 1]
            .then(Normalize::imagenet())
            .then(ToChw),                  // channels first
    )
    .collator(ImageCollator::default().reuse_buffers(8)) // one reused buffer per batch
    .build();
```

In Python, the same operations are in `dataloader_rs.transforms`, combined with
`transforms.Compose`. A batch from `ImageCollator` is viewed by `np.asarray(batch)` or
`torch.from_dlpack(batch)` without a copy, and `Image.from_array` brings a NumPy array or a
torch tensor back into Rust, so Rust and torch operations can be mixed in one transform. The
details are in the [Python guide](python/README.md#image-transforms-in-rust), in
[src/image](src/image/README.md) (limits and the parity tests) and in
[src/collator](src/collator/README.md#imagecollator-feature-image) (buffer reuse).

## Performance

Measured on an Intel Core i9-13900H pinned to 4 cores (the size of the CI runners), on CPython
3.13 (with the GIL), 3.14t and 3.15t, against `torch.utils.data.DataLoader` with the best
configuration of each loader. The method, every table and the caveats are in
[BENCH.md](BENCH.md); results for every commit are plotted on the dashboards for
[Rust](https://azhicham.github.io/dataloader-rs/benchmarks/rust/) and
[Python 3.14t](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| Python workload | Result against PyTorch |
|---|---|
| Loader overhead (items already in memory) | **4.9 to 6.6 times faster** |
| Network-bound items (10 ms each), with `item_concurrency` | **3.2 to 3.9 times faster**; the first batch in 21 ms instead of 320 ms or more |
| 2,048 HTTP tiles of 64 KiB (10 ms each): `HttpDataset` against PyTorch with 8 or 16 worker processes | **4.3 to 7.8 times faster** (0.38 s instead of 1.64 to 2.99 s), on 3.13 and 3.14t |
| 2,048 PNG tiles decoded, padded and normalised: the Rust image transforms against Pillow and torchvision in 4 worker processes | **2.4 times faster**, with 2.4 times less CPU time, and the same values |
| The first batch of an epoch | **a few milliseconds**, instead of 11 to 334 ms for PyTorch's fresh worker processes |
| Array items (read and normalise a file, augment an image), 4 workers | **2.5 to 3.5 times faster** on 3.14t and 3.15t, 1.0 to 1.5 times on 3.13, with the allocator settings of [BENCH.md, section 8](BENCH.md#8-caveats-and-known-bottlenecks) |
| Pure-Python CPU work, with per-thread tables, on 3.14t and 3.15t | **about 1.2 times faster** |
| Pure-Python CPU work on shared objects, or with the GIL | PyTorch is 2.4 to 2.9 times faster (its processes share nothing) |

## Examples

[examples/README.md](examples/README.md) lists the examples, in Rust and in Python, with the
command that runs each one: the basics, collators, hand-written async datasets, `HttpDataset`
with templates, headers and retries, files and object stores, image operations, Rust and torch
operations mixed, and tiles of a large image fetched by coordinates. They all run offline.

## Documentation

| Document | Content |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the loader works inside: the paths of an epoch, threads, ordering, bounded memory, failures, the Python binding, and the design choices. |
| [BENCH.md](BENCH.md) | What is measured and how, the results, the comparison with PyTorch, caveats, and how to run the benchmarks again. |
| [python/README.md](python/README.md) | The guide to the Python package (also its page on PyPI). |
| [src/README.md](src/README.md) | A map of the crate's code, and the `Dataset` and `AsyncDataset` contracts. |
| [src/loader](src/loader/README.md), [src/sampler](src/sampler/README.md), [src/collator](src/collator/README.md), [src/image](src/image/README.md), [src/python](src/python/README.md) | One guide per module (also on docs.rs). |
| [examples/](examples/README.md), [tests/](tests/README.md), [benches/](benches/README.md), [bench/](bench/README.md) | What each example, test file, Rust benchmark and Python benchmark covers. |
| [CHANGELOG.md](CHANGELOG.md) | The release notes, generated by semantic-release. |

The Python package's code is in [`dataloader_rs/`](dataloader_rs/__init__.pyi): it re-exports
the Rust module and holds the type stubs.

## Building and testing

You need Rust 1.95 or later, and [uv](https://docs.astral.sh/uv/) for Python. The development
environment uses the free-threaded CPython 3.14t (set in `.python-version`); to use a build with
the GIL, add `--python 3.13` to `uv run`.

```bash
# Rust
cargo build
cargo test --features ndarray,python          # add torch-rs to test tch (it downloads libtorch)
cargo test --all-features                     # everything, as CI does

# Python
uv sync --group dev                            # maturin, pytest, NumPy, torch, aiohttp
uv run maturin develop                         # build the extension into the virtual environment
uv run pytest                                  # tests/ (the benchmarks are deselected)

# Lints, as CI runs them
CI=1 prek run --all-files                      # ruff, ty, cargo fmt, clippy with -D warnings
```

Running the benchmarks is described in [BENCH.md, section 9](BENCH.md#9-running-it-again).

Commits follow [Conventional Commits](https://www.conventionalcommits.org/). Releases, the
changelog, the crate and the wheels are published by semantic-release from the `main` branch.

## Minimum supported Rust version

Rust **1.95**, as set by `rust-version` in `Cargo.toml`. The `Rust tests (MSRV)` CI job checks
it.

## License

MIT; see [LICENSE](LICENSE).
