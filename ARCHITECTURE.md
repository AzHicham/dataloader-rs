# Architecture

How `dataloader-rs` turns a dataset into an ordered stream of batches, where the threads are,
what bounds memory, and what happens on errors, panics and early exits — in Rust and through
the Python binding.

Contents:

1. [Goals](#1-goals)
2. [The pieces](#2-the-pieces)
3. [Life of an epoch](#3-life-of-an-epoch)
4. [Failures and early exits](#4-failures-and-early-exits)
5. [Memory safety](#5-memory-safety)
6. [The Python binding](#6-the-python-binding)
7. [Design choices](#7-design-choices)

## 1. Goals

- **PyTorch's model**: dataset, sampler, collator and loader as separate, replaceable pieces,
  with the same knobs (`batch_size`, `num_workers`, `drop_last`, `shuffle`, `collate_fn`).
- **Threads, not processes**: workers share the dataset in memory; a sample is handed to the
  consumer by moving a value, never by serialising it.
- **Typed and monomorphized**: `DataLoader<D, S, C>` is generic over dataset, sampler and
  collator, so the hot path has no dynamic dispatch and no per-item boxing.
- **Predictable**: batches come out in epoch order whatever the scheduling; memory held by
  prefetching is bounded by settings; errors and panics reach the caller.

## 2. The pieces

```text
            ┌──────────────┐  indices  ┌──────────────┐  batches of indices
 epoch ───▶ │   Sampler    │ ────────▶ │ BatchSampler │ ─────────────────────┐
            └──────────────┘           └──────────────┘                      │
                                                                             ▼
 consumer ◀── DataLoaderIter ◀── collate ◀── transform (per item) ◀── fetch (Dataset) ◀── batch i
```

| Piece | Role | Code |
|---|---|---|
| `Dataset` | `get(index)`, `len()`; `get_batch(indices)` defaults to `get` per index | `src/dataset/sync.rs` |
| `AsyncDataset` | the same with `get` returning a future (feature `async`); `HttpDataset`, `FileDataset`, `ObjectStoreDataset` | `src/dataset/` |
| `Transform<In>` | per-item CPU work `apply(index, item)`, a loader setting (`builder.transform(t)`, default `Identity`); closures, `then` | `src/transform.rs` |
| `Sampler` | the index sequence of one epoch (`indices(dataset_len)`), and its length | `src/sampler/` |
| `BatchSampler` | cuts the indices into `batch_size` chunks, honours `drop_last` | `src/sampler/batch_sampler.rs` |
| `Collator<Item>` | merges a batch's `Vec<Item>` into one `Batch` value | `src/collator/` |
| image transforms, `ImageCollator` | `Decode`, `PadTo`, `Normalize`, … as `Transform`s on `Image<T, L>`; images of one shape into one `ImageBatch` buffer (feature `image`) | `src/image/`, `src/collator/image/` |
| `DataLoader` | owns the pieces and the settings; `iter()` starts an epoch | `src/loader/dataloader.rs` |
| `SyncSource`, `AsyncSource` | the dataset as the loader holds it (`builder` / `builder_async`), behind the sealed `LoaderSource`; each starts its own kind of epoch | `src/loader/source.rs` |
| `DataLoaderIter` | yields `Result<Batch>` in epoch order | `src/loader/iter.rs` |

A dataset is a **source**: sync (`Dataset`) or async (`AsyncDataset`), it only fetches. The
**transform** is a loader parameter, always sync, always run on the loader's threads, with the
item's index; it changes the builder's item type as `collator(c)` changes its batch type, and
the collator receives transformed items. `get_batch` must return one item per index: the loader
fails the batch otherwise, so items and indices never shift.

`DataLoader::iter(&mut self)` advances the sampler (a `RandomSampler` draws a new permutation
each epoch) but the iterator it returns does not borrow the loader: it holds `Arc` clones of the
dataset, the transform and the collator. Every path yields batches in epoch order, a failed one
included; the transform sees each item's index, so per-item metadata (coordinates, labels) is
looked up there rather than carried through the dataset.

Image work follows the same split (feature `image`): decoding, color, crop, pad, resize, float
conversion and normalisation are each a `Transform` of one item, chained with `then` and typed
(`Image<u8>` → `Image<f32>`, HWC → CHW), so they run in parallel on the loader's pool like any
transform; only the copy into the batch buffer is the collator's. A batch's images are written
once into one contiguous buffer (`ImageCollator`); with `reuse_buffers(n)` the buffers of
dropped batches are kept for the next ones. Taking a buffer never waits: an empty pool
allocates, so a consumer that holds many batches costs memory, not a deadlock (a pool that made
collators wait for a free buffer would block pool threads that the batches being held would
need).

## 3. Life of an epoch

### 3.1 The pool

Two settings decide the parallelism, and nothing else does:

| Setting | Meaning | Default |
|---|---|---|
| `num_workers(n)` | batches processed in parallel; `0` processes batches in `next()` | `0` |
| `item_concurrency(k)` | items of one batch fetched in parallel | `1` |
| `prefetch_depth(d)` | how far ahead of the consumer batches may run: at most `n + d` in flight or buffered | `2` |

When either parallel setting is above its default, `build()` creates **one rayon pool per
loader** with `max(n, 1) × k` threads, so at most that many fetches run at once. The pool lives
as long as the loader and is reused by every epoch; its threads are named `dataloader-N` and have
a main thread's 8 MiB stack. Sizing it for the machine is the caller's choice.

### 3.2 Choosing a path

`iter()` asks the batch sampler for every batch of the epoch up front (a `Vec<Vec<usize>>`), then
picks one of two paths:

```text
num_workers == 0                         num_workers > 0
┌───────────────────────────┐            ┌──────────────────────────────────────────┐
│ Direct path               │            │ Parallel path                            │
│ next() fetches + collates │            │ batch tasks on the loader's pool claim   │
│ batch i on the caller's   │            │ batches in epoch order, at most          │
│ thread                    │            │ num_workers + prefetch_depth ahead       │
└───────────────────────────┘            └──────────────────────────────────────────┘
```

### 3.3 Direct path (`num_workers = 0`)

`next()` takes the next chunk of indices, fetches the samples (`Dataset::get_batch`), transforms
and collates them on the calling thread. No thread hand-off: for cheap samples it is the fastest path
([BENCH.md](BENCH.md)).

### 3.4 Parallel path (`num_workers > 0`)

The epoch is a shared state (`Epoch`): the batches, a claim counter `next_claim`, the consumer's
position `next_out`, and a result channel. Up to `num_workers` **batch tasks** run on the pool;
each one loops:

1. claim the next batch `i` — but only while `i < next_out + num_workers + prefetch_depth`
   (the **window**);
2. fetch, transform and collate it, and send `(i, result)` to the iterator;
3. when nothing is claimable, **exit** instead of blocking a pool thread.

The iterator returns batch `next_out` as soon as it has arrived, publishes its new position, and
starts tasks again if some had exited. A task that sees no claimable batch re-checks after
leaving, so a batch is never left unclaimed between a task's exit and the consumer's advance.

```text
                 window: next_out .. next_out + num_workers + prefetch_depth
 batches  [ done | returned ... | next_out | in flight / buffered ... | not yet claimable ]
                                     ▲
 pool:  task ── claim ─▶ fetch ─▶ collate ─▶ send ──▶ reorder buffer ──▶ next()
```

**Backpressure.** Nothing beyond the window is claimed, so at most `num_workers + prefetch_depth`
batches are in flight or waiting, however slow one batch is.

**Ordering.** Tasks finish out of order; early results wait for their turn in a ring of `window`
slots, batch `i` in slot `i % window`. Every batch in flight or waiting lies in
`next_out..next_out + window`, so no two share a slot and the window bounds the buffer.

**Cost.** Claiming, publishing the position and restarting tasks cost a few atomic operations
and an occasional task spawn per batch. That is negligible once a batch does real work, but
dominates for nearly free batches (`batch_size` ≤ 8 of in-memory items, [BENCH.md](BENCH.md)).

### 3.5 Item-level parallelism (`item_concurrency`)

With `item_concurrency = 1`, a batch is one `Dataset::get_batch` call on the thread that
processes it. With `k > 1`, the batch is fetched by `k` **lanes** on the pool: each lane takes the
batch's next item, calls `Dataset::get` and applies the transform, until none is left, so the
items of a batch are transformed in parallel. A batch therefore never has more
than `k` items in flight, and a fast lane takes more items, so uneven items still balance. After
an error, lanes stop taking items and the first error by position is the batch's result. On the
direct path (`num_workers = 0`) the batch's lanes run on the pool too.

### 3.6 End of the epoch

When the last batch has been returned, the iterator yields `None`. The pool's threads stay
alive, idle, for the next epoch.

### 3.7 Async datasets (`builder_async`)

`DataLoader::builder_async(dataset)` builds the same `DataLoader` and `DataLoaderIter` types
around an `AsyncSource<A>`: the `AsyncDataset` and the tokio runtime that drives it — the
loader's own (built by `build()`, shut down
with `shutdown_background`, so dropping a loader inside async code does not panic) or the
caller's (`runtime(handle)`). The own runtime has `runtime_threads(n)` worker threads, by default
`std::thread::available_parallelism()` (affinity and cgroup quota included; 1 if unknown), at
most `MAX_RUNTIME_THREADS` = 1024; `0`, more than the maximum, or `runtime_threads` together with
`runtime` panic in the builder. `builder(dataset)` holds a `Dataset` in a `SyncSource<D>`; the
two sources implement the sealed `LoaderSource` trait, through which `build()` sizes the pool
and `iter()` starts the source's own epoch, so each path is chosen by type at compile time and
the sync path above is unchanged.
The pool transforms and collates: with a transform, `max(num_workers, 1) × item_concurrency`
threads, but no more than the CPUs available (at least `max(num_workers, 1)`): more would only
add wake-ups to CPU work; without one (`Identity`), `max(num_workers, 1)`: it only collates.

```text
driver task:  stream::iter(batches).map(spawn batch task).buffered(num_workers)
batch task:   stream::iter(indices).map(get, then transform on the pool)
                .buffered(item_concurrency).try_collect()  ──▶  collate on the pool
              (item_concurrency = 1: get_batch, then transform + collate in one pool job)
driver:       for each batch, in order: wait for room (semaphore of prefetch_depth permits),
              send (batch, permit) on a channel  ──▶  DataLoaderIter::next() receives
```

- **Bounds**: at most `num_workers` batch tasks exist, each with at most `item_concurrency`
  futures, so at most `num_workers × item_concurrency` fetches run at once; at most
  `prefetch_depth` finished batches wait in the channel, one more waits for room.
- **Order**: `buffered` yields batches in epoch order; no reorder buffer is needed.
- **Each batch is a task**, not only a future of the driver's stream: its fetches keep being
  polled while the driver waits for room, so a slow consumer cannot make requests time out.
- **The consumer** blocks on the channel (a crossbeam receiver, so the Python binding waits with
  a deadline, as on the sync path).
- **Transforms** run on the loader's pool as each item arrives, while the batch's other fetches
  are in flight; the result comes back to the future through a oneshot channel, so CPU work never
  runs on the runtime's threads. `Identity` (`Transform::is_identity`) skips the hop.

Sizing: `num_workers × item_concurrency` is the number of requests in flight; `runtime_threads`
is the CPU for network receive, TLS and body copies (one thread drives many requests); the
loader's pool is the CPU for transforms and collation.

The built-in datasets (`src/dataset/io/`) share two pieces. Where item `i` lives
(`Locations<L>`): a list parsed when the dataset is built, a `Template` parsed once whose
location is made and parsed when the item is fetched, or a function of the index. A template
writes string values per target (`Escape`): percent-encoded in URLs, as they are in paths and
object keys, where values holding `/` (or `\`, NUL in paths) or equal to `.`/`..` are refused
at build, and an item whose values make a `.`/`..` component with the template's text fails.
How often to try again (`Retry`): `Retry::run` loops with exponential backoff and jitter, and
each dataset classifies its failures (`Transient`): HTTP statuses and transport errors, file
I/O kinds and its own per-attempt timeout. `ObjectStoreDataset` turns `Retry` into
`object_store`'s `RetryConfig` for the store it opens, which retries by itself; no second loop
runs on top.

## 4. Failures and early exits

### 4.1 Errors

`Dataset::get`, `get_batch`, `Transform::apply` and `Collator::collate` return `Result`. An error belongs to its
batch: the iterator yields `Some(Err(e))` for that batch and continues with the next one, on both
paths. The error type is `Box<dyn Error + Send + Sync>`, so any error converts with `?`.

### 4.2 Panics

On the direct path a panic in the dataset, transform or collator unwinds through `next()` as usual. On the
parallel path, batch tasks run each batch under `catch_unwind` (a panic escaping a rayon task
would abort the process), send the payload to the iterator and cancel the epoch; the iterator
re-raises it with `resume_unwind`, so the caller sees the same panic as with `num_workers = 0`. The payload is re-raised when it arrives, which can be before
earlier batches have been returned. On the async path, tokio catches a batch task's panic (a
panicking collation or transform is caught on the pool and resumed in the task), the driver
forwards the payload and the iterator re-raises it in order; the epoch then ends.

### 4.3 Dropping an iterator early

Dropping an iterator mid-epoch (a `break`, an error, `drop`) marks the epoch cancelled: batch
tasks stop claiming. A task in the middle of a batch finishes it in the background and its
result is discarded; `drop` itself **does not wait**. On the async path, dropping the iterator
aborts the driver task, which drops (and so aborts) every batch task and their futures: requests
in flight are cancelled, not finished. The Python binding waits for those tasks before it drops
an iterator (section 6.5).

### 4.4 `fork()`

A process created with `fork()` (e.g. Python `multiprocessing` with the `fork` start method)
inherits the pool but none of its threads. The pool records the PID that built it; `iter()` in
another process builds a new pool. The inherited one is leaked, not dropped: dropping it would
signal threads that do not exist, through locks a parent thread may have held at fork time.
An async loader's runtime is not rebuilt: an epoch started in the child returns an error.

## 5. Memory safety

The crate is `#![forbid(unsafe_code)]`. Workers receive `Arc` clones of the dataset and the
collator, never references into the loader, so neither leaking an iterator (`mem::forget`) nor
dropping the loader while an iterator lives can leave a thread reading freed memory: a leaked
iterator keeps the dataset alive.

`Dataset` and `Collator` require `Send + Sync + 'static` and items `Send + 'static`, which is
what sharing them across threads needs.

## 6. The Python binding

Behind the `python` feature, `src/python/` builds the `dataloader_rs` extension module
(`#[pymodule(gil_used = false)]`, so it also loads on free-threaded CPython).

| Python | Rust |
|---|---|
| `PyDataset` (subclass it) | `Py<PyDatasetBase>`, which implements `Dataset` by calling `__getitem__` / `__getitems__` / `__len__` |
| `PyDataloader` | `DataLoader<SyncSource<PyDataset>, PySampler, PyCollator, PyTransform>` |
| `PyDataloaderIter` | the epoch: a direct path, or the core parallel iterator |
| `PyAsyncDataset` (subclass it) | `PyAsyncItems`, an `AsyncDataset` running the coroutines on an asyncio loop thread |
| `HttpDataset`, `FileDataset`, `ObjectStoreDataset` | the Rust datasets (items `Bytes`) |
| `transform=f` | `PyTransform`, the loader's `Transform`: `f(index, item)`; for a Rust dataset `bytes` first |
| `transform=dataloader_rs.transforms.Compose([...])` | `PyTransform::Native`: the Rust image transforms (`python/image/ops.rs`), no interpreter |
| `collate_fn=ImageCollator()` | `PyCollator::Images`: `ImageCollator` in Rust, batches as `ImageBatch` objects |

`PyDataloader(dataset)` picks the loader from the dataset's base class: `PyDataset` → the sync
loader; `PyAsyncDataset` → `DataLoader<AsyncSource<PyAsyncItems>, …>`; a Rust dataset →
`DataLoader<AsyncSource<RustSource>, …>`, whose `PyTransform` makes the `bytes`. At
construction it refuses `async def` methods on a `PyDataset`, plain ones on a `PyAsyncDataset`,
both bases, and a missing `__len__`; a plain `__getitem__` returning an awaitable fails on the
first item of a batch (the coroutines are closed).

### 6.1 Fetching and collating

- **`num_workers = 0`**: `__next__` already holds the interpreter, so it calls the dataset
  directly: `__getitems__(indices)` once per batch if the dataset defines it, else a bound
  `__getitem__` (looked up once per epoch) per index; then `transform` per item and
  `collate_fn`, or a `list`.
- **`num_workers > 0`**: each batch task attaches to the interpreter **once per batch**
  (`Dataset::get_batch`), not once per item. The pool's threads are created with the loader and
  reused by every epoch, and each keeps one Python thread state for its whole life
  (`python/pool.rs`): a task's attach only re-activates it, where creating and destroying a thread
  state per attach would cost about 100 µs on free-threaded builds. The consumer releases the
  interpreter (`py.detach`) only when it has to wait for the next batch, so workers can run, and
  waits in slices of 100 ms with signal handlers in between, so Ctrl-C interrupts the wait; a
  batch that already arrived is returned without releasing it.
- **`item_concurrency > 1`**: a batch's lanes call `__getitem__` and `transform` concurrently,
  each attaching to the interpreter for its own calls (`__getitems__` is not used); such epochs always use the core
  iterator, even with `num_workers = 0`, because the consumer must wait with the interpreter
  released.

On regular CPython the tasks' Python code is serialised by the GIL; work that releases it
(file I/O, NumPy, hashlib) runs in parallel. On free-threaded CPython 3.14t all of it does.

### 6.2 Async and Rust datasets

- **`PyAsyncDataset`**: the loader starts an asyncio event loop on a daemon thread and stops it
  when the loader goes, after awaiting the dataset's `aclose()` if it defines one (at most 5 s,
  the dropping thread waits for it; errors and timeouts are logged). For each item, a thread of rayon's global pool calls `__getitem__` (or
  `__getitems__`) and submits the coroutine with `asyncio.run_coroutine_threadsafe`; a done
  callback sends the result through a oneshot channel to the waiting future. The global pool's
  threads stay attached to the interpreter like the loader's: attaching a tokio thread per item
  would create a thread state each time, three times the cost of everything else on
  free-threaded builds. Dropping the future (an aborted epoch) cancels the coroutine. The
  stopped loop cancels the coroutines still pending and waits for them (at most 5 s) before it
  closes, so that each coroutine created is awaited.
- **Rust datasets**: fetching holds no interpreter; the conversion to `bytes` and `transform`
  run as the loader's transform on its pool, attaching only for the calls. With a
  `dataloader_rs.transforms` transform and `ImageCollator`, nothing attaches until `__next__`
  wraps the batch: the items stay Rust values (`PyItem::Native`) from fetch to collation.

### 6.2.1 Image transforms and zero-copy batches

`dataloader_rs.transforms` builds `Transform` objects holding a list of `Op`s, each wrapping the
Rust transform of the same name; the image flowing through them is a `Value` (bytes, or an
`Image` of one of four type × layout combinations), checked at each step with an error saying
what the step takes. Passed as `transform=`, the list runs as `PyTransform::Native`: on a
Python dataset's items it attaches only to copy the `bytes`; on a Rust dataset's `Bytes` it does
not attach. The items reach the collator as `PyItem::Native`; `ImageCollator` collates them
without the interpreter, any other `collate_fn` (or none) gets them as `Image` objects.

`Image` and `ImageBatch` keep their values in Rust's buffer and describe it with
`__array_interface__` (address, shape, type string, C-contiguous): `np.asarray(batch)` makes a
view whose `base` is the batch, so the buffer lives as long as any view. `__dlpack__` delegates
to that view's, so `torch.from_dlpack(batch)` shares the memory too. Batches are writable (a
tensor's memory must be), images read-only; no unsafe code is involved, the buffer's address is
an integer. A batch from a collator with `buffers=n` returns its buffer to the pool when the
batch object and every view of it are gone.
- The global pool is built with the first `PyAsyncDataset` loader, with the interpreter released: rayon
  waits for its threads to start, and a starting thread may wait for every attached thread to
  pause (free-threaded builds).

### 6.3 Samplers

`shuffle=True` uses the Rust `RandomSampler` (seeded with `seed`, or from OS entropy); `sampler=`
accepts any finite Python iterable of `int`. The Python sampler is read whole, under the
interpreter, at the start of each epoch in `iter(loader)`, with signal handlers run every 65,536
indices so that Ctrl-C stops one that never ends. `len(loader)` uses `len(sampler)` when a sampler
is given, so a sampler without `__len__` raises `TypeError`, as in PyTorch.

### 6.4 Errors

Python exceptions cross the Rust core as boxed `PyErr`s and are restored with their original
type and traceback (`into_py_err`); only errors that did not come from Python become
`RuntimeError`, except the built-in datasets' `FetchError`, which becomes `IndexError`,
`ValueError` (over `max_bytes`), `TimeoutError` or `OSError`. An exception is raised for its batch and iteration can continue, with or without
workers, as in PyTorch. A sampler or `__len__` that raises makes `iter(loader)` or `len(loader)`
raise.

### 6.5 Dropping an iterator

Dropping a `PyDataloaderIter` cancels its epoch (section 4.3), then waits, with the interpreter
released, for the batch tasks in flight to finish their batch and exit, at most 5 s: they hold
the dataset, transform and collator, so a `break` or a garbage-collected iterator cannot
deadlock, and once the drop returns nothing but the loader refers to them. Loaders and iterators
are not tied to a thread: any thread may use or drop them (one at a time; PyO3's borrow check
refuses concurrent use), and they report the Python objects they hold to the garbage collector
(`__traverse__`), so a reference cycle through a loader is freed by one `gc.collect()` once no
batch is running Python code on it. While the interpreter shuts down (after `atexit`), drops do
not wait: the pool's threads can no longer attach.

## 7. Design choices

| Choice | Why | Cost |
|---|---|---|
| Threads instead of worker processes | samples move by value, no pickling, no per-epoch process start; the dataset is shared, not copied | pure-Python code is serialised by the GIL on regular CPython (use 3.14t) |
| One pool per loader, batch tasks on it | no thread spawn per epoch; batch and item work share threads instead of oversubscribing the CPU | a fixed cost per batch that shows on nearly free batches; the pool must be rebuilt after `fork()` |
| Generic `DataLoader<Src, S, C, T>` | monomorphized fetch, transform and collate, no boxing | the item and batch types are part of the loader's type |
| Transform on the loader, not the dataset | a dataset is only a source; one place, one thread rule (the loader's), the item's index for free | `Transform::is_identity` lets the async path skip the pool for `Identity` |
| Image operations as per-item transforms, the collator only copies | decoding and padding run in parallel across a batch's items; the collator, serial per batch, does one copy | `ImageCollator` cannot fuse padding into its copy |
| Batch buffers reused through a pool that never waits | a fresh 38.5 MB buffer spends three quarters of its collation in page faults; a waiting pool can deadlock when the consumer holds batches | up to `n` idle buffers stay allocated |
| Batches computed at `iter()` | the epoch's order is fixed before any work starts; `len()` is exact | one `Vec<Vec<usize>>` per epoch |
| Claim window | at most `num_workers + prefetch_depth` batches in flight or buffered, even behind one slow batch | a slow batch stalls the window instead of letting later batches run far ahead |
| Errors per batch | one bad sample does not end a Rust epoch | callers must handle `Result` per batch |
| Panics forwarded | same behaviour with and without workers; no silent truncation | a panic is re-raised as soon as it arrives |
| Sync and async datasets as two sources of the same `DataLoader` | one loader type and iterator, options and semantics; each path monomorphized, chosen at compile time | the source is part of the loader's type (`SyncSource<D>` / `AsyncSource<A>`) |
| Futures for fetching, rayon for CPU work | thousands of requests in flight on a few runtime threads; transforms and collation never block the runtime | a hand-off through a oneshot channel per transform and per batch |
| Rust datasets return `Bytes`, decoded by the transform | fetching, retries and size caps live in one place; decoding is the user's choice | — |
| One `Retry` and one list / template / function of locations for every built-in dataset; what is transient per backend | the same options everywhere; each backend knows its failures; `object_store` keeps its own retry loop | object stores also stop retrying 3 minutes after the first attempt (`object_store`'s default) |
