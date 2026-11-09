# Architecture

This document explains how `dataloader-rs` turns a dataset into an ordered stream of batches:
which threads do the work, what keeps memory bounded, how batches stay in order, and what
happens on errors, panics and early exits, in Rust and through the Python binding. It assumes
you know the user-facing concepts of the [README](README.md#concepts).

The design follows four goals:

- **PyTorch's model.** The dataset, sampler, transform, collator and loader are separate,
  replaceable pieces, with the settings PyTorch users know (`batch_size`, `num_workers`,
  `drop_last`, `shuffle`, `collate_fn`).
- **Threads, not processes.** The workers share the dataset in memory, and an item reaches the
  consumer by moving a value, never by serialising it.
- **Typed code without dynamic dispatch.** The loader is generic over its dataset, transform and
  collator, so the compiler generates a specialised loop for each combination, with no boxing
  per item.
- **Predictable behaviour.** Batches come out in epoch order however the work is scheduled,
  the memory used by prefetching is bounded by the settings, and errors and panics reach the
  caller.

Contents:

1. [The pieces](#1-the-pieces)
2. [Life of an epoch](#2-life-of-an-epoch)
3. [Transforms and collation](#3-transforms-and-collation)
4. [Failures and early exits](#4-failures-and-early-exits)
5. [Memory safety](#5-memory-safety)
6. [The Python binding](#6-the-python-binding)
7. [Design choices](#7-design-choices)

## 1. The pieces

```text
            ┌──────────────┐  indices  ┌──────────────┐  batches of indices
 epoch ───▶ │   Sampler    │ ────────▶ │ BatchSampler │ ─────────────────────┐
            └──────────────┘           └──────────────┘                      │
                                                                             ▼
 consumer ◀── DataLoaderIter ◀── collate ◀── transform (per item) ◀── fetch (Dataset) ◀── batch i
```

The sampler orders the indices of an epoch, the batch sampler cuts them into batches, and each
batch is fetched from the dataset, transformed item by item, and collated into one value that
the iterator returns.

| Piece | Role | Code |
|---|---|---|
| `Dataset` | Returns item `index` with `get(index)`, and the number of items with `len()`. `get_batch(indices)` fetches a whole batch; by default it calls `get` for each index. | `src/dataset/sync.rs` |
| `AsyncDataset` | The same, but `get` returns a future (feature `async`). `HttpDataset`, `FileDataset` and `ObjectStoreDataset` implement it. | `src/dataset/asynchronous.rs`, `src/dataset/io/` |
| `Transform<In>` | The per-item work, `apply(index, item)`. It is a setting of the loader (`Identity` by default), not of the dataset. Closures are transforms, and `then` chains two. | `src/transform.rs` |
| `Sampler` | Produces the indices of one epoch, in order, with `indices(dataset_len)`. | `src/sampler/` |
| `BatchSampler` | Cuts the indices into chunks of `batch_size`, and drops a short last chunk if `drop_last` is set. | `src/sampler/batch_sampler.rs` |
| `Collator<Item>` | Turns the items of a batch, a `Vec<Item>`, into one batch value. | `src/collator/` |
| Image operations and `ImageCollator` | `Decode`, `PadTo`, `Normalize` and the others are transforms of `Image<T, L>`; `ImageCollator` copies the images of a batch into one buffer (feature `image`). | `src/image/`, `src/collator/image/` |
| `DataLoader` | Holds the pieces and the settings; `iter()` starts an epoch. | `src/loader/dataloader.rs` |
| `SyncSource`, `AsyncSource` | The dataset as the loader holds it: `DataLoader::builder` wraps a `Dataset` in a `SyncSource`, `DataLoader::builder_async` wraps an `AsyncDataset` and its tokio runtime in an `AsyncSource`. | `src/loader/source/` |
| `DataLoaderIter` | Yields one `Result<Batch>` per batch, in epoch order. | `src/loader/iter.rs` |

A dataset is only a **source**: it fetches items and nothing else. The **transform** belongs to
the loader. It is always synchronous, always runs on the loader's threads, and receives each
item's index, so it can look up per-item information (a label, a position) that the dataset
does not return. Setting a transform changes the builder's item type, just as setting a collator
changes its batch type, and the collator receives the transformed items.

The two sources implement one sealed trait, `LoaderSource`. Through it, `build()` sizes the
thread pool and `iter()` starts the source's own kind of epoch. The path is therefore chosen by
the type of the dataset, at compile time, and both paths share the same `DataLoader` and
`DataLoaderIter` types and the same settings.

## 2. Life of an epoch

### 2.1 Starting an epoch

`DataLoader::iter(&mut self)` asks the sampler for the epoch's indices (a `RandomSampler` draws a
new permutation each time) and cuts them into all the batches of the epoch at once, as a
`Vec<Vec<usize>>`. The order of the epoch is therefore fixed before any work starts, and the
iterator knows exactly how many batches it will return.

The iterator does not borrow the loader. It holds `Arc` clones of the dataset, the transform and
the collator, so it can outlive the loader, and the worker threads never hold a reference into
it.

### 2.2 The thread pool

Two settings decide the parallelism:

| Setting | Meaning | Default |
|---|---|---|
| `num_workers(n)` | The number of batches processed in parallel. With `0`, batches are processed inside `next()`. | `0` |
| `item_concurrency(k)` | The number of items of one batch fetched in parallel. | `1` |
| `prefetch_depth(d)` | How far ahead of the consumer the batches may run: at most `n + d` batches are being processed or waiting. | `2` |

For a sync dataset, when either `num_workers` or `item_concurrency` is above its default,
`build()` creates **one rayon thread pool per loader**, with `max(n, 1) × k` threads, so that
`max(n, 1) × k` fetches can run at once. The pool lives as long as the loader and serves every
epoch, so no thread is started per epoch. Its threads are named `dataloader-N` and have an
8 MiB stack, the usual size for a main thread, so deep recursion in a dataset behaves as it
would on the caller's thread. Sizing the pool for the machine is left to the caller. The pool of
an async loader is sized differently (section 2.6).

### 2.3 Sync dataset, direct path (`num_workers = 0`)

`next()` takes the next batch of indices, fetches the items with `Dataset::get_batch`, applies
the transform to each item, and collates them, all on the calling thread. Nothing is handed
between threads, which makes this the fastest path for cheap items.

When `item_concurrency` is above 1, the batch's items are fetched by lanes on the pool (section
2.5) and `next()` waits for them.

### 2.4 Sync dataset, parallel path (`num_workers > 0`)

The epoch is a shared state: the list of batches, a claim counter (the next batch nobody has
claimed), the consumer's position (the next batch the consumer will return), and a channel for
results. Up to `num_workers` **batch tasks** run on the pool, and each one loops:

1. It claims the next batch `i`, but only if `i` is inside the **window**: less than the
   consumer's position plus `num_workers + prefetch_depth`.
2. It fetches, transforms and collates the batch, and sends `(i, result)` to the iterator.
3. When no batch can be claimed, it **exits** instead of blocking a pool thread.

When the consumer's batch arrives, the iterator returns it, publishes its new position, and
starts tasks again if some had exited. A task that found nothing to claim checks again after it
has left, so a batch is never left unclaimed between the moment a task exits and the moment the
consumer moves on.

```text
                 window: next_out .. next_out + num_workers + prefetch_depth
 batches  [ done | returned ... | next_out | in flight / buffered ... | not yet claimable ]
                                     ▲
 pool:  task ── claim ─▶ fetch ─▶ collate ─▶ send ──▶ reorder buffer ──▶ next()
```

**Bounded memory.** No batch beyond the window is ever claimed, so at most
`num_workers + prefetch_depth` batches are being processed or waiting, however slow one batch
is. A slow batch holds the window back instead of letting later batches pile up.

**Ordering.** Tasks finish in any order. A result that arrives early waits in a **reorder ring**
of `num_workers + prefetch_depth` slots, batch `i` in slot `i % window`. Every batch that is
being processed or waiting lies inside the window, so no two batches ever need the same slot,
and the ring never grows.

**Cost.** Claiming a batch, publishing the consumer's position and restarting tasks cost a few
atomic operations and, now and then, a task spawn: about half a microsecond per batch. That is
negligible once a batch does real work, but it dominates for nearly free batches (8 in-memory
items or fewer per batch, see [benches/README.md](benches/README.md#reference-results)), where the direct path is faster.

### 2.5 Item lanes (`item_concurrency`)

With `item_concurrency = 1`, a batch is one `Dataset::get_batch` call on the thread that
processes it, followed by the transform of each item.

With `item_concurrency = k > 1`, the batch is fetched by `k` **lanes** running on the pool. Each
lane takes the batch's next item, calls `Dataset::get`, applies the transform on the same
thread, and takes another item, until none is left. A batch therefore never has more than `k`
items in flight, the items are transformed in parallel, and a fast lane simply takes more items,
so uneven items still balance across the lanes. After an error, the lanes stop taking items, and
the batch's result is the first error by position in the batch. Lanes are used on both sync
paths.

### 2.6 Async dataset (`DataLoader::builder_async`)

An `AsyncSource` holds the `AsyncDataset` and the tokio runtime that drives its futures. The
runtime is either the loader's own, built by `build()`, or the caller's, passed with
`runtime(handle)`. The loader's own runtime has `runtime_threads(n)` threads; by default, as many
as `std::thread::available_parallelism()` reports (which accounts for CPU affinity and cgroup
quotas, and is 1 if unknown), and at most 1,024. It is shut down with `shutdown_background`, so
dropping a loader inside async code does not panic.

The loader's rayon pool only transforms and collates here, since fetching is done by futures.
With a transform, it has `max(num_workers, 1) × item_concurrency` threads, but no more than the
CPUs available, because more threads would only add wake-ups to CPU-bound work. Without a
transform (`Identity`), it has `max(num_workers, 1)` threads, one per batch being collated.

```text
driver task:  stream::iter(batches).map(spawn batch task).buffered(num_workers)
batch task:   stream::iter(indices).map(get, then transform on the pool)
                .buffered(item_concurrency).try_collect()  ──▶  collate on the pool
              (item_concurrency = 1: get_batch, then transform + collate in one pool job)
driver:       for each batch, in order: wait for room (semaphore of prefetch_depth permits),
              send (batch, permit) on a channel  ──▶  DataLoaderIter::next() receives
```

- **The driver task** spawns one batch task per batch, in epoch order, and keeps at most
  `num_workers` of them running (at least one). Its stream yields the batches in order, so no
  reorder ring is needed.
- **Each batch task** keeps up to `item_concurrency` item futures in flight. At most
  `num_workers × item_concurrency` fetches therefore run at once. Each is a future, not a
  thread, so a few runtime threads can drive thousands of requests.
- **Each batch is a separate task**, not only a future inside the driver's stream. Its fetches
  keep making progress while the driver waits for room, so a slow consumer cannot make requests
  time out.
- **Transforms** run on the loader's pool as each item arrives, while the batch's other fetches
  are still in flight. The result returns to the future through a one-shot channel, so CPU work
  never runs on the runtime's threads. When the transform is `Identity`
  (`Transform::is_identity`), this hop is skipped.
- **Delivery**: the driver waits for a permit from a semaphore of `prefetch_depth` permits, then
  sends the batch and its permit on a channel. The consumer releases the permit when it takes the
  batch. At most `prefetch_depth` finished batches wait in the channel, and one more waits for
  room.
- **The consumer** blocks on that channel, a crossbeam receiver, so the Python binding can wait
  with a deadline, as it does on the sync path.

To size an async loader: `num_workers × item_concurrency` is the number of requests in flight;
`runtime_threads` is the CPU given to the network (receiving data, TLS, copying bodies), and one
thread can drive many requests; the loader's pool is the CPU given to transforms and collation.

### 2.7 End of the epoch

After the last batch has been returned, the iterator yields `None`. The pool's threads stay
alive, idle, for the next epoch; an async loader's runtime does too.

## 3. Transforms and collation

**Transforms.** A transform runs where the item is loaded: right after `get` (or `get_batch`) on
the worker's thread for a `Dataset`, in each lane when there are several, and on the loader's
pool as each item arrives for an `AsyncDataset`. It receives the item's dataset index, whatever
order the sampler chose. `get_batch` must return exactly one item per index; the loader fails
the batch otherwise, so items and indices can never be paired wrongly. `Transform::apply_batch`
transforms a whole batch in one call; by default it applies the transform to each item in turn,
and `Identity` returns the items as they are, without a pass over them.

**Image operations** (feature `image`) follow the same split. Decoding, colour conversion, crop,
padding, resizing, conversion to floats and normalisation are each a transform of one item,
chained with `then`. The chain is typed by sample type and layout (`Image<u8>` to `Image<f32>`,
height × width × channels to channels × height × width), so a wrong order does not compile, and
the items of a batch are processed in parallel like any other transform. Only the copy into the
batch buffer is the collator's job.

**`ImageCollator`** writes the images of a batch once into one contiguous buffer. With
`reuse_buffers(n)`, the buffers of up to `n` dropped batches are kept and reused, which avoids
the page faults of fresh memory (for a 38.5 MB batch, three quarters of the collation time).
Taking a buffer never waits: when no kept buffer is free, a new one is allocated. A consumer that
holds many batches therefore costs memory, never a deadlock; a pool that made collators wait for
a free buffer would block the pool threads that the batches being held need.

**Built-in datasets** (`src/dataset/io/`) share two pieces. `Locations` says where item `i` is: a
list parsed when the dataset is built, a function of the index, or a `Template` parsed once whose
location is made only when the item is fetched. A template writes string values according to
the target (`Escape`): percent-encoded in URLs, and as they are in paths and object keys, where a
value containing `/` (or `\` or NUL in a path) or equal to `.` or `..` is refused when the dataset
is built, and an item whose values would make a `.` or `..` path component fails. `Retry` says
when to try again: `Retry::run` loops with exponential backoff and random jitter, and each
dataset classifies its own failures as transient or not (HTTP statuses and transport errors,
file I/O error kinds, its own per-attempt time-out). `ObjectStoreDataset` instead turns its
`Retry` into the `RetryConfig` of the store it opens, which retries by itself, so no second loop
runs on top.

## 4. Failures and early exits

### 4.1 Errors

`Dataset::get`, `get_batch`, `Transform::apply` and `Collator::collate` return a `Result`. An
error belongs to its batch: the iterator yields `Some(Err(e))` for that batch and continues with
the next one, on every path. The error type is `Box<dyn Error + Send + Sync>`, so any error
converts into it with `?`.

### 4.2 Panics

On the direct path, a panic in the dataset, the transform or the collator unwinds through
`next()` as usual. On the parallel path, each batch task runs its batch inside `catch_unwind`,
because a panic escaping a rayon task would abort the process. The task sends the panic's payload
to the iterator and cancels the epoch, and the iterator re-raises it with `resume_unwind`, so the
caller sees the same panic as with `num_workers = 0`. The payload is re-raised as soon as it
arrives, which can be before earlier batches have been returned.

On the async path, a panic in a transform or a collation is caught on the pool and resumed inside
the batch task; tokio catches the batch task's panic, the driver forwards it, and the iterator
re-raises it at that batch's turn. The epoch then ends.

### 4.3 Dropping an iterator early

Dropping an iterator in the middle of an epoch (with `break`, after an error, or with `drop`)
marks the epoch as cancelled, and the batch tasks stop claiming batches. A task in the middle of
a batch finishes that batch in the background and its result is discarded; `drop` itself does
**not** wait. On the async path, dropping the iterator aborts the driver task, which drops, and
so aborts, every batch task and its futures: requests in flight are cancelled, not completed.
The Python binding adds a wait before the drop returns (section 6.6).

### 4.4 `fork()`

A process created with `fork()` (for example by Python's `multiprocessing` with the `fork` start
method) inherits the pool's memory but none of its threads. The pool records the process ID that
built it, and `iter()` in another process builds a new pool. The inherited pool is leaked rather
than dropped: dropping it would signal threads that do not exist, through locks that a parent
thread may have held at the moment of the fork. An async loader's runtime is not rebuilt: an
epoch started in the child process returns an error.

### 4.5 Interpreter shutdown (Python)

While the Python interpreter shuts down, after the `atexit` handlers, pool threads can no longer
attach to it. Dropping an iterator then does not wait for its tasks, and a task that would need
the interpreter fails with a `RuntimeError` instead of panicking. An `atexit` handler also keeps
every loader's pool alive until the process ends: a pool freed during shutdown would wake its
threads into a finalizing interpreter, which aborts the process on CPython 3.11 to 3.13.

## 5. Memory safety

The crate is `#![forbid(unsafe_code)]`: it contains no `unsafe` block, and the compiler checks
it. The workers receive `Arc` clones of the dataset, the transform and the collator, never
references into the loader. Neither leaking an iterator (with `mem::forget`) nor dropping the
loader while an iterator is alive can leave a thread reading freed memory: a leaked iterator
simply keeps the dataset alive.

`Dataset`, `Transform` and `Collator` must be `Send + Sync + 'static`, and items and batches
`Send + 'static`, which is what sharing them across threads requires.

## 6. The Python binding

With the `python` feature, `src/python/` builds the `dataloader_rs` extension module. It is
declared with `#[pymodule(gil_used = false)]`, so it also loads on free-threaded CPython without
turning the GIL back on.

| Python | Rust |
|---|---|
| `PyDataset` (subclass it) | `Py<PyDatasetBase>`, which implements `Dataset` by calling `__getitem__`, `__getitems__` and `__len__` |
| `PyAsyncDataset` (subclass it) | `PyAsyncItems`, an `AsyncDataset` that runs the coroutines on an asyncio event loop thread |
| `HttpDataset`, `FileDataset`, `ObjectStoreDataset` | The Rust datasets, whose items are `Bytes` |
| `PyDataloader` | A `DataLoader` over one of the three kinds of dataset above, with `PySampler`, `PyCollator` and `PyTransform` |
| `PyDataloaderIter` | One epoch: the direct path, the sync parallel iterator, or the async iterator |
| `transform=f` | `PyTransform`, the loader's `Transform`, which calls `f(index, item)`; for a Rust dataset it makes `bytes` first |
| `transform=dataloader_rs.transforms.Compose([...])` | `PyTransform::Native`: the Rust image operations, run without the interpreter |
| `collate_fn=ImageCollator()` | `PyCollator::Images`: `ImageCollator` in Rust; batches are `ImageBatch` objects |

`PyDataloader(dataset)` chooses the loader from the dataset's base class: a `PyDataset` gets the
sync loader, and a `PyAsyncDataset` or a Rust dataset gets the async loader. When the loader is
built, it refuses `async def` methods on a `PyDataset`, plain methods on a `PyAsyncDataset`, a
class with both bases, and a missing `__len__`, with a `TypeError`. A plain `__getitem__` that
returns an awaitable fails on the first item of a batch, and the coroutines are closed.

### 6.1 Holding and releasing the GIL

- **`num_workers = 0`.** `__next__` already holds the interpreter, so it calls the dataset
  directly: `__getitems__(indices)` once per batch if the dataset defines it, otherwise
  `__getitem__` for each index (looked up once per epoch), then `transform` for each item, then
  `collate_fn`, or a plain `list` without one.
- **`num_workers > 0`.** Each batch task attaches to the interpreter **once per batch**, in
  `Dataset::get_batch`, not once per item. The consumer releases the interpreter only when it has
  to wait for the next batch, so the workers can run; a batch that has already arrived is
  returned without releasing it. The wait happens in slices of 100 ms, with Python's signal
  handlers run in between, so Ctrl-C interrupts it.
- **`item_concurrency > 1`.** The lanes of a batch call `__getitem__` and `transform`
  concurrently, each attaching to the interpreter only for its own calls, and `__getitems__` is
  not used. Such epochs always use the parallel iterator, even with `num_workers = 0`, because
  the consumer must wait with the interpreter released.

On regular CPython, the Python code of the tasks runs one thread at a time, under the GIL, while
work that releases it (file I/O, NumPy, hashing, Rust code) runs in parallel. On free-threaded
CPython, all of it runs in parallel.

### 6.2 Pool threads stay attached

The loader's pool threads are created with the loader and serve every epoch. Each one attaches
to the interpreter once, when it starts, keeps its Python thread state for its whole life, and
runs rayon's loop with the interpreter released (`src/python/pool.rs`). When a task attaches
later, it only re-activates that thread state. Creating and destroying a thread state on every
attach would cost about 100 µs on free-threaded builds, paid by every batch and every item lane.

### 6.3 `PyAsyncDataset`: an asyncio loop of its own

For a `PyAsyncDataset`, the loader starts an asyncio event loop on a daemon thread named
`dataloader-asyncio`. For each item, a thread of rayon's global pool calls `__getitem__` (or
`__getitems__`) to create the coroutine, and submits it to the loop with
`asyncio.run_coroutine_threadsafe`. When the coroutine completes, a done-callback sends the
result through a one-shot channel to the future that waits for it in the batch task. The global
pool's threads stay attached to the interpreter, like the loader's own; attaching a tokio thread
for every item would create a thread state each time, three times the cost of everything else on
free-threaded builds. The global pool is built with the first such loader, with the interpreter
released, because a starting thread may need every attached thread to pause.

Dropping an item's future (when an epoch is cancelled) cancels its coroutine. When the loader is
dropped, it awaits the dataset's `aclose()` on the loop, if the dataset defines one (for at most
5 seconds; errors and time-outs are logged), then stops the loop. The loop cancels the
coroutines still pending and waits for them (again for at most 5 seconds) before it closes, so
that every coroutine created is awaited.

### 6.4 Rust datasets and image batches without copies

A Rust dataset fetches without the interpreter. Turning an item into `bytes` and calling a
Python `transform` run on the loader's pool, attaching only for those calls.

With a `dataloader_rs.transforms` transform, the items stay Rust values. Such a transform holds a
list of operations, each wrapping the Rust transform of the same name. The value flowing through
them (bytes, or an image of one of four combinations of sample type and layout) is checked at
each step, with an error that says what the step accepts. On a Rust dataset's items, the whole
chain runs without attaching; on a Python dataset's items, it attaches only to copy the `bytes`.
`ImageCollator` then collates the items without the interpreter, so nothing attaches until
`__next__` wraps the batch. With another `collate_fn`, or none, the items reach Python as `Image`
objects.

`Image` and `ImageBatch` keep their pixels in Rust's buffer and describe it with
`__array_interface__` (address, shape, type and C-contiguous layout). `np.asarray(batch)` is
therefore a view whose `base` is the batch, so the buffer lives as long as any view of it.
`__dlpack__` delegates to that NumPy view, so `torch.from_dlpack(batch)` shares the same memory.
Batches are writable (a tensor's memory must be) and single images are read-only. No `unsafe`
code is involved: the buffer's address is passed as an integer. A batch from a collator with
`buffers=n` returns its buffer to the pool once the batch object and every view of it are gone.

### 6.5 Samplers and errors

`shuffle=True` uses the Rust `RandomSampler`, seeded with `seed` or from the operating system.
`sampler=` accepts any finite Python iterable of `int`. It is read whole, under the interpreter,
at the start of each epoch in `iter(loader)`, with signal handlers run every 65,536 indices so
that Ctrl-C stops a sampler that never ends. `len(loader)` uses `len(sampler)` when a sampler is
given, so a sampler without `__len__` raises `TypeError`, as in PyTorch.

Python exceptions cross the Rust core as boxed `PyErr` values and are restored with their
original type and traceback. Errors that did not come from Python become `RuntimeError`, except
those of the built-in datasets, which become `IndexError`, `ValueError` (an item over
`max_bytes`), `TimeoutError` or `OSError`. An exception is raised for its batch, and iteration
can continue, with or without workers, as in PyTorch. A sampler or a `__len__` that raises makes
`iter(loader)` or `len(loader)` raise: an epoch never comes out silently empty or short.

### 6.6 Dropping an iterator

Dropping a `PyDataloaderIter` cancels its epoch (section 4.3), then waits, with the interpreter
released and for at most 5 seconds, for the batch tasks in flight to finish their batch and
exit. Those tasks hold the dataset, the transform and the collator, so after the drop returns,
nothing but the loader refers to them: a `break`, or an iterator collected by the garbage
collector, cannot leave Python objects in use on another thread.

Loaders and iterators are not tied to a thread: any thread may use or drop them, one at a time
(PyO3's borrow check refuses concurrent use). They report the Python objects they hold to the
garbage collector (`__traverse__`), so a reference cycle through a loader is freed by one
`gc.collect()` once no batch is running Python code on it.

## 7. Design choices

| Choice | Why | Cost |
|---|---|---|
| Threads instead of worker processes | Items move by value, with no pickling and no process to start per epoch; the dataset is shared, not copied. | On regular CPython, pure-Python code is serialised by the GIL (the free-threaded builds remove this). |
| One pool per loader, with batch tasks on it | No thread is spawned per epoch, and batch-level and item-level work share threads instead of oversubscribing the CPU. | A fixed cost per batch, visible on nearly free batches; the pool must be rebuilt after `fork()`. |
| A generic `DataLoader<Src, S, C, T>` | Fetching, transforming and collating are compiled for the exact types, with no boxing. | The item and batch types are part of the loader's type. |
| The transform belongs to the loader, not to the dataset | A dataset is only a source; the transform has one place and one threading rule (the loader's threads), and it gets the item's index. | `Transform::is_identity` is needed so the async path can skip the pool for `Identity`. |
| Image operations as per-item transforms; the collator only copies | Decoding and padding run in parallel across a batch's items; the collator, which runs once per batch, does a single copy. | `ImageCollator` cannot fuse padding into its copy. |
| Batch buffers reused through a pool that never waits | A fresh 38.5 MB buffer spends three quarters of its collation in page faults; a pool that waited could deadlock while the consumer holds batches. | Up to `n` idle buffers stay allocated. |
| All the batches of an epoch computed by `iter()` | The epoch's order is fixed before any work starts, and `len()` is exact. | One `Vec<Vec<usize>>` per epoch. |
| A claim window | At most `num_workers + prefetch_depth` batches are in flight or waiting, even behind one slow batch. | A slow batch holds up the window instead of letting later batches run far ahead. |
| Errors reported per batch | One bad item does not end a Rust epoch. | Callers handle a `Result` per batch. |
| Panics forwarded to the consumer | The behaviour is the same with and without workers, and an epoch is never cut short silently. | A panic is re-raised as soon as it arrives. |
| Sync and async datasets as two sources of one `DataLoader` | One loader type, one iterator, the same settings and semantics; each path is compiled separately and chosen at compile time. | The source is part of the loader's type (`SyncSource<D>` or `AsyncSource<A>`). |
| Futures for fetching, rayon for CPU work | Thousands of requests in flight on a few runtime threads; transforms and collation never block the runtime. | A hand-off through a one-shot channel for each transform and each batch. |
| The built-in datasets return bytes, decoded by the transform | Fetching, retries and size limits live in one place; how to decode is the user's choice. | None. |
| One `Retry`, and one list, template or function of locations, for every built-in dataset | The same options everywhere; each backend knows its own transient failures; `object_store` keeps its own retry loop. | Object stores also stop retrying 3 minutes after the first attempt (the `object_store` default). |
