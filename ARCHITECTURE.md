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
 consumer ◀── DataLoaderIter ◀── collate (Collator) ◀── fetch (Dataset) ◀── batch i
```

| Piece | Role | Code |
|---|---|---|
| `Dataset` | `get(index)`, `len()`; `get_batch(indices)` defaults to `get` per index | `src/dataset.rs` |
| `Sampler` | the index sequence of one epoch (`indices(dataset_len)`), and its length | `src/sampler/` |
| `BatchSampler` | cuts the indices into `batch_size` chunks, honours `drop_last` | `src/sampler/batch_sampler.rs` |
| `Collator<Item>` | merges a batch's `Vec<Item>` into one `Batch` value | `src/collator/` |
| `DataLoader` | owns the pieces and the settings; `iter()` starts an epoch | `src/loader/core.rs` |
| `DataLoaderIter` | yields `Result<Batch>` in epoch order | `src/loader/iter.rs` |

`DataLoader::iter(&mut self)` advances the sampler (a `RandomSampler` draws a new permutation
each epoch) but the iterator it returns does not borrow the loader: it holds `Arc` clones of the
dataset and the collator.

## 3. Life of an epoch

### 3.1 The pool

`build()` creates **one rayon pool per loader** when `num_workers > 0` or `parallel_items` is on,
with `num_threads` threads (default: `num_workers`, or the number of CPUs with
`parallel_items`; never fewer than `num_workers`). The pool lives as long as the loader and is
reused by every epoch; its threads are named `dataloader-N`.

| Setting | Meaning |
|---|---|
| `num_workers(n)` | at most `n` batches in flight; `0` selects the direct path |
| `parallel_items(true)` | the items of a batch are fetched in parallel on the same pool |
| `num_threads(t)` | the pool's size |
| `prefetch_depth(d)` | how far ahead of the consumer batches may run: at most `n + d` in flight or buffered |

`intra_workers(n)` is a deprecated alias for `parallel_items(n > 0)` plus `num_threads(n)`.

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

`next()` takes the next chunk of indices, fetches the samples (`Dataset::get_batch`) and collates
them on the calling thread. No thread hand-off: for cheap samples it is the fastest path
([BENCH.md](BENCH.md)).

### 3.4 Parallel path (`num_workers > 0`)

The epoch is a shared state (`Epoch`): the batches, a claim counter `next_claim`, the consumer's
position `next_out`, and a result channel. Up to `num_workers` **batch tasks** run on the pool;
each one loops:

1. claim the next batch `i` — but only while `i < next_out + num_workers + prefetch_depth`
   (the **window**);
2. fetch and collate it, and send `(i, result)` to the iterator;
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

**Ordering.** Tasks finish out of order; a `HashMap<batch index, result>` holds early results
until their turn. The window also bounds this buffer.

**Cost.** Claiming, publishing the position and restarting tasks cost a few atomic operations
and an occasional task spawn per batch. That is negligible once a batch does real work, but
dominates for nearly free batches (`batch_size` ≤ 8 of in-memory items, [BENCH.md](BENCH.md)).

### 3.5 Item-level parallelism (`parallel_items`)

With `parallel_items`, a batch's items are fetched with `Dataset::get` in parallel (`par_iter`)
on the same pool, instead of `get_batch` on one thread. Batch tasks and item work share the
pool's threads (idle threads steal either kind), so the two levels never oversubscribe the CPU.
On the direct path, items are fetched in parallel on the pool too.

### 3.6 End of the epoch

When the last batch has been returned, the iterator yields `None`. The pool's threads stay
alive, idle, for the next epoch.

## 4. Failures and early exits

### 4.1 Errors

`Dataset::get`, `get_batch` and `Collator::collate` return `Result`. An error belongs to its
batch: the iterator yields `Some(Err(e))` for that batch and continues with the next one, on both
paths. The error type is `Box<dyn Error + Send + Sync>`, so any error converts with `?`.

### 4.2 Panics

On the direct path a panic in the dataset or collator unwinds through `next()` as usual. On the
parallel path, batch tasks run each batch under `catch_unwind` (a panic escaping a rayon task
would abort the process), send the payload to the iterator and cancel the epoch; the iterator
re-raises it with `resume_unwind`, so the caller sees the same panic as with `num_workers = 0`. The payload is re-raised when it arrives, which can be before
earlier batches have been returned.

### 4.3 Dropping an iterator early

Dropping an iterator mid-epoch (a `break`, an error, `drop`) marks the epoch cancelled: batch
tasks stop claiming. A task in the middle of a batch finishes it in the background and its
result is discarded; `drop` itself **does not wait**.

### 4.4 `fork()`

A process created with `fork()` (e.g. Python `multiprocessing` with the `fork` start method)
inherits the pool but none of its threads. The pool records the PID that built it; `iter()` in
another process builds a new pool. The inherited one is leaked, not dropped: dropping it would
signal threads that do not exist, through locks a parent thread may have held at fork time.

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
| `PyDataloader` | `DataLoader<PyDataset, SharedPySampler, PyCollator>` |
| `PyDataloaderIter` | the epoch: a direct path, or the core parallel iterator |

### 6.1 Fetching and collating

- **`num_workers = 0`**: `__next__` already holds the interpreter, so it calls the dataset
  directly: `__getitems__(indices)` once per batch if the dataset defines it, else a bound
  `__getitem__` (looked up once per epoch) per index; then `collate_fn`, or a `list`.
- **`num_workers > 0`**: each batch task attaches to the interpreter **once per batch**
  (`Dataset::get_batch`), not once per item. The pool's threads are created with the loader and
  reused by every epoch. The consumer releases the interpreter
  (`py.detach`) while it waits for the next batch, so workers can run.

On regular CPython the tasks' Python code is serialised by the GIL; work that releases it
(file I/O, NumPy, hashlib) runs in parallel. On free-threaded CPython 3.14t all of it does.

### 6.2 Samplers

`shuffle=True` uses the Rust `RandomSampler` (seeded with `seed`, or from OS entropy); `sampler=`
accepts any Python iterable of `int`. The Python sampler is iterated, under the interpreter, at
the start of each epoch in `iter(loader)`. `len(loader)` uses `len(sampler)` when a sampler is
given, so a sampler without `__len__` raises `TypeError`, as in PyTorch.

### 6.3 Errors

Python exceptions cross the Rust core as boxed `PyErr`s and are restored with their original
type and traceback (`into_py_err`); only errors that did not come from Python become
`RuntimeError`. With `num_workers = 0` an exception is raised for its batch and iteration can
continue; with workers it ends the epoch. A sampler or `__len__` that raises makes
`iter(loader)` or `len(loader)` raise.

### 6.4 Dropping an iterator

Dropping a `PyDataloaderIter` cancels its epoch without waiting (section 4.3), and releases the
interpreter first: a task waiting to attach (to call `__getitem__`) can then finish its batch, so a
`break` or a garbage-collected iterator cannot deadlock. A batch already running keeps calling the
dataset in the background until it completes.

## 7. Design choices

| Choice | Why | Cost |
|---|---|---|
| Threads instead of worker processes | samples move by value, no pickling, no per-epoch process start; the dataset is shared, not copied | pure-Python code is serialised by the GIL on regular CPython (use 3.14t) |
| One pool per loader, batch tasks on it | no thread spawn per epoch; batch and item work share threads instead of oversubscribing the CPU | a fixed cost per batch that shows on nearly free batches; the pool must be rebuilt after `fork()` |
| Generic `DataLoader<D, S, C>` | monomorphized fetch and collate, no boxing | the batch type is part of the loader's type |
| Batches computed at `iter()` | the epoch's order is fixed before any work starts; `len()` is exact | one `Vec<Vec<usize>>` per epoch |
| Claim window | at most `num_workers + prefetch_depth` batches in flight or buffered, even behind one slow batch | a slow batch stalls the window instead of letting later batches run far ahead |
| Errors per batch | one bad sample does not end a Rust epoch | callers must handle `Result` per batch |
| Panics forwarded | same behaviour with and without workers; no silent truncation | a panic is re-raised as soon as it arrives |
