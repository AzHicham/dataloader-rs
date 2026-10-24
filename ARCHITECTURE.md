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

### 3.1 Choosing a path

`iter()` asks the batch sampler for every batch of the epoch up front (a `Vec<Vec<usize>>`), then
picks one of two paths:

```text
num_workers == 0                         num_workers > 0
┌───────────────────────────┐            ┌──────────────────────────────────────────┐
│ Direct path               │            │ Parallel path                            │
│ next() fetches + collates │            │ num_workers threads, spawned per epoch,  │
│ batch i on the caller's   │            │ pull batches from a queue and send       │
│ thread                    │            │ results through a bounded channel        │
└───────────────────────────┘            └──────────────────────────────────────────┘
```

### 3.2 Direct path (`num_workers = 0`)

`next()` takes the next chunk of indices, fetches the samples (`Dataset::get_batch`) and collates
them on the calling thread. No thread, no channel, no allocation beyond the batch itself: for
cheap samples it is the fastest path ([BENCH.md](BENCH.md)).

### 3.3 Parallel path (`num_workers > 0`)

At the start of the epoch:

1. every batch is pushed, with its position `i`, on an **unbounded work queue**
   (`crossbeam_channel`), which is then closed;
2. `num_workers` OS threads are spawned; each one loops: take a batch, fetch it, collate it, send
   `(i, result)` on the **result channel**, whose capacity is `prefetch_depth`;
3. the iterator receives results and returns them **in epoch order**.

```text
 work queue (all batches) ──▶ worker 1 ─┐
                          ──▶ worker 2 ─┼──▶ result channel ──▶ reorder buffer ──▶ next()
                          ──▶ worker N ─┘   (capacity = prefetch_depth)   (by batch index)
```

**Backpressure.** A worker blocks on `send` when `prefetch_depth` results are already waiting, so
at most `num_workers` batches are being built and `prefetch_depth` are finished but not yet
consumed.

**Ordering.** Workers finish out of order. The iterator keeps results that arrive early in a
`HashMap<batch index, result>` and returns batch `next_out` as soon as it is there. While it
waits for one slow batch it keeps draining the channel, so batches finished after it accumulate
in that buffer: one very slow batch can make the iterator hold many later ones.

### 3.4 Item-level parallelism (`intra_workers > 0`)

`build()` creates a rayon pool of `intra_workers` threads, kept for the life of the loader. On
either path, a batch's items are then fetched with `Dataset::get` in parallel on that pool
(`par_iter`), instead of `get_batch` on the batch's thread. With both settings, the loader runs
`num_workers + intra_workers` threads during an epoch.

### 3.5 End of the epoch

When the last batch has been returned, the iterator yields `None`. On the parallel path, dropping
the iterator joins the worker threads (section 4.3); a new `iter()` spawns new ones.

## 4. Failures and early exits

### 4.1 Errors

`Dataset::get`, `get_batch` and `Collator::collate` return `Result`. An error belongs to its
batch: the iterator yields `Some(Err(e))` for that batch and continues with the next one, on both
paths. The error type is `Box<dyn Error + Send + Sync>`, so any error converts with `?`.

### 4.2 Panics

On the direct path a panic in the dataset or collator unwinds through `next()` as usual. On the
parallel path, workers run each batch under `catch_unwind`, send the panic payload to the
iterator and stop; the iterator re-raises it with `resume_unwind`, so the caller sees the same
panic as with `num_workers = 0`. The payload is re-raised when it arrives, which can be before
earlier batches have been returned.

### 4.3 Dropping an iterator early

Dropping an iterator mid-epoch (a `break`, an error, `drop`) sets a cancel flag, drops the result
receiver — a worker blocked on `send` gets an error and exits — and **joins** the workers. A
worker in the middle of a batch finishes that batch first, so `drop` can take as long as one
batch.

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
- **`num_workers > 0`**: each worker attaches to the interpreter **once per batch**
  (`Dataset::get_batch`), not once per item. The consumer releases the interpreter
  (`py.detach`) while it waits for the next batch, so workers can run.

On regular CPython the workers' Python code is serialised by the GIL; work that releases it
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

`PyDataloaderIter`'s `Drop` releases the interpreter before joining the workers: a worker waiting
to attach (to call `__getitem__`) can then finish, so a `break` or a garbage-collected iterator
cannot deadlock. A worker panic is likewise re-raised only after the workers are joined.

## 7. Design choices

| Choice | Why | Cost |
|---|---|---|
| Threads instead of worker processes | samples move by value, no pickling, no per-epoch process start; the dataset is shared, not copied | pure-Python code is serialised by the GIL on regular CPython (use 3.14t) |
| Generic `DataLoader<D, S, C>` | monomorphized fetch and collate, no boxing | the batch type is part of the loader's type |
| Batches computed at `iter()` | the epoch's order is fixed before any work starts; `len()` is exact | one `Vec<Vec<usize>>` per epoch |
| Bounded result channel | memory of finished batches is bounded by `prefetch_depth` | the reorder buffer can still grow behind one slow batch (section 3.3) |
| Errors per batch | one bad sample does not end a Rust epoch | callers must handle `Result` per batch |
| Panics forwarded | same behaviour with and without workers; no silent truncation | a panic is re-raised as soon as it arrives |
