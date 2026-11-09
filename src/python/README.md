# The Python binding (feature `python`)

The PyO3 module `dataloader_rs`, built by maturin into the `dataloader-rs` Python package. This is
the implementation guide; the user guide is [python/README.md](../../python/README.md).

In this guide, a thread *attaches* to the interpreter when it takes the right to run Python code
(the GIL, on builds that have one) and *detaches* when it gives it back; these are PyO3's terms
(`Python::attach`, `py.detach`).

| File | What |
|---|---|
| `module.rs` | The `#[pymodule]`, declared with `gil_used = false` so that it loads on free-threaded CPython without enabling the GIL. |
| `error.rs` | `into_py_err`, which turns the core's errors back into Python exceptions. |
| `dataset/sync.rs` | `PyDatasetBase` (the `PyDataset` base class), `Dataset` for `Py<PyDatasetBase>`, `Fetcher` (`__getitems__` or `__getitem__`) |
| `dataloader.rs` | `PyDataloader`: arguments, validation, `__iter__`, `__len__` |
| `iterator.rs` | `PyDataloaderIter`: the direct, threaded and async epochs, `__next__` |
| `transform.rs` | `PyTransform`: `transform=f`, called as `f(index, item)`, or the Rust image transforms (`Native`); the loader's `Transform` for Python items and for a Rust dataset's `Bytes`; `PyItem`, a Python object or a Rust value |
| `sampler.rs` | `PySampler`: sequential, random, or a Python iterable, and the exception it raised |
| `collator.rs` | `PyCollator`: `collate_fn`, an `ImageCollator` (collated in Rust), or a plain `list` |
| `image/ops.rs` | `Value` (bytes or an image of one type and layout) and `Op` (one Rust image transform), applied without static types |
| `image/transforms.rs` | `dataloader_rs.transforms`: `Transform` (a list of `Op`s, callable) and its constructors `Decode`, `PadTo`, …, `Compose` |
| `image/collator.rs` | `ImageCollator` in Python: Rust values, or `Image` objects and `(Image, …)` tuples |
| `image/batch.rs` | `Image` and `ImageBatch`: `__array_interface__` views and `__dlpack__` |
| `image/array.rs` | `Image.from_array`: a numpy array or CPU tensor (through `np.asarray` and the buffer protocol), checked against `Decode`'s limits, copied into an HWC image |
| `pool.rs` | `run_attached`: the pool threads' main function, one thread state per thread; `attach_global_pool` does the same for rayon's global pool |
| `dataset/asynchronous.rs` | `PyAsyncDatasetBase` (the `PyAsyncDataset` base class), `PyAsyncItems`: coroutines on an asyncio loop thread, submitted from the global pool |
| `dataset/event_loop.py` | that loop: `start(aclose)` returns `submit(coroutine)` and `stop()`; compiled into the module |
| `dataset/rust.rs` | `RustDataset` and its `HttpDataset` / `FileDataset` / `ObjectStoreDataset` subclasses |
| `dataset/kind.rs` | a dataset's kind from its base class, and the checks its methods must pass |

## How a Python epoch runs

```text
iter(loader)
  ├─ len(dataset)                     raises if __len__ raises
  ├─ sampler → batches of indices     a Python sampler is iterated here, under the interpreter
  ├─ num_workers == 0, item_concurrency == 1 → Direct    __next__ fetches, transforms, collates
  ├─ otherwise → Threaded              the Rust core; one attach per batch, or per item
  └─ PyAsyncDataset or Rust dataset → Async   the async core (futures on tokio), always
```

- **Direct**: the `Fetcher` (a bound `__getitems__`, or a bound `__getitem__`) is resolved once
  per epoch; `__next__` calls it, then `transform` per item and `collate_fn`.
- **Threaded**: the core's worker threads call `Dataset::get_batch` on `Py<PyDatasetBase>`, which
  attaches to the interpreter once for the whole batch and resolves the `Fetcher` there.
  `__next__` returns a batch that already arrived directly, and releases the interpreter
  (`py.detach`) only when it has to wait on the channel, in 100 ms slices that end with
  `check_signals` (Ctrl-C).
- **Pool threads keep a thread state** (`pool.rs`): each attaches once when it starts and runs
  rayon's loop detached, so attaching from a task re-activates its thread state instead of
  creating and destroying one, which costs about 100 µs on free-threaded builds.
- **`item_concurrency > 1`**: the epoch uses the core iterator even with `num_workers = 0`; the
  batch's lanes fetch its items with one `Dataset::get` (one attach, one `__getitem__`) each.
- **Async**: `PyDataloader` builds `DataLoader::builder_async` around `PyAsyncItems` or a Rust
  dataset, with `PyTransform` (which also makes a Rust dataset's items `bytes`); `__next__` waits on it like on the threaded core
  (`wait_next`). See [ARCHITECTURE.md, 6.3 and 6.4](../../ARCHITECTURE.md#63-pyasyncdataset-an-asyncio-loop-of-its-own).

## Rules this code keeps

- **Never wait for workers while holding the interpreter.** A worker may be waiting to attach
  (inside `__getitem__` or `collate_fn`); `__next__` detaches before it waits on the channel, and
  dropping an iterator detaches before it waits (at most 5 s) for the batches in flight.
- **Python exceptions keep their type.** They cross the core as boxed `PyErr`s; `into_py_err`
  downcasts them back, and only non-Python errors become `RuntimeError`.
- **Nothing ends an epoch silently.** A raising `__len__`, a sampler that raises or yields a
  non-index, a sampler without `__len__` in `len(loader)`: each raises its own exception instead
  of producing an empty or shorter epoch.
- **The loader outlives its iterators.** `PyDataloaderIter` holds a strong reference to its
  `PyDataloader`.
- **The garbage collector sees every Python reference, once.** `__traverse__` reports the
  dataset, the Python sampler, `transform` and `collate_fn` from the loader, and `_owner` plus the direct path's
  own references from an iterator, so cycles through them are freed. Running epochs share the
  loader's objects through `Arc`s and hold no references of their own; reporting one twice would
  let the collector free a live object.
