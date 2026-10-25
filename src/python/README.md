# The Python binding (feature `python`)

The PyO3 module `dataloader_rs`, built by maturin into the `dataloader-rs` Python package. This is
the implementation guide; the user guide is [python/README.md](../../python/README.md).

| File | What |
|---|---|
| `mod.rs` | the `#[pymodule]` (`gil_used = false`: loads on free-threaded CPython), `into_py_err` |
| `dataset.rs` | `PyDatasetBase` (the `PyDataset` base class), `Dataset` for `Py<PyDatasetBase>`, `Fetcher` (`__getitems__` or `__getitem__`) |
| `dataloader.rs` | `PyDataloader`: arguments, validation, `__iter__`, `__len__` |
| `iterator.rs` | `PyDataloaderIter`: the direct and threaded epochs, `__next__` |
| `sampler.rs` | `PySampler` (sequential, random, or a Python iterable) behind `SharedPySampler` |
| `collator.rs` | `PyCollator`: `collate_fn`, or a plain `list` |
| `pool.rs` | `run_attached`: the pool threads' main function, one thread state per thread |

## How a Python epoch runs

```text
iter(loader)
  ├─ len(dataset)                     raises if __len__ raises
  ├─ sampler → batches of indices     a Python sampler is iterated here, under the interpreter
  ├─ num_workers == 0, items_in_parallel == 1 → Direct    __next__ fetches + collates itself
  └─ otherwise → Threaded              the Rust core; one attach per batch, or per item
```

- **Direct**: the `Fetcher` (a bound `__getitems__`, or a bound `__getitem__`) is resolved once
  per epoch; `__next__` calls it, then `collate_fn`.
- **Threaded**: the core's worker threads call `Dataset::get_batch` on `Py<PyDatasetBase>`, which
  attaches to the interpreter once for the whole batch and resolves the `Fetcher` there.
  `__next__` returns a batch that already arrived directly, and releases the interpreter
  (`py.detach`) only when it has to wait on the channel.
- **Pool threads keep a thread state** (`pool.rs`): each attaches once when it starts and runs
  rayon's loop detached, so attaching from a task re-activates its thread state instead of
  creating and destroying one, which costs about 100 µs on free-threaded builds.
- **`items_in_parallel > 1`**: the epoch uses the core iterator even with `num_workers = 0`; the
  batch's lanes fetch its items with one `Dataset::get` (one attach, one `__getitem__`) each.

## Rules this code keeps

- **Never wait for workers while holding the interpreter.** A worker may be waiting to attach
  (inside `__getitem__` or `collate_fn`); `__next__` detaches before it waits on the channel, and
  dropping an iterator does not wait at all.
- **Python exceptions keep their type.** They cross the core as boxed `PyErr`s; `into_py_err`
  downcasts them back, and only non-Python errors become `RuntimeError`.
- **Nothing ends an epoch silently.** A raising `__len__`, a sampler that raises or yields a
  non-index, a sampler without `__len__` in `len(loader)`: each raises its own exception instead
  of producing an empty or shorter epoch.
- **The loader outlives its iterators.** `PyDataloaderIter` holds a strong reference to its
  `PyDataloader`.
