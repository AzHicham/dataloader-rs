# dataloader-rs for Python

A `DataLoader` for Python with the `torch.utils.data.DataLoader` interface and a Rust core. Workers
are threads, not processes: samples are handed over by reference instead of being pickled, an
epoch starts in milliseconds, and against PyTorch it is **4.9–6.6× faster on the loader's own
overhead, 3.2–3.9× on network-bound datasets and 2.5–3.5× on array samples** with free-threaded
Python — with no PyTorch dependency.

- **PyTorch's arguments**: `batch_size`, `shuffle`, `sampler`, `num_workers`, `collate_fn`,
  `drop_last`, plus `seed` for reproducible shuffling, `prefetch_depth`, and `items_in_parallel`
  to fetch the items of a batch concurrently.
- **Any batch type**: lists by default, NumPy arrays or tensors from your `collate_fn`.
- **Batched fetching**: `__getitems__(indices)` is used when your dataset defines it.
- **Exceptions keep their type**: a `KeyError` in `__getitem__` reaches you as a `KeyError`.
- **Free-threaded Python**: built for CPython 3.14t and 3.15t, where worker threads run Python
  code in parallel.

Requires Python ≥ 3.13.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [PyDataloader arguments](#pydataloader-arguments)
- [Batches and `collate_fn`](#batches-and-collate_fn)
- [Samplers](#samplers)
- [Workers, threads and the GIL](#workers-threads-and-the-gil)
- [I/O-bound datasets](#io-bound-datasets)
- [Errors](#errors)
- [Coming from `torch.utils.data.DataLoader`](#coming-from-torchutilsdatadataloader)
- [Performance](#performance)

## Installation

```bash
pip install dataloader-rs
```

| Wheels | Platforms |
|---|---|
| free-threaded CPython 3.14t, 3.15t | Linux x86_64 and arm64 (glibc ≥ 2.28), macOS arm64 (11.0+) |

Other interpreters and platforms build from the source distribution, which needs a Rust
toolchain; from a checkout: `pip install maturin && maturin develop --release --features python`.

## Quick start

```python
import numpy as np
from dataloader_rs import PyDataloader, PyDataset


class Images(PyDataset):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return np.load(self.paths[index])


loader = PyDataloader(
    Images(paths),
    batch_size=32,
    shuffle=True,
    seed=0,  # reproducible order, a new permutation every epoch
    num_workers=4,  # 4 worker threads
    prefetch_depth=4,  # load up to 4 batches ahead of the 4 in progress
    collate_fn=np.stack,  # one (32, ...) array per batch
    drop_last=True,
)

for epoch in range(10):
    for batch in loader:  # iterating again starts a new epoch
        ...
```

## Datasets

Subclass `PyDataset` and implement `__len__` and `__getitem__(index)`. Call `super().__init__()`
if you define `__init__`.

Optionally implement `__getitems__(indices)` — PyTorch's batched-fetch protocol — to fetch a whole
batch in one call: one database query, one NumPy fancy-indexing, one Python call instead of
`batch_size`. It receives a `list[int]` and must return one sample per index, in order (any
iterable); a wrong count raises `ValueError`. Setting `__getitems__ = None` in a subclass turns it
off, as in PyTorch.

```python
class Rows(PyDataset):
    def __init__(self, table: np.ndarray) -> None:
        super().__init__()
        self.table = table

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.table[index]

    def __getitems__(self, indices: list[int]) -> list[np.ndarray]:
        return list(self.table[indices])  # one vectorised read per batch
```

With `num_workers > 0`, `__getitem__` / `__getitems__` run on several threads at once: keep the
dataset read-only, or protect shared state with a lock.

## PyDataloader arguments

| Argument | Default | Meaning |
|---|---|---|
| `dataset` | — | a `PyDataset` |
| `batch_size` | `1` | samples per batch (> 0) |
| `shuffle` | `False` | a new random order every epoch; exclusive with `sampler` |
| `seed` | `None` | seed for `shuffle=True` (requires it): same seed, same sequence of epoch orders |
| `sampler` | `None` | any iterable of `int` indices, iterated at the start of each epoch |
| `num_workers` | `0` | batches loaded in parallel; `0` loads each batch in the calling thread |
| `prefetch_depth` | `2` | how far ahead workers may load: at most `num_workers + prefetch_depth` batches in progress or waiting (with `num_workers > 0`, > 0) |
| `collate_fn` | `None` | `list[sample] -> batch`; without it a batch is a `list` |
| `drop_last` | `False` | drop a final batch smaller than `batch_size` |
| `items_in_parallel` | `1` | items of one batch fetched in parallel, one `__getitem__` call each (see [I/O-bound datasets](#io-bound-datasets)) |

`len(loader)` is the number of batches per epoch; `len(iterator)` the number left in the current
one.

## Batches and `collate_fn`

Without `collate_fn` a batch is the `list` of its samples. `collate_fn` receives that list and
returns anything: `np.stack`, `torch.stack`, a dict of arrays. It runs on the worker thread that
loaded the batch (or the calling thread with `num_workers=0`), so heavy collation is parallel too.

## Samplers

- `shuffle=True`: a Rust permutation, new every epoch; with `seed=` two loaders produce the same
  sequence of orders.
- `sampler=`: any iterable of indices — a list, a generator-producing object, a weighted sampler.
  It is iterated once at the start of each epoch. `len(loader)` uses `len(sampler)`, so a sampler
  without `__len__` makes `len(loader)` raise `TypeError`, as in PyTorch; iteration still works.

```python
import random


class Weighted:
    def __init__(self, weights: list[float]) -> None:
        self.weights = weights

    def __len__(self) -> int:
        return len(self.weights)

    def __iter__(self):
        population = range(len(self.weights))
        yield from random.choices(population, weights=self.weights, k=len(self.weights))


loader = PyDataloader(dataset, batch_size=8, sampler=Weighted(weights))
```

## Workers, threads and the GIL

`num_workers > 0` loads that many batches in parallel on a thread pool of
`num_workers × items_in_parallel` threads, created once and reused by every epoch; they load
batches ahead while your loop runs and the loader returns them in order.

- **Regular CPython**: the threads share the GIL. Work that releases it — file reads, NumPy,
  image decoding, hashing, most C extensions — runs in parallel; pure-Python code in
  `__getitem__` does not.
- **Free-threaded CPython 3.14t / 3.15t**: no GIL, so pure-Python code runs in parallel too.

  ```bash
  uv python install 3.14t
  uv run --python 3.14t python -X gil=0 train.py
  ```

- **Shared Python objects on free-threaded CPython**: an object's reference count is cheap only for
  the thread that created it; other threads update it atomically. Threads that all read the same
  dict or list millions of times per second (a vocabulary, a lookup table) contend on it and stop
  scaling: a pure-Python tokeniser over one shared vocabulary runs *slower* on 8 threads than on 1
  with 3.14t. Give each thread its own copy of hot lookup tables (`threading.local`), or on 3.15t
  key them with interned strings (`sys.intern`), which no longer carry this cost. Processes
  (PyTorch's workers) never share objects, so they do not see this.

- **Array samples on Linux: tune the allocator NumPy uses.** With default settings, the memory of
  every batch goes back to the system and is faulted back in, which serialises worker threads.
  The settings are process-wide, so they are yours to make, at start-up:

  ```bash
  # CPython 3.13 and 3.14t: NumPy allocates with glibc's malloc
  GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 python train.py
  # CPython 3.15t: NumPy's buffers come from CPython's mimalloc
  MIMALLOC_PURGE_DELAY=-1 python train.py
  ```

  On 3.14t this is worth up to 2× for array samples with 4 workers, and the process keeps some
  free memory for reuse (80–100 MB in the benchmarks). With glibc, raise both thresholds: either
  one alone turns off glibc's adjustment of the other and is slower than the defaults. Batches of
  16 MiB and more hit a mimalloc path no setting avoids on 3.15t: smaller batches are faster
  there.
- **Breaking out of a loop** returns immediately; batches already being loaded finish in the
  background and are discarded.
- **`multiprocessing` with `fork`**: a child process can use a loader its parent already
  iterated; the loader rebuilds its threads there.

## I/O-bound datasets

By default a worker fetches the items of its batch one after another, so a dataset that waits
on the network (an object store, an HTTP API, a database) keeps only `num_workers` requests in
flight. With `items_in_parallel=k`, up to `k` items of each batch are fetched concurrently, one
`__getitem__` call each, so `num_workers × items_in_parallel` requests are in flight:

```python
loader = PyDataloader(
    S3Images(keys),
    batch_size=32,
    num_workers=4,  # 4 batches in parallel
    items_in_parallel=16,  # 16 items of each batch at once: 64 requests in flight
)
```

Waiting on I/O releases the GIL, so this works on regular CPython too. `__getitems__` is not used
when `items_in_parallel > 1`: each item is its own call.

## Errors

- An exception raised by `__getitem__`, `__getitems__` or `collate_fn` reaches you with its own
  type and message. With `num_workers=0`, the next `next()` continues with the following batch;
  with workers, the epoch ends.
- A `__len__` or a sampler that raises, or a sampler that yields something other than a
  non-negative `int`, makes `iter(loader)` raise — an epoch never silently comes out empty or
  short.
- Invalid arguments raise `ValueError` (`batch_size=0`, `shuffle` with `sampler`, `seed` without
  `shuffle`) or `TypeError` (a non-iterable `sampler`).

## Coming from `torch.utils.data.DataLoader`

| PyTorch | dataloader-rs |
|---|---|
| `torch.utils.data.Dataset` | subclass `PyDataset` (same `__len__` / `__getitem__` / `__getitems__`) |
| `batch_size`, `shuffle`, `sampler`, `num_workers`, `collate_fn`, `drop_last` | same |
| `generator=` for a reproducible shuffle | `seed=` |
| `prefetch_factor` (per worker) | `prefetch_depth` (total, beyond the `num_workers` batches in progress) |
| worker processes, `persistent_workers`, `worker_init_fn`, `multiprocessing_context` | threads: nothing to start, copy or initialise per worker |
| `pin_memory`, `timeout`, `batch_sampler`, `IterableDataset` | not supported |
| default collation into tensors | a `list`; pass `collate_fn` (e.g. `np.stack`, `torch.stack`) |

## Performance

Against `torch.utils.data.DataLoader` with the same dataset code, on 4 cores of an Intel Core
i9-13900H, with the allocator settings of [Workers, threads and the GIL](#workers-threads-and-the-gil)
for both: 4.9–6.6× faster on the loader's own overhead, 3.2–3.9× on network-bound datasets
(`items_in_parallel`), first batches in milliseconds against 11–334 ms for torch's fresh worker
processes, and 2.5–3.5× on array samples (file reads, NumPy augmentation) with free-threaded
3.14t / 3.15t, 1.0–1.5× on 3.13. Pure-Python CPU work with per-thread tables is about 1.2×
faster on 3.14t and 3.15t; on shared objects or under the GIL, torch's processes are 2.4–2.9×
faster. Every number, the method and the caveats:
[BENCH.md](https://github.com/AzHicham/dataloader-rs/blob/dev/BENCH.md).

Source, issues and the Rust crate: [github.com/AzHicham/dataloader-rs](https://github.com/AzHicham/dataloader-rs).
