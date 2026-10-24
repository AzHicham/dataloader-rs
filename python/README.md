# dataloader-rs for Python

A `DataLoader` for Python with the `torch.utils.data.DataLoader` interface and a Rust core. Workers
are threads, not processes: samples are handed over by reference instead of being pickled, which
makes it about **7× faster than PyTorch on array samples** — with no PyTorch dependency.

- **PyTorch's arguments**: `batch_size`, `shuffle`, `sampler`, `num_workers`, `collate_fn`,
  `drop_last`, plus `seed` for reproducible shuffling and `prefetch_depth`.
- **Any batch type**: lists by default, NumPy arrays or tensors from your `collate_fn`.
- **Batched fetching**: `__getitems__(indices)` is used when your dataset defines it.
- **Exceptions keep their type**: a `KeyError` in `__getitem__` reaches you as a `KeyError`.
- **Free-threaded Python**: built for CPython 3.14t, where worker threads run Python code in
  parallel.

Requires Python ≥ 3.13.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [PyDataloader arguments](#pydataloader-arguments)
- [Batches and `collate_fn`](#batches-and-collate_fn)
- [Samplers](#samplers)
- [Workers, threads and the GIL](#workers-threads-and-the-gil)
- [Errors](#errors)
- [Coming from `torch.utils.data.DataLoader`](#coming-from-torchutilsdatadataloader)
- [Performance](#performance)

## Installation

```bash
pip install dataloader-rs
```

From source (needs a Rust toolchain): `pip install maturin && maturin develop --release --features python`.

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
| `num_workers` | `0` | worker threads; `0` loads each batch in the calling thread |
| `prefetch_depth` | `1` | how far ahead workers may load: at most `num_workers + prefetch_depth` batches in progress or waiting (with `num_workers > 0`, > 0) |
| `collate_fn` | `None` | `list[sample] -> batch`; without it a batch is a `list` |
| `drop_last` | `False` | drop a final batch smaller than `batch_size` |

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

`num_workers > 0` gives the loader a pool of `num_workers` threads, created once and reused by
every epoch; they load batches ahead while your loop runs and the loader returns them in order.

- **Regular CPython**: the threads share the GIL. Work that releases it — file reads, NumPy,
  image decoding, hashing, most C extensions — runs in parallel; pure-Python code in
  `__getitem__` does not.
- **Free-threaded CPython 3.14t**: no GIL, so pure-Python code runs in parallel too.

  ```bash
  uv python install 3.14t
  uv run --python 3.14t python -X gil=0 train.py
  ```

- **Breaking out of a loop** returns immediately; batches already being loaded finish in the
  background and are discarded.
- **`multiprocessing` with `fork`**: a child process can use a loader its parent already
  iterated; the loader rebuilds its threads there.

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

On an Intel Core i9-13900H, against `torch.utils.data.DataLoader` with the same dataset code:
≈ 7× faster on array samples (file reads, image augmentation), 1.5–61× faster without workers,
and slower (torch 1.6×) on pure-Python CPU work under the GIL, where processes win — use 3.14t.
Every number, the method and the caveats:
[BENCH.md](https://github.com/AzHicham/dataloader-rs/blob/dev/BENCH.md).

Source, issues and the Rust crate: [github.com/AzHicham/dataloader-rs](https://github.com/AzHicham/dataloader-rs).
