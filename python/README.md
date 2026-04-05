# dataloader-rs

A high-performance DataLoader for Python, backed by a Rust core.
Drop-in replacement for `torch.utils.data.DataLoader` — same interface, lower overhead, no PyTorch dependency. Works with any batch type: plain Python objects, NumPy arrays, or tensors via a custom `collate_fn`.

## Installation

```bash
pip install dataloader-rs
```

Requires **Python ≥ 3.13**. Free-threaded builds (`3.13t`) are supported and recommended for parallel workloads — see [GIL note](#free-threaded-python-313t) below.

## Quick start

### 1 — Define a dataset

Subclass `PyDataset` and implement `__len__` and `__getitem__`:

```python
from dataloader_rs import PyDataset, PyDataloader

class RangeDataset(PyDataset):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return {"x": index, "y": index ** 2}
```

### 2 — Iterate

```python
loader = PyDataloader(RangeDataset(100), batch_size=16)

for batch in loader:
    # batch is a list of 16 dicts by default
    process(batch)
```

The loader is **reusable**: iterating it again starts a new epoch from scratch.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `PyDataset` | — | Map-style dataset (required) |
| `batch_size` | `int` | `1` | Number of samples per batch |
| `shuffle` | `bool` | `False` | Shuffle indices each epoch |
| `sampler` | `Iterable[int]` | `None` | Custom index iterable; mutually exclusive with `shuffle` |
| `num_workers` | `int` | `0` | Worker threads for parallel prefetch |
| `collate_fn` | `Callable[[list], Any]` | `None` | Merge a list of samples into a batch; defaults to returning a plain `list` |
| `drop_last` | `bool` | `False` | Drop the final batch if it is smaller than `batch_size` |
| `prefetch_depth` | `int` | `1` | Number of batches to buffer ahead (only relevant when `num_workers > 0`) |

## Custom collate

```python
import numpy as np

def collate_numpy(samples: list[dict]) -> dict:
    return {
        "x": np.array([s["x"] for s in samples]),
        "y": np.array([s["y"] for s in samples]),
    }

loader = PyDataloader(RangeDataset(100), batch_size=32, collate_fn=collate_numpy)
```

## Custom sampler

Any iterable of `int` indices works:

```python
import random

class WeightedSampler:
    def __init__(self, weights: list[float]) -> None:
        self.weights = weights

    def __iter__(self):
        population = range(len(self.weights))
        yield from random.choices(population, weights=self.weights, k=len(self.weights))

loader = PyDataloader(dataset, batch_size=8, sampler=WeightedSampler(weights))
```

## Parallel prefetch

Set `num_workers > 0` to prefetch batches on background threads while your training loop runs:

```python
loader = PyDataloader(
    dataset,
    batch_size=64,
    num_workers=4,
    prefetch_depth=8,
    collate_fn=collate_numpy,
)
```

> **Note** — worker threads call `dataset.__getitem__` concurrently. Make sure your dataset is **thread-safe** (read-only state, or protected with a lock).

## Free-threaded Python (3.13t)

With a standard CPython build the GIL serialises worker threads, limiting parallel speedup. Install a **free-threaded** Python 3.13t build to remove this constraint:

```bash
# with uv
uv python install 3.13t
uv run --python 3.13t python -X gil=0 train.py
```

On a CPU-bound dataset, free-threaded mode delivers near-linear scaling with `num_workers`.

## Compared to PyTorch DataLoader

| Feature | `dataloader-rs` | `torch.utils.data.DataLoader` |
|---------|-----------------|-------------------------------|
| Dependency | none | PyTorch (large) |
| Worker model | threads | processes |
| GIL bypass | Python 3.13t | always (separate process) |
| Tensor collation | bring your own | built-in |
| Drop-in API | yes | — |

Use `dataloader-rs` when you want a lightweight, dependency-free loader — it handles any batch type, including tensors produced by a custom `collate_fn`.
Use PyTorch's loader when you need its built-in tensor collation or are already deep in the PyTorch ecosystem.

## Links

- [GitHub](https://github.com/AzHicham/dataloader-rs)
- [Rust crate docs](https://docs.rs/dataloader-rs)
