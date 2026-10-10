# dataloader-rs

A high-performance DataLoader for Rust with a PyTorch-compatible Python API.

[![CI](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/dataloader-rs)](https://crates.io/crates/dataloader-rs)
[![PyPI](https://img.shields.io/pypi/v/dataloader-rs)](https://pypi.org/project/dataloader-rs/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Typed, streaming batch pipeline with a familiar builder API. Supports sequential
and parallel (inter-batch threads + intra-batch rayon) execution, pluggable
samplers and collators, and Python bindings via PyO3.

See [bench/bench.md](bench/bench.md) for performance comparisons against
`torch.utils.data.DataLoader`.

---

## Rust quick start

Add to `Cargo.toml`:

```toml
[dependencies]
dataloader-rs = "0.1"
```

### Define a dataset

```rust
use dataloader_rs::{Dataset, error::Result};

struct ImageDataset {
    paths: Vec<std::path::PathBuf>,
}

impl Dataset for ImageDataset {
    type Item = Vec<u8>;

    fn get(&self, index: usize) -> Result<Vec<u8>> {
        Ok(std::fs::read(&self.paths[index])?)
    }

    fn len(&self) -> usize {
        self.paths.len()
    }
}
```

### Build a DataLoader and iterate

```rust
use dataloader_rs::{DataLoader, RandomSampler};

let mut loader = DataLoader::builder(ImageDataset { paths })
    .batch_size(32)
    .sampler(RandomSampler::new(42))  // reproducible shuffle
    .num_workers(4)                   // inter-batch worker threads
    .intra_workers(2)                 // rayon threads per batch
    .prefetch_depth(8)                // batches in-flight
    .drop_last(true)
    .build();

for batch in &mut loader {
    let batch: Vec<Vec<u8>> = batch?;
    // train, write to disk, etc.
}
```

### Custom collator

```rust
use dataloader_rs::collator::Collator;

struct StackCollator;

impl Collator<Vec<f32>> for StackCollator {
    type Batch = Vec<Vec<f32>>;

    fn collate(&self, items: Vec<Vec<f32>>) -> dataloader_rs::error::Result<Self::Batch> {
        Ok(items)
    }
}
```

### Async datasets (feature `async`)

For samples behind network calls (object stores, HTTP, databases), implement
`AsyncDataset`: many requests stay in flight on a few threads instead of one
thread per request. The loader only composes futures, so it runs under any
executor (tokio, smol, async-std, `futures::executor`).

```rust,ignore
use dataloader_rs::{AsyncDataLoader, AsyncDataset, error::Result};

struct S3Images { client: Client, keys: Vec<String> }

impl AsyncDataset for S3Images {
    type Item = Vec<u8>;
    async fn get(&self, index: usize) -> Result<Vec<u8>> {
        Ok(self.client.get(&self.keys[index]).await?)
    }
    fn len(&self) -> usize { self.keys.len() }
}

let mut loader = AsyncDataLoader::builder(S3Images { client, keys })
    .batch_size(64)
    .concurrency(4)       // batches in flight, each with all samples concurrent
    .max_concurrency(128) // optional: at most 128 samples in flight overall
    .collate_threads(2)   // CPU collation off the executor
    .build();

// From your runtime: a Stream of batches, in sampler order.
let mut batches = std::pin::pin!(loader.stream());
while let Some(batch) = batches.next().await { /* ... */ }

// Or from synchronous code: a blocking iterator driven on a background thread.
for batch in loader.iter() { /* ... */ }
```

`iter()` drives the stream with `futures::executor`, which cannot run futures
that need tokio (reqwest, the AWS SDK, `tokio::time`). Enable the `tokio`
feature to drive it on a tokio runtime instead: one the loader creates and
keeps for its lifetime, or yours via `.tokio_handle(handle)`.

---

## Python quick start

Install:

```bash
pip install dataloader-rs
```

Or build from source (requires a Rust toolchain):

```bash
pip install maturin
maturin develop --features python
```

### Define a dataset

```python
from dataloader_rs import PyDataset

class MyDataset(PyDataset):
    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with open(self.paths[index], "rb") as f:
            return f.read()
```

### Build a DataLoader and iterate

```python
from dataloader_rs import PyDataloader

loader = PyDataloader(
    MyDataset(paths),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_depth=8,
    drop_last=True,
)

for batch in loader:
    # batch: list[bytes]
    ...
```

### Custom collate function

```python
import numpy as np

def numpy_collate(items: list[bytes]) -> np.ndarray:
    return np.stack([np.frombuffer(b, dtype=np.uint8) for b in items])

loader = PyDataloader(MyDataset(paths), batch_size=32, num_workers=4,
                      collate_fn=numpy_collate)
```

### Custom sampler

Any iterable of indices is accepted:

```python
import random

class WeightedSampler:
    def __init__(self, weights: list[float]):
        self.weights = weights

    def __iter__(self):
        population = list(range(len(self.weights)))
        return iter(random.choices(population, weights=self.weights, k=len(population)))

loader = PyDataloader(dataset, batch_size=32, sampler=WeightedSampler(weights))
```

---

## Architecture

### Execution paths

`DataLoader::iter()` selects one of two paths:

```
inter_workers == 0               inter_workers > 0
┌─────────────────────┐          ┌──────────────────────────────┐
│     Direct path     │          │        Parallel path          │
│                     │          │                               │
│  Process each batch │          │  N worker threads pull from   │
│  inline in next()   │          │  unbounded work queue, push   │
│                     │          │  results to bounded channel   │
│  Zero allocation,   │          │  (capacity = prefetch_depth)  │
│  no thread overhead │          │                               │
└─────────────────────┘          └──────────────────────────────┘
```

Both paths can enable a **rayon thread pool** (`intra_workers > 0`) to fetch
items within a single batch in parallel.

### Key design points

**Fully generic — monomorphized hot path.**
`DataLoader<D, S, C>` is generic over dataset, sampler, and collator. Every
code path is monomorphized — no virtual dispatch, no boxing, no dynamic
allocation in the iteration loop.

**Bounded channel as backpressure.**
`prefetch_depth` is the exact capacity of the `crossbeam` result channel.
Workers block when the channel is full, giving precise control over memory
without over-prefetching.

**Reorder buffer preserves epoch order.**
Workers complete batches out of order. A `HashMap<batch_idx, Batch>` reorder
buffer in the iterator reassembles results in epoch order regardless of
scheduling.

**No unsafe code.**
The dataset and collator live behind `Arc`; each worker holds its own clone,
so an iterator never borrows the loader and leaking it (e.g. `mem::forget`)
cannot leave a worker reading freed memory. The crate is `#![forbid(unsafe_code)]`.

**Python: GIL acquired once per batch.**
`PyDataset::get_batch` acquires the Python thread state once for all items in
a batch, reducing GIL overhead from O(batch\_size) to O(1) per batch in the
threaded path.

### Module layout

```
src/
├── dataset.rs          Dataset + IterableDataset traits
├── error.rs            Error / Result type
├── sampler/
│   ├── sequential.rs   SequentialSampler
│   ├── random.rs       RandomSampler (inside-out Fisher-Yates, fastrand wyrand)
│   ├── distributed.rs  DistributedSampler (padded strided sharding)
│   └── batch_sampler.rs BatchSampler (chunks indices into batches)
├── collator/
│   ├── vec_collator.rs  VecCollator (collect items into Vec — default)
│   ├── default_collator/ DefaultCollator (tensor stacking, ndarray, primitives)
│   └── torch.rs         TorchPinnedCollator (pin_memory for tch tensors)
└── loader/
    ├── builder.rs      DataLoaderBuilder (type-safe builder)
    ├── core.rs         DataLoader struct + IntoIterator
    ├── iter.rs         DataLoaderIter, ParallelCore
    └── worker.rs       process_batch, worker_loop
```

---

## Building & testing

### Prerequisites

- Rust stable ≥ 1.91
- Python ≥ 3.13 (free-threaded 3.13t / 3.14t recommended for parallel benchmarks)
- [uv](https://docs.astral.sh/uv/) for Python environment management

### Rust

```bash
# Library
cargo build

# With Python bindings
cargo build --features python

# Tests
cargo test
cargo test --features python
```

### Python

```bash
# Install dev dependencies (maturin, pytest, torch, ...)
uv sync --group dev

# Build and install the extension in development mode
uv run maturin develop --features python

# Run all Python tests
uv run pytest -v
```

### Benchmarks

#### Rust (Criterion)

```bash
cargo bench                              # all benches
cargo bench --bench bench_inter_workers  # single bench
cargo bench -- num_workers/4            # filter within a bench
```

HTML reports: `target/criterion/`.

#### Python regression (pytest-benchmark)

```bash
uv run pytest bench/ -m bench \
    --benchmark-enable \
    --benchmark-warmup=on \
    --benchmark-min-rounds=10 \
    -v
```

#### Python vs PyTorch (full sweep)

```bash
uv run python bench/bench_batch_size.py    --warmup 2 --repeats 10
uv run python bench/bench_inter_workers.py --warmup 2 --repeats 10
uv run python bench/bench_prefetch_depth.py --warmup 2 --repeats 10
uv run python bench/bench_sampler.py       --warmup 2 --repeats 10
```

---

## Features

| Feature | Default | Description |
|---|---|---|
| `python` | ❌ | PyO3 bindings (`PyDataloader`, `PyDataset`) |
| `ndarray` | ❌ | `DefaultCollator` for `ndarray::Array` types |
| `torch-rs` | ❌ | `TorchPinnedCollator` for `tch::Tensor` + `pin_memory` |
| `async` | ❌ | `AsyncDataset` + `AsyncDataLoader` (runtime-agnostic, adds `futures`) |
| `tokio` | ❌ | Implies `async`; `AsyncDataLoader::iter` runs on a tokio runtime |

---

## License

MIT — see [LICENSE](LICENSE).
