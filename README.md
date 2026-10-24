# dataloader-rs

[![CI](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/AzHicham/dataloader-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/dataloader-rs)](https://crates.io/crates/dataloader-rs)
[![docs.rs](https://img.shields.io/docsrs/dataloader-rs)](https://docs.rs/dataloader-rs)
[![PyPI](https://img.shields.io/pypi/v/dataloader-rs)](https://pypi.org/project/dataloader-rs/)
[![MSRV](https://img.shields.io/badge/rustc-1.91%2B-orange.svg)](#minimum-supported-rust-version)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PyTorch's `DataLoader` model — dataset, sampler, collator, loader — as a typed Rust library,
with a Python binding that runs the same pipeline on threads instead of worker processes.

- **Familiar model, typed end to end**: `DataLoader<Dataset, Sampler, Collator>` is generic and
  monomorphized; batches come out in epoch order as `Result<Batch>`.
- **Threads, not processes**: workers share the dataset through `Arc`, so samples are handed
  over by reference instead of being pickled across processes. On array samples the Python
  binding is about **7× `torch.utils.data.DataLoader`** ([BENCH.md](BENCH.md)).
- **One thread pool, two levels of parallelism**: `num_workers` batches in flight and, with
  `parallel_items`, the items of each batch fetched in parallel — on one rayon pool built with
  the loader and reused by every epoch, so the two never oversubscribe the CPU.
- **Bounded, ordered prefetch**: at most `num_workers + prefetch_depth` batches are in flight or
  waiting, even behind one slow batch; they come out in epoch order.
- **Failures surface**: a dataset or collator error is returned for its batch, a panic is
  re-raised on the consumer, and Python exceptions keep their type.
- **Python binding**: `PyDataloader` mirrors the PyTorch arguments (`batch_size`, `shuffle`,
  `sampler`, `collate_fn`, `drop_last`, `seed`, `__getitems__`); wheels for free-threaded
  CPython 3.14t and 3.15t on Linux (x86_64, arm64) and macOS (arm64).
- **No unsafe code**: `#![forbid(unsafe_code)]`.

## Contents

- [Quick start: Rust](#quick-start-rust)
- [Quick start: Python](#quick-start-python)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [Feature flags](#feature-flags)
- [Performance](#performance)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Building and testing](#building-and-testing)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [License](#license)

## Quick start: Rust

```toml
[dependencies]
dataloader-rs = "0.1"
```

```rust
use dataloader_rs::{DataLoader, Dataset, RandomSampler, error::Result};

/// One sample per file; `get` is called from worker threads.
struct Files(Vec<std::path::PathBuf>);

impl Dataset for Files {
    type Item = Vec<u8>;

    fn get(&self, index: usize) -> Result<Vec<u8>> {
        Ok(std::fs::read(&self.0[index])?)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn main() -> Result<()> {
    let paths = std::fs::read_dir("data")?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;

    let mut loader = DataLoader::builder(Files(paths))
        .batch_size(32)
        .sampler(RandomSampler::new(42)) // reproducible shuffle
        .num_workers(4)                  // 4 batches in flight
        .prefetch_depth(8)               // run up to 8 batches ahead
        .drop_last(true)
        .build();

    for epoch in 0..3 {
        for batch in &mut loader {
            let batch: Vec<Vec<u8>> = batch?; // VecCollator: a Vec of samples
            println!("epoch {epoch}: {} samples", batch.len());
        }
    }
    Ok(())
}
```

`cargo run --example basic` runs a complete example.

## Quick start: Python

```bash
pip install dataloader-rs
```

```python
from dataloader_rs import PyDataloader, PyDataset


class Files(PyDataset):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> bytes:
        with open(self.paths[index], "rb") as f:
            return f.read()


loader = PyDataloader(Files(paths), batch_size=32, shuffle=True, seed=0, num_workers=4)
for batch in loader:  # a list of 32 `bytes`, or whatever collate_fn returns
    ...
```

The full Python guide — arguments, samplers, `collate_fn`, `__getitems__`, errors, threads and
the GIL — is [python/README.md](python/README.md).

## Concepts

| Piece | Trait / type | Provided | Guide |
|---|---|---|---|
| Dataset | `Dataset`: `get(index)`, `len()`, optional `get_batch(indices)` | — (yours) | [src/README.md](src/README.md) |
| Sampler | `Sampler`: the index order of one epoch | `SequentialSampler`, `RandomSampler`, `DistributedSampler` | [src/sampler](src/sampler/README.md) |
| Collator | `Collator<Item>`: merge a batch's samples | `VecCollator` (default), `DefaultCollator`, `TorchPinnedCollator` | [src/collator](src/collator/README.md) |
| Loader | `DataLoader` / `DataLoaderBuilder` / `DataLoaderIter` | — | [src/loader](src/loader/README.md) |

One epoch: the sampler produces indices, `BatchSampler` cuts them into batches of `batch_size`,
each batch is fetched (`Dataset::get_batch`, or `get` per item in parallel) and collated, and the
iterator yields the batches in order. How the threads, the bounded prefetch and the reorder
buffer fit together is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration

Every setting is a `DataLoaderBuilder` method; the Python argument of the same name maps to it.

| Builder method | Default | Effect |
|---|---|---|
| `batch_size(n)` | `1` | samples per batch |
| `drop_last(bool)` | `false` | drop a final batch smaller than `batch_size` |
| `sampler(s)` | `SequentialSampler` | index order of each epoch |
| `collator(c)` | `VecCollator` | how a batch's samples are merged |
| `num_workers(n)` | `0` | batches processed concurrently on the loader's pool; `0` processes batches inside `next()` |
| `parallel_items(bool)` | `false` | also fetch the items of each batch in parallel, on the same pool |
| `num_threads(t)` | `num_workers`, or CPUs with `parallel_items` | the pool's size, never below `num_workers` |
| `prefetch_depth(n)` | `1` | how far batches may run ahead: at most `num_workers + n` in flight or waiting |
| `pin_memory(bool)` | — | pin `tch::Tensor` batches in host memory (feature `torch-rs`) |

`intra_workers(n)` still works, deprecated: it is `parallel_items(n > 0)` plus `num_threads(n)`.

Rules of thumb: `num_workers = 0` is the fastest path for cheap samples (no thread hand-off);
use workers once a batch costs more than a few tens of microseconds (a nearly free batch pays a
fixed scheduling cost); for blocking I/O, more threads than cores; `prefetch_depth` 2–8 is
enough, beyond that it only holds memory ([BENCH.md](BENCH.md#4-results-rust)).

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `ndarray` | off | `DefaultCollator` for `ndarray::Array` (stacked on a new axis 0) |
| `torch-rs` | off | `DefaultCollator` for `tch::Tensor` (stacked), `TorchPinnedCollator`, `pin_memory` |
| `python` | off | the PyO3 module behind the `dataloader-rs` Python package |

## Performance

Measured on an Intel Core i9-13900H (20 threads); method, every table and the caveats are in
[BENCH.md](BENCH.md). Continuous results per commit: dashboards for [Rust](https://azhicham.github.io/dataloader-rs/benchmarks/rust/) and [Python 3.14t](https://azhicham.github.io/dataloader-rs/benchmarks/python-3.14t/).

| Python workload, against `torch.utils.data.DataLoader` | Result |
|---|---|
| Array samples: read + normalise a file, augment an image (4 workers) | **≈ 7× faster** |
| Sequential path (`num_workers=0`), any batch size | **1.5–61× faster** |
| Parallel, dataset releasing the GIL (hashlib, NumPy, file I/O) | 2–6 % faster |
| Pure-Python CPU work holding the GIL, regular CPython | torch 1.6× faster (processes); use 3.14t |

## Documentation

| Document | Content |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | execution paths, threads, backpressure, ordering, errors, panics, the Python binding |
| [BENCH.md](BENCH.md) | what is measured and how, results, comparison with PyTorch, caveats, how to rerun |
| [python/README.md](python/README.md) | the Python package (also its PyPI page) |
| [src/README.md](src/README.md) | code map, and the `Dataset` contract |
| [src/loader](src/loader/README.md), [src/sampler](src/sampler/README.md), [src/collator](src/collator/README.md), [src/python](src/python/README.md) | one guide per module (also on docs.rs) |
| [bench/](bench/README.md), [benches/](benches/README.md), [tests/](tests/README.md), [examples/](examples/README.md) | what each script, suite and test file covers |
| [CHANGELOG.md](CHANGELOG.md) | release notes (generated by semantic-release) |

## Project layout

| Path | What |
|---|---|
| [`src/`](src/README.md) | the crate: `dataset`, `sampler`, `collator`, `loader`, `python` |
| [`dataloader_rs/`](dataloader_rs/__init__.pyi) | Python package: re-exports and type stubs |
| [`python/`](python/README.md) | Python package README (PyPI) |
| [`tests/`](tests/README.md) | Rust integration tests and the pytest suite |
| [`benches/`](benches/README.md) | Criterion benchmarks (Rust) |
| [`bench/`](bench/README.md) | Python benchmarks: pytest-benchmark guards and comparisons with PyTorch |
| [`examples/`](examples/README.md) | runnable Rust examples |

## Building and testing

Prerequisites: Rust ≥ 1.91, Python ≥ 3.13, [uv](https://docs.astral.sh/uv/).

```bash
# Rust
cargo build
cargo test --features ndarray,python          # add torch-rs to test tch (downloads libtorch)

# Python
uv sync --group dev                            # maturin, pytest, numpy, torch
uv run maturin develop --features python       # build the extension into the venv
uv run pytest                                  # tests/ (benchmarks are deselected)

# Lint, as CI does
CI=1 prek run --all-files                      # ruff, ty, cargo fmt, clippy -D warnings
```

Benchmarks: [BENCH.md, section 9](BENCH.md#9-running-it-again).

Commits follow [Conventional Commits](https://www.conventionalcommits.org/); releases, the
changelog, the crate and the wheels are published by semantic-release from `main`.

## Minimum supported Rust version

Rust **1.91** (`rust-version` in `Cargo.toml`), checked in CI by the `Rust tests (MSRV)` job.

## License

MIT — see [LICENSE](LICENSE).
