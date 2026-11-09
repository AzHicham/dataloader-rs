# `tests/`: integration tests

There are two suites: Rust integration tests against the public API, and pytest tests against
the Python package. Unit tests live next to the code, in the `#[cfg(test)]` modules of `src/`.
The Rust examples of `examples/rust/` are compiled by `cargo clippy --all-targets`.

## Rust

| File | Covers | Feature |
|---|---|---|
| `loader_sync.rs` | the public API end to end: sequential, parallel and item-level paths give the same items; batch sizes and `drop_last`; random and distributed samplers; collators; transforms; errors per batch; early drop; reuse across epochs | |
| `loader_async.rs` | `builder_async`: order, futures in flight, transforms, errors and panics per batch, early drop, the runtime settings | `async` |
| `integration_collate_default.rs` | `DefaultCollator` on nested structures through a loader | |
| `integration_collate_ndarray.rs` | `DefaultCollator` stacking `ndarray::Array` | `ndarray` |
| `integration_collate_torch.rs` | `DefaultCollator` stacking `tch::Tensor`, `pin_memory` | `torch-rs` |
| `image.rs` | the image transforms against reference files made with Pillow, NumPy and torchvision (`fixtures/image/make_fixtures.py`); hostile input (huge declared sizes, truncated, corrupted or random data); `ImageCollator` through a loader | `image` |

```bash
cargo test                           # unit + integration + doc tests
cargo test --features ndarray,python
cargo test --features image --test image   # regenerate the golden files: tests/fixtures/image/make_fixtures.py
cargo test --all-features            # as CI does; torch-rs downloads libtorch
```

## Python

| File | Covers |
|---|---|
| `test_dataloader_sequential.py` | `num_workers=0`: order, `drop_last`, `shuffle` / `seed`, edge sizes |
| `test_dataloader_parallel.py` | `num_workers > 0`: coverage, order, `drop_last`, reuse, early drop, `collate_fn`, large `prefetch_depth` |
| `test_dataloader_iterator_semantics.py` | the iterator protocol: `len(iterator)`, exhaustion, several iterators of one loader |
| `test_dataloader_validation.py` | argument validation, `len(loader)` with samplers and `drop_last`, what `collate_fn` receives |
| `test_dataloader_errors.py` | exceptions from `__getitem__`, `collate_fn`, `__len__` and samplers keep their type and do not end an epoch silently |
| `test_dataloader_getitems.py` | batched fetching with `__getitems__` |
| `test_dataloader_safety.py` | early drop, garbage collection and cancellation under load: no hang, no deadlock |
| `test_dataloader_integration.py` | end-to-end scenarios: batches without `collate_fn`, `drop_last`, `collate_fn` sequential and parallel, a Python sampler's order |
| `test_dataset_async.py` | `PyAsyncDataset` and the Rust datasets (`HttpDataset`, `FileDataset`, `ObjectStoreDataset`): base-class checks, order, cancellation, `aclose`, errors, templates, retry settings |
| `test_transform.py` | `transform=f`, called as `f(index, item)`, on every kind of dataset |
| `test_image_transforms.py` | `dataloader_rs.transforms` and `ImageCollator`: reference files, errors, every kind of dataset, `(Image, coords)` tuples, zero-copy numpy and DLPack views, buffer reuse; `Image.from_array` (round trips, CHW and strided input, refused types, shapes and sizes, a loader that goes from Rust to torch and back) |
| `test_examples.py` | every Python example of `examples/python/` runs to the end and prints what it should (an example is skipped when NumPy, aiohttp, torch or torchvision, as it needs, is missing) |
| `py_dataloader_test_utils.py` | shared datasets (`ListDataset`, `ToyDataset`) and helpers |

```bash
uv sync --group dev
uv run maturin develop --features python
uv run pytest                       # tests/; benchmark cases in bench/ are deselected
uv run --python 3.13 pytest        # with the GIL (default: free-threaded 3.14t, as CI)
```
