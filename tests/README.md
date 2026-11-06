# `tests/`: integration tests

Two suites: Rust integration tests against the public API, and pytest against the Python
package. Unit tests live next to the code (`#[cfg(test)]` modules in `src/`).

## Rust

| File | Covers | Feature |
|---|---|---|
| `integration.rs` | the public API end to end: sequential, parallel and item-level paths give the same items; batch sizes and `drop_last`; random and distributed samplers; collators; errors per batch; early drop; reuse across epochs | |
| `integration_collate_default.rs` | `DefaultCollator` on nested structures through a loader | |
| `integration_collate_ndarray.rs` | `DefaultCollator` stacking `ndarray::Array` | `ndarray` |
| `integration_collate_torch.rs` | `DefaultCollator` stacking `tch::Tensor`, `pin_memory` | `torch-rs` |
| `integration_collate_into.rs` | `collate_into`: buffers reused (stable pointers), pool bounded under a slow consumer, buffers returned on drop, errors, panics, early break, partial last batch, epochs, workers | |
| `integration_sources.rs` | `MemorySource`, `FileSource` (size cap, missing file, directory, `/dev/zero`), `HttpSource` against a local server (`support/http_server.rs`): URLs and templates, encoding, headers, connection reuse, retries and backoff, `Retry-After`, timeouts, size caps, truncated / endless / garbage responses, redirects, invalid settings, a failed item failing its batch only, an `ImageCollator` fed over HTTP with metadata | `sources` (`image`) |
| `integration_image.rs` | the `image` pipeline against Pillow and torchvision golden fixtures (`fixtures/image/make_fixtures.py`), hostile inputs, a loader with workers reusing pooled buffers, run-time dtypes, a bad sample failing its batch only | `image` |

```bash
cargo test                           # unit + integration + doc tests
cargo test --features ndarray,python,image
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
| `test_image_pipeline.py` | `ImageCollator`: validation, values against the golden fixtures, metadata, partial batches, any thread count, shuffling, zero-copy NumPy / DLPack views, buffer reuse and lifetime, the `max_held_batches` guard, errors |
| `py_dataloader_test_utils.py` | shared datasets (`ListDataset`, `ToyDataset`) and helpers |

```bash
uv sync --group dev
uv run maturin develop --features python
uv run pytest                       # tests/; benchmark cases in bench/ are deselected
uv run --python 3.13 pytest        # with the GIL (default: free-threaded 3.14t, as CI)
```
