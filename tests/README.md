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

```bash
cargo test                           # unit + integration + doc tests
cargo test --features ndarray,python
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
| `py_dataloader_test_utils.py` | shared datasets (`ListDataset`, `ToyDataset`) and helpers |

```bash
uv sync --group dev
uv run maturin develop --features python
uv run pytest                       # tests/; benchmark cases in bench/ are deselected
uv run --python 3.14t python -X gil=0 -m pytest   # free-threaded, as CI
```
