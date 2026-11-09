# Examples

Small programs that each show one part of `dataloader-rs`, in Rust and in Python. Every example
runs offline: the ones that need files, images or a server create them, and remove them when
they finish. Each one prints what it loaded and exits.

The [README](../README.md) has the two quick starts; this page lists the rest.

## Rust

Run an example from the repository's root with `cargo run --example <name>`, adding the
features it needs with `--features`. All of them are compiled by
`cargo clippy --all-targets --all-features`.

| Example | What it shows | Run |
|---|---|---|
| [`basic.rs`](rust/basic.rs) | An in-memory `Dataset`; a shuffled loader with worker threads and prefetching; a custom collator; the number of batches; stopping an epoch early. | `cargo run --example basic` |
| [`collators.rs`](rust/collators.rs) | The default `VecCollator`, `DefaultCollator` (PyTorch's `default_collate`: a batch of tuples becomes a tuple of columns), and a custom `Collator` that builds a struct. | `cargo run --example collators` |
| [`async_dataset.rs`](rust/async_dataset.rs) | A hand-written `AsyncDataset` whose items each wait 20 ms: 16 of them in flight at once on one runtime thread, with a transform on the loader's thread pool. | `cargo run --example async_dataset --features async` |
| [`http_dataset.rs`](rust/http_dataset.rs) | `HttpDataset` with a URL template and columns of values, an `Authorization` header, retries of `503` answers, and a `404` that fails only its own batch. Uses a small local server. | `cargo run --example http_dataset --features http` |
| [`files_and_objects.rs`](rust/files_and_objects.rs) | `FileDataset` with a path template, retries and a time limit; `ObjectStoreDataset` over an in-memory store, by key template and by byte ranges. | `cargo run --example files_and_objects --features object-store` |
| [`image_ops.rs`](rust/image_ops.rs) | A chain of image operations (`Decode`, `ToRgb`, a closure of our own, `PadTo`, `Resize`, `HFlip`, `ToFloat`, `Normalize`, `ToChw`) and `ImageCollator` with reused buffers; a corrupt image fails only its batch. | `cargo run --example image_ops --features image` |
| [`tiles_by_coordinates.rs`](rust/tiles_by_coordinates.rs) | Tiles of a large image fetched over HTTP by `(x, y)`, decoded, and batched as `(images, coordinates)` by a custom collator around `ImageCollator`. Uses a small local tile server. | `cargo run --example tiles_by_coordinates --features http,image` |
| [`stress_safety.rs`](rust/stress_safety.rs) | A stress test: iterators dropped early, many times, with workers, a shallow prefetch and item-level parallelism; the loader never hangs and still runs full epochs. | `cargo run --release --example stress_safety` |

[`local_server.rs`](rust/local_server.rs) is the minimal HTTP server (standard library only)
that the two HTTP examples start; it is not part of the library.

## Python

Build the package into the virtual environment once, then run any example with Python:

```bash
uv run maturin develop --release
uv run python examples/python/basic.py
```

Some examples need NumPy, aiohttp, torch or torchvision, as listed below. `tests/test_examples.py`
runs every example, and skips those whose dependencies are not installed.

| Example | What it shows | Needs |
|---|---|---|
| [`basic.py`](python/basic.py) | A `PyDataset`; `shuffle`, `seed`, `num_workers`, `prefetch_depth`, `drop_last`; `len(loader)`; batches in order; stopping an epoch early. | — |
| [`collate_fn.py`](python/collate_fn.py) | Batches as lists, then as a dict of NumPy arrays made by `collate_fn`; a dataset that fetches whole batches with `__getitems__`. | NumPy |
| [`async_dataset.py`](python/async_dataset.py) | A `PyAsyncDataset` with `async def __getitem__` using an aiohttp session, created on the loader's event loop and closed in `aclose()`; 16 requests in flight. | aiohttp |
| [`http_dataset.py`](python/http_dataset.py) | `HttpDataset.from_template` with columns, a header, retries of `503` answers; a `404` raised for its own batch only. | — |
| [`files_and_objects.py`](python/files_and_objects.py) | `FileDataset.from_template` with a time limit and retries; `ObjectStoreDataset` over a local directory served by HTTP, standing in for a bucket. | — |
| [`tiles_by_coordinates.py`](python/tiles_by_coordinates.py) | Tiles of a large image fetched by `(x, y)` with a URL template, decoded by `transforms.Compose` in Rust, batched as `(ImageBatch, coordinates)` by `ImageCollator`; the batch viewed by NumPy without a copy. | NumPy |
| [`rust_transforms.py`](python/rust_transforms.py) | Every step in Rust: `FileDataset`, `transforms.Compose([Decode, ToRgb, PadTo, ToFloat, Normalize, ToChw])`, `ImageCollator`; `torch.from_dlpack(batch)` without a copy. | NumPy, torch |
| [`mixed_rust_torch.py`](python/mixed_rust_torch.py) | Rust and torch operations in one transform: Rust decodes, a torch operation rotates the image, `Image.from_array` brings it back to Rust, Rust normalises and batches. Batches are `(images, coordinates)`. | NumPy, torch |
| [`torch_transforms.py`](python/torch_transforms.py) | Files read in Rust, then plain torchvision transforms in a Python `transform`, and `collate_fn=torch.stack`; `torch.set_num_threads(1)` to avoid oversubscription. | NumPy, torch, torchvision |

Two helper modules are shared by the examples: [`tiles.py`](python/tiles.py) writes small PNG
tiles with NumPy and the standard library, and [`local_server.py`](python/local_server.py) runs a
minimal HTTP server on a background thread.

The module guides under [`src/`](../src/README.md) and the [Python guide](../python/README.md)
have more short examples, each focused on one API.
