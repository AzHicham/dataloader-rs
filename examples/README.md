# `examples/`

## Rust

| Example | What it shows | Run |
|---|---|---|
| `basic.rs` | a minimal dataset; random batches with `VecCollator`, sequential batches with a summing collator | `cargo run --example basic` |
| `stress_safety.rs` | drop safety under stress: iterators dropped early with workers, with a shallow prefetch, with item-level parallelism, repeatedly | `cargo run --release --example stress_safety` |

## Python

[`python/`](python/): small offline examples of image loading. Each writes its own PNG tiles (a
3 × 3 grid of a slide, the last column narrower) into a temporary directory and reads them with
`FileDataset`; `HttpDataset.from_template` reads the same tiles over HTTP.

| Example | What it shows | Needs |
|---|---|---|
| `rust_transforms.py` | all in Rust, no interpreter per item: `transforms.Compose([Decode, ToRgb, PadTo, ToFloat, Normalize, ToChw])` + `ImageCollator`, `torch.from_dlpack(batch)` without a copy | numpy, torch |
| `mixed_rust_torch.py` | Rust decode → a torch op on `torch.from_dlpack(image)` → `Image.from_array(t, layout="chw")` → Rust `Normalize` → `ImageCollator`; items `(Image, coords[i])`, batches `(batch, coords)`; the GIL and copies of each step | numpy, torch |
| `torch_transforms.py` | Rust fetch, plain torchvision transforms in `transform=`, `collate_fn=torch.stack`; `torch.set_num_threads(1)` against oversubscription | numpy, torch, torchvision |

```bash
uv run maturin develop
uv run python examples/python/rust_transforms.py
```

`tests/test_examples.py` runs them (skipped without torch or torchvision).

The Rust and Python quick starts are in the [README](../README.md); every module guide in
[`src/`](../src/README.md) has compiled examples.
