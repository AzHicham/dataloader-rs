# `examples/`

| Example | What it shows | Run |
|---|---|---|
| `basic.rs` | a minimal dataset; random batches with `VecCollator`, sequential batches with a summing collator | `cargo run --example basic` |
| `stress_safety.rs` | drop safety under stress: iterators dropped early with workers, with a shallow prefetch, with item-level parallelism, repeatedly | `cargo run --release --example stress_safety` |

The Rust and Python quick starts are in the [README](../README.md); every module guide in
[`src/`](../src/README.md) has compiled examples.
