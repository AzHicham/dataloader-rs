# `benches/`: Rust benchmarks (Criterion)

Each suite measures one setting of the loader without Python. One Criterion iteration is a whole
epoch, iterator creation included; the loader is built once, outside the timed loop. Results and
their reading: [BENCH.md, section 4](../BENCH.md#4-results-rust).

| Suite | Dataset | Swept | Fixed | Question |
|---|---|---|---|---|
| `bench_batch_size` | `InMemoryDs(4096)` | `batch_size` 1 … 4096 | sequential; parallel with 4 workers, depth 16 | per-batch overhead and its amortisation |
| `bench_inter_workers` | `HeavyCpuDs(256)` | `num_workers` 0, 1, 2, 4, 8 | `batch_size` 16, depth 32 | scaling with batch-level threads |
| `bench_intra_workers` | `HeavyCpuDs(128)` | `intra_workers` 0, 1, 2, 4, 8; and 4 + 4 | `batch_size` 64, 2 workers, depth 8 | scaling with item-level threads |
| `bench_prefetch_depth` | `LightCpuDs(128)` | `prefetch_depth` 1, 2, 4, 8, 16 | `batch_size` 8, 4 workers | does a deeper prefetch help? |
| `bench_sampler` | `InMemoryDs(N)` | N 1k, 10k, 100k; sequential vs random | `batch_size` 128, no workers | sampler cost |
| `bench_real_world` | 64 KB files; 2 ms blocking latency | `num_workers` 0, 2, 4, 8; threads 4, 16, 64 | `batch_size` 32 | real per-sample work: I/O fan-out, network latency |

`common.rs` holds the shared datasets and collators:

| Item | What |
|---|---|
| `InMemoryDs` | returns the index: the loader's own cost only |
| `LightCpuDs` | 1 000 LCG iterations per item (µs) |
| `HeavyCpuDs` | 50 000 LCG iterations per item (≈ 1 ms), 256 bytes out |
| `SumCollator`, `CatCollator` | sum `u64`s; concatenate byte vectors |

## Running

```bash
cargo bench                              # every suite
cargo bench --bench bench_prefetch_depth # one suite
cargo bench -- depth/4                   # one case
```

Reports: `target/criterion/<group>/report/index.html`. To compare two versions, delete
`target/criterion` between builds and alternate them (BENCH.md, section 9.4).

## In CI

`.github/workflows/bench.yml` runs every suite on pull requests and pushes to `dev` (when `src/`,
`benches/`, `bench/` or `Cargo.*` change), with `--output-format bencher`, and stores the results
per commit on the [dashboard](https://azhicham.github.io/dataloader-rs/benchmarks/rust/). Adding a
suite: a `[[bench]]` entry with `harness = false` in `Cargo.toml`, and its name in the workflow's
`matrix.bench` list.
