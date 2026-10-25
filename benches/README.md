# `benches/`: Rust benchmarks (Criterion)

Each suite answers one question about the loader, without Python. One Criterion iteration is a
whole epoch, iterator creation included; the loader is built once, outside the timed loop. The
suites are sized for a 4-vCPU machine, the CI runner: at most 4 threads, except where waiting on
the network is the point. Results and their reading:
[BENCH.md, section 4](../BENCH.md#4-results-rust).

| Suite | Question | Dataset | Swept | Fixed |
|---|---|---|---|---|
| `bench_batch_size` | the loader's own cost per batch | `InMemoryDs(4096)` | `batch_size` 1 … 4096 | sequential; 4 workers, depth 16 |
| `bench_num_workers` | scaling with batch-level threads | `HeavyCpuDs(256)` | `num_workers` 0, 1, 2, 4 | `batch_size` 16, depth 4 |
| `bench_items_in_parallel` | item- against batch-level parallelism at equal threads | `HeavyCpuDs(128)` | (workers, items) (1, 1), (1, 2), (1, 4), (2, 2), (4, 1) | `batch_size` 64: 2 batches |
| `bench_prefetch_depth` | does loading overlap the consumer's work? | `HeavyCpuDs(64)` | sequential; `prefetch_depth` 1, 2, 4, 8 | `batch_size` 4, 2 workers, a consumer step of ≈ 0.22 ms per batch |
| `bench_first_batch` | latency to the first batch | `HeavyCpuDs(32)` | (workers, items) (0, 1), (1, 1), (0, 2), (0, 4), (1, 4) | one batch of 32 |
| `bench_sampler` | sampler cost | `InMemoryDs(N)` | N 1k, 10k, 100k; sequential vs random | `batch_size` 128, no workers |
| `bench_real_world` | real per-item work | 4 096 files of 16 KB; 2 ms blocking latency | `num_workers` 0, 1, 2, 4; 4, 16, 64 requests in flight | `batch_size` 32; glibc's malloc thresholds pinned |

`common.rs` holds the shared datasets, collators and consumer work:

| Item | What |
|---|---|
| `InMemoryDs` | returns the index: the loader's own cost only |
| `HeavyCpuDs` | `spin(ITEM_WORK)` per item, ≈ 110 µs, 256 bytes out |
| `spin(n)` | `n` steps of a xorshift-multiply chain, ≈ 1.1 ns each; unlike an LCG, the compiler can neither shorten nor vectorise it |
| `SumCollator`, `CatCollator` | sum `u64`s; concatenate byte vectors |

## Running

```bash
taskset -c 0,2,4,6 cargo bench            # every suite, pinned to 4 cores like the runner
cargo bench --bench bench_prefetch_depth  # one suite
cargo bench -- depth/4                    # one case
```

Criterion is built with its minimum features: results go to the console (and `target/criterion`
for its comparisons); HTML reports need its `html_reports` and `plotters` features. To compare
two versions, delete `target/criterion` between builds and alternate them ([BENCH.md, section 9.4](../BENCH.md#94-comparing-two-versions)).

## In CI

`.github/workflows/bench.yml` runs every suite on pull requests and pushes to `dev` (when `src/`,
`benches/`, `bench/` or `Cargo.*` change), with `--output-format bencher`, and stores the results
per commit on the [dashboard](https://azhicham.github.io/dataloader-rs/benchmarks/rust/). Adding a
suite: a `[[bench]]` entry with `harness = false` in `Cargo.toml`, and its name in the workflow's
`matrix.bench` list.
