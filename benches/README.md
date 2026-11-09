# `benches/`: Rust benchmarks (Criterion)

Each suite answers one question about the loader, without Python. One Criterion iteration is a
whole epoch, iterator creation included; the loader is built once, outside the timed loop. The
suites are sized for a 4-vCPU machine, the CI runner: at most 4 threads, except where waiting on
the network is the point. [Reference results](#reference-results) below; the comparison with
PyTorch is in [BENCH.md](../BENCH.md).

| Suite | Question | Dataset | Swept | Fixed |
|---|---|---|---|---|
| `bench_batch_size` | the loader's own cost per batch | `InMemoryDs(4096)` | `batch_size` 1 … 4096 | sequential; 4 workers, depth 16 |
| `bench_num_workers` | scaling with batch-level threads | `HeavyCpuDs(256)` | `num_workers` 0, 1, 2, 4 | `batch_size` 16, depth 4 |
| `bench_item_concurrency` | item- against batch-level parallelism at equal threads | `HeavyCpuDs(128)` | (workers, items) (1, 1), (1, 2), (1, 4), (2, 2), (4, 1) | `batch_size` 64: 2 batches |
| `bench_prefetch_depth` | does loading overlap the consumer's work? | `HeavyCpuDs(64)` | sequential; `prefetch_depth` 1, 2, 4, 8 | `batch_size` 4, 2 workers, a consumer step of ≈ 0.22 ms per batch |
| `bench_first_batch` | latency to the first batch | `HeavyCpuDs(32)` | (workers, items) (0, 1), (1, 1), (0, 2), (0, 4), (1, 4) | one batch of 32 |
| `bench_sampler` | sampler cost | `InMemoryDs(N)` | N 1k, 10k, 100k; sequential vs random | `batch_size` 128, no workers |
| `bench_real_world` | real per-item work | 4 096 files of 16 KB; 2 ms blocking latency | `num_workers` 0, 1, 2, 4; 4, 16, 64 requests in flight | `batch_size` 32; glibc's malloc thresholds pinned |
| `bench_async` | futures (`builder_async` + `HttpDataset`) against threads (sync loader, one blocking request per thread) at equal requests in flight; needs `--features http` | 512 GETs on a local keep-alive server, 64 KiB after 10 ms | (workers, items) (1, 64), (2, 32), (4, 16), (8, 1) | `batch_size` 64 |
| `bench_image` | the image transforms and `ImageCollator`, alone and through a loader; with and without reused batch buffers; needs `--features image` | 512 in-memory 224 × 224 RGB PNGs (every eighth 224 × 150): decode, RGB, `PadTo(224)`, `ToFloat`, `Normalize`, CHW | one item; collation NCHW / NHWC, buffers 0 / 4; loader (workers, items) (1, 4), (4, 1), buffers 0 / 8 | `batch_size` 64, `f32` |

`common.rs` holds the shared datasets, collators and consumer work:

| Item | What |
|---|---|
| `InMemoryDs` | returns the index: the loader's own cost only |
| `HeavyCpuDs` | `spin(ITEM_WORK)` per item, ≈ 110 µs, 256 bytes out |
| `spin(n)` | `n` steps of a xorshift-multiply chain, ≈ 1.1 ns each; unlike an LCG, the compiler can neither shorten nor vectorise it |
| `SumCollator`, `CatCollator` | sum `u64`s; concatenate byte vectors |

## Reference results

Criterion estimates from one run on an Intel Core i9-13900H laptop pinned to 4 performance cores
(`bench_async` on 12 hardware threads, `bench_image` on 8):

- **Loader overhead**: 27 ns per batch without workers, 0.7 to 1 G items per second from
  `batch_size` 512; with workers, about 0.5 µs of scheduling per batch.
- **CPU-bound items**: 1.95× with 2 workers, 3.1× with 4. With only 2 batches per epoch,
  `item_concurrency` 4 reaches 3.4× where 4 batch workers stop at 2.0×; it also cuts the wait for
  the first batch 3.7×.
- **Overlap**: `prefetch_depth = 1` already hides loading behind the consumer's step.
- **Sampler**: 2.4 ns per index in order, 3.7 ns shuffled.
- **Waiting items**: throughput follows the requests in flight, 12.6× from 4 to 64. Futures and
  blocking threads cost about the same in Rust at 64 in flight (futures 1 % slower to 19 % faster).
- **Images**: 506 µs per tile through the chain, 314 µs of it decoding. Collating 64 tiles takes
  13.8 ms into fresh memory and 3.6 ms into a reused buffer, so `reuse_buffers` makes an epoch
  19 to 33 % faster.

## Running

```bash
taskset -c 0,2,4,6 cargo bench --features http,image  # every suite, pinned to 4 cores like the runner
cargo bench --bench bench_prefetch_depth  # one suite
cargo bench -- depth/4                    # one case
```

Criterion is built with its minimum features: results go to the console (and `target/criterion`
for its comparisons); HTML reports need its `html_reports` and `plotters` features. To compare
two versions, delete `target/criterion` between builds and run the two builds alternately in one
session (A, B, A, B, …): the machine drifts more between sessions than between interleaved runs.

## In CI

`.github/workflows/bench.yml` runs every suite on pull requests and pushes to `dev` (when `src/`,
`benches/`, `bench/` or `Cargo.*` change), with `--output-format bencher`, and stores the results
per commit on the [dashboard](https://azhicham.github.io/dataloader-rs/benchmarks/rust/). Adding a
suite: a `[[bench]]` entry with `harness = false` in `Cargo.toml`, and its name in the workflow's
`matrix.bench` list; a suite with `required-features` (`bench_async`, `bench_image`) goes in `matrix.include`
with its `features` instead, since Cargo skips it silently without them.
