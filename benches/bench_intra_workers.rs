//! Benchmark: item-level parallelism (`parallel_items` + `num_threads`).
//!
//! Series ids keep the historical `intra_workers` names: `intra_workers/N`
//! is `parallel_items(true).num_threads(N)`, and `N = 0` is `parallel_items(false)`.
//!
//! FIXED:
//!   dataset    = HeavyCpuDs(128)  — CPU-bound, benefits from intra-batch rayon
//!   batch_size = 64               — 2 batches total; large batch = room for rayon
//!   num_workers    = 2            — enough pipeline depth
//!   prefetch_depth = 8            — enough pipeline depth
//!
//! SWEPT:
//!   num_threads in [0 (items sequential), 1, 2, 4, 8]
//!   Also: num_workers=4 + num_threads=4 — combined parallelism case
//!
//! Throughput unit: Elements(128) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 128;
const BS: usize = 64;
const INTER: usize = 2;
const PREFETCH: usize = 8;

fn bench_intra_workers(c: &mut Criterion) {
    let mut group = c.benchmark_group("intra_workers");
    group.throughput(Throughput::Elements(N as u64));

    // Sweep the pool size with item-level parallelism, num_workers=2
    for &intra in &[0usize, 1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("intra_workers", intra),
            &intra,
            |b, &iw| {
                // Build ONCE outside b.iter()
                let builder = DataLoader::builder(HeavyCpuDs(N))
                    .batch_size(BS)
                    .num_workers(INTER)
                    .parallel_items(iw > 0)
                    .prefetch_depth(PREFETCH)
                    .collator(CatCollator);
                let builder = if iw > 0 {
                    builder.num_threads(iw)
                } else {
                    builder
                };
                let mut loader = builder.build();

                b.iter(|| {
                    for batch in loader.iter() {
                        std::hint::black_box(batch.unwrap());
                    }
                });
            },
        );
    }

    // Combined: num_workers=4 + parallel items on 4 threads — both axes at once
    group.bench_function("inter4_intra4", |b| {
        let mut loader = DataLoader::builder(HeavyCpuDs(N))
            .batch_size(BS)
            .num_workers(4)
            .parallel_items(true)
            .num_threads(4)
            .prefetch_depth(PREFETCH)
            .collator(CatCollator)
            .build();

        b.iter(|| {
            for batch in loader.iter() {
                std::hint::black_box(batch.unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_intra_workers);
criterion_main!(benches);
