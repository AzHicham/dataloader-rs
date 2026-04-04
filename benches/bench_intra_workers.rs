//! Benchmark: item-level rayon parallelism (`intra_workers`).
//!
//! FIXED:
//!   dataset    = HeavyCpuDs(128)  — CPU-bound, benefits from intra-batch rayon
//!   batch_size = 64               — 2 batches total; large batch = room for rayon
//!   inter_workers  = 2            — enough pipeline depth
//!   prefetch_depth = 8            — enough pipeline depth
//!
//! SWEPT:
//!   intra_workers in [0, 1, 2, 4, 8]
//!   Also: (inter=4, intra=4) — combined parallelism case
//!
//! Throughput unit: Elements(128) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dataloader_rs::DataLoader;

const N: usize = 128;
const BS: usize = 64;
const INTER: usize = 2;
const PREFETCH: usize = 8;

fn bench_intra_workers(c: &mut Criterion) {
    let mut group = c.benchmark_group("intra_workers");
    group.throughput(Throughput::Elements(N as u64));

    // Sweep intra_workers with fixed inter_workers=2
    for &intra in &[0usize, 1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("intra_workers", intra),
            &intra,
            |b, &iw| {
                // Build ONCE outside b.iter()
                let mut loader = DataLoader::builder(HeavyCpuDs(N))
                    .batch_size(BS)
                    .num_workers(INTER)
                    .intra_workers(iw)
                    .prefetch_depth(PREFETCH)
                    .collator(CatCollator)
                    .build();

                b.iter(|| {
                    for batch in loader.iter() {
                        black_box(batch.unwrap());
                    }
                });
            },
        );
    }

    // Combined: inter=4 + intra=4 — both axes at once
    group.bench_function("inter4_intra4", |b| {
        let mut loader = DataLoader::builder(HeavyCpuDs(N))
            .batch_size(BS)
            .num_workers(4)
            .intra_workers(4)
            .prefetch_depth(PREFETCH)
            .collator(CatCollator)
            .build();

        b.iter(|| {
            for batch in loader.iter() {
                black_box(batch.unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_intra_workers);
criterion_main!(benches);
