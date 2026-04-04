//! Benchmark: cost of cancellation / shutdown (`early_drop`).
//!
//! FIXED:
//!   dataset    = InMemoryDs(1000) — fast items so early-drop cost dominates;
//!                                    slow items would make shutdown latency
//!                                    negligible relative to in-flight work
//!   batch_size = 10               — 100 batches total
//!
//! SWEPT:
//!   num_workers in [0, 1, 2, 4]
//!
//! Two cases per worker count:
//!   a. consume 1 batch then drop  — exercises rx-drop + join after partial epoch
//!   b. drop immediately (0 batches consumed) — exercises clean shutdown before
//!                                               any batch is consumed
//!
//! Loaders are built ONCE outside b.iter() — we measure shutdown cost,
//! not construction cost.
//!
//! Group: "early_drop"

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dataloader_rs::DataLoader;

const N: usize = 1000;
const BS: usize = 10;

fn bench_early_drop(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_drop");

    for &workers in &[0usize, 1, 2, 4] {
        // Case a: consume 1 batch, then drop
        group.bench_with_input(
            BenchmarkId::new("consume_1_then_drop", workers),
            &workers,
            |b, &w| {
                let mut loader = DataLoader::builder(InMemoryDs(N))
                    .batch_size(BS)
                    .num_workers(w)
                    .prefetch_depth(4)
                    .build();

                b.iter(|| {
                    let mut iter = loader.iter();
                    // Consume exactly one batch, then drop the iterator mid-epoch
                    black_box(iter.next().unwrap().unwrap());
                    drop(iter); // triggers rx-drop + prefetch thread join
                });
            },
        );

        // Case b: drop iterator immediately (0 batches consumed)
        group.bench_with_input(
            BenchmarkId::new("drop_immediately", workers),
            &workers,
            |b, &w| {
                let mut loader = DataLoader::builder(InMemoryDs(N))
                    .batch_size(BS)
                    .num_workers(w)
                    .prefetch_depth(4)
                    .build();

                b.iter(|| {
                    let iter = loader.iter();
                    drop(iter); // clean shutdown before any batch consumed
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_early_drop);
criterion_main!(benches);
