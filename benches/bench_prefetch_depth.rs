//! Benchmark: overlapping loading with the consumer's work (`prefetch_depth`).
//!
//! The consumer does work for every batch, like a training step, so a loader
//! that prefetches well hides its loading time behind that work.
//!
//! FIXED:
//!   dataset     = HeavyCpuDs(64)  — CPU-bound items (≈ 110 µs each)
//!   batch_size  = 4               — 16 batches, ≈ 0.45 ms to load each
//!   consumer    = spin(2 × ITEM_WORK) — ≈ 0.22 ms of work per batch
//!   num_workers = 2               — a batch every ≈ 0.22 ms: as fast as the consumer
//!
//! SWEPT:
//!   sequential (num_workers 0: load and consume one after the other, ≈ 0.67 ms per batch)
//!   prefetch_depth in [1, 2, 4, 8]
//!
//! Ideal with full overlap: ≈ 16 × 0.22 ms + one batch to fill the pipeline.
//! Throughput unit: Elements(64) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 64;
const BS: usize = 4;
const CONSUMER_WORK: u64 = 2 * ITEM_WORK;

fn bench_prefetch_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_depth");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function("sequential", |b| {
        let mut loader = DataLoader::builder(HeavyCpuDs(N))
            .batch_size(BS)
            .collator(CatCollator)
            .build();
        b.iter(|| {
            for batch in loader.iter() {
                std::hint::black_box(batch.unwrap());
                spin(CONSUMER_WORK);
            }
        });
    });

    for &depth in &[1usize, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("depth", depth), &depth, |b, &d| {
            let mut loader = DataLoader::builder(HeavyCpuDs(N))
                .batch_size(BS)
                .num_workers(2)
                .prefetch_depth(d)
                .collator(CatCollator)
                .build();
            b.iter(|| {
                for batch in loader.iter() {
                    std::hint::black_box(batch.unwrap());
                    spin(CONSUMER_WORK);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prefetch_depth);
criterion_main!(benches);
