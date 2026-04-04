//! Benchmark: per-batch amortization (`batch_size`).
//!
//! FIXED:
//!   dataset = InMemoryDs(4096)  — per-item cost is trivial; batch overhead
//!                                  and collation amortization are visible
//!
//! SWEPT:
//!   batch_size in [1, 8, 32, 128, 512, 1024, 4096]
//!
//! Two scenarios per batch size:
//!   a. sequential: num_workers=0 — shows per-batch amortization on main thread
//!   b. parallel:   num_workers=4, prefetch_depth=16 — amortization with workers
//!
//! Groups:
//!   "batch_size/sequential"
//!   "batch_size/parallel"
//!
//! Throughput unit: Elements(4096) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 4096;

fn bench_batch_size_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size/sequential");
    group.throughput(Throughput::Elements(N as u64));

    for &bs in &[1usize, 8, 32, 128, 512, 1024, 4096] {
        group.bench_with_input(BenchmarkId::new("bs", bs), &bs, |b, &batch_size| {
            // Build ONCE outside b.iter()
            let mut loader = DataLoader::builder(InMemoryDs(N))
                .batch_size(batch_size)
                .num_workers(0)
                .collator(SumCollator)
                .build();

            b.iter(|| {
                let total: u64 = loader.iter().map(|b| black_box(b.unwrap())).sum();
                black_box(total);
            });
        });
    }

    group.finish();
}

fn bench_batch_size_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size/parallel");
    group.throughput(Throughput::Elements(N as u64));

    for &bs in &[1usize, 8, 32, 128, 512, 1024, 4096] {
        group.bench_with_input(BenchmarkId::new("bs", bs), &bs, |b, &batch_size| {
            // Build ONCE outside b.iter()
            let mut loader = DataLoader::builder(InMemoryDs(N))
                .batch_size(batch_size)
                .num_workers(4)
                .prefetch_depth(16)
                .collator(SumCollator)
                .build();

            b.iter(|| {
                let total: u64 = loader.iter().map(|b| black_box(b.unwrap())).sum();
                black_box(total);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_size_sequential,
    bench_batch_size_parallel
);
criterion_main!(benches);
