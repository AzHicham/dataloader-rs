//! Benchmark: sampler overhead (sequential vs random).
//!
//! FIXED:
//!   batch_size   = 128
//!   num_workers  = 0   — direct path, pure sampler measurement; workers would
//!                         obscure the index-generation cost
//!
//! SWEPT:
//!   dataset size N in [1_000, 10_000, 100_000]
//!   sampler type: sequential (SequentialSampler) vs random (RandomSampler)
//!
//! Throughput unit: Elements(N) per iteration (full epoch).
//! Group: "sampler"

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::{DataLoader, sampler::RandomSampler};

const BS: usize = 128;

fn bench_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler");

    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));

        // Sequential sampler (default)
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, &size| {
            // Build ONCE outside b.iter()
            let mut loader = DataLoader::builder(InMemoryDs(size))
                .batch_size(BS)
                .num_workers(0)
                .collator(SumCollator)
                .build();

            b.iter(|| {
                let total: u64 = loader
                    .iter()
                    .map(|b| std::hint::black_box(b.unwrap()))
                    .sum();
                std::hint::black_box(total);
            });
        });

        // Random sampler (Fisher-Yates shuffle)
        group.bench_with_input(BenchmarkId::new("random", n), &n, |b, &size| {
            // Build ONCE outside b.iter()
            let mut loader = DataLoader::builder(InMemoryDs(size))
                .batch_size(BS)
                .num_workers(0)
                .sampler(RandomSampler::new(42))
                .collator(SumCollator)
                .build();

            b.iter(|| {
                let total: u64 = loader
                    .iter()
                    .map(|b| std::hint::black_box(b.unwrap()))
                    .sum();
                std::hint::black_box(total);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sampler);
criterion_main!(benches);
