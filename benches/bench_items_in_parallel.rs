//! Benchmark: item-level parallelism (`items_in_parallel`).
//!
//! FIXED:
//!   dataset        = HeavyCpuDs(128)  — CPU-bound items
//!   batch_size     = 64               — 2 batches: little batch-level parallelism to use
//!   num_workers    = 2
//!   prefetch_depth = 8
//!
//! SWEPT:
//!   items_in_parallel in [1, 2, 4, 8]  (1 = one get_batch call per batch)
//!   Also: num_workers=4 with items_in_parallel=4
//!
//! Throughput unit: Elements(128) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 128;
const BS: usize = 64;
const WORKERS: usize = 2;
const PREFETCH: usize = 8;

fn bench_items_in_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("items_in_parallel");
    group.throughput(Throughput::Elements(N as u64));

    for &items in &[1usize, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("items", items), &items, |b, &items| {
            // Build ONCE outside b.iter()
            let mut loader = DataLoader::builder(HeavyCpuDs(N))
                .batch_size(BS)
                .num_workers(WORKERS)
                .items_in_parallel(items)
                .prefetch_depth(PREFETCH)
                .collator(CatCollator)
                .build();

            b.iter(|| {
                for batch in loader.iter() {
                    std::hint::black_box(batch.unwrap());
                }
            });
        });
    }

    // Both axes at once.
    group.bench_function("workers4_items4", |b| {
        let mut loader = DataLoader::builder(HeavyCpuDs(N))
            .batch_size(BS)
            .num_workers(4)
            .items_in_parallel(4)
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

criterion_group!(benches, bench_items_in_parallel);
criterion_main!(benches);
