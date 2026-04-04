//! Benchmark: inter-batch worker concurrency (`num_workers`).
//!
//! FIXED:
//!   dataset    = HeavyCpuDs(256)   — CPU-bound, benefits from parallelism
//!   batch_size = 16                — 16 batches total
//!   prefetch_depth = 32            — 2× max workers, never starves workers
//!   intra_workers  = 0             — isolate inter effect only
//!
//! SWEPT:
//!   num_workers in [0, 1, 2, 4, 8]
//!
//! Throughput unit: Elements(256) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 256;
const BS: usize = 16;
const PREFETCH: usize = 32;

fn bench_inter_workers(c: &mut Criterion) {
    let mut group = c.benchmark_group("inter_workers");
    group.throughput(Throughput::Elements(N as u64));

    for &workers in &[0usize, 1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("num_workers", workers),
            &workers,
            |b, &w| {
                // Build the loader ONCE outside b.iter() — we measure epoch
                // throughput, not thread-pool construction cost.
                let mut loader = DataLoader::builder(HeavyCpuDs(N))
                    .batch_size(BS)
                    .num_workers(w)
                    .intra_workers(0)
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

    group.finish();
}

criterion_group!(benches, bench_inter_workers);
criterion_main!(benches);
