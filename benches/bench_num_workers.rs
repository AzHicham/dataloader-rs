//! Benchmark: batch-level parallelism (`num_workers`), calibrated for a 4-vCPU runner.
//!
//! FIXED:
//!   dataset        = HeavyCpuDs(256)  — CPU-bound items (≈ 1 ms each)
//!   batch_size     = 16               — 16 batches per epoch
//!   prefetch_depth = 4
//!
//! SWEPT:
//!   num_workers in [0, 1, 2, 4]  (0 = batches processed in `next()`)
//!
//! Question: how close to linear does throughput scale with batch workers?
//! Throughput unit: Elements(256) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 256;
const BS: usize = 16;

fn bench_num_workers(c: &mut Criterion) {
    let mut group = c.benchmark_group("num_workers");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    for &workers in &[0usize, 1, 2, 4] {
        group.bench_with_input(BenchmarkId::new("workers", workers), &workers, |b, &w| {
            let mut loader = DataLoader::builder(HeavyCpuDs(N))
                .batch_size(BS)
                .num_workers(w)
                .prefetch_depth(4)
                .collator(CatCollator)
                .build();
            b.iter(|| {
                for batch in loader.iter() {
                    std::hint::black_box(batch.unwrap());
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_num_workers);
criterion_main!(benches);
