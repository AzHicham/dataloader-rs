//! Benchmark: latency to the first batch.
//!
//! An epoch of exactly one batch, so one iteration is `iter()` until that
//! batch is ready: the time a training loop waits before its first step, and
//! the per-epoch start-up cost of the workers.
//!
//! FIXED:
//!   dataset    = HeavyCpuDs(32)  — CPU-bound items (≈ 110 µs each)
//!   batch_size = 32              — one batch, ≈ 3.5 ms of item work
//!
//! SWEPT (num_workers × items_in_parallel), ≤ 4 threads:
//!   w0_items1  w1_items1  w0_items2  w0_items4  w1_items4
//!
//! Throughput unit: Elements(32) per iteration (one batch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const BS: usize = 32;

fn bench_first_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_batch");
    group.throughput(Throughput::Elements(BS as u64));
    group.sample_size(30);

    for &(workers, items) in &[(0usize, 1usize), (1, 1), (0, 2), (0, 4), (1, 4)] {
        group.bench_function(format!("w{workers}_items{items}"), |b| {
            let mut loader = DataLoader::builder(HeavyCpuDs(BS))
                .batch_size(BS)
                .num_workers(workers)
                .items_in_parallel(items)
                .collator(CatCollator)
                .build();
            b.iter(|| std::hint::black_box(loader.iter().next().unwrap().unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_first_batch);
criterion_main!(benches);
