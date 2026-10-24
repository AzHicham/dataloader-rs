//! Benchmark: item-level against batch-level parallelism, at equal thread counts
//! (≤ 4, calibrated for a 4-vCPU runner).
//!
//! FIXED:
//!   dataset    = HeavyCpuDs(128)  — CPU-bound items (≈ 1 ms each)
//!   batch_size = 64               — 2 batches per epoch: few batches, large ones
//!
//! SWEPT (num_workers × items_in_parallel = threads):
//!   w1_items1 (1)  w1_items2 (2)  w1_items4 (4)  w2_items2 (4)  w4_items1 (4)
//!
//! Question: with few large batches, does spreading a batch's items over the
//! threads beat processing more batches at once?
//! Throughput unit: Elements(128) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::DataLoader;

const N: usize = 128;
const BS: usize = 64;

fn bench_items_in_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("items_in_parallel");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    for &(workers, items) in &[(1usize, 1usize), (1, 2), (1, 4), (2, 2), (4, 1)] {
        group.bench_function(format!("w{workers}_items{items}"), |b| {
            let mut loader = DataLoader::builder(HeavyCpuDs(N))
                .batch_size(BS)
                .num_workers(workers)
                .items_in_parallel(items)
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

criterion_group!(benches, bench_items_in_parallel);
criterion_main!(benches);
