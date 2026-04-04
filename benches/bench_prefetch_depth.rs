//! Benchmark: prefetch buffer depth (`prefetch_depth`).
//!
//! FIXED:
//!   dataset    = LightCpuDs(128)  — ~µs per item (1_000 LCG iters) so the
//!                                    pipeline fill effect is visible; if items
//!                                    are too heavy, all depths saturate equally
//!   batch_size = 8                — 16 batches total
//!   inter_workers  = 4            — workers that can outpace consumer; shallow
//!                                    depth will starve consumer, deep depth
//!                                    allows overlap
//!   intra_workers  = 0            — isolate prefetch effect only
//!
//! SWEPT:
//!   prefetch_depth in [1, 2, 4, 8, 16]
//!
//! Expected pattern: depth=1 starves consumer; depth>=4 shows full overlap.
//!
//! Throughput unit: Elements(128) per iteration (full epoch).

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dataloader_rs::DataLoader;

const N: usize = 128;
const BS: usize = 8;
const INTER: usize = 4;

fn bench_prefetch_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_depth");
    group.throughput(Throughput::Elements(N as u64));

    for &depth in &[1usize, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            &depth,
            |b, &d| {
                // Build ONCE outside b.iter()
                let mut loader = DataLoader::builder(LightCpuDs(N))
                    .batch_size(BS)
                    .num_workers(INTER)
                    .intra_workers(0)
                    .prefetch_depth(d)
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

criterion_group!(benches, bench_prefetch_depth);
criterion_main!(benches);
