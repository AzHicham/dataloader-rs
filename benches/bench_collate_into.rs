//! Benchmark: allocating every batch against collating into a reused buffer.
//!
//! FIXED:
//!   batch = 64 samples of 3×224×224 (an NCHW batch of RGB tiles): 38.5 MB in
//!   f32, 19.3 MB in 16-bit (`u16` stands in for `f16`: same size, same copy)
//!   samples = shared `Arc<[T]>` rows, so fetching costs nothing and only
//!   collation is measured
//!
//! SWEPT:
//!   element type: f32, 16-bit
//!   collation: `alloc` (a new `Vec` per batch, filled with `extend_from_slice`)
//!              or `pooled` (`collate_into` a pooled `HeapAllocator` buffer)
//!
//! Groups:
//!   "collate_into/collate"      — one `collate` call, batch dropped at once
//!   "collate_into/loader_w0"    — an epoch of 8 batches, `num_workers(0)`
//!   "collate_into/loader_w2"    — an epoch of 8 batches, `num_workers(2)`
//!
//! Throughput unit: bytes of batch per iteration.

use std::{hint::black_box, sync::Arc};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::{
    Collator, DataLoader, Dataset,
    collator::{ConcatCollator, HeapAllocator, PooledCollator},
    error::Result,
};

const BATCH: usize = 64;
const ITEM_LEN: usize = 3 * 224 * 224;
const BATCHES_PER_EPOCH: usize = 8;

/// Hands out clones of a few shared rows.
struct RowsDs<T> {
    rows: Vec<Arc<[T]>>,
    len: usize,
}

impl<T: Copy + Default + Send + Sync + 'static> RowsDs<T> {
    fn new(len: usize, value: impl Fn(usize) -> T) -> Self {
        let rows = (0..4)
            .map(|r| (0..ITEM_LEN).map(|i| value(r + i)).collect())
            .collect();
        Self { rows, len }
    }
}

impl<T: Send + Sync + 'static> Dataset for RowsDs<T> {
    type Item = Arc<[T]>;

    fn get(&self, index: usize) -> Result<Arc<[T]>> {
        Ok(Arc::clone(&self.rows[index % self.rows.len()]))
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// The allocate-per-batch baseline: a fresh `Vec` per batch.
struct AllocCollator;

impl<T: Copy + Send + 'static> Collator<Arc<[T]>> for AllocCollator {
    type Batch = Vec<T>;

    fn collate(&self, items: Vec<Arc<[T]>>) -> Result<Vec<T>> {
        let total = items
            .len()
            .checked_mul(ITEM_LEN)
            .ok_or("batch size overflow")?;
        let mut batch = Vec::with_capacity(total);
        for item in &items {
            batch.extend_from_slice(item);
        }
        Ok(batch)
    }
}

fn items<T: Copy + Default + Send + Sync + 'static>(ds: &RowsDs<T>) -> Vec<Arc<[T]>> {
    (0..BATCH).map(|i| ds.get(i).unwrap()).collect()
}

fn bench_type<T>(c: &mut Criterion, name: &str, value: impl Fn(usize) -> T + Copy)
where
    T: Copy + Default + Send + Sync + 'static,
{
    let bytes = (BATCH * ITEM_LEN * size_of::<T>()) as u64;
    let alloc = || HeapAllocator::<T>::for_shape(&[BATCH, ITEM_LEN]).unwrap();

    let mut group = c.benchmark_group("collate_into/collate");
    group.throughput(Throughput::Bytes(bytes));
    let ds = RowsDs::new(BATCH, value);
    group.bench_function(BenchmarkId::new("alloc", name), |b| {
        b.iter(|| black_box(AllocCollator.collate(items(&ds)).unwrap()));
    });
    let pooled = PooledCollator::new(ConcatCollator::<T>::new(ITEM_LEN), alloc(), 2);
    group.bench_function(BenchmarkId::new("pooled", name), |b| {
        b.iter(|| black_box(pooled.collate(items(&ds)).unwrap()));
    });
    group.finish();

    for workers in [0usize, 2] {
        let mut group = c.benchmark_group(format!("collate_into/loader_w{workers}"));
        group.throughput(Throughput::Bytes(bytes * BATCHES_PER_EPOCH as u64));
        let len = BATCH * BATCHES_PER_EPOCH;

        let mut loader = DataLoader::builder(RowsDs::new(len, value))
            .batch_size(BATCH)
            .num_workers(workers)
            .collator(AllocCollator)
            .build();
        group.bench_function(BenchmarkId::new("alloc", name), |b| {
            b.iter(|| {
                loader
                    .iter()
                    .for_each(|batch| drop(black_box(batch.unwrap())))
            });
        });

        let mut loader = DataLoader::builder(RowsDs::new(len, value))
            .batch_size(BATCH)
            .num_workers(workers)
            .collate_into(ConcatCollator::<T>::new(ITEM_LEN), alloc())
            .build();
        group.bench_function(BenchmarkId::new("pooled", name), |b| {
            b.iter(|| {
                loader
                    .iter()
                    .for_each(|batch| drop(black_box(batch.unwrap())))
            });
        });
        group.finish();
    }
}

fn bench_collate_into(c: &mut Criterion) {
    bench_type::<f32>(c, "f32", |i| i as f32);
    bench_type::<u16>(c, "f16", |i| i as u16);
}

criterion_group!(benches, bench_collate_into);
criterion_main!(benches);
