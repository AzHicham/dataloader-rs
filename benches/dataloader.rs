//! Benchmarks for the DataLoader pipeline.
//!
//! Run with:
//!   cargo bench
//!   cargo bench -- <filter>       # e.g. "throughput/seq"
//!
//! Results are printed to stdout. Criterion also writes HTML reports to
//! `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dataloader_rs::{
    collator::Collator, error::Result, sampler::RandomSampler, DataLoader, Dataset,
};
use std::{thread, time::Duration};

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark datasets
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-cost in-memory dataset: returns the index as-is.
struct InMemoryDs(usize);

impl Dataset for InMemoryDs {
    type Item = u64;
    fn get(&self, index: usize) -> Result<u64> {
        Ok(index as u64)
    }
    fn len(&self) -> usize {
        self.0
    }
}

/// Simulates light I/O (e.g. a fast SSD read).
struct LightIoDs(usize);

impl Dataset for LightIoDs {
    type Item = u64;
    fn get(&self, index: usize) -> Result<u64> {
        thread::sleep(Duration::from_micros(50));
        Ok(index as u64)
    }
    fn len(&self) -> usize {
        self.0
    }
}

/// Simulates a heavier CPU transform (e.g. JPEG decode).
struct HeavyCpuDs(usize);

impl Dataset for HeavyCpuDs {
    type Item = Vec<u8>;
    fn get(&self, index: usize) -> Result<Vec<u8>> {
        // Spin to burn CPU cycles without allocating anything huge.
        let mut acc: u64 = index as u64;
        for i in 0..10_000u64 {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        Ok(vec![(acc & 0xff) as u8; 256])
    }
    
    fn len(&self) -> usize {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collators
// ─────────────────────────────────────────────────────────────────────────────

/// Sums all u64 items into a single value — trivially cheap.
struct SumCollator;

impl Collator<u64> for SumCollator {
    type Batch = u64;
    fn collate(&self, items: Vec<u64>) -> Result<u64> {
        Ok(items.iter().sum())
    }
}

/// Concatenates byte slices into one flat buffer.
struct CatCollator;

impl Collator<Vec<u8>> for CatCollator {
    type Batch = Vec<u8>;
    fn collate(&self, items: Vec<Vec<u8>>) -> Result<Vec<u8>> {
        Ok(items.into_iter().flatten().collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Sequential vs. parallel throughput (in-memory dataset)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_throughput_inmemory(c: &mut Criterion) {
    let n = 1_000;
    let batch_sizes = [32, 128, 512];

    let mut group = c.benchmark_group("throughput/inmemory");

    for &bs in &batch_sizes {
        // Items processed per iteration = n
        group.throughput(Throughput::Elements(n as u64));

        // Sequential (num_workers=0) — loader built once, iterated each sample.
        group.bench_with_input(
            BenchmarkId::new("sequential", bs),
            &bs,
            |b, &batch_size| {
                let mut loader = DataLoader::builder(InMemoryDs(n))
                    .batch_size(batch_size)
                    .collator(SumCollator)
                    .prefetch_depth(4)
                    .build();
                b.iter(|| {
                    let total: u64 = loader.iter().map(|b| black_box(b.unwrap())).sum();
                    black_box(total);
                });
            },
        );

        // Parallel (num_workers=4) — pool built once, reused per epoch.
        group.bench_with_input(
            BenchmarkId::new("parallel_4w", bs),
            &bs,
            |b, &batch_size| {
                let mut loader = DataLoader::builder(InMemoryDs(n))
                    .batch_size(batch_size)
                    .collator(SumCollator)
                    .num_workers(4)
                    .prefetch_depth(4)
                    .build();
                b.iter(|| {
                    let total: u64 = loader.iter().map(|b| black_box(b.unwrap())).sum();
                    black_box(total);
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. CPU-bound dataset: sequential vs parallel workers
// ─────────────────────────────────────────────────────────────────────────────

fn bench_throughput_cpu_bound(c: &mut Criterion) {
    // n and bs sized so each batch has real CPU work but not too slow for CI.
    let n = 128;
    let bs = 16;

    let mut group = c.benchmark_group("throughput/cpu_bound");
    group.throughput(Throughput::Elements(n as u64));

    for workers in [0usize, 1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("workers", workers),
            &workers,
            |b, &w| {
                // Build the loader (and its ThreadPool) ONCE outside the hot
                // loop. Otherwise we measure thread-spawn cost, not throughput.
                let mut loader = DataLoader::builder(HeavyCpuDs(n))
                    .batch_size(bs)
                    .collator(CatCollator)
                    .num_workers(w)
                    .prefetch_depth(4)
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

// ─────────────────────────────────────────────────────────────────────────────
// 3. Prefetch depth impact
// ─────────────────────────────────────────────────────────────────────────────

fn bench_prefetch_depth(c: &mut Criterion) {
    // Use the light-I/O dataset so prefetch depth has visible impact.
    let n = 64;
    let bs = 8;

    let mut group = c.benchmark_group("prefetch_depth");
    group.throughput(Throughput::Elements(n as u64));

    for &depth in &[1usize, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            &depth,
            |b, &d| {
                b.iter(|| {
                    let mut loader = DataLoader::builder(LightIoDs(n))
                        .batch_size(bs)
                        .collator(SumCollator)
                        .num_workers(2)
                        .prefetch_depth(d)
                        .build();
                    for batch in &mut loader {
                        black_box(batch.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Batch size impact on in-memory throughput
// ─────────────────────────────────────────────────────────────────────────────

fn bench_batch_size(c: &mut Criterion) {
    let n = 512;

    let mut group = c.benchmark_group("batch_size");
    group.throughput(Throughput::Elements(n as u64));

    for &bs in &[1usize, 8, 32, 128, 512] {
        group.bench_with_input(
            BenchmarkId::new("bs", bs),
            &bs,
            |b, &batch_size| {
                b.iter(|| {
                    let mut loader = DataLoader::builder(InMemoryDs(n))
                        .batch_size(batch_size)
                        .collator(SumCollator)
                        .prefetch_depth(4)
                        .build();
                    let s: u64 = loader.iter().map(|b| b.unwrap()).sum();
                    black_box(s);
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Sampler overhead: sequential vs random
// ─────────────────────────────────────────────────────────────────────────────

fn bench_sampler_overhead(c: &mut Criterion) {
    // Isolate the cost of index generation (sequential range vs Fisher-Yates
    // shuffle). Build once — the sampler state evolves across epochs, which
    // is the realistic use-case.
    let n = 10_000;
    let bs = 128;

    let mut group = c.benchmark_group("sampler_overhead");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("sequential", |b| {
        let mut loader = DataLoader::builder(InMemoryDs(n))
            .batch_size(bs)
            .collator(SumCollator)
            .prefetch_depth(4)
            .build();
        b.iter(|| {
            let s: u64 = loader.iter().map(|b| b.unwrap()).sum();
            black_box(s);
        });
    });

    group.bench_function("random", |b| {
        let mut loader = DataLoader::builder(InMemoryDs(n))
            .batch_size(bs)
            .sampler(RandomSampler::new(42))
            .collator(SumCollator)
            .prefetch_depth(4)
            .build();
        b.iter(|| {
            let s: u64 = loader.iter().map(|b| b.unwrap()).sum();
            black_box(s);
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Early-drop overhead
// ─────────────────────────────────────────────────────────────────────────────

fn bench_early_drop(c: &mut Criterion) {
    // Measure the cost of starting an iterator, consuming one batch, and
    // dropping it mid-epoch (exercises the rx-drop + thread-join path).
    // Loaders are built ONCE so we measure shutdown, not construction.
    let mut group = c.benchmark_group("early_drop");

    group.bench_function("sequential", |b| {
        let mut loader = DataLoader::builder(InMemoryDs(1000))
            .batch_size(32)
            .prefetch_depth(4)
            .build();
        b.iter(|| {
            let mut iter = loader.iter();
            black_box(iter.next().unwrap().unwrap());
            drop(iter); // triggers rx drop + prefetch thread join
        });
    });

    group.bench_function("parallel_4w", |b| {
        let mut loader = DataLoader::builder(InMemoryDs(1000))
            .batch_size(32)
            .num_workers(4)
            .prefetch_depth(4)
            .build();
        b.iter(|| {
            let mut iter = loader.iter();
            black_box(iter.next().unwrap().unwrap());
            drop(iter);
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
criterion_group!(
    benches,
    bench_throughput_inmemory,
    bench_throughput_cpu_bound,
    bench_prefetch_depth,
    bench_batch_size,
    bench_sampler_overhead,
    bench_early_drop,
);
criterion_main!(benches);
