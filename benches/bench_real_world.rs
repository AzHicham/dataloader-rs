//! Benchmark: real-world per-sample work through the thread-pool loader.
//!
//! The dataset does what a real one does; the benchmark measures how well the
//! loader spreads it across workers:
//!
//!   local_files  read a 64 KB file + normalise to f32   (I/O + light CPU)
//!   remote       2 ms blocking network round trip        (I/O-bound)
//!
//! Throughput unit: samples per epoch.

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::{DataLoader, Dataset, error::Result};

const BATCH_SIZE: usize = 32;
const FILE_BYTES: usize = 64 * 1024;

fn file_tree(n: usize) -> PathBuf {
    let root = std::env::temp_dir().join(format!("dataloader_rs_rust_bench_{n}"));
    std::fs::create_dir_all(&root).unwrap();
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    for i in 0..n {
        let path = root.join(format!("{i:06}.bin"));
        if std::fs::metadata(&path).map(|m| m.len() as usize).ok() != Some(FILE_BYTES) {
            let bytes: Vec<u8> = (0..FILE_BYTES)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (seed >> 56) as u8
                })
                .collect();
            std::fs::write(&path, bytes).unwrap();
        }
    }
    root
}

struct LocalFiles(Vec<PathBuf>);

impl LocalFiles {
    fn new(root: &Path, n: usize) -> Self {
        Self((0..n).map(|i| root.join(format!("{i:06}.bin"))).collect())
    }
}

impl Dataset for LocalFiles {
    type Item = Vec<f32>;

    fn get(&self, index: usize) -> Result<Vec<f32>> {
        let raw = std::fs::read(&self.0[index])?;
        Ok(raw
            .iter()
            .map(|&b| (b as f32 / 255.0 - 0.5) / 0.25)
            .collect())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

struct Remote(usize);

impl Dataset for Remote {
    type Item = u64;

    fn get(&self, index: usize) -> Result<u64> {
        std::thread::sleep(Duration::from_millis(2));
        Ok(index as u64)
    }

    fn len(&self) -> usize {
        self.0
    }
}

fn bench_local_files(c: &mut Criterion) {
    const N: usize = 1024;
    let root = file_tree(N);
    let mut group = c.benchmark_group("real_world/local_files");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);
    for workers in [0usize, 2, 4, 8] {
        let mut loader = DataLoader::builder(LocalFiles::new(&root, N))
            .batch_size(BATCH_SIZE)
            .num_workers(workers)
            .prefetch_depth(2 * workers.max(1))
            .build();
        group.bench_with_input(
            BenchmarkId::new("num_workers", workers),
            &workers,
            |b, _| {
                b.iter(|| loader.iter().for_each(|batch| drop(batch.unwrap())));
            },
        );
    }
    group.finish();
}

fn bench_remote(c: &mut Criterion) {
    const N: usize = 512;
    let mut group = c.benchmark_group("real_world/remote_2ms");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);
    // Blocking I/O: one thread per in-flight request.
    for workers in [4usize, 16, 64] {
        let mut loader = DataLoader::builder(Remote(N))
            .batch_size(BATCH_SIZE)
            .num_workers(workers.min(N / BATCH_SIZE))
            .intra_workers(workers)
            .prefetch_depth(4)
            .build();
        group.bench_with_input(BenchmarkId::new("threads", workers), &workers, |b, _| {
            b.iter(|| loader.iter().for_each(|batch| drop(batch.unwrap())));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_local_files, bench_remote);
criterion_main!(benches);
