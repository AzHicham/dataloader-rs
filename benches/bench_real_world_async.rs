//! Benchmark: network-bound samples through `AsyncDataLoader` (feature `async`).
//!
//! Same 2 ms round trip as `bench_real_world`'s `remote` group, but awaited:
//! many requests in flight on a couple of runtime threads instead of one
//! blocked thread per request.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::{AsyncDataLoader, AsyncDataset, error::Result};
use futures::StreamExt;

const BATCH_SIZE: usize = 32;
const N: usize = 512;

struct AsyncRemote(usize);

impl AsyncDataset for AsyncRemote {
    type Item = u64;

    async fn get(&self, index: usize) -> Result<u64> {
        tokio::time::sleep(Duration::from_millis(2)).await;
        Ok(index as u64)
    }

    fn len(&self) -> usize {
        self.0
    }
}

fn bench_async_remote(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("real_world/remote_2ms");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);
    for max_in_flight in [16usize, 64, 256] {
        let mut loader = AsyncDataLoader::builder(AsyncRemote(N))
            .batch_size(BATCH_SIZE)
            .concurrency(8)
            .max_concurrency(max_in_flight)
            .build();
        group.bench_with_input(
            BenchmarkId::new("async_max_concurrency", max_in_flight),
            &max_in_flight,
            |b, _| {
                b.iter(|| {
                    runtime.block_on(async {
                        let mut stream = std::pin::pin!(loader.stream());
                        while let Some(batch) = stream.next().await {
                            batch.unwrap();
                        }
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_async_remote);
criterion_main!(benches);
