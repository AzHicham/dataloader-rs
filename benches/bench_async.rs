//! Async datasets over HTTP: futures (`builder_async` + `HttpDataset`) against threads (the
//! sync loader, one blocking request per thread) at equal requests in flight.
//!
//! A local keep-alive HTTP/1.1 server, on a runtime of its own, answers every request with a
//! 64 KiB body after `LATENCY` (a remote tile server). An epoch is 512 items in batches of 64
//! (the msintuit-ec feature pass, shortened for CI): 8 batches, one per worker at 8 workers.
//! 64 requests in flight need at least 512 / 64 × 10 ms = 80 ms per epoch, 8 in flight 640 ms.
//! Flat sampling (10 epochs) keeps each benchmark to a few seconds.
//!
//! ```text
//! cargo bench --features http --bench bench_async
//! ```

use std::net::SocketAddr;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use dataloader_rs::{
    AsyncDataset, DataLoader, Dataset,
    dataset::{HttpDataset, Retry},
    error::Result,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::{Builder, Runtime};
use tokio::time::sleep;

const ITEMS: usize = 512;
const BATCH: usize = 64;
const BODY: usize = 64 << 10;
const LATENCY: Duration = Duration::from_millis(10);

/// Start the server on its own threads; it lives until the process ends.
fn start_server() -> SocketAddr {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let runtime = Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            let body = Bytes::from(vec![7u8; BODY]);
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                tokio::spawn(serve(stream, body.clone()));
            }
        });
    });
    rx.recv().unwrap()
}

/// Answer the requests of one keep-alive connection until the client closes it.
async fn serve(mut stream: TcpStream, body: Bytes) {
    stream.set_nodelay(true).ok();
    let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {BODY}\r\n\r\n");
    let mut request = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        while let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            request.drain(..end + 4);
            sleep(LATENCY).await;
            if stream.write_all(head.as_bytes()).await.is_err()
                || stream.write_all(&body).await.is_err()
            {
                return;
            }
        }
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    }
}

fn http_dataset(addr: SocketAddr) -> HttpDataset {
    HttpDataset::builder()
        .retry(Retry::NONE)
        .max_idle_per_host(256)
        .url_fn(ITEMS, move |i| format!("http://{addr}/tile/{i}"))
        .unwrap()
}

/// The same requests through a blocking dataset: each `get` waits on its own thread for the
/// request, made on a runtime of its own.
struct Blocking {
    dataset: HttpDataset,
    runtime: Runtime,
}

impl Blocking {
    fn new(dataset: HttpDataset) -> Self {
        let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
        Self { dataset, runtime }
    }
}

impl Dataset for Blocking {
    type Item = Bytes;

    fn get(&self, index: usize) -> Result<Bytes> {
        self.runtime.block_on(self.dataset.get(index))
    }

    fn len(&self) -> usize {
        AsyncDataset::len(&self.dataset)
    }
}

fn bench_http(c: &mut Criterion) {
    let addr = start_server();
    let mut group = c.benchmark_group("async_http");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    // (num_workers, item_concurrency): 64 requests in flight, or 8 like torch with 8 workers.
    for (workers, concurrency) in [(1, 64), (2, 32), (4, 16), (8, 1)] {
        let mut futures = DataLoader::builder_async(http_dataset(addr))
            .batch_size(BATCH)
            .num_workers(workers)
            .item_concurrency(concurrency)
            .build();
        group.bench_function(format!("futures/w{workers}_c{concurrency}"), |b| {
            b.iter(|| {
                assert_eq!(
                    futures
                        .iter()
                        .map(|batch| batch.unwrap().len())
                        .sum::<usize>(),
                    ITEMS
                )
            })
        });

        let blocking = Blocking::new(http_dataset(addr));
        let mut threads = DataLoader::builder(blocking)
            .batch_size(BATCH)
            .num_workers(workers)
            .item_concurrency(concurrency)
            .build();
        group.bench_function(format!("threads/w{workers}_c{concurrency}"), |b| {
            b.iter(|| {
                assert_eq!(
                    threads
                        .iter()
                        .map(|batch| batch.unwrap().len())
                        .sum::<usize>(),
                    ITEMS
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_http);
criterion_main!(benches);
