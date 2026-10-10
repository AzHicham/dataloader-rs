//! AsyncDataLoader: runtime-agnostic async datasets.

use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use dataloader_rs::{
    AsyncDataLoader, AsyncDataset, RandomSampler, collator::Collator, error::Result,
};
use futures::{StreamExt, channel::oneshot};

/// Runtime-agnostic sleep: a helper thread completes a oneshot.
async fn sleep(duration: Duration) {
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        std::thread::sleep(duration);
        let _ = tx.send(());
    });
    rx.await.unwrap();
}

/// Each `get` "waits on the network" for `latency`, tracking peak in-flight.
struct LatencyDs {
    len: usize,
    latency: Duration,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

impl LatencyDs {
    fn new(len: usize, latency: Duration) -> Self {
        Self {
            len,
            latency,
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }
}

impl AsyncDataset for LatencyDs {
    type Item = usize;

    async fn get(&self, index: usize) -> Result<usize> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        sleep(self.latency).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(index)
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Instant dataset; `fail_at` errors, `panic_at` panics.
struct InstantDs {
    len: usize,
    fail_at: Option<usize>,
    panic_at: Option<usize>,
}

impl InstantDs {
    fn new(len: usize) -> Self {
        Self {
            len,
            fail_at: None,
            panic_at: None,
        }
    }
}

impl AsyncDataset for InstantDs {
    type Item = usize;

    async fn get(&self, index: usize) -> Result<usize> {
        if self.panic_at == Some(index) {
            panic!("async dataset panicked at {index}");
        }
        if self.fail_at == Some(index) {
            return Err(format!("fetch failed at {index}").into());
        }
        Ok(index)
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[test]
fn iter_yields_batches_in_order() {
    let mut loader = AsyncDataLoader::builder(InstantDs::new(10))
        .batch_size(4)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(
        batches,
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
    );
    assert_eq!(loader.batch_len(), 3);
}

#[test]
fn iter_is_exact_size() {
    let mut loader = AsyncDataLoader::builder(InstantDs::new(10))
        .batch_size(4)
        .drop_last(true)
        .build();
    let mut iter = loader.iter();
    assert_eq!(iter.len(), 2);
    iter.next();
    assert_eq!(iter.len(), 1);
}

#[test]
fn requests_overlap_across_items_and_batches() {
    // 64 samples x 50 ms: sequential fetching would take 3.2 s.
    let latency = Duration::from_millis(50);
    let mut loader = AsyncDataLoader::builder(LatencyDs::new(64, latency))
        .batch_size(8)
        .concurrency(4)
        .build();

    let start = Instant::now();
    let seen: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    let elapsed = start.elapsed();

    assert_eq!(seen, (0..64).collect::<Vec<_>>(), "order is preserved");
    assert!(elapsed < Duration::from_millis(1000), "took {elapsed:?}");
    let peak = loader.dataset().peak.load(Ordering::SeqCst);
    assert!(peak > 8, "batches must overlap, peak in flight was {peak}");
    assert!(
        peak <= 4 * 8,
        "at most concurrency * batch_size in flight, got {peak}"
    );
}

#[test]
fn stream_runs_on_futures_executor() {
    let mut loader = AsyncDataLoader::builder(LatencyDs::new(12, Duration::from_millis(5)))
        .batch_size(4)
        .build();
    let batches: Vec<Vec<usize>> =
        futures::executor::block_on(loader.stream().map(|b| b.unwrap()).collect::<Vec<_>>());
    assert_eq!(
        batches,
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]
    );
}

/// A dataset whose futures need tokio (its timer), polled through `stream`.
struct TokioDs;

impl AsyncDataset for TokioDs {
    type Item = usize;

    async fn get(&self, index: usize) -> Result<usize> {
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok(index * 2)
    }

    fn len(&self) -> usize {
        32
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_runs_inside_tokio() {
    let mut loader = AsyncDataLoader::builder(TokioDs)
        .batch_size(8)
        .concurrency(4)
        .build();
    let start = Instant::now();
    let mut stream = std::pin::pin!(loader.stream());
    let mut out = Vec::new();
    while let Some(batch) = stream.next().await {
        out.extend(batch.unwrap());
    }
    assert_eq!(out, (0..32).map(|i| i * 2).collect::<Vec<_>>());
    // 32 x 20 ms sequentially = 640 ms; all overlap -> ~20 ms.
    assert!(start.elapsed() < Duration::from_millis(400));
}

#[test]
fn every_index_covered_with_random_sampler() {
    let mut loader = AsyncDataLoader::builder(InstantDs::new(50))
        .batch_size(7)
        .sampler(RandomSampler::new(3))
        .build();
    let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seen.len(), 50);
}

#[test]
fn error_fails_its_batch_only() {
    let mut loader = AsyncDataLoader::builder(InstantDs {
        fail_at: Some(5),
        ..InstantDs::new(12)
    })
    .batch_size(4)
    .build();
    let results: Vec<_> = loader.iter().collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    let err = results[1].as_ref().unwrap_err().to_string();
    assert!(err.contains("fetch failed at 5"), "{err}");
    assert!(results[2].is_ok());
}

#[test]
#[should_panic(expected = "async dataset panicked at 6")]
fn panic_is_reraised_by_iter() {
    let mut loader = AsyncDataLoader::builder(InstantDs {
        panic_at: Some(6),
        ..InstantDs::new(12)
    })
    .batch_size(4)
    .build();
    loader.iter().for_each(drop);
}

#[test]
fn dropping_iterator_early_does_not_hang() {
    let mut loader = AsyncDataLoader::builder(LatencyDs::new(400, Duration::from_millis(5)))
        .batch_size(4)
        .concurrency(2)
        .build();
    let mut iter = loader.iter();
    iter.next().unwrap().unwrap();
    drop(iter);
    // The loader is reusable for a full epoch.
    assert_eq!(loader.iter().count(), 100);
}

#[test]
fn get_batch_override_is_called_once_per_batch() {
    struct BatchedDs(Mutex<Vec<Vec<usize>>>);
    impl AsyncDataset for BatchedDs {
        type Item = usize;
        async fn get(&self, _index: usize) -> Result<usize> {
            unreachable!("get_batch is overridden")
        }
        async fn get_batch(&self, indices: &[usize]) -> Result<Vec<usize>> {
            self.0.lock().unwrap().push(indices.to_vec());
            Ok(indices.iter().map(|i| i + 100).collect())
        }
        fn len(&self) -> usize {
            6
        }
    }

    let mut loader = AsyncDataLoader::builder(BatchedDs(Mutex::new(Vec::new())))
        .batch_size(3)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches, vec![vec![100, 101, 102], vec![103, 104, 105]]);
    assert_eq!(
        *loader.dataset().0.lock().unwrap(),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
}

#[test]
fn collate_runs_on_collate_pool() {
    struct ThreadCollator(Arc<Mutex<HashSet<String>>>);
    impl Collator<usize> for ThreadCollator {
        type Batch = usize;
        fn collate(&self, items: Vec<usize>) -> Result<usize> {
            let name = std::thread::current().name().unwrap_or("?").to_owned();
            self.0.lock().unwrap().insert(name);
            Ok(items.into_iter().sum())
        }
    }

    let names = Arc::new(Mutex::new(HashSet::new()));
    let mut loader = AsyncDataLoader::builder(InstantDs::new(8))
        .batch_size(4)
        .collator(ThreadCollator(Arc::clone(&names)))
        .collate_threads(2)
        .build();
    let sums: Vec<usize> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(sums, vec![6, 22]);
    let names = names.lock().unwrap();
    assert!(
        names.iter().all(|n| n.starts_with("dataloader-collate-")),
        "{names:?}"
    );
}

// ── max_concurrency ───────────────────────────────────────────────────────────

#[test]
fn max_concurrency_caps_samples_in_flight_across_batches() {
    let mut loader = AsyncDataLoader::builder(LatencyDs::new(30, Duration::from_millis(10)))
        .batch_size(8)
        .concurrency(4)
        .max_concurrency(5)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();

    // Batches are regrouped in order, including the shorter last one.
    let expected: Vec<Vec<usize>> = (0..30)
        .collect::<Vec<_>>()
        .chunks(8)
        .map(<[_]>::to_vec)
        .collect();
    assert_eq!(batches, expected);
    let peak = loader.dataset().peak.load(Ordering::SeqCst);
    assert_eq!(peak, 5, "the limit is reached but never exceeded");
}

#[test]
fn max_concurrency_with_drop_last_and_shuffle() {
    let mut loader = AsyncDataLoader::builder(LatencyDs::new(30, Duration::from_millis(1)))
        .batch_size(8)
        .drop_last(true)
        .sampler(RandomSampler::new(9))
        .max_concurrency(3)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 3);
    assert!(batches.iter().all(|b| b.len() == 8));
    let distinct: HashSet<usize> = batches.into_iter().flatten().collect();
    assert_eq!(distinct.len(), 24);
}

#[test]
fn max_concurrency_error_fails_its_batch_only() {
    let mut loader = AsyncDataLoader::builder(InstantDs {
        fail_at: Some(5),
        ..InstantDs::new(12)
    })
    .batch_size(4)
    .max_concurrency(2)
    .build();
    let results: Vec<_> = loader.iter().collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok() && results[1].is_err() && results[2].is_ok());
}

#[test]
fn max_concurrency_works_through_stream() {
    let mut loader = AsyncDataLoader::builder(InstantDs::new(10))
        .batch_size(3)
        .max_concurrency(2)
        .build();
    let batches: Vec<Vec<usize>> =
        futures::executor::block_on(loader.stream().map(|b| b.unwrap()).collect::<Vec<_>>());
    assert_eq!(
        batches,
        vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]
    );
}

// ── tokio feature: sync iter() over tokio-based datasets ──────────────────────

#[cfg(feature = "tokio")]
#[test]
fn iter_drives_tokio_futures_on_owned_runtime() {
    let mut loader = AsyncDataLoader::builder(TokioDs)
        .batch_size(8)
        .concurrency(4)
        .build();
    // Several epochs on the same runtime.
    for _ in 0..2 {
        let start = Instant::now();
        let out: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(out, (0..32).map(|i| i * 2).collect::<Vec<_>>());
        assert!(start.elapsed() < Duration::from_millis(400));
    }
}

#[cfg(feature = "tokio")]
#[test]
fn iter_uses_supplied_tokio_handle() {
    struct WhichRuntime;
    impl AsyncDataset for WhichRuntime {
        type Item = String;
        async fn get(&self, _index: usize) -> Result<String> {
            let id = tokio::runtime::Handle::current().id();
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(id.to_string())
        }
        fn len(&self) -> usize {
            4
        }
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let mut loader = AsyncDataLoader::builder(WhichRuntime)
        .batch_size(2)
        .tokio_handle(runtime.handle().clone())
        .build();
    let ids: HashSet<String> = loader.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(ids, HashSet::from([runtime.handle().id().to_string()]));
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn loader_can_be_built_and_dropped_inside_tokio() {
    // Dropping the loader's own runtime from async code must not panic.
    let loader = AsyncDataLoader::builder(TokioDs).batch_size(4).build();
    drop(loader);
}
