//! `DataLoader::builder_async`: an `AsyncDataset` fetched on tokio (feature `async`).
#![cfg(feature = "async")]

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use dataloader_rs::{
    AsyncDataset, DataLoader, MAX_RUNTIME_THREADS, RandomSampler, collator::Collator, error::Result,
};

/// Counts a future as alive from its creation to its drop, and tracks the peak.
struct Live {
    now: Arc<AtomicUsize>,
}

impl Live {
    fn new(now: &Arc<AtomicUsize>, peak: &AtomicUsize) -> Self {
        let current = now.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(current, Ordering::SeqCst);
        Self {
            now: Arc::clone(now),
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.now.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Item `i` is `i`, after a delay; optionally fails or panics at one index.
#[derive(Default)]
struct Sleepy {
    len: usize,
    delay: Duration,
    /// Varies the delay per item so that items finish out of order.
    jitter: bool,
    fail_at: Option<usize>,
    panic_at: Option<usize>,
    now: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    started: Arc<AtomicUsize>,
    batch_calls: Arc<AtomicUsize>,
}

impl Sleepy {
    fn new(len: usize, delay: Duration) -> Self {
        Self {
            len,
            delay,
            ..Self::default()
        }
    }
}

impl AsyncDataset for Sleepy {
    type Item = usize;

    async fn get(&self, index: usize) -> Result<usize> {
        let _live = Live::new(&self.now, &self.peak);
        self.started.fetch_add(1, Ordering::SeqCst);
        let delay = if self.jitter {
            self.delay * u32::try_from(index % 5).unwrap()
        } else {
            self.delay
        };
        tokio::time::sleep(delay).await;
        if self.panic_at == Some(index) {
            panic!("boom at {index}");
        }
        if self.fail_at == Some(index) {
            return Err(format!("fail at {index}").into());
        }
        Ok(index)
    }

    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<usize>> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        let mut items = Vec::with_capacity(indices.len());
        for &index in indices {
            items.push(self.get(index).await?);
        }
        Ok(items)
    }

    fn len(&self) -> usize {
        self.len
    }
}

fn flatten(batches: Vec<Vec<usize>>) -> Vec<usize> {
    batches.into_iter().flatten().collect()
}

#[test]
fn batches_come_in_epoch_order() {
    let mut ds = Sleepy::new(64, Duration::from_millis(1));
    ds.jitter = true;
    let mut loader = DataLoader::builder_async(ds)
        .batch_size(5)
        .num_workers(4)
        .item_concurrency(3)
        .build();
    for _ in 0..2 {
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 13);
        assert_eq!(batches[12], vec![60, 61, 62, 63]);
        assert_eq!(flatten(batches), (0..64).collect::<Vec<_>>());
    }
}

#[test]
fn shuffled_epochs_cover_every_item() {
    let mut loader = DataLoader::builder_async(Sleepy::new(50, Duration::ZERO))
        .batch_size(8)
        .num_workers(2)
        .item_concurrency(4)
        .sampler(RandomSampler::new(7))
        .drop_last(true)
        .build();
    assert_eq!(loader.batch_len(), 6);
    let mut it = loader.iter();
    assert_eq!(it.len(), 6);
    let mut items = flatten(it.by_ref().map(|b| b.unwrap()).collect());
    assert_eq!(it.len(), 0);
    assert_eq!(items.len(), 48);
    items.sort_unstable();
    items.dedup();
    assert_eq!(items.len(), 48);
}

#[test]
fn in_flight_is_bounded_by_workers_times_item_concurrency() {
    let ds = Sleepy::new(96, Duration::from_millis(20));
    let peak = Arc::clone(&ds.peak);
    let mut loader = DataLoader::builder_async(ds)
        .batch_size(8)
        .num_workers(3)
        .item_concurrency(4)
        .prefetch_depth(4)
        .build();
    let n: usize = loader.iter().map(|b| b.unwrap().len()).sum();
    assert_eq!(n, 96);
    // Reached, and never exceeded: 3 batches × 4 items.
    assert_eq!(peak.load(Ordering::SeqCst), 12);
}

#[test]
fn a_slow_consumer_bounds_the_batches_loaded_ahead() {
    let ds = Sleepy::new(400, Duration::ZERO);
    let started = Arc::clone(&ds.started);
    let mut loader = DataLoader::builder_async(ds)
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(4)
        .prefetch_depth(3)
        .build();
    let mut it = loader.iter();
    it.next().unwrap().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    // The batch taken, 3 in the channel, one waiting for room, one more in flight.
    let batches_started = started.load(Ordering::SeqCst).div_ceil(4);
    assert!(
        batches_started <= 1 + 3 + 2,
        "{batches_started} batches started"
    );
    assert_eq!(it.map(|b| b.unwrap().len()).sum::<usize>(), 396);
}

#[test]
fn item_concurrency_one_uses_get_batch() {
    let ds = Sleepy::new(10, Duration::ZERO);
    let batch_calls = Arc::clone(&ds.batch_calls);
    let mut loader = DataLoader::builder_async(ds).batch_size(4).build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(flatten(batches), (0..10).collect::<Vec<_>>());
    assert_eq!(batch_calls.load(Ordering::SeqCst), 3);
}

#[test]
fn a_failed_item_fails_its_batch_only() {
    for item_concurrency in [1, 4] {
        let mut ds = Sleepy::new(20, Duration::from_millis(1));
        ds.fail_at = Some(9);
        let mut loader = DataLoader::builder_async(ds)
            .batch_size(4)
            .num_workers(2)
            .item_concurrency(item_concurrency)
            .build();
        let results: Vec<_> = loader.iter().collect();
        assert_eq!(results.len(), 5);
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Err(e) => {
                    assert_eq!(i, 2);
                    assert_eq!(e.to_string(), "fail at 9");
                }
                Ok(batch) => assert_eq!(batch, (i * 4..i * 4 + 4).collect::<Vec<_>>()),
            }
        }
    }
}

#[test]
fn a_panic_in_get_is_re_raised_and_the_loader_stays_usable() {
    let mut ds = Sleepy::new(16, Duration::ZERO);
    ds.panic_at = Some(5);
    let mut loader = DataLoader::builder_async(ds)
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(2)
        .build();
    let mut it = loader.iter();
    assert_eq!(it.next().unwrap().unwrap(), vec![0, 1, 2, 3]);
    let panic = catch_unwind(AssertUnwindSafe(|| it.next())).unwrap_err();
    assert_eq!(panic.downcast_ref::<String>().unwrap(), "boom at 5");
    assert!(it.next().is_none(), "the epoch ends after a panic");
    drop(it);
    // The next epoch panics again at the same place, and starts fine.
    assert_eq!(loader.iter().next().unwrap().unwrap(), vec![0, 1, 2, 3]);
}

struct PanickingCollator;

impl Collator<usize> for PanickingCollator {
    type Batch = usize;

    fn collate(&self, items: Vec<usize>) -> Result<usize> {
        if items.contains(&6) {
            panic!("collator boom");
        }
        Ok(items.into_iter().sum())
    }
}

#[test]
fn a_panic_in_the_collator_is_re_raised() {
    let mut loader = DataLoader::builder_async(Sleepy::new(12, Duration::ZERO))
        .batch_size(4)
        .num_workers(2)
        .collator(PanickingCollator)
        .build();
    let mut it = loader.iter();
    assert_eq!(it.next().unwrap().unwrap(), 6);
    let panic = catch_unwind(AssertUnwindSafe(|| it.next())).unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"collator boom"));
}

#[test]
fn dropping_the_iterator_mid_epoch_drops_every_future_at_once() {
    let ds = Sleepy::new(1000, Duration::from_secs(3600));
    let now = Arc::clone(&ds.now);
    let mut loader = DataLoader::builder_async(ds)
        .batch_size(8)
        .num_workers(4)
        .item_concurrency(8)
        .build();
    let it = loader.iter();
    let deadline = Instant::now() + Duration::from_secs(5);
    while now.load(Ordering::SeqCst) < 32 {
        assert!(Instant::now() < deadline, "futures never started");
        std::thread::sleep(Duration::from_millis(1));
    }
    let start = Instant::now();
    drop(it);
    assert!(start.elapsed() < Duration::from_millis(100), "drop waited");
    while now.load(Ordering::SeqCst) > 0 {
        assert!(Instant::now() < deadline, "futures still alive after drop");
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(loader);
}

#[test]
fn a_new_epoch_after_an_early_drop_is_complete() {
    let mut loader = DataLoader::builder_async(Sleepy::new(40, Duration::from_millis(1)))
        .batch_size(4)
        .num_workers(3)
        .item_concurrency(2)
        .build();
    for round in 0..5 {
        let mut it = loader.iter();
        for _ in 0..round {
            it.next().unwrap().unwrap();
        }
    }
    let items = flatten(loader.iter().map(|b| b.unwrap()).collect());
    assert_eq!(items, (0..40).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_loader_inside_a_runtime_does_not_panic() {
    let mut loader = DataLoader::builder_async(Sleepy::new(8, Duration::from_secs(3600)))
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(4)
        .build();
    let it = loader.iter();
    tokio::time::sleep(Duration::from_millis(10)).await;
    drop(it);
    drop(loader);
}

#[test]
fn runs_on_the_callers_runtime() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_time()
        .build()
        .unwrap();
    let mut loader = DataLoader::builder_async(Sleepy::new(9, Duration::from_millis(1)))
        .runtime(runtime.handle().clone())
        .batch_size(3)
        .num_workers(2)
        .item_concurrency(3)
        .build();
    let items = flatten(loader.iter().map(|b| b.unwrap()).collect());
    assert_eq!(items, (0..9).collect::<Vec<_>>());
}

/// `(index, item)`: what a transform sees.
fn pair(index: usize, item: usize) -> Result<(usize, usize)> {
    Ok((index, item))
}

#[test]
fn the_transform_receives_each_items_index() {
    for item_concurrency in [1, 3] {
        let mut ds = Sleepy::new(40, Duration::from_micros(200));
        ds.jitter = true;
        let mut loader = DataLoader::builder_async(ds)
            .batch_size(6)
            .num_workers(2)
            .item_concurrency(item_concurrency)
            .sampler(RandomSampler::new(3))
            .transform(pair)
            .build();
        let pairs: Vec<(usize, usize)> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert!(pairs.iter().all(|(index, item)| index == item));
        let mut indices: Vec<usize> = pairs.iter().map(|&(index, _)| index).collect();
        assert_ne!(indices, (0..40).collect::<Vec<_>>(), "shuffled");
        indices.sort_unstable();
        assert_eq!(indices, (0..40).collect::<Vec<_>>());
    }
}

#[test]
fn a_failed_transform_fails_its_batch_only() {
    for item_concurrency in [1, 4] {
        let mut loader = DataLoader::builder_async(Sleepy::new(12, Duration::ZERO))
            .batch_size(4)
            .num_workers(2)
            .item_concurrency(item_concurrency)
            .transform(|index: usize, item: usize| -> Result<String> {
                match index {
                    5 => Err("no five".into()),
                    _ => Ok(format!("#{item}")),
                }
            })
            .build();
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches[0].as_ref().unwrap()[3], "#3");
        assert_eq!(batches[1].as_ref().unwrap_err().to_string(), "no five");
        assert!(batches[2].is_ok());
    }
}

#[test]
fn a_panic_in_the_transform_is_re_raised() {
    for item_concurrency in [1, 4] {
        let mut loader = DataLoader::builder_async(Sleepy::new(8, Duration::ZERO))
            .batch_size(4)
            .num_workers(2)
            .item_concurrency(item_concurrency)
            .transform(|index: usize, item: usize| -> Result<usize> {
                assert_ne!(index, 6, "boom in transform");
                Ok(item)
            })
            .build();
        let panic = catch_unwind(AssertUnwindSafe(|| loader.iter().count())).unwrap_err();
        let message = panic.downcast_ref::<String>().unwrap();
        assert!(message.contains("boom in transform"), "{message}");
        // The loader stays usable.
        assert_eq!(loader.iter().next().unwrap().unwrap(), vec![0, 1, 2, 3]);
    }
}

#[test]
fn the_loader_gives_back_its_dataset() {
    let loader = DataLoader::builder_async(Sleepy::new(5, Duration::ZERO)).build();
    assert_eq!(AsyncDataset::len(loader.dataset()), 5);
    assert_eq!(loader.len(), 5);
}

#[test]
fn one_runtime_thread_drives_64_requests_in_flight() {
    let ds = Sleepy::new(256, Duration::from_millis(20));
    let peak = Arc::clone(&ds.peak);
    let mut loader = DataLoader::builder_async(ds)
        .runtime_threads(1)
        .batch_size(32)
        .num_workers(2)
        .item_concurrency(32)
        .build();
    let n: usize = loader.iter().map(|b| b.unwrap().len()).sum();
    assert_eq!(n, 256);
    assert_eq!(peak.load(Ordering::SeqCst), 64);
}

#[test]
#[should_panic(expected = "runtime_threads must be between 1 and 1024, got 0")]
fn zero_runtime_threads_is_rejected() {
    let _ = DataLoader::builder_async(Sleepy::new(1, Duration::ZERO)).runtime_threads(0);
}

#[test]
#[should_panic(expected = "runtime_threads must be between 1 and 1024, got 1025")]
fn too_many_runtime_threads_are_rejected() {
    assert_eq!(MAX_RUNTIME_THREADS, 1024);
    let _ = DataLoader::builder_async(Sleepy::new(1, Duration::ZERO))
        .runtime_threads(MAX_RUNTIME_THREADS + 1);
}

#[test]
fn runtime_and_runtime_threads_are_exclusive() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();
    for threads_first in [true, false] {
        let handle = runtime.handle().clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let builder = DataLoader::builder_async(Sleepy::new(1, Duration::ZERO));
            let _ = if threads_first {
                builder.runtime_threads(2).runtime(handle)
            } else {
                builder.runtime(handle).runtime_threads(2)
            };
        }));
        let panic = result.expect_err("both runtime and runtime_threads were accepted");
        let message = panic.downcast_ref::<String>().map(String::as_str);
        let message = message.or_else(|| panic.downcast_ref::<&str>().copied());
        assert!(
            message.is_some_and(|m| m.contains("mutually exclusive")),
            "{message:?}"
        );
    }
}
