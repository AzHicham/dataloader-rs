use std::sync::Arc;

use crate::{
    collator::{Collator, VecCollator},
    dataset::Dataset,
    error::Result,
    loader::{builder::DataLoaderBuilder, iter::DataLoaderIter},
    sampler::{BatchSampler, Sampler, SequentialSampler},
};

/// High-performance DataLoader with a PyTorch-like interface.
pub struct DataLoader<D, S: Sampler, C> {
    pub(super) dataset: Arc<D>,
    pub(super) batch_sampler: BatchSampler<S>,
    pub(super) collator: Arc<C>,
    pub(super) prefetch_depth: usize,
    /// Batches processed in parallel. `0` means the direct path: batches are
    /// processed in `Iterator::next`.
    pub(super) num_workers: usize,
    /// Items of one batch fetched in parallel; `1` uses `Dataset::get_batch`.
    pub(super) items_in_parallel: usize,
    /// The rayon pool running batch tasks and item lanes, with
    /// `max(num_workers, 1) * items_in_parallel` threads. `None` when both
    /// settings are at their defaults.
    pub(super) pool: Option<LoaderPool>,
}

/// Main function of a pool thread: runs `ThreadBuilder::run`, possibly
/// inside some per-thread setup.
pub(crate) type ThreadMain = fn(rayon::ThreadBuilder);

/// The loader's rayon pool, rebuilt after a `fork()`.
///
/// A forked child inherits the pool but none of its threads, so batch tasks
/// spawned on it would never run and the epoch would hang (e.g. Python
/// `multiprocessing` with the `fork` start method).
pub(super) struct LoaderPool {
    pool: Arc<rayon::ThreadPool>,
    threads: usize,
    thread_main: Option<ThreadMain>,
    pid: u32,
}

impl LoaderPool {
    pub(super) fn new(threads: usize, thread_main: Option<ThreadMain>) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("dataloader-{i}"))
            .spawn_handler(move |thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                builder.spawn(move || match thread_main {
                    Some(main) => main(thread),
                    None => thread.run(),
                })?;
                Ok(())
            })
            .build()
            .expect("failed to build rayon thread pool");
        Self {
            pool: Arc::new(pool),
            threads,
            thread_main,
            pid: std::process::id(),
        }
    }

    #[cfg(test)]
    pub(super) fn threads(&self) -> usize {
        self.threads
    }

    /// The pool, rebuilt first if this process is a fork of the one that
    /// built it.
    fn get(&mut self) -> &Arc<rayon::ThreadPool> {
        if self.pid != std::process::id() {
            // Leak the inherited pool: dropping it would signal threads that
            // do not exist here, through locks a parent thread may have held
            // at fork time.
            let fresh = Self::new(self.threads, self.thread_main);
            std::mem::forget(std::mem::replace(self, fresh));
        }
        &self.pool
    }
}

impl<D: Dataset> DataLoader<D, SequentialSampler, VecCollator> {
    /// Create a [`DataLoaderBuilder`] starting with [`SequentialSampler`] and
    /// [`VecCollator`] as defaults.
    pub fn builder(dataset: D) -> DataLoaderBuilder<D, SequentialSampler, VecCollator> {
        DataLoaderBuilder::new(dataset)
    }
}

impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collator<D::Item>,
{
    /// Start one epoch of iteration.
    ///
    /// Advances the sampler (hence `&mut self`); the returned iterator does
    /// not borrow the loader.
    pub fn iter(&mut self) -> DataLoaderIter<D, C>
    where
        C::Batch: Send + 'static,
    {
        let chunks = self.batch_sampler.batch_indices(self.dataset.len());
        self.iter_chunks(chunks)
    }

    /// Start one epoch over `chunks`, the batches of indices of
    /// [`epoch_chunks`](Self::epoch_chunks).
    pub(crate) fn iter_chunks(&mut self, chunks: Vec<Vec<usize>>) -> DataLoaderIter<D, C>
    where
        C::Batch: Send + 'static,
    {
        DataLoaderIter::new(
            &self.dataset,
            &self.collator,
            self.pool.as_mut().map(LoaderPool::get),
            self.items_in_parallel,
            self.num_workers,
            self.prefetch_depth,
            chunks,
        )
    }

    /// Reference to the underlying dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Reference to the collator.
    pub fn collator(&self) -> &C {
        &self.collator
    }

    /// Whether batches run on the loader's pool (`num_workers > 0` or
    /// `items_in_parallel > 1`) rather than in `next` on the caller's thread.
    #[cfg(feature = "python")]
    pub(crate) fn has_pool(&self) -> bool {
        self.pool.is_some()
    }

    #[cfg(feature = "python")]
    pub(crate) fn sampler(&self) -> &S {
        &self.batch_sampler.sampler
    }

    #[cfg(feature = "python")]
    pub(crate) fn sampler_mut(&mut self) -> &mut S {
        &mut self.batch_sampler.sampler
    }

    /// Generate batch index chunks for one epoch, advancing the sampler.
    #[cfg(feature = "python")]
    pub(crate) fn epoch_chunks(&mut self) -> Vec<Vec<usize>> {
        self.batch_sampler.batch_indices(self.dataset.len())
    }

    /// Number of batches `n_indices` sampled indices are grouped into, for a
    /// sampler whose length is measured outside the core.
    #[cfg(feature = "python")]
    pub(crate) fn num_batches(&self, n_indices: usize) -> usize {
        self.batch_sampler.num_batches(n_indices)
    }

    /// Total number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Total number of batches for one epoch with current batch settings.
    ///
    /// Counts what the sampler yields, so a sharding or subsampling sampler
    /// (e.g. [`DistributedSampler`](crate::DistributedSampler)) is accounted for.
    pub fn batch_len(&self) -> usize {
        self.batch_sampler.len(self.dataset.len())
    }

    /// Returns `true` when the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

impl<D, S, C> IntoIterator for &mut DataLoader<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    type Item = Result<C::Batch>;
    type IntoIter = DataLoaderIter<D, C>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::Arc as StdArc,
        sync::atomic::{AtomicUsize, Ordering},
        thread,
        time::Duration,
    };

    use crate::{
        collator::{Collator, VecCollator},
        dataset::Dataset,
        error::Result,
        loader::{DataLoader, core::LoaderPool},
        sampler::{RandomSampler, Sampler, SequentialSampler},
    };

    /// Returns `index` as its item. Never fails.
    struct CountingDs {
        len: usize,
    }

    impl Dataset for CountingDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<usize> {
            Ok(index)
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    /// Tracks concurrent calls to `get` so tests can assert parallel execution.
    struct ConcurrentDs {
        len: usize,
        peak: StdArc<AtomicUsize>,
        active: StdArc<AtomicUsize>,
    }

    impl Dataset for ConcurrentDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<usize> {
            let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(current, Ordering::SeqCst);
            thread::sleep(Duration::from_micros(100));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(index)
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    /// Always returns an error.
    struct ErrDs;

    impl Dataset for ErrDs {
        type Item = u8;

        fn get(&self, _: usize) -> Result<u8> {
            Err("dataset failure".into())
        }

        fn len(&self) -> usize {
            4
        }
    }

    struct PartialErrDs {
        fail_at: usize,
        len: usize,
    }

    impl Dataset for PartialErrDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<Self::Item> {
            if index == self.fail_at {
                Err(format!("fail at index {index}").into())
            } else {
                Ok(index)
            }
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    fn counting(len: usize) -> DataLoader<CountingDs, SequentialSampler, VecCollator> {
        DataLoader::builder(CountingDs { len }).build()
    }

    #[test]
    fn builder_defaults() {
        let mut loader = counting(5);
        assert_eq!(loader.iter().count(), 5);
    }

    #[test]
    fn builder_batch_size() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn builder_drop_last() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .drop_last(true)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.len(), 3);
        }
    }

    #[test]
    fn builder_no_drop_last_exact_divisor() {
        let mut loader = DataLoader::builder(CountingDs { len: 9 })
            .batch_size(3)
            .drop_last(true)
            .build();
        assert_eq!(loader.iter().count(), 3);
    }

    #[test]
    #[should_panic]
    fn builder_zero_batch_size_panics() {
        DataLoader::builder(CountingDs { len: 5 })
            .batch_size(0)
            .build();
    }

    #[test]
    #[should_panic]
    fn builder_zero_prefetch_depth_panics() {
        DataLoader::builder(CountingDs { len: 5 })
            .prefetch_depth(0)
            .build();
    }

    #[test]
    fn sequential_batch_values() {
        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(2)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
    }

    #[test]
    fn sequential_single_item_batches() {
        let mut loader = counting(4);
        let items: Vec<usize> = loader
            .iter()
            .map(|b| b.unwrap().into_iter().next().unwrap())
            .collect();
        assert_eq!(items, vec![0, 1, 2, 3]);
    }

    #[test]
    fn empty_dataset_yields_no_batches() {
        let mut loader = counting(0);
        assert_eq!(loader.iter().count(), 0);
    }

    #[test]
    fn single_item_dataset() {
        let mut loader = counting(1);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].as_ref().unwrap(), &[0]);
    }

    #[test]
    fn all_indices_covered_sequential() {
        let n = 25;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(4)
            .build();
        let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn all_indices_covered_drop_last() {
        let n = 7;
        let bs = 3;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(bs)
            .drop_last(true)
            .build();
        let seen: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn exact_size_no_drop_last() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .build();
        let iter = loader.iter();
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn exact_size_with_drop_last() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .drop_last(true)
            .build();
        let iter = loader.iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn exact_size_decrements_as_consumed() {
        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(2)
            .build();
        let mut iter = loader.iter();
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
        iter.next();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn loader_is_reusable_across_epochs() {
        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(2)
            .build();
        let epoch1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        let epoch2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(epoch1, epoch2);
    }

    #[test]
    fn random_sampler_differs_across_epochs() {
        let mut loader = DataLoader::builder(CountingDs { len: 20 })
            .batch_size(4)
            .sampler(RandomSampler::new(0))
            .build();
        let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        let e2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_ne!(e1, e2);
    }

    #[test]
    fn all_indices_covered_random() {
        let n = 50;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(5)
            .sampler(RandomSampler::new(1))
            .build();
        let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn parallel_fetch_produces_same_items() {
        let n = 30;
        let bs = 5;

        let mut seq = DataLoader::builder(CountingDs { len: n })
            .batch_size(bs)
            .build();
        let mut par = DataLoader::builder(CountingDs { len: n })
            .batch_size(bs)
            .num_workers(4)
            .build();

        let seq_out: Vec<Vec<usize>> = seq.iter().map(|b| b.unwrap()).collect();
        let par_out: Vec<Vec<usize>> = par.iter().map(|b| b.unwrap()).collect();
        assert_eq!(seq_out, par_out);
    }

    #[test]
    fn parallel_fetch_actually_concurrent() {
        let peak = StdArc::new(AtomicUsize::new(0));
        let active = StdArc::new(AtomicUsize::new(0));

        let ds = ConcurrentDs {
            len: 16,
            peak: StdArc::clone(&peak),
            active: StdArc::clone(&active),
        };

        let mut loader = DataLoader::builder(ds)
            .batch_size(8)
            .num_workers(4)
            .prefetch_depth(2)
            .build();

        loader.iter().for_each(|b| {
            b.unwrap();
        });

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "expected concurrent get() calls, but peak concurrency was 1"
        );
    }

    #[test]
    fn dataset_error_propagates() {
        let mut loader = DataLoader::builder(ErrDs).batch_size(2).build();
        let first = loader.iter().next().unwrap();
        assert!(first.is_err());
    }

    #[test]
    fn dataset_error_does_not_panic_loader() {
        let mut loader = DataLoader::builder(ErrDs).batch_size(2).build();
        for batch in loader.iter() {
            let _ = batch;
        }
        let count = loader.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn iter_catches_error_from_dataset_get() {
        let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
            .batch_size(4)
            .build();

        let first = loader.iter().next().expect("missing first batch");
        assert!(
            first.is_err(),
            "first batch should surface dataset get() error"
        );
    }

    #[test]
    fn collate_is_skipped_when_get_fails_and_error_is_iterated() {
        struct CountingCollator {
            calls: StdArc<AtomicUsize>,
        }

        impl Collator<usize> for CountingCollator {
            type Batch = Vec<usize>;

            fn collate(&self, items: Vec<usize>) -> Result<Self::Batch> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(items)
            }
        }

        let calls = StdArc::new(AtomicUsize::new(0));
        let collator = CountingCollator {
            calls: StdArc::clone(&calls),
        };

        let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
            .batch_size(4)
            .collator(collator)
            .build();

        let results: Vec<_> = loader.iter().collect();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_err(), "first batch should be an error");
        assert!(results[1].is_ok(), "second batch should succeed");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "collate should be called only for successful batches"
        );
    }

    #[test]
    fn custom_collator_sum() {
        struct SumCollator;
        impl Collator<usize> for SumCollator {
            type Batch = usize;
            fn collate(&self, items: Vec<usize>) -> Result<usize> {
                Ok(items.iter().sum())
            }
        }

        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(3)
            .collator(SumCollator)
            .build();

        let sums: Vec<usize> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(sums, vec![3, 12]);
    }

    #[test]
    fn items_in_parallel_produces_correct_results() {
        let n = 20;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(4)
            .items_in_parallel(2)
            .build();
        let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn items_in_parallel_with_num_workers() {
        let n = 24;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(4)
            .num_workers(2)
            .items_in_parallel(2)
            .build();
        let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn items_in_parallel_fetches_a_single_item_with_get() {
        struct GetOnly;
        impl Dataset for GetOnly {
            type Item = usize;
            fn get(&self, index: usize) -> Result<usize> {
                Ok(index)
            }
            fn get_batch(&self, _: &[usize]) -> Result<Vec<usize>> {
                Err("get_batch must not be used with items_in_parallel > 1".into())
            }
            fn len(&self) -> usize {
                7
            }
        }

        let mut loader = DataLoader::builder(GetOnly)
            .batch_size(3)
            .items_in_parallel(2)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
    }

    #[test]
    fn early_drop_does_not_hang() {
        let mut loader = DataLoader::builder(CountingDs { len: 100 })
            .batch_size(5)
            .prefetch_depth(2)
            .num_workers(2)
            .build();

        let mut iter = loader.iter();
        iter.next();
        drop(iter);

        let count = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(2)
            .build()
            .iter()
            .count();
        assert_eq!(count, 5);
    }

    #[test]
    fn early_break_then_new_epoch() {
        let mut loader = DataLoader::builder(CountingDs { len: 20 })
            .batch_size(4)
            .prefetch_depth(4)
            .num_workers(2)
            .build();

        for (i, b) in (&mut loader).into_iter().enumerate() {
            b.unwrap();
            if i == 1 {
                break;
            }
        }

        let second: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(second.len(), 5);

        let all_items: HashSet<usize> = second.into_iter().flatten().collect();
        assert_eq!(all_items.len(), 20);
    }

    struct PanicDs;

    impl Dataset for PanicDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<usize> {
            if index == 5 {
                panic!("dataset panicked at {index}");
            }
            Ok(index)
        }

        fn len(&self) -> usize {
            40
        }
    }

    #[test]
    #[should_panic(expected = "dataset panicked at 5")]
    fn worker_panic_propagates_to_consumer() {
        let mut loader = DataLoader::builder(PanicDs)
            .batch_size(4)
            .num_workers(2)
            .build();
        // Before the fix the epoch silently ended after the first batch.
        for batch in loader.iter() {
            let _ = batch;
        }
    }

    #[test]
    #[should_panic(expected = "dataset panicked at 5")]
    fn items_in_parallel_panic_propagates_to_consumer() {
        let mut loader = DataLoader::builder(PanicDs)
            .batch_size(4)
            .num_workers(2)
            .items_in_parallel(2)
            .build();
        for batch in loader.iter() {
            let _ = batch;
        }
    }

    #[test]
    fn loader_is_reusable_after_worker_panic() {
        let mut loader = DataLoader::builder(PanicDs)
            .batch_size(4)
            .num_workers(2)
            .build();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loader.iter().for_each(drop);
        }));
        assert!(caught.is_err());
        // The next epoch still reaches (and re-raises) the same panic rather
        // than hanging or truncating.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loader.iter().for_each(drop);
        }));
        assert!(caught.is_err());
    }

    /// Batch 0 sleeps; every other batch is instant.
    struct SlowFirstDs(StdArc<AtomicUsize>);

    impl Dataset for SlowFirstDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<usize> {
            if index == 0 {
                thread::sleep(Duration::from_millis(200));
            }
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(index)
        }

        fn len(&self) -> usize {
            1000
        }
    }

    #[test]
    fn slow_batch_does_not_buffer_whole_epoch() {
        // While the consumer waits on batch 0, fast batches must not pile up
        // in the reorder buffer beyond num_workers + prefetch_depth.
        let (workers, depth) = (2, 2);
        let fetched = StdArc::new(AtomicUsize::new(0));
        let mut loader = DataLoader::builder(SlowFirstDs(StdArc::clone(&fetched)))
            .num_workers(workers)
            .prefetch_depth(depth)
            .build();

        let mut iter = loader.iter();
        assert_eq!(iter.next().unwrap().unwrap(), vec![0]);
        // Returning batch 0 may let one more batch start.
        let fetched_now = fetched.load(Ordering::SeqCst);
        assert!(
            fetched_now <= workers + depth + 1,
            "{fetched_now} batches fetched while waiting on batch 0"
        );
        assert_eq!(iter.count(), 999);
    }

    #[test]
    fn dropping_iterator_does_not_wait_for_running_batch() {
        let mut loader = DataLoader::builder(SlowFirstDs(StdArc::new(AtomicUsize::new(0))))
            .num_workers(2)
            .build();
        let iter = loader.iter();
        thread::sleep(Duration::from_millis(20)); // batch 0 is now sleeping
        let start = std::time::Instant::now();
        drop(iter);
        assert!(start.elapsed() < Duration::from_millis(100));
        // The loader is immediately reusable.
        assert_eq!(loader.iter().nth(1).unwrap().unwrap(), vec![1]);
    }

    #[test]
    fn batch_and_item_work_share_one_pool() {
        struct ThreadNameDs(std::sync::Mutex<HashSet<String>>);
        impl Dataset for ThreadNameDs {
            type Item = usize;
            fn get(&self, index: usize) -> Result<usize> {
                let name = thread::current().name().unwrap_or("?").to_owned();
                self.0.lock().unwrap().insert(name);
                thread::sleep(Duration::from_micros(200));
                Ok(index)
            }
            fn len(&self) -> usize {
                64
            }
        }

        let mut loader = DataLoader::builder(ThreadNameDs(Default::default()))
            .batch_size(8)
            .num_workers(2)
            .items_in_parallel(3)
            .build();
        loader.iter().for_each(|b| drop(b.unwrap()));

        let names = loader.dataset().0.lock().unwrap().clone();
        assert!(
            names.iter().all(|n| n.starts_with("dataloader-")),
            "items fetched outside the loader pool: {names:?}"
        );
        assert!(names.len() <= 6, "pool must have 2 x 3 threads: {names:?}");
    }

    fn pool_threads<S: Sampler, C>(loader: &DataLoader<CountingDs, S, C>) -> Option<usize> {
        loader.pool.as_ref().map(LoaderPool::threads)
    }

    #[test]
    fn pool_size_is_workers_times_items_in_parallel() {
        let ds = || CountingDs { len: 8 };
        assert_eq!(pool_threads(&DataLoader::builder(ds()).build()), None);
        assert_eq!(
            pool_threads(&DataLoader::builder(ds()).num_workers(3).build()),
            Some(3)
        );
        assert_eq!(
            pool_threads(&DataLoader::builder(ds()).items_in_parallel(4).build()),
            Some(4)
        );
        assert_eq!(
            pool_threads(
                &DataLoader::builder(ds())
                    .num_workers(2)
                    .items_in_parallel(5)
                    .build()
            ),
            Some(10)
        );
    }

    /// Sleeps per item and records the most `get` calls running at once.
    struct ConcurrencyDs {
        running: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Dataset for ConcurrencyDs {
        type Item = usize;
        fn get(&self, index: usize) -> Result<usize> {
            let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(10));
            self.running.fetch_sub(1, Ordering::SeqCst);
            Ok(index)
        }
        fn len(&self) -> usize {
            24
        }
    }

    #[test]
    fn items_in_parallel_bounds_item_concurrency() {
        let ds = || ConcurrencyDs {
            running: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        };
        // One batch at a time (direct path): exactly 3 items in flight.
        let mut loader = DataLoader::builder(ds())
            .batch_size(12)
            .items_in_parallel(3)
            .build();
        let items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(items, (0..24).collect::<Vec<_>>());
        assert_eq!(loader.dataset().peak.load(Ordering::SeqCst), 3);

        // Two batches in parallel, 2 items each: at most 4 in flight.
        let mut loader = DataLoader::builder(ds())
            .batch_size(12)
            .num_workers(2)
            .items_in_parallel(2)
            .build();
        let items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(items, (0..24).collect::<Vec<_>>());
        let peak = loader.dataset().peak.load(Ordering::SeqCst);
        assert!((2..=4).contains(&peak), "peak {peak}");
    }

    #[test]
    fn leaked_iterator_keeps_dataset_alive() {
        struct DropFlagDs(StdArc<AtomicUsize>);
        impl Dataset for DropFlagDs {
            type Item = usize;
            fn get(&self, index: usize) -> Result<usize> {
                Ok(index)
            }
            fn len(&self) -> usize {
                64
            }
        }
        impl Drop for DropFlagDs {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = StdArc::new(AtomicUsize::new(0));
        let mut loader = DataLoader::builder(DropFlagDs(StdArc::clone(&drops)))
            .num_workers(2)
            .build();
        std::mem::forget(loader.iter());
        drop(loader);
        // Leaked workers still hold the dataset: it must not have been freed.
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn batch_len_follows_distributed_sampler() {
        use crate::sampler::DistributedSampler;

        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(2)
            .sampler(DistributedSampler::new(SequentialSampler, 0, 4))
            .build();
        // 10 items over 4 ranks -> 3 per rank -> 2 batches of size 2.
        assert_eq!(loader.batch_len(), 2);
        assert_eq!(loader.iter().count(), 2);
    }

    #[test]
    fn loader_len() {
        let loader = counting(42);
        assert_eq!(loader.len(), 42);
        assert!(!loader.is_empty());
    }

    #[test]
    fn loader_is_empty() {
        let loader = counting(0);
        assert!(loader.is_empty());
        assert_eq!(loader.len(), 0);
    }

    #[test]
    fn dataset_accessor() {
        let loader = counting(7);
        assert_eq!(loader.dataset().len(), 7);
    }

    #[cfg(feature = "torch-rs")]
    #[test]
    fn pin_memory_true_keeps_tensor_batches_valid() {
        use crate::collator::DefaultCollator;
        use tch::{Device, Kind, Tensor};

        struct TensorDs(usize);
        impl Dataset for TensorDs {
            type Item = Tensor;

            fn get(&self, _index: usize) -> Result<Self::Item> {
                Tensor::f_zeros([2, 3], (Kind::Float, Device::Cpu)).map_err(|e| e.into())
            }

            fn len(&self) -> usize {
                self.0
            }
        }

        let mut loader = DataLoader::builder(TensorDs(4))
            .batch_size(2)
            .collator(DefaultCollator)
            .pin_memory(true)
            .build();

        for batch in loader.iter() {
            let batch = batch.unwrap();
            assert_eq!(batch.size(), vec![2, 2, 3]);
        }
    }
}
