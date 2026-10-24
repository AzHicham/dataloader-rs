use std::sync::Arc;

#[cfg(feature = "torch-rs")]
use crate::collator::TorchPinnedCollator;
use crate::{
    collator::{Collator, VecCollator},
    dataset::Dataset,
    loader::core::{DataLoader, LoaderPool},
    sampler::{BatchSampler, Sampler, SequentialSampler},
};

/// Builder for [`DataLoader`].
pub struct DataLoaderBuilder<D, S, C> {
    dataset: D,
    sampler: S,
    collator: C,
    batch_size: usize,
    drop_last: bool,
    prefetch_depth: usize,
    /// Batches processed concurrently.
    num_workers: usize,
    /// Fetch the items of a batch in parallel on the pool.
    parallel_items: bool,
    /// Pool size requested with [`num_threads`](Self::num_threads).
    num_threads: Option<usize>,
}

impl<D: Dataset> DataLoaderBuilder<D, SequentialSampler, VecCollator> {
    pub(super) fn new(dataset: D) -> Self {
        DataLoaderBuilder {
            dataset,
            sampler: SequentialSampler,
            collator: VecCollator,
            batch_size: 1,
            drop_last: false,
            prefetch_depth: 1,
            num_workers: 0,
            parallel_items: false,
            num_threads: None,
        }
    }
}

impl<D, S, C> DataLoaderBuilder<D, S, C> {
    /// Number of items per batch. Default: `1`.
    pub fn batch_size(mut self, n: usize) -> Self {
        assert!(n > 0, "batch_size must be > 0");
        self.batch_size = n;
        self
    }

    /// Number of batches workers may run ahead of the consumer. At most
    /// `num_workers + prefetch_depth` batches are in flight or buffered.
    /// Default: `1`.
    pub fn prefetch_depth(mut self, n: usize) -> Self {
        assert!(n > 0, "prefetch_depth must be > 0");
        self.prefetch_depth = n;
        self
    }

    /// Whether to drop a non-full final batch. Default: `false`.
    pub fn drop_last(mut self, v: bool) -> Self {
        self.drop_last = v;
        self
    }

    /// Number of batches processed concurrently — PyTorch's `num_workers`.
    ///
    /// Batches run as tasks on the loader's thread pool (see
    /// [`num_threads`](Self::num_threads)). `0` (default) processes batches
    /// directly in `Iterator::next` with no spawned threads.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Fetch the items of each batch in parallel. Default: `false`.
    ///
    /// Items are fetched with [`Dataset::get`] on the loader's thread pool,
    /// the same one that runs batches, so batch- and item-level work share
    /// threads instead of oversubscribing the CPU. Useful for CPU-bound Rust
    /// datasets. When `false`, items are fetched sequentially via
    /// [`Dataset::get_batch`].
    pub fn parallel_items(mut self, enabled: bool) -> Self {
        self.parallel_items = enabled;
        self
    }

    /// Size of the loader's thread pool, never below `num_workers`.
    ///
    /// Default: `num_workers`, or the number of available CPUs when
    /// [`parallel_items`](Self::parallel_items) is enabled. The pool is built
    /// once and reused by every epoch; it is not created when `num_workers` is
    /// `0` and `parallel_items` is off.
    pub fn num_threads(mut self, n: usize) -> Self {
        assert!(n > 0, "num_threads must be > 0");
        self.num_threads = Some(n);
        self
    }

    /// Fetch the items of each batch in parallel with a pool of at least `n`
    /// threads.
    ///
    /// Equivalent to `parallel_items(n > 0)` plus `num_threads(n)` when
    /// `n > 0`. The pool is shared with batch-level work and items use every
    /// idle thread, so `n` only sets the pool size, not a per-batch limit.
    #[deprecated(note = "use `parallel_items(true)` and `num_threads(n)` instead")]
    pub fn intra_workers(mut self, n: usize) -> Self {
        self.parallel_items = n > 0;
        self.num_threads = (n > 0).then_some(n);
        self
    }

    /// Pin Tensor batches in host memory before they are sent to the consumer.
    ///
    /// Available only for `tch::Tensor` batches under `torch-rs` feature.
    #[cfg(feature = "torch-rs")]
    pub fn pin_memory(self, enabled: bool) -> DataLoaderBuilder<D, S, TorchPinnedCollator<C>>
    where
        D: Dataset,
        C: Collator<D::Item, Batch = tch::Tensor>,
    {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collator: TorchPinnedCollator::new(self.collator, enabled),
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            prefetch_depth: self.prefetch_depth,
            num_workers: self.num_workers,
            parallel_items: self.parallel_items,
            num_threads: self.num_threads,
        }
    }

    /// Replace the sampler and preserve all other settings.
    pub fn sampler<S2: Sampler>(self, sampler: S2) -> DataLoaderBuilder<D, S2, C> {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler,
            collator: self.collator,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            prefetch_depth: self.prefetch_depth,
            num_workers: self.num_workers,
            parallel_items: self.parallel_items,
            num_threads: self.num_threads,
        }
    }

    /// Replace the collator and preserve all other settings.
    pub fn collator<C2>(self, collator: C2) -> DataLoaderBuilder<D, S, C2> {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collator,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            prefetch_depth: self.prefetch_depth,
            num_workers: self.num_workers,
            parallel_items: self.parallel_items,
            num_threads: self.num_threads,
        }
    }

    /// Finalize configuration and construct a [`DataLoader`].
    pub fn build(self) -> DataLoader<D, S, C>
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item>,
    {
        // One pool for batch-level and item-level parallelism, so the two
        // never oversubscribe the CPU and idle threads steal either kind.
        let default_threads = if self.parallel_items {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            self.num_workers
        };
        let threads = self
            .num_threads
            .unwrap_or(default_threads)
            .max(self.num_workers);
        let pool = (threads > 0).then(|| LoaderPool::new(threads));

        DataLoader {
            dataset: Arc::new(self.dataset),
            batch_sampler: BatchSampler::new(self.sampler, self.batch_size, self.drop_last),
            collator: Arc::new(self.collator),
            prefetch_depth: self.prefetch_depth,
            num_workers: self.num_workers,
            parallel_items: self.parallel_items,
            pool,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collator::Collator, dataset::Dataset, error::Result, loader::DataLoader, sampler::Sampler,
    };

    struct TinyDs(usize);

    impl Dataset for TinyDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<Self::Item> {
            Ok(index)
        }

        fn len(&self) -> usize {
            self.0
        }
    }

    struct ReverseSampler;

    impl Sampler for ReverseSampler {
        fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
            (0..dataset_len).rev().collect()
        }
    }

    struct SumCollator;

    impl Collator<usize> for SumCollator {
        type Batch = usize;

        fn collate(&self, items: Vec<usize>) -> Result<Self::Batch> {
            Ok(items.into_iter().sum())
        }
    }

    #[test]
    fn builder_defaults_create_single_item_batches() {
        let mut loader = DataLoader::builder(TinyDs(4)).build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn builder_replaces_sampler_and_collator() {
        let mut loader = DataLoader::builder(TinyDs(4))
            .batch_size(2)
            .sampler(ReverseSampler)
            .collator(SumCollator)
            .build();

        let sums: Vec<usize> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(sums, vec![5, 1]); // [3,2] and [1,0]
    }

    #[test]
    #[should_panic]
    fn builder_rejects_zero_batch_size() {
        let _ = DataLoader::builder(TinyDs(4)).batch_size(0);
    }

    #[test]
    #[should_panic]
    fn builder_rejects_zero_prefetch_depth() {
        let _ = DataLoader::builder(TinyDs(4)).prefetch_depth(0);
    }
}
