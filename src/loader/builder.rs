use std::sync::Arc;

#[cfg(feature = "torch-rs")]
use crate::collator::TorchPinnedCollator;
use crate::{
    collator::{Collator, VecCollator},
    dataset::Dataset,
    loader::core::DataLoader,
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
    /// Number of independent worker threads (inter-batch concurrency).
    inter_workers: usize,
    /// Number of rayon threads for intra-batch item-level parallelism.
    intra_workers: usize,
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
            inter_workers: 0,
            intra_workers: 0,
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
    /// Batches run as tasks on the loader's thread pool, which has
    /// `max(num_workers, intra_workers)` threads. `0` (default) processes
    /// batches directly in `Iterator::next` with no spawned threads.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.inter_workers = n;
        self
    }

    /// Fetch the items of each batch in parallel, with at least `n` threads.
    ///
    /// When `> 0`, items within a batch are fetched in parallel on the same
    /// pool that runs batches (sized `max(num_workers, intra_workers)`), so
    /// batch- and item-level work share threads instead of oversubscribing
    /// the CPU. Useful for CPU-bound Rust datasets. `0` (default) fetches
    /// items sequentially via [`Dataset::get_batch`].
    pub fn intra_workers(mut self, n: usize) -> Self {
        self.intra_workers = n;
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
            inter_workers: self.inter_workers,
            intra_workers: self.intra_workers,
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
            inter_workers: self.inter_workers,
            intra_workers: self.intra_workers,
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
            inter_workers: self.inter_workers,
            intra_workers: self.intra_workers,
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
        let threads = self.inter_workers.max(self.intra_workers);
        let pool = (threads > 0).then(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("dataloader-{i}"))
                .build()
                .expect("failed to build rayon thread pool");
            Arc::new(pool)
        });

        DataLoader {
            dataset: Arc::new(self.dataset),
            batch_sampler: BatchSampler::new(self.sampler, self.batch_size, self.drop_last),
            collator: Arc::new(self.collator),
            prefetch_depth: self.prefetch_depth,
            inter_workers: self.inter_workers,
            parallel_items: self.intra_workers > 0,
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
