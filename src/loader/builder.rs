use std::sync::Arc;

use crate::{
    collator::{Collator, VecCollator},
    dataset::Dataset,
    loader::core::DataLoader,
    sampler::{BatchSampler, Sampler, SequentialSampler},
};
#[cfg(feature = "torch-rs")]
use crate::collator::TorchPinnedCollator;

/// Builder for [`DataLoader`].
pub struct DataLoaderBuilder<D, S, C> {
    dataset: D,
    sampler: S,
    collator: C,
    batch_size: usize,
    drop_last: bool,
    prefetch_depth: usize,
    num_workers: usize,
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

    /// Maximum number of batches to keep prefetched. Default: `1`.
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

    /// Number of rayon worker threads used for intra-batch parallelism.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
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
        }
    }

    /// Finalize configuration and construct a [`DataLoader`].
    pub fn build(self) -> DataLoader<D, S, C>
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item>,
    {
        let pool = if self.num_workers > 0 {
            let tp = rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_workers)
                .build()
                .expect("failed to build rayon thread pool");
            Some(Arc::new(tp))
        } else {
            None
        };

        DataLoader {
            dataset: self.dataset,
            batch_sampler: BatchSampler::new(self.sampler, self.batch_size, self.drop_last),
            collator: self.collator,
            prefetch_depth: self.prefetch_depth,
            pool,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collator::Collator,
        dataset::Dataset,
        error::Result,
        loader::DataLoader,
        sampler::Sampler,
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
