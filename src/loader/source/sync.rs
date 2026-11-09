use std::sync::Arc;

use super::{LoaderSource, sealed::Sealed, settings::EpochSettings, threads::pool_threads};
use crate::{collator::Collator, dataset::Dataset, loader::sync::SyncEpoch, transform::Transform};

/// A [`Dataset`], as a [`DataLoader`](crate::DataLoader) built by
/// [`DataLoader::builder`](crate::DataLoader::builder) holds it.
pub struct SyncSource<D>(Arc<D>);

impl<D> SyncSource<D> {
    pub(in crate::loader) fn new(dataset: D) -> Self {
        Self(Arc::new(dataset))
    }
}

impl<D: Dataset> LoaderSource for SyncSource<D> {
    type Dataset = D;

    fn dataset(&self) -> &D {
        &self.0
    }
}

impl<D: Dataset> Sealed for SyncSource<D> {
    type Item = D::Item;
    type Epoch<T, C, O>
        = SyncEpoch<D, T, C>
    where
        T: Transform<D::Item, Output = O>,
        C: Collator<O>;

    fn len(&self) -> usize {
        self.0.len()
    }

    /// One pool for batch-level and item-level work: `num_workers` batches, each with
    /// `item_concurrency` items in flight; none when both are at their defaults.
    fn pool_threads(
        &self,
        num_workers: usize,
        item_concurrency: usize,
        _transform_is_identity: bool,
    ) -> Option<usize> {
        (num_workers > 0 || item_concurrency > 1)
            .then(|| pool_threads(num_workers, item_concurrency))
    }

    fn epoch<T, C, O>(
        &self,
        transform: &Arc<T>,
        collator: &Arc<C>,
        pool: Option<&Arc<rayon::ThreadPool>>,
        settings: EpochSettings,
        chunks: Vec<Vec<usize>>,
    ) -> SyncEpoch<D, T, C>
    where
        T: Transform<D::Item, Output = O>,
        C: Collator<O>,
    {
        SyncEpoch::start(&self.0, transform, collator, pool, settings, chunks)
    }
}
