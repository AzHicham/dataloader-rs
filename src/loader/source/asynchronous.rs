use std::sync::Arc;

use super::{
    LoaderSource,
    sealed::Sealed,
    settings::EpochSettings,
    threads::{available_cpus, pool_threads},
};
use crate::{
    collator::Collator,
    dataset::AsyncDataset,
    loader::asynchronous::{AsyncEpoch, LazyRuntime},
    transform::Transform,
};

/// An `AsyncDataset` and the tokio runtime that fetches its items, as a
/// [`DataLoader`](crate::DataLoader) built by
/// [`DataLoader::builder_async`](crate::DataLoader::builder_async) holds them.
pub struct AsyncSource<A> {
    pub(in crate::loader) dataset: Arc<A>,
    pub(in crate::loader) runtime: LazyRuntime,
}

impl<A> AsyncSource<A> {
    pub(in crate::loader) fn new(dataset: A) -> Self {
        Self {
            dataset: Arc::new(dataset),
            runtime: LazyRuntime::new(),
        }
    }
}

impl<A: AsyncDataset> LoaderSource for AsyncSource<A> {
    type Dataset = A;

    fn dataset(&self) -> &A {
        &self.dataset
    }
}

impl<A: AsyncDataset> Sealed for AsyncSource<A> {
    type Item = A::Item;
    type Epoch<T, C, O>
        = AsyncEpoch<C::Batch>
    where
        T: Transform<A::Item, Output = O>,
        C: Collator<O>;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Items are futures, not threads: the pool only transforms and collates, CPU work that
    /// more threads than CPUs would not speed up; without a transform it only collates, one
    /// batch per worker.
    fn pool_threads(
        &self,
        num_workers: usize,
        item_concurrency: usize,
        transform_is_identity: bool,
    ) -> Option<usize> {
        let batches = num_workers.max(1);
        Some(match transform_is_identity {
            true => batches,
            false => pool_threads(num_workers, item_concurrency)
                .min(available_cpus())
                .max(batches),
        })
    }

    /// The runtime is built here, not on the first epoch.
    fn start(&self) {
        self.runtime.get();
    }

    fn epoch<T, C, O>(
        &self,
        transform: &Arc<T>,
        collator: &Arc<C>,
        pool: Option<&Arc<rayon::ThreadPool>>,
        settings: EpochSettings,
        chunks: Vec<Vec<usize>>,
    ) -> AsyncEpoch<C::Batch>
    where
        T: Transform<A::Item, Output = O>,
        C: Collator<O>,
    {
        let pool = pool.expect("an async loader has a pool");
        AsyncEpoch::start(self, transform, collator, pool, settings, chunks)
    }
}
