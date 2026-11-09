//! Where a [`DataLoader`](crate::DataLoader) gets its items: a [`Dataset`] in a
//! [`SyncSource`], or an `AsyncDataset` and the tokio runtime that fetches its items in an
//! `AsyncSource` (feature `async`).

use std::sync::Arc;

#[cfg(feature = "async")]
use std::thread;

use crate::{collator::Collator, dataset::Dataset, loader::iter::SyncEpoch, transform::Transform};
#[cfg(feature = "async")]
use crate::{
    dataset::AsyncDataset,
    loader::async_epoch::{AsyncEpoch, LazyRuntime},
};

/// How an epoch runs: batches in flight, items in flight per batch, batches buffered.
pub struct EpochSettings {
    pub(super) num_workers: usize,
    pub(super) item_concurrency: usize,
    pub(super) prefetch_depth: usize,
}

/// The dataset of a [`DataLoader`](crate::DataLoader), as the loader holds it: a
/// [`SyncSource`] from [`DataLoader::builder`](crate::DataLoader::builder), or an
/// `AsyncSource` from `DataLoader::builder_async` (feature `async`). Sealed: these two are
/// the only sources.
pub trait LoaderSource: sealed::Sealed {
    /// The dataset: a [`Dataset`] or an `AsyncDataset`.
    type Dataset;

    /// The dataset.
    fn dataset(&self) -> &Self::Dataset;
}

mod sealed {
    use std::sync::Arc;

    use crate::{
        collator::Collator,
        loader::{iter::BatchStream, source::EpochSettings},
        transform::Transform,
    };

    /// What the loader does with its source; see [`LoaderSource`](super::LoaderSource).
    pub trait Sealed {
        /// The items the dataset fetches.
        type Item: Send + 'static;

        /// One epoch over the source. `O` is `T::Output`, a parameter of its own: rustc
        /// cannot prove `C: Collator<T::Output>` from a `T: Transform<Self::Item>` bound when
        /// it compares a trait's where clauses with an impl's.
        type Epoch<T, C, O>: BatchStream<C::Batch> + Send
        where
            T: Transform<Self::Item, Output = O>,
            C: Collator<O>;

        /// Number of items in the dataset.
        fn len(&self) -> usize;

        /// Threads of the loader's rayon pool, `None` for no pool.
        fn pool_threads(
            &self,
            num_workers: usize,
            item_concurrency: usize,
            transform_is_identity: bool,
        ) -> Option<usize>;

        /// Get ready for the first epoch, at [`build`](crate::DataLoaderBuilder::build).
        fn start(&self) {}

        /// Start an epoch over `chunks`, the indices of its batches.
        fn epoch<T, C, O>(
            &self,
            transform: &Arc<T>,
            collator: &Arc<C>,
            pool: Option<&Arc<rayon::ThreadPool>>,
            settings: EpochSettings,
            chunks: Vec<Vec<usize>>,
        ) -> Self::Epoch<T, C, O>
        where
            T: Transform<Self::Item, Output = O>,
            C: Collator<O>;
    }
}

/// A [`Dataset`], as a [`DataLoader`](crate::DataLoader) built by
/// [`DataLoader::builder`](crate::DataLoader::builder) holds it.
pub struct SyncSource<D>(Arc<D>);

impl<D> SyncSource<D> {
    pub(super) fn new(dataset: D) -> Self {
        Self(Arc::new(dataset))
    }
}

impl<D: Dataset> LoaderSource for SyncSource<D> {
    type Dataset = D;

    fn dataset(&self) -> &D {
        &self.0
    }
}

impl<D: Dataset> sealed::Sealed for SyncSource<D> {
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

/// `max(num_workers, 1) × item_concurrency`.
fn pool_threads(num_workers: usize, item_concurrency: usize) -> usize {
    num_workers
        .max(1)
        .checked_mul(item_concurrency)
        .expect("num_workers × item_concurrency overflows usize")
}

/// An `AsyncDataset` and the tokio runtime that fetches its items, as a
/// [`DataLoader`](crate::DataLoader) built by
/// [`DataLoader::builder_async`](crate::DataLoader::builder_async) holds them.
#[cfg(feature = "async")]
pub struct AsyncSource<A> {
    pub(super) dataset: Arc<A>,
    pub(super) runtime: LazyRuntime,
}

#[cfg(feature = "async")]
impl<A> AsyncSource<A> {
    pub(super) fn new(dataset: A) -> Self {
        Self {
            dataset: Arc::new(dataset),
            runtime: LazyRuntime::new(),
        }
    }
}

#[cfg(feature = "async")]
impl<A: AsyncDataset> LoaderSource for AsyncSource<A> {
    type Dataset = A;

    fn dataset(&self) -> &A {
        &self.dataset
    }
}

#[cfg(feature = "async")]
impl<A: AsyncDataset> sealed::Sealed for AsyncSource<A> {
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

/// The CPUs available to the process (affinity and cgroup quota included), 1 if unknown.
#[cfg(feature = "async")]
fn available_cpus() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}
