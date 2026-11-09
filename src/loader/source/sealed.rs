//! What the loader does with its source, out of users' reach: the trait is public so that
//! [`LoaderSource`](super::LoaderSource) can extend it, but its module is private, so no
//! other crate can implement it.

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
