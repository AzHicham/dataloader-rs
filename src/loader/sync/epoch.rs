//! [`SyncEpoch`]: one epoch of a [`Dataset`], on the calling thread or the loader's pool.

use std::sync::Arc;
#[cfg(feature = "python")]
use std::time::Instant;

use super::{reorder::ParallelCore, worker::process_batch};
use crate::{
    collator::Collator,
    dataset::Dataset,
    error::Result,
    loader::{
        iter::{BatchStream, forked},
        source::EpochSettings,
    },
    transform::Transform,
};

/// One epoch of a [`Dataset`].
pub enum SyncEpoch<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
{
    /// num_workers=0: process batches on the calling thread, zero allocation.
    /// Concrete types + monomorphized `process_batch` — no virtual dispatch.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        dataset: Arc<D>,
        transform: Arc<T>,
        collator: Arc<C>,
        item_concurrency: usize,
        pool: Option<Arc<rayon::ThreadPool>>,
        /// The process that started the epoch: item lanes run on the pool,
        /// which has no threads in a `fork()`ed child.
        pid: u32,
    },
    /// num_workers>0: batch tasks on the loader's rayon pool.
    Parallel(ParallelCore<D, T, C>),
}

impl<D, T, C> SyncEpoch<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
{
    pub(crate) fn start(
        dataset: &Arc<D>,
        transform: &Arc<T>,
        collator: &Arc<C>,
        pool: Option<&Arc<rayon::ThreadPool>>,
        settings: EpochSettings,
        chunks: Vec<Vec<usize>>,
    ) -> Self {
        let EpochSettings {
            num_workers,
            item_concurrency,
            prefetch_depth,
        } = settings;
        match pool {
            Some(pool) if num_workers > 0 => Self::Parallel(ParallelCore::new(
                dataset,
                transform,
                collator,
                pool,
                item_concurrency,
                num_workers,
                prefetch_depth,
                chunks,
            )),
            _ => Self::Direct {
                chunks: chunks.into_iter(),
                dataset: Arc::clone(dataset),
                transform: Arc::clone(transform),
                collator: Arc::clone(collator),
                item_concurrency,
                pool: pool.cloned(),
                pid: std::process::id(),
            },
        }
    }
}

impl<D, T, C> BatchStream<C::Batch> for SyncEpoch<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
{
    fn next(&mut self) -> Option<Result<C::Batch>> {
        match self {
            Self::Direct {
                chunks,
                dataset,
                transform,
                collator,
                item_concurrency,
                pool,
                pid,
            } => {
                let indices = chunks.next()?;
                if pool.is_some() && *pid != std::process::id() {
                    *chunks = Vec::new().into_iter();
                    return Some(Err(forked()));
                }
                Some(process_batch(
                    &**dataset,
                    &**transform,
                    &indices,
                    &**collator,
                    *item_concurrency,
                    pool.as_deref(),
                ))
            }
            Self::Parallel(core) => core.next(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Direct { chunks, .. } => chunks.len(),
            Self::Parallel(core) => core.len(),
        }
    }

    /// The direct path fetches the batch itself: it never has one ready, and
    /// given a deadline it does the work, which no deadline can cut short.
    #[cfg(feature = "python")]
    fn next_until(&mut self, deadline: Option<Instant>) -> Option<Option<Result<C::Batch>>> {
        match self {
            Self::Parallel(core) => core.next_until(deadline),
            Self::Direct { .. } => deadline.map(|_| self.next()),
        }
    }
}
