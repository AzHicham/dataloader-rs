//! [`DataLoaderIter`]: one epoch of a loader, whatever its source.

#[cfg(feature = "python")]
use std::time::Instant;

use crate::{
    collator::Collator,
    error::{Error, Result},
    loader::source::LoaderSource,
    transform::{Identity, Transform},
};

/// The error of an epoch used in a process `fork()`ed after it started: the
/// pool's threads do not exist there, so its batches would never arrive.
pub(super) fn forked() -> Error {
    "this epoch was started before fork(); start a new one in this process (iter())".into()
}

/// What an epoch of a loader yields its batches through: [`SyncEpoch`] for a
/// [`Dataset`], `AsyncEpoch` for an `AsyncDataset`.
pub trait BatchStream<B> {
    /// The next batch in epoch order, `None` at the end of the epoch.
    fn next(&mut self) -> Option<Result<B>>;

    /// Batches left in the epoch.
    fn len(&self) -> usize;

    /// The next batch if it arrives by `deadline`, or (`None`) if it has
    /// already arrived; `None` if not, so that the caller can wait again.
    #[cfg(feature = "python")]
    fn next_until(&mut self, deadline: Option<Instant>) -> Option<Option<Result<B>>>;
}

// ── DataLoaderIter ────────────────────────────────────────────────────────────

/// Iterator over one epoch of a [`DataLoader`](crate::DataLoader).
///
/// Owns shared handles to the dataset, transform and collator, so it does not
/// borrow the loader and can be stored, sent to another thread, or outlive it.
pub struct DataLoaderIter<Src, C, T = Identity>
where
    Src: LoaderSource,
    T: Transform<Src::Item>,
    C: Collator<T::Output>,
{
    epoch: Src::Epoch<T, C, T::Output>,
}

impl<Src, C, T> DataLoaderIter<Src, C, T>
where
    Src: LoaderSource,
    T: Transform<Src::Item>,
    C: Collator<T::Output>,
{
    pub(super) fn new(epoch: Src::Epoch<T, C, T::Output>) -> Self {
        Self { epoch }
    }

    /// The next item if it is ready by `deadline`, or (`None`) right away;
    /// `None` if not. Lets the Python binding keep the interpreter when a
    /// batch is already there, and wait in slices that signals can end.
    #[cfg(feature = "python")]
    pub(crate) fn next_until(
        &mut self,
        deadline: Option<Instant>,
    ) -> Option<Option<Result<C::Batch>>> {
        self.epoch.next_until(deadline)
    }
}

impl<Src, C, T> Iterator for DataLoaderIter<Src, C, T>
where
    Src: LoaderSource,
    T: Transform<Src::Item>,
    C: Collator<T::Output>,
{
    type Item = Result<C::Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.epoch.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.epoch.len();
        (n, Some(n))
    }
}

impl<Src, C, T> ExactSizeIterator for DataLoaderIter<Src, C, T>
where
    Src: LoaderSource,
    T: Transform<Src::Item>,
    C: Collator<T::Output>,
{
}
