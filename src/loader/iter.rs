use std::sync::Arc;
#[cfg(feature = "python")]
use std::time::Instant;

#[cfg(feature = "python")]
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::{Receiver, TryRecvError, unbounded};

use crate::{
    collator::Collator,
    dataset::Dataset,
    error::{Error, Result},
    loader::{
        source::{EpochSettings, LoaderSource},
        worker::{Epoch, WorkerMsg, process_batch},
    },
    transform::{Identity, Transform},
};

/// The error of an epoch used in a process `fork()`ed after it started: the
/// pool's threads do not exist there, so its batches would never arrive.
pub(super) fn forked() -> Error {
    "this epoch was started before fork(); start a new one in this process (iter())".into()
}

// ── ParallelCore ──────────────────────────────────────────────────────────────

// Batch tasks run on the loader's rayon pool and share the dataset,
// transform and collator through `Arc` clones, so the iterator owns everything they touch
// and does not borrow the loader. Dropping (or leaking) the iterator never
// waits on them: a task in flight finishes its batch and exits.

pub struct ParallelCore<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
{
    epoch: Arc<Epoch<D, T, C>>,
    result_rx: Receiver<WorkerMsg<C::Batch>>,
    /// Results waiting to be returned in epoch order: batch `i` waits in slot
    /// `i % window`. Batches in flight or waiting always lie in
    /// `next_out..next_out + window` (see [`Epoch`]), so no two share a slot.
    reorder: Vec<Option<Result<C::Batch>>>,
    next_out: usize,
    /// The process that started the epoch (see [`forked`]).
    pid: u32,
}

impl<D, T, C> ParallelCore<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
    C::Batch: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset: &Arc<D>,
        transform: &Arc<T>,
        collator: &Arc<C>,
        pool: &Arc<rayon::ThreadPool>,
        item_concurrency: usize,
        num_workers: usize,
        prefetch_depth: usize,
        chunks: Vec<Vec<usize>>,
    ) -> Self {
        let (result_tx, result_rx) = unbounded();
        // Batches in flight: one per worker plus the prefetch depth, and no
        // more than the epoch has, so a huge prefetch_depth allocates nothing.
        let window = num_workers
            .saturating_add(prefetch_depth)
            .min(chunks.len())
            .max(1);
        let epoch = Arc::new(Epoch::new(
            Arc::clone(dataset),
            Arc::clone(transform),
            Arc::clone(collator),
            item_concurrency,
            Arc::clone(pool),
            chunks,
            window,
            num_workers,
            result_tx,
        ));
        epoch.spawn_tasks();
        Self {
            epoch,
            result_rx,
            reorder: std::iter::repeat_with(|| None).take(window).collect(),
            next_out: 0,
            pid: std::process::id(),
        }
    }

    /// End the epoch with [`forked`] if this process is not the one that
    /// started it. Checked only when about to wait, where a `getpid` costs
    /// nothing next to the wait.
    fn check_forked(&mut self) -> Result<()> {
        if self.pid == std::process::id() {
            return Ok(());
        }
        self.next_out = self.epoch.len();
        Err(forked())
    }

    fn next(&mut self) -> Option<Result<C::Batch>> {
        if self.next_out == self.epoch.len() {
            return None;
        }
        let window = self.reorder.len();
        loop {
            if let Some(batch) = self.reorder[self.next_out % window].take() {
                self.next_out += 1;
                self.epoch.advance(self.next_out);
                return Some(batch);
            }
            let msg = match self.result_rx.try_recv() {
                Ok(msg) => msg,
                // About to wait: first make sure a batch can still arrive.
                Err(TryRecvError::Empty) => {
                    if let Err(e) = self.check_forked() {
                        return Some(Err(e));
                    }
                    match self.result_rx.recv() {
                        Ok(msg) => msg,
                        Err(_) => return None,
                    }
                }
                Err(TryRecvError::Disconnected) => return None,
            };
            self.store(msg);
        }
    }

    /// The next batch if it arrives by `deadline`, or (`None`) if it has
    /// already arrived; `None` if not, so that the caller can wait again.
    #[cfg(feature = "python")]
    fn next_until(&mut self, deadline: Option<Instant>) -> Option<Option<Result<C::Batch>>> {
        if self.next_out == self.epoch.len() {
            return Some(None);
        }
        let window = self.reorder.len();
        while self.reorder[self.next_out % window].is_none() {
            let msg = match deadline {
                None => self.result_rx.try_recv().ok()?,
                Some(deadline) => match self.check_forked() {
                    Err(e) => return Some(Some(Err(e))),
                    Ok(()) => match self.result_rx.recv_deadline(deadline) {
                        Ok(msg) => msg,
                        Err(RecvTimeoutError::Timeout) => return None,
                        // The epoch holds a sender, so this does not happen;
                        // `next` ends the epoch in that case too.
                        Err(RecvTimeoutError::Disconnected) => return Some(None),
                    },
                },
            };
            self.store(msg);
        }
        Some(self.next())
    }

    /// File a task's result in its slot, or re-raise its panic.
    fn store(&mut self, (idx, result): WorkerMsg<C::Batch>) {
        match result {
            Ok(batch) => {
                let window = self.reorder.len();
                self.reorder[idx % window] = Some(batch);
            }
            // A task panicked: re-raise on the consumer thread, as
            // `num_workers(0)` would. `Drop` stops the other tasks.
            Err(payload) => {
                self.next_out = self.epoch.len();
                std::panic::resume_unwind(payload);
            }
        }
    }

    fn len(&self) -> usize {
        self.epoch.len() - self.next_out
    }
}

impl<D, T, C> Drop for ParallelCore<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
{
    fn drop(&mut self) {
        // Tasks stop claiming; one already processing finishes its batch and
        // fails to send once the receiver is gone.
        self.epoch.cancel();
    }
}

// ── SyncEpoch ─────────────────────────────────────────────────────────────────

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
    pub(super) fn start(
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
