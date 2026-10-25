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
    loader::worker::{Epoch, WorkerMsg, process_batch},
};

/// The error of an epoch used in a process `fork()`ed after it started: the
/// pool's threads do not exist there, so its batches would never arrive.
fn forked() -> Error {
    "this epoch was started before fork(); start a new one in this process (iter())".into()
}

// ── ParallelCore ──────────────────────────────────────────────────────────────

// Batch tasks run on the loader's rayon pool and share the dataset and
// collator through `Arc` clones, so the iterator owns everything they touch
// and does not borrow the loader. Dropping (or leaking) the iterator never
// waits on them: a task in flight finishes its batch and exits.

struct ParallelCore<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    epoch: Arc<Epoch<D, C>>,
    result_rx: Receiver<WorkerMsg<C::Batch>>,
    /// Results waiting to be returned in epoch order: batch `i` waits in slot
    /// `i % window`. Batches in flight or waiting always lie in
    /// `next_out..next_out + window` (see [`Epoch`]), so no two share a slot.
    reorder: Vec<Option<Result<C::Batch>>>,
    next_out: usize,
    /// The process that started the epoch (see [`forked`]).
    pid: u32,
}

impl<D, C> ParallelCore<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    fn new(
        dataset: &Arc<D>,
        collator: &Arc<C>,
        pool: &Arc<rayon::ThreadPool>,
        items_in_parallel: usize,
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
            Arc::clone(collator),
            items_in_parallel,
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

impl<D, C> Drop for ParallelCore<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    fn drop(&mut self) {
        // Tasks stop claiming; one already processing finishes its batch and
        // fails to send once the receiver is gone.
        self.epoch.cancel();
    }
}

// ── DataLoaderIter ────────────────────────────────────────────────────────────

enum Inner<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    /// num_workers=0: process batches on the calling thread, zero allocation.
    /// Concrete types + monomorphized `process_batch` — no virtual dispatch.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        dataset: Arc<D>,
        collator: Arc<C>,
        items_in_parallel: usize,
        pool: Option<Arc<rayon::ThreadPool>>,
        /// The process that started the epoch: item lanes run on the pool,
        /// which has no threads in a `fork()`ed child.
        pid: u32,
    },
    /// num_workers>0: batch tasks on the loader's rayon pool.
    Parallel(ParallelCore<D, C>),
}

/// Iterator over one epoch of a [`DataLoader`](crate::DataLoader).
///
/// Owns shared handles to the dataset and collator, so it does not borrow the
/// loader and can be stored, sent to another thread, or outlive it.
pub struct DataLoaderIter<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    inner: Inner<D, C>,
}

impl<D, C> DataLoaderIter<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    pub(super) fn new(
        dataset: &Arc<D>,
        collator: &Arc<C>,
        pool: Option<&Arc<rayon::ThreadPool>>,
        items_in_parallel: usize,
        num_workers: usize,
        prefetch_depth: usize,
        chunks: Vec<Vec<usize>>,
    ) -> Self {
        let inner = match pool {
            Some(pool) if num_workers > 0 => Inner::Parallel(ParallelCore::new(
                dataset,
                collator,
                pool,
                items_in_parallel,
                num_workers,
                prefetch_depth,
                chunks,
            )),
            _ => Inner::Direct {
                chunks: chunks.into_iter(),
                dataset: Arc::clone(dataset),
                collator: Arc::clone(collator),
                items_in_parallel,
                pool: pool.cloned(),
                pid: std::process::id(),
            },
        };
        Self { inner }
    }

    /// The next item if it is ready by `deadline`, or (`None`) right away;
    /// `None` if not. Lets the Python binding keep the interpreter when a
    /// batch is already there, and wait in slices that signals can end.
    ///
    /// The direct path fetches the batch itself: it never has one ready, and
    /// given a deadline it does the work, which no deadline can cut short.
    #[cfg(feature = "python")]
    pub(crate) fn next_until(
        &mut self,
        deadline: Option<Instant>,
    ) -> Option<Option<Result<C::Batch>>> {
        if let Inner::Parallel(core) = &mut self.inner {
            return core.next_until(deadline);
        }
        deadline.map(|_| self.next())
    }
}

impl<D, C> Iterator for DataLoaderIter<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    type Item = Result<C::Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            Inner::Direct {
                chunks,
                dataset,
                collator,
                items_in_parallel,
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
                    &indices,
                    &**collator,
                    *items_in_parallel,
                    pool.as_deref(),
                ))
            }
            Inner::Parallel(core) => core.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.inner {
            Inner::Direct { chunks, .. } => chunks.len(),
            Inner::Parallel(core) => core.len(),
        };
        (n, Some(n))
    }
}

impl<D, C> ExactSizeIterator for DataLoaderIter<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
}
