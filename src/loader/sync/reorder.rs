//! [`ParallelCore`]: batch tasks on the loader's pool, their results put back in epoch order.

use std::convert::Infallible;
use std::sync::Arc;
#[cfg(feature = "python")]
use std::time::{Duration, Instant};

#[cfg(feature = "python")]
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, unbounded};

use super::worker::{Epoch, WorkerMsg};
use crate::{
    collator::Collator, dataset::Dataset, error::Result, loader::iter::forked, transform::Transform,
};

// Batch tasks run on the loader's rayon pool and share the dataset,
// transform and collator through `Arc` clones, so the iterator owns everything they touch
// and does not borrow the loader. Dropping (or leaking) the iterator never
// waits on them: a task in flight finishes its batch and exits. The Python
// binding waits for them first (`stop`), so that the Python objects they hold
// are released before the iterator is gone.

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
    /// Cloned into each batch task; `None` once the epoch is stopped.
    alive: Option<Sender<Infallible>>,
    /// Disconnected once every batch task has exited.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    exited: Receiver<Infallible>,
}

impl<D, T, C> ParallelCore<D, T, C>
where
    D: Dataset,
    T: Transform<D::Item>,
    C: Collator<T::Output>,
    C::Batch: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
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
        let (alive, exited) = bounded(0);
        epoch.spawn_tasks(&alive);
        Self {
            epoch,
            result_rx,
            reorder: std::iter::repeat_with(|| None).take(window).collect(),
            next_out: 0,
            pid: std::process::id(),
            alive: Some(alive),
            exited,
        }
    }

    /// End the epoch: no batch starts any more, and wait up to `timeout` for
    /// the tasks in flight to finish their batch and exit, so that none holds
    /// the dataset, transform or collator after this returns `true`.
    #[cfg(feature = "python")]
    pub(super) fn stop(&mut self, timeout: Duration) -> bool {
        self.epoch.cancel();
        self.next_out = self.epoch.len();
        self.alive = None;
        // A forked child has none of the parent's tasks: they never exit there.
        self.pid == std::process::id()
            && matches!(
                self.exited.recv_timeout(timeout),
                Err(RecvTimeoutError::Disconnected)
            )
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

    pub(super) fn next(&mut self) -> Option<Result<C::Batch>> {
        if self.next_out == self.epoch.len() {
            return None;
        }
        let window = self.reorder.len();
        loop {
            if let Some(batch) = self.reorder[self.next_out % window].take() {
                self.next_out += 1;
                if let Some(alive) = &self.alive {
                    self.epoch.advance(self.next_out, alive);
                }
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
    pub(super) fn next_until(
        &mut self,
        deadline: Option<Instant>,
    ) -> Option<Option<Result<C::Batch>>> {
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

    pub(super) fn len(&self) -> usize {
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
