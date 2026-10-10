use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crossbeam_channel::Sender;
use rayon::prelude::*;

use crate::{collator::Collator, dataset::Dataset, error::Result};

/// Message sent from a batch task to the consumer: the batch position plus
/// either the batch result or, if the dataset/collator panicked, the panic
/// payload.
pub(super) type WorkerMsg<B> = (usize, std::thread::Result<Result<B>>);

/// Fetch items for one batch (optionally in parallel) then collate them.
pub(super) fn process_batch<D, C>(
    dataset: &D,
    indices: &[usize],
    collator: &C,
    pool: Option<&rayon::ThreadPool>,
) -> Result<C::Batch>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    let items = match pool {
        // From inside the pool `install` runs inline; either way the items
        // are spread over the same pool by work-stealing.
        Some(pool) => pool.install(|| {
            indices
                .par_iter()
                .map(|&i| dataset.get(i))
                .collect::<Result<Vec<_>>>()
        })?,
        None => dataset.get_batch(indices)?,
    };
    collator.collate(items)
}

/// State of one epoch, shared by the consumer and the batch tasks running on
/// the loader's rayon pool.
///
/// Batches are claimed in epoch order through `next_claim`, but only while
/// `batch < next_out + window`: at most `window` batches are ever in flight
/// or waiting in the consumer's reorder buffer, however slow one batch is.
/// A task that finds nothing to claim exits instead of blocking a pool
/// thread; the consumer starts tasks again when it advances.
pub(super) struct Epoch<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    dataset: Arc<D>,
    collator: Arc<C>,
    /// Pool for item-level parallelism inside a batch, if enabled.
    item_pool: Option<Arc<rayon::ThreadPool>>,
    chunks: Vec<Vec<usize>>,
    next_claim: AtomicUsize,
    /// Mirror of the consumer's position, published after each batch.
    next_out: AtomicUsize,
    window: usize,
    /// Running batch tasks, at most `max_tasks`.
    tasks: AtomicUsize,
    max_tasks: usize,
    cancelled: AtomicBool,
    result_tx: Sender<WorkerMsg<C::Batch>>,
}

impl<D, C> Epoch<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        dataset: Arc<D>,
        collator: Arc<C>,
        item_pool: Option<Arc<rayon::ThreadPool>>,
        chunks: Vec<Vec<usize>>,
        window: usize,
        max_tasks: usize,
        result_tx: Sender<WorkerMsg<C::Batch>>,
    ) -> Self {
        Self {
            dataset,
            collator,
            item_pool,
            chunks,
            next_claim: AtomicUsize::new(0),
            next_out: AtomicUsize::new(0),
            window,
            tasks: AtomicUsize::new(0),
            max_tasks,
            cancelled: AtomicBool::new(false),
            result_tx,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the next unclaimed batch exists and is inside the window.
    fn claimable(&self, claim: usize) -> bool {
        // SeqCst pairs `advance`/`spawn_tasks` with a task's exit path (see
        // `run`): either the consumer sees the task gone and starts another,
        // or the task sees the new position and keeps going.
        !self.cancelled.load(Ordering::SeqCst)
            && claim < self.chunks.len()
            && claim < self.next_out.load(Ordering::SeqCst) + self.window
    }

    fn claim(&self) -> Option<usize> {
        let mut claim = self.next_claim.load(Ordering::SeqCst);
        loop {
            if !self.claimable(claim) {
                return None;
            }
            match self.next_claim.compare_exchange_weak(
                claim,
                claim + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return Some(claim),
                Err(current) => claim = current,
            }
        }
    }

    fn try_reserve_task(&self) -> bool {
        self.tasks
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n < self.max_tasks).then_some(n + 1)
            })
            .is_ok()
    }

    /// Start batch tasks on `pool` until `max_tasks` run or nothing is
    /// claimable. Costs a few atomic loads when all tasks are running.
    pub(super) fn spawn_tasks(self: &Arc<Self>, pool: &rayon::ThreadPool) {
        while self.claimable(self.next_claim.load(Ordering::SeqCst)) && self.try_reserve_task() {
            let epoch = Arc::clone(self);
            pool.spawn(move || epoch.run());
        }
    }

    /// Publish the consumer's position, opening the window for more batches.
    pub(super) fn advance(self: &Arc<Self>, next_out: usize, pool: &rayon::ThreadPool) {
        self.next_out.store(next_out, Ordering::SeqCst);
        self.spawn_tasks(pool);
    }

    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// A batch task: process claimable batches until none is left or the
    /// window is full, then exit.
    fn run(&self) {
        loop {
            while let Some(batch_idx) = self.claim() {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    process_batch(
                        &*self.dataset,
                        &self.chunks[batch_idx],
                        &*self.collator,
                        self.item_pool.as_deref(),
                    )
                }));
                // A panic is forwarded for the consumer to re-raise (rayon
                // would otherwise abort the process); stop claiming after it.
                let panicked = result.is_err();
                if self.result_tx.send((batch_idx, result)).is_err() || panicked {
                    self.cancel();
                }
            }
            self.tasks.fetch_sub(1, Ordering::SeqCst);
            // The consumer may have advanced after our last failed claim but
            // before we left; if so, it may also have seen us still running
            // and not started a replacement. Re-check and resume.
            if !(self.claimable(self.next_claim.load(Ordering::SeqCst)) && self.try_reserve_task())
            {
                return;
            }
        }
    }
}
