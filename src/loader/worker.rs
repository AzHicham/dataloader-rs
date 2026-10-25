use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crossbeam_channel::Sender;

use crate::{collator::Collator, dataset::Dataset, error::Result};

/// Message sent from a batch task to the consumer: the batch position plus
/// either the batch result or, if the dataset/collator panicked, the panic
/// payload.
pub(super) type WorkerMsg<B> = (usize, std::thread::Result<Result<B>>);

/// Fetch the items of one batch, then collate them.
///
/// With `items_in_parallel == 1` the batch is one `Dataset::get_batch` call;
/// otherwise each item is one `Dataset::get` call, made by lanes on `pool`
/// (see [`fetch_in_lanes`]).
pub(super) fn process_batch<D, C>(
    dataset: &D,
    indices: &[usize],
    collator: &C,
    items_in_parallel: usize,
    pool: Option<&rayon::ThreadPool>,
) -> Result<C::Batch>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    let items = match (pool, indices) {
        // A single item needs no lane, but is still fetched with `get`.
        (Some(_), &[index]) if items_in_parallel > 1 => vec![dataset.get(index)?],
        (Some(pool), _) if items_in_parallel > 1 => {
            fetch_in_lanes(dataset, indices, items_in_parallel, pool)?
        }
        _ => dataset.get_batch(indices)?,
    };
    collator.collate(items)
}

/// Fetch `indices` with at most `lanes` concurrent `Dataset::get` calls.
///
/// Each lane takes the batch's next item until none is left, so a fast lane
/// takes more items and uneven items still balance, while the batch never has
/// more than `lanes` items in flight. After an error, lanes stop taking items
/// and the first error by position is returned. A panic in a lane is resumed
/// here once every lane has finished.
fn fetch_in_lanes<D: Dataset>(
    dataset: &D,
    indices: &[usize],
    lanes: usize,
    pool: &rayon::ThreadPool,
) -> Result<Vec<D::Item>> {
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<Result<D::Item>>>> =
        indices.iter().map(|_| Mutex::new(None)).collect();
    pool.scope(|scope| {
        for _ in 0..lanes.min(indices.len()) {
            scope.spawn(|_| {
                while !failed.load(Ordering::Relaxed) {
                    let position = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&index) = indices.get(position) else {
                        break;
                    };
                    let item = dataset.get(index);
                    if item.is_err() {
                        failed.store(true, Ordering::Relaxed);
                    }
                    *slots[position]
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Some(item);
                }
            });
        }
    });
    let mut items = Vec::with_capacity(indices.len());
    for slot in slots {
        match slot.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Some(Ok(item)) => items.push(item),
            Some(Err(e)) => return Err(e),
            // Not fetched: a lane failed first, and its error comes later.
            None => {}
        }
    }
    Ok(items)
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
    /// Items of one batch fetched in parallel, on `pool`.
    items_in_parallel: usize,
    pool: Arc<rayon::ThreadPool>,
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
        items_in_parallel: usize,
        pool: Arc<rayon::ThreadPool>,
        chunks: Vec<Vec<usize>>,
        window: usize,
        max_tasks: usize,
        result_tx: Sender<WorkerMsg<C::Batch>>,
    ) -> Self {
        Self {
            dataset,
            collator,
            items_in_parallel,
            pool,
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
        self.next_claim
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |claim| {
                self.claimable(claim).then_some(claim + 1)
            })
            .ok()
    }

    /// Reserve a task if a batch is claimable and fewer than `max_tasks` run.
    fn reserve_task(&self) -> bool {
        self.claimable(self.next_claim.load(Ordering::SeqCst))
            && self
                .tasks
                .try_update(Ordering::SeqCst, Ordering::SeqCst, |running| {
                    (running < self.max_tasks).then_some(running + 1)
                })
                .is_ok()
    }

    /// Start batch tasks until `max_tasks` run or nothing is claimable.
    /// Costs a few atomic loads when all tasks are running.
    pub(super) fn spawn_tasks(self: &Arc<Self>) {
        while self.reserve_task() {
            let epoch = Arc::clone(self);
            self.pool.spawn(move || epoch.run());
        }
    }

    /// Publish the consumer's position, opening the window for more batches.
    pub(super) fn advance(self: &Arc<Self>, next_out: usize) {
        self.next_out.store(next_out, Ordering::SeqCst);
        self.spawn_tasks();
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
                        self.items_in_parallel,
                        Some(&self.pool),
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
            if !self.reserve_task() {
                return;
            }
        }
    }
}
