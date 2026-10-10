use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crossbeam_channel::{Receiver, Sender};
use rayon::prelude::*;

use crate::{collator::Collator, dataset::Dataset, error::Result};

/// A single unit of work dispatched to a worker thread.
pub(super) struct WorkItem {
    /// Identifies this batch's position in the epoch so the consumer can
    /// reassemble results in the original order.
    pub(super) batch_idx: usize,
    pub(super) indices: Vec<usize>,
}

/// Limits how far workers run ahead of the consumer.
///
/// Batches are queued in epoch order, but the consumer returns them strictly
/// in order, so results of fast batches pile up in its reorder buffer while
/// it waits on a slow one. A worker may only start batch `i` once
/// `i < next_out + window`, which caps that buffer at `window` batches.
///
/// In steady state the gate stays open and costs one atomic load per batch;
/// workers only park on the condvar while some batch is far behind.
pub(super) struct Gate {
    next_out: AtomicUsize,
    window: usize,
    cancelled: AtomicBool,
    /// Workers currently parked; lets `advance` skip the lock when zero.
    waiters: AtomicUsize,
    lock: Mutex<()>,
    cvar: Condvar,
}

impl Gate {
    pub(super) fn new(window: usize) -> Self {
        debug_assert!(window >= 2, "hysteresis in `advance` needs window >= 2");
        Self {
            next_out: AtomicUsize::new(0),
            window,
            cancelled: AtomicBool::new(false),
            waiters: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cvar: Condvar::new(),
        }
    }

    fn is_open(&self, batch_idx: usize) -> bool {
        // SeqCst pairs with `advance`: either the consumer sees our waiter
        // count and notifies, or we see its new `next_out`.
        batch_idx < self.next_out.load(Ordering::SeqCst) + self.window
            || self.cancelled.load(Ordering::SeqCst)
    }

    /// Block until batch `batch_idx` may start. Returns `false` if cancelled.
    fn wait_for(&self, batch_idx: usize) -> bool {
        if !self.is_open(batch_idx) {
            let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.waiters.fetch_add(1, Ordering::SeqCst);
            while !self.is_open(batch_idx) {
                guard = self.cvar.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
            self.waiters.fetch_sub(1, Ordering::SeqCst);
        }
        !self.cancelled.load(Ordering::Acquire)
    }

    /// Called by the consumer after returning batch `next_out - 1`.
    ///
    /// Parked workers are woken only every `window / 2` batches, so each
    /// wake-up buys them a run of work instead of a single batch. This is
    /// still in time: a parked batch `i` opens at `next_out = i - window + 1`
    /// but is only needed at `next_out = i`, and any `window - 1` consecutive
    /// values of `next_out` include a multiple of `window / 2`.
    pub(super) fn advance(&self, next_out: usize) {
        self.next_out.store(next_out, Ordering::SeqCst);
        if next_out % (self.window / 2) == 0 && self.waiters.load(Ordering::SeqCst) > 0 {
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.cvar.notify_all();
        }
    }

    /// Stop all workers, including parked ones.
    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.cvar.notify_all();
    }
}

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

/// Per-worker loop: drain the work queue, process each batch, send results.
pub(super) fn worker_loop<D, C>(
    dataset: &D,
    collator: &C,
    work_rx: Receiver<WorkItem>,
    result_tx: Sender<(usize, Result<C::Batch>)>,
    pool: Option<Arc<rayon::ThreadPool>>,
    gate: Arc<Gate>,
) where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send,
{
    while let Ok(item) = work_rx.recv() {
        if !gate.wait_for(item.batch_idx) {
            break;
        }
        let result = process_batch(dataset, &item.indices, collator, pool.as_deref());
        if result_tx.send((item.batch_idx, result)).is_err() {
            break;
        }
    }
}
