use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
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

/// Message sent from a worker to the consumer: the batch position plus either
/// the batch result or, if the dataset/collator panicked, the panic payload.
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
///
/// A panic in the dataset or collator is caught and forwarded to the
/// consumer, which re-raises it; the worker then stops.
pub(super) fn worker_loop<D, C>(
    dataset: &D,
    collator: &C,
    work_rx: Receiver<WorkItem>,
    result_tx: Sender<WorkerMsg<C::Batch>>,
    pool: Option<Arc<rayon::ThreadPool>>,
    cancel: Arc<AtomicBool>,
) where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send,
{
    while let Ok(item) = work_rx.recv() {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            process_batch(dataset, &item.indices, collator, pool.as_deref())
        }));
        let panicked = result.is_err();
        if result_tx.send((item.batch_idx, result)).is_err() || panicked {
            break;
        }
    }
}
