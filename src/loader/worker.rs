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
/// # Safety
///
/// `dataset` and `collator` must remain valid for the entire duration of this
/// call. The caller is responsible for joining this thread before the values
/// pointed to are dropped.
pub(super) unsafe fn worker_loop<D, C>(
    dataset: *const D,
    collator: *const C,
    work_rx: Receiver<WorkItem>,
    result_tx: Sender<(usize, Result<C::Batch>)>,
    pool: Option<Arc<rayon::ThreadPool>>,
    cancel: Arc<AtomicBool>,
) where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send,
{
    // SAFETY: guaranteed by the DataLoaderIter / OwnedDataLoaderIter lifetime
    // contract — see their respective safety comments.
    let (dataset, collator) = unsafe { (&*dataset, &*collator) };

    while let Ok(item) = work_rx.recv() {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let result = process_batch(dataset, &item.indices, collator, pool.as_deref());
        if result_tx.send((item.batch_idx, result)).is_err() {
            break;
        }
    }
}
