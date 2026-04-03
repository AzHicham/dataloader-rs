use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crossbeam_channel::Sender;
use rayon::prelude::*;

use crate::{collator::Collator, dataset::Dataset, error::Result};

pub(super) fn prefetch_loop<D, C>(
    dataset: &D,
    chunks: Vec<Vec<usize>>,
    collator: &C,
    tx: Sender<Result<C::Batch>>,
    pool: Option<Arc<rayon::ThreadPool>>,
    cancel: Arc<AtomicBool>,
) where
    D: Dataset,
    C: Collator<D::Item>,
{
    for batch_indices in chunks {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let result = if let Some(ref p) = pool {
            p.install(|| fetch_batch(dataset, &batch_indices, collator, true))
        } else {
            fetch_batch(dataset, &batch_indices, collator, false)
        };

        if tx.send(result).is_err() {
            break;
        }
    }
}

fn fetch_batch<D, C>(dataset: &D, indices: &[usize], collator: &C, parallel: bool) -> Result<C::Batch>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    let items: Result<Vec<D::Item>> = if parallel {
        indices.par_iter().map(|&i| dataset.get(i)).collect()
    } else {
        dataset.get_batch(indices)
    };

    items.and_then(|v| collator.collate(v))
}
