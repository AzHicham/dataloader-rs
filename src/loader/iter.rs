use std::sync::Arc;

use crossbeam_channel::{Receiver, unbounded};
use hashbrown::HashMap;

use crate::{
    collator::Collator,
    dataset::Dataset,
    error::Result,
    loader::worker::{Epoch, WorkerMsg, process_batch},
};

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
    pool: Arc<rayon::ThreadPool>,
    result_rx: Receiver<WorkerMsg<C::Batch>>,
    /// Out-of-order results waiting to be returned in epoch order; at most
    /// `window` batches, see [`Epoch`].
    reorder: HashMap<usize, Result<C::Batch>>,
    next_out: usize,
}

impl<D, C> ParallelCore<D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    fn spawn(
        dataset: &Arc<D>,
        collator: &Arc<C>,
        pool: &Arc<rayon::ThreadPool>,
        parallel_items: bool,
        num_workers: usize,
        prefetch_depth: usize,
        chunks: Vec<Vec<usize>>,
    ) -> Self {
        let (result_tx, result_rx) = unbounded();
        let epoch = Arc::new(Epoch::new(
            Arc::clone(dataset),
            Arc::clone(collator),
            parallel_items.then(|| Arc::clone(pool)),
            chunks,
            // Batches in flight: one per worker plus the prefetch depth.
            num_workers + prefetch_depth,
            num_workers,
            result_tx,
        ));
        epoch.spawn_tasks(pool);
        Self {
            epoch,
            pool: Arc::clone(pool),
            result_rx,
            reorder: HashMap::new(),
            next_out: 0,
        }
    }

    fn next(&mut self) -> Option<Result<C::Batch>> {
        if self.next_out == self.epoch.len() {
            return None;
        }
        loop {
            if let Some(batch) = self.reorder.remove(&self.next_out) {
                self.next_out += 1;
                self.epoch.advance(self.next_out, &self.pool);
                return Some(batch);
            }
            match self.result_rx.recv() {
                Ok((idx, Ok(batch))) => {
                    self.reorder.insert(idx, batch);
                }
                // A task panicked: re-raise on the consumer thread, as
                // `num_workers(0)` would. `Drop` stops the other tasks.
                Ok((_, Err(payload))) => {
                    self.next_out = self.epoch.len();
                    std::panic::resume_unwind(payload);
                }
                Err(_) => return None,
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
    /// inter_workers=0: process batches on the calling thread, zero allocation.
    /// Concrete types + monomorphized `process_batch` — no virtual dispatch.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        remaining: usize,
        dataset: Arc<D>,
        collator: Arc<C>,
        pool: Option<Arc<rayon::ThreadPool>>,
    },
    /// inter_workers>0: batch tasks on the loader's rayon pool.
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
        parallel_items: bool,
        num_workers: usize,
        prefetch_depth: usize,
        chunks: Vec<Vec<usize>>,
    ) -> Self {
        let inner = match pool {
            Some(pool) if num_workers > 0 => Inner::Parallel(ParallelCore::spawn(
                dataset,
                collator,
                pool,
                parallel_items,
                num_workers,
                prefetch_depth,
                chunks,
            )),
            _ => Inner::Direct {
                remaining: chunks.len(),
                chunks: chunks.into_iter(),
                dataset: Arc::clone(dataset),
                collator: Arc::clone(collator),
                pool: pool.filter(|_| parallel_items).cloned(),
            },
        };
        Self { inner }
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
                remaining,
                dataset,
                collator,
                pool,
            } => {
                let indices = chunks.next()?;
                *remaining -= 1;
                Some(process_batch(
                    &**dataset,
                    &indices,
                    &**collator,
                    pool.as_deref(),
                ))
            }
            Inner::Parallel(core) => core.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.inner {
            Inner::Direct { remaining, .. } => *remaining,
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
