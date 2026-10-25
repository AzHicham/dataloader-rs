use std::sync::Arc;

use crossbeam_channel::{Receiver, unbounded};

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
    result_rx: Receiver<WorkerMsg<C::Batch>>,
    /// Results waiting to be returned in epoch order: batch `i` waits in slot
    /// `i % window`. Batches in flight or waiting always lie in
    /// `next_out..next_out + window` (see [`Epoch`]), so no two share a slot.
    reorder: Vec<Option<Result<C::Batch>>>,
    next_out: usize,
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
        // Batches in flight: one per worker plus the prefetch depth.
        let window = num_workers + prefetch_depth;
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
        }
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
            match self.result_rx.recv() {
                Ok(msg) => self.store(msg),
                Err(_) => return None,
            }
        }
    }

    /// The next batch if it has already arrived; `None` if `next` would wait.
    #[cfg(feature = "python")]
    fn try_next(&mut self) -> Option<Option<Result<C::Batch>>> {
        if self.next_out == self.epoch.len() {
            return Some(None);
        }
        let window = self.reorder.len();
        while self.reorder[self.next_out % window].is_none() {
            self.store(self.result_rx.try_recv().ok()?);
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
            },
        };
        Self { inner }
    }

    /// The next item if it is ready without waiting: `None` if `next` would
    /// wait for a worker, or do the work itself (direct path). Lets the
    /// Python binding keep the interpreter when a batch is already there.
    #[cfg(feature = "python")]
    pub(crate) fn try_next(&mut self) -> Option<Option<Result<C::Batch>>> {
        match &mut self.inner {
            Inner::Parallel(core) => core.try_next(),
            Inner::Direct { .. } => None,
        }
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
            } => {
                let indices = chunks.next()?;
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
