use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, bounded, unbounded};
use hashbrown::HashMap;

use crate::{
    collator::Collator,
    dataset::Dataset,
    error::Result,
    loader::worker::{Gate, WorkItem, process_batch, worker_loop},
};

// ── ParallelCore ──────────────────────────────────────────────────────────────

// Worker threads share the dataset and collator through `Arc` clones, so the
// iterator owns everything its workers touch and does not borrow the loader.
// Leaking the iterator (e.g. `mem::forget`) only leaks threads; it can never
// leave a worker reading freed memory.

struct ParallelCore<B> {
    result_rx: Option<Receiver<(usize, Result<B>)>>,
    /// Out-of-order results waiting to be returned in epoch order; bounded
    /// by the gate's window.
    reorder: HashMap<usize, Result<B>>,
    next_out: usize,
    remaining: usize,
    handles: Vec<JoinHandle<()>>,
    gate: Arc<Gate>,
}

impl<B: Send + 'static> ParallelCore<B> {
    fn spawn<D, C>(
        dataset: &Arc<D>,
        collator: &Arc<C>,
        pool: Option<&Arc<rayon::ThreadPool>>,
        num_workers: usize,
        prefetch_depth: usize,
        chunks: Vec<Vec<usize>>,
    ) -> Self
    where
        D: Dataset,
        C: Collator<D::Item, Batch = B>,
    {
        let n_batches = chunks.len();
        let (work_tx, work_rx) = unbounded::<WorkItem>();
        let (result_tx, result_rx) = bounded(prefetch_depth);

        for (batch_idx, indices) in chunks.into_iter().enumerate() {
            let _ = work_tx.send(WorkItem { batch_idx, indices });
        }
        drop(work_tx);

        // Twice the steady-state pipeline (one batch per worker plus the
        // prefetch depth): wide enough that the gate stays open unless one
        // batch falls far behind.
        let gate = Arc::new(Gate::new(2 * (num_workers + prefetch_depth)));

        let handles = (0..num_workers)
            .map(|_| {
                let dataset = Arc::clone(dataset);
                let collator = Arc::clone(collator);
                let work_rx = work_rx.clone();
                let result_tx = result_tx.clone();
                let pool = pool.cloned();
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    worker_loop(&*dataset, &*collator, work_rx, result_tx, pool, gate);
                })
            })
            .collect();

        drop(result_tx);

        Self {
            result_rx: Some(result_rx),
            reorder: HashMap::new(),
            next_out: 0,
            remaining: n_batches,
            handles,
            gate,
        }
    }

    fn next(&mut self) -> Option<Result<B>> {
        if self.remaining == 0 {
            return None;
        }
        loop {
            if let Some(batch) = self.reorder.remove(&self.next_out) {
                self.next_out += 1;
                self.remaining -= 1;
                self.gate.advance(self.next_out);
                return Some(batch);
            }
            match self.result_rx.as_ref()?.recv() {
                Ok((idx, batch)) => {
                    self.reorder.insert(idx, batch);
                }
                Err(_) => return None,
            }
        }
    }

    fn len(&self) -> usize {
        self.remaining
    }
}

impl<B> Drop for ParallelCore<B> {
    fn drop(&mut self) {
        self.gate.cancel();
        // Drop receiver so workers blocked on send() get Err and exit.
        drop(self.result_rx.take());
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
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
    /// inter_workers>0: N workers, optional rayon intra-batch pool.
    Parallel(ParallelCore<C::Batch>),
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
        num_workers: usize,
        prefetch_depth: usize,
        chunks: Vec<Vec<usize>>,
    ) -> Self {
        let inner = if num_workers == 0 {
            Inner::Direct {
                remaining: chunks.len(),
                chunks: chunks.into_iter(),
                dataset: Arc::clone(dataset),
                collator: Arc::clone(collator),
                pool: pool.cloned(),
            }
        } else {
            Inner::Parallel(ParallelCore::spawn(
                dataset,
                collator,
                pool,
                num_workers,
                prefetch_depth,
                chunks,
            ))
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
