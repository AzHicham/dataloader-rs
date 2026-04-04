use std::marker::PhantomData;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, bounded, unbounded};
use hashbrown::HashMap;

use crate::{
    collator::Collator,
    dataset::Dataset,
    error::Result,
    loader::{
        core::DataLoader,
        worker::{WorkItem, process_batch, worker_loop},
    },
    sampler::Sampler,
};

// ── ParallelCore ──────────────────────────────────────────────────────────────

// Safety contract (shared by DataLoaderIter<'a> and OwnedDataLoaderIter):
//
//   Worker threads receive raw pointers into the parent DataLoader's dataset
//   and collator.  The caller guarantees:
//
//   • DataLoaderIter<'a>  — the mutable borrow lifetime 'a prevents any use
//     of the loader while the iterator is live; Drop joins threads before 'a
//     ends.
//   • OwnedDataLoaderIter — the Python layer stores a Py<PyDataloader> strong
//     ref in PyDataloaderIter::_owner; Drop joins threads before that ref can
//     be released.

struct ParallelCore<B> {
    result_rx: Option<Receiver<(usize, Result<B>)>>,
    /// Out-of-order results waiting to be returned in epoch order.
    reorder: HashMap<usize, Result<B>>,
    next_out: usize,
    remaining: usize,
    handles: Vec<JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
}

impl<B: Send + 'static> ParallelCore<B> {
    fn spawn<D, S, C>(loader: &mut DataLoader<D, S, C>, chunks: Vec<Vec<usize>>) -> Self
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item, Batch = B>,
    {
        let n_batches = chunks.len();
        let (work_tx, work_rx) = unbounded::<WorkItem>();
        let (result_tx, result_rx) = bounded(loader.prefetch_depth);

        for (batch_idx, indices) in chunks.into_iter().enumerate() {
            let _ = work_tx.send(WorkItem { batch_idx, indices });
        }
        drop(work_tx);

        let cancel = Arc::new(AtomicBool::new(false));
        let dataset_ptr = &loader.dataset as *const D as usize;
        let collator_ptr = &loader.collator as *const C as usize;

        let handles = (0..loader.inter_workers)
            .map(|_| {
                let work_rx = work_rx.clone();
                let result_tx = result_tx.clone();
                let pool = loader.pool.clone();
                let cancel = Arc::clone(&cancel);
                std::thread::spawn(move || unsafe {
                    worker_loop(
                        dataset_ptr as *const D,
                        collator_ptr as *const C,
                        work_rx,
                        result_tx,
                        pool,
                        cancel,
                    );
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
            cancel,
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
        self.cancel.store(true, Ordering::Release);
        // Drop receiver so workers blocked on send() get Err and exit.
        drop(self.result_rx.take());
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ── DataLoaderIter ────────────────────────────────────────────────────────────

enum Inner<'a, D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    /// inter_workers=0: process batches on the calling thread, zero allocation.
    /// Concrete references + monomorphized `process_batch` — no virtual dispatch.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        remaining: usize,
        dataset: &'a D,
        collator: &'a C,
        pool: Option<Arc<rayon::ThreadPool>>,
    },
    /// inter_workers>0: N workers, optional rayon intra-batch pool.
    Parallel(ParallelCore<C::Batch>),
}

pub struct DataLoaderIter<'a, D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
{
    inner: Inner<'a, D, C>,
    _borrow: PhantomData<&'a mut ()>,
}

impl<'a, D, C> DataLoaderIter<'a, D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    pub(super) fn new<S: Sampler>(loader: &'a mut DataLoader<D, S, C>) -> Self {
        let chunks = loader.batch_sampler.batch_indices(loader.dataset.len());

        if loader.inter_workers == 0 {
            let remaining = chunks.len();
            Self {
                inner: Inner::Direct {
                    chunks: chunks.into_iter(),
                    remaining,
                    dataset: &loader.dataset,
                    collator: &loader.collator,
                    pool: loader.pool.clone(),
                },
                _borrow: PhantomData,
            }
        } else {
            Self {
                inner: Inner::Parallel(ParallelCore::spawn(loader, chunks)),
                _borrow: PhantomData,
            }
        }
    }
}

impl<'a, D, C> Iterator for DataLoaderIter<'a, D, C>
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
                    *dataset,
                    &indices,
                    *collator,
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

impl<'a, D, C> ExactSizeIterator for DataLoaderIter<'a, D, C>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
}

// ── OwnedDataLoaderIter (Python FFI) ──────────────────────────────────────────

/// Lifetime-free iterator for the Python FFI boundary.
/// Safety is upheld by `PyDataloaderIter::_owner` keeping the loader alive.
#[cfg(feature = "python")]
pub(crate) struct OwnedDataLoaderIter<B>(ParallelCore<B>);

#[cfg(feature = "python")]
impl<B: Send + 'static> OwnedDataLoaderIter<B> {
    pub(super) fn new<D, S, C>(loader: &mut DataLoader<D, S, C>) -> Self
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item, Batch = B>,
    {
        let chunks = loader.batch_sampler.batch_indices(loader.dataset.len());
        Self(ParallelCore::spawn(loader, chunks))
    }
}

#[cfg(feature = "python")]
impl<B: Send + 'static> Iterator for OwnedDataLoaderIter<B> {
    type Item = Result<B>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.len();
        (n, Some(n))
    }
}

#[cfg(feature = "python")]
impl<B: Send + 'static> ExactSizeIterator for OwnedDataLoaderIter<B> {}
