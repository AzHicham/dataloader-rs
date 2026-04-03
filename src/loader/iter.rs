use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crossbeam_channel::{bounded, Receiver};

use crate::{
    collator::Collator,
    dataset::Dataset,
    error::Result,
    loader::{core::DataLoader, prefetch::prefetch_loop},
    sampler::Sampler,
};

/// Iterator over one epoch of batches, produced by `DataLoader::iter`.
pub struct DataLoaderIter<'a, B> {
    pub(super) rx: Option<Receiver<Result<B>>>,
    pub(super) remaining: usize,
    pub(super) handle: Option<std::thread::JoinHandle<()>>,
    pub(super) cancel: Arc<AtomicBool>,
    pub(super) _borrow: PhantomData<&'a mut ()>,
}

impl<'a, B: Send + 'static> DataLoaderIter<'a, B> {
    pub(super) fn new<D, S, C>(loader: &'a mut DataLoader<D, S, C>) -> Self
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item, Batch = B>,
    {
        let chunks = loader.batch_sampler.batch_indices(loader.dataset.len());
        let n_batches = chunks.len();
        let (tx, rx) = bounded::<Result<B>>(loader.prefetch_depth);

        // SAFETY: pointers are only used by the worker thread while the iterator
        // exists. Iterator drop joins the thread before loader internals are reused.
        let dataset_ptr = &loader.dataset as *const D as usize;
        let collator_ptr = &loader.collator as *const C as usize;
        let pool = loader.pool.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);

        let handle = std::thread::spawn(move || {
            // SAFETY: see comment above.
            let dataset: &D = unsafe { &*(dataset_ptr as *const D) };
            // SAFETY: see comment above.
            let collator: &C = unsafe { &*(collator_ptr as *const C) };

            prefetch_loop(dataset, chunks, collator, tx, pool, cancel_worker);
        });

        Self {
            rx: Some(rx),
            remaining: n_batches,
            handle: Some(handle),
            cancel,
            _borrow: PhantomData,
        }
    }
}


impl<'a, B: Send + 'static> Iterator for DataLoaderIter<'a, B> {
    type Item = Result<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let rx = self.rx.as_ref()?;
        match rx.recv() {
            Ok(batch) => {
                self.remaining -= 1;
                Some(batch)
            }
            Err(_) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, B: Send + 'static> ExactSizeIterator for DataLoaderIter<'a, B> {}

/// Owned iterator over one epoch of batches.
///
/// This variant does not borrow the parent loader and is intended for FFI
/// boundaries (e.g. Python bindings), where borrowed iterators are awkward to
/// represent safely.
///
/// # SAFETY WARNING
///
/// This type is intentionally `pub(crate)` because it depends on a strict
/// lifetime contract that the Rust type system cannot express directly:
/// the parent `DataLoader` (and therefore its dataset/collator internals)
/// must outlive this iterator.
///
/// Internally, the prefetch worker uses raw-pointer access to loader-owned
/// state. Dropping the loader before this iterator is undefined behavior.
/// Python bindings enforce this by storing a strong owner reference to the
/// loader inside `PyDataloaderIter`.
#[cfg(feature = "python")]
pub(crate) struct OwnedDataLoaderIter<B> {
    pub(super) rx: Option<Receiver<Result<B>>>,
    pub(super) remaining: usize,
    pub(super) handle: Option<std::thread::JoinHandle<()>>,
    pub(super) cancel: Arc<AtomicBool>,
}

#[cfg(feature = "python")]
impl<B: Send + 'static> OwnedDataLoaderIter<B> {
    pub(super) fn new<D, S, C>(loader: &mut DataLoader<D, S, C>) -> Self
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item, Batch = B>,
    {
        let chunks = loader.batch_sampler.batch_indices(loader.dataset.len());
        let n_batches = chunks.len();
        let (tx, rx) = bounded::<Result<B>>(loader.prefetch_depth);

        // SAFETY: pointers are only used by the worker thread while the iterator
        // exists. Iterator drop joins the thread before loader internals are reused.
        let dataset_ptr = &loader.dataset as *const D as usize;
        let collator_ptr = &loader.collator as *const C as usize;
        let pool = loader.pool.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);

        let handle = std::thread::spawn(move || {
            // SAFETY: see comment above.
            let dataset: &D = unsafe { &*(dataset_ptr as *const D) };
            // SAFETY: see comment above.
            let collator: &C = unsafe { &*(collator_ptr as *const C) };

            prefetch_loop(dataset, chunks, collator, tx, pool, cancel_worker);
        });

        Self {
            rx: Some(rx),
            remaining: n_batches,
            handle: Some(handle),
            cancel,
        }
    }
}

#[cfg(feature = "python")]
impl<B: Send + 'static> Iterator for OwnedDataLoaderIter<B> {
    type Item = Result<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let rx = self.rx.as_ref()?;
        match rx.recv() {
            Ok(batch) => {
                self.remaining -= 1;
                Some(batch)
            }
            Err(_) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(feature = "python")]
impl<B: Send + 'static> ExactSizeIterator for OwnedDataLoaderIter<B> {}

impl<'a, B> Drop for DataLoaderIter<'a, B> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        drop(self.rx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(feature = "python")]
impl<B> Drop for OwnedDataLoaderIter<B> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        drop(self.rx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
