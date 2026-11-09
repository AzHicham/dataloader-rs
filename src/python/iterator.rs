use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use crate::loader::{AsyncSource, DataLoaderIter, LoaderSource, SyncSource};
use crate::python::collator::{PyBatch, PyCollator};
use crate::python::dataloader::PyDataloader;
use crate::python::dataset::{Fetcher, PyAsyncItems, PyDataset, RustSource};
use crate::python::error::into_py_err;
use crate::python::pool::exiting;
use crate::python::transform::{PyItem, PyTransform};
use crate::transform::Transform;

type SyncIter = DataLoaderIter<SyncSource<PyDataset>, PyCollator, PyTransform>;
pub(crate) type AsyncIter<A> = DataLoaderIter<AsyncSource<A>, PyCollator, PyTransform>;
/// The indices of an epoch's batches, in epoch order.
pub(crate) type BatchIndices = std::vec::IntoIter<Vec<usize>>;

/// How long `__next__` waits with the interpreter released before it runs
/// signal handlers: Ctrl-C ends a wait within this.
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) enum PyIterInner {
    /// Sequential (num_workers=0): call Python directly in `__next__` using the
    /// already-held `py` token — no thread, no channel, no extra GIL acquire.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        /// `__getitems__` or cached `__getitem__`, resolved once per epoch.
        fetcher: Fetcher,
        /// Transform and collator cloned at `__iter__` time so `__next__` skips
        /// borrowing `_owner`.
        transform: PyTransform,
        collator: PyCollator,
    },
    /// Parallel (num_workers>0): batch tasks on the loader's pool.
    Threaded(SyncIter),
    /// A `PyAsyncDataset`: coroutines on the loader's event loop.
    Async(AsyncIter<PyAsyncItems>),
    /// A Rust dataset: futures on the loader's runtime.
    Rust(AsyncIter<RustSource>),
}

/// The next batch of a core iterator, `None` at the end of the epoch. A batch
/// that already arrived is taken without releasing the interpreter. Otherwise
/// wait with it released, so workers can attach, in slices: between two,
/// signal handlers run, so Ctrl-C raises `KeyboardInterrupt` here (the outer
/// error: no batch was taken) and the iterator stays usable. A worker's panic
/// unwinds from here to PyO3, which raises it; the epoch is then over.
fn wait_next<Src>(
    py: Python<'_>,
    inner: &mut DataLoaderIter<Src, PyCollator, PyTransform>,
) -> PyResult<Option<PyResult<PyBatch>>>
where
    Src: LoaderSource,
    PyTransform: Transform<Src::Item, Output = PyItem>,
{
    let next = match inner.next_until(None) {
        Some(next) => next,
        None => loop {
            let deadline = Instant::now() + SIGNAL_CHECK_INTERVAL;
            if let Some(next) = py.detach(|| inner.next_until(Some(deadline))) {
                break next;
            }
            py.check_signals()?;
        },
    };
    Ok(next.map(|batch| batch.map_err(into_py_err)))
}

/// How long dropping an iterator waits for the batches in flight.
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

#[pyclass(name = "PyDataloaderIter", module = "dataloader_rs")]
pub struct PyDataloaderIter {
    /// Dropped first: the epoch ends before its loader is freed.
    inner: PyIterInner,
    /// With `return_indices=True`: the indices of the batches not yet returned, one
    /// `Vec` per batch. Every path returns batches in epoch order, a failed one included.
    indices: Option<BatchIndices>,
    /// Keeps the `PyDataloader` alive while the iterator exists.
    _owner: Py<PyDataloader>,
}

/// Stop the epoch before it is dropped: wait, with the interpreter released,
/// for the tasks in flight to finish their batch (at most [`STOP_TIMEOUT`]).
/// They hold the dataset, transform and collator; once they are gone, this
/// thread drops the last references the epoch has, so a reference cycle
/// through them is freed by the `gc.collect()` that frees the iterator.
/// Not while the interpreter shuts down: the tasks cannot attach any more.
impl Drop for PyDataloaderIter {
    fn drop(&mut self) {
        if exiting() {
            return;
        }
        Python::try_attach(|py| {
            py.detach(|| match &mut self.inner {
                PyIterInner::Direct { .. } => true,
                PyIterInner::Threaded(inner) => inner.stop(STOP_TIMEOUT),
                PyIterInner::Async(inner) => inner.stop(STOP_TIMEOUT),
                PyIterInner::Rust(inner) => inner.stop(STOP_TIMEOUT),
            })
        });
    }
}

impl PyDataloaderIter {
    /// An epoch of `owner`.
    pub(crate) fn new(
        owner: Py<PyDataloader>,
        inner: PyIterInner,
        indices: Option<BatchIndices>,
    ) -> Self {
        Self {
            inner,
            indices,
            _owner: owner,
        }
    }

    /// The next batch, `None` at the end of the epoch. The outer error means no
    /// batch was taken (Ctrl-C); the inner one is the batch's own.
    fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<PyResult<PyBatch>>> {
        match &mut self.inner {
            PyIterInner::Direct {
                chunks,
                fetcher,
                transform,
                collator,
            } => {
                let Some(chunk) = chunks.next() else {
                    return Ok(None);
                };
                // We already hold `py` — call Python directly with no extra GIL acquire.
                let batch = fetcher.fetch(py, &chunk).and_then(|items| {
                    let items = items
                        .into_iter()
                        .zip(&chunk)
                        .map(|(item, &index)| transform.call(py, index, item))
                        .collect::<PyResult<Vec<_>>>()?;
                    collator.collate_with_py(py, items).map_err(into_py_err)
                });
                Ok(Some(batch))
            }
            PyIterInner::Threaded(inner) => wait_next(py, inner),
            PyIterInner::Async(inner) => wait_next(py, inner),
            PyIterInner::Rust(inner) => wait_next(py, inner),
        }
    }
}

#[pymethods]
impl PyDataloaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next batch, or `(batch, indices)` with `return_indices=True`: `indices` is a
    /// `list` of the dataset indices of the batch's items, built here, once per batch.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(batch) = self.next_batch(py)? else {
            return Ok(None);
        };
        // Taken before the batch's own error is raised, so the next batch gets its own.
        let indices = self.indices.as_mut().map(|indices| indices.next());
        let batch = batch?.into_py(py)?;
        let Some(indices) = indices else {
            return Ok(Some(batch));
        };
        let indices = PyList::new(py, indices.unwrap_or_default())?;
        Ok(Some(
            PyTuple::new(py, [batch.into_bound(py), indices.into_any()])?
                .into_any()
                .unbind(),
        ))
    }

    /// The loader, plus the references the direct path holds itself; a
    /// threaded epoch shares the loader's objects.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self._owner)?;
        if let PyIterInner::Direct {
            fetcher,
            transform,
            collator,
            ..
        } = &self.inner
        {
            visit.call(fetcher.method())?;
            visit.call(transform.function())?;
            visit.call(collator.collate_fn())?;
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            PyIterInner::Direct { chunks, .. } => chunks.len(),
            PyIterInner::Threaded(inner) => inner.len(),
            PyIterInner::Async(inner) => inner.len(),
            PyIterInner::Rust(inner) => inner.len(),
        }
    }
}
