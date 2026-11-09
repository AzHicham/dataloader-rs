use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::{PyTraverseError, PyVisit};

use crate::loader::{AsyncSource, DataLoaderIter, LoaderSource, SyncSource};
use crate::python::async_dataset::PyAsyncItems;
use crate::python::collator::{PyBatch, PyCollator};
use crate::python::dataloader::PyDataloader;
use crate::python::dataset::{Fetcher, PyDataset};
use crate::python::into_py_err;
use crate::python::rust_dataset::RustSource;
use crate::python::transform::PyTransform;
use crate::transform::Transform;

type SyncIter = DataLoaderIter<SyncSource<PyDataset>, PyCollator, PyTransform>;
pub(crate) type AsyncIter<A> = DataLoaderIter<AsyncSource<A>, PyCollator, PyTransform>;

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
    PyTransform: Transform<Src::Item, Output = Py<PyAny>>,
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

#[pyclass(name = "PyDataloaderIter", module = "dataloader_rs")]
pub struct PyDataloaderIter {
    /// Keeps the `PyDataloader` alive while the iterator exists.
    _owner: Py<PyDataloader>,
    inner: PyIterInner,
}

impl PyDataloaderIter {
    /// An epoch of `owner`.
    pub(crate) fn new(owner: Py<PyDataloader>, inner: PyIterInner) -> Self {
        Self {
            _owner: owner,
            inner,
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

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self.next_batch(py)? {
            Some(batch) => Ok(Some(batch?.into_py(py)?)),
            None => Ok(None),
        }
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
