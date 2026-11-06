use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{PyTraverseError, PyVisit};

use crate::collator::Collator;
use crate::dataset::Dataset;
use crate::error::Result as CoreResult;
use crate::loader::DataLoaderIter;
use crate::python::collator::PyCollator;
use crate::python::dataloader::PyDataloader;
use crate::python::dataset::{Fetcher, PyDataset};
use crate::python::image::{ImagePoolCollator, ImageSource, PyImageBatch};
use crate::python::into_py_err;

type CorePyDataloaderIter = DataLoaderIter<PyDataset, PyCollator>;
type CoreImageIter = DataLoaderIter<ImageSource, ImagePoolCollator>;

/// An epoch of a loader with an `ImageCollator`.
pub(crate) struct ImageEpoch {
    pub(crate) epoch: CoreImageIter,
    pub(crate) sample_shape: [usize; 3],
    /// `ImageBatch`es alive, shared with the loader and its other epochs.
    pub(crate) alive: Arc<AtomicUsize>,
    pub(crate) max_held_batches: usize,
}

/// The next batch of a core epoch: taken at once if it is ready, else waited for with the
/// interpreter released, in slices between which signal handlers run, so Ctrl-C raises
/// `KeyboardInterrupt` here and the iterator stays usable. A worker's panic unwinds from here to
/// PyO3, which raises it; the epoch is then over.
fn next_batch<D, C>(
    py: Python<'_>,
    inner: &mut DataLoaderIter<D, C>,
) -> PyResult<Option<CoreResult<C::Batch>>>
where
    D: Dataset,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
    DataLoaderIter<D, C>: Send,
{
    if let Some(next) = inner.next_until(None) {
        return Ok(next);
    }
    loop {
        let deadline = Instant::now() + SIGNAL_CHECK_INTERVAL;
        if let Some(next) = py.detach(|| inner.next_until(Some(deadline))) {
            return Ok(next);
        }
        py.check_signals()?;
    }
}

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
        /// Collator cloned at `__iter__` time so `__next__` skips borrowing `_owner`.
        collator: PyCollator,
    },
    /// Parallel (num_workers>0): batch tasks on the loader's pool.
    Threaded(CorePyDataloaderIter),
    /// Images decoded and collated in Rust; `__next__` returns `ImageBatch`es.
    Image(ImageEpoch),
}

#[pyclass(name = "PyDataloaderIter", module = "dataloader_rs")]
pub struct PyDataloaderIter {
    /// Keeps the `PyDataloader` alive while the iterator exists.
    pub(crate) _owner: Py<PyDataloader>,
    pub(crate) inner: PyIterInner,
}

#[pymethods]
impl PyDataloaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let batch = match &mut self.inner {
            PyIterInner::Direct {
                chunks,
                fetcher,
                collator,
            } => {
                let Some(chunk) = chunks.next() else {
                    return Ok(None);
                };
                // We already hold `py` — call Python directly with no extra GIL acquire.
                let items = fetcher.fetch(py, &chunk)?;
                collator.collate_with_py(py, items).map_err(into_py_err)?
            }

            PyIterInner::Threaded(inner) => {
                // Workers attach to the interpreter: wait with it released.
                let Some(batch) = next_batch(py, inner)? else {
                    return Ok(None);
                };
                batch.map_err(into_py_err)?
            }

            PyIterInner::Image(image) => {
                // Every buffer of the pool may be held by `ImageBatch`es: asking for one more
                // would wait forever for the consumer to drop one.
                let alive = image.alive.load(Ordering::Acquire);
                if alive >= image.max_held_batches && image.epoch.len() > 0 {
                    return Err(PyRuntimeError::new_err(format!(
                        "{alive} ImageBatch objects are still alive and max_held_batches is {}: \
                         drop them (and the arrays viewing them) before asking for the next \
                         batch, copy what you keep (numpy.array(batch)), or raise \
                         max_held_batches",
                        image.max_held_batches
                    )));
                }
                let Some(batch) = next_batch(py, &mut image.epoch)? else {
                    return Ok(None);
                };
                let batch = batch.map_err(into_py_err)?;
                let batch = PyImageBatch::new(batch, image.sample_shape, &image.alive);
                return Ok(Some(Py::new(py, batch)?.into_any()));
            }
        };
        batch.into_py(py).map(Some)
    }

    /// The loader, plus the references the direct path holds itself; a
    /// threaded epoch shares the loader's objects.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self._owner)?;
        if let PyIterInner::Direct {
            fetcher, collator, ..
        } = &self.inner
        {
            visit.call(fetcher.method())?;
            visit.call(collator.collate_fn())?;
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            PyIterInner::Direct { chunks, .. } => chunks.len(),
            PyIterInner::Threaded(inner) => inner.len(),
            PyIterInner::Image(image) => image.epoch.len(),
        }
    }
}
