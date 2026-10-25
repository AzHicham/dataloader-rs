use pyo3::prelude::*;

use crate::python::collator::PyCollator;
use crate::python::dataloader::PyDataloader;
use crate::python::dataset::{Fetcher, PyDataset};
use crate::python::into_py_err;

type CorePyDataloaderIter = crate::loader::DataLoaderIter<PyDataset, PyCollator>;

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
                // A batch that already arrived is taken without releasing the
                // interpreter; otherwise wait with it released, so workers can
                // attach. A worker's panic unwinds from here to PyO3, which
                // raises it; the epoch is then over.
                let next = match inner.try_next() {
                    Some(next) => next,
                    None => py.detach(|| inner.next()),
                };
                let Some(batch) = next else {
                    return Ok(None);
                };
                batch.map_err(into_py_err)?
            }
        };
        batch.into_py(py).map(Some)
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            PyIterInner::Direct { chunks, .. } => chunks.len(),
            PyIterInner::Threaded(inner) => inner.len(),
        }
    }
}
