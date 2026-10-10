use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::python::collator::{PyBatch, PyCollator};
use crate::python::dataloader::PyDataloader;
use crate::python::dataset::PyDataset;
use crate::python::into_py_err;

type CorePyDataloaderIter = crate::loader::DataLoaderIter<PyDataset, PyCollator>;

pub(crate) enum PyIterInner {
    /// Sequential (num_workers=0): call Python directly in `__next__` using the
    /// already-held `py` token — no thread, no channel, no extra GIL acquire.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        remaining: usize,
        /// Cached bound `__getitem__` method — avoids one attribute lookup per item.
        getitem: Py<PyAny>,
        /// Collator cloned at `__iter__` time so `__next__` skips borrowing `_owner`.
        collator: PyCollator,
    },
    /// Parallel (num_workers>0): threaded prefetch with crossbeam channel.
    Threaded(Option<CorePyDataloaderIter>),
}

#[pyclass(name = "PyDataloaderIter", module = "dataloader_rs", unsendable)]
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
        match &mut self.inner {
            PyIterInner::Direct {
                chunks,
                remaining,
                getitem,
                collator,
            } => {
                let Some(chunk) = chunks.next() else {
                    return Ok(None);
                };
                *remaining -= 1;

                // We already hold `py` — call Python directly with no extra GIL acquire.
                // Use the cached bound method to skip one attribute lookup per item.
                let items: PyResult<Vec<Py<PyAny>>> =
                    chunk.iter().map(|&i| getitem.call1(py, (i,))).collect();
                let items = items?;

                let batch = collator.collate_with_py(py, items).map_err(into_py_err)?;

                let out = match batch {
                    PyBatch::Ready(obj) => obj,
                    PyBatch::Items(items) => PyList::new(py, items)?.unbind().into_any(),
                };
                Ok(Some(out))
            }

            PyIterInner::Threaded(inner_opt) => {
                let Some(mut inner) = inner_opt.take() else {
                    return Ok(None);
                };
                // Release Python thread state while blocked on the channel.
                let next_item = py.detach(|| inner.next());
                match next_item {
                    Some(Ok(batch)) => {
                        let out = match batch {
                            PyBatch::Ready(obj) => obj,
                            PyBatch::Items(items) => PyList::new(py, items)?.unbind().into_any(),
                        };
                        *inner_opt = Some(inner);
                        Ok(Some(out))
                    }
                    Some(Err(e)) => {
                        py.detach(|| drop(inner));
                        Err(into_py_err(e))
                    }
                    None => {
                        py.detach(|| drop(inner));
                        Ok(None)
                    }
                }
            }
        }
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            PyIterInner::Direct { remaining, .. } => *remaining,
            PyIterInner::Threaded(inner) => inner.as_ref().map_or(0, |it| it.len()),
        }
    }
}

impl Drop for PyDataloaderIter {
    /// Joining the workers must happen with the thread state detached: a
    /// worker blocked in `Python::attach` (inside `__getitem__` or
    /// `collate_fn`) can only finish once this thread lets go of the GIL.
    fn drop(&mut self) {
        if let PyIterInner::Threaded(inner) = &mut self.inner
            && let Some(inner) = inner.take()
        {
            Python::attach(|py| py.detach(|| drop(inner)));
        }
    }
}
