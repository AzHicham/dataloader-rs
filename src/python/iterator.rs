use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::python::dataloader::PyDataloader;
use crate::python::collator::PyBatch;

type CorePyDataloaderIter = crate::loader::OwnedDataLoaderIter<PyBatch>;

#[pyclass(name = "PyDataloaderIter", module = "dataloader_rs", unsendable)]
pub struct PyDataloaderIter {
    // Keep the Python loader object alive while this iterator exists.
    // `iter_owned()` relies on loader-owned internals staying valid.
    pub(crate) _owner: Py<PyDataloader>,
    pub(crate) inner: Option<CorePyDataloaderIter>,
}

#[pymethods]
impl PyDataloaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(mut inner) = self.inner.take() else {
            return Ok(None);
        };
        // Avoid blocking while attached to Python runtime.
        // In free-threaded builds this prevents deadlock-prone global sync waits.
        let next_item = py.detach(|| inner.next());
        match next_item {
            Some(Ok(v)) => {
                let out = match v {
                    PyBatch::Ready(obj) => obj,
                    PyBatch::Items(items) => PyList::new(py, items)?.unbind().into_any(),
                };
                self.inner = Some(inner);
                Ok(Some(out))
            }
            Some(Err(e)) => {
                py.detach(|| drop(inner));
                Err(PyRuntimeError::new_err(e.to_string()))
            }
            None => {
                py.detach(|| drop(inner));
                Ok(None)
            }
        }
    }

    fn __len__(&self) -> usize {
        self.inner.as_ref().map_or(0, |it| it.len())
    }

    fn __del__(&mut self, py: Python<'_>) {
        if let Some(inner) = self.inner.take() {
            py.detach(|| drop(inner));
        }
    }
}
