use crate::{dataset::Dataset, error::Result};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyclass(frozen, subclass)]
pub struct PyDatasetBase;

#[pymethods]
impl PyDatasetBase {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    fn __len__(&self) -> PyResult<usize> {
        Err(PyNotImplementedError::new_err(
            "PyDataset.__len__ must be implemented by subclasses",
        ))
    }

    fn __getitem__(&self, _index: usize) -> PyResult<Py<PyAny>> {
        Err(PyNotImplementedError::new_err(
            "PyDataset.__getitem__ must be implemented by subclasses",
        ))
    }
}

pub(crate) type PyDataset = Py<PyDatasetBase>;

pub(crate) fn len_py(dataset: &PyDataset, py: Python<'_>) -> PyResult<usize> {
    dataset.call_method0(py, "__len__")?.extract(py)
}

pub(crate) fn get_item_py(dataset: &PyDataset, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
    dataset.call_method1(py, "__getitem__", (index,))
}

impl Dataset for PyDataset {
    type Item = Py<PyAny>;

    fn get(&self, index: usize) -> Result<Self::Item> {
        Python::attach(|py| get_item_py(self, py, index).map_err(|e| e.into()))
    }

    fn len(&self) -> usize {
        Python::attach(|py| {
            len_py(self, py)
                .unwrap_or_else(|e| panic!("PyDataset.__len__ failed: {e}"))
        })
    }
}
