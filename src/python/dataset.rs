use crate::{dataset::Dataset, error::Result};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::time::Instant;

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
    dataset
        .call_method0(py, intern!(py, "__len__"))?
        .extract(py)
}

pub(crate) fn get_item_py(
    dataset: &PyDataset,
    py: Python<'_>,
    index: usize,
) -> PyResult<Py<PyAny>> {
    dataset.call_method1(py, intern!(py, "__getitem__"), (index,))
}

impl Dataset for PyDataset {
    type Item = Py<PyAny>;

    fn get(&self, index: usize) -> Result<Self::Item> {
        Python::attach(|py| get_item_py(self, py, index).map_err(|e| e.into()))
    }

    /// Acquire the Python thread state once for the entire batch rather than
    /// once per item — reduces GIL attach/release overhead from O(batch_size)
    /// to O(1) per batch in the threaded (num_workers>0) code path.
    ///
    /// Also caches the bound `__getitem__` method for the duration of the batch,
    /// saving one attribute lookup per item (same trick used in the direct path).
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Py<PyAny>>> {
        Python::attach(|py| {
            let getitem = self
                .getattr(py, intern!(py, "__getitem__"))
                .map_err(crate::error::Error::from)?;
            indices
                .iter()
                .map(|&i| getitem.call1(py, (i,)).map_err(crate::error::Error::from))
                .collect()
        })
    }

    fn len(&self) -> usize {
        Python::attach(|py| {
            len_py(self, py).unwrap_or_else(|e| panic!("PyDataset.__len__ failed: {e}"))
        })
    }
}

#[pyfunction]
pub(crate) fn bench_dataset_get_dispatch(dataset: PyDataset, iters: usize) -> PyResult<f64> {
    if iters == 0 {
        return Err(PyValueError::new_err("iters must be > 0"));
    }
    let n = Python::attach(|py| len_py(&dataset, py))?;
    if n == 0 {
        return Err(PyValueError::new_err("dataset length must be > 0"));
    }

    let start = Instant::now();
    for i in 0..iters {
        <PyDataset as Dataset>::get(&dataset, i % n)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    Ok(start.elapsed().as_secs_f64())
}
