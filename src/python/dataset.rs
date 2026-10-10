use crate::{dataset::Dataset, error::Result};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

/// How a batch of samples is fetched from a Python dataset.
///
/// Mirrors PyTorch's fetcher: a dataset defining `__getitems__(indices)` is
/// called once per batch with the list of indices and returns the list of
/// samples; otherwise `__getitem__` is called once per index.
pub(crate) enum Fetcher {
    /// Bound `__getitems__` method.
    Batched(Py<PyAny>),
    /// Bound `__getitem__` method, cached to skip one lookup per item.
    PerItem(Py<PyAny>),
}

impl Fetcher {
    pub(crate) fn new(dataset: &PyDataset, py: Python<'_>) -> PyResult<Self> {
        let dataset = dataset.bind(py);
        Ok(match dataset.getattr_opt(intern!(py, "__getitems__"))? {
            Some(getitems) => Self::Batched(getitems.unbind()),
            None => Self::PerItem(dataset.getattr(intern!(py, "__getitem__"))?.unbind()),
        })
    }

    /// The bound method used to fetch: `__getitems__` or `__getitem__`.
    pub(crate) fn method(&self) -> &Py<PyAny> {
        match self {
            Self::Batched(method) | Self::PerItem(method) => method,
        }
    }

    pub(crate) fn is_batched(&self) -> bool {
        matches!(self, Self::Batched(_))
    }

    /// Whether the fetch method is `async def`.
    pub(crate) fn is_async(&self, py: Python<'_>) -> PyResult<bool> {
        py.import(intern!(py, "inspect"))?
            .call_method1(intern!(py, "iscoroutinefunction"), (self.method(),))?
            .is_truthy()
    }

    pub(crate) fn fetch(&self, py: Python<'_>, indices: &[usize]) -> PyResult<Vec<Py<PyAny>>> {
        match self {
            Self::PerItem(getitem) => indices.iter().map(|&i| getitem.call1(py, (i,))).collect(),
            Self::Batched(getitems) => {
                let samples = getitems.bind(py).call1((PyList::new(py, indices)?,))?;
                let samples = samples
                    .try_iter()?
                    .map(|sample| sample.map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()?;
                // A short or long list would silently change batch sizes.
                if samples.len() != indices.len() {
                    return Err(PyValueError::new_err(format!(
                        "__getitems__ returned {} samples for {} indices",
                        samples.len(),
                        indices.len()
                    )));
                }
                Ok(samples)
            }
        }
    }
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
    /// Uses `__getitems__` when the dataset defines it, else a cached bound
    /// `__getitem__` (one attribute lookup per batch, not per item).
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Py<PyAny>>> {
        Python::attach(|py| Fetcher::new(self, py)?.fetch(py, indices)).map_err(Into::into)
    }

    fn len(&self) -> usize {
        Python::attach(|py| {
            len_py(self, py).unwrap_or_else(|e| panic!("PyDataset.__len__ failed: {e}"))
        })
    }
}
