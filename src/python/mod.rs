#![doc = include_str!("README.md")]

mod async_dataset;
mod collator;
mod dataloader;
mod dataset;
mod iterator;
mod pool;
mod rust_dataset;
mod sampler;
mod transform;

use async_dataset::PyAsyncDatasetBase;
use dataloader::PyDataloader;
use dataset::PyDatasetBase;
use iterator::PyDataloaderIter;
use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use rust_dataset::{PyFileDataset, PyHttpDataset, PyObjectStoreDataset, PyRustDataset};

use crate::dataset::FetchError;

/// Convert a core error back into the Python exception it came from.
///
/// Errors raised by `__getitem__` or `collate_fn` travel through the core as
/// boxed `PyErr`s; unwrapping them keeps the original exception type and
/// traceback. Errors that did not originate in Python become `RuntimeError`.
///
/// A built-in dataset's [`FetchError`] becomes `IndexError` (index out of
/// range), `ValueError` (item over `max_bytes`), `TimeoutError` or `OSError`.
pub(crate) fn into_py_err(err: crate::error::Error) -> PyErr {
    let err = match err.downcast::<PyErr>() {
        Ok(err) => return *err,
        Err(err) => err,
    };
    let Some(fetch) = err.downcast_ref::<FetchError>() else {
        return PyRuntimeError::new_err(err.to_string());
    };
    let message = err.to_string();
    match fetch {
        FetchError::OutOfRange { .. } => PyIndexError::new_err(message),
        FetchError::TooLarge { .. } => PyValueError::new_err(message),
        fetch if fetch.is_timeout() => PyTimeoutError::new_err(message),
        _ => PyOSError::new_err(message),
    }
}

#[pymodule(gil_used = false)]
fn dataloader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataloader>()?;
    m.add_class::<PyDataloaderIter>()?;
    m.add_class::<PyDatasetBase>()?;
    m.add_class::<PyAsyncDatasetBase>()?;
    m.add_class::<PyRustDataset>()?;
    m.add_class::<PyHttpDataset>()?;
    m.add_class::<PyFileDataset>()?;
    m.add_class::<PyObjectStoreDataset>()?;
    m.add_function(wrap_pyfunction!(pool::keep_pools_alive, m)?)?;
    Ok(())
}
