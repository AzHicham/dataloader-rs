#![doc = include_str!("README.md")]

mod collator;
mod dataloader;
mod dataset;
mod image;
mod iterator;
mod pool;
mod sampler;

use dataloader::PyDataloader;
use dataset::PyDatasetBase;
use image::{PyCrop, PyImageBatch, PyImageCollator, PyNormalize, PyPad, PyResize, PyToFloat};
use iterator::PyDataloaderIter;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

/// Convert a core error back into the Python exception it came from.
///
/// Errors raised by `__getitem__` or `collate_fn` travel through the core as
/// boxed `PyErr`s; unwrapping them keeps the original exception type and
/// traceback. Errors that did not originate in Python become `RuntimeError`.
pub(crate) fn into_py_err(err: crate::error::Error) -> PyErr {
    match err.downcast::<PyErr>() {
        Ok(err) => *err,
        Err(err) => PyRuntimeError::new_err(err.to_string()),
    }
}

#[pymodule(gil_used = false)]
fn dataloader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataloader>()?;
    m.add_class::<PyDataloaderIter>()?;
    m.add_class::<PyDatasetBase>()?;
    m.add_class::<PyImageCollator>()?;
    m.add_class::<PyImageBatch>()?;
    m.add_class::<PyResize>()?;
    m.add_class::<PyCrop>()?;
    m.add_class::<PyPad>()?;
    m.add_class::<PyToFloat>()?;
    m.add_class::<PyNormalize>()?;
    m.add_function(wrap_pyfunction!(pool::keep_pools_alive, m)?)?;
    Ok(())
}
