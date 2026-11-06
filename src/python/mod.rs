#![doc = include_str!("README.md")]

mod collator;
mod dataloader;
mod dataset;
mod image;
mod iterator;
mod pool;
mod sampler;
mod source;

use dataloader::PyDataloader;
use dataset::PyDatasetBase;
use image::{PyCrop, PyImageBatch, PyImageCollator, PyNormalize, PyPad, PyResize, PyToFloat};
use iterator::PyDataloaderIter;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use source::PySource;

use crate::sources::SourceError;

/// Convert a core error back into the Python exception it came from.
///
/// Errors raised by `__getitem__` or `collate_fn` travel through the core as
/// boxed `PyErr`s; unwrapping them keeps the original exception type and
/// traceback. A `SourceError` becomes the matching `OSError` (see `source::source_error`);
/// other errors that did not originate in Python become `RuntimeError`.
pub(crate) fn into_py_err(err: crate::error::Error) -> PyErr {
    let err = match err.downcast::<PyErr>() {
        Ok(err) => return *err,
        Err(err) => err,
    };
    match err.downcast_ref::<SourceError>() {
        Some(err) => source::source_error(err),
        None => PyRuntimeError::new_err(err.to_string()),
    }
}

#[pymodule(gil_used = false)]
fn dataloader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataloader>()?;
    m.add_class::<PyDataloaderIter>()?;
    m.add_class::<PyDatasetBase>()?;
    m.add_class::<PyImageCollator>()?;
    m.add_class::<PyImageBatch>()?;
    m.add_class::<PySource>()?;
    m.add_class::<PyResize>()?;
    m.add_class::<PyCrop>()?;
    m.add_class::<PyPad>()?;
    m.add_class::<PyToFloat>()?;
    m.add_class::<PyNormalize>()?;
    m.add_function(wrap_pyfunction!(pool::keep_pools_alive, m)?)?;
    Ok(())
}
