//! The `#[pymodule]`: the classes and functions of the `dataloader_rs` Python module.

use pyo3::prelude::*;

use crate::python::{
    dataloader::PyDataloader,
    dataset::{
        PyAsyncDatasetBase, PyDatasetBase, PyFileDataset, PyHttpDataset, PyObjectStoreDataset,
        PyRustDataset,
    },
    iterator::PyDataloaderIter,
    pool,
};

#[pymodule(gil_used = false)]
fn dataloader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // PyO3 asserts, on the first attach from a thread that is not attached, that the
    // interpreter is initialized. Do that first attach here: a pool thread attaching while
    // the interpreter finishes shutting down would panic on it instead of being refused.
    m.py().detach(|| Python::try_attach(|_| ()));
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
