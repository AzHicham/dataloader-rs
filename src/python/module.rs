//! The `#[pymodule]`: the classes and functions of the `dataloader_rs` Python module.

use pyo3::prelude::*;

use crate::python::{
    dataloader::PyDataloader,
    dataset::{
        PyAsyncDatasetBase, PyDatasetBase, PyFileDataset, PyHttpDataset, PyObjectStoreDataset,
        PyRustDataset,
    },
    image::{self, PyImage, PyImageBatch, PyImageCollator, PyImageTransform},
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
    m.add_function(wrap_pyfunction!(pool::attach_count, m)?)?;
    m.add_class::<PyImage>()?;
    m.add_class::<PyImageBatch>()?;
    m.add_class::<PyImageCollator>()?;
    m.add_class::<PyImageTransform>()?;
    // `dataloader_rs.transforms` re-exports these from the extension module.
    for transform in [
        wrap_pyfunction!(image::decode, m)?,
        wrap_pyfunction!(image::to_rgb, m)?,
        wrap_pyfunction!(image::to_rgba, m)?,
        wrap_pyfunction!(image::to_gray, m)?,
        wrap_pyfunction!(image::crop, m)?,
        wrap_pyfunction!(image::center_crop, m)?,
        wrap_pyfunction!(image::pad, m)?,
        wrap_pyfunction!(image::pad_to, m)?,
        wrap_pyfunction!(image::resize, m)?,
        wrap_pyfunction!(image::hflip, m)?,
        wrap_pyfunction!(image::vflip, m)?,
        wrap_pyfunction!(image::rotate90, m)?,
        wrap_pyfunction!(image::to_float, m)?,
        wrap_pyfunction!(image::normalize, m)?,
        wrap_pyfunction!(image::to_chw, m)?,
        wrap_pyfunction!(image::to_hwc, m)?,
        wrap_pyfunction!(image::compose, m)?,
    ] {
        m.add_function(transform)?;
    }
    Ok(())
}
