mod collator;
mod dataloader;
mod dataset;
mod iterator;
mod sampler;

use dataloader::PyDataloader;
use dataset::{PyDatasetBase, bench_dataset_get_dispatch};
use iterator::PyDataloaderIter;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule(gil_used = false)]
fn dataloader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataloader>()?;
    m.add_class::<PyDataloaderIter>()?;
    m.add_class::<PyDatasetBase>()?;
    m.add_function(wrap_pyfunction!(bench_dataset_get_dispatch, m)?)?;
    Ok(())
}
