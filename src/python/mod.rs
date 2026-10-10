mod collator;
mod dataloader;
mod dataset;
mod iterator;
mod sampler;

use dataloader::PyDataloader;
use dataset::PyDatasetBase;
use iterator::PyDataloaderIter;
use pyo3::prelude::*;
use sampler::PyDistributedSampler;

#[pymodule(gil_used = false)]
fn dataloader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataloader>()?;
    m.add_class::<PyDataloaderIter>()?;
    m.add_class::<PyDatasetBase>()?;
    m.add_class::<PyDistributedSampler>()?;
    Ok(())
}
