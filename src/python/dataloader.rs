use crate::loader as core_loader;
use crate::sampler::{RandomSampler, SequentialSampler};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::python::collator::PyCollator;
use crate::python::dataset::PyDataset;
use crate::python::iterator::PyDataloaderIter;
use crate::python::sampler::{validate_python_sampler, PySampler, SharedPySampler};

type CorePyLoader = core_loader::DataLoader<PyDataset, SharedPySampler, PyCollator>;

#[pyclass(name = "PyDataloader", module = "dataloader_rs", unsendable)]
pub struct PyDataloader {
    inner: CorePyLoader,
}

#[pymethods]
impl PyDataloader {
    #[new]
    #[pyo3(signature = (
        dataset,
        batch_size=1,
        prefetch_depth=1,
        shuffle=false,
        sampler=None,
        num_workers=0,
        collate_fn=None,
        drop_last=false
    ))]
    fn new(
        dataset: PyDataset,
        batch_size: usize,
        prefetch_depth: usize,
        shuffle: bool,
        sampler: Option<Py<PyAny>>,
        num_workers: usize,
        collate_fn: Option<Py<PyAny>>,
        drop_last: bool,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }
        if prefetch_depth == 0 {
            return Err(PyValueError::new_err("prefetch_depth must be > 0"));
        }
        if sampler.is_some() && shuffle {
            return Err(PyValueError::new_err(
                "sampler and shuffle are mutually exclusive",
            ));
        }

        let sampler = match sampler {
            Some(py_sampler) => {
                validate_python_sampler(&py_sampler).map_err(|e| PyValueError::new_err(e.to_string()))?;
                SharedPySampler::new(PySampler::Python(py_sampler))
            }
            None if shuffle => SharedPySampler::new(PySampler::Random(RandomSampler::from_entropy())),
            None => SharedPySampler::new(PySampler::Sequential(SequentialSampler)),
        };

        let collator = PyCollator::new(collate_fn);
        let loader = core_loader::DataLoader::builder(dataset)
            .batch_size(batch_size)
            .prefetch_depth(prefetch_depth)
            .drop_last(drop_last)
            .num_workers(num_workers)
            .sampler(sampler)
            .collator(collator)
            .build();

        Ok(Self {
            inner: loader,
        })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let inner = {
            let mut loader = slf.borrow_mut(py);
            loader.inner.iter_owned()
        };
        Ok(PyDataloaderIter {
            _owner: slf,
            inner: Some(inner),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.batch_len()
    }
}
