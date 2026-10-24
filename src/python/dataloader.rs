use crate::loader as core_loader;
use crate::sampler::{RandomSampler, SequentialSampler};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::python::collator::PyCollator;
use crate::python::dataset::{Fetcher, PyDataset, len_py};
use crate::python::into_py_err;
use crate::python::iterator::{PyDataloaderIter, PyIterInner};
use crate::python::sampler::{PySampler, SharedPySampler, validate_python_sampler};

type CorePyLoader = core_loader::DataLoader<PyDataset, SharedPySampler, PyCollator>;

#[pyclass(name = "PyDataloader", module = "dataloader_rs", unsendable)]
pub struct PyDataloader {
    inner: CorePyLoader,
    /// The user's sampler, kept to answer `len(loader)` with `len(sampler)`.
    py_sampler: Option<Py<PyAny>>,
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
        drop_last=false,
        seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dataset: PyDataset,
        batch_size: usize,
        prefetch_depth: usize,
        shuffle: bool,
        sampler: Option<Py<PyAny>>,
        num_workers: usize,
        collate_fn: Option<Py<PyAny>>,
        drop_last: bool,
        seed: Option<u64>,
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
        if seed.is_some() && !shuffle {
            return Err(PyValueError::new_err("seed only applies when shuffle=True"));
        }

        let py_sampler = sampler.as_ref().map(|s| s.clone_ref(py));
        let sampler = match sampler {
            Some(py_sampler) => {
                validate_python_sampler(&py_sampler).map_err(into_py_err)?;
                SharedPySampler::new(PySampler::Python(py_sampler))
            }
            None if shuffle => {
                let sampler = seed.map_or_else(RandomSampler::from_entropy, RandomSampler::new);
                SharedPySampler::new(PySampler::Random(sampler))
            }
            None => SharedPySampler::new(PySampler::Sequential(SequentialSampler)),
        };

        // `num_workers` maps to batch-level workers; parallel_items stays off for
        // Python datasets (rayon cannot call Python's __getitem__ in parallel).
        let inner = core_loader::DataLoader::builder(dataset)
            .batch_size(batch_size)
            .prefetch_depth(prefetch_depth)
            .drop_last(drop_last)
            .num_workers(num_workers)
            .sampler(sampler)
            .collator(PyCollator::new(collate_fn))
            .build();

        Ok(Self { inner, py_sampler })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let mut loader = slf.borrow_mut(py);
        // `Dataset::len` cannot fail; surface a raising `__len__` here instead
        // of letting the core panic on it.
        len_py(loader.inner.dataset(), py)?;

        if !loader.inner.has_workers() {
            // Direct path (num_workers=0): call Python directly inside __next__
            // using the py token already held — zero extra GIL acquisitions.
            let chunks = loader.inner.epoch_chunks();
            let remaining = chunks.len();
            let fetcher = Fetcher::new(loader.inner.dataset(), py)?;
            let collator = loader.inner.collator().clone();
            drop(loader);
            return Ok(PyDataloaderIter {
                _owner: slf,
                inner: PyIterInner::Direct {
                    chunks: chunks.into_iter(),
                    remaining,
                    fetcher,
                    collator,
                },
            });
        }

        // Parallel path: core spawns N inter-batch worker threads.
        // Each worker calls dataset.get_batch() (one Python::attach per batch).
        let inner = loader.inner.iter();
        drop(loader);
        Ok(PyDataloaderIter {
            _owner: slf,
            inner: PyIterInner::Threaded(Some(inner)),
        })
    }

    /// Batches per epoch. With a Python sampler this is `len(sampler)`
    /// batched, so a sampler without `__len__` raises `TypeError` like
    /// PyTorch, and an exception from `__len__` propagates.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.py_sampler {
            Some(sampler) => Ok(self.inner.num_batches(sampler.bind(py).len()?)),
            None => {
                len_py(self.inner.dataset(), py)?;
                Ok(self.inner.batch_len())
            }
        }
    }
}
