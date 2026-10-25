use crate::loader as core_loader;
use crate::sampler::RandomSampler;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::python::collator::PyCollator;
use crate::python::dataset::{Fetcher, PyDataset, len_py};
use crate::python::iterator::{PyDataloaderIter, PyIterInner};
use crate::python::pool::run_attached;
use crate::python::sampler::PySampler;

type CorePyLoader = core_loader::DataLoader<PyDataset, PySampler, PyCollator>;

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
        prefetch_depth=2,
        shuffle=false,
        sampler=None,
        num_workers=0,
        collate_fn=None,
        drop_last=false,
        seed=None,
        items_in_parallel=1
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
        items_in_parallel: usize,
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
        if items_in_parallel == 0 {
            return Err(PyValueError::new_err("items_in_parallel must be > 0"));
        }
        if num_workers.max(1).checked_mul(items_in_parallel).is_none() {
            return Err(PyValueError::new_err(
                "num_workers * items_in_parallel is too large",
            ));
        }

        let sampler = match sampler {
            Some(sampler) => PySampler::python(py, sampler)?,
            None if shuffle => {
                PySampler::random(seed.map_or_else(RandomSampler::from_entropy, RandomSampler::new))
            }
            None => PySampler::sequential(),
        };

        // With items_in_parallel > 1, lanes on the pool call `__getitem__` for
        // the items of a batch concurrently, each attaching to the interpreter.
        let inner = core_loader::DataLoader::builder(dataset)
            .batch_size(batch_size)
            .prefetch_depth(prefetch_depth)
            .drop_last(drop_last)
            .num_workers(num_workers)
            .items_in_parallel(items_in_parallel)
            .thread_main(run_attached)
            .sampler(sampler)
            .collator(PyCollator::new(collate_fn))
            .build();

        Ok(Self { inner })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let mut loader = slf.borrow_mut(py);
        let core = &mut loader.inner;
        // `Dataset::len` cannot fail; surface a raising `__len__` here instead
        // of letting the core panic on it.
        len_py(core.dataset(), py)?;
        let chunks = core.epoch_chunks();
        if let Some(err) = core.sampler_mut().take_error() {
            return Err(err);
        }

        let inner = if core.has_pool() {
            // Core iterator: batch tasks on the loader's pool (one Python::attach
            // per batch via get_batch), or with items_in_parallel > 1 one attach
            // per item in the batch's lanes. `__next__` waits with the interpreter
            // released, which item lanes need even with num_workers=0.
            PyIterInner::Threaded(core.iter_chunks(chunks))
        } else {
            // Direct path (num_workers=0): call Python directly inside __next__
            // using the py token already held — zero extra GIL acquisitions.
            PyIterInner::Direct {
                chunks: chunks.into_iter(),
                fetcher: Fetcher::new(core.dataset(), py)?,
                collator: core.collator().clone_ref(py),
            }
        };
        drop(loader);
        Ok(PyDataloaderIter { _owner: slf, inner })
    }

    /// Batches per epoch. With a Python sampler this is `len(sampler)`
    /// batched, so a sampler without `__len__` raises `TypeError` like
    /// PyTorch, and an exception from `__len__` propagates.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        match self.inner.sampler().python_sampler() {
            Some(sampler) => Ok(self.inner.num_batches(sampler.bind(py).len()?)),
            None => {
                len_py(self.inner.dataset(), py)?;
                Ok(self.inner.batch_len())
            }
        }
    }
}
