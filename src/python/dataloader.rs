use std::collections::VecDeque;
use std::sync::Arc;

use crate::loader as core_loader;
use crate::sampler::{RandomSampler, SequentialSampler};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

use crate::python::collator::PyCollator;
use crate::python::dataset::{Fetcher, PyDataset, len_py};
use crate::python::into_py_err;
use crate::python::iterator::{PyDataloaderIter, PyIterInner, preserving_exception};
use crate::python::sampler::{
    PyDistributedSampler, PySampler, SharedPySampler, validate_python_sampler,
};

type CorePyLoader = core_loader::DataLoader<PyDataset, SharedPySampler, PyCollator>;

#[pyclass(name = "PyDataloader", module = "dataloader_rs", unsendable)]
pub struct PyDataloader {
    inner: CorePyLoader,
    /// Shared with `inner`; reports exceptions raised by a Python sampler.
    sampler: SharedPySampler,
    /// Event loop thread for async datasets, created on first use and kept
    /// for the loader's lifetime (see `dataloader_rs._async`).
    async_loop: Option<Py<PyAny>>,
    /// `asyncio.Semaphore(max_concurrency)` shared by all epochs, if set.
    async_limit: Option<Py<PyAny>>,
    num_workers: usize,
    prefetch_depth: usize,
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
        seed=None,
        max_concurrency=None
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
        max_concurrency: Option<usize>,
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
        let async_limit = match max_concurrency {
            None => None,
            Some(0) => return Err(PyValueError::new_err("max_concurrency must be > 0")),
            Some(n) => {
                if !Fetcher::new(&dataset, py)?.is_async(py)? {
                    return Err(PyValueError::new_err(
                        "max_concurrency requires an async def __getitem__ or __getitems__",
                    ));
                }
                let semaphore = py
                    .import(intern!(py, "asyncio"))?
                    .getattr(intern!(py, "Semaphore"))?
                    .call1((n,))?;
                Some(semaphore.unbind())
            }
        };

        let sampler = match sampler {
            Some(py_sampler) => {
                // Our own DistributedSampler runs natively: no Python call
                // per epoch, and set_epoch on it reaches the loader.
                let native = Python::attach(|py| {
                    py_sampler
                        .bind(py)
                        .cast::<PyDistributedSampler>()
                        .ok()
                        .map(|s| Arc::clone(&s.get().inner))
                });
                if let Some(shared) = native {
                    SharedPySampler::new(PySampler::Distributed(shared))
                } else {
                    validate_python_sampler(&py_sampler).map_err(into_py_err)?;
                    SharedPySampler::new(PySampler::Python(py_sampler))
                }
            }
            None if shuffle => {
                let sampler = seed.map_or_else(RandomSampler::from_entropy, RandomSampler::new);
                SharedPySampler::new(PySampler::Random(sampler))
            }
            None => SharedPySampler::new(PySampler::Sequential(SequentialSampler)),
        };

        // `num_workers` maps to inter-batch workers; intra_workers stays 0 for
        // Python datasets (rayon cannot call Python's __getitem__ in parallel).
        let inner = core_loader::DataLoader::builder(dataset)
            .batch_size(batch_size)
            .prefetch_depth(prefetch_depth)
            .drop_last(drop_last)
            .num_workers(num_workers)
            .sampler(sampler.clone())
            .collator(PyCollator::new(collate_fn))
            .build();

        Ok(Self {
            inner,
            sampler,
            async_loop: None,
            async_limit,
            num_workers,
            prefetch_depth,
        })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let mut loader = slf.borrow_mut(py);
        // `Dataset::len` cannot fail; surface a raising `__len__` here instead
        // of letting the core panic on it.
        len_py(loader.inner.dataset(), py)?;

        let fetcher = Fetcher::new(loader.inner.dataset(), py)?;
        if fetcher.is_async(py)? {
            // Async dataset: batches are awaited concurrently on one event
            // loop thread; worker threads are not used.
            let async_mod = py.import(intern!(py, "dataloader_rs._async"))?;
            let runner = match &loader.async_loop {
                Some(runner) => runner.clone_ref(py),
                None => {
                    let runner = async_mod
                        .getattr(intern!(py, "LoopThread"))?
                        .call0()?
                        .unbind();
                    loader.async_loop = Some(runner.clone_ref(py));
                    runner
                }
            };
            let chunks = loader.inner.epoch_chunks();
            if let Some(err) = loader.sampler.take_error() {
                return Err(err);
            }
            let window = loader.num_workers.max(1) + loader.prefetch_depth;
            let collator = loader.inner.collator().clone();
            let limit = loader.async_limit.as_ref().map(|l| l.clone_ref(py));
            drop(loader);
            return Ok(PyDataloaderIter {
                _owner: slf,
                inner: PyIterInner::Async {
                    limit,
                    remaining: chunks.len(),
                    chunks: chunks.into_iter(),
                    pending: VecDeque::with_capacity(window),
                    window,
                    runner,
                    fetch_batch: async_mod.getattr(intern!(py, "fetch_batch"))?.unbind(),
                    batched: fetcher.is_batched(),
                    method: fetcher.method().clone_ref(py),
                    collator,
                },
            });
        }

        if !loader.inner.has_workers() {
            // Direct path (num_workers=0): call Python directly inside __next__
            // using the py token already held — zero extra GIL acquisitions.
            let chunks = loader.inner.epoch_chunks();
            if let Some(err) = loader.sampler.take_error() {
                return Err(err);
            }
            let remaining = chunks.len();
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
        if let Some(err) = loader.sampler.take_error() {
            // Join the (idle) workers with the GIL released.
            py.detach(|| drop(inner));
            return Err(err);
        }
        drop(loader);
        Ok(PyDataloaderIter {
            _owner: slf,
            inner: PyIterInner::Threaded(Some(inner)),
        })
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        len_py(self.inner.dataset(), py)?;
        Ok(self.inner.batch_len())
    }
}

impl Drop for PyDataloader {
    fn drop(&mut self) {
        if let Some(runner) = self.async_loop.take() {
            Python::attach(|py| {
                preserving_exception(py, || {
                    let _ = runner.call_method0(py, intern!(py, "close"));
                });
            });
        }
    }
}
