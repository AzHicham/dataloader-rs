use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::loader as core_loader;
use crate::sampler::RandomSampler;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{PyTraverseError, PyVisit};

use crate::python::collator::PyCollator;
use crate::python::dataset::{Fetcher, PyDataset, len_py};
use crate::python::image::{ImagePoolCollator, ImageSource, PyImageCollator};
use crate::python::iterator::{ImageEpoch, PyDataloaderIter, PyIterInner};
use crate::python::pool::{run_attached, track};
use crate::python::sampler::PySampler;

type CorePyLoader = core_loader::DataLoader<PyDataset, PySampler, PyCollator>;
type CoreImageLoader = core_loader::DataLoader<ImageSource, PySampler, ImagePoolCollator>;

/// A loader whose batches an `ImageCollator` builds in Rust.
pub(crate) struct ImageLoader {
    core: CoreImageLoader,
    sample_shape: [usize; 3],
    /// `ImageBatch`es alive, against `max_held_batches`.
    alive: Arc<AtomicUsize>,
    max_held_batches: usize,
}

enum Mode {
    /// Python samples, a Python `collate_fn` or lists.
    Python(CorePyLoader),
    /// Encoded images, decoded and collated in Rust.
    Image(ImageLoader),
}

#[pyclass(name = "PyDataloader", module = "dataloader_rs")]
pub struct PyDataloader {
    mode: Mode,
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
        items_in_parallel=1,
        max_held_batches=2
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dataset: Bound<'_, PyAny>,
        batch_size: usize,
        prefetch_depth: usize,
        shuffle: bool,
        sampler: Option<Py<PyAny>>,
        num_workers: usize,
        collate_fn: Option<Py<PyAny>>,
        drop_last: bool,
        seed: Option<u64>,
        items_in_parallel: usize,
        max_held_batches: usize,
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
        if max_held_batches == 0 {
            return Err(PyValueError::new_err("max_held_batches must be > 0"));
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

        let image_collator = collate_fn
            .as_ref()
            .and_then(|f| f.bind(py).cast::<PyImageCollator>().ok().map(Bound::get));
        if let Some(image_collator) = image_collator {
            // Decoding, transforms and collation run in Rust, on threads detached from the
            // interpreter; Python only runs to fetch the encoded bytes of a Python dataset.
            let collator = image_collator.collator()?;
            let allocator = collator
                .allocator(batch_size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let sample_shape = collator.sample_shape();
            let core = core_loader::DataLoader::builder(ImageSource::new(&dataset)?)
                .batch_size(batch_size)
                .prefetch_depth(prefetch_depth)
                .drop_last(drop_last)
                .num_workers(num_workers)
                .items_in_parallel(items_in_parallel)
                .thread_main(run_attached)
                .sampler(sampler)
                .collate_into(collator, allocator)
                .max_held_batches(max_held_batches)
                .build();
            if let Some(pool) = core.pool_handle() {
                track(pool);
            }
            return Ok(Self {
                mode: Mode::Image(ImageLoader {
                    core,
                    sample_shape,
                    alive: Arc::new(AtomicUsize::new(0)),
                    max_held_batches,
                }),
            });
        }

        let dataset: PyDataset = dataset.extract()?;
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
        if let Some(pool) = inner.pool_handle() {
            track(pool);
        }

        Ok(Self {
            mode: Mode::Python(inner),
        })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let mut loader = slf.borrow_mut(py);
        let inner = match &mut loader.mode {
            Mode::Python(core) => {
                // `Dataset::len` cannot fail; surface a raising `__len__` here instead
                // of letting the core panic on it.
                len_py(core.dataset(), py)?;
                let chunks = core.epoch_chunks();
                if let Some(err) = core.sampler_mut().take_error() {
                    return Err(err);
                }

                if core.has_pool() {
                    // Core iterator: batch tasks on the loader's pool (one Python::attach
                    // per batch via get_batch), or with items_in_parallel > 1 one attach
                    // per item in the batch's lanes. `__next__` waits with the interpreter
                    // released, which item lanes need even with num_workers=0.
                    let epoch = core.iter_chunks(chunks);
                    // A forked child gets a new pool on its first epoch.
                    if let Some(pool) = core.pool_handle() {
                        track(pool);
                    }
                    PyIterInner::Threaded(epoch)
                } else {
                    // Direct path (num_workers=0): call Python directly inside __next__
                    // using the py token already held — zero extra GIL acquisitions.
                    PyIterInner::Direct {
                        chunks: chunks.into_iter(),
                        fetcher: Fetcher::new(core.dataset(), py)?,
                        collator: core.collator().clone_ref(py),
                    }
                }
            }
            Mode::Image(image) => {
                let core = &mut image.core;
                if let Some(dataset) = core.dataset().python() {
                    len_py(dataset, py)?;
                }
                let chunks = core.epoch_chunks();
                if let Some(err) = core.sampler_mut().take_error() {
                    return Err(err);
                }
                // Always the core iterator: without workers `__next__` runs the batch
                // itself, detached from the interpreter.
                let epoch = core.iter_chunks(chunks);
                if let Some(pool) = core.pool_handle() {
                    track(pool);
                }
                PyIterInner::Image(ImageEpoch {
                    epoch,
                    sample_shape: image.sample_shape,
                    alive: Arc::clone(&image.alive),
                    max_held_batches: image.max_held_batches,
                })
            }
        };
        drop(loader);
        Ok(PyDataloaderIter { _owner: slf, inner })
    }

    /// Tell the garbage collector which Python objects this loader holds, so
    /// a reference cycle through them (a `collate_fn` bound to an object that
    /// keeps the loader, a dataset that refers back to it) can be freed.
    /// Running epochs share these objects through `Arc`s and add no
    /// references of their own.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.mode {
            Mode::Python(inner) => {
                visit.call(inner.dataset())?;
                visit.call(inner.sampler().python_sampler())?;
                visit.call(inner.collator().collate_fn())
            }
            Mode::Image(image) => {
                image.core.dataset().traverse(&visit)?;
                visit.call(image.core.sampler().python_sampler())
            }
        }
    }

    /// Batches per epoch. With a Python sampler this is `len(sampler)`
    /// batched, so a sampler without `__len__` raises `TypeError` like
    /// PyTorch, and an exception from `__len__` propagates.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.mode {
            Mode::Python(inner) => match inner.sampler().python_sampler() {
                Some(sampler) => Ok(inner.num_batches(sampler.bind(py).len()?)),
                None => {
                    len_py(inner.dataset(), py)?;
                    Ok(inner.batch_len())
                }
            },
            Mode::Image(image) => {
                let core = &image.core;
                match core.sampler().python_sampler() {
                    Some(sampler) => Ok(core.num_batches(sampler.bind(py).len()?)),
                    None => {
                        if let Some(dataset) = core.dataset().python() {
                            len_py(dataset, py)?;
                        }
                        Ok(core.batch_len())
                    }
                }
            }
        }
    }
}
