use pyo3::exceptions::{PyDeprecationWarning, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{PyTraverseError, PyVisit};

use crate::collator::VecCollator;
use crate::dataset::AsyncDataset;
use crate::loader::{
    AsyncSource, DataLoader, DataLoaderBuilder, LoaderSource, MAX_RUNTIME_THREADS, SyncSource,
};
use crate::python::collator::PyCollator;
use crate::python::dataset::{
    DatasetKind, Fetcher, PyAsyncItems, PyDataset, RustSource, dataset_kind, len_py,
};
use crate::python::iterator::{AsyncIter, BatchIndices, PyDataloaderIter, PyIterInner};
use crate::python::pool::{attach_global_pool, run_attached, track};
use crate::python::sampler::PySampler;
use crate::python::transform::{PyItem, PyTransform};
use crate::sampler::{RandomSampler, SequentialSampler};
use crate::transform::Transform;

type SyncLoader = DataLoader<SyncSource<PyDataset>, PySampler, PyCollator, PyTransform>;
type AsyncLoader<A> = DataLoader<AsyncSource<A>, PySampler, PyCollator, PyTransform>;

/// The core loader, by kind of dataset. For async ones, `dataset` is the Python object, for
/// `len()` and the garbage collector.
enum Core {
    /// A `PyDataset`: `__getitem__` on the loader's threads.
    Sync(SyncLoader),
    /// A `PyAsyncDataset`: coroutines on the loader's event loop.
    Async {
        loader: AsyncLoader<PyAsyncItems>,
        dataset: Py<PyAny>,
    },
    /// A Rust dataset: futures on the loader's runtime, items made `bytes` by the transform.
    Rust {
        loader: AsyncLoader<RustSource>,
        dataset: Py<PyAny>,
    },
}

/// Settings shared by both kinds of loaders.
struct Options {
    batch_size: usize,
    prefetch_depth: usize,
    drop_last: bool,
    num_workers: usize,
    item_concurrency: usize,
    /// Worker threads of an async loader's runtime; `None`: one per available CPU.
    runtime_threads: Option<usize>,
    sampler: PySampler,
    transform: PyTransform,
    collator: PyCollator,
}

impl Options {
    fn build<Src>(
        self,
        builder: DataLoaderBuilder<Src, SequentialSampler, VecCollator>,
    ) -> DataLoader<Src, PySampler, PyCollator, PyTransform>
    where
        Src: LoaderSource,
        PyTransform: Transform<Src::Item, Output = PyItem>,
    {
        let loader = builder
            .batch_size(self.batch_size)
            .prefetch_depth(self.prefetch_depth)
            .drop_last(self.drop_last)
            .num_workers(self.num_workers)
            .item_concurrency(self.item_concurrency)
            .thread_main(run_attached)
            .sampler(self.sampler)
            .transform(self.transform)
            .collator(self.collator)
            .build();
        if let Some(pool) = loader.pool_handle() {
            track(pool);
        }
        loader
    }

    /// An async loader over `dataset`.
    fn build_async<A>(self, dataset: A) -> AsyncLoader<A>
    where
        A: AsyncDataset,
        PyTransform: Transform<A::Item, Output = PyItem>,
    {
        let mut builder = DataLoader::builder_async(dataset);
        if let Some(threads) = self.runtime_threads {
            builder = builder.runtime_threads(threads);
        }
        self.build(builder)
    }
}

/// Pick the loader for `dataset`'s kind.
fn core_for(py: Python<'_>, dataset: &Bound<'_, PyAny>, options: Options) -> PyResult<Core> {
    match dataset_kind(dataset)? {
        DatasetKind::Rust(source) => Ok(Core::Rust {
            loader: options.build_async(source),
            dataset: dataset.clone().unbind(),
        }),
        DatasetKind::Sync(_) if options.runtime_threads.is_some() => Err(PyValueError::new_err(
            "runtime_threads only applies to async datasets \
             (PyAsyncDataset, HttpDataset, FileDataset, ObjectStoreDataset)",
        )),
        DatasetKind::Sync(sync) => Ok(Core::Sync(options.build(DataLoader::builder(sync)))),
        DatasetKind::Async { batched, aclose } => {
            let items = PyAsyncItems::new(py, dataset, batched, aclose)?;
            // Built before the loader, so that its threads attach once.
            attach_global_pool(py);
            Ok(Core::Async {
                loader: options.build_async(items),
                dataset: dataset.clone().unbind(),
            })
        }
    }
}

/// The epoch's indices, batch by batch, for `__next__` to return with each batch; `None`
/// unless `return_indices` is set.
fn batch_indices(return_indices: bool, chunks: &[Vec<usize>]) -> Option<BatchIndices> {
    return_indices.then(|| chunks.to_vec()).map(Vec::into_iter)
}

/// Start an epoch of an async loader, with its indices if `return_indices` is set.
fn start_async<A>(
    py: Python<'_>,
    loader: &mut AsyncLoader<A>,
    dataset: &Py<PyAny>,
    return_indices: bool,
) -> PyResult<(AsyncIter<A>, Option<BatchIndices>)>
where
    A: AsyncDataset,
    PyTransform: Transform<A::Item, Output = PyItem>,
{
    dataset.bind(py).len()?;
    let chunks = loader.epoch_chunks();
    if let Some(err) = loader.sampler_mut().take_error() {
        return Err(err);
    }
    let indices = batch_indices(return_indices, &chunks);
    let epoch = loader.iter_chunks(chunks);
    if let Some(pool) = loader.pool_handle() {
        track(pool);
    }
    Ok((epoch, indices))
}

/// The Python objects an async loader holds.
fn traverse_async<A>(
    visit: &PyVisit<'_>,
    loader: &AsyncLoader<A>,
    dataset: &Py<PyAny>,
) -> Result<(), PyTraverseError>
where
    A: AsyncDataset,
    PyTransform: Transform<A::Item, Output = PyItem>,
{
    visit.call(dataset)?;
    visit.call(loader.sampler().python_sampler())?;
    visit.call(loader.transform().function())?;
    visit.call(loader.collator().collate_fn())
}

/// Batches per epoch of an async loader; see `PyDataloader.__len__`.
fn len_async<A>(py: Python<'_>, loader: &AsyncLoader<A>, dataset: &Py<PyAny>) -> PyResult<usize>
where
    A: AsyncDataset,
    PyTransform: Transform<A::Item, Output = PyItem>,
{
    match loader.sampler().python_sampler() {
        Some(sampler) => Ok(loader.num_batches(sampler.bind(py).len()?)),
        None => {
            dataset.bind(py).len()?;
            Ok(loader.batch_len())
        }
    }
}

#[pyclass(name = "PyDataloader", module = "dataloader_rs")]
pub struct PyDataloader {
    inner: Core,
    /// Each batch is `(batch, indices)`.
    return_indices: bool,
}

/// An async dataset's `aclose()` is awaited here, on the thread that drops the
/// loader, before the drop returns. The event loop cannot wait for the last
/// reference to the dataset: tasks of a finished or dropped epoch may still
/// hold one for a moment, on their own threads.
impl Drop for PyDataloader {
    fn drop(&mut self) {
        if let Core::Async { loader, .. } = &self.inner {
            loader.dataset().close();
        }
    }
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
        item_concurrency=1,
        items_in_parallel=None,
        runtime_threads=None,
        transform=None,
        return_indices=false
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
        collate_fn: Option<Bound<'_, PyAny>>,
        drop_last: bool,
        seed: Option<u64>,
        mut item_concurrency: usize,
        items_in_parallel: Option<usize>,
        runtime_threads: Option<usize>,
        transform: Option<Bound<'_, PyAny>>,
        return_indices: bool,
    ) -> PyResult<Self> {
        if let Some(n) = items_in_parallel {
            PyErr::warn(
                py,
                &py.get_type::<PyDeprecationWarning>(),
                c"items_in_parallel is deprecated, use item_concurrency",
                1,
            )?;
            item_concurrency = n;
        }
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
        if item_concurrency == 0 {
            return Err(PyValueError::new_err("item_concurrency must be > 0"));
        }
        if runtime_threads.is_some_and(|n| !(1..=MAX_RUNTIME_THREADS).contains(&n)) {
            return Err(PyValueError::new_err(format!(
                "runtime_threads must be between 1 and {MAX_RUNTIME_THREADS}"
            )));
        }
        if transform.as_ref().is_some_and(|f| !f.is_callable()) {
            return Err(PyTypeError::new_err("transform must be callable"));
        }
        if num_workers.max(1).checked_mul(item_concurrency).is_none() {
            return Err(PyValueError::new_err(
                "num_workers * item_concurrency is too large",
            ));
        }

        let sampler = match sampler {
            Some(sampler) => PySampler::python(py, sampler)?,
            None if shuffle => {
                PySampler::random(seed.map_or_else(RandomSampler::from_entropy, RandomSampler::new))
            }
            None => PySampler::sequential(),
        };

        // With item_concurrency > 1, lanes on the pool call `__getitem__` for
        // the items of a batch concurrently, each attaching to the interpreter;
        // for an async dataset it is the number of coroutines or requests in
        // flight per batch.
        let options = Options {
            batch_size,
            prefetch_depth,
            drop_last,
            num_workers,
            item_concurrency,
            runtime_threads,
            sampler,
            transform: PyTransform::new(transform.as_ref()),
            collator: PyCollator::new(collate_fn.as_ref()),
        };
        Ok(Self {
            inner: core_for(py, &dataset, options)?,
            return_indices,
        })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let mut loader = slf.borrow_mut(py);
        let return_indices = loader.return_indices;
        let (inner, indices) = match &mut loader.inner {
            Core::Sync(core) => {
                // `Dataset::len` cannot fail; surface a raising `__len__` here instead
                // of letting the core panic on it.
                len_py(core.dataset(), py)?;
                let chunks = core.epoch_chunks();
                if let Some(err) = core.sampler_mut().take_error() {
                    return Err(err);
                }
                let indices = batch_indices(return_indices, &chunks);
                let inner = if core.has_pool() {
                    // Core iterator: batch tasks on the loader's pool (one Python::attach
                    // per batch via get_batch), or with item_concurrency > 1 one attach
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
                        transform: core.transform().clone_ref(py),
                        collator: core.collator().clone_ref(py),
                    }
                };
                (inner, indices)
            }
            Core::Async { loader, dataset } => {
                let (epoch, indices) = start_async(py, loader, dataset, return_indices)?;
                (PyIterInner::Async(epoch), indices)
            }
            Core::Rust { loader, dataset } => {
                let (epoch, indices) = start_async(py, loader, dataset, return_indices)?;
                (PyIterInner::Rust(epoch), indices)
            }
        };
        drop(loader);
        Ok(PyDataloaderIter::new(slf, inner, indices))
    }

    /// Tell the garbage collector which Python objects this loader holds, so
    /// a reference cycle through them (a `collate_fn` bound to an object that
    /// keeps the loader, a dataset that refers back to it) can be freed.
    /// Running epochs share these objects through `Arc`s and add no
    /// references of their own.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.inner {
            Core::Sync(core) => {
                visit.call(core.dataset())?;
                visit.call(core.sampler().python_sampler())?;
                visit.call(core.transform().function())?;
                visit.call(core.collator().collate_fn())
            }
            Core::Async { loader, dataset } => traverse_async(&visit, loader, dataset),
            Core::Rust { loader, dataset } => traverse_async(&visit, loader, dataset),
        }
    }

    /// Batches per epoch. With a Python sampler this is `len(sampler)`
    /// batched, so a sampler without `__len__` raises `TypeError` like
    /// PyTorch, and an exception from `__len__` propagates.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.inner {
            Core::Sync(core) => match core.sampler().python_sampler() {
                Some(sampler) => Ok(core.num_batches(sampler.bind(py).len()?)),
                None => {
                    len_py(core.dataset(), py)?;
                    Ok(core.batch_len())
                }
            },
            Core::Async { loader, dataset } => len_async(py, loader, dataset),
            Core::Rust { loader, dataset } => len_async(py, loader, dataset),
        }
    }
}
