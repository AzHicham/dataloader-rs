use pyo3::exceptions::{PyDeprecationWarning, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{PyTraverseError, PyVisit, intern};

use crate::collator::VecCollator;
use crate::dataset::AsyncDataset;
use crate::loader::{
    AsyncSource, DataLoader, DataLoaderBuilder, LoaderSource, MAX_RUNTIME_THREADS, SyncSource,
};
use crate::python::async_dataset::{PyAsyncDatasetBase, PyAsyncItems};
use crate::python::collator::PyCollator;
use crate::python::dataset::{Fetcher, PyDataset, PyDatasetBase, len_py};
use crate::python::iterator::{AsyncIter, PyDataloaderIter, PyIterInner};
use crate::python::pool::{attach_global_pool, run_attached, track};
use crate::python::rust_dataset::{PyRustDataset, RustSource};
use crate::python::sampler::PySampler;
use crate::python::transform::PyTransform;
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
        PyTransform: Transform<Src::Item, Output = Py<PyAny>>,
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
        PyTransform: Transform<A::Item, Output = Py<PyAny>>,
    {
        let mut builder = DataLoader::builder_async(dataset);
        if let Some(threads) = self.runtime_threads {
            builder = builder.runtime_threads(threads);
        }
        self.build(builder)
    }
}

/// Whether `dataset.<name>` exists and is an `async def` function.
fn is_async_method(dataset: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<bool>> {
    let py = dataset.py();
    match dataset.getattr_opt(name)? {
        Some(method) if method.is_truthy()? => {
            let inspect = py.import(intern!(py, "inspect"))?;
            let is_async = inspect
                .call_method1(intern!(py, "iscoroutinefunction"), (method,))?
                .is_truthy()?;
            Ok(Some(is_async))
        }
        _ => Ok(None),
    }
}

/// Check `dataset` against its base class and pick the loader for it.
fn core_for(py: Python<'_>, dataset: &Bound<'_, PyAny>, options: Options) -> PyResult<Core> {
    if let Ok(rust) = dataset.cast::<PyRustDataset>() {
        return Ok(Core::Rust {
            loader: options.build_async(rust.get().source()),
            dataset: dataset.clone().unbind(),
        });
    }
    let is_sync = dataset.is_instance_of::<PyDatasetBase>();
    let is_async = dataset.is_instance_of::<PyAsyncDatasetBase>();
    let kind = match (is_sync, is_async) {
        (true, true) => {
            return Err(PyTypeError::new_err(
                "a dataset subclasses PyDataset or PyAsyncDataset, not both",
            ));
        }
        (true, false) => "PyDataset",
        (false, true) => "PyAsyncDataset",
        (false, false) => {
            return Err(PyTypeError::new_err(format!(
                "dataset must be a PyDataset, a PyAsyncDataset or a Rust dataset \
                 (HttpDataset, FileDataset, ObjectStoreDataset), not {}",
                dataset.get_type().name()?
            )));
        }
    };
    let ty = dataset.get_type();
    let overrides_len = match is_sync {
        true => !ty.getattr(intern!(py, "__len__"))?.is(&py
            .get_type::<PyDatasetBase>()
            .getattr(intern!(py, "__len__"))?),
        false => ty.hasattr(intern!(py, "__len__"))?,
    };
    if !overrides_len {
        return Err(PyTypeError::new_err(format!(
            "{kind} subclasses must define __len__"
        )));
    }
    let getitem = is_async_method(dataset, "__getitem__")?;
    let getitems = is_async_method(dataset, "__getitems__")?;
    if is_sync {
        if getitem == Some(true) || getitems == Some(true) {
            return Err(PyTypeError::new_err(
                "PyDataset.__getitem__ / __getitems__ must not be async; \
                 subclass PyAsyncDataset for async def methods",
            ));
        }
        if options.runtime_threads.is_some() {
            return Err(PyValueError::new_err(
                "runtime_threads only applies to async datasets \
                 (PyAsyncDataset, HttpDataset, FileDataset, ObjectStoreDataset)",
            ));
        }
        let dataset: PyDataset = dataset.cast::<PyDatasetBase>()?.clone().unbind();
        return Ok(Core::Sync(options.build(DataLoader::builder(dataset))));
    }
    if getitem == Some(false) || getitems == Some(false) {
        return Err(PyTypeError::new_err(
            "PyAsyncDataset.__getitem__ / __getitems__ must be async def; \
             subclass PyDataset for plain methods",
        ));
    }
    if getitem.is_none() {
        return Err(PyTypeError::new_err(
            "PyAsyncDataset subclasses must define async def __getitem__",
        ));
    }
    let aclose = is_async_method(dataset, "aclose")?;
    if aclose == Some(false) {
        return Err(PyTypeError::new_err(
            "PyAsyncDataset.aclose must be async def",
        ));
    }
    let items = PyAsyncItems::new(py, dataset, getitems.is_some(), aclose.is_some())?;
    // Built before the loader, so that its threads attach once.
    attach_global_pool(py);
    Ok(Core::Async {
        loader: options.build_async(items),
        dataset: dataset.clone().unbind(),
    })
}

/// Start an epoch of an async loader.
fn start_async<A>(
    py: Python<'_>,
    loader: &mut AsyncLoader<A>,
    dataset: &Py<PyAny>,
) -> PyResult<AsyncIter<A>>
where
    A: AsyncDataset,
    PyTransform: Transform<A::Item, Output = Py<PyAny>>,
{
    dataset.bind(py).len()?;
    let chunks = loader.epoch_chunks();
    if let Some(err) = loader.sampler_mut().take_error() {
        return Err(err);
    }
    let epoch = loader.iter_chunks(chunks);
    if let Some(pool) = loader.pool_handle() {
        track(pool);
    }
    Ok(epoch)
}

/// The Python objects an async loader holds.
fn traverse_async<A>(
    visit: &PyVisit<'_>,
    loader: &AsyncLoader<A>,
    dataset: &Py<PyAny>,
) -> Result<(), PyTraverseError>
where
    A: AsyncDataset,
    PyTransform: Transform<A::Item, Output = Py<PyAny>>,
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
    PyTransform: Transform<A::Item, Output = Py<PyAny>>,
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
        transform=None
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
        mut item_concurrency: usize,
        items_in_parallel: Option<usize>,
        runtime_threads: Option<usize>,
        transform: Option<Bound<'_, PyAny>>,
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
            transform: PyTransform::new(transform.map(Bound::unbind)),
            collator: PyCollator::new(collate_fn),
        };
        Ok(Self {
            inner: core_for(py, &dataset, options)?,
        })
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<PyDataloaderIter> {
        let mut loader = slf.borrow_mut(py);
        let inner = match &mut loader.inner {
            Core::Sync(core) => {
                // `Dataset::len` cannot fail; surface a raising `__len__` here instead
                // of letting the core panic on it.
                len_py(core.dataset(), py)?;
                let chunks = core.epoch_chunks();
                if let Some(err) = core.sampler_mut().take_error() {
                    return Err(err);
                }
                if core.has_pool() {
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
                }
            }
            Core::Async { loader, dataset } => {
                PyIterInner::Async(start_async(py, loader, dataset)?)
            }
            Core::Rust { loader, dataset } => PyIterInner::Rust(start_async(py, loader, dataset)?),
        };
        drop(loader);
        Ok(PyDataloaderIter::new(slf, inner))
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
