use std::sync::{Arc, Mutex};

use crate::{
    error::Error,
    sampler::{DistributedSampler, RandomSampler, Sampler, SequentialSampler},
};
use pyo3::{
    exceptions::PyValueError,
    intern,
    prelude::*,
    types::{PyIterator, PyList},
};

pub(crate) enum PySampler {
    Sequential(SequentialSampler),
    Random(RandomSampler),
    Python(Py<PyAny>),
    /// Shared with the Python `DistributedSampler` object, so `set_epoch`
    /// called from Python applies to the loader's next epoch.
    Distributed(SharedDistributed),
}

/// Order a [`DistributedSampler`] shards: sequential or shuffled.
pub(crate) enum BaseSampler {
    Sequential(SequentialSampler),
    Random(RandomSampler),
}

impl Sampler for BaseSampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        match self {
            Self::Sequential(s) => s.indices(dataset_len),
            Self::Random(s) => s.indices(dataset_len),
        }
    }

    fn set_epoch(&mut self, epoch: u64) {
        match self {
            Self::Sequential(s) => s.set_epoch(epoch),
            Self::Random(s) => s.set_epoch(epoch),
        }
    }
}

type SharedDistributed = Arc<Mutex<DistributedSampler<BaseSampler>>>;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Python-facing `DistributedSampler`, backed by the Rust one.
///
/// Same constructor as `torch.utils.data.DistributedSampler`. Pass it as
/// `sampler=` to `PyDataloader` and call `set_epoch(epoch)` before each epoch.
#[pyclass(name = "DistributedSampler", module = "dataloader_rs", frozen)]
pub struct PyDistributedSampler {
    pub(crate) inner: SharedDistributed,
    dataset: Py<PyAny>,
}

#[pymethods]
impl PyDistributedSampler {
    #[new]
    #[pyo3(signature = (dataset, num_replicas=None, rank=None, shuffle=true, seed=0, drop_last=false))]
    fn new(
        py: Python<'_>,
        dataset: Py<PyAny>,
        num_replicas: Option<usize>,
        rank: Option<usize>,
        shuffle: bool,
        seed: u64,
        drop_last: bool,
    ) -> PyResult<Self> {
        let (num_replicas, rank) = match (num_replicas, rank) {
            (Some(n), Some(r)) => (n, r),
            (n, r) => {
                let (world, current) = torch_distributed_world(py)?;
                (n.unwrap_or(world), r.unwrap_or(current))
            }
        };
        if num_replicas == 0 {
            return Err(PyValueError::new_err("num_replicas must be > 0"));
        }
        if rank >= num_replicas {
            return Err(PyValueError::new_err(format!(
                "Invalid rank {rank}, rank should be in the interval [0, {}]",
                num_replicas - 1
            )));
        }
        let base = if shuffle {
            BaseSampler::Random(RandomSampler::new(seed))
        } else {
            BaseSampler::Sequential(SequentialSampler)
        };
        let sampler = DistributedSampler::new(base, rank, num_replicas).drop_last(drop_last);
        Ok(Self {
            inner: Arc::new(Mutex::new(sampler)),
            dataset,
        })
    }

    /// Set the epoch used to shuffle; call it before each epoch so every rank
    /// draws the same new order.
    fn set_epoch(&self, epoch: u64) {
        lock(&self.inner).set_epoch(epoch);
    }

    #[getter]
    fn num_replicas(&self) -> usize {
        lock(&self.inner).world_size()
    }

    #[getter]
    fn rank(&self) -> usize {
        lock(&self.inner).rank()
    }

    #[getter]
    fn epoch(&self) -> u64 {
        lock(&self.inner).epoch()
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        let dataset_len = self.dataset.bind(py).len()?;
        Ok(lock(&self.inner).len(dataset_len))
    }

    /// This rank's indices for the current epoch (does not advance the epoch).
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let dataset_len = self.dataset.bind(py).len()?;
        let indices = lock(&self.inner).indices(dataset_len);
        PyList::new(py, indices)?.try_iter()
    }
}

/// `(world_size, rank)` from an initialized `torch.distributed` process group.
fn torch_distributed_world(py: Python<'_>) -> PyResult<(usize, usize)> {
    let missing = || {
        PyValueError::new_err(
            "num_replicas and rank are required unless torch.distributed is initialized",
        )
    };
    let Ok(dist) = py.import(intern!(py, "torch.distributed")) else {
        return Err(missing());
    };
    let ready = dist
        .call_method0(intern!(py, "is_available"))?
        .is_truthy()?
        && dist
            .call_method0(intern!(py, "is_initialized"))?
            .is_truthy()?;
    if !ready {
        return Err(missing());
    }
    Ok((
        dist.call_method0(intern!(py, "get_world_size"))?
            .extract()?,
        dist.call_method0(intern!(py, "get_rank"))?.extract()?,
    ))
}

impl Sampler for PySampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        match self {
            Self::Sequential(s) => s.indices(dataset_len),
            Self::Random(s) => s.indices(dataset_len),
            Self::Python(py_sampler) => Python::attach(|py| {
                let iter = match PyIterator::from_object(py_sampler.bind(py)) {
                    Ok(i) => i,
                    Err(_) => return Vec::new(),
                };
                let mut indices = Vec::new();
                for item in iter {
                    let Ok(item) = item else { return Vec::new() };
                    let Ok(idx) = item.extract::<usize>() else {
                        return Vec::new();
                    };
                    indices.push(idx);
                }
                indices
            }),
            Self::Distributed(sampler) => lock(sampler).indices(dataset_len),
        }
    }
}

#[derive(Clone)]
pub(crate) struct SharedPySampler {
    inner: Arc<Mutex<PySampler>>,
}

impl SharedPySampler {
    pub(crate) fn new(sampler: PySampler) -> Self {
        Self {
            inner: Arc::new(Mutex::new(sampler)),
        }
    }
}

impl PySampler {
    /// `len(sampler)` for a Python sampler that defines `__len__`; otherwise
    /// the dataset length (a sampler without `__len__` cannot be measured
    /// without consuming it).
    fn py_len(&self, dataset_len: usize) -> usize {
        match self {
            Self::Python(py_sampler) => {
                Python::attach(|py| py_sampler.bind(py).len()).unwrap_or(dataset_len)
            }
            Self::Distributed(sampler) => lock(sampler).len(dataset_len),
            _ => dataset_len,
        }
    }
}

impl Sampler for SharedPySampler {
    fn len(&self, dataset_len: usize) -> usize {
        match self.inner.lock() {
            Ok(guard) => guard.py_len(dataset_len),
            Err(_) => dataset_len,
        }
    }

    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        let Ok(mut guard) = self.inner.lock() else {
            return Vec::new();
        };
        guard.indices(dataset_len)
    }
}

pub(crate) fn validate_python_sampler(sampler: &Py<PyAny>) -> Result<(), Error> {
    Python::attach(|py| {
        let _ = PyIterator::from_object(sampler.bind(py))?;
        Ok::<_, PyErr>(())
    })
    .map_err(|e| e.into())
}
