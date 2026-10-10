use std::sync::{Arc, Mutex};

use crate::{
    error::Error,
    sampler::{RandomSampler, Sampler, SequentialSampler},
};
use pyo3::{prelude::*, types::PyIterator};

pub(crate) enum PySampler {
    Sequential(SequentialSampler),
    Random(RandomSampler),
    Python(Py<PyAny>),
}

impl Sampler for PySampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        match self {
            Self::Sequential(s) => s.indices(dataset_len),
            Self::Random(s) => s.indices(dataset_len),
            // Callers go through `SharedPySampler`, which records the error.
            Self::Python(py_sampler) => {
                Python::attach(|py| python_indices(py, py_sampler)).unwrap_or_default()
            }
        }
    }
}

/// Iterate a Python sampler, failing on the first exception or non-index item.
fn python_indices(py: Python<'_>, sampler: &Py<PyAny>) -> PyResult<Vec<usize>> {
    PyIterator::from_object(sampler.bind(py))?
        .map(|item| item?.extract::<usize>())
        .collect()
}

/// Sampler handed to the core loader.
///
/// [`Sampler::indices`] cannot fail, so an exception raised by a Python
/// sampler is stored here and re-raised by `PyDataloader.__iter__` via
/// [`take_error`](Self::take_error) instead of silently producing an empty
/// epoch.
#[derive(Clone)]
pub(crate) struct SharedPySampler {
    inner: Arc<Mutex<PySampler>>,
    error: Arc<Mutex<Option<PyErr>>>,
}

impl SharedPySampler {
    pub(crate) fn new(sampler: PySampler) -> Self {
        Self {
            inner: Arc::new(Mutex::new(sampler)),
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// The exception raised while producing the last epoch's indices, if any.
    pub(crate) fn take_error(&self) -> Option<PyErr> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

impl Sampler for SharedPySampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let PySampler::Python(py_sampler) = &*guard else {
            return guard.indices(dataset_len);
        };
        Python::attach(|py| python_indices(py, py_sampler)).unwrap_or_else(|err| {
            *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(err);
            Vec::new()
        })
    }
}

pub(crate) fn validate_python_sampler(sampler: &Py<PyAny>) -> Result<(), Error> {
    Python::attach(|py| {
        let _ = PyIterator::from_object(sampler.bind(py))?;
        Ok::<_, PyErr>(())
    })
    .map_err(|e| e.into())
}
