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

impl Sampler for SharedPySampler {
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
