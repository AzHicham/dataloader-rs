use crate::sampler::{RandomSampler, Sampler, SequentialSampler};
use pyo3::{prelude::*, types::PyIterator};

/// The loader's sampler: sequential, shuffled by the Rust `RandomSampler`,
/// or a Python iterable of indices.
///
/// [`Sampler::indices`] cannot fail, so an exception raised by a Python
/// sampler is kept here and re-raised by `PyDataloader.__iter__` via
/// [`take_error`](Self::take_error) instead of silently producing an empty
/// epoch.
pub(crate) struct PySampler {
    order: Order,
    error: Option<PyErr>,
}

enum Order {
    Sequential,
    Random(RandomSampler),
    Python(Py<PyAny>),
}

impl PySampler {
    pub(crate) fn sequential() -> Self {
        Self::new(Order::Sequential)
    }

    pub(crate) fn random(sampler: RandomSampler) -> Self {
        Self::new(Order::Random(sampler))
    }

    /// A Python sampler: any iterable of `int`; anything else raises
    /// `TypeError`, as in PyTorch.
    pub(crate) fn python(py: Python<'_>, sampler: Py<PyAny>) -> PyResult<Self> {
        PyIterator::from_object(sampler.bind(py))?;
        Ok(Self::new(Order::Python(sampler)))
    }

    fn new(order: Order) -> Self {
        Self { order, error: None }
    }

    /// The user's sampler object, if one was given.
    pub(crate) fn python_sampler(&self) -> Option<&Py<PyAny>> {
        match &self.order {
            Order::Python(sampler) => Some(sampler),
            Order::Sequential | Order::Random(_) => None,
        }
    }

    /// The exception raised while producing the last epoch's indices, if any.
    pub(crate) fn take_error(&mut self) -> Option<PyErr> {
        self.error.take()
    }
}

/// Indices read from a Python sampler between two runs of the signal handlers.
const SIGNAL_CHECK_EVERY: usize = 1 << 16;

/// Iterate a Python sampler, failing on the first exception or non-index
/// item. An epoch's indices are all read up front, so a sampler that never
/// ends would fill memory: signal handlers run every 65,536 indices, so
/// Ctrl-C stops it even when it runs no bytecode (`itertools.cycle`), where
/// Python would not check them itself.
fn python_indices(py: Python<'_>, sampler: &Py<PyAny>) -> PyResult<Vec<usize>> {
    let mut indices = Vec::new();
    for item in PyIterator::from_object(sampler.bind(py))? {
        if indices.len() % SIGNAL_CHECK_EVERY == 0 {
            py.check_signals()?;
        }
        indices.push(item?.extract::<usize>()?);
    }
    Ok(indices)
}

// `len` keeps the trait default, which is right for the built-in samplers.
// `PyDataloader.__len__` calls `len()` on a Python sampler itself so that a
// missing or raising `__len__` reaches Python instead of being guessed.
impl Sampler for PySampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        match &mut self.order {
            Order::Sequential => SequentialSampler.indices(dataset_len),
            Order::Random(sampler) => sampler.indices(dataset_len),
            Order::Python(sampler) => Python::attach(|py| python_indices(py, sampler))
                .unwrap_or_else(|err| {
                    // Re-raised by `PyDataloader.__iter__`; the empty epoch is
                    // never iterated.
                    self.error = Some(err);
                    Vec::new()
                }),
        }
    }
}
