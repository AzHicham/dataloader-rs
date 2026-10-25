use crate::{collator::Collator, error::Result};
use pyo3::{prelude::*, types::PyList};

pub(crate) enum PyBatch {
    Ready(Py<PyAny>),
    Items(Vec<Py<PyAny>>),
}

impl PyBatch {
    /// The object `__next__` returns: `collate_fn`'s result, or a `list`.
    pub(crate) fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Ready(obj) => Ok(obj),
            Self::Items(items) => Ok(PyList::new(py, items)?.into_any().unbind()),
        }
    }
}

pub(crate) struct PyCollator {
    collate_fn: Option<Py<PyAny>>,
}

impl PyCollator {
    pub(crate) fn new(collate_fn: Option<Py<PyAny>>) -> Self {
        Self { collate_fn }
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            collate_fn: self.collate_fn.as_ref().map(|f| f.clone_ref(py)),
        }
    }

    /// Like `collate` but uses an already-held `py` token — no `Python::attach()` overhead.
    pub(crate) fn collate_with_py(&self, py: Python<'_>, items: Vec<Py<PyAny>>) -> Result<PyBatch> {
        match &self.collate_fn {
            None => Ok(PyBatch::Items(items)),
            Some(f) => {
                let list = PyList::new(py, items)?;
                Ok(PyBatch::Ready(f.call1(py, (list,))?))
            }
        }
    }
}

impl Collator<Py<PyAny>> for PyCollator {
    type Batch = PyBatch;

    fn collate(&self, items: Vec<Py<PyAny>>) -> Result<Self::Batch> {
        match &self.collate_fn {
            // Fast path: keep raw items and convert to Python list in __next__,
            // avoiding Python API work in the worker thread.
            None => Ok(PyBatch::Items(items)),
            Some(_) => Python::attach(|py| self.collate_with_py(py, items)),
        }
    }
}
