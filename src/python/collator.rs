use crate::{collator::Collator, error::Result};
use pyo3::{prelude::*, types::PyList};

pub(crate) enum PyBatch {
    Ready(Py<PyAny>),
    Items(Vec<Py<PyAny>>),
}

pub(crate) struct PyCollator {
    collate_fn: Option<Py<PyAny>>,
}

impl PyCollator {
    pub(crate) fn new(collate_fn: Option<Py<PyAny>>) -> Self {
        Self { collate_fn }
    }
}

impl Clone for PyCollator {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            collate_fn: self.collate_fn.as_ref().map(|f| f.clone_ref(py)),
        })
    }
}

impl Collator<Py<PyAny>> for PyCollator {
    type Batch = PyBatch;

    fn collate(&self, items: Vec<Py<PyAny>>) -> Result<Self::Batch> {
        match &self.collate_fn {
            // Fast path: keep raw items and convert to Python list in __next__,
            // avoiding Python API work in the worker thread.
            None => Ok(PyBatch::Items(items)),
            Some(f) => Python::attach(|py| {
                let items = PyList::new(py, items)?;
                f.call1(py, (items,)).map(PyBatch::Ready)
            })
            .map_err(|e| e.into()),
        }
    }
}
