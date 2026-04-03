use crate::{collator::Collator, error::Result};
use pyo3::{prelude::*, types::PyList};

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
    type Batch = Py<PyAny>;

    fn collate(&self, items: Vec<Py<PyAny>>) -> Result<Self::Batch> {
        Python::attach(|py| {
            let items = PyList::new(py, items)?;
            match &self.collate_fn {
                Some(f) => f.call1(py, (items,)),
                None => Ok(items.unbind().into_any()),
            }
        })
        .map_err(|e| e.into())
    }
}
