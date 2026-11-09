//! `transform=`: the Python function the loader applies to every item.

use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{error::Result, transform::Transform};

/// `PyDataloader(transform=f)`: `f(index, item)` for each item, on the loader's threads,
/// holding the interpreter only for the call. Without `f`, items are left as they are, except a Rust
/// dataset's, which become `bytes`.
pub(crate) struct PyTransform {
    function: Option<Py<PyAny>>,
}

impl PyTransform {
    pub(crate) fn new(function: Option<Py<PyAny>>) -> Self {
        Self { function }
    }

    /// The user's function, if any.
    pub(crate) fn function(&self) -> Option<&Py<PyAny>> {
        self.function.as_ref()
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            function: self.function.as_ref().map(|f| f.clone_ref(py)),
        }
    }

    /// Transform `item`, the item at `index`, with an already-held `py`.
    pub(crate) fn call(
        &self,
        py: Python<'_>,
        index: usize,
        item: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match &self.function {
            None => Ok(item),
            Some(f) => f.call1(py, (index, item)),
        }
    }
}

/// The items of a `PyDataset` or a `PyAsyncDataset`.
impl Transform<Py<PyAny>> for PyTransform {
    type Output = Py<PyAny>;

    fn apply(&self, index: usize, item: Py<PyAny>) -> Result<Py<PyAny>> {
        match &self.function {
            None => Ok(item),
            Some(f) => Ok(Python::attach(|py| f.call1(py, (index, item)))?),
        }
    }

    fn is_identity(&self) -> bool {
        self.function.is_none()
    }
}

/// The items of a Rust dataset: `bytes`, then `f(index, bytes)`.
impl Transform<Bytes> for PyTransform {
    type Output = Py<PyAny>;

    fn apply(&self, index: usize, item: Bytes) -> Result<Py<PyAny>> {
        Ok(Python::attach(|py| {
            let item = PyBytes::new(py, &item).into_any().unbind();
            self.call(py, index, item)
        })?)
    }
}
