//! `transform=`: what the loader applies to every item, a Python function or the Rust image
//! transforms.

use std::sync::Arc;

use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::python::error::into_py_err;
use crate::python::image::{Op, PyImageTransform, Value, apply_all, object_of, value_of};
use crate::{error::Result, python::pool::attach, transform::Transform};

/// An item between the transform and the collator: a Python object, or a value of the Rust
/// image transforms, made a Python object only if the collator needs one.
pub(crate) enum PyItem {
    Object(Py<PyAny>),
    Native(Value),
}

impl PyItem {
    /// The item as a Python object.
    pub(crate) fn into_object(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Object(object) => Ok(object),
            Self::Native(value) => object_of(py, value),
        }
    }
}

/// `PyDataloader(transform=...)`.
pub(crate) enum PyTransform {
    /// No transform: items are left as they are, except a Rust dataset's, which become `bytes`.
    Identity,
    /// `f(index, item)` for each item, on the loader's threads, holding the interpreter only
    /// for the call.
    Function(Py<PyAny>),
    /// A `dataloader_rs.transforms` object: run in Rust, without the interpreter except to
    /// read a Python item's bytes.
    Native(Arc<[Op]>),
}

impl PyTransform {
    pub(crate) fn new(transform: Option<&Bound<'_, PyAny>>) -> Self {
        match transform {
            None => Self::Identity,
            Some(t) => match t.cast::<PyImageTransform>() {
                Ok(native) => Self::Native(native.get().ops()),
                Err(_) => Self::Function(t.clone().unbind()),
            },
        }
    }

    /// The user's Python function, if any.
    pub(crate) fn function(&self) -> Option<&Py<PyAny>> {
        match self {
            Self::Function(f) => Some(f),
            _ => None,
        }
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Function(f) => Self::Function(f.clone_ref(py)),
            Self::Native(ops) => Self::Native(Arc::clone(ops)),
        }
    }

    /// Transform `item`, the item at `index`, with an already-held `py`.
    pub(crate) fn call(&self, py: Python<'_>, index: usize, item: Py<PyAny>) -> PyResult<PyItem> {
        match self {
            Self::Identity => Ok(PyItem::Object(item)),
            Self::Function(f) => Ok(PyItem::Object(f.call1(py, (index, item))?)),
            Self::Native(ops) => {
                let value = value_of(item.bind(py))?;
                let value = py
                    .detach(|| apply_all(ops, index, value))
                    .map_err(into_py_err)?;
                Ok(PyItem::Native(value))
            }
        }
    }
}

/// The items of a `PyDataset` or a `PyAsyncDataset`.
impl Transform<Py<PyAny>> for PyTransform {
    type Output = PyItem;

    fn apply(&self, index: usize, item: Py<PyAny>) -> Result<PyItem> {
        match self {
            Self::Identity => Ok(PyItem::Object(item)),
            Self::Function(f) => Ok(PyItem::Object(attach(|py| f.call1(py, (index, item)))?)),
            Self::Native(ops) => {
                let value = attach(|py| value_of(item.bind(py)))?;
                Ok(PyItem::Native(apply_all(ops, index, value)?))
            }
        }
    }

    fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

/// The items of a Rust dataset: `bytes`, then `f(index, bytes)`; or the Rust image transforms,
/// without the interpreter.
impl Transform<Bytes> for PyTransform {
    type Output = PyItem;

    fn apply(&self, index: usize, item: Bytes) -> Result<PyItem> {
        if let Self::Native(ops) = self {
            return Ok(PyItem::Native(apply_all(ops, index, Value::Bytes(item))?));
        }
        Ok(PyItem::Object(attach(|py| {
            let item = PyBytes::new(py, &item).into_any().unbind();
            self.call(py, index, item)
                .and_then(|item| item.into_object(py))
        })?))
    }
}
