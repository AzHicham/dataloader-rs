//! `collate_fn=`: a Python function, an `ImageCollator`, or none (a `list`).

use pyo3::{prelude::*, types::PyList};

use crate::python::image::{Batch, ImageCollate, PyImageBatch, PyImageCollator, Value};
use crate::python::transform::PyItem;
use crate::{collator::Collator, error::Result, python::pool::attach};

pub(crate) enum PyBatch {
    Ready(Py<PyAny>),
    Items(Vec<PyItem>),
    /// An `ImageCollator`'s batch, collated without the interpreter.
    Images(Batch),
}

impl PyBatch {
    /// The object `__next__` returns: `collate_fn`'s result, a `list`, or an `ImageBatch`.
    pub(crate) fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Ready(obj) => Ok(obj),
            Self::Items(items) => {
                let items = objects(py, items)?;
                Ok(PyList::new(py, items)?.into_any().unbind())
            }
            Self::Images(batch) => Ok(Py::new(py, PyImageBatch::new(batch))?.into_any()),
        }
    }
}

/// `items` as Python objects.
fn objects(py: Python<'_>, items: Vec<PyItem>) -> PyResult<Vec<Py<PyAny>>> {
    items.into_iter().map(|item| item.into_object(py)).collect()
}

/// The values of `items` if they are all Rust images (the Rust transforms' output).
fn native(items: &[PyItem]) -> Option<Vec<&Value>> {
    items
        .iter()
        .map(|item| match item {
            PyItem::Native(value) => Some(value),
            PyItem::Object(_) => None,
        })
        .collect()
}

pub(crate) enum PyCollator {
    /// No `collate_fn`: the batch is the `list` of its items.
    List,
    Function(Py<PyAny>),
    Images(ImageCollate),
}

impl PyCollator {
    pub(crate) fn new(collate_fn: Option<&Bound<'_, PyAny>>) -> Self {
        match collate_fn {
            None => Self::List,
            Some(f) => match f.cast::<PyImageCollator>() {
                Ok(images) => Self::Images(images.get().collate()),
                Err(_) => Self::Function(f.clone().unbind()),
            },
        }
    }

    /// The user's `collate_fn`, if it is a Python function.
    pub(crate) fn collate_fn(&self) -> Option<&Py<PyAny>> {
        match self {
            Self::Function(f) => Some(f),
            _ => None,
        }
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::List => Self::List,
            Self::Function(f) => Self::Function(f.clone_ref(py)),
            Self::Images(images) => Self::Images(images.clone()),
        }
    }

    /// Like `collate` but uses an already-held `py` token — no `Python::attach()` overhead.
    pub(crate) fn collate_with_py(&self, py: Python<'_>, items: Vec<PyItem>) -> Result<PyBatch> {
        match self {
            Self::List => Ok(PyBatch::Items(items)),
            Self::Function(f) => {
                let list = PyList::new(py, objects(py, items)?)?;
                Ok(PyBatch::Ready(f.call1(py, (list,))?))
            }
            Self::Images(images) => {
                if let Some(values) = native(&items) {
                    return Ok(PyBatch::Images(py.detach(|| images.collate(&values))?));
                }
                let objects: Vec<Bound<'_, PyAny>> = objects(py, items)?
                    .into_iter()
                    .map(|object| object.into_bound(py))
                    .collect();
                Ok(PyBatch::Ready(images.collate_objects(py, &objects)?))
            }
        }
    }
}

impl Collator<PyItem> for PyCollator {
    type Batch = PyBatch;

    fn collate(&self, items: Vec<PyItem>) -> Result<Self::Batch> {
        match self {
            // Fast path: keep raw items and convert to Python list in __next__,
            // avoiding Python API work in the worker thread.
            Self::List => Ok(PyBatch::Items(items)),
            // The Rust transforms' images: collated without the interpreter.
            Self::Images(images) => match native(&items) {
                Some(values) => Ok(PyBatch::Images(images.collate(&values)?)),
                None => attach(|py| self.collate_with_py(py, items)),
            },
            Self::Function(_) => attach(|py| self.collate_with_py(py, items)),
        }
    }
}
