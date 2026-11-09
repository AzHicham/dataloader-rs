//! The kind of a Python dataset, from its base class: `PyDataset`, `PyAsyncDataset` or a Rust
//! dataset, with the checks its methods must pass.

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;

use super::{PyAsyncDatasetBase, PyDataset, PyDatasetBase, PyRustDataset, RustSource};

/// A dataset `PyDataloader` accepts, by base class.
pub(crate) enum DatasetKind {
    /// A `PyDataset`.
    Sync(PyDataset),
    /// A `PyAsyncDataset`: whether it defines `__getitems__` and `aclose`.
    Async { batched: bool, aclose: bool },
    /// A Rust dataset.
    Rust(RustSource),
}

/// Whether `dataset.<name>` exists and is an `async def` function.
fn is_async_method(dataset: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<bool>> {
    let py = dataset.py();
    match dataset.getattr_opt(name)? {
        Some(method) if method.is_truthy()? => {
            let inspect = py.import(intern!(py, "inspect"))?;
            let is_async = inspect
                .call_method1(intern!(py, "iscoroutinefunction"), (method,))?
                .is_truthy()?;
            Ok(Some(is_async))
        }
        _ => Ok(None),
    }
}

/// Check `dataset` against its base class and tell its kind.
pub(crate) fn dataset_kind(dataset: &Bound<'_, PyAny>) -> PyResult<DatasetKind> {
    let py = dataset.py();
    if let Ok(rust) = dataset.cast::<PyRustDataset>() {
        return Ok(DatasetKind::Rust(rust.get().source()));
    }
    let is_sync = dataset.is_instance_of::<PyDatasetBase>();
    let is_async = dataset.is_instance_of::<PyAsyncDatasetBase>();
    let kind = match (is_sync, is_async) {
        (true, true) => {
            return Err(PyTypeError::new_err(
                "a dataset subclasses PyDataset or PyAsyncDataset, not both",
            ));
        }
        (true, false) => "PyDataset",
        (false, true) => "PyAsyncDataset",
        (false, false) => {
            return Err(PyTypeError::new_err(format!(
                "dataset must be a PyDataset, a PyAsyncDataset or a Rust dataset \
                 (HttpDataset, FileDataset, ObjectStoreDataset), not {}",
                dataset.get_type().name()?
            )));
        }
    };
    let ty = dataset.get_type();
    let overrides_len = match is_sync {
        true => !ty.getattr(intern!(py, "__len__"))?.is(&py
            .get_type::<PyDatasetBase>()
            .getattr(intern!(py, "__len__"))?),
        false => ty.hasattr(intern!(py, "__len__"))?,
    };
    if !overrides_len {
        return Err(PyTypeError::new_err(format!(
            "{kind} subclasses must define __len__"
        )));
    }
    let getitem = is_async_method(dataset, "__getitem__")?;
    let getitems = is_async_method(dataset, "__getitems__")?;
    if is_sync {
        if getitem == Some(true) || getitems == Some(true) {
            return Err(PyTypeError::new_err(
                "PyDataset.__getitem__ / __getitems__ must not be async; \
                 subclass PyAsyncDataset for async def methods",
            ));
        }
        let dataset: PyDataset = dataset.cast::<PyDatasetBase>()?.clone().unbind();
        return Ok(DatasetKind::Sync(dataset));
    }
    if getitem == Some(false) || getitems == Some(false) {
        return Err(PyTypeError::new_err(
            "PyAsyncDataset.__getitem__ / __getitems__ must be async def; \
             subclass PyDataset for plain methods",
        ));
    }
    if getitem.is_none() {
        return Err(PyTypeError::new_err(
            "PyAsyncDataset subclasses must define async def __getitem__",
        ));
    }
    let aclose = is_async_method(dataset, "aclose")?;
    if aclose == Some(false) {
        return Err(PyTypeError::new_err(
            "PyAsyncDataset.aclose must be async def",
        ));
    }
    Ok(DatasetKind::Async {
        batched: getitems.is_some(),
        aclose: aclose.is_some(),
    })
}
