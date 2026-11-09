//! Core errors back into Python exceptions.

use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;

use crate::{dataset::FetchError, error::Error, image::ImageError};

/// Convert a core error back into the Python exception it came from.
///
/// Errors raised by `__getitem__` or `collate_fn` travel through the core as
/// boxed `PyErr`s; unwrapping them keeps the original exception type and
/// traceback. Errors that did not originate in Python become `RuntimeError`.
///
/// A built-in dataset's [`FetchError`] becomes `IndexError` (index out of
/// range), `ValueError` (item over `max_bytes`), `TimeoutError` or `OSError`; an
/// [`ImageError`] (an image refused by a transform or the image collator) becomes `ValueError`.
pub(crate) fn into_py_err(err: Error) -> PyErr {
    let err = match err.downcast::<PyErr>() {
        Ok(err) => return *err,
        Err(err) => err,
    };
    if err.is::<ImageError>() {
        return PyValueError::new_err(err.to_string());
    }
    let Some(fetch) = err.downcast_ref::<FetchError>() else {
        return PyRuntimeError::new_err(err.to_string());
    };
    let message = err.to_string();
    match fetch {
        FetchError::OutOfRange { .. } => PyIndexError::new_err(message),
        FetchError::TooLarge { .. } => PyValueError::new_err(message),
        fetch if fetch.is_timeout() => PyTimeoutError::new_err(message),
        _ => PyOSError::new_err(message),
    }
}
