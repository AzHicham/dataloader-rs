//! The Rust datasets in Python: `HttpDataset`, `FileDataset`, `ObjectStoreDataset`. Items are
//! fetched without the interpreter; the loader's transform turns them into `bytes` (and applies
//! `transform=`) on its pool.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use bytes::Bytes;
use pyo3::PyClass;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::{PyBytes, PyDict, PyString};

use crate::{
    dataset::{
        AsyncDataset, Column, DEFAULT_MAX_BYTES, FileDataset, FileDatasetBuilder, HttpDataset,
        HttpDatasetBuilder, ObjectStoreDataset, ObjectStoreDatasetBuilder, Retry,
    },
    error::{Error, Result},
};

/// The Rust dataset behind a Python one.
#[derive(Clone)]
pub(crate) enum RustSource {
    Http(HttpDataset),
    File(FileDataset),
    Object(ObjectStoreDataset),
}

impl AsyncDataset for RustSource {
    type Item = Bytes;

    async fn get(&self, index: usize) -> Result<Bytes> {
        match self {
            Self::Http(dataset) => dataset.get(index).await,
            Self::File(dataset) => dataset.get(index).await,
            Self::Object(dataset) => dataset.get(index).await,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Http(dataset) => AsyncDataset::len(dataset),
            Self::File(dataset) => AsyncDataset::len(dataset),
            Self::Object(dataset) => AsyncDataset::len(dataset),
        }
    }
}

/// Base class of the datasets implemented in Rust. Items are `bytes`.
#[pyclass(frozen, subclass, name = "RustDataset", module = "dataloader_rs")]
pub struct PyRustDataset {
    source: RustSource,
}

impl PyRustDataset {
    fn new(source: RustSource) -> Self {
        Self { source }
    }

    /// The dataset the loader runs.
    pub(crate) fn source(&self) -> RustSource {
        self.source.clone()
    }
}

#[pymethods]
impl PyRustDataset {
    fn __len__(&self) -> usize {
        AsyncDataset::len(&self.source)
    }
}

/// Seconds as a `Duration`; negative, NaN or infinite values are refused.
fn seconds(name: &str, value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be a finite number >= 0")))
}

fn value_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// `retries`, `backoff` and `max_backoff` (seconds), the same for every Rust dataset.
fn retry(retries: u32, backoff: f64, max_backoff: f64) -> PyResult<Retry> {
    Ok(Retry {
        max_retries: retries,
        initial_backoff: seconds("backoff", backoff)?,
        max_backoff: seconds("max_backoff", max_backoff)?,
    })
}

/// The values of a template column: ints (any sequence or numpy array of them) or strs.
fn column(name: &str, values: &Bound<'_, PyAny>) -> PyResult<Column> {
    if !values.is_instance_of::<PyString>() && !values.is_instance_of::<PyBytes>() {
        if let Ok(values) = values.extract::<Vec<i64>>() {
            return Ok(Column::I64(values));
        }
        if let Ok(values) = values.extract::<Vec<u64>>() {
            return Ok(Column::U64(values));
        }
        if let Ok(values) = values.extract::<Vec<String>>() {
            return Ok(Column::Str(values));
        }
    }
    Err(PyTypeError::new_err(format!(
        "column {name:?} must be a sequence of ints (in the int64 or uint64 range) or of strs"
    )))
}

/// The `columns` of `from_template`, `{name: values}`: the names, and the values in the same
/// order.
fn columns(columns: Option<Bound<'_, PyDict>>) -> PyResult<(Vec<String>, Vec<Column>)> {
    let mut names = Vec::new();
    let mut values = Vec::new();
    for (name, column_values) in columns.iter().flat_map(|columns| columns.iter()) {
        let name: String = name
            .extract()
            .map_err(|_| PyTypeError::new_err("column names must be strs"))?;
        values.push(column(&name, &column_values)?);
        names.push(name);
    }
    Ok((names, values))
}

fn initializer<T: PyClass<BaseType = PyRustDataset>>(
    source: RustSource,
    subclass: T,
) -> PyClassInitializer<T> {
    PyClassInitializer::from(PyRustDataset::new(source)).add_subclass(subclass)
}

/// The settings shared by `HttpDataset(urls)` and `HttpDataset.from_template`.
#[allow(clippy::too_many_arguments)]
fn http_builder(
    headers: Option<HashMap<String, String>>,
    timeout: f64,
    connect_timeout: f64,
    retries: u32,
    backoff: f64,
    max_backoff: f64,
    max_bytes: u64,
) -> PyResult<HttpDatasetBuilder> {
    let mut builder = HttpDataset::builder()
        .timeout(seconds("timeout", timeout)?)
        .connect_timeout(seconds("connect_timeout", connect_timeout)?)
        .retry(retry(retries, backoff, max_backoff)?)
        .max_bytes(max_bytes);
    for (name, value) in headers.unwrap_or_default() {
        builder = builder.header(name, value);
    }
    Ok(builder)
}

/// One HTTP GET per item: `bytes` of the response to `urls[i]`.
#[pyclass(frozen, extends = PyRustDataset, name = "HttpDataset", module = "dataloader_rs")]
pub struct PyHttpDataset;

#[pymethods]
impl PyHttpDataset {
    #[new]
    #[pyo3(signature = (
        urls,
        *,
        headers=None,
        timeout=30.0,
        connect_timeout=10.0,
        retries=3,
        backoff=0.1,
        max_backoff=5.0,
        max_bytes=DEFAULT_MAX_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        urls: Vec<String>,
        headers: Option<HashMap<String, String>>,
        timeout: f64,
        connect_timeout: f64,
        retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let builder = http_builder(
            headers,
            timeout,
            connect_timeout,
            retries,
            backoff,
            max_backoff,
            max_bytes,
        )?;
        let dataset = builder.urls(urls).map_err(value_error)?;
        Ok(initializer(RustSource::Http(dataset), Self))
    }

    /// URLs made in Rust, item by item, from `template` and one value per item in each of
    /// `columns`; no URL string is kept per item.
    #[staticmethod]
    #[pyo3(signature = (
        template,
        columns=None,
        *,
        length=None,
        headers=None,
        timeout=30.0,
        connect_timeout=10.0,
        retries=3,
        backoff=0.1,
        max_backoff=5.0,
        max_bytes=DEFAULT_MAX_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_template(
        py: Python<'_>,
        template: &str,
        columns: Option<Bound<'_, PyDict>>,
        length: Option<usize>,
        headers: Option<HashMap<String, String>>,
        timeout: f64,
        connect_timeout: f64,
        retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
    ) -> PyResult<Py<Self>> {
        let builder = http_builder(
            headers,
            timeout,
            connect_timeout,
            retries,
            backoff,
            max_backoff,
            max_bytes,
        )?;
        let (names, values) = self::columns(columns)?;
        let columns = names.iter().map(String::as_str).zip(values);
        let dataset = match length {
            Some(len) => builder.url_template_len(len, template, columns),
            None => builder.url_template(template, columns),
        }
        .map_err(value_error)?;
        Py::new(py, initializer(RustSource::Http(dataset), Self))
    }
}

/// The settings shared by `FileDataset(paths)` and `FileDataset.from_template`.
fn file_builder(
    timeout: Option<f64>,
    retries: u32,
    backoff: f64,
    max_backoff: f64,
    max_bytes: u64,
) -> PyResult<FileDatasetBuilder> {
    let mut builder = FileDataset::builder()
        .retry(retry(retries, backoff, max_backoff)?)
        .max_bytes(max_bytes);
    if let Some(timeout) = timeout {
        builder = builder.timeout(seconds("timeout", timeout)?);
    }
    Ok(builder)
}

/// One local file per item: `bytes` of `paths[i]`.
#[pyclass(frozen, extends = PyRustDataset, name = "FileDataset", module = "dataloader_rs")]
pub struct PyFileDataset;

#[pymethods]
impl PyFileDataset {
    #[new]
    #[pyo3(signature = (
        paths,
        *,
        timeout=None,
        retries=0,
        backoff=0.1,
        max_backoff=5.0,
        max_bytes=DEFAULT_MAX_BYTES
    ))]
    fn new(
        paths: Vec<PathBuf>,
        timeout: Option<f64>,
        retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let builder = file_builder(timeout, retries, backoff, max_backoff, max_bytes)?;
        let dataset = builder.paths(paths).map_err(value_error)?;
        Ok(initializer(RustSource::File(dataset), Self))
    }

    /// Paths made in Rust, item by item, from `template` and one value per item in each of
    /// `columns`; no path is kept per item.
    #[staticmethod]
    #[pyo3(signature = (
        template,
        columns=None,
        *,
        length=None,
        timeout=None,
        retries=0,
        backoff=0.1,
        max_backoff=5.0,
        max_bytes=DEFAULT_MAX_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_template(
        py: Python<'_>,
        template: &str,
        columns: Option<Bound<'_, PyDict>>,
        length: Option<usize>,
        timeout: Option<f64>,
        retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
    ) -> PyResult<Py<Self>> {
        let builder = file_builder(timeout, retries, backoff, max_backoff, max_bytes)?;
        let (names, values) = self::columns(columns)?;
        let columns = names.iter().map(String::as_str).zip(values);
        let dataset = match length {
            Some(len) => builder.path_template_len(len, template, columns),
            None => builder.path_template(template, columns),
        }
        .map_err(value_error)?;
        Py::new(py, initializer(RustSource::File(dataset), Self))
    }
}

/// The settings shared by `ObjectStoreDataset(url, keys)` and
/// `ObjectStoreDataset.from_template`.
fn object_store_builder(
    url: &str,
    options: Option<HashMap<String, String>>,
    timeout: Option<f64>,
    retries: u32,
    backoff: f64,
    max_backoff: f64,
    max_bytes: u64,
) -> PyResult<ObjectStoreDatasetBuilder> {
    let mut builder = ObjectStoreDataset::from_url(url, options.unwrap_or_default())
        .retry(retry(retries, backoff, max_backoff)?)
        .max_bytes(max_bytes);
    if let Some(timeout) = timeout {
        builder = builder.timeout(seconds("timeout", timeout)?);
    }
    Ok(builder)
}

/// One object per item from S3, GCS, Azure or an HTTP store: `bytes` of key `keys[i]`.
#[pyclass(frozen, extends = PyRustDataset, name = "ObjectStoreDataset", module = "dataloader_rs")]
pub struct PyObjectStoreDataset;

#[pymethods]
impl PyObjectStoreDataset {
    #[new]
    #[pyo3(signature = (
        url,
        keys,
        *,
        options=None,
        timeout=None,
        retries=3,
        backoff=0.1,
        max_backoff=5.0,
        max_bytes=DEFAULT_MAX_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        url: &str,
        keys: Vec<String>,
        options: Option<HashMap<String, String>>,
        timeout: Option<f64>,
        retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let builder = object_store_builder(
            url,
            options,
            timeout,
            retries,
            backoff,
            max_backoff,
            max_bytes,
        )?;
        let dataset = builder.keys(keys).map_err(value_error)?;
        Ok(initializer(RustSource::Object(dataset), Self))
    }

    /// Keys made in Rust, item by item, from `template` (relative to the prefix of `url`) and
    /// one value per item in each of `columns`; no key is kept per item.
    #[staticmethod]
    #[pyo3(signature = (
        url,
        template,
        columns=None,
        *,
        length=None,
        options=None,
        timeout=None,
        retries=3,
        backoff=0.1,
        max_backoff=5.0,
        max_bytes=DEFAULT_MAX_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_template(
        py: Python<'_>,
        url: &str,
        template: &str,
        columns: Option<Bound<'_, PyDict>>,
        length: Option<usize>,
        options: Option<HashMap<String, String>>,
        timeout: Option<f64>,
        retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
    ) -> PyResult<Py<Self>> {
        let builder = object_store_builder(
            url,
            options,
            timeout,
            retries,
            backoff,
            max_backoff,
            max_bytes,
        )?;
        let (names, values) = self::columns(columns)?;
        let columns = names.iter().map(String::as_str).zip(values);
        let dataset = match length {
            Some(len) => builder.key_template_len(len, template, columns),
            None => builder.key_template(template, columns),
        }
        .map_err(value_error)?;
        Py::new(py, initializer(RustSource::Object(dataset), Self))
    }
}
