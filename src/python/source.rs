//! `Source`: a dataset whose items Rust fetches itself, from memory, files or HTTP, with
//! per-item metadata. With `collate_fn=ImageCollator(...)`, an item is fetched, decoded,
//! transformed and collated without the interpreter.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyOSError, PyPermissionError, PyTimeoutError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict};
use pyo3::{PyTraverseError, PyVisit};

use crate::python::dataset::PyDatasetBase;
use crate::python::image::MemoryImages;
use crate::sources::{
    DEFAULT_MAX_BYTES, FileSource, HttpSource, HttpSourceBuilder, Retry, Source, SourceError,
};

/// The Python exception of a [`SourceError`]: `ValueError` for an invalid configuration,
/// `IndexError` for an index out of range, `TimeoutError`, `FileNotFoundError`,
/// `PermissionError` or else `OSError` for an item that cannot be fetched.
pub(crate) fn source_error(error: &SourceError) -> PyErr {
    let message = error.to_string();
    match error {
        SourceError::Invalid(_) | SourceError::MetaLength { .. } => PyValueError::new_err(message),
        SourceError::OutOfRange { .. } => PyIndexError::new_err(message),
        SourceError::Request { timeout: true, .. } => PyTimeoutError::new_err(message),
        SourceError::File { source, .. } => match source.kind() {
            ErrorKind::NotFound => PyFileNotFoundError::new_err(message),
            ErrorKind::PermissionDenied => PyPermissionError::new_err(message),
            _ => PyOSError::new_err(message),
        },
        _ => PyOSError::new_err(message),
    }
}

/// The files or URLs of a `Source`, and the metadata of each item.
pub(crate) struct FetchSource {
    kind: FetchKind,
    metas: Vec<Option<Py<PyAny>>>,
}

enum FetchKind {
    File(FileSource),
    Http(HttpSource),
}

impl FetchSource {
    /// The bytes of item `index`; runs without the interpreter.
    pub(crate) fn fetch(&self, index: usize) -> Result<Vec<u8>, SourceError> {
        match &self.kind {
            FetchKind::File(source) => source.fetch(index),
            FetchKind::Http(source) => source.fetch(index),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.metas.len()
    }

    pub(crate) fn meta(&self, index: usize) -> Option<&Py<PyAny>> {
        self.metas.get(index).and_then(Option::as_ref)
    }
}

/// What a `Source` holds.
#[derive(Clone)]
pub(crate) enum SourceItems {
    /// `Source.from_bytes`: the bytes, read once.
    Memory(Arc<MemoryImages>),
    /// `Source.from_files`, `from_urls`, `from_template`: fetched per item.
    Fetch(Arc<FetchSource>),
}

/// A dataset of encoded items that Rust fetches itself: `bytes` from memory, files, or HTTP
/// `GET`s, each with optional metadata. Item `i` is `(bytes, meta[i])`.
///
/// With `collate_fn=ImageCollator(...)`, `PyDataloader` fetches, decodes, transforms and
/// collates each item on its worker threads without the interpreter. As a plain `PyDataset`,
/// `source[i]` fetches item `i` with the interpreter released.
#[pyclass(name = "Source", module = "dataloader_rs", frozen, extends = PyDatasetBase)]
pub struct PySource {
    items: SourceItems,
}

impl PySource {
    pub(crate) fn items(&self) -> &SourceItems {
        &self.items
    }

    fn create(py: Python<'_>, items: SourceItems) -> PyResult<Py<Self>> {
        Py::new(
            py,
            PyClassInitializer::from(PyDatasetBase).add_subclass(Self { items }),
        )
    }

    fn fetch_source(
        py: Python<'_>,
        kind: FetchKind,
        len: usize,
        meta: Option<Vec<Py<PyAny>>>,
    ) -> PyResult<Py<Self>> {
        let metas = metas(len, meta)?;
        Self::create(
            py,
            SourceItems::Fetch(Arc::new(FetchSource { kind, metas })),
        )
    }
}

/// One `Option` per item: the given metadata, or `None` for each.
fn metas(len: usize, meta: Option<Vec<Py<PyAny>>>) -> PyResult<Vec<Option<Py<PyAny>>>> {
    match meta {
        None => Ok((0..len).map(|_| None).collect()),
        Some(meta) if meta.len() == len => Ok(meta.into_iter().map(Some).collect()),
        Some(meta) => Err(source_error(&SourceError::MetaLength {
            items: len,
            meta: meta.len(),
        })),
    }
}

/// Seconds as a [`Duration`]; `ValueError` if negative, NaN or too large.
fn seconds(name: &str, value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be a number of seconds >= 0")))
}

/// The HTTP settings shared by `from_urls` and `from_template`.
struct HttpSettings {
    headers: Option<HashMap<String, String>>,
    timeout: f64,
    connect_timeout: f64,
    max_retries: u32,
    backoff: f64,
    max_backoff: f64,
    max_bytes: u64,
    max_redirects: u32,
    max_idle_connections: usize,
}

impl HttpSettings {
    fn builder(self) -> PyResult<HttpSourceBuilder> {
        let mut builder = HttpSource::builder()
            .timeout(seconds("timeout", self.timeout)?)
            .connect_timeout(seconds("connect_timeout", self.connect_timeout)?)
            .retry(Retry {
                max_retries: self.max_retries,
                initial_backoff: seconds("backoff", self.backoff)?,
                max_backoff: seconds("max_backoff", self.max_backoff)?,
            })
            .max_bytes(self.max_bytes)
            .max_redirects(self.max_redirects)
            .max_idle_connections(self.max_idle_connections);
        let mut headers: Vec<_> = self.headers.unwrap_or_default().into_iter().collect();
        headers.sort();
        for (name, value) in headers {
            builder = builder.header(name, value);
        }
        Ok(builder)
    }
}

#[pymethods]
impl PySource {
    /// Items already in memory: `bytes` (or `bytearray`, copied), read once.
    #[staticmethod]
    #[pyo3(signature = (items, meta = None))]
    fn from_bytes(
        py: Python<'_>,
        items: Vec<PyBackedBytes>,
        meta: Option<Vec<Py<PyAny>>>,
    ) -> PyResult<Py<Self>> {
        let metas = metas(items.len(), meta)?;
        Self::create(
            py,
            SourceItems::Memory(Arc::new(MemoryImages::new(items, metas))),
        )
    }

    /// One file per item, read whole; a file above `max_bytes` fails its item.
    #[staticmethod]
    #[pyo3(signature = (paths, meta = None, *, max_bytes = DEFAULT_MAX_BYTES))]
    fn from_files(
        py: Python<'_>,
        paths: Vec<PathBuf>,
        meta: Option<Vec<Py<PyAny>>>,
        max_bytes: u64,
    ) -> PyResult<Py<Self>> {
        if max_bytes == 0 {
            return Err(PyValueError::new_err("max_bytes must be > 0"));
        }
        let len = paths.len();
        let source = FileSource::new(paths).max_bytes(max_bytes);
        Self::fetch_source(py, FetchKind::File(source), len, meta)
    }

    /// One HTTP `GET` per URL. See the class documentation for the settings.
    #[staticmethod]
    #[pyo3(signature = (
        urls,
        meta = None,
        *,
        headers = None,
        timeout = 30.0,
        connect_timeout = 10.0,
        max_retries = 3,
        backoff = 0.1,
        max_backoff = 5.0,
        max_bytes = DEFAULT_MAX_BYTES,
        max_redirects = 5,
        max_idle_connections = 64
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_urls(
        py: Python<'_>,
        urls: Vec<String>,
        meta: Option<Vec<Py<PyAny>>>,
        headers: Option<HashMap<String, String>>,
        timeout: f64,
        connect_timeout: f64,
        max_retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
        max_redirects: u32,
        max_idle_connections: usize,
    ) -> PyResult<Py<Self>> {
        let builder = HttpSettings {
            headers,
            timeout,
            connect_timeout,
            max_retries,
            backoff,
            max_backoff,
            max_bytes,
            max_redirects,
            max_idle_connections,
        }
        .builder()?;
        let len = urls.len();
        let source = builder.urls(&urls).map_err(|e| source_error(&e))?;
        Self::fetch_source(py, FetchKind::Http(source), len, meta)
    }

    /// One HTTP `GET` per entry of `params`: `template` with each `{name}` replaced by
    /// `str(params[i][name])`, percent-encoded.
    #[staticmethod]
    #[pyo3(signature = (
        template,
        params,
        meta = None,
        *,
        headers = None,
        timeout = 30.0,
        connect_timeout = 10.0,
        max_retries = 3,
        backoff = 0.1,
        max_backoff = 5.0,
        max_bytes = DEFAULT_MAX_BYTES,
        max_redirects = 5,
        max_idle_connections = 64
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_template(
        py: Python<'_>,
        template: &str,
        params: Vec<Bound<'_, PyDict>>,
        meta: Option<Vec<Py<PyAny>>>,
        headers: Option<HashMap<String, String>>,
        timeout: f64,
        connect_timeout: f64,
        max_retries: u32,
        backoff: f64,
        max_backoff: f64,
        max_bytes: u64,
        max_redirects: u32,
        max_idle_connections: usize,
    ) -> PyResult<Py<Self>> {
        let builder = HttpSettings {
            headers,
            timeout,
            connect_timeout,
            max_retries,
            backoff,
            max_backoff,
            max_bytes,
            max_redirects,
            max_idle_connections,
        }
        .builder()?;
        let entries = params
            .iter()
            .map(|entry| {
                entry
                    .iter()
                    .map(|(name, value)| Ok((name.extract::<String>()?, value.str()?.to_string())))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let len = entries.len();
        let source = builder
            .template(template, entries)
            .map_err(|e| source_error(&e))?;
        Self::fetch_source(py, FetchKind::Http(source), len, meta)
    }

    fn __len__(&self) -> usize {
        match &self.items {
            SourceItems::Memory(images) => images.len(),
            SourceItems::Fetch(source) => source.len(),
        }
    }

    /// `(bytes, meta)` of item `index`, fetched with the interpreter released.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Py<PyAny>)> {
        let len = self.__len__();
        if index >= len {
            return Err(source_error(&SourceError::OutOfRange { index, len }));
        }
        match &self.items {
            SourceItems::Memory(images) => Ok((
                PyBytes::new(py, images.bytes(index)),
                images
                    .meta(index)
                    .map_or_else(|| py.None(), |meta| meta.clone_ref(py)),
            )),
            SourceItems::Fetch(source) => {
                let bytes = py
                    .detach(|| source.fetch(index))
                    .map_err(|e| source_error(&e))?;
                Ok((
                    PyBytes::new(py, &bytes),
                    source
                        .meta(index)
                        .map_or_else(|| py.None(), |meta| meta.clone_ref(py)),
                ))
            }
        }
    }

    fn __repr__(&self) -> String {
        let kind = match &self.items {
            SourceItems::Memory(_) => "bytes",
            SourceItems::Fetch(source) => match source.kind {
                FetchKind::File(_) => "files",
                FetchKind::Http(_) => "urls",
            },
        };
        format!("Source({kind}, len={})", self.__len__())
    }

    /// The metadata is held here only: loaders and batches reach it through this object.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.items {
            SourceItems::Memory(images) => images.traverse(&visit),
            SourceItems::Fetch(source) => source.metas.iter().try_for_each(|meta| visit.call(meta)),
        }
    }
}
