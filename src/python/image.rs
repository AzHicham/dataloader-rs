//! The Rust image pipeline from Python: `ImageCollator` and its operations, image datasets, and
//! `ImageBatch`, a pooled batch buffer handed to NumPy and DLPack without a copy.
//!
//! With `collate_fn=ImageCollator(...)`, `PyDataloader` decodes, transforms and collates in Rust
//! on its worker threads, detached from the interpreter: Python only runs to fetch the encoded
//! bytes of a batch (`__getitem__` / `__getitems__`), or not at all for a `list` of bytes.

use std::result::Result as StdResult;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use crate::collator::{PooledBatch, PooledCollator};
use crate::dataset::Dataset;
use crate::error::Result;
use crate::image::{
    Channels, DType, DynImageCollator, ImageBuffer, ImageBufferAllocator, ImageError, ImageItem,
    ImageLimits, ImageTransform, Layout, Op, ResizeFilter, SampleInfo,
};
use crate::python::dataset::{Fetcher, PyDataset, get_item_py, len_py};
use crate::python::source::{FetchSource, PySource, SourceItems};

// ── Operations ───────────────────────────────────────────────────────────────

fn parse_filter(name: &str) -> PyResult<ResizeFilter> {
    Ok(match name {
        "nearest" => ResizeFilter::Nearest,
        "box" => ResizeFilter::Box,
        "bilinear" => ResizeFilter::Bilinear,
        "hamming" => ResizeFilter::Hamming,
        "bicubic" => ResizeFilter::Bicubic,
        "mitchell" => ResizeFilter::Mitchell,
        "lanczos3" | "lanczos" => ResizeFilter::Lanczos3,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown filter {other:?}: expected nearest, box, bilinear, hamming, bicubic, \
                 mitchell or lanczos3"
            )));
        }
    })
}

fn filter_name(filter: ResizeFilter) -> &'static str {
    match filter {
        ResizeFilter::Nearest => "nearest",
        ResizeFilter::Box => "box",
        ResizeFilter::Bilinear => "bilinear",
        ResizeFilter::Hamming => "hamming",
        ResizeFilter::Bicubic => "bicubic",
        ResizeFilter::Mitchell => "mitchell",
        ResizeFilter::Lanczos3 => "lanczos3",
    }
}

/// Resize to `width × height` with `filter` (default `"bilinear"`).
#[pyclass(name = "Resize", module = "dataloader_rs", frozen)]
pub struct PyResize {
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    filter: ResizeFilter,
}

#[pymethods]
impl PyResize {
    #[new]
    #[pyo3(signature = (width, height, filter = "bilinear"))]
    fn new(width: u32, height: u32, filter: &str) -> PyResult<Self> {
        Ok(Self {
            width,
            height,
            filter: parse_filter(filter)?,
        })
    }

    #[getter]
    fn filter(&self) -> &'static str {
        filter_name(self.filter)
    }

    fn __repr__(&self) -> String {
        op_repr(&Op::Resize {
            width: self.width,
            height: self.height,
            filter: self.filter,
        })
    }
}

/// Keep the `width × height` region whose top-left pixel is (`left`, `top`).
#[pyclass(name = "Crop", module = "dataloader_rs", frozen)]
pub struct PyCrop {
    #[pyo3(get)]
    left: u32,
    #[pyo3(get)]
    top: u32,
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
}

#[pymethods]
impl PyCrop {
    #[new]
    fn new(left: u32, top: u32, width: u32, height: u32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Crop({}, {}, {}, {})",
            self.left, self.top, self.width, self.height
        )
    }
}

/// Grow the image to `width × height`, anchored top-left, filling with `fill`.
#[pyclass(name = "Pad", module = "dataloader_rs", frozen)]
pub struct PyPad {
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    #[pyo3(get)]
    fill: u8,
}

#[pymethods]
impl PyPad {
    #[new]
    #[pyo3(signature = (width, height, fill = 0))]
    fn new(width: u32, height: u32, fill: u8) -> Self {
        Self {
            width,
            height,
            fill,
        }
    }

    fn __repr__(&self) -> String {
        format!("Pad({}, {}, fill={})", self.width, self.height, self.fill)
    }
}

/// Divide by 255, as torchvision's `ToTensor`.
#[pyclass(name = "ToFloat", module = "dataloader_rs", frozen)]
pub struct PyToFloat;

#[pymethods]
impl PyToFloat {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "ToFloat()"
    }
}

/// `(x - mean[c]) / std[c]` per channel, as torchvision's `Normalize`.
#[pyclass(name = "Normalize", module = "dataloader_rs", frozen)]
pub struct PyNormalize {
    #[pyo3(get)]
    mean: Vec<f32>,
    #[pyo3(get)]
    std: Vec<f32>,
}

#[pymethods]
impl PyNormalize {
    #[new]
    fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    fn __repr__(&self) -> String {
        format!("Normalize(mean={:?}, std={:?})", self.mean, self.std)
    }
}

fn extract_op(op: &Bound<'_, PyAny>) -> PyResult<Op> {
    if let Ok(r) = op.cast::<PyResize>() {
        let r = r.get();
        return Ok(Op::Resize {
            width: r.width,
            height: r.height,
            filter: r.filter,
        });
    }
    if let Ok(c) = op.cast::<PyCrop>() {
        let c = c.get();
        return Ok(Op::Crop {
            left: c.left,
            top: c.top,
            width: c.width,
            height: c.height,
        });
    }
    if let Ok(p) = op.cast::<PyPad>() {
        let p = p.get();
        return Ok(Op::Pad {
            width: p.width,
            height: p.height,
            fill: p.fill,
        });
    }
    if op.cast::<PyToFloat>().is_ok() {
        return Ok(Op::ToFloat);
    }
    if let Ok(n) = op.cast::<PyNormalize>() {
        let n = n.get();
        return Ok(Op::Normalize {
            mean: n.mean.clone(),
            std: n.std.clone(),
        });
    }
    Err(PyTypeError::new_err(format!(
        "{} is not an image operation: expected Resize, Crop, Pad, ToFloat or Normalize",
        op.repr()?
    )))
}

/// The Python spelling of an operation.
fn op_repr(op: &Op) -> String {
    match op {
        Op::Resize {
            width,
            height,
            filter,
        } => format!(
            "Resize({width}, {height}, filter={:?})",
            filter_name(*filter)
        ),
        Op::Crop {
            left,
            top,
            width,
            height,
        } => format!("Crop({left}, {top}, {width}, {height})"),
        Op::Pad {
            width,
            height,
            fill,
        } => format!("Pad({width}, {height}, fill={fill})"),
        Op::ToFloat => "ToFloat()".to_owned(),
        Op::Normalize { mean, std } => format!("Normalize(mean={mean:?}, std={std:?})"),
    }
}

// ── ImageCollator ────────────────────────────────────────────────────────────

fn parse_dtype(name: &str) -> PyResult<DType> {
    Ok(match name {
        "uint8" | "u8" => DType::U8,
        "float16" | "f16" | "half" => DType::F16,
        "float32" | "f32" | "float" => DType::F32,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown dtype {other:?}: expected uint8, float16 or float32"
            )));
        }
    })
}

fn image_error(err: ImageError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// The image pipeline, as a `collate_fn` for `PyDataloader`: decode, operations, dtype, layout.
#[pyclass(name = "ImageCollator", module = "dataloader_rs", frozen)]
pub struct PyImageCollator {
    transform: ImageTransform,
    dtype: DType,
    sample_shape: [usize; 3],
}

impl PyImageCollator {
    /// A new collator for one loader.
    pub(crate) fn collator(&self) -> PyResult<DynImageCollator> {
        DynImageCollator::new(self.transform.clone(), self.dtype).map_err(image_error)
    }
}

#[pymethods]
impl PyImageCollator {
    #[new]
    #[pyo3(signature = (
        ops = Vec::new(),
        *,
        channels = "rgb",
        dtype = "float32",
        layout = "nchw",
        input_size = None,
        max_width = 16_384,
        max_height = 16_384,
        max_bytes = 256 << 20
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        ops: Vec<Bound<'_, PyAny>>,
        channels: &str,
        dtype: &str,
        layout: &str,
        input_size: Option<(u32, u32)>,
        max_width: u32,
        max_height: u32,
        max_bytes: u64,
    ) -> PyResult<Self> {
        let channels = match channels {
            "rgb" | "RGB" => Channels::Rgb,
            "gray" | "L" => Channels::Gray,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown channels {other:?}: expected rgb or gray"
                )));
            }
        };
        let layout = match layout {
            "nchw" | "NCHW" => Layout::Nchw,
            "nhwc" | "NHWC" => Layout::Nhwc,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown layout {other:?}: expected nchw or nhwc"
                )));
            }
        };
        let ops = ops.iter().map(extract_op).collect::<PyResult<Vec<_>>>()?;
        let mut transform = ImageTransform::new(channels)
            .ops(ops)
            .layout(layout)
            .limits(ImageLimits {
                max_width,
                max_height,
                max_bytes,
            });
        if let Some((width, height)) = input_size {
            transform = transform.input_size(width, height);
        }
        let dtype = parse_dtype(dtype)?;
        // Validate now: an invalid pipeline raises here, not at the first batch.
        let collator = DynImageCollator::new(transform.clone(), dtype).map_err(image_error)?;
        Ok(Self {
            transform,
            dtype,
            sample_shape: collator.sample_shape(),
        })
    }

    /// The shape of one sample: `(C, H, W)` or `(H, W, C)`.
    #[getter]
    fn sample_shape(&self) -> (usize, usize, usize) {
        let [a, b, c] = self.sample_shape;
        (a, b, c)
    }

    /// NumPy name of the batch's dtype.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }

    fn __repr__(&self) -> String {
        let ops: Vec<String> = self.transform.operations().iter().map(op_repr).collect();
        let [a, b, c] = self.sample_shape;
        format!(
            "ImageCollator([{}], dtype={:?}, sample_shape=({a}, {b}, {c}))",
            ops.join(", "),
            self.dtype.name(),
        )
    }
}

// ── Datasets ─────────────────────────────────────────────────────────────────

/// A `list` of encoded images held by the loader: no Python runs per item.
pub(crate) struct MemoryImages {
    bytes: Vec<PyBackedBytes>,
    metas: Vec<Option<Py<PyAny>>>,
}

impl MemoryImages {
    /// `bytes[i]` with `metas[i]`; both have the same length.
    pub(crate) fn new(bytes: Vec<PyBackedBytes>, metas: Vec<Option<Py<PyAny>>>) -> Self {
        debug_assert_eq!(bytes.len(), metas.len());
        Self { bytes, metas }
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn bytes(&self, index: usize) -> &[u8] {
        &self.bytes[index]
    }

    pub(crate) fn meta(&self, index: usize) -> Option<&Py<PyAny>> {
        self.metas.get(index).and_then(Option::as_ref)
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> StdResult<(), PyTraverseError> {
        self.metas.iter().try_for_each(|meta| visit.call(meta))
    }
}

/// Where an image loader gets its encoded images.
pub(crate) enum ImageSource {
    /// A `PyDataset` whose items are `bytes` or `(bytes, meta)`.
    Python(PyDataset),
    /// A `list` of `bytes` or `(bytes, meta)`, read once.
    Memory(Arc<MemoryImages>),
    /// A `Source`: Rust fetches the items itself. Its metadata is reported to the garbage
    /// collector by `object` only.
    Source {
        object: Py<PySource>,
        items: SourceItems,
    },
}

/// `bytes` / `bytearray`, or a `(bytes, meta)` pair.
fn split_item(item: &Bound<'_, PyAny>) -> PyResult<(PyBackedBytes, Option<Py<PyAny>>)> {
    if let Ok(bytes) = item.extract::<PyBackedBytes>() {
        return Ok((bytes, None));
    }
    if let Ok(pair) = item.cast::<PyTuple>()
        && pair.len() == 2
        && let Ok(bytes) = pair.get_item(0)?.extract::<PyBackedBytes>()
    {
        return Ok((bytes, Some(pair.get_item(1)?.unbind())));
    }
    Err(PyTypeError::new_err(format!(
        "an image sample is bytes or (bytes, meta), not {}",
        item.get_type().name()?
    )))
}

impl ImageSource {
    /// A `PyDataset`, or a `list` / `tuple` of samples read now.
    pub(crate) fn new(dataset: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(source) = dataset.cast::<PySource>() {
            return Ok(Self::Source {
                items: source.get().items().clone(),
                object: source.clone().unbind(),
            });
        }
        if let Ok(dataset) = dataset.extract::<PyDataset>() {
            return Ok(Self::Python(dataset));
        }
        if dataset.is_instance_of::<PyList>() || dataset.is_instance_of::<PyTuple>() {
            let (bytes, metas) = dataset
                .try_iter()?
                .map(|item| split_item(&item?))
                .collect::<PyResult<(Vec<_>, Vec<_>)>>()?;
            return Ok(Self::Memory(Arc::new(MemoryImages { bytes, metas })));
        }
        Err(PyTypeError::new_err(format!(
            "an image dataset is a PyDataset or a list of bytes, not {}",
            dataset.get_type().name()?
        )))
    }

    /// The Python dataset, if any.
    pub(crate) fn python(&self) -> Option<&PyDataset> {
        match self {
            Self::Python(dataset) => Some(dataset),
            Self::Memory(_) | Self::Source { .. } => None,
        }
    }

    /// Visit the Python objects the source holds.
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> StdResult<(), PyTraverseError> {
        match self {
            Self::Python(dataset) => visit.call(dataset),
            Self::Memory(images) => images.traverse(visit),
            Self::Source { object, .. } => visit.call(object),
        }
    }

    fn item(item: &Bound<'_, PyAny>) -> PyResult<PyImageItem> {
        let (bytes, meta) = split_item(item)?;
        Ok(PyImageItem::Owned { bytes, meta })
    }
}

/// One encoded image and its metadata.
pub(crate) enum PyImageItem {
    Owned {
        bytes: PyBackedBytes,
        meta: Option<Py<PyAny>>,
    },
    Memory {
        images: Arc<MemoryImages>,
        index: usize,
    },
    /// Fetched by a `Source`, without the interpreter.
    Fetched {
        bytes: Vec<u8>,
        source: Arc<FetchSource>,
        index: usize,
    },
}

/// The metadata of a sample, resolved to a Python object when Python asks for it.
pub(crate) enum PyMeta {
    Owned(Option<Py<PyAny>>),
    Memory(Arc<MemoryImages>, usize),
    Fetched(Arc<FetchSource>, usize),
}

impl PyMeta {
    fn to_object(&self, py: Python<'_>) -> Py<PyAny> {
        let meta = match self {
            Self::Owned(meta) => meta.as_ref(),
            Self::Memory(images, index) => images.meta(*index),
            Self::Fetched(source, index) => source.meta(*index),
        };
        meta.map_or_else(|| py.None(), |m| m.clone_ref(py))
    }
}

impl ImageItem for PyImageItem {
    type Meta = PyMeta;

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned { bytes, .. } => bytes,
            Self::Memory { images, index } => images.bytes(*index),
            Self::Fetched { bytes, .. } => bytes,
        }
    }

    fn into_meta(self) -> PyMeta {
        match self {
            Self::Owned { meta, .. } => PyMeta::Owned(meta),
            Self::Memory { images, index } => PyMeta::Memory(images, index),
            Self::Fetched { source, index, .. } => PyMeta::Fetched(source, index),
        }
    }
}

impl Dataset for ImageSource {
    type Item = PyImageItem;

    fn get(&self, index: usize) -> Result<PyImageItem> {
        match self {
            Self::Python(dataset) => Python::attach(|py| {
                let item = get_item_py(dataset, py, index)?;
                Self::item(item.bind(py))
            })
            .map_err(Into::into),
            Self::Source { items, .. } => match items {
                SourceItems::Memory(images) => Self::memory_item(images, index),
                SourceItems::Fetch(source) => {
                    let bytes = source.fetch(index)?;
                    Ok(PyImageItem::Fetched {
                        bytes,
                        source: Arc::clone(source),
                        index,
                    })
                }
            },
            Self::Memory(images) => Self::memory_item(images, index),
        }
    }

    /// Attach once per batch, fetch the encoded images and nothing else: decoding runs after,
    /// detached.
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<PyImageItem>> {
        match self {
            Self::Python(dataset) => Python::attach(|py| {
                Fetcher::new(dataset, py)?
                    .fetch(py, indices)?
                    .iter()
                    .map(|item| Self::item(item.bind(py)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .map_err(Into::into),
            Self::Memory(_) | Self::Source { .. } => indices.iter().map(|&i| self.get(i)).collect(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Python(dataset) => Python::attach(|py| {
                len_py(dataset, py).unwrap_or_else(|e| panic!("PyDataset.__len__ failed: {e}"))
            }),
            Self::Memory(images) => images.len(),
            Self::Source { items, .. } => match items {
                SourceItems::Memory(images) => images.len(),
                SourceItems::Fetch(source) => source.len(),
            },
        }
    }
}

impl ImageSource {
    fn memory_item(images: &Arc<MemoryImages>, index: usize) -> Result<PyImageItem> {
        if index >= images.len() {
            return Err(PyIndexError::new_err(format!(
                "index {index} out of range for {} images",
                images.len()
            ))
            .into());
        }
        Ok(PyImageItem::Memory {
            images: Arc::clone(images),
            index,
        })
    }
}

pub(crate) type ImagePoolCollator = PooledCollator<DynImageCollator, ImageBufferAllocator>;
type CoreImageBatch = PooledBatch<ImageBuffer, Vec<SampleInfo<PyMeta>>>;

// ── ImageBatch ───────────────────────────────────────────────────────────────

/// Counts the `ImageBatch`es alive, so that `__next__` raises instead of waiting forever for a
/// buffer the consumer keeps.
pub(crate) struct Alive(Arc<AtomicUsize>);

impl Alive {
    fn new(count: &Arc<AtomicUsize>) -> Self {
        count.fetch_add(1, Ordering::AcqRel);
        Self(Arc::clone(count))
    }
}

impl Drop for Alive {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// NumPy's `typestr` of a dtype, in the machine's byte order.
fn typestr(dtype: DType) -> &'static str {
    let little = cfg!(target_endian = "little");
    match (dtype, little) {
        (DType::U8, _) => "|u1",
        (DType::F16, true) => "<f2",
        (DType::F16, false) => ">f2",
        (DType::F32, true) => "<f4",
        (DType::F32, false) => ">f4",
    }
}

/// A batch collated in Rust, in a buffer of the loader's pool.
///
/// `numpy.asarray(batch)` (through `__array_interface__`) and `torch.from_dlpack(batch)` view the
/// buffer without copying it; the views keep the batch alive, and the buffer returns to the pool
/// when the batch and every view of it are gone.
#[pyclass(name = "ImageBatch", module = "dataloader_rs", frozen)]
pub struct PyImageBatch {
    /// The buffer: Rust never reads or writes its values while the batch lives (Python may
    /// write them through a view).
    batch: CoreImageBatch,
    /// `(n, C, H, W)` or `(n, H, W, C)`, `n` the samples of this batch.
    shape: [usize; 4],
    dtype: DType,
    /// Address of the first value; stable, the buffer does not move.
    address: usize,
    _alive: Alive,
}

impl PyImageBatch {
    pub(crate) fn new(
        batch: CoreImageBatch,
        sample_shape: [usize; 3],
        alive: &Arc<AtomicUsize>,
    ) -> Self {
        let [a, b, c] = sample_shape;
        let dtype = batch.dtype();
        // `Vec::as_ptr` does not read or reference the values.
        let address = batch.as_ptr() as usize;
        Self {
            shape: [batch.num_samples(), a, b, c],
            dtype,
            address,
            batch,
            _alive: Alive::new(alive),
        }
    }

    fn as_numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py().import("numpy")?.getattr("asarray")?.call1((slf,))
    }
}

#[pymethods]
impl PyImageBatch {
    /// Samples in the batch (fewer than `batch_size` for the last batch of an epoch).
    fn __len__(&self) -> usize {
        self.shape[0]
    }

    /// `(n, C, H, W)` or `(n, H, W, C)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let [n, a, b, c] = self.shape;
        (n, a, b, c)
    }

    /// NumPy name of the dtype.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }

    /// The metadata of each sample: the `meta` of `(bytes, meta)` samples, else `None`.
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.batch.output().iter().map(|s| s.meta.to_object(py)))
    }

    /// The decoded `(width, height)` of each sample, before any operation.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.batch.output().iter().map(|s| (s.width, s.height)))
    }

    /// NumPy's array interface (version 3): the buffer, writable, C-contiguous.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let interface = PyDict::new(py);
        interface.set_item("version", 3)?;
        interface.set_item("shape", self.shape())?;
        interface.set_item("typestr", typestr(self.dtype))?;
        interface.set_item("data", (self.address, false))?;
        interface.set_item("strides", py.None())?;
        Ok(interface)
    }

    /// `numpy.asarray(self)`: a view of the buffer, no copy.
    fn numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Self::as_numpy(slf)
    }

    /// DLPack export through NumPy's: the capsule views the buffer and keeps the batch alive.
    #[pyo3(signature = (**kwargs))]
    fn __dlpack__<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::as_numpy(slf)?.call_method("__dlpack__", (), kwargs)
    }

    /// `(kDLCPU, 0)`.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageBatch(shape={:?}, dtype={:?})",
            self.shape(),
            self.dtype.name()
        )
    }
}
