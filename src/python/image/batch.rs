//! `Image` and `ImageBatch` in Python: the values stay in Rust's buffer and numpy reads them in
//! place through `__array_interface__`; `__dlpack__` hands the same memory to torch.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::array::value_of_array;
use super::ops::Value;
use crate::collator::ImageBatch;
use crate::image::{Chw, Hwc, Layout, Sample};

/// A numpy view of a buffer: what `__array_interface__` describes.
struct ArrayView<'a> {
    address: usize,
    shape: &'a [usize],
    /// numpy's type string: `|u1`, `<f4`.
    typestr: &'static str,
    readonly: bool,
}

/// numpy's type string of `T`.
fn typestr<T: Sample>() -> &'static str {
    match (T::NAME, cfg!(target_endian = "little")) {
        ("uint8", _) => "|u1",
        (_, true) => "<f4",
        (_, false) => ">f4",
    }
}

impl ArrayView<'_> {
    fn of<'a, T: Sample>(data: &[T], shape: &'a [usize], readonly: bool) -> ArrayView<'a> {
        ArrayView {
            // The address only: numpy reads the memory, Rust keeps owning it, and the array
            // keeps this object (its `base`) alive.
            address: data.as_ptr() as usize,
            shape,
            typestr: typestr::<T>(),
            readonly,
        }
    }

    /// The `__array_interface__` dict (version 3, C-contiguous).
    fn interface<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item(intern!(py, "shape"), PyTuple::new(py, self.shape)?)?;
        dict.set_item(intern!(py, "typestr"), self.typestr)?;
        dict.set_item(intern!(py, "data"), (self.address, self.readonly))?;
        dict.set_item(intern!(py, "strides"), py.None())?;
        dict.set_item(intern!(py, "version"), 3)?;
        Ok(dict)
    }
}

/// `np.asarray(obj).__dlpack__(**kwargs)`: the array is a view of `obj`, so DLPack hands
/// torch (or any consumer) the same memory, kept alive through the array.
fn dlpack<'py>(
    obj: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let array = py
        .import(intern!(py, "numpy"))?
        .call_method1(intern!(py, "asarray"), (obj,))?;
    array.call_method(intern!(py, "__dlpack__"), (), kwargs)
}

/// DLPack's CPU device: `(kDLCPU, 0)`.
const CPU: (i32, i32) = (1, 0);

/// One image from the Rust transforms. `np.asarray(image)` is a read-only view of it, `HWC` or
/// `CHW` as `layout` says, and `ImageCollator` batches it.
#[pyclass(frozen, name = "Image", module = "dataloader_rs")]
pub struct PyImage {
    value: Value,
    shape: [usize; 3],
}

impl PyImage {
    /// `value`, which is an image (not bytes).
    pub(crate) fn new(value: Value) -> Self {
        let shape = match &value {
            Value::U8(i) => i.shape(),
            Value::U8Chw(i) => i.shape(),
            Value::F32(i) => i.shape(),
            Value::F32Chw(i) => i.shape(),
            Value::Bytes(_) => unreachable!("PyImage of bytes"),
        };
        Self { value, shape }
    }

    pub(crate) fn value(&self) -> &Value {
        &self.value
    }

    fn view(&self) -> ArrayView<'_> {
        match &self.value {
            Value::U8(i) => ArrayView::of(i.data(), &self.shape, true),
            Value::U8Chw(i) => ArrayView::of(i.data(), &self.shape, true),
            Value::F32(i) => ArrayView::of(i.data(), &self.shape, true),
            Value::F32Chw(i) => ArrayView::of(i.data(), &self.shape, true),
            Value::Bytes(_) => unreachable!("PyImage of bytes"),
        }
    }

    fn layout(&self) -> &'static str {
        match &self.value {
            Value::U8(_) | Value::F32(_) => Hwc::NAME,
            _ => Chw::NAME,
        }
    }
}

#[pymethods]
impl PyImage {
    /// An image of `obj`: a numpy array, a CPU torch tensor or anything `np.asarray` takes,
    /// of uint8 or float32 values, `(height, width)` or 3 dimensions in `layout` (`"hwc"` or
    /// `"chw"`) with 1, 3 or 4 channels, as an HWC image. A contiguous HWC array is copied
    /// once; a CHW or strided one is made contiguous by numpy first.
    #[staticmethod]
    #[pyo3(signature = (obj, layout = "hwc"))]
    fn from_array(obj: &Bound<'_, PyAny>, layout: &str) -> PyResult<Self> {
        value_of_array(obj, layout).map(Self::new)
    }

    /// `(height, width, channels)` or `(channels, height, width)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let [a, b, c] = self.shape;
        (a, b, c)
    }

    /// `"uint8"` or `"float32"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.value {
            Value::U8(_) | Value::U8Chw(_) => u8::NAME,
            _ => f32::NAME,
        }
    }

    /// `"HWC"` or `"CHW"`.
    #[getter(layout)]
    fn layout_name(&self) -> &'static str {
        self.layout()
    }

    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.view().interface(py)
    }

    #[pyo3(signature = (**kwargs))]
    fn __dlpack__<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dlpack(slf.as_any(), kwargs)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        CPU
    }

    fn __repr__(&self) -> String {
        let [a, b, c] = self.shape;
        format!("Image({a}x{b}x{c}, {}, {})", self.dtype(), self.layout())
    }
}

/// A batch from `ImageCollator`, in Rust's buffer.
pub(crate) enum Batch {
    U8(ImageBatch<u8>),
    F32(ImageBatch<f32>),
}

/// A batch of images in one buffer, from `ImageCollator`. `np.asarray(batch)` and
/// `torch.from_dlpack(batch)` are views of it, without a copy; the buffer lives as long as the
/// batch or any view of it.
#[pyclass(frozen, name = "ImageBatch", module = "dataloader_rs")]
pub struct PyImageBatch {
    batch: Batch,
    shape: [usize; 4],
}

impl PyImageBatch {
    pub(crate) fn new(batch: Batch) -> Self {
        let shape = match &batch {
            Batch::U8(b) => b.shape(),
            Batch::F32(b) => b.shape(),
        };
        Self { batch, shape }
    }
}

#[pymethods]
impl PyImageBatch {
    /// `(batch, channels, height, width)` or `(batch, height, width, channels)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let [n, a, b, c] = self.shape;
        (n, a, b, c)
    }

    /// `"uint8"` or `"float32"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.batch {
            Batch::U8(_) => u8::NAME,
            Batch::F32(_) => f32::NAME,
        }
    }

    /// `"NCHW"` or `"NHWC"`.
    #[getter]
    fn layout(&self) -> &'static str {
        match &self.batch {
            Batch::U8(b) => b.layout().name(),
            Batch::F32(b) => b.layout().name(),
        }
    }

    fn __len__(&self) -> usize {
        self.shape[0]
    }

    /// Writable, as a tensor's memory must be for torch; Rust does not read it again.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match &self.batch {
            Batch::U8(b) => ArrayView::of(b.data(), &self.shape, false).interface(py),
            Batch::F32(b) => ArrayView::of(b.data(), &self.shape, false).interface(py),
        }
    }

    #[pyo3(signature = (**kwargs))]
    fn __dlpack__<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dlpack(slf.as_any(), kwargs)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        CPU
    }

    fn __repr__(&self) -> String {
        let [n, a, b, c] = self.shape;
        format!(
            "ImageBatch({n}x{a}x{b}x{c}, {}, {})",
            self.dtype(),
            self.layout()
        )
    }
}
