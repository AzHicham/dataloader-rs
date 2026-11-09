//! `dataloader_rs.transforms`: the Rust image transforms under torchvision-like names, each a
//! `Transform` object; `Compose` chains them. Passed as `transform=`, a `Transform` runs in
//! Rust on the loader's threads, without the interpreter.

use std::sync::Arc;

use bytes::Bytes;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use super::batch::PyImage;
use super::ops::{Op, Value, apply_all};
use crate::image::{
    Anchor, CenterCrop, Crop, Decode, DecodeLimits, Fill, Filter, Normalize, Pad, PadMode, PadTo,
    Resize, Rotate90, ToFloat,
};
use crate::python::error::into_py_err;

/// Image transforms implemented in Rust, applied in order.
#[pyclass(frozen, name = "Transform", module = "dataloader_rs.transforms")]
pub struct PyImageTransform {
    ops: Arc<[Op]>,
    repr: String,
}

impl PyImageTransform {
    fn one(op: Op, repr: String) -> Self {
        Self {
            ops: Arc::new([op]),
            repr,
        }
    }

    /// The transforms, for the loader.
    pub(crate) fn ops(&self) -> Arc<[Op]> {
        Arc::clone(&self.ops)
    }
}

#[pymethods]
impl PyImageTransform {
    /// Apply the transforms to `item` (`bytes` or an `Image`), releasing the interpreter.
    fn __call__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value = value_of(item)?;
        let value = py
            .detach(|| apply_all(&self.ops, 0, value))
            .map_err(into_py_err)?;
        object_of(py, value)
    }

    fn __repr__(&self) -> &str {
        &self.repr
    }
}

/// The value of a Python item: `bytes` / `bytearray`, or an `Image`.
pub(crate) fn value_of(item: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(image) = item.cast::<PyImage>() {
        return Ok(image.get().value().clone());
    }
    match item.extract::<PyBackedBytes>() {
        Ok(bytes) => Ok(Value::Bytes(Bytes::copy_from_slice(&bytes))),
        Err(_) => Err(PyTypeError::new_err(format!(
            "image transforms take bytes or an Image, not {}",
            item.get_type().name()?
        ))),
    }
}

/// `value` as a Python object: `bytes` or an `Image`.
pub(crate) fn object_of(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Bytes(bytes) => Ok(PyBytes::new(py, &bytes).into_any().unbind()),
        image => Ok(Py::new(py, PyImage::new(image))?.into_any()),
    }
}

/// `(height, width)` from an int (square) or a pair.
fn size(name: &str, size: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    if let Ok(side) = size.extract::<usize>() {
        return Ok((side, side));
    }
    size.extract::<(usize, usize)>()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be an int or (height, width)")))
}

/// A fill: one number, or one per channel.
fn fill(fill: &Bound<'_, PyAny>) -> PyResult<Fill> {
    if let Ok(value) = fill.extract::<f32>() {
        return Ok(Fill::from(value));
    }
    fill.extract::<Vec<f32>>()
        .map(Fill::from)
        .map_err(|_| PyTypeError::new_err("fill must be a number or a sequence of numbers"))
}

/// The Python repr of a fill, `0` without one.
fn repr_or_zero(fill: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    fill.map_or(Ok("0".into()), |fill| Ok(fill.repr()?.to_string()))
}

/// Decode PNG, JPEG, WebP, TIFF or BMP bytes into a uint8 HWC image with the image's own
/// channels; images above the limits are refused from their header.
#[pyfunction(name = "Decode")]
#[pyo3(signature = (*, max_width=16_384, max_height=16_384, max_alloc=256 << 20))]
pub(crate) fn decode(max_width: u32, max_height: u32, max_alloc: u64) -> PyImageTransform {
    let limits = DecodeLimits {
        max_width,
        max_height,
        max_alloc,
    };
    let repr =
        format!("Decode(max_width={max_width}, max_height={max_height}, max_alloc={max_alloc})");
    PyImageTransform::one(Op::Decode(Decode::with_limits(limits)), repr)
}

/// To 3 channels, as Pillow's `convert("RGB")`.
#[pyfunction(name = "ToRgb")]
pub(crate) fn to_rgb() -> PyImageTransform {
    PyImageTransform::one(Op::ToRgb, "ToRgb()".into())
}

/// To 4 channels, as Pillow's `convert("RGBA")`.
#[pyfunction(name = "ToRgba")]
pub(crate) fn to_rgba() -> PyImageTransform {
    PyImageTransform::one(Op::ToRgba, "ToRgba()".into())
}

/// To 1 channel, as Pillow's `convert("L")`.
#[pyfunction(name = "ToGray")]
pub(crate) fn to_gray() -> PyImageTransform {
    PyImageTransform::one(Op::ToGray, "ToGray()".into())
}

/// The `height × width` region at `(top, left)`, as torchvision's `F.crop`.
#[pyfunction(name = "Crop")]
pub(crate) fn crop(top: usize, left: usize, height: usize, width: usize) -> PyImageTransform {
    let repr = format!("Crop({top}, {left}, {height}, {width})");
    PyImageTransform::one(Op::Crop(Crop::new(top, left, height, width)), repr)
}

/// The central region, as torchvision's `CenterCrop(size)`.
#[pyfunction(name = "CenterCrop")]
pub(crate) fn center_crop(size: &Bound<'_, PyAny>) -> PyResult<PyImageTransform> {
    let (height, width) = self::size("size", size)?;
    let repr = format!("CenterCrop(({height}, {width}))");
    Ok(PyImageTransform::one(
        Op::CenterCrop(CenterCrop::new(height, width)),
        repr,
    ))
}

/// Padding, as torchvision's `Pad(padding, fill, padding_mode)`: `padding` is an int, `(left
/// and right, top and bottom)` or `(left, top, right, bottom)`.
#[pyfunction(name = "Pad")]
#[pyo3(signature = (padding, fill=None, padding_mode="constant"))]
pub(crate) fn pad(
    padding: &Bound<'_, PyAny>,
    fill: Option<&Bound<'_, PyAny>>,
    padding_mode: &str,
) -> PyResult<PyImageTransform> {
    let [left, top, right, bottom] = if let Ok(all) = padding.extract::<usize>() {
        [all; 4]
    } else if let Ok((x, y)) = padding.extract::<(usize, usize)>() {
        [x, y, x, y]
    } else {
        padding.extract::<[usize; 4]>().map_err(|_| {
            PyTypeError::new_err("padding must be an int, a pair or (left, top, right, bottom)")
        })?
    };
    let mode = match (padding_mode, fill) {
        ("constant", None) => PadMode::Constant(Fill::from(0.0)),
        ("constant", Some(value)) => PadMode::Constant(self::fill(value)?),
        (_, Some(_)) => {
            return Err(PyValueError::new_err(
                "fill only applies to padding_mode=\"constant\"",
            ));
        }
        ("edge", None) => PadMode::Edge,
        ("reflect", None) => PadMode::Reflect,
        ("symmetric", None) => PadMode::Symmetric,
        (other, None) => {
            return Err(PyValueError::new_err(format!(
                "padding_mode must be constant, edge, reflect or symmetric, not {other:?}"
            )));
        }
    };
    let repr = format!(
        "Pad(({left}, {top}, {right}, {bottom}), fill={}, padding_mode='{padding_mode}')",
        repr_or_zero(fill)?
    );
    Ok(PyImageTransform::one(
        Op::Pad(Pad::new(left, top, right, bottom, mode)),
        repr,
    ))
}

/// Pad with `fill` up to `width × height`, never cropping. `anchor="top_left"` pads right and
/// bottom; `"center"` pads evenly.
#[pyfunction(name = "PadTo")]
#[pyo3(signature = (width, height, fill=None, anchor="top_left"))]
pub(crate) fn pad_to(
    width: usize,
    height: usize,
    fill: Option<&Bound<'_, PyAny>>,
    anchor: &str,
) -> PyResult<PyImageTransform> {
    let anchor_value = match anchor {
        "top_left" => Anchor::TopLeft,
        "center" => Anchor::Center,
        other => {
            return Err(PyValueError::new_err(format!(
                "anchor must be \"top_left\" or \"center\", not {other:?}"
            )));
        }
    };
    let repr = format!(
        "PadTo({width}, {height}, fill={}, anchor='{anchor}')",
        repr_or_zero(fill)?
    );
    let fill = fill.map(self::fill).transpose()?.unwrap_or(Fill::from(0.0));
    let op = PadTo::new(width, height, fill).anchor(anchor_value);
    Ok(PyImageTransform::one(Op::PadTo(op), repr))
}

/// Resize a uint8 image, as torchvision's `Resize(size, interpolation, max_size, antialias)`:
/// `size` is `(height, width)`, or an int for the shorter side. `interpolation`: `"nearest"`,
/// `"bilinear"`, `"bicubic"` or `"lanczos3"`.
#[pyfunction(name = "Resize")]
#[pyo3(signature = (size, interpolation="bilinear", max_size=None, antialias=true))]
pub(crate) fn resize(
    size: &Bound<'_, PyAny>,
    interpolation: &str,
    max_size: Option<usize>,
    antialias: bool,
) -> PyResult<PyImageTransform> {
    let filter = match interpolation {
        "nearest" => Filter::Nearest,
        "bilinear" => Filter::Bilinear,
        "bicubic" => Filter::Bicubic,
        "lanczos3" => Filter::Lanczos3,
        other => {
            return Err(PyValueError::new_err(format!(
                "interpolation must be nearest, bilinear, bicubic or lanczos3, not {other:?}"
            )));
        }
    };
    let mut op = match size.extract::<usize>() {
        Ok(shorter) => Resize::shorter(shorter),
        Err(_) => {
            let (height, width) = self::size("size", size)?;
            Resize::new(height, width)
        }
    };
    if let Some(max_size) = max_size {
        op = op.max_size(max_size);
    }
    let op = op.filter(filter).antialias(antialias);
    let repr = format!(
        "Resize({}, interpolation='{interpolation}', max_size={max_size:?}, antialias={antialias})",
        size.repr()?
    );
    Ok(PyImageTransform::one(Op::Resize(op), repr))
}

/// Mirror left to right.
#[pyfunction(name = "HFlip")]
pub(crate) fn hflip() -> PyImageTransform {
    PyImageTransform::one(Op::HFlip, "HFlip()".into())
}

/// Mirror top to bottom.
#[pyfunction(name = "VFlip")]
pub(crate) fn vflip() -> PyImageTransform {
    PyImageTransform::one(Op::VFlip, "VFlip()".into())
}

/// Rotate by `k` quarter turns counter-clockwise, as `numpy.rot90(image, k)`.
#[pyfunction(name = "Rotate90")]
#[pyo3(signature = (k=1))]
pub(crate) fn rotate90(k: i64) -> PyImageTransform {
    PyImageTransform::one(Op::Rotate90(Rotate90::new(k)), format!("Rotate90({k})"))
}

/// uint8 to float32, divided by 255 if `scale`: torchvision's `ToTensor` values.
#[pyfunction(name = "ToFloat")]
#[pyo3(signature = (scale=true))]
pub(crate) fn to_float(scale: bool) -> PyImageTransform {
    let repr = format!("ToFloat(scale={})", if scale { "True" } else { "False" });
    PyImageTransform::one(Op::ToFloat(ToFloat::new(scale)), repr)
}

/// `(x - mean) / std` per channel of a float32 image, as torchvision's `Normalize`.
#[pyfunction(name = "Normalize")]
pub(crate) fn normalize(mean: Vec<f32>, std: Vec<f32>) -> PyResult<PyImageTransform> {
    let repr = format!("Normalize(mean={mean:?}, std={std:?})");
    let op = Normalize::new(mean, std).map_err(into_py_err)?;
    Ok(PyImageTransform::one(Op::Normalize(op), repr))
}

/// HWC to CHW: the layout change of torchvision's `ToTensor`.
#[pyfunction(name = "ToChw")]
pub(crate) fn to_chw() -> PyImageTransform {
    PyImageTransform::one(Op::ToChw, "ToChw()".into())
}

/// CHW to HWC.
#[pyfunction(name = "ToHwc")]
pub(crate) fn to_hwc() -> PyImageTransform {
    PyImageTransform::one(Op::ToHwc, "ToHwc()".into())
}

/// The `transforms` in order, as one `Transform`.
#[pyfunction(name = "Compose")]
pub(crate) fn compose(transforms: Vec<Bound<'_, PyImageTransform>>) -> PyImageTransform {
    let ops: Vec<Op> = transforms
        .iter()
        .flat_map(|t| t.get().ops.iter().cloned())
        .collect();
    let reprs: Vec<&str> = transforms.iter().map(|t| t.get().repr.as_str()).collect();
    PyImageTransform {
        ops: ops.into(),
        repr: format!("Compose([{}])", reprs.join(", ")),
    }
}
