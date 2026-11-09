//! `Image.from_array`: an image from a numpy array, a CPU torch tensor or any object with the
//! buffer protocol, copied into Rust's buffer.
//!
//! numpy views the object (`np.asarray`: no copy for arrays and CPU tensors) and the buffer
//! protocol copies the view into the image's `Vec`: one copy for a contiguous HWC array. A CHW
//! or strided array is first made contiguous in HWC order by numpy: one more copy, but numpy's
//! strided copy is several times faster than the buffer protocol's (0.12 vs 0.73 ms for
//! 3x224x224 values).

use pyo3::buffer::{Element, PyUntypedBuffer};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;

use super::ops::Value;
use crate::error::Result;
use crate::image::{DecodeLimits, Image, ImageError, Sample, invalid, sample_count};
use crate::python::error::into_py_err;

/// DLPack's device type of CPU memory.
const DLPACK_CPU: i32 = 1;

/// How the array's dimensions are ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrayLayout {
    /// `(height, width, channels)`, as numpy and Pillow.
    Hwc,
    /// `(channels, height, width)`, as torch.
    Chw,
}

impl ArrayLayout {
    fn parse(name: &str) -> PyResult<Self> {
        if name.eq_ignore_ascii_case("hwc") {
            Ok(Self::Hwc)
        } else if name.eq_ignore_ascii_case("chw") {
            Ok(Self::Chw)
        } else {
            Err(PyValueError::new_err(format!(
                "layout must be \"hwc\" or \"chw\", not {name:?}"
            )))
        }
    }
}

/// The type of the array's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleType {
    U8,
    F32,
}

impl SampleType {
    /// The type of a buffer of `format` (a `struct` format string): `uint8`, or `float32` in
    /// this machine's byte order; `None` for any other.
    fn of(format: &[u8]) -> Option<Self> {
        match format {
            [b'B'] | [b'@' | b'=', b'B'] => Some(Self::U8),
            [b'f'] | [b'@' | b'=', b'f'] => Some(Self::F32),
            _ => None,
        }
    }

    /// Bytes per value.
    fn size(self) -> usize {
        match self {
            Self::U8 => size_of::<u8>(),
            Self::F32 => size_of::<f32>(),
        }
    }
}

/// `(height, width, channels)` of an array of `shape` in `layout`, holding `len_bytes` bytes of
/// `item_size`-byte values: an image of 1, 3 or 4 channels, within the size limits of `Decode`,
/// whose bytes are exactly those of its shape.
fn image_shape(
    shape: &[usize],
    layout: ArrayLayout,
    item_size: usize,
    len_bytes: usize,
    limits: &DecodeLimits,
) -> Result<(usize, usize, usize)> {
    let (height, width, channels) = match (shape, layout) {
        (&[height, width], _) => (height, width, 1),
        (&[height, width, channels], ArrayLayout::Hwc) => (height, width, channels),
        (&[channels, height, width], ArrayLayout::Chw) => (height, width, channels),
        _ => {
            return Err(invalid(format!(
                "an array of shape {shape:?}: expected 2 dimensions (height, width) or 3"
            )));
        }
    };
    if height == 0 || width == 0 || channels == 0 {
        return Err(invalid(format!("an empty array of shape {shape:?}")));
    }
    if !matches!(channels, 1 | 3 | 4) {
        return Err(invalid(format!(
            "an array of shape {shape:?} in {layout:?} layout has {channels} channels, \
             expected 1, 3 or 4"
        )));
    }
    let (max_width, max_height) = (limits.max_width as usize, limits.max_height as usize);
    if width > max_width || height > max_height {
        return Err(too_large(format!(
            "{width}x{height}, above the limit of {max_width}x{max_height}"
        )));
    }
    let bytes = sample_count(width, height, channels)?
        .checked_mul(item_size)
        .ok_or_else(|| too_large(format!("an array of shape {shape:?} overflows")))?;
    if bytes as u64 > limits.max_alloc {
        return Err(too_large(format!(
            "{bytes} bytes, above the limit of {}",
            limits.max_alloc
        )));
    }
    if bytes != len_bytes {
        return Err(invalid(format!(
            "a buffer of {len_bytes} bytes for an array of shape {shape:?} ({bytes} bytes)"
        )));
    }
    Ok((height, width, channels))
}

/// An [`ImageError::Limits`] as a crate error.
fn too_large(message: String) -> crate::Error {
    ImageError::Limits(message).into()
}

/// The image of `obj` (any object numpy can view, or with the buffer protocol): uint8 or
/// float32 values, `(height, width)` or 3 dimensions in `layout`, as an HWC image.
pub(crate) fn value_of_array(obj: &Bound<'_, PyAny>, layout: &str) -> PyResult<Value> {
    let py = obj.py();
    let layout = ArrayLayout::parse(layout)?;
    ensure_cpu(obj)?;
    let numpy = py.import(intern!(py, "numpy")).ok();
    let array = match &numpy {
        Some(numpy) => numpy.call_method1(intern!(py, "asarray"), (obj,))?,
        None => obj.clone(),
    };
    // A view of the array as given, for its type and shape: checked before any copy.
    let buffer = PyUntypedBuffer::get(&array).map_err(|err| {
        let detail = match (&numpy, array.getattr(intern!(py, "shape"))) {
            (Some(_), Ok(shape)) => format!("an array of shape {shape}"),
            (Some(_), Err(_)) => String::new(),
            (None, _) => "without numpy, only objects with the buffer protocol".into(),
        };
        let type_name = obj
            .get_type()
            .name()
            .map_or_else(|_| "?".into(), |name| name.to_string());
        PyTypeError::new_err(format!(
            "Image.from_array cannot read a {type_name} ({detail}): {err}"
        ))
    })?;
    let Some(sample) = SampleType::of(buffer.format().to_bytes()) else {
        let dtype = match array.getattr(intern!(py, "dtype")) {
            Ok(dtype) => dtype.str()?.to_string(),
            Err(_) => format!("values of format {:?}", buffer.format()),
        };
        return Err(PyTypeError::new_err(format!(
            "Image.from_array takes uint8 or float32 values, not {dtype}"
        )));
    };
    let shape = image_shape(
        buffer.shape(),
        layout,
        sample.size(),
        buffer.len_bytes(),
        &DecodeLimits::default(),
    )
    .map_err(into_py_err)?;
    let transpose = layout == ArrayLayout::Chw && buffer.dimensions() == 3;
    let buffer = match &numpy {
        // The HWC values in one block: the array itself if it is one (a contiguous HWC
        // array or CPU tensor), else numpy's copy (a CHW or strided array): numpy copies
        // strided arrays several times faster than the buffer protocol does.
        Some(numpy) => {
            drop(buffer);
            let view = if transpose {
                array.call_method1(intern!(py, "transpose"), (1, 2, 0))?
            } else {
                array
            };
            let hwc = numpy.call_method1(intern!(py, "ascontiguousarray"), (view,))?;
            PyUntypedBuffer::get(&hwc)?
        }
        None if transpose => {
            return Err(PyTypeError::new_err(
                "Image.from_array needs numpy for layout=\"chw\"",
            ));
        }
        None => buffer,
    };
    Ok(match sample {
        SampleType::U8 => Value::U8(copy(py, buffer, shape)?),
        SampleType::F32 => Value::F32(copy(py, buffer, shape)?),
    })
}

/// A torch tensor (or any DLPack array) outside CPU memory is an error, not a hidden copy.
fn ensure_cpu(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = obj.py();
    if !obj.hasattr(intern!(py, "__dlpack_device__"))? {
        return Ok(());
    }
    let (device, _): (i32, i32) = obj
        .call_method0(intern!(py, "__dlpack_device__"))?
        .extract()?;
    if device == DLPACK_CPU {
        return Ok(());
    }
    let place = match obj.getattr(intern!(py, "device")) {
        Ok(place) => place.str()?.to_string(),
        Err(_) => format!("DLPack device {device}"),
    };
    Err(PyTypeError::new_err(format!(
        "Image.from_array reads CPU memory, not {place}: move the tensor first (tensor.cpu())"
    )))
}

/// The values of `buffer`, an HWC view of `(height, width, channels)`, copied into an image.
fn copy<T: Sample + Element>(
    py: Python<'_>,
    buffer: PyUntypedBuffer,
    (height, width, channels): (usize, usize, usize),
) -> PyResult<Image<T>> {
    // The shape checked before, unless another thread reshaped the array since.
    let hwc = [height, width, channels];
    if buffer.shape() != &hwc[..buffer.dimensions()] {
        return Err(PyValueError::new_err("the array changed shape while read"));
    }
    let data = buffer.into_typed::<T>()?.to_vec(py)?;
    Image::new(width, height, channels, data).map_err(into_py_err)
}

#[cfg(test)]
mod tests {
    use super::{ArrayLayout, SampleType, image_shape};
    use crate::image::DecodeLimits;

    fn shape(
        dims: &[usize],
        layout: ArrayLayout,
        len_bytes: usize,
    ) -> Result<(usize, usize, usize), String> {
        image_shape(dims, layout, 1, len_bytes, &DecodeLimits::default()).map_err(|e| e.to_string())
    }

    #[test]
    fn shapes_in_both_layouts() {
        assert_eq!(shape(&[2, 5], ArrayLayout::Hwc, 10), Ok((2, 5, 1)));
        assert_eq!(shape(&[2, 5], ArrayLayout::Chw, 10), Ok((2, 5, 1)));
        assert_eq!(shape(&[2, 5, 3], ArrayLayout::Hwc, 30), Ok((2, 5, 3)));
        assert_eq!(shape(&[3, 2, 5], ArrayLayout::Chw, 30), Ok((2, 5, 3)));
        assert_eq!(shape(&[4, 2, 5], ArrayLayout::Chw, 40), Ok((2, 5, 4)));
    }

    #[test]
    fn bad_shapes_are_refused() {
        for (dims, len) in [(&[6][..], 6), (&[1, 2, 3, 1][..], 6), (&[2, 0, 3][..], 0)] {
            assert!(shape(dims, ArrayLayout::Hwc, len).is_err(), "{dims:?}");
        }
        let err = shape(&[2, 5, 2], ArrayLayout::Hwc, 20).unwrap_err();
        assert!(err.contains("2 channels, expected 1, 3 or 4"), "{err}");
        let err = shape(&[224, 224, 3], ArrayLayout::Chw, 224 * 224 * 3).unwrap_err();
        assert!(err.contains("224 channels"), "{err}");
    }

    #[test]
    fn the_buffer_must_have_the_bytes_of_its_shape() {
        let err = shape(&[2, 5, 3], ArrayLayout::Hwc, 29).unwrap_err();
        assert!(err.contains("29 bytes"), "{err}");
        let float = image_shape(
            &[2, 5, 3],
            ArrayLayout::Hwc,
            4,
            30,
            &DecodeLimits::default(),
        );
        assert!(float.is_err());
    }

    #[test]
    fn the_limits_of_decode_apply() {
        let err = shape(&[16_385, 1, 1], ArrayLayout::Hwc, 16_385).unwrap_err();
        assert!(err.contains("image too large"), "{err}");
        // 16384 x 16384 x 4 float32 values: 1 GiB, above 256 MiB.
        let big = image_shape(
            &[16_384, 16_384, 4],
            ArrayLayout::Hwc,
            4,
            1 << 30,
            &DecodeLimits::default(),
        );
        assert!(
            big.unwrap_err()
                .to_string()
                .contains("above the limit of 268435456")
        );
        // Overflow never wraps.
        let huge = image_shape(
            &[usize::MAX, usize::MAX, 3],
            ArrayLayout::Hwc,
            4,
            0,
            &DecodeLimits::default(),
        );
        assert!(huge.is_err());
        let limits = DecodeLimits {
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_alloc: u64::MAX,
        };
        let side = u32::MAX as usize;
        let huge = image_shape(&[side, side, 4], ArrayLayout::Hwc, 4, 0, &limits);
        assert!(huge.unwrap_err().to_string().contains("overflows"));
    }

    #[test]
    fn only_native_uint8_and_float32() {
        assert_eq!(SampleType::of(b"B"), Some(SampleType::U8));
        assert_eq!(SampleType::of(b"=B"), Some(SampleType::U8));
        assert_eq!(SampleType::of(b"f"), Some(SampleType::F32));
        assert_eq!(SampleType::of(b"@f"), Some(SampleType::F32));
        for other in [
            &b"b"[..],
            b"?",
            b"c",
            b"d",
            b"e",
            b">f",
            b"<f",
            b"!f",
            b"H",
            b"Bf",
            b"",
        ] {
            assert_eq!(SampleType::of(other), None, "{other:?}");
        }
    }
}
