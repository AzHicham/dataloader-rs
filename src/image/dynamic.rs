//! The value type chosen at run time: one collator and one buffer type for `u8`, `f16` and `f32`
//! batches, for bindings and configuration files.

use half::f16;

use crate::collator::{BufferAllocator, CollateInto, HeapAllocator};
use crate::error::Result;

use super::{ImageCollator, ImageError, ImageItem, ImageResult, ImageTransform, SampleInfo};

/// The value type of a batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    /// 8-bit pixels, no value operation (`to_float`, `normalize`) allowed.
    U8,
    /// IEEE half precision, rounded to nearest even from `f32` (as `Tensor.half()`).
    F16,
    /// IEEE single precision.
    #[default]
    F32,
}

impl DType {
    /// Bytes per value: 1, 2 or 4.
    pub fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    /// NumPy's name of the type: `uint8`, `float16` or `float32`.
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::F16 => "float16",
            Self::F32 => "float32",
        }
    }
}

/// A contiguous batch buffer whose value type is known at run time.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageBuffer {
    /// `u8` values.
    U8(Vec<u8>),
    /// `f16` values.
    F16(Vec<f16>),
    /// `f32` values.
    F32(Vec<f32>),
}

impl ImageBuffer {
    /// The value type.
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
        }
    }

    /// Values in the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    /// Whether the buffer holds no value.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Address of the first value, e.g. for NumPy's `__array_interface__`. The buffer never
    /// reallocates: the address is stable while it lives.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::U8(v) => v.as_ptr(),
            Self::F16(v) => v.as_ptr().cast(),
            Self::F32(v) => v.as_ptr().cast(),
        }
    }

    /// The values, if they are `u8`.
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(v) => Some(v),
            _ => None,
        }
    }

    /// The values, if they are `f16`.
    pub fn as_f16(&self) -> Option<&[f16]> {
        match self {
            Self::F16(v) => Some(v),
            _ => None,
        }
    }

    /// The values, if they are `f32`.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }
}

/// Allocates [`ImageBuffer`]s of one dtype and length, on the heap.
#[derive(Clone, Copy, Debug)]
pub struct ImageBufferAllocator {
    dtype: DType,
    len: usize,
}

impl ImageBufferAllocator {
    /// Buffers of `shape.iter().product()` values of `dtype`; fails on overflow or above
    /// `isize::MAX` bytes.
    pub fn for_shape(dtype: DType, shape: &[usize]) -> ImageResult<Self> {
        let too_large = |_| ImageError::Limits(format!("batch of shape {shape:?} is too large"));
        let len = match dtype {
            DType::U8 => HeapAllocator::<u8>::for_shape(shape)
                .map_err(too_large)?
                .len(),
            DType::F16 => HeapAllocator::<f16>::for_shape(shape)
                .map_err(too_large)?
                .len(),
            DType::F32 => HeapAllocator::<f32>::for_shape(shape)
                .map_err(too_large)?
                .len(),
        };
        Ok(Self { dtype, len })
    }

    /// The value type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Values per buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether buffers hold no value.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl BufferAllocator for ImageBufferAllocator {
    type Buffer = ImageBuffer;

    fn allocate(&self) -> Result<ImageBuffer> {
        // The length was checked against `isize::MAX` bytes; `try_reserve_exact` turns an
        // allocation failure into an error.
        fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>> {
            let mut v = Vec::new();
            v.try_reserve_exact(len)?;
            v.resize(len, T::default());
            Ok(v)
        }
        Ok(match self.dtype {
            DType::U8 => ImageBuffer::U8(zeroed(self.len)?),
            DType::F16 => ImageBuffer::F16(zeroed(self.len)?),
            DType::F32 => ImageBuffer::F32(zeroed(self.len)?),
        })
    }
}

enum Typed {
    U8(ImageCollator<u8>),
    F16(ImageCollator<f16>),
    F32(ImageCollator<f32>),
}

/// An [`ImageCollator`] whose value type is a [`DType`] value rather than a type parameter: built
/// from plain values only, it collates into [`ImageBuffer`]s from
/// [`allocator`](Self::allocator).
///
/// ```rust
/// use dataloader_rs::image::{Channels, DType, DynImageCollator, ImageTransform, ResizeFilter};
///
/// let collator = DynImageCollator::new(
///     ImageTransform::new(Channels::Rgb)
///         .resize(224, 224, ResizeFilter::Bilinear)
///         .to_float(),
///     DType::F16,
/// )
/// .unwrap();
/// assert_eq!(collator.batch_shape(64), [64, 3, 224, 224]);
/// let allocator = collator.allocator(64).unwrap();
/// assert_eq!(allocator.len(), 64 * 3 * 224 * 224);
/// ```
pub struct DynImageCollator {
    typed: Typed,
}

impl DynImageCollator {
    /// Collate with `transform` into buffers of `dtype`; fails if the pipeline is invalid, or
    /// produces floats and `dtype` is `U8`.
    pub fn new(transform: ImageTransform, dtype: DType) -> ImageResult<Self> {
        let typed = match dtype {
            DType::U8 => Typed::U8(ImageCollator::new(transform)?),
            DType::F16 => Typed::F16(ImageCollator::new(transform)?),
            DType::F32 => Typed::F32(ImageCollator::new(transform)?),
        };
        Ok(Self { typed })
    }

    /// The value type.
    pub fn dtype(&self) -> DType {
        match self.typed {
            Typed::U8(_) => DType::U8,
            Typed::F16(_) => DType::F16,
            Typed::F32(_) => DType::F32,
        }
    }

    /// The pipeline.
    pub fn transform(&self) -> &ImageTransform {
        match &self.typed {
            Typed::U8(c) => c.transform(),
            Typed::F16(c) => c.transform(),
            Typed::F32(c) => c.transform(),
        }
    }

    /// The shape of one sample: `[C, H, W]` (NCHW) or `[H, W, C]` (NHWC).
    pub fn sample_shape(&self) -> [usize; 3] {
        match &self.typed {
            Typed::U8(c) => c.sample_shape(),
            Typed::F16(c) => c.sample_shape(),
            Typed::F32(c) => c.sample_shape(),
        }
    }

    /// Values per sample.
    pub fn sample_len(&self) -> usize {
        match &self.typed {
            Typed::U8(c) => c.sample_len(),
            Typed::F16(c) => c.sample_len(),
            Typed::F32(c) => c.sample_len(),
        }
    }

    /// The shape of a batch of `batch_size` samples.
    pub fn batch_shape(&self, batch_size: usize) -> [usize; 4] {
        let [a, b, c] = self.sample_shape();
        [batch_size, a, b, c]
    }

    /// An allocator of buffers for batches of `batch_size` samples.
    pub fn allocator(&self, batch_size: usize) -> ImageResult<ImageBufferAllocator> {
        ImageBufferAllocator::for_shape(self.dtype(), &self.batch_shape(batch_size))
    }

    /// Decode and transform one image into `row`, which must have the collator's dtype and hold
    /// exactly [`sample_len`](Self::sample_len) values. Returns the decoded size.
    pub fn write_sample(&self, bytes: &[u8], row: &mut ImageBuffer) -> ImageResult<(u32, u32)> {
        match (&self.typed, row) {
            (Typed::U8(c), ImageBuffer::U8(row)) => c.write_sample(bytes, row),
            (Typed::F16(c), ImageBuffer::F16(row)) => c.write_sample(bytes, row),
            (Typed::F32(c), ImageBuffer::F32(row)) => c.write_sample(bytes, row),
            (_, row) => Err(self.dtype_mismatch(row)),
        }
    }

    fn dtype_mismatch(&self, buffer: &ImageBuffer) -> ImageError {
        ImageError::Buffer(format!(
            "buffer of {}, the collator writes {}",
            buffer.dtype().name(),
            self.dtype().name()
        ))
    }
}

impl<I: ImageItem> CollateInto<I> for DynImageCollator {
    type Buffer = ImageBuffer;
    type Output = Vec<SampleInfo<I::Meta>>;

    fn collate_into(&self, items: Vec<I>, buffer: &mut ImageBuffer) -> Result<Self::Output> {
        match (&self.typed, buffer) {
            (Typed::U8(c), ImageBuffer::U8(b)) => c.collate_into(items, b),
            (Typed::F16(c), ImageBuffer::F16(b)) => c.collate_into(items, b),
            (Typed::F32(c), ImageBuffer::F32(b)) => c.collate_into(items, b),
            (_, buffer) => Err(self.dtype_mismatch(buffer).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, DynImageCollator, ImageBuffer, ImageBufferAllocator};
    use crate::collator::{BufferAllocator, CollateInto};
    use crate::image::{Channels, ImageError, ImageTransform, Layout};

    #[test]
    fn allocates_the_dtype() {
        for (dtype, size) in [(DType::U8, 1), (DType::F16, 2), (DType::F32, 4)] {
            assert_eq!(dtype.size(), size);
            let buffer = ImageBufferAllocator::for_shape(dtype, &[2, 3])
                .unwrap()
                .allocate()
                .unwrap();
            assert_eq!((buffer.dtype(), buffer.len()), (dtype, 6));
        }
        assert!(matches!(
            ImageBufferAllocator::for_shape(DType::F32, &[usize::MAX, 2]),
            Err(ImageError::Limits(_))
        ));
        assert!(ImageBufferAllocator::for_shape(DType::F32, &[usize::MAX / 4 + 1]).is_err());
    }

    #[test]
    fn rejects_value_operations_into_u8_and_other_dtypes() {
        let t = ImageTransform::new(Channels::Gray).input_size(2, 2);
        assert!(matches!(
            DynImageCollator::new(t.clone().to_float(), DType::U8),
            Err(ImageError::InvalidPipeline(_))
        ));
        let c = DynImageCollator::new(t.layout(Layout::Nhwc), DType::F16).unwrap();
        assert_eq!(c.batch_shape(3), [3, 2, 2, 1]);
        let mut wrong = ImageBuffer::F32(vec![0.0; 12]);
        let err = c
            .collate_into(Vec::<Vec<u8>>::new(), &mut wrong)
            .unwrap_err();
        assert!(err.to_string().contains("buffer of float32"), "{err}");
        assert!(c.write_sample(&[], &mut wrong).is_err());
    }
}
