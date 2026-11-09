//! [`Image`], an owned image, and [`Sample`], the type of its values.

use std::fmt::{self, Debug};
use std::marker::PhantomData;

use super::{Hwc, ImageError, Layout};
use crate::error::Result;

/// The type of an image's values: `u8` for decoded pixels, `f32` after
/// [`ToFloat`](super::ToFloat).
pub trait Sample: Copy + Default + PartialEq + Debug + Send + Sync + 'static {
    /// numpy's name of the type.
    const NAME: &'static str;

    /// `value` as a sample, if it is one: an integer in `0..=255` for `u8`.
    fn from_f32(value: f32) -> Option<Self>;
}

impl Sample for u8 {
    const NAME: &'static str = "uint8";

    fn from_f32(value: f32) -> Option<u8> {
        let fits = value.fract() == 0.0 && (0.0..=255.0).contains(&value);
        fits.then_some(value as u8)
    }
}

impl Sample for f32 {
    const NAME: &'static str = "float32";

    fn from_f32(value: f32) -> Option<f32> {
        Some(value)
    }
}

/// The most channels an image has: gray, gray + alpha, RGB, RGBA.
pub const MAX_CHANNELS: usize = 4;

/// An owned image: `height × width × channels` values of type `T` (`u8` or `f32`), in layout `L`
/// ([`Hwc`] unless converted with [`ToChw`](super::ToChw)).
///
/// It is never empty and has 1 to 4 channels; [`new`](Self::new) checks that the data has the
/// size of the shape.
#[derive(Clone, PartialEq)]
pub struct Image<T, L = Hwc> {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<T>,
    layout: PhantomData<L>,
}

impl<T: Sample, L: Layout> Image<T, L> {
    /// An image of `data`, `height × width × channels` values in layout `L`.
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<T>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(invalid(format!("empty image of {width}x{height}")));
        }
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(invalid(format!(
                "{channels} channels, expected 1 to {MAX_CHANNELS}"
            )));
        }
        let len = sample_count(width, height, channels)?;
        if data.len() != len {
            return Err(invalid(format!(
                "{} values for a {width}x{height} image of {channels} channels ({len})",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
            layout: PhantomData,
        })
    }

    /// Width, in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height, in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Values per pixel.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The shape in this layout: `[height, width, channels]` or `[channels, height, width]`.
    pub fn shape(&self) -> [usize; 3] {
        L::shape(self.height, self.width, self.channels)
    }

    /// The values, in layout `L`.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The values, mutable.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The values, in layout `L`.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

/// The shape, type and layout; not the values.
impl<T: Sample, L: Layout> Debug for Image<T, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [a, b, c] = self.shape();
        write!(f, "Image<{}, {}>({a}x{b}x{c})", T::NAME, L::NAME)
    }
}

/// An [`ImageError::Invalid`] as a crate error.
pub(crate) fn invalid(message: String) -> crate::Error {
    ImageError::Invalid(message).into()
}

/// An [`ImageError::Limits`] as a crate error.
pub(crate) fn too_large(message: String) -> crate::Error {
    ImageError::Limits(message).into()
}

/// `width × height × channels`, or an error if it overflows.
pub(crate) fn sample_count(width: usize, height: usize, channels: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| too_large(format!("{width}x{height}x{channels} overflows")))
}

/// An empty `Vec` with room for `len` values, or an error instead of an abort when the memory
/// is not there.
pub(crate) fn with_capacity<T>(len: usize) -> Result<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| too_large(format!("cannot allocate {len} values")))?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::{Image, Sample};
    use crate::image::{Chw, Hwc};

    #[test]
    fn new_checks_the_shape() {
        let image = Image::<u8, Hwc>::new(3, 2, 1, vec![0; 6]).unwrap();
        assert_eq!(image.shape(), [2, 3, 1]);
        let planar = Image::<u8, Chw>::new(3, 2, 4, vec![0; 24]).unwrap();
        assert_eq!(planar.shape(), [4, 2, 3]);
        for (w, h, c, len) in [(0, 2, 1, 0), (3, 2, 5, 30), (3, 2, 0, 0), (3, 2, 1, 5)] {
            assert!(Image::<u8>::new(w, h, c, vec![0; len]).is_err());
        }
        let huge = Image::<u8>::new(usize::MAX, 2, 1, Vec::new()).unwrap_err();
        assert!(huge.to_string().contains("overflows"), "{huge}");
    }

    #[test]
    fn fill_values_must_fit_the_sample_type() {
        assert_eq!(u8::from_f32(255.0), Some(255));
        assert_eq!(u8::from_f32(256.0), None);
        assert_eq!(u8::from_f32(-1.0), None);
        assert_eq!(u8::from_f32(0.5), None);
        assert_eq!(f32::from_f32(0.5), Some(0.5));
    }
}
