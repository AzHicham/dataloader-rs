//! The memory layout of an [`Image`]: [`Hwc`] (interleaved, the default) or [`Chw`] (planar,
//! PyTorch's), and [`ToChw`] / [`ToHwc`] between them.

use std::fmt::Debug;

use super::buffer::{filled, with_capacity};
use super::{Image, Sample};
use crate::{error::Result, transform::Transform};

mod sealed {
    pub trait Sealed {}
}

/// The order of an image's values in memory. Implemented by [`Hwc`] and [`Chw`] only.
pub trait Layout:
    sealed::Sealed + Clone + Copy + Debug + Default + PartialEq + Send + Sync + 'static
{
    /// `"HWC"` or `"CHW"`.
    const NAME: &'static str;

    /// The shape of a `height × width × channels` image in this layout.
    fn shape(height: usize, width: usize, channels: usize) -> [usize; 3];
}

/// Row by row, the channels of each pixel together: `[height, width, channels]`. What decoders
/// produce and every geometric transform takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hwc;

/// Channel by channel, each a `height × width` plane: `[channels, height, width]`, the layout of
/// a PyTorch image tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chw;

impl sealed::Sealed for Hwc {}
impl sealed::Sealed for Chw {}

impl Layout for Hwc {
    const NAME: &'static str = "HWC";

    fn shape(height: usize, width: usize, channels: usize) -> [usize; 3] {
        [height, width, channels]
    }
}

impl Layout for Chw {
    const NAME: &'static str = "CHW";

    fn shape(height: usize, width: usize, channels: usize) -> [usize; 3] {
        [channels, height, width]
    }
}

/// From [`Hwc`] to [`Chw`]: the layout change of torchvision's `ToTensor`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToChw;

/// From [`Chw`] back to [`Hwc`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToHwc;

impl<T: Sample> Transform<Image<T, Hwc>> for ToChw {
    type Output = Image<T, Chw>;

    fn apply(&self, _index: usize, image: Image<T, Hwc>) -> Result<Image<T, Chw>> {
        let (w, h, c) = (image.width(), image.height(), image.channels());
        let mut data = with_capacity(image.data().len())?;
        for channel in 0..c {
            data.extend(image.data().iter().skip(channel).step_by(c));
        }
        Image::new(w, h, c, data)
    }
}

impl<T: Sample> Transform<Image<T, Chw>> for ToHwc {
    type Output = Image<T, Hwc>;

    fn apply(&self, _index: usize, image: Image<T, Chw>) -> Result<Image<T, Hwc>> {
        let (w, h, c) = (image.width(), image.height(), image.channels());
        let mut data = filled(image.data().len(), T::default())?;
        for (channel, plane) in image.data().chunks_exact(w * h).enumerate() {
            for (out, &value) in data.iter_mut().skip(channel).step_by(c).zip(plane) {
                *out = value;
            }
        }
        Image::new(w, h, c, data)
    }
}

#[cfg(test)]
mod tests {
    use super::{Chw, Hwc, ToChw, ToHwc};
    use crate::image::Image;
    use crate::transform::Transform;

    #[test]
    fn round_trip() {
        // 1 row of 2 RGB pixels.
        let hwc = Image::<u8, Hwc>::new(2, 1, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let chw: Image<u8, Chw> = ToChw.apply(0, hwc.clone()).unwrap();
        assert_eq!(chw.shape(), [3, 1, 2]);
        assert_eq!(chw.data(), [1, 4, 2, 5, 3, 6]);
        assert_eq!(ToHwc.apply(0, chw).unwrap(), hwc);
    }
}
