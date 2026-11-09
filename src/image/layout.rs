//! The memory layout of an [`Image`](super::Image): [`Hwc`] (interleaved, the default) or
//! [`Chw`] (planar, PyTorch's).

use std::fmt::Debug;

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
