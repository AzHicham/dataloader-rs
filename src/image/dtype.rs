//! [`ToFloat`]: 8-bit samples to `f32`, as torchvision's `ToTensor`.

use super::buffer::with_capacity;
use super::{Image, Layout};
use crate::{error::Result, transform::Transform};

/// `u8` samples to `f32`, divided by 255 by default: torchvision's `ToTensor` (without its
/// layout change, see [`ToChw`](super::ToChw)) and `ToDtype(torch.float32, scale=True)`.
/// The same values bit for bit: `x as f32 / 255.0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToFloat {
    scale: bool,
}

impl ToFloat {
    /// With `scale`, values are divided by 255 (to `0.0..=1.0`); without, kept as they are.
    pub fn new(scale: bool) -> Self {
        Self { scale }
    }
}

impl Default for ToFloat {
    /// Divided by 255.
    fn default() -> Self {
        Self::new(true)
    }
}

impl<L: Layout> Transform<Image<u8, L>> for ToFloat {
    type Output = Image<f32, L>;

    fn apply(&self, _index: usize, image: Image<u8, L>) -> Result<Image<f32, L>> {
        let mut data = with_capacity(image.data().len())?;
        match self.scale {
            true => data.extend(image.data().iter().map(|&v| f32::from(v) / 255.0)),
            false => data.extend(image.data().iter().map(|&v| f32::from(v))),
        }
        Image::new(image.width(), image.height(), image.channels(), data)
    }
}

#[cfg(test)]
mod tests {
    use super::ToFloat;
    use crate::image::{Chw, Image};
    use crate::transform::Transform;

    #[test]
    fn scaled_and_not() {
        let image = Image::<u8, Chw>::new(3, 1, 1, vec![0, 51, 255]).unwrap();
        let scaled = ToFloat::default().apply(0, image.clone()).unwrap();
        assert_eq!(scaled.data(), [0.0, 0.2, 1.0]);
        assert_eq!(scaled.shape(), [1, 1, 3]);
        let plain = ToFloat::new(false).apply(0, image).unwrap();
        assert_eq!(plain.data(), [0.0, 51.0, 255.0]);
    }
}
