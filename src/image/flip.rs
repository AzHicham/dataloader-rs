//! [`HFlip`] and [`VFlip`], as torchvision's.

use super::buffer::with_capacity;
use super::{Image, Sample};
use crate::{error::Result, transform::Transform};

/// Mirror left to right, as torchvision's `F.hflip`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HFlip;

/// Mirror top to bottom, as torchvision's `F.vflip`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VFlip;

impl<T: Sample> Transform<Image<T>> for HFlip {
    type Output = Image<T>;

    fn apply(&self, _index: usize, image: Image<T>) -> Result<Image<T>> {
        let (w, h, c) = (image.width(), image.height(), image.channels());
        let mut data = with_capacity(image.data().len())?;
        for row in image.data().chunks_exact(w * c) {
            data.extend(row.chunks_exact(c).rev().flatten());
        }
        Image::new(w, h, c, data)
    }
}

impl<T: Sample> Transform<Image<T>> for VFlip {
    type Output = Image<T>;

    fn apply(&self, _index: usize, image: Image<T>) -> Result<Image<T>> {
        let (w, h, c) = (image.width(), image.height(), image.channels());
        let mut data = with_capacity(image.data().len())?;
        for row in image.data().chunks_exact(w * c).rev() {
            data.extend_from_slice(row);
        }
        Image::new(w, h, c, data)
    }
}

#[cfg(test)]
mod tests {
    use super::{HFlip, VFlip};
    use crate::image::Image;
    use crate::transform::Transform;

    #[test]
    fn flips() {
        // 2 rows of 2 pixels of 2 channels.
        let image = Image::new(2, 2, 2, (0u8..8).collect()).unwrap();
        let h = HFlip.apply(0, image.clone()).unwrap();
        assert_eq!(h.data(), [2, 3, 0, 1, 6, 7, 4, 5]);
        let v = VFlip.apply(0, image).unwrap();
        assert_eq!(v.data(), [4, 5, 6, 7, 0, 1, 2, 3]);
    }
}
