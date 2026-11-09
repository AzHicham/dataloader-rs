//! [`Crop`] and [`CenterCrop`], as torchvision's.

use super::buffer::{invalid, sample_count, with_capacity};
use super::pad::{Fill, PadMode, pad};
use super::{Image, Sample};
use crate::{error::Result, transform::Transform};

/// The `height × width` region at `(top, left)`, as torchvision's `F.crop`; a region not fully
/// inside the image is an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    top: usize,
    left: usize,
    height: usize,
    width: usize,
}

impl Crop {
    /// The region of `height × width` pixels whose top-left pixel is `(top, left)`.
    pub fn new(top: usize, left: usize, height: usize, width: usize) -> Self {
        Self {
            top,
            left,
            height,
            width,
        }
    }
}

impl<T: Sample> Transform<Image<T>> for Crop {
    type Output = Image<T>;

    fn apply(&self, _index: usize, image: Image<T>) -> Result<Image<T>> {
        crop(&image, self.top, self.left, self.height, self.width)
    }
}

/// The central `height × width` region, as torchvision's `CenterCrop`: an image smaller than
/// that is first padded with zeros, evenly (the extra pixel right or bottom); the region's
/// offset is rounded half to even, as Python's `round`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenterCrop {
    height: usize,
    width: usize,
}

impl CenterCrop {
    /// The central `height × width` region.
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

impl<T: Sample> Transform<Image<T>> for CenterCrop {
    type Output = Image<T>;

    fn apply(&self, _index: usize, mut image: Image<T>) -> Result<Image<T>> {
        let (height, width) = (self.height, self.width);
        if height > image.height() || width > image.width() {
            let pad_w = width.saturating_sub(image.width());
            let pad_h = height.saturating_sub(image.height());
            let (left, top) = (pad_w / 2, pad_h / 2);
            let zeros = PadMode::Constant(Fill::from(0.0));
            image = pad(&image, [left, top, pad_w - left, pad_h - top], &zeros)?;
        }
        let top = half_to_even(image.height() - height);
        let left = half_to_even(image.width() - width);
        crop(&image, top, left, height, width)
    }
}

/// `round(n / 2)` in Python: halves go to the even neighbour.
fn half_to_even(n: usize) -> usize {
    let half = n / 2;
    match n % 2 == 1 && half % 2 == 1 {
        true => half + 1,
        false => half,
    }
}

/// The `height × width` region of `image` at `(top, left)`.
pub(crate) fn crop<T: Sample>(
    image: &Image<T>,
    top: usize,
    left: usize,
    height: usize,
    width: usize,
) -> Result<Image<T>> {
    let inside =
        |start: usize, len: usize, max: usize| start.checked_add(len).is_some_and(|end| end <= max);
    if !inside(top, height, image.height()) || !inside(left, width, image.width()) {
        return Err(invalid(format!(
            "crop of {height}x{width} at ({top}, {left}) is outside the {}x{} image",
            image.height(),
            image.width()
        )));
    }
    let channels = image.channels();
    let row_len = image.width() * channels;
    let mut data = with_capacity(sample_count(width, height, channels)?)?;
    for row in image.data().chunks_exact(row_len).skip(top).take(height) {
        data.extend_from_slice(&row[left * channels..(left + width) * channels]);
    }
    Image::new(width, height, channels, data)
}

#[cfg(test)]
mod tests {
    use super::{CenterCrop, Crop, half_to_even};
    use crate::image::Image;
    use crate::transform::Transform;

    /// A `height × width` gray image whose values are their row-major position.
    fn ramp(height: usize, width: usize) -> Image<u8> {
        Image::new(width, height, 1, (0..(width * height) as u8).collect()).unwrap()
    }

    #[test]
    fn crop_takes_the_region() {
        let out = Crop::new(1, 2, 2, 3).apply(0, ramp(4, 5)).unwrap();
        assert_eq!(out.shape(), [2, 3, 1]);
        assert_eq!(out.data(), [7, 8, 9, 12, 13, 14]);
    }

    #[test]
    fn crop_outside_is_an_error() {
        for crop in [
            Crop::new(3, 0, 2, 1),
            Crop::new(0, 4, 1, 2),
            Crop::new(usize::MAX, 0, 2, 1),
            Crop::new(0, 0, 1, 0),
        ] {
            assert!(crop.apply(0, ramp(4, 5)).is_err(), "{crop:?}");
        }
    }

    #[test]
    fn python_rounding() {
        let rounded: Vec<usize> = (0..8).map(half_to_even).collect();
        // round(0/2), round(1/2) = 0, round(3/2) = 2, round(5/2) = 2, round(7/2) = 4
        assert_eq!(rounded, [0, 0, 1, 2, 2, 2, 3, 4]);
    }

    #[test]
    fn center_crop_pads_small_images() {
        let out = CenterCrop::new(3, 4).apply(0, ramp(1, 1)).unwrap();
        assert_eq!(out.shape(), [3, 4, 1]);
        assert_eq!(out.data(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let out = CenterCrop::new(1, 2).apply(0, ramp(3, 5)).unwrap();
        // top = round(1) = 1, left = round(1.5) = 2
        assert_eq!(out.data(), [7, 8]);
    }
}
