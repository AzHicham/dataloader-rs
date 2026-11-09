//! Channel conversions of 8-bit images, as Pillow's `convert`: [`ToRgb`], [`ToRgba`],
//! [`ToGray`].

use super::Image;
use super::buffer::{invalid, sample_count, with_capacity};
use crate::{error::Result, transform::Transform};

/// To 3 channels, as Pillow's `convert("RGB")`: gray is repeated, alpha is dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToRgb;

/// To 4 channels, as Pillow's `convert("RGBA")`: gray is repeated, alpha is 255 if missing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToRgba;

/// To 1 channel, as Pillow's `convert("L")` (ITU-R 601-2 luma, rounded the same way): alpha is
/// dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToGray;

impl Transform<Image<u8>> for ToRgb {
    type Output = Image<u8>;

    fn apply(&self, _index: usize, image: Image<u8>) -> Result<Image<u8>> {
        convert(image, 3)
    }
}

impl Transform<Image<u8>> for ToRgba {
    type Output = Image<u8>;

    fn apply(&self, _index: usize, image: Image<u8>) -> Result<Image<u8>> {
        convert(image, 4)
    }
}

impl Transform<Image<u8>> for ToGray {
    type Output = Image<u8>;

    fn apply(&self, _index: usize, image: Image<u8>) -> Result<Image<u8>> {
        convert(image, 1)
    }
}

/// Pillow's luma (`L24` in `Convert.c`): `(19595 R + 38470 G + 7471 B + 0x8000) >> 16`.
fn luma(r: u8, g: u8, b: u8) -> u8 {
    let sum = 19595 * u32::from(r) + 38470 * u32::from(g) + 7471 * u32::from(b) + 0x8000;
    (sum >> 16) as u8
}

/// `image` with `to` channels: 1 (gray), 3 (RGB) or 4 (RGBA).
fn convert(image: Image<u8>, to: usize) -> Result<Image<u8>> {
    let from = image.channels();
    if from == to {
        return Ok(image);
    }
    let (width, height) = (image.width(), image.height());
    let mut out = with_capacity(sample_count(width, height, to)?)?;
    let pixels = image.data().chunks_exact(from);
    match (from, to) {
        (1, 3) => out.extend(pixels.flat_map(|p| [p[0], p[0], p[0]])),
        (2, 3) => out.extend(pixels.flat_map(|p| [p[0], p[0], p[0]])),
        (4, 3) => out.extend(pixels.flat_map(|p| [p[0], p[1], p[2]])),
        (1, 4) => out.extend(pixels.flat_map(|p| [p[0], p[0], p[0], 255])),
        (2, 4) => out.extend(pixels.flat_map(|p| [p[0], p[0], p[0], p[1]])),
        (3, 4) => out.extend(pixels.flat_map(|p| [p[0], p[1], p[2], 255])),
        (2, 1) => out.extend(pixels.map(|p| p[0])),
        (3 | 4, 1) => out.extend(pixels.map(|p| luma(p[0], p[1], p[2]))),
        _ => {
            return Err(invalid(format!("cannot convert {from} channels to {to}")));
        }
    }
    Image::new(width, height, to, out)
}

#[cfg(test)]
mod tests {
    use super::{ToGray, ToRgb, ToRgba};
    use crate::image::Image;
    use crate::transform::Transform;

    fn image(channels: usize, data: &[u8]) -> Image<u8> {
        Image::new(1, data.len() / channels, channels, data.to_vec()).unwrap()
    }

    #[test]
    fn every_channel_count_converts() {
        let gray = image(1, &[7]);
        let la = image(2, &[7, 9]);
        let rgb = image(3, &[10, 20, 30]);
        let rgba = image(4, &[10, 20, 30, 40]);
        let rgb_of = |i: &Image<u8>| ToRgb.apply(0, i.clone()).unwrap().into_vec();
        let rgba_of = |i: &Image<u8>| ToRgba.apply(0, i.clone()).unwrap().into_vec();
        let gray_of = |i: &Image<u8>| ToGray.apply(0, i.clone()).unwrap().into_vec();
        assert_eq!(rgb_of(&gray), [7, 7, 7]);
        assert_eq!(rgb_of(&la), [7, 7, 7]);
        assert_eq!(rgb_of(&rgb), [10, 20, 30]);
        assert_eq!(rgb_of(&rgba), [10, 20, 30]);
        assert_eq!(rgba_of(&gray), [7, 7, 7, 255]);
        assert_eq!(rgba_of(&la), [7, 7, 7, 9]);
        assert_eq!(rgba_of(&rgb), [10, 20, 30, 255]);
        assert_eq!(gray_of(&la), [7]);
        // (19595·10 + 38470·20 + 7471·30 + 32768) >> 16 = 18
        assert_eq!(gray_of(&rgb), [18]);
        assert_eq!(gray_of(&rgba), [18]);
        assert_eq!(gray_of(&image(3, &[255, 255, 255])), [255]);
    }

    #[test]
    fn only_rgb_gray_and_rgba_are_targets() {
        let err = super::convert(image(3, &[1, 2, 3]), 2).unwrap_err();
        assert!(err.to_string().contains("cannot convert 3 channels to 2"));
    }
}
