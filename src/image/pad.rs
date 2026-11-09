//! [`Pad`] (torchvision's, with its four modes) and [`PadTo`] (up to a fixed size, as
//! prime_dx's `ConstantFill2d`).

use super::buffer::{invalid, sample_count, too_large, with_capacity};
use super::{Image, Sample};
use crate::{error::Result, transform::Transform};

/// The value of padded pixels: one for every channel, or one per channel. Values must be
/// samples of the image's type: integers in `0..=255` for `u8` images.
#[derive(Clone, Debug, PartialEq)]
pub struct Fill(Vec<f32>);

impl Fill {
    /// The pixel of `channels` values this fill gives.
    fn pixel<T: Sample>(&self, channels: usize) -> Result<Vec<T>> {
        let values: Vec<f32> = match self.0.len() {
            1 => vec![self.0[0]; channels],
            n if n == channels => self.0.clone(),
            n => {
                return Err(invalid(format!(
                    "a fill of {n} values for an image of {channels} channels"
                )));
            }
        };
        values
            .into_iter()
            .map(|v| {
                T::from_f32(v)
                    .ok_or_else(|| invalid(format!("fill {v} is not a {} value", T::NAME)))
            })
            .collect()
    }
}

impl From<f32> for Fill {
    fn from(value: f32) -> Self {
        Self(vec![value])
    }
}

impl From<Vec<f32>> for Fill {
    fn from(values: Vec<f32>) -> Self {
        Self(values)
    }
}

impl<const N: usize> From<[f32; N]> for Fill {
    fn from(values: [f32; N]) -> Self {
        Self(values.to_vec())
    }
}

/// What padded pixels hold, as torchvision's `padding_mode` (the same as `numpy.pad`'s modes of
/// these names).
#[derive(Clone, Debug, PartialEq)]
pub enum PadMode {
    /// A fixed value: `abc → 00abc00`.
    Constant(Fill),
    /// The edge pixel repeated: `abc → aaabccc`.
    Edge,
    /// Mirrored without the edge pixel: `abc → cbabcba`.
    Reflect,
    /// Mirrored with the edge pixel: `abc → babccba`.
    Symmetric,
}

/// Add `left`, `top`, `right` and `bottom` pixels around the image, as torchvision's `Pad`.
/// Padding wider than the image keeps mirroring, as `numpy.pad` (torchvision on PIL images).
#[derive(Clone, Debug, PartialEq)]
pub struct Pad {
    sides: [usize; 4],
    mode: PadMode,
}

impl Pad {
    /// Padding of the four sides, in torchvision's order, filled per `mode`.
    pub fn new(left: usize, top: usize, right: usize, bottom: usize, mode: PadMode) -> Self {
        Self {
            sides: [left, top, right, bottom],
            mode,
        }
    }
}

impl<T: Sample> Transform<Image<T>> for Pad {
    type Output = Image<T>;

    fn apply(&self, _index: usize, image: Image<T>) -> Result<Image<T>> {
        pad(&image, self.sides, &self.mode)
    }
}

/// Where [`PadTo`] puts the image in the padded one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Anchor {
    /// Padding only right and bottom, as prime_dx's `ConstantFill2d`.
    #[default]
    TopLeft,
    /// Padding split evenly, the extra pixel right or bottom.
    Center,
}

/// Pad with a constant up to `width × height`, never cropping: a side already that long or
/// longer is left as it is. With the default top-left anchor this is prime_dx's
/// `ConstantFill2d`, `np.pad(img, ((0, H - h), (0, W - w), (0, 0)))`.
#[derive(Clone, Debug, PartialEq)]
pub struct PadTo {
    width: usize,
    height: usize,
    mode: PadMode,
    anchor: Anchor,
}

impl PadTo {
    /// Pad up to `width × height` with `fill`, right and bottom.
    pub fn new(width: usize, height: usize, fill: impl Into<Fill>) -> Self {
        Self {
            width,
            height,
            mode: PadMode::Constant(fill.into()),
            anchor: Anchor::TopLeft,
        }
    }

    /// Where the image goes; default [`Anchor::TopLeft`].
    pub fn anchor(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }
}

impl<T: Sample> Transform<Image<T>> for PadTo {
    type Output = Image<T>;

    fn apply(&self, _index: usize, image: Image<T>) -> Result<Image<T>> {
        let pad_w = self.width.saturating_sub(image.width());
        let pad_h = self.height.saturating_sub(image.height());
        if pad_w == 0 && pad_h == 0 {
            return Ok(image);
        }
        let (left, top) = match self.anchor {
            Anchor::TopLeft => (0, 0),
            Anchor::Center => (pad_w / 2, pad_h / 2),
        };
        pad(&image, [left, top, pad_w - left, pad_h - top], &self.mode)
    }
}

/// The source position of output position `at` along an axis of `len` pixels with `before`
/// padded ones in front, or `None` for a constant.
fn source(at: usize, before: usize, len: usize, mode: &PadMode) -> Option<usize> {
    // i128 holds any difference of two usizes.
    let p = at as i128 - before as i128;
    let n = len as i128;
    if (0..n).contains(&p) {
        return Some(p as usize);
    }
    let q = match mode {
        PadMode::Constant(_) => return None,
        PadMode::Edge => p.clamp(0, n - 1),
        PadMode::Reflect if n == 1 => 0,
        PadMode::Reflect => {
            let q = p.rem_euclid(2 * (n - 1));
            if q < n { q } else { 2 * (n - 1) - q }
        }
        PadMode::Symmetric => {
            let q = p.rem_euclid(2 * n);
            if q < n { q } else { 2 * n - 1 - q }
        }
    };
    Some(q as usize)
}

/// `image` with `[left, top, right, bottom]` pixels added per `mode`.
pub(crate) fn pad<T: Sample>(
    image: &Image<T>,
    sides: [usize; 4],
    mode: &PadMode,
) -> Result<Image<T>> {
    let [left, top, right, bottom] = sides;
    let (w, h, c) = (image.width(), image.height(), image.channels());
    let overflow = || too_large(format!("padding {sides:?} of a {h}x{w} image overflows"));
    let out_w = w
        .checked_add(left)
        .and_then(|n| n.checked_add(right))
        .ok_or_else(overflow)?;
    let out_h = h
        .checked_add(top)
        .and_then(|n| n.checked_add(bottom))
        .ok_or_else(overflow)?;
    let fill = match mode {
        PadMode::Constant(fill) => fill.pixel::<T>(c)?,
        _ => Vec::new(),
    };
    let mut data = with_capacity(sample_count(out_w, out_h, c)?)?;
    let rows: Vec<&[T]> = image.data().chunks_exact(w * c).collect();
    for y in 0..out_h {
        let Some(row) = source(y, top, h, mode).map(|sy| rows[sy]) else {
            for _ in 0..out_w {
                data.extend_from_slice(&fill);
            }
            continue;
        };
        for x in 0..left {
            push_pixel(&mut data, row, source(x, left, w, mode), c, &fill);
        }
        data.extend_from_slice(row);
        for x in w + left..out_w {
            push_pixel(&mut data, row, source(x, left, w, mode), c, &fill);
        }
    }
    Image::new(out_w, out_h, c, data)
}

/// Pixel `x` of `row`, or `fill`.
fn push_pixel<T: Sample>(data: &mut Vec<T>, row: &[T], x: Option<usize>, c: usize, fill: &[T]) {
    match x {
        Some(x) => data.extend_from_slice(&row[x * c..(x + 1) * c]),
        None => data.extend_from_slice(fill),
    }
}

#[cfg(test)]
mod tests {
    use super::{Anchor, Fill, Pad, PadMode, PadTo, source};
    use crate::image::Image;
    use crate::transform::Transform;

    fn row(values: &[u8]) -> Image<u8> {
        Image::new(values.len(), 1, 1, values.to_vec()).unwrap()
    }

    fn pad_row(values: &[u8], before: usize, after: usize, mode: PadMode) -> Vec<u8> {
        Pad::new(before, 0, after, 0, mode)
            .apply(0, row(values))
            .unwrap()
            .into_vec()
    }

    #[test]
    fn the_four_modes() {
        let zero = PadMode::Constant(Fill::from(0.0));
        assert_eq!(pad_row(&[1, 2, 3], 2, 2, zero), [0, 0, 1, 2, 3, 0, 0]);
        assert_eq!(
            pad_row(&[1, 2, 3], 2, 2, PadMode::Edge),
            [1, 1, 1, 2, 3, 3, 3]
        );
        assert_eq!(
            pad_row(&[1, 2, 3], 2, 2, PadMode::Reflect),
            [3, 2, 1, 2, 3, 2, 1]
        );
        assert_eq!(
            pad_row(&[1, 2, 3], 2, 2, PadMode::Symmetric),
            [2, 1, 1, 2, 3, 3, 2]
        );
    }

    #[test]
    fn padding_wider_than_the_image_keeps_mirroring() {
        // numpy.pad([1, 2, 3], 5, mode) for each mode.
        assert_eq!(
            pad_row(&[1, 2, 3], 5, 5, PadMode::Reflect),
            [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2]
        );
        assert_eq!(
            pad_row(&[1, 2, 3], 5, 5, PadMode::Symmetric),
            [2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2]
        );
        assert_eq!(pad_row(&[7], 2, 1, PadMode::Reflect), [7, 7, 7, 7]);
        assert_eq!(source(0, usize::MAX, 3, &PadMode::Edge), Some(0));
    }

    #[test]
    fn fill_per_channel_and_type_checks() {
        let rgb = Image::new(1, 1, 3, vec![9u8, 9, 9]).unwrap();
        let fill = PadMode::Constant(Fill::from([1.0, 2.0, 3.0]));
        let out = Pad::new(1, 0, 0, 0, fill).apply(0, rgb.clone()).unwrap();
        assert_eq!(out.data(), [1, 2, 3, 9, 9, 9]);
        for bad in [Fill::from([1.0, 2.0]), Fill::from(256.0), Fill::from(0.5)] {
            let pad = Pad::new(1, 0, 0, 0, PadMode::Constant(bad));
            assert!(pad.apply(0, rgb.clone()).is_err());
        }
        let float = Image::new(1, 1, 1, vec![0.5f32]).unwrap();
        let pad = Pad::new(1, 0, 0, 0, PadMode::Constant(Fill::from(-1.5)));
        assert_eq!(pad.apply(0, float).unwrap().data(), [-1.5, 0.5]);
    }

    #[test]
    fn huge_padding_is_an_error() {
        let pad = Pad::new(usize::MAX, 0, 1, 0, PadMode::Edge);
        assert!(pad.apply(0, row(&[1])).is_err());
        let pad = Pad::new(1 << 40, 0, 0, 1 << 40, PadMode::Edge);
        assert!(pad.apply(0, row(&[1])).is_err());
    }

    #[test]
    fn pad_to_never_crops() {
        let image = Image::new(3, 2, 1, vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        let out = PadTo::new(4, 3, 0.0).apply(0, image.clone()).unwrap();
        assert_eq!(out.data(), [1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0]);
        let out = PadTo::new(2, 3, 0.0).apply(0, image.clone()).unwrap();
        assert_eq!(out.shape(), [3, 3, 1]);
        let same = PadTo::new(1, 1, 0.0).apply(0, image.clone()).unwrap();
        assert_eq!(same, image);
        let centered = PadTo::new(5, 2, 0.0)
            .anchor(Anchor::Center)
            .apply(0, image)
            .unwrap();
        assert_eq!(centered.data(), [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]);
    }
}
