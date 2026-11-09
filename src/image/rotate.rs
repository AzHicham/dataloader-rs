//! [`Rotate90`]: lossless rotations by quarter turns.

use super::buffer::with_capacity;
use super::{Image, Sample};
use crate::{error::Result, transform::Transform};

/// Rotate by `k` quarter turns counter-clockwise, as `numpy.rot90(image, k)` and
/// `torch.rot90(image, k, dims=(-2, -1))`; a negative `k` turns clockwise. Lossless.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rotate90 {
    /// Quarter turns counter-clockwise, in `0..4`.
    turns: u8,
}

impl Rotate90 {
    /// `k` quarter turns counter-clockwise.
    pub fn new(k: i64) -> Self {
        Self {
            turns: k.rem_euclid(4) as u8,
        }
    }
}

impl<T: Sample> Transform<Image<T>> for Rotate90 {
    type Output = Image<T>;

    fn apply(&self, _index: usize, image: Image<T>) -> Result<Image<T>> {
        let (w, h, c) = (image.width(), image.height(), image.channels());
        let (out_w, out_h) = match self.turns {
            0 => return Ok(image),
            2 => (w, h),
            _ => (h, w),
        };
        let src = image.data();
        // Output pixel (y, x) comes from input pixel (sy, sx).
        let at = |y: usize, x: usize| -> (usize, usize) {
            match self.turns {
                1 => (x, w - 1 - y),
                2 => (h - 1 - y, w - 1 - x),
                _ => (h - 1 - x, y),
            }
        };
        let mut data = with_capacity(src.len())?;
        for y in 0..out_h {
            for x in 0..out_w {
                let (sy, sx) = at(y, x);
                let start = (sy * w + sx) * c;
                data.extend_from_slice(&src[start..start + c]);
            }
        }
        Image::new(out_w, out_h, c, data)
    }
}

#[cfg(test)]
mod tests {
    use super::Rotate90;
    use crate::image::Image;
    use crate::transform::Transform;

    #[test]
    fn quarter_turns_like_numpy() {
        // [[0, 1, 2], [3, 4, 5]]
        let image = Image::new(3, 2, 1, (0u8..6).collect()).unwrap();
        let turn = |k| Rotate90::new(k).apply(0, image.clone()).unwrap();
        // np.rot90(a, 1) = [[2, 5], [1, 4], [0, 3]]
        assert_eq!(turn(1).shape(), [3, 2, 1]);
        assert_eq!(turn(1).data(), [2, 5, 1, 4, 0, 3]);
        assert_eq!(turn(2).data(), [5, 4, 3, 2, 1, 0]);
        // np.rot90(a, 3) = [[3, 0], [4, 1], [5, 2]]
        assert_eq!(turn(3).data(), [3, 0, 4, 1, 5, 2]);
        assert_eq!(turn(-1), turn(3));
        assert_eq!(turn(4), image);
    }
}
