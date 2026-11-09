//! [`Normalize`]: `(x - mean) / std` per channel, as torchvision's.

use super::buffer::invalid;
use super::{Chw, Hwc, Image};
use crate::{error::Result, transform::Transform};

/// `(x - mean[c]) / std[c]` for each value of channel `c` of an `f32` image, in either layout:
/// torchvision's `Normalize`, bit for bit (a subtraction then a division, in `f32`).
///
/// One value applies to every channel; otherwise there is one per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// Normalize with `mean` and `std`: as many values each, none of `std` zero, all finite.
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Result<Self> {
        if mean.is_empty() || mean.len() != std.len() {
            return Err(invalid(format!(
                "Normalize needs as many means as stds, at least one: {} and {}",
                mean.len(),
                std.len()
            )));
        }
        if std.contains(&0.0) || !mean.iter().chain(&std).all(|v| v.is_finite()) {
            return Err(invalid(format!(
                "Normalize needs finite values and non-zero stds: mean {mean:?}, std {std:?}"
            )));
        }
        Ok(Self { mean, std })
    }

    /// ImageNet's mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    /// The `(mean, std)` of each of `channels`.
    fn per_channel(&self, channels: usize) -> Result<Vec<(f32, f32)>> {
        match self.mean.len() {
            1 => Ok(vec![(self.mean[0], self.std[0]); channels]),
            n if n == channels => Ok(self
                .mean
                .iter()
                .copied()
                .zip(self.std.iter().copied())
                .collect()),
            n => Err(invalid(format!(
                "Normalize of {n} channels on an image of {channels}"
            ))),
        }
    }
}

impl Transform<Image<f32, Hwc>> for Normalize {
    type Output = Image<f32, Hwc>;

    fn apply(&self, _index: usize, mut image: Image<f32, Hwc>) -> Result<Image<f32, Hwc>> {
        let channels = self.per_channel(image.channels())?;
        for pixel in image.data_mut().chunks_exact_mut(channels.len()) {
            for (value, &(mean, std)) in pixel.iter_mut().zip(&channels) {
                *value = (*value - mean) / std;
            }
        }
        Ok(image)
    }
}

impl Transform<Image<f32, Chw>> for Normalize {
    type Output = Image<f32, Chw>;

    fn apply(&self, _index: usize, mut image: Image<f32, Chw>) -> Result<Image<f32, Chw>> {
        let channels = self.per_channel(image.channels())?;
        let plane = image.width() * image.height();
        for (values, &(mean, std)) in image.data_mut().chunks_exact_mut(plane).zip(&channels) {
            for value in values {
                *value = (*value - mean) / std;
            }
        }
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::Normalize;
    use crate::image::{Chw, Hwc, Image};
    use crate::transform::Transform;

    #[test]
    fn both_layouts_give_the_same_values() {
        let norm = Normalize::new(vec![0.5, 0.0], vec![0.5, 2.0]).unwrap();
        let hwc = Image::<f32, Hwc>::new(2, 1, 2, vec![1.0, 4.0, 0.0, 2.0]).unwrap();
        assert_eq!(norm.apply(0, hwc).unwrap().data(), [1.0, 2.0, -1.0, 1.0]);
        let chw = Image::<f32, Chw>::new(2, 1, 2, vec![1.0, 0.0, 4.0, 2.0]).unwrap();
        assert_eq!(norm.apply(0, chw).unwrap().data(), [1.0, -1.0, 2.0, 1.0]);
    }

    #[test]
    fn bad_settings_are_errors() {
        assert!(Normalize::new(vec![], vec![]).is_err());
        assert!(Normalize::new(vec![0.0], vec![1.0, 2.0]).is_err());
        assert!(Normalize::new(vec![0.0], vec![0.0]).is_err());
        assert!(Normalize::new(vec![f32::NAN], vec![1.0]).is_err());
        let one = Normalize::new(vec![1.0], vec![2.0]).unwrap();
        let rgb = Image::<f32>::new(1, 1, 3, vec![3.0; 3]).unwrap();
        assert_eq!(one.apply(0, rgb.clone()).unwrap().data(), [1.0; 3]);
        let two = Normalize::new(vec![0.0; 2], vec![1.0; 2]).unwrap();
        assert!(two.apply(0, rgb).is_err());
    }
}
