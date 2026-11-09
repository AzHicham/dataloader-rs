//! [`Resize`] with `fast_image_resize`, sizes computed as torchvision's.

use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use super::buffer::{invalid, sample_count, too_large, with_capacity};
use super::{DecodeLimits, Image};
use crate::{error::Result, transform::Transform};

/// The interpolation of [`Resize`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Filter {
    /// Nearest neighbour.
    Nearest,
    /// Linear (triangle) filter: torchvision's default.
    #[default]
    Bilinear,
    /// Catmull-Rom cubic (a = -0.5), as Pillow's `BICUBIC`.
    Bicubic,
    /// Lanczos with 3 lobes, as Pillow's `LANCZOS`.
    Lanczos3,
}

/// The size [`Resize`] gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Size {
    Exact {
        height: usize,
        width: usize,
    },
    Shorter {
        size: usize,
        max_size: Option<usize>,
    },
}

/// Resize an 8-bit image, as torchvision's `Resize`: to `height × width`, or with
/// [`shorter`](Self::shorter) its shorter side to `size` keeping the aspect ratio (the longer
/// side truncated, as torchvision does), optionally capped by `max_size`.
///
/// Antialiased by default, as torchvision v2 and Pillow: a downscale filters over the pixels
/// each output pixel covers; [`antialias(false)`](Self::antialias) keeps the filter's own
/// support (torch's `interpolate`). With alpha, colors are premultiplied, as Pillow. Values
/// are close to Pillow's, not bit-exact (fixed-point rounding differs). Resize before
/// [`ToFloat`](super::ToFloat); outputs above 16384 pixels per side or 256 MiB are an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resize {
    size: Size,
    filter: Filter,
    antialias: bool,
}

impl Resize {
    /// Resize to exactly `height × width`.
    pub fn new(height: usize, width: usize) -> Self {
        Self::with_size(Size::Exact { height, width })
    }

    /// Resize the shorter side to `size`, keeping the aspect ratio.
    pub fn shorter(size: usize) -> Self {
        Self::with_size(Size::Shorter {
            size,
            max_size: None,
        })
    }

    fn with_size(size: Size) -> Self {
        Self {
            size,
            filter: Filter::default(),
            antialias: true,
        }
    }

    /// With [`shorter`](Self::shorter): if the longer side would exceed `max_size`, it is
    /// `max_size` and the shorter one is scaled down to match. Ignored for an exact size.
    pub fn max_size(mut self, max_size: usize) -> Self {
        if let Size::Shorter { size, .. } = self.size {
            self.size = Size::Shorter {
                size,
                max_size: Some(max_size),
            };
        }
        self
    }

    /// The interpolation; default [`Filter::Bilinear`].
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Whether a downscale is antialiased; default `true`.
    pub fn antialias(mut self, antialias: bool) -> Self {
        self.antialias = antialias;
        self
    }

    /// The `(height, width)` an image of `height × width` is resized to.
    fn output_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        let (size, max_size) = match self.size {
            Size::Exact { height, width } => return Ok((height, width)),
            Size::Shorter { size, max_size } => (size, max_size),
        };
        // torchvision's `_compute_resized_output_size`, in floats as Python does.
        let (short, long) = (width.min(height), width.max(height));
        let (mut new_short, mut new_long) =
            (size, (size as f64 * long as f64 / short as f64) as usize);
        if let Some(max_size) = max_size {
            if max_size <= size {
                return Err(invalid(format!(
                    "max_size {max_size} must be above size {size}"
                )));
            }
            if new_long > max_size {
                new_short = (max_size as f64 * new_short as f64 / new_long as f64) as usize;
                new_long = max_size;
            }
        }
        Ok(match width <= height {
            true => (new_long, new_short),
            false => (new_short, new_long),
        })
    }

    fn algorithm(&self) -> ResizeAlg {
        let filter = match self.filter {
            Filter::Nearest => return ResizeAlg::Nearest,
            Filter::Bilinear => FilterType::Bilinear,
            Filter::Bicubic => FilterType::CatmullRom,
            Filter::Lanczos3 => FilterType::Lanczos3,
        };
        match self.antialias {
            true => ResizeAlg::Convolution(filter),
            false => ResizeAlg::Interpolation(filter),
        }
    }
}

impl Transform<Image<u8>> for Resize {
    type Output = Image<u8>;

    fn apply(&self, _index: usize, image: Image<u8>) -> Result<Image<u8>> {
        let (w, h, c) = (image.width(), image.height(), image.channels());
        let (out_h, out_w) = self.output_size(h, w)?;
        if (out_h, out_w) == (h, w) {
            return Ok(image);
        }
        let limits = DecodeLimits::default();
        let len = sample_count(out_w, out_h, c)?;
        let max_side = limits.max_width as usize;
        if out_w == 0
            || out_h == 0
            || out_w > max_side
            || out_h > max_side
            || len as u64 > limits.max_alloc
        {
            return Err(too_large(format!(
                "resize of a {h}x{w} image to {out_h}x{out_w}: above {max_side} per side or {} bytes, or empty",
                limits.max_alloc
            )));
        }
        let pixel_type = match c {
            1 => PixelType::U8,
            2 => PixelType::U8x2,
            3 => PixelType::U8x3,
            _ => PixelType::U8x4,
        };
        // Both sides are at most 16384 here, and the input's are below u32::MAX by
        // construction of an `Image` from a decoder or from these limits.
        let dims = |w: usize, h: usize| (u32::try_from(w), u32::try_from(h));
        let (Ok(src_w), Ok(src_h)) = dims(w, h) else {
            return Err(too_large(format!("{h}x{w} is too large to resize")));
        };
        let src = FirImage::from_vec_u8(src_w, src_h, image.into_vec(), pixel_type)
            .map_err(|e| invalid(e.to_string()))?;
        let mut out = with_capacity(len)?;
        out.resize(len, 0);
        let mut dst = FirImage::from_vec_u8(out_w as u32, out_h as u32, out, pixel_type)
            .map_err(|e| invalid(e.to_string()))?;
        let options = ResizeOptions::new().resize_alg(self.algorithm());
        Resizer::new()
            .resize(&src, &mut dst, &options)
            .map_err(|e| invalid(e.to_string()))?;
        Image::new(out_w, out_h, c, dst.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::{Filter, Resize};
    use crate::image::Image;
    use crate::transform::Transform;

    #[test]
    fn output_sizes_as_torchvision() {
        // (h, w) → Resize(224) / Resize(224, max_size=300)
        assert_eq!(
            Resize::shorter(224).output_size(480, 640).unwrap(),
            (224, 298)
        );
        assert_eq!(
            Resize::shorter(224).output_size(640, 480).unwrap(),
            (298, 224)
        );
        assert_eq!(
            Resize::shorter(224)
                .max_size(250)
                .output_size(480, 640)
                .unwrap(),
            (187, 250)
        );
        assert_eq!(Resize::shorter(10).output_size(7, 7).unwrap(), (10, 10));
        assert!(Resize::shorter(10).max_size(10).output_size(7, 9).is_err());
        assert_eq!(
            Resize::new(3, 4).max_size(1).output_size(9, 9).unwrap(),
            (3, 4)
        );
    }

    #[test]
    fn every_channel_count_and_filter_resizes() {
        for c in 1..=4 {
            for filter in [
                Filter::Nearest,
                Filter::Bilinear,
                Filter::Bicubic,
                Filter::Lanczos3,
            ] {
                let image = Image::new(8, 6, c, vec![255u8; 48 * c]).unwrap();
                let out = Resize::new(3, 5).filter(filter).apply(0, image).unwrap();
                assert_eq!(out.shape(), [3, 5, c]);
                assert!(out.data().iter().all(|&v| v == 255), "{c} {filter:?}");
            }
        }
    }

    #[test]
    fn hostile_aspect_ratios_are_capped() {
        // 1 × 16384 with its shorter side to 224 would be 224 × 3.6 M pixels.
        let image = Image::new(16_384, 1, 3, vec![0u8; 16_384 * 3]).unwrap();
        assert!(Resize::shorter(224).apply(0, image.clone()).is_err());
        assert!(Resize::new(0, 3).apply(0, image).is_err());
    }
}
