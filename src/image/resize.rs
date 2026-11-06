//! Resizing 8-bit images with `fast_image_resize` (pure Rust, SIMD).

use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image, ImageRef},
};

use super::{Channels, ImageError, ImageResult};

/// The resampling filter of a resize.
///
/// Every filter but `Nearest` is a convolution whose support grows with the downscale factor
/// (antialiasing), as in Pillow's `Image.resize`; the names follow Pillow's filters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ResizeFilter {
    /// The nearest source pixel: no interpolation, no antialiasing.
    Nearest,
    /// Box (area average when downscaling); Pillow's `BOX`.
    Box,
    /// Triangle filter; Pillow's `BILINEAR`, torchvision's default with `antialias=True`.
    #[default]
    Bilinear,
    /// Hamming-windowed sinc; Pillow's `HAMMING`.
    Hamming,
    /// Catmull-Rom cubic (`a = -0.5`); Pillow's `BICUBIC`.
    Bicubic,
    /// Mitchell–Netravali cubic (`B = C = 1/3`).
    Mitchell,
    /// Lanczos with 3 lobes; Pillow's `LANCZOS`.
    Lanczos3,
}

impl ResizeFilter {
    fn options(self) -> ResizeOptions {
        let algorithm = match self {
            Self::Nearest => ResizeAlg::Nearest,
            Self::Box => ResizeAlg::Convolution(FilterType::Box),
            Self::Bilinear => ResizeAlg::Convolution(FilterType::Bilinear),
            Self::Hamming => ResizeAlg::Convolution(FilterType::Hamming),
            Self::Bicubic => ResizeAlg::Convolution(FilterType::CatmullRom),
            Self::Mitchell => ResizeAlg::Convolution(FilterType::Mitchell),
            Self::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
        };
        ResizeOptions::new().resize_alg(algorithm)
    }
}

fn pixel_type(channels: Channels) -> PixelType {
    match channels {
        Channels::Rgb => PixelType::U8x3,
        Channels::Gray => PixelType::U8,
    }
}

/// Resize `src`, `width × height` interleaved pixels of `channels`, into `dst` as
/// `out_width × out_height` pixels. Sizes are within the pipeline's limits, so they fit `u32`.
pub(crate) fn resize(
    resizer: &mut Resizer,
    src: &[u8],
    (width, height): (usize, usize),
    dst: &mut Vec<u8>,
    (out_width, out_height): (usize, usize),
    channels: Channels,
    filter: ResizeFilter,
) -> ImageResult<()> {
    let too_large = || ImageError::Limits("resize: image too large".into());
    let len = out_width
        .checked_mul(out_height)
        .and_then(|n| n.checked_mul(channels.count()))
        .ok_or_else(too_large)?;
    dst.clear();
    dst.try_reserve_exact(len)?;
    dst.resize(len, 0);
    let side = |n: usize| u32::try_from(n).map_err(|_| too_large());
    let pixel_type = pixel_type(channels);
    let src = ImageRef::new(side(width)?, side(height)?, src, pixel_type)?;
    let mut out = Image::from_slice_u8(side(out_width)?, side(out_height)?, dst, pixel_type)?;
    resizer.resize(&src, &mut out, &filter.options())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use fast_image_resize::Resizer;

    use super::{ResizeFilter, resize};
    use crate::image::Channels;

    fn run(src: &[u8], from: (usize, usize), to: (usize, usize), f: ResizeFilter) -> Vec<u8> {
        let mut out = Vec::new();
        resize(
            &mut Resizer::new(),
            src,
            from,
            &mut out,
            to,
            Channels::Rgb,
            f,
        )
        .unwrap();
        out
    }

    #[test]
    fn constant_images_stay_constant() {
        let src = [7u8, 100, 250].repeat(13 * 9);
        for filter in [
            ResizeFilter::Nearest,
            ResizeFilter::Box,
            ResizeFilter::Bilinear,
            ResizeFilter::Hamming,
            ResizeFilter::Bicubic,
            ResizeFilter::Mitchell,
            ResizeFilter::Lanczos3,
        ] {
            for to in [(5, 4), (13, 9), (31, 17)] {
                let out = run(&src, (13, 9), to, filter);
                assert_eq!(
                    out,
                    [7u8, 100, 250].repeat(to.0 * to.1),
                    "{filter:?} {to:?}"
                );
            }
        }
    }

    #[test]
    fn bilinear_halving_keeps_a_step() {
        // A step from 0 to 200, halved: the left pixel stays dark, the right one bright.
        let src: Vec<u8> = [0u8, 0, 200, 200].iter().flat_map(|&v| [v, v, v]).collect();
        let out = run(&src, (4, 1), (2, 1), ResizeFilter::Bilinear);
        assert!(out[0] < 50 && out[3] > 150, "{out:?}");
    }

    #[test]
    fn gray_images() {
        let src = vec![9u8; 6 * 6];
        let mut out = Vec::new();
        resize(
            &mut Resizer::new(),
            &src,
            (6, 6),
            &mut out,
            (3, 2),
            Channels::Gray,
            ResizeFilter::Bilinear,
        )
        .unwrap();
        assert_eq!(out, vec![9u8; 6]);
    }

    #[test]
    fn short_source_is_an_error() {
        let mut out = Vec::new();
        let err = resize(
            &mut Resizer::new(),
            &[0u8; 5],
            (2, 2),
            &mut out,
            (1, 1),
            Channels::Rgb,
            ResizeFilter::Bilinear,
        );
        assert!(err.is_err());
    }
}
