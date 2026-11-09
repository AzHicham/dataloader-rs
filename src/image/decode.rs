//! [`Decode`]: encoded bytes (PNG, JPEG, WebP, TIFF, BMP) into an 8-bit [`Image`], with the
//! `image` crate's pure-Rust decoders and bounded memory.

use std::io::Cursor;

use ::image::error::ImageError as CodecError;
use ::image::{ColorType, ImageDecoder, ImageFormat, ImageReader, Limits};
use bytes::Bytes;

use super::buffer::{invalid, too_large, with_capacity};
use super::{Image, ImageError};
use crate::{error::Result, transform::Transform};

/// The largest image [`Decode`] accepts, checked against the header before any pixel is
/// decoded. Above them an image is an error for its batch, never an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Widest image, in pixels. Default: 16384.
    pub max_width: u32,
    /// Tallest image, in pixels. Default: 16384.
    pub max_height: u32,
    /// Bytes the decoder may allocate, the decoded image included. Default: 256 MiB.
    pub max_alloc: u64,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_width: 16_384,
            max_height: 16_384,
            max_alloc: 256 << 20,
        }
    }
}

/// Decode PNG, JPEG, WebP (lossless and lossy), TIFF or BMP bytes, the format told by their
/// content, into an 8-bit [`Image`] with the image's own channels: 1 (gray), 2 (gray + alpha),
/// 3 (RGB) or 4 (RGBA). Palette images are expanded to RGB or RGBA; 16-bit images keep their
/// 8 high bits, as Pillow does for RGB.
///
/// The input is [`Bytes`], the items of the built-in async datasets; a `Vec<u8>` becomes one
/// without a copy (`Bytes::from(vec)`). [`decode`](Self::decode) takes any `&[u8]`.
///
/// Images above [`DecodeLimits`] (default 16384 × 16384 and 256 MiB) are refused from their
/// header, and a truncated JPEG is an error, as in Pillow (the decoder would fill the missing
/// part with gray).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Decode {
    limits: DecodeLimits,
}

impl Decode {
    /// A decoder with the default [`DecodeLimits`].
    pub fn new() -> Self {
        Self::default()
    }

    /// A decoder with `limits`.
    pub fn with_limits(limits: DecodeLimits) -> Self {
        Self { limits }
    }

    /// Its limits.
    pub fn limits(&self) -> DecodeLimits {
        self.limits
    }

    /// Decode `bytes`.
    pub fn decode(&self, bytes: &[u8]) -> Result<Image<u8>> {
        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| decode_error(e.to_string()))?;
        match reader.format() {
            None => {
                return Err(decode_error(
                    "unknown format (expected PNG, JPEG, WebP, TIFF or BMP)".into(),
                ));
            }
            Some(ImageFormat::Jpeg) if !jpeg_is_complete(bytes) => {
                return Err(decode_error(
                    "truncated JPEG: no end-of-image marker after the last scan".into(),
                ));
            }
            Some(_) => {}
        }
        let mut reader = reader;
        reader.limits(self.codec_limits());
        let decoder = reader.into_decoder().map_err(codec_error)?;

        let (width, height) = decoder.dimensions();
        self.check_size(width, height)?;
        let color = decoder.color_type();
        let channels = channels(color)?;
        let total = decoder.total_bytes();
        if total > self.limits.max_alloc {
            return Err(too_large(format!(
                "{width}x{height} {color:?} is {total} bytes, above the limit of {}",
                self.limits.max_alloc
            )));
        }
        let total = usize::try_from(total).map_err(|_| too_large(format!("{total} bytes")))?;
        let mut raw = with_capacity(total)?;
        raw.resize(total, 0);
        decoder.read_image(&mut raw).map_err(codec_error)?;
        let data = match color.bytes_per_pixel() as usize == channels {
            true => raw,
            // 16 bits per value, native endian: keep the high byte.
            false => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| (u16::from_ne_bytes(pair) >> 8) as u8)
                .collect(),
        };
        Image::new(width as usize, height as usize, channels, data)
    }

    /// The limits as the `image` crate takes them.
    fn codec_limits(&self) -> Limits {
        let mut limits = Limits::default();
        limits.max_image_width = Some(self.limits.max_width);
        limits.max_image_height = Some(self.limits.max_height);
        limits.max_alloc = Some(self.limits.max_alloc);
        limits
    }

    /// Refuse an empty image or one above the limits, from its header.
    fn check_size(&self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(decode_error(format!("empty image of {width}x{height}")));
        }
        let DecodeLimits {
            max_width,
            max_height,
            ..
        } = self.limits;
        if width > max_width || height > max_height {
            return Err(too_large(format!(
                "{width}x{height} is above the limit of {max_width}x{max_height}"
            )));
        }
        Ok(())
    }
}

/// One input type only, so that a chain starting with `Decode` infers its types.
impl Transform<Bytes> for Decode {
    type Output = Image<u8>;

    fn apply(&self, _index: usize, bytes: Bytes) -> Result<Image<u8>> {
        self.decode(&bytes)
    }
}

/// The channels of an image decoded as `color`; 32-bit float images are refused.
fn channels(color: ColorType) -> Result<usize> {
    match color {
        ColorType::L8 | ColorType::L16 => Ok(1),
        ColorType::La8 | ColorType::La16 => Ok(2),
        ColorType::Rgb8 | ColorType::Rgb16 => Ok(3),
        ColorType::Rgba8 | ColorType::Rgba16 => Ok(4),
        other => Err(invalid(format!("unsupported pixel type {other:?}"))),
    }
}

/// Whether a JPEG stream ends: its last end-of-image marker (`FF D9`) comes after its last
/// start-of-scan (`FF DA`). Neither marker can appear inside entropy-coded data, where a `FF` is
/// followed by `00`.
fn jpeg_is_complete(bytes: &[u8]) -> bool {
    let Some(end) = bytes.windows(2).rposition(|w| w == [0xFF, 0xD9]) else {
        return false;
    };
    !bytes[end..].windows(2).any(|w| w == [0xFF, 0xDA])
}

fn decode_error(message: String) -> crate::Error {
    ImageError::Decode(message).into()
}

/// A decoder's error: over a limit, or anything else that stopped it.
fn codec_error(error: CodecError) -> crate::Error {
    match error {
        CodecError::Limits(limits) => too_large(limits.to_string()),
        other => decode_error(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Decode, DecodeLimits, jpeg_is_complete};
    use crate::image::ImageError;

    #[test]
    fn unknown_bytes_are_an_error() {
        for bytes in [&b""[..], b"hello", &[0u8; 64]] {
            let err = Decode::new().decode(bytes).unwrap_err();
            let err = err.downcast_ref::<ImageError>().unwrap();
            assert!(matches!(err, ImageError::Decode(_)), "{err}");
        }
    }

    #[test]
    fn a_jpeg_is_complete_when_it_ends_after_its_last_scan() {
        assert!(jpeg_is_complete(&[
            0xFF, 0xD8, 0xFF, 0xDA, 1, 2, 0xFF, 0xD9
        ]));
        assert!(!jpeg_is_complete(&[0xFF, 0xD8, 0xFF, 0xDA, 1, 2]));
        assert!(!jpeg_is_complete(&[0xFF, 0xD9, 0xFF, 0xDA, 1, 2]));
    }

    #[test]
    fn default_limits() {
        let limits = Decode::new().limits();
        assert_eq!(limits, DecodeLimits::default());
        assert_eq!((limits.max_width, limits.max_height), (16_384, 16_384));
        assert_eq!(limits.max_alloc, 256 << 20);
    }
}
