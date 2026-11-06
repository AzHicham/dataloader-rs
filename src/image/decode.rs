//! Decoding encoded bytes into 8-bit pixels of the pipeline's channel count, with the `image`
//! crate's pure-Rust decoders and bounded memory.

use std::io::Cursor;

use ::image::{ColorType, ImageDecoder, ImageFormat, ImageReader, Limits};

use super::{Channels, ImageError, ImageResult};

/// Bounds on the images a pipeline accepts, checked against the encoded header before any pixel
/// is decoded, and against every target size of the pipeline when it is built.
///
/// The defaults accept 16384 × 16384 pixels and 256 MiB per working image; an image above them
/// is an error for its batch, never an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLimits {
    /// Widest image, in pixels.
    pub max_width: u32,
    /// Tallest image, in pixels.
    pub max_height: u32,
    /// Bytes of one decoded image, of what the decoder allocates for it, and of any working image
    /// of the pipeline (`width × height × channels`).
    pub max_bytes: u64,
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_width: 16_384,
            max_height: 16_384,
            max_bytes: 256 << 20,
        }
    }
}

impl ImageLimits {
    /// Whether a working image of `width × height × channels` bytes is within the limits.
    pub(crate) fn check(&self, width: usize, height: usize, channels: usize) -> ImageResult<()> {
        let bytes = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(channels))
            .and_then(|n| u64::try_from(n).ok());
        let within = u32::try_from(width).is_ok_and(|w| w <= self.max_width)
            && u32::try_from(height).is_ok_and(|h| h <= self.max_height)
            && bytes.is_some_and(|b| b <= self.max_bytes);
        if within {
            Ok(())
        } else {
            Err(ImageError::Limits(format!(
                "image of {width}x{height}x{channels} is above the limits of {}x{} and {} bytes",
                self.max_width, self.max_height, self.max_bytes
            )))
        }
    }
}

/// Decode `bytes` (PNG, JPEG or WebP, detected from their content) into `out` as
/// `width × height` pixels of `channels` interleaved 8-bit values. `raw` is scratch space.
///
/// Returns `(width, height)`.
pub(crate) fn decode(
    bytes: &[u8],
    channels: Channels,
    limits: ImageLimits,
    raw: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> ImageResult<(usize, usize)> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    match reader.format() {
        None => {
            return Err(ImageError::Decode(
                "unrecognised format (expected PNG, JPEG or WebP)".into(),
            ));
        }
        Some(ImageFormat::Jpeg) if !jpeg_is_complete(bytes) => {
            return Err(ImageError::Decode(
                "truncated JPEG: no end-of-image marker after the last scan".into(),
            ));
        }
        Some(_) => {}
    }
    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(limits.max_width);
    decoder_limits.max_image_height = Some(limits.max_height);
    decoder_limits.max_alloc = Some(limits.max_bytes);
    reader.limits(decoder_limits);
    let decoder = reader.into_decoder()?;

    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 {
        return Err(ImageError::Decode("empty image".into()));
    }
    if width > limits.max_width || height > limits.max_height {
        return Err(ImageError::Limits(format!(
            "image of {width}x{height} is above the limit of {}x{}",
            limits.max_width, limits.max_height
        )));
    }
    // The converted image has the pipeline's channel count, which may exceed the decoded one.
    limits.check(width as usize, height as usize, channels.count())?;
    let color = decoder.color_type();
    let total = decoder.total_bytes();
    if total > limits.max_bytes {
        return Err(ImageError::Limits(format!(
            "decoded image of {total} bytes is above the limit of {}",
            limits.max_bytes
        )));
    }
    let too_large = || ImageError::Limits("decoded image too large".into());
    let total = usize::try_from(total).map_err(|_| too_large())?;
    let (width, height) = (width as usize, height as usize);
    let pixels = width.checked_mul(height).ok_or_else(too_large)?;

    let direct = matches!(
        (color, channels),
        (ColorType::Rgb8, Channels::Rgb) | (ColorType::L8, Channels::Gray)
    );
    let target = if direct { &mut *out } else { &mut *raw };
    target.clear();
    target.try_reserve_exact(total)?;
    target.resize(total, 0);
    decoder.read_image(target)?;
    if !direct {
        convert(raw, color, channels, pixels, out)?;
    }
    Ok((width, height))
}

/// Whether a JPEG stream ends: its last end-of-image marker (`FF D9`) comes after its last
/// start-of-scan (`FF DA`). zune-jpeg decodes a truncated file into a partly gray image without an
/// error; Pillow raises on it, and so does the pipeline. Neither marker can appear inside
/// entropy-coded data (a `FF` there is followed by `00`).
fn jpeg_is_complete(bytes: &[u8]) -> bool {
    let Some(eoi) = bytes.windows(2).rposition(|w| w == [0xFF, 0xD9]) else {
        return false;
    };
    !bytes[eoi..].windows(2).any(|w| w == [0xFF, 0xDA])
}

/// Pillow's `L` conversion (ITU-R 601-2 luma, `L24` in `Convert.c`).
#[inline]
fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 19595 + u32::from(g) * 38470 + u32::from(b) * 7471 + 0x8000) >> 16) as u8
}

/// The 8 most significant bits of a native-endian 16-bit sample, as Pillow reads 16-bit RGB(A)
/// PNGs.
#[inline]
fn high_byte(pair: &[u8]) -> u8 {
    (u16::from_ne_bytes([pair[0], pair[1]]) >> 8) as u8
}

/// Convert `pixels` decoded pixels of `color` to `channels`: alpha is dropped (as Pillow's
/// `convert("RGB")` does), gray is replicated, RGB to gray uses Pillow's luma.
fn convert(
    raw: &[u8],
    color: ColorType,
    channels: Channels,
    pixels: usize,
    out: &mut Vec<u8>,
) -> ImageResult<()> {
    let n = channels.count();
    out.clear();
    let len = pixels
        .checked_mul(n)
        .ok_or_else(|| ImageError::Limits("decoded image too large".into()))?;
    out.try_reserve_exact(len)?;
    let step = color.bytes_per_pixel() as usize;
    let sixteen = matches!(
        color,
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    );
    let gray_source = matches!(
        color,
        ColorType::L8 | ColorType::La8 | ColorType::L16 | ColorType::La16
    );
    if !matches!(
        color,
        ColorType::L8
            | ColorType::La8
            | ColorType::Rgb8
            | ColorType::Rgba8
            | ColorType::L16
            | ColorType::La16
            | ColorType::Rgb16
            | ColorType::Rgba16
    ) {
        return Err(ImageError::Decode(format!(
            "unsupported pixel type {color:?}"
        )));
    }
    let sample = |px: &[u8], i: usize| -> u8 {
        if sixteen {
            high_byte(&px[2 * i..2 * i + 2])
        } else {
            px[i]
        }
    };
    for px in raw.chunks_exact(step).take(pixels) {
        match (gray_source, channels) {
            (true, Channels::Gray) => out.push(sample(px, 0)),
            (true, Channels::Rgb) => {
                let v = sample(px, 0);
                out.extend_from_slice(&[v, v, v]);
            }
            (false, Channels::Gray) => out.push(luma(sample(px, 0), sample(px, 1), sample(px, 2))),
            (false, Channels::Rgb) => {
                out.extend_from_slice(&[sample(px, 0), sample(px, 1), sample(px, 2)]);
            }
        }
    }
    if out.len() != len {
        return Err(ImageError::Decode(
            "decoded image is shorter than its size".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ::image::{
        ExtendedColorType, ImageEncoder,
        codecs::{png::PngEncoder, webp::WebPEncoder},
    };

    use super::{ImageLimits, decode, jpeg_is_complete, luma};
    use crate::image::{Channels, ImageError, ImageResult};

    const LIMITS: ImageLimits = ImageLimits {
        max_width: 4096,
        max_height: 4096,
        max_bytes: 1 << 26,
    };

    pub(crate) fn png(pixels: &[u8], w: u32, h: u32, color: ExtendedColorType) -> Vec<u8> {
        let mut out = Vec::new();
        PngEncoder::new(Cursor::new(&mut out))
            .write_image(pixels, w, h, color)
            .unwrap();
        out
    }

    fn run(bytes: &[u8], channels: Channels) -> ImageResult<(usize, usize, Vec<u8>)> {
        let (mut raw, mut out) = (Vec::new(), Vec::new());
        let (w, h) = decode(bytes, channels, LIMITS, &mut raw, &mut out)?;
        Ok((w, h, out))
    }

    #[test]
    fn decodes_png_and_webp() {
        let rgb: Vec<u8> = (0..5 * 3 * 3).map(|i| (i * 17) as u8).collect();
        let png = png(&rgb, 5, 3, ExtendedColorType::Rgb8);
        assert_eq!(run(&png, Channels::Rgb).unwrap(), (5, 3, rgb.clone()));

        let mut webp = Vec::new();
        WebPEncoder::new_lossless(Cursor::new(&mut webp))
            .write_image(&rgb, 5, 3, ExtendedColorType::Rgb8)
            .unwrap();
        assert_eq!(run(&webp, Channels::Rgb).unwrap(), (5, 3, rgb));
    }

    #[test]
    fn converts_colors() {
        let rgba = [10u8, 20, 30, 0, 200, 100, 50, 255];
        let png_rgba = png(&rgba, 2, 1, ExtendedColorType::Rgba8);
        assert_eq!(
            run(&png_rgba, Channels::Rgb).unwrap().2,
            [10, 20, 30, 200, 100, 50]
        );
        assert_eq!(
            run(&png_rgba, Channels::Gray).unwrap().2,
            [luma(10, 20, 30), luma(200, 100, 50)]
        );
        let png_gray = png(&[7, 9], 2, 1, ExtendedColorType::L8);
        assert_eq!(run(&png_gray, Channels::Rgb).unwrap().2, [7, 7, 7, 9, 9, 9]);
        assert_eq!(run(&png_gray, Channels::Gray).unwrap().2, [7, 9]);
        // 16-bit: the high byte.
        let wide: Vec<u8> = [0x1234u16, 0xabcd, 0xff00]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let png16 = png(&wide, 1, 1, ExtendedColorType::Rgb16);
        assert_eq!(run(&png16, Channels::Rgb).unwrap().2, [0x12, 0xab, 0xff]);
    }

    #[test]
    fn luma_matches_pillow() {
        // Values from Pillow 13: Image.new("RGB", (1, 1), c).convert("L").getpixel((0, 0)).
        assert_eq!(luma(255, 255, 255), 255);
        assert_eq!(luma(0, 0, 0), 0);
        assert_eq!(luma(255, 0, 0), 76);
        assert_eq!(luma(0, 255, 0), 150);
        assert_eq!(luma(0, 0, 255), 29);
        assert_eq!(luma(12, 200, 99), 132);
    }

    #[test]
    fn rejects_garbage_and_truncation() {
        assert!(run(b"", Channels::Rgb).is_err());
        assert!(run(b"not an image at all", Channels::Rgb).is_err());
        // GIF is recognised but not compiled in.
        assert!(run(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", Channels::Rgb).is_err());
        let rgb: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 7) as u8).collect();
        let png = png(&rgb, 16, 16, ExtendedColorType::Rgb8);
        for len in 0..png.len() - 1 {
            let _ = run(&png[..len], Channels::Rgb);
        }
        assert!(run(&png[..png.len() / 2], Channels::Rgb).is_err());
    }

    #[test]
    fn jpeg_completeness() {
        let eoi = [0xFF, 0xD9];
        let sos = [0xFF, 0xDA];
        assert!(jpeg_is_complete(&[&sos[..], &[1, 2], &eoi].concat()));
        // Trailing bytes after the end are fine, as in Pillow.
        assert!(jpeg_is_complete(&[&sos[..], &eoi, &[0, 0]].concat()));
        assert!(!jpeg_is_complete(&[&sos[..], &[1, 2]].concat()));
        // An end marker of an embedded thumbnail, then a scan without end.
        assert!(!jpeg_is_complete(&[&eoi[..], &sos, &[1]].concat()));
        assert!(!jpeg_is_complete(&[]));
    }

    #[test]
    fn rejects_oversized_headers() {
        // A valid PNG whose header claims 100000 x 100000 pixels.
        let mut png = png(&[1, 2, 3], 1, 1, ExtendedColorType::Rgb8);
        set_png_size(&mut png, 100_000, 100_000);
        let err = run(&png, Channels::Rgb).unwrap_err();
        assert!(matches!(err, ImageError::Limits(_)), "{err}");
        // Within the sides, above the byte budget.
        let small = ImageLimits {
            max_width: 4096,
            max_height: 4096,
            max_bytes: 1000,
        };
        set_png_size(&mut png, 4000, 4000);
        let (mut raw, mut out) = (Vec::new(), Vec::new());
        assert!(matches!(
            decode(&png, Channels::Rgb, small, &mut raw, &mut out),
            Err(ImageError::Limits(_))
        ));
        assert!(out.capacity() < 1 << 20);
    }

    /// Rewrite the IHDR width and height of `png`, and its CRC.
    pub(crate) fn set_png_size(png: &mut [u8], w: u32, h: u32) {
        // 8-byte signature, 4-byte length, "IHDR", then width and height.
        png[16..20].copy_from_slice(&w.to_be_bytes());
        png[20..24].copy_from_slice(&h.to_be_bytes());
        let crc = crc32(&png[12..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in data {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }
}
