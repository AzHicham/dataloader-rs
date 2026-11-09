//! Image transforms against Pillow, numpy and torchvision (golden files made by
//! `tests/fixtures/image/make_fixtures.py`), and against hostile input (feature `image`).
#![cfg(feature = "image")]

use std::path::PathBuf;

use bytes::Bytes;
use dataloader_rs::Transform;
use dataloader_rs::error::Error;
use dataloader_rs::image::{Decode, DecodeLimits, Image, ImageError, ToGray, ToRgb, ToRgba};

/// Width and height of the fixtures.
const W: usize = 37;
const H: usize = 29;

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image");
    std::fs::read(path.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn decode(name: &str) -> Image<u8> {
    Decode::new()
        .decode(&fixture(name))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The `ImageError` inside a crate error.
fn image_error(error: &Error) -> &ImageError {
    error
        .downcast_ref::<ImageError>()
        .unwrap_or_else(|| panic!("not an ImageError: {error}"))
}

/// Largest absolute difference and mean absolute difference.
fn diff(a: &[u8], b: &[u8]) -> (u8, f64) {
    assert_eq!(a.len(), b.len());
    let max = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    (max, sum as f64 / a.len() as f64)
}

// ── Decoding: the same pixels as Pillow ──────────────────────────────────────

#[test]
fn lossless_formats_decode_to_the_exact_pixels() {
    let expected = fixture("rgb.u8");
    for name in [
        "rgb.png",
        "rgb.bmp",
        "rgb.tiff",
        "rgb_lossless.webp",
        "rgb16.png",
    ] {
        let image = decode(name);
        assert_eq!(image.shape(), [H, W, 3], "{name}");
        assert!(image.data() == expected.as_slice(), "{name}");
    }
}

/// `image` converted to `channels`: 1 gray, 3 RGB, 4 RGBA.
fn to_channels(image: Image<u8>, channels: usize) -> Image<u8> {
    match channels {
        1 => ToGray.apply(0, image),
        3 => ToRgb.apply(0, image),
        4 => ToRgba.apply(0, image),
        _ => unreachable!(),
    }
    .unwrap()
}

#[test]
fn channel_variants_convert_like_pillow() {
    for (input, channels, expected) in [
        ("rgba.png", 4, "rgba.u8"),
        ("rgb.png", 1, "rgb_to_gray.u8"),
        ("rgba.png", 1, "rgba_to_gray.u8"),
        ("gray.png", 3, "gray_to_rgb.u8"),
        ("la.png", 4, "la_to_rgba.u8"),
        ("palette.png", 3, "palette_to_rgb.u8"),
        ("palette_trns.png", 3, "palette_to_rgb.u8"),
    ] {
        let image = to_channels(decode(input), channels);
        assert_eq!(image.shape(), [H, W, channels], "{input} → {expected}");
        assert!(
            image.data() == fixture(expected).as_slice(),
            "{input} → {expected}"
        );
    }
}

#[test]
fn sixteen_bit_images_keep_their_high_byte() {
    let gray = decode("gray16.png");
    assert_eq!(gray.shape(), [H, W, 1]);
    let green: Vec<u8> = fixture("rgb.u8").chunks(3).map(|p| p[1]).collect();
    assert_eq!(gray.data(), green.as_slice());
}

/// zune-jpeg and image-webp against libjpeg-turbo and libwebp (Pillow): the IDCT, chroma
/// upsampling and rounding differ slightly.
#[test]
fn lossy_formats_decode_within_tolerance() {
    for (name, expected, max_diff, mean_diff) in [
        // max 3, mean 0.36 with Pillow 13 (libjpeg-turbo).
        ("rgb_q90.jpg", "rgb_q90_jpg.u8", 4, 0.5),
        // Equal today (max 0); a newer libwebp may round differently.
        ("rgb_q90.webp", "rgb_q90_webp.u8", 1, 0.01),
    ] {
        let image = decode(name);
        assert_eq!(image.shape(), [H, W, 3], "{name}");
        let (max, mean) = diff(image.data(), &fixture(expected));
        assert!(
            max <= max_diff && mean <= mean_diff,
            "{name}: max {max}, mean {mean:.3}"
        );
    }
}

#[test]
fn decode_is_a_transform_of_bytes() {
    let png = fixture("rgb.png");
    let chain = Decode::new().then(ToGray);
    let image = chain.apply(0, Bytes::from(png)).unwrap();
    assert!(image.data() == fixture("rgb_to_gray.u8").as_slice());
}

// ── Hostile input: clean errors, no panic, no unbounded allocation ───────────

/// CRC-32 of PNG chunks.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// `rgb.png` claiming to be `width × height`, with a valid header checksum.
fn png_claiming(width: u32, height: u32) -> Vec<u8> {
    let mut png = fixture("rgb.png");
    png[16..20].copy_from_slice(&width.to_be_bytes());
    png[20..24].copy_from_slice(&height.to_be_bytes());
    let crc = crc32(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
    png
}

/// `rgb.bmp` claiming to be `width × height`.
fn bmp_claiming(width: i32, height: i32) -> Vec<u8> {
    let mut bmp = fixture("rgb.bmp");
    bmp[18..22].copy_from_slice(&width.to_le_bytes());
    bmp[22..26].copy_from_slice(&height.to_le_bytes());
    bmp
}

/// `rgb_q90.jpg` claiming to be `width × height` in its start-of-frame.
fn jpeg_claiming(width: u16, height: u16) -> Vec<u8> {
    let mut jpeg = fixture("rgb_q90.jpg");
    let sof = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("baseline JPEG");
    jpeg[sof + 5..sof + 7].copy_from_slice(&height.to_be_bytes());
    jpeg[sof + 7..sof + 9].copy_from_slice(&width.to_be_bytes());
    jpeg
}

fn assert_limits(bytes: &[u8], decode: Decode, case: &str) {
    let err = decode.decode(bytes).unwrap_err();
    assert!(
        matches!(image_error(&err), ImageError::Limits(_)),
        "{case}: {err}"
    );
}

#[test]
fn huge_declared_sizes_are_refused_from_the_header() {
    let decode = Decode::new();
    assert_limits(&png_claiming(1 << 30, 1 << 30), decode, "png 2^30");
    assert_limits(&png_claiming(20_000, 10), decode, "png wide");
    // Within 16384 × 16384, but 16000 × 16000 × 3 bytes is above 256 MiB.
    assert_limits(&png_claiming(16_000, 16_000), decode, "png alloc");
    assert_limits(&bmp_claiming(20_000, 4), decode, "bmp wide");
    // The BMP decoder refuses more than 65535 by itself.
    assert!(decode.decode(&bmp_claiming(1 << 30, 4)).is_err());
    assert_limits(&bmp_claiming(16_000, -16_000), decode, "bmp alloc");
    assert_limits(&jpeg_claiming(60_000, 60_000), decode, "jpeg");
}

#[test]
fn limits_are_configurable() {
    let small = Decode::with_limits(DecodeLimits {
        max_width: 36,
        ..DecodeLimits::default()
    });
    assert_limits(&fixture("rgb.png"), small, "width 37 > 36");
    let tiny = Decode::with_limits(DecodeLimits {
        max_alloc: 1000,
        ..DecodeLimits::default()
    });
    for name in [
        "rgb.png",
        "rgb_q90.jpg",
        "rgb_lossless.webp",
        "rgb.tiff",
        "rgb.bmp",
    ] {
        assert_limits(&fixture(name), tiny, name);
    }
}

const ENCODED: [&str; 11] = [
    "rgb.png",
    "rgba.png",
    "gray.png",
    "la.png",
    "palette_trns.png",
    "rgb16.png",
    "rgb.bmp",
    "rgb.tiff",
    "rgb_lossless.webp",
    "rgb_q90.webp",
    "rgb_q90.jpg",
];

#[test]
fn truncated_files_are_errors() {
    for name in ENCODED {
        let bytes = fixture(name);
        for len in [
            0,
            8,
            16,
            40,
            bytes.len() / 2,
            bytes.len() * 3 / 4,
            bytes.len() - 16,
        ] {
            let Err(err) = Decode::new().decode(&bytes[..len]) else {
                panic!("{name}[..{len}] of {} decoded", bytes.len());
            };
            assert!(
                matches!(image_error(&err), ImageError::Decode(_)),
                "{name}[..{len}]: {err}"
            );
        }
    }
}

#[test]
fn garbage_is_an_error() {
    let mut rng = fastrand::Rng::with_seed(7);
    for len in [1, 2, 7, 64, 4096] {
        let bytes: Vec<u8> = (0..len).map(|_| rng.u8(..)).collect();
        assert!(Decode::new().decode(&bytes).is_err(), "{len} random bytes");
    }
    // A valid signature followed by noise.
    for magic in [
        &b"\x89PNG\r\n\x1a\n"[..],
        b"\xFF\xD8\xFF",
        b"RIFF\0\0\0\0WEBP",
        b"II*\0",
        b"BM",
    ] {
        let mut bytes = magic.to_vec();
        bytes.extend((0..512).map(|_| rng.u8(..)));
        assert!(Decode::new().decode(&bytes).is_err(), "{magic:?} + noise");
    }
}

/// Random byte flips in every format: each decode returns, `Ok` or `Err`, without panicking,
/// and a decoded image always has the size of its shape.
#[test]
fn corrupted_files_never_panic() {
    let mut rng = fastrand::Rng::with_seed(42);
    for name in ENCODED {
        let original = fixture(name);
        for _ in 0..300 {
            let mut bytes = original.clone();
            for _ in 0..rng.usize(1..8) {
                let at = rng.usize(..bytes.len());
                bytes[at] = rng.u8(..);
            }
            if let Ok(image) = Decode::new().decode(&bytes) {
                let [a, b, c] = image.shape();
                assert_eq!(image.data().len(), a * b * c, "{name}");
            }
        }
    }
}
