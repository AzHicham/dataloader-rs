//! Image transforms against Pillow, numpy and torchvision (golden files made by
//! `tests/fixtures/image/make_fixtures.py`), against hostile input, and through a loader with
//! `ImageCollator` (feature `image`).
#![cfg(feature = "image")]

use std::path::PathBuf;

use bytes::Bytes;
use dataloader_rs::collator::{BatchLayout, ImageCollator};
use dataloader_rs::error::Error;
use dataloader_rs::image::{
    CenterCrop, Chw, Crop, Decode, DecodeLimits, Fill, Filter, HFlip, Image, ImageError, Normalize,
    Pad, PadMode, PadTo, Resize, Rotate90, ToChw, ToFloat, ToGray, ToRgb, ToRgba, VFlip,
};
use dataloader_rs::{DataLoader, Dataset, Transform};

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

// ── Geometry: the same pixels as torchvision, numpy and prime_dx ─────────────

fn rgb() -> Image<u8> {
    decode("rgb.png")
}

fn assert_exact(image: &Image<u8>, shape: [usize; 3], expected: &str) {
    assert_eq!(image.shape(), shape, "{expected}");
    assert!(image.data() == fixture(expected).as_slice(), "{expected}");
}

#[test]
fn crops_flips_and_rotations_are_exact() {
    assert_exact(
        &Crop::new(3, 5, 20, 30).apply(0, rgb()).unwrap(),
        [20, 30, 3],
        "crop_3_5_20_30.u8",
    );
    for (h, w) in [(20, 30), (40, 45), (31, 20)] {
        let out = CenterCrop::new(h, w).apply(0, rgb()).unwrap();
        assert_exact(&out, [h, w, 3], &format!("center_crop_{h}_{w}.u8"));
    }
    assert_exact(&HFlip.apply(0, rgb()).unwrap(), [H, W, 3], "hflip.u8");
    assert_exact(&VFlip.apply(0, rgb()).unwrap(), [H, W, 3], "vflip.u8");
    for k in 1..=3 {
        let shape = if k == 2 { [H, W, 3] } else { [W, H, 3] };
        assert_exact(
            &Rotate90::new(k).apply(0, rgb()).unwrap(),
            shape,
            &format!("rot90_{k}.u8"),
        );
    }
}

#[test]
fn padding_modes_are_exact() {
    let fill = PadMode::Constant(Fill::from([10.0, 20.0, 30.0]));
    let shape = [H + 3, W + 7, 3];
    assert_exact(
        &Pad::new(3, 1, 4, 2, fill).apply(0, rgb()).unwrap(),
        shape,
        "pad_constant.u8",
    );
    for (name, mode) in [
        ("edge", PadMode::Edge),
        ("reflect", PadMode::Reflect),
        ("symmetric", PadMode::Symmetric),
    ] {
        let out = Pad::new(3, 1, 4, 2, mode.clone()).apply(0, rgb()).unwrap();
        assert_exact(&out, shape, &format!("pad_{name}.u8"));
        let wide = Pad::new(40, 30, 2, 50, mode).apply(0, rgb()).unwrap();
        assert_exact(&wide, [H + 80, W + 42, 3], &format!("pad_wide_{name}.u8"));
    }
}

/// prime_dx's `ConstantFill2d`: pad right and bottom up to the tile size.
#[test]
fn pad_to_is_constant_fill_2d() {
    assert_exact(
        &PadTo::new(64, 64, 0.0).apply(0, rgb()).unwrap(),
        [64, 64, 3],
        "pad_to_64.u8",
    );
    let out = PadTo::new(37, 40, 255.0).apply(0, rgb()).unwrap();
    assert_exact(&out, [40, 37, 3], "pad_to_37_40.u8");
    assert_eq!(PadTo::new(20, 20, 0.0).apply(0, rgb()).unwrap(), rgb());
}

/// fast_image_resize against Pillow (antialiased) and torch (not): the same filters with
/// different fixed-point rounding; the cubic and Lanczos kernels overshoot, so their largest
/// differences are at hard edges. Measured: nearest 0; bilinear max 1, mean 0.18–0.20;
/// bicubic max 3–4, mean 0.21–0.23; Lanczos max 6–7, mean 0.21–0.29. (torch's bicubic uses
/// a = -0.75, Pillow's and ours -0.5: not compared.)
#[test]
fn resize_is_close_to_pillow_and_torch() {
    let down = || Resize::new(13, 17);
    let up = || Resize::new(41, 53);
    let cases = [
        ("down_nearest", down().filter(Filter::Nearest), 0, 0.0),
        ("down_bilinear", down(), 1, 0.25),
        ("down_bicubic", down().filter(Filter::Bicubic), 4, 0.3),
        ("down_lanczos3", down().filter(Filter::Lanczos3), 8, 0.35),
        ("up_nearest", up().filter(Filter::Nearest), 0, 0.0),
        ("up_bilinear", up(), 1, 0.25),
        ("up_bicubic", up().filter(Filter::Bicubic), 4, 0.3),
        ("up_lanczos3", up().filter(Filter::Lanczos3), 8, 0.35),
        ("down_bilinear_no_aa", down().antialias(false), 1, 0.25),
    ];
    for (name, resize, max_diff, mean_diff) in cases {
        let out = resize.apply(0, rgb()).unwrap();
        assert_eq!(out.shape()[2], 3);
        let (max, mean) = diff(out.data(), &fixture(&format!("resize_{name}.u8")));
        assert!(
            max <= max_diff && mean <= mean_diff,
            "{name}: max {max}, mean {mean:.3}"
        );
    }
}

// ── Floats: torchvision's ToTensor and Normalize, bit for bit ────────────────

fn f32_fixture(name: &str) -> Vec<f32> {
    let bytes = fixture(name);
    let (floats, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "{name}");
    floats.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// Equal bit for bit (so `-0.0 != 0.0` and NaNs compare by payload).
fn assert_bits(actual: &[f32], expected: &[f32], case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}");
    let differ = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differ, 0, "{case}: {differ} values differ");
}

#[test]
fn to_float_and_normalize_are_torchvision_bit_for_bit() {
    let to_tensor = ToChw.then(ToFloat::default());
    let chw: Image<f32, Chw> = to_tensor.apply(0, rgb()).unwrap();
    assert_eq!(chw.shape(), [3, H, W]);
    assert_bits(chw.data(), &f32_fixture("to_tensor.f32"), "ToTensor");
    let normalized = Normalize::imagenet().apply(0, chw).unwrap();
    assert_bits(
        normalized.data(),
        &f32_fixture("to_tensor_normalize.f32"),
        "Normalize",
    );

    // The other order: float and normalize in HWC, then CHW.
    let float = ToFloat::default().apply(0, rgb()).unwrap();
    let normalized = Normalize::imagenet().apply(0, float).unwrap();
    let hwc = ToChw.apply(0, normalized).unwrap();
    assert_bits(
        hwc.data(),
        &f32_fixture("to_tensor_normalize.f32"),
        "HWC first",
    );
}

/// msintuit-ec's feature pass: prime_dx's `ConstantFill2d(48)` + `ToTensor` + `Normalize` on
/// PIL images converted to RGB, against the same chain in Rust, from the PNG bytes.
#[test]
fn the_msintuit_chain_is_prime_dx_bit_for_bit() {
    let chain = Decode::new()
        .then(ToRgb)
        .then(PadTo::new(48, 48, 0.0))
        .then(ToFloat::default())
        .then(Normalize::imagenet())
        .then(ToChw);
    for name in ["tile_full", "rgb", "rgba", "gray", "palette"] {
        let png = Bytes::from(fixture(&format!("{name}.png")));
        let tensor = chain.apply(0, png).unwrap();
        assert_eq!(tensor.shape(), [3, 48, 48], "{name}");
        assert_bits(
            tensor.data(),
            &f32_fixture(&format!("msintuit_{name}.f32")),
            name,
        );
    }
}

// ── ImageCollator through a loader ───────────────────────────────────────────

/// The PNG fixtures as a dataset of `Bytes`.
struct Pngs(Vec<Bytes>);

impl Dataset for Pngs {
    type Item = Bytes;

    fn get(&self, index: usize) -> dataloader_rs::Result<Bytes> {
        Ok(self.0[index].clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

#[test]
fn the_loader_transforms_in_parallel_and_collates_one_buffer() {
    let names = ["tile_full", "rgb", "rgba", "gray", "palette"];
    let pngs = Pngs(
        names
            .iter()
            .map(|n| Bytes::from(fixture(&format!("{n}.png"))))
            .collect(),
    );
    let chain = Decode::new()
        .then(ToRgb)
        .then(PadTo::new(48, 48, 0.0))
        .then(ToFloat::default())
        .then(Normalize::imagenet());
    for (workers, buffers) in [(0, 0), (2, 0), (2, 2)] {
        let mut loader = DataLoader::builder(Pngs(pngs.0.clone()))
            .batch_size(2)
            .num_workers(workers)
            .item_concurrency(2)
            .transform(chain.clone())
            .collator(ImageCollator::default().reuse_buffers(buffers))
            .build();
        for _epoch in 0..2 {
            let batches: Vec<_> = loader.iter().map(Result::unwrap).collect();
            let shapes: Vec<_> = batches.iter().map(|b| b.shape()).collect();
            assert_eq!(shapes, [[2, 3, 48, 48], [2, 3, 48, 48], [1, 3, 48, 48]]);
            let values: Vec<f32> = batches.iter().flat_map(|b| b.data().to_vec()).collect();
            let expected: Vec<f32> = names
                .iter()
                .flat_map(|n| f32_fixture(&format!("msintuit_{n}.f32")))
                .collect();
            assert_bits(
                &values,
                &expected,
                &format!("workers {workers}, buffers {buffers}"),
            );
        }
    }
}

#[test]
fn a_bad_image_fails_its_batch_only() {
    let mut items: Vec<Bytes> = (0..4).map(|_| Bytes::from(fixture("rgb.png"))).collect();
    items[2] = Bytes::from_static(b"not an image");
    let mut loader = DataLoader::builder(Pngs(items))
        .batch_size(2)
        .num_workers(2)
        .transform(Decode::new())
        .collator(ImageCollator::new(BatchLayout::Nhwc))
        .build();
    let results: Vec<_> = loader.iter().collect();
    assert_eq!(results[0].as_ref().unwrap().shape(), [2, H, W, 3]);
    let err = results[1].as_ref().unwrap_err();
    assert!(matches!(image_error(err), ImageError::Decode(_)), "{err}");
}
