//! Feature `image`: `ImageCollator` against Pillow and torchvision (golden fixtures from
//! `tests/fixtures/image/make_fixtures.py`), and through a loader with workers and pooled buffers.
#![cfg(feature = "image")]

use std::{collections::HashSet, fs, path::PathBuf, sync::Arc};

use dataloader_rs::{
    DataLoader, Dataset,
    collator::{CollateInto, HeapAllocator},
    error::Result,
    image::{
        Channels, DType, DynImageCollator, ImageBuffer, ImageCollator, ImageError, ImageLimits,
        ImageTransform, Layout, ResizeFilter,
    },
};
use half::f16;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Largest |Δ| against Pillow 13, in 8-bit levels, measured on these fixtures (see the image
/// guide, src/image/README.md).
const JPEG_TOLERANCE: u8 = 3;
/// `(filter, fixture name, max |Δ| downscaling 160x120 → 64x48, upscaling 64x48 → 96x72)`.
///
/// Nearest upscaling picks another source pixel than Pillow on a quarter of the values
/// (`fast_image_resize` rounds the source position differently), hence no bound; downscaling is
/// identical.
const RESIZE_TOLERANCE: [(ResizeFilter, &str, u8, u8); 6] = [
    (ResizeFilter::Nearest, "nearest", 0, u8::MAX),
    (ResizeFilter::Box, "box", 1, 0),
    (ResizeFilter::Bilinear, "bilinear", 1, 1),
    (ResizeFilter::Hamming, "hamming", 1, 1),
    (ResizeFilter::Bicubic, "bicubic", 2, 8),
    (ResizeFilter::Lanczos3, "lanczos3", 4, 14),
];

fn fixture(name: &str) -> Vec<u8> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests/fixtures/image", name]
        .iter()
        .collect();
    fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn f32s(name: &str) -> Vec<f32> {
    fixture(name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f16s(name: &str) -> Vec<f16> {
    fixture(name)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// Run `transform` on one encoded image.
fn run<T: dataloader_rs::image::Element>(transform: ImageTransform, bytes: &[u8]) -> Vec<T> {
    let collator = ImageCollator::<T>::new(transform).unwrap();
    let mut row = vec![T::default(); collator.sample_len()];
    collator.write_sample(bytes, &mut row).unwrap();
    row
}

fn max_diff_u8(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap()
}

fn max_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rgb_hwc(width: u32, height: u32) -> ImageTransform {
    ImageTransform::new(Channels::Rgb)
        .input_size(width, height)
        .layout(Layout::Nhwc)
}

#[test]
fn png_and_lossless_webp_decode_bit_identical_to_pillow() {
    let expected = fixture("rgb.u8");
    for name in ["rgb.png", "rgb_lossless.webp", "rgba.png"] {
        assert_eq!(
            run::<u8>(rgb_hwc(64, 48), &fixture(name)),
            expected,
            "{name}"
        );
    }
    let gray = ImageTransform::new(Channels::Gray).input_size(64, 48);
    assert_eq!(
        run::<u8>(gray, &fixture("rgb.png")),
        fixture("rgb_to_gray_pil.u8")
    );
}

#[test]
fn jpeg_decode_close_to_pillow() {
    let ours = run::<u8>(rgb_hwc(64, 48), &fixture("rgb_q90.jpg"));
    let diff = max_diff_u8(&ours, &fixture("rgb_q90_pil.u8"));
    eprintln!("jpeg q90 vs Pillow: max |delta| {diff}");
    assert!(diff <= JPEG_TOLERANCE, "jpeg max |delta| {diff}");
}

#[test]
fn resize_close_to_pillow() {
    let big = fixture("big.png");
    let small = fixture("rgb.png");
    for (filter, name, max_down, max_up) in RESIZE_TOLERANCE {
        let down = run::<u8>(
            ImageTransform::new(Channels::Rgb)
                .resize(64, 48, filter)
                .layout(Layout::Nhwc),
            &big,
        );
        let up = run::<u8>(
            ImageTransform::new(Channels::Rgb)
                .resize(96, 72, filter)
                .layout(Layout::Nhwc),
            &small,
        );
        let down = max_diff_u8(&down, &fixture(&format!("down_{name}.u8")));
        let up = max_diff_u8(&up, &fixture(&format!("up_{name}.u8")));
        eprintln!("resize {name} vs Pillow: max |delta| down {down}, up {up}");
        assert!(
            down <= max_down && up <= max_up,
            "{name}: down {down}, up {up}"
        );
    }
}

#[test]
fn to_float_and_normalize_match_torchvision() {
    let png = fixture("rgb.png");
    let rgb = || ImageTransform::new(Channels::Rgb).input_size(64, 48);

    let to_tensor = run::<f32>(rgb().to_float(), &png);
    let diff = max_diff_f32(&to_tensor, &f32s("rgb_to_tensor.f32"));
    eprintln!("ToTensor vs torchvision: max |delta| {diff:e}");
    assert!(diff <= 1e-6, "{diff}");

    let normalize = rgb().to_float().normalize(MEAN, STD);
    let ours = run::<f32>(normalize.clone(), &png);
    let diff = max_diff_f32(&ours, &f32s("rgb_normalize.f32"));
    eprintln!("ToTensor + Normalize vs torchvision: max |delta| {diff:e}");
    assert!(diff <= 1e-6, "{diff}");

    let half = run::<f16>(normalize, &png);
    let mismatches = half
        .iter()
        .zip(f16s("rgb_normalize.f16"))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("ToTensor + Normalize + half() vs torchvision: {mismatches} values differ");
    assert_eq!(mismatches, 0);
}

#[test]
fn resize_then_normalize_close_to_torchvision() {
    let t = ImageTransform::new(Channels::Rgb)
        .resize(64, 48, ResizeFilter::Bilinear)
        .to_float()
        .normalize(MEAN, STD);
    let ours = run::<f32>(t, &fixture("big.png"));
    let diff = max_diff_f32(&ours, &f32s("big_pipeline.f32"));
    eprintln!("resize + ToTensor + Normalize vs torchvision: max |delta| {diff:e}");
    // One 8-bit level of the resize, through the smallest std.
    assert!(diff <= 1.0 / 255.0 / 0.224 + 1e-6, "{diff}");
}

#[test]
fn hostile_inputs_are_errors() {
    let collator = ImageCollator::<f32>::new(
        ImageTransform::new(Channels::Rgb)
            .resize(8, 8, ResizeFilter::Bilinear)
            .limits(ImageLimits {
                max_width: 4096,
                max_height: 4096,
                max_bytes: 1 << 20,
            }),
    )
    .unwrap();
    let mut row = vec![0f32; collator.sample_len()];
    let png = fixture("rgb.png");
    let mut huge = png.clone();
    set_png_size(&mut huge, 1 << 30, 1 << 30);
    let mut over_budget = png.clone();
    set_png_size(&mut over_budget, 4000, 4000);
    let jpeg = fixture("rgb_q90.jpg");
    let webp = fixture("rgb_lossless.webp");
    for (what, bytes) in [
        ("empty", &[][..]),
        ("text", b"definitely not an image"),
        ("gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
        ("huge header", &huge),
        ("over the byte budget", &over_budget),
        ("truncated png", &png[..png.len() / 2]),
        ("truncated jpeg", &jpeg[..jpeg.len() / 2]),
        ("truncated webp", &webp[..webp.len() / 2]),
    ] {
        assert!(collator.write_sample(bytes, &mut row).is_err(), "{what}");
    }
    // Every prefix and a sweep of corrupted bytes: errors or images, never a panic.
    for source in [&png, &jpeg, &webp] {
        for len in (0..source.len()).step_by(7) {
            let _ = collator.write_sample(&source[..len], &mut row);
        }
        for i in (0..source.len()).step_by(11) {
            let mut bad = source.clone();
            bad[i] ^= 0x5a;
            let _ = collator.write_sample(&bad, &mut row);
        }
    }
    // A size other than input_size.
    let strict = ImageCollator::<u8>::new(rgb_hwc(32, 32)).unwrap();
    let err = strict
        .write_sample(&png, &mut vec![0; strict.sample_len()])
        .unwrap_err();
    assert!(err.to_string().contains("expected 32x32"), "{err}");
}

/// Rewrite the IHDR width and height of `png`, and its CRC.
fn set_png_size(png: &mut [u8], w: u32, h: u32) {
    png[16..20].copy_from_slice(&w.to_be_bytes());
    png[20..24].copy_from_slice(&h.to_be_bytes());
    let mut crc = !0u32;
    for &b in &png[12..29] {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    png[29..33].copy_from_slice(&(!crc).to_be_bytes());
}

/// Encoded images of several sizes and formats; sample `i` is labelled `i`. One index can hold
/// garbage.
struct Encoded {
    images: Vec<Arc<[u8]>>,
    bad_at: Option<usize>,
}

impl Encoded {
    fn new(len: usize) -> Self {
        let sources = [
            fixture("rgb.png"),
            fixture("big.png"),
            fixture("rgb_q90.jpg"),
            fixture("rgb_lossless.webp"),
        ];
        Self {
            images: (0..len)
                .map(|i| Arc::from(sources[i % sources.len()].as_slice()))
                .collect(),
            bad_at: None,
        }
    }
}

impl Dataset for Encoded {
    type Item = (Arc<[u8]>, usize);

    fn get(&self, index: usize) -> Result<(Arc<[u8]>, usize)> {
        if self.bad_at == Some(index) {
            return Ok((Arc::from(&b"garbage"[..]), index));
        }
        Ok((Arc::clone(&self.images[index]), index))
    }

    fn len(&self) -> usize {
        self.images.len()
    }
}

/// Images of different sizes resized and padded to one sample shape.
fn pipeline() -> ImageTransform {
    ImageTransform::new(Channels::Rgb)
        .resize(40, 30, ResizeFilter::Bilinear)
        .pad(48, 32, 0)
        .to_float()
        .normalize(MEAN, STD)
}

#[test]
fn loader_with_workers_reuses_pooled_buffers() {
    const BATCH: usize = 4;
    let dataset = Encoded::new(10);
    let reference = ImageCollator::<f32>::new(pipeline()).unwrap();
    let expected: Vec<Vec<f32>> = dataset
        .images
        .iter()
        .map(|bytes| {
            let mut row = vec![0f32; reference.sample_len()];
            reference.write_sample(bytes, &mut row).unwrap();
            row
        })
        .collect();
    let len = reference.sample_len();
    let shape = reference.batch_shape(BATCH);
    assert_eq!(shape, [BATCH, 3, 32, 48]);

    let mut loader = DataLoader::builder(dataset)
        .batch_size(BATCH)
        .num_workers(3)
        .prefetch_depth(2)
        .collate_into(
            ImageCollator::<f32>::new(pipeline()).unwrap(),
            HeapAllocator::<f32>::for_shape(&shape).unwrap(),
        )
        .build();
    let mut buffers = HashSet::new();
    for _epoch in 0..3 {
        let mut seen = Vec::new();
        for batch in &mut loader {
            let batch = batch.unwrap();
            buffers.insert(batch.as_ptr() as usize);
            assert_eq!(batch.output().len(), batch.num_samples());
            for (info, row) in batch.output().iter().zip(batch.chunks_exact(len)) {
                assert_eq!(row, expected[info.meta].as_slice(), "sample {}", info.meta);
                assert!(
                    (info.width, info.height) == (64, 48)
                        || (info.width, info.height) == (160, 120)
                );
                seen.push(info.meta);
            }
        }
        // 10 samples in batches of 4: the last one has 2.
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
    }
    let pool = loader.collator().pool();
    assert!(
        pool.allocated() <= 3 + 2 + 1,
        "{} buffers",
        pool.allocated()
    );
    assert_eq!(buffers.len(), pool.allocated());
}

#[test]
fn a_bad_image_fails_its_batch_only() {
    let mut dataset = Encoded::new(8);
    dataset.bad_at = Some(5);
    let collator = ImageCollator::<f16>::new(pipeline().layout(Layout::Nhwc)).unwrap();
    let shape = collator.batch_shape(2);
    let mut loader = DataLoader::builder(dataset)
        .batch_size(2)
        .num_workers(2)
        .collate_into(collator, HeapAllocator::<f16>::for_shape(&shape).unwrap())
        .build();
    let results: Vec<_> = loader.iter().map(|b| b.map(|b| b.num_samples())).collect();
    assert_eq!(results.len(), 4);
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(n) => assert!(i != 2 && *n == 2),
            Err(e) => {
                assert!(
                    i == 2 && e.to_string().contains("image 1 of the batch"),
                    "{e}"
                );
                let err = e.downcast_ref::<ImageError>().expect("an ImageError");
                assert_eq!(err.sample(), Some(1));
                assert!(matches!(err.kind(), ImageError::Decode(_)), "{err:?}");
            }
        }
    }
    // The failed batch returned its buffer: the next epoch runs.
    assert_eq!(loader.iter().filter(Result::is_ok).count(), 3);
}

#[test]
fn buffer_too_small_is_an_error() {
    let collator = ImageCollator::<u8>::new(rgb_hwc(64, 48)).unwrap();
    let mut buffer = vec![0u8; collator.sample_len() * 2 - 1];
    let items = vec![fixture("rgb.png"), fixture("rgb.png")];
    assert!(collator.collate_into(items, &mut buffer).is_err());
}

#[test]
fn dtype_chosen_at_run_time() {
    const BATCH: usize = 4;
    for dtype in [DType::U8, DType::F16, DType::F32] {
        let transform = ImageTransform::new(Channels::Rgb).resize(40, 30, ResizeFilter::Bilinear);
        let transform = match dtype {
            DType::U8 => transform,
            _ => transform.to_float().normalize(MEAN, STD),
        };
        let typed = ImageCollator::<f32>::new(transform.clone()).unwrap();
        let collator = DynImageCollator::new(transform, dtype).unwrap();
        let allocator = collator.allocator(BATCH).unwrap();
        let len = collator.sample_len();
        let mut loader = DataLoader::builder(Encoded::new(6))
            .batch_size(BATCH)
            .num_workers(2)
            .collate_into(collator, allocator)
            .build();
        let mut seen = 0;
        for batch in &mut loader {
            let batch = batch.unwrap();
            assert_eq!(batch.dtype(), dtype);
            assert_eq!(batch.len(), BATCH * len);
            for (row, info) in batch.output().iter().enumerate() {
                let mut expected = vec![0f32; len];
                typed
                    .write_sample(&loader_bytes(info.meta), &mut expected)
                    .unwrap();
                let range = row * len..(row + 1) * len;
                match &*batch {
                    ImageBuffer::U8(v) => assert!(
                        v[range]
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| f32::from(*a) == *b)
                    ),
                    ImageBuffer::F16(v) => assert!(
                        v[range]
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| *a == f16::from_f32(*b))
                    ),
                    ImageBuffer::F32(v) => assert_eq!(&v[range], expected.as_slice()),
                }
                seen += 1;
            }
        }
        assert_eq!(seen, 6);
    }
}

fn loader_bytes(index: usize) -> Vec<u8> {
    Encoded::new(index + 1).images[index].to_vec()
}
