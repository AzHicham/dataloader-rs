//! Benchmark: the `image` pipeline into pooled batch buffers against a naive per-sample one.
//!
//! FIXED:
//!   batch = 64 lossless WebP images (8 distinct ones, cycled), decoded to RGB, then
//!   `ToTensor` + `Normalize` (ImageNet mean/std) into an f32 NCHW batch of 64×3×224×224
//!   (38.5 MB)
//!
//! SWEPT:
//!   input: 224×224 (no resize) or 512×512 resized to 224×224 (bilinear)
//!   collation: `naive` (per sample: `image::load_from_memory`, `to_rgb8`, a new resizer and
//!              output image, a new CHW `Vec<f32>`; then a new batch `Vec`) or `pooled`
//!              (`ImageCollator` writing each sample into its row of a pooled buffer)
//!
//! Groups:
//!   "image/collate_224"         — one batch, on the bench thread
//!   "image/collate_512_to_224"  — one batch, on the bench thread
//!   "image/loader_w4_224"       — an epoch of 8 batches, `num_workers(4)`
//!
//! Throughput unit: images per iteration.
//!
//! Run on the E-cores of the reference laptop:
//!   taskset -c 12-19 cargo bench --features image --bench bench_image

use std::{hint::black_box, io::Cursor, sync::Arc};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::{
    Collator, DataLoader, Dataset,
    collator::{HeapAllocator, PooledCollator},
    error::Result,
    image::{Channels, ImageCollator, ImageTransform, ResizeFilter},
};
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image, ImageRef},
};
use image::{
    ExtendedColorType, ImageEncoder, RgbImage, codecs::webp::WebPEncoder, load_from_memory,
};

const BATCH: usize = 64;
const SIDE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const BATCHES_PER_EPOCH: usize = 8;

/// A lossless WebP of `side × side`: gradients, a checkerboard and some noise, so it neither
/// compresses to nothing nor is pure noise.
fn webp(side: u32, seed: u32) -> Arc<[u8]> {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state % 24) as i32 - 12
    };
    let mut pixels = Vec::with_capacity((side * side * 3) as usize);
    for y in 0..side {
        for x in 0..side {
            let dark = ((x / 8 + y / 8) % 2 == 0) as i32;
            for base in [x * 255 / side, y * 255 / side, (x + y + seed * 40) % 256] {
                let v = base as i32 * (5 - 2 * dark) / 5 + noise();
                pixels.push(v.clamp(0, 255) as u8);
            }
        }
    }
    let mut out = Vec::new();
    WebPEncoder::new_lossless(Cursor::new(&mut out))
        .write_image(&pixels, side, side, ExtendedColorType::Rgb8)
        .unwrap();
    out.into()
}

/// Encoded images, cycled.
struct Encoded {
    images: Vec<Arc<[u8]>>,
    len: usize,
}

impl Encoded {
    fn new(side: u32, len: usize) -> Self {
        Self {
            images: (0..8).map(|seed| webp(side, seed)).collect(),
            len,
        }
    }
}

impl Dataset for Encoded {
    type Item = Arc<[u8]>;

    fn get(&self, index: usize) -> Result<Arc<[u8]>> {
        Ok(Arc::clone(&self.images[index % self.images.len()]))
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// The naive pipeline: every sample and every batch allocates.
struct NaiveCollator {
    resize: bool,
}

impl NaiveCollator {
    fn sample(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let mut rgb = load_from_memory(bytes)?.to_rgb8();
        if self.resize {
            let src = ImageRef::new(rgb.width(), rgb.height(), rgb.as_raw(), PixelType::U8x3)?;
            let mut dst = Image::new(SIDE, SIDE, PixelType::U8x3);
            let options =
                ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
            Resizer::new().resize(&src, &mut dst, &options)?;
            rgb = RgbImage::from_raw(SIDE, SIDE, dst.into_vec()).ok_or("resize")?;
        }
        let plane = rgb.as_raw().len() / 3;
        let mut chw = vec![0f32; plane * 3];
        for (i, px) in rgb.as_raw().as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c];
            }
        }
        Ok(chw)
    }
}

impl Collator<Arc<[u8]>> for NaiveCollator {
    type Batch = Vec<f32>;

    fn collate(&self, items: Vec<Arc<[u8]>>) -> Result<Vec<f32>> {
        let samples = items
            .iter()
            .map(|bytes| self.sample(bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(samples.concat())
    }
}

fn transform(resize: bool) -> ImageTransform {
    let t = ImageTransform::new(Channels::Rgb);
    let t = if resize {
        t.resize(SIDE, SIDE, ResizeFilter::Bilinear)
    } else {
        t.input_size(SIDE, SIDE)
    };
    t.to_float().normalize(MEAN, STD)
}

fn pooled(resize: bool) -> PooledCollator<ImageCollator<f32>, HeapAllocator<f32>> {
    let collator = ImageCollator::<f32>::new(transform(resize)).unwrap();
    let alloc = HeapAllocator::<f32>::for_shape(&collator.batch_shape(BATCH)).unwrap();
    PooledCollator::new(collator, alloc, 2)
}

fn bench_collate(c: &mut Criterion) {
    for (group, side, resize) in [
        ("image/collate_224", SIDE, false),
        ("image/collate_512_to_224", 512, true),
    ] {
        let ds = Encoded::new(side, BATCH);
        let items = || (0..BATCH).map(|i| ds.get(i).unwrap()).collect::<Vec<_>>();
        let naive = NaiveCollator { resize };
        let pooled = pooled(resize);
        // Same values from both pipelines.
        let a = naive.collate(items()).unwrap();
        let b = pooled.collate(items()).unwrap();
        assert_eq!(a, b[..]);

        let mut group = c.benchmark_group(group);
        group.throughput(Throughput::Elements(BATCH as u64));
        group.sample_size(20);
        group.bench_function(BenchmarkId::from_parameter("naive"), |b| {
            b.iter(|| black_box(naive.collate(items()).unwrap()));
        });
        group.bench_function(BenchmarkId::from_parameter("pooled"), |b| {
            b.iter(|| black_box(pooled.collate(items()).unwrap()));
        });
        group.finish();
    }
}

fn bench_loader(c: &mut Criterion) {
    let len = BATCH * BATCHES_PER_EPOCH;
    let mut group = c.benchmark_group("image/loader_w4_224");
    group.throughput(Throughput::Elements(len as u64));
    group.sample_size(10);

    let mut naive = DataLoader::builder(Encoded::new(SIDE, len))
        .batch_size(BATCH)
        .num_workers(4)
        .collator(NaiveCollator { resize: false })
        .build();
    group.bench_function(BenchmarkId::from_parameter("naive"), |b| {
        b.iter(|| {
            naive
                .iter()
                .for_each(|batch| drop(black_box(batch.unwrap())))
        });
    });

    let collator = ImageCollator::<f32>::new(transform(false)).unwrap();
    let alloc = HeapAllocator::<f32>::for_shape(&collator.batch_shape(BATCH)).unwrap();
    let mut pooled = DataLoader::builder(Encoded::new(SIDE, len))
        .batch_size(BATCH)
        .num_workers(4)
        .collate_into(collator, alloc)
        .build();
    group.bench_function(BenchmarkId::from_parameter("pooled"), |b| {
        b.iter(|| {
            pooled
                .iter()
                .for_each(|batch| drop(black_box(batch.unwrap())))
        });
    });
    group.finish();
}

criterion_group!(benches, bench_collate, bench_loader);
criterion_main!(benches);
