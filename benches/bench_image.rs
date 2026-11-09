//! Image transforms: msintuit-ec's tile chain (decode PNG, RGB, pad to 224, float, normalize)
//! per item, the image collator with and without reused buffers, and both through a loader
//! (feature `image`).
//!
//! Tiles are 224 × 224 RGB PNGs of gradients and noise (every eighth one 224 × 150, an
//! undersized edge tile), encoded once in memory: no I/O, the work measured is CPU only.
//!
//! ```text
//! cargo bench --features image --bench bench_image
//! ```

use std::io::Cursor;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::collator::{BatchLayout, ImageCollator};
use dataloader_rs::image::{Chw, Decode, Image, Normalize, PadTo, ToChw, ToFloat, ToRgb};
use dataloader_rs::{DataLoader, Dataset, Transform, error::Result};
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};

const TILE: usize = 224;
const BATCH: usize = 64;
/// Items per loader epoch: 8 batches.
const ITEMS: usize = 512;

/// A `width × height` RGB PNG: gradients plus noise, about the size of a tissue tile.
fn png(width: usize, height: usize, seed: u64) -> Bytes {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let noise = rng.u8(..32);
            pixels.extend([
                (x as u8).wrapping_add(noise),
                (y as u8).wrapping_add(noise),
                noise,
            ]);
        }
    }
    let mut out = Cursor::new(Vec::new());
    PngEncoder::new(&mut out)
        .write_image(
            &pixels,
            width as u32,
            height as u32,
            ExtendedColorType::Rgb8,
        )
        .unwrap();
    Bytes::from(out.into_inner())
}

/// `ITEMS` tiles, every eighth undersized.
struct Tiles(Vec<Bytes>);

impl Tiles {
    fn new() -> Self {
        let full = png(TILE, TILE, 1);
        let edge = png(TILE, 150, 2);
        Self(
            (0..ITEMS)
                .map(|i| {
                    if i % 8 == 7 {
                        edge.clone()
                    } else {
                        full.clone()
                    }
                })
                .collect(),
        )
    }
}

impl Dataset for Tiles {
    type Item = Bytes;

    fn get(&self, index: usize) -> Result<Bytes> {
        Ok(self.0[index].clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// The msintuit-ec feature pass: Decode → ToRgb → PadTo(224) → ToFloat → Normalize → CHW.
fn chain() -> impl Transform<Bytes, Output = Image<f32, Chw>> + Clone {
    Decode::new()
        .then(ToRgb)
        .then(PadTo::new(TILE, TILE, 0.0))
        .then(ToFloat::default())
        .then(Normalize::imagenet())
        .then(ToChw)
}

/// One tile through the chain, and the collation of a batch of them.
fn per_item(c: &mut Criterion) {
    let tiles = Tiles::new();
    let mut group = c.benchmark_group("image_item");
    group.throughput(Throughput::Elements(1));
    let chain = chain();
    group.bench_function("decode_pad_float_normalize", |b| {
        b.iter(|| chain.apply(0, tiles.0[0].clone()).unwrap())
    });
    let decode = Decode::new();
    group.bench_function("decode", |b| b.iter(|| decode.decode(&tiles.0[0]).unwrap()));
    group.finish();

    let items: Vec<Image<f32, Chw>> = (0..BATCH)
        .map(|i| chain.apply(i, tiles.0[i].clone()).unwrap())
        .collect();
    let refs: Vec<&Image<f32, Chw>> = items.iter().collect();
    let mut group = c.benchmark_group("image_collate");
    group.throughput(Throughput::Elements(BATCH as u64));
    for (layout, buffers) in [
        (BatchLayout::Nchw, 0),
        (BatchLayout::Nchw, 4),
        (BatchLayout::Nhwc, 0),
    ] {
        let collator = ImageCollator::new(layout).reuse_buffers(buffers);
        let name = format!("{}_buffers{buffers}", layout.name());
        group.bench_function(BenchmarkId::new("f32_64x3x224x224", name), |b| {
            b.iter(|| collator.collate_refs(&refs).unwrap())
        });
    }
    group.finish();
}

/// An epoch of `ITEMS` tiles in batches of 64, transformed on the loader's pool.
fn loader(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_loader");
    group.throughput(Throughput::Elements(ITEMS as u64));
    group.sample_size(10);
    // 4 threads either way: one batch at a time with 4 items in parallel, or 4 batches.
    for (workers, buffers) in [(1, 0), (1, 8), (4, 0), (4, 8)] {
        let mut loader = DataLoader::builder(Tiles::new())
            .batch_size(BATCH)
            .num_workers(workers)
            .item_concurrency(4 / workers)
            .transform(chain())
            .collator(ImageCollator::default().reuse_buffers(buffers))
            .build();
        let name = format!("w{workers}_buffers{buffers}");
        group.bench_function(BenchmarkId::new("png_224_f32_nchw", name), |b| {
            b.iter(|| {
                loader
                    .iter()
                    .map(|batch| batch.unwrap().shape()[0])
                    .sum::<usize>()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, per_item, loader);
criterion_main!(benches);
