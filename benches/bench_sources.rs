//! Benchmark: `HttpSource` + `ImageCollator` against a local HTTP server, next to the same
//! images from a `MemorySource`, so the difference is the cost of HTTP.
//!
//! FIXED:
//!   epoch = 512 lossless WebP images of 224×224 (8 distinct ones, cycled; about 116 KiB each),
//!   decoded to RGB, then `ToTensor` + `Normalize` into f32 NCHW batches of 64; `num_workers(4)`;
//!   the server (`tests/support/http_server.rs`) runs in the bench process, one thread per
//!   keep-alive connection
//!
//! SWEPT:
//!   source: `memory` or `http`
//!   items_in_parallel: 1 or 4 (requests in flight: 4 or 16)
//!
//! Groups:
//!   "sources/image_epoch_224"   — `<source>/in_parallel/<n>`
//!
//! Throughput unit: images per epoch.
//!
//! Run on the E-cores of the reference laptop:
//!   taskset -c 12-19 cargo bench --features sources,image --bench bench_sources

#[path = "../tests/support/http_server.rs"]
mod http_server;

use std::{hint::black_box, io::Cursor, sync::Arc};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dataloader_rs::{
    DataLoader, Dataset,
    collator::HeapAllocator,
    image::{Channels, ImageCollator, ImageTransform},
    sources::{HttpSource, MemorySource, SourceDataset},
};
use http_server::{Reply, Server};
use image::{ExtendedColorType, ImageEncoder, codecs::webp::WebPEncoder};

const BATCH: usize = 64;
const SIDE: u32 = 224;
const LEN: usize = BATCH * 8;
const DISTINCT: usize = 8;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A lossless WebP of `side × side`: gradients, a checkerboard and some noise (as in
/// `bench_image`).
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

fn bench_epoch<D>(c: &mut Criterion, group: &str, name: &str, dataset: D)
where
    D: Dataset + Clone,
    D::Item: dataloader_rs::image::ImageItem,
{
    let mut group = c.benchmark_group(group);
    group.throughput(Throughput::Elements(LEN as u64));
    group.sample_size(10);
    for in_parallel in [1usize, 4] {
        let transform = ImageTransform::new(Channels::Rgb)
            .input_size(SIDE, SIDE)
            .to_float()
            .normalize(MEAN, STD);
        let collator = ImageCollator::<f32>::new(transform).unwrap();
        let alloc = HeapAllocator::<f32>::for_shape(&collator.batch_shape(BATCH)).unwrap();
        let mut loader = DataLoader::builder(dataset.clone())
            .batch_size(BATCH)
            .num_workers(4)
            .items_in_parallel(in_parallel)
            .collate_into(collator, alloc)
            .build();
        group.bench_function(
            BenchmarkId::new(format!("{name}/in_parallel"), in_parallel),
            |b| {
                b.iter(|| {
                    loader
                        .iter()
                        .for_each(|batch| drop(black_box(batch.unwrap())))
                });
            },
        );
    }
    group.finish();
}

fn bench_sources(c: &mut Criterion) {
    let images: Vec<Arc<[u8]>> = (0..DISTINCT as u32).map(|seed| webp(SIDE, seed)).collect();

    let memory = MemorySource::new((0..LEN).map(|i| images[i % DISTINCT].clone()));
    bench_epoch(
        c,
        "sources/image_epoch_224",
        "memory",
        SourceDataset::new(memory),
    );

    let served = images.clone();
    let server = Server::start(move |request| {
        let index: usize = request
            .target
            .trim_start_matches("/img/")
            .parse()
            .unwrap_or(0);
        Reply::ok(served[index % DISTINCT].clone())
    });
    let http = HttpSource::urls((0..LEN).map(|i| server.url(&format!("/img/{i}")))).unwrap();
    bench_epoch(
        c,
        "sources/image_epoch_224",
        "http",
        SourceDataset::new(http),
    );
}

criterion_group!(benches, bench_sources);
criterion_main!(benches);
