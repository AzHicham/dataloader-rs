# Images (feature `image`)

Decode encoded images, transform them, and write each one straight into its row of a reusable
batch buffer: the data path of an image classifier's loader, with no tensor allocated per sample.

```rust
use dataloader_rs::{
    DataLoader, Dataset,
    collator::HeapAllocator,
    error::Result,
    image::{Channels, ImageCollator, ImageTransform, ResizeFilter},
};

/// Encoded images (PNG, JPEG or WebP) with a label each.
struct Images(Vec<(Vec<u8>, u32)>);

impl Dataset for Images {
    type Item = (Vec<u8>, u32);

    fn get(&self, index: usize) -> Result<(Vec<u8>, u32)> {
        Ok(self.0[index].clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

let transform = ImageTransform::new(Channels::Rgb)
    .resize(256, 256, ResizeFilter::Bilinear)
    .crop(16, 16, 224, 224)
    .to_float()
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
let collator = ImageCollator::<f32>::new(transform)?;
let shape = collator.batch_shape(64); // [64, 3, 224, 224]
let mut loader = DataLoader::builder(Images(Vec::new()))
    .batch_size(64)
    .num_workers(4)
    .collate_into(collator, HeapAllocator::<f32>::for_shape(&shape)?)
    .build();
for batch in &mut loader {
    let batch = batch?;
    // `&batch[..]` is the NCHW batch; the first `num_samples()` rows hold this batch.
    let _labels: Vec<u32> = batch.output().iter().map(|info| info.meta).collect();
}
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

## Pipeline

An `ImageTransform` is plain data: the channels, a list of `Op`s, the layout and the limits. It is
validated once, when the collator is built; an invalid pipeline is an
`ImageError::InvalidPipeline` naming the operation.

| Step | Effect |
|---|---|
| decode | PNG, JPEG or WebP, detected from the bytes, to 8-bit `Channels::Rgb` (gray replicated, alpha dropped, as Pillow's `convert("RGB")`) or `Channels::Gray` (Pillow's ITU-R 601-2 luma); 16-bit PNGs keep their high byte |
| `input_size(w, h)` | optional: every image must be `w × h` |
| `Op::Resize { width, height, filter }` | `fast_image_resize` convolution; `ResizeFilter::Bilinear` by default |
| `Op::Crop { left, top, width, height }` | keep a region; one that leaves the image is an error |
| `Op::Pad { width, height, fill }` | grow to `width × height`, image at the top left, `fill` elsewhere |
| `Op::ToFloat` | `x / 255`, as `ToTensor` |
| `Op::Normalize { mean, std }` | `(x - mean[c]) / std[c]`, as `Normalize`; one value applies to every channel |
| `Layout::Nchw` / `Layout::Nhwc` | channel planes (PyTorch) or interleaved pixels |

Geometric operations come first and work on 8-bit pixels. The value operations are computed once
per channel and input value (a 256-entry table), with the same `f32` steps as torchvision, and
applied while the sample is written into its row. The sample size must be known when the pipeline
is built: set `input_size`, or end the geometric operations with a resize, crop or pad.

The batch's value type is the collator's `T`: `u8` (no value operation), `f32` or `half::f16`.
`DynImageCollator` takes it as a `DType` value instead and collates into `ImageBuffer`s (an enum
of `Vec<u8>`, `Vec<f16>`, `Vec<f32>`) from its `allocator`: a configuration without type
parameters, for bindings and configuration files.

`ImageCollator`'s batch output is one `SampleInfo` per sample: the decoded size and the item's
metadata. Items are `Vec<u8>`, `Arc<[u8]>`, `Box<[u8]>`, `&'static [u8]`, `Cow<'static, [u8]>`, or
`(bytes, meta)` pairs whose `meta` is passed through (a label, a path).

Each batch is collated on the thread that loads it (a worker with `num_workers > 0`), with working
images and a resizer reused across batches: one set per batch collated at once.

## Errors and limits

Every error is an `ImageError`. A sample that fails makes its batch fail with
`ImageError::Sample { index, source }`, boxed in the loader's error (`downcast_ref::<ImageError>()`);
the buffer returns to the pool and the next batches load.

`ImageLimits` (default: 16384 × 16384 pixels and 256 MiB per working image) bounds the image the
header announces before any pixel is decoded, the decoder's own allocations, and every size of the
pipeline. Sizes are checked with overflow-checked arithmetic and buffers reserved with
`try_reserve`, so a hostile header is an `ImageError::Limits`, not an allocation. Corrupt and
truncated files are `ImageError::Decode` (a JPEG without its end marker too: zune-jpeg would
return a partly gray image, Pillow raises).

## Dependencies

Pure Rust, no C library, no `-sys` crate, no FFI: `image` with its `png`, `jpeg` (zune-jpeg) and
`webp` (image-webp) decoders only, `fast_image_resize` (SIMD, its `unsafe` stays inside it) and
`half`. This crate keeps `#![forbid(unsafe_code)]`.

## Parity with Pillow and torchvision

Golden fixtures from Pillow 13.0 and torchvision 0.29 (`tests/fixtures/image/make_fixtures.py`),
checked by `tests/integration_image.rs`; max |Δ| in 8-bit levels unless noted.

| Step | Against | Max \|Δ\| |
|---|---|---|
| PNG (RGB, RGBA), lossless WebP decode | `Image.open().convert("RGB")` | 0 (bit-identical) |
| RGB → gray | `convert("L")` | 0 |
| JPEG (quality 90) decode | Pillow (libjpeg-turbo) | 3 |
| `ToTensor` | torchvision | 0 |
| `ToTensor` + `Normalize` (ImageNet), `f32` | torchvision | 0 |
| the same, `f16` | `.half()` | 0 (bit-identical) |
| resize 160×120 → 64×48 / 64×48 → 96×72, `Bilinear` | `Image.resize(BILINEAR)` | 1 / 1 |
| `Box` | `BOX` | 1 / 0 |
| `Hamming` | `HAMMING` | 1 / 1 |
| `Bicubic` | `BICUBIC` | 2 / 8 |
| `Lanczos3` | `LANCZOS` | 4 / 14 |
| `Nearest` | `NEAREST` | 0 / another source pixel on 25 % of the values |
| resize `Bilinear` + `ToTensor` + `Normalize` | Pillow + torchvision | 0.0175 (one level / std) |

Resizing uses `fast_image_resize`, not a port of Pillow: the convolution filters agree to one
level when downscaling with `Bilinear` (the default), and drift more with the sharper filters when
upscaling; nearest upscaling rounds the source position differently.
