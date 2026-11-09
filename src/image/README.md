# Image transforms (feature `image`)

Image operations on one item at a time, each a [`Transform`](crate::Transform): the loader runs
them on its thread pool, on the items of a batch in parallel. A chain of them is typed, so a
wrong order does not compile:

```rust
use bytes::Bytes;
use dataloader_rs::Transform;
use dataloader_rs::image::{Decode, Image, ToRgb};

let png = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/image/gray.png"));
let decode = Decode::new().then(ToRgb);
let image: Image<u8> = decode.apply(0, Bytes::from(png.unwrap())).unwrap();
assert_eq!(image.shape(), [29, 37, 3]);
```

| Type | What |
|---|---|
| `Image<T, L>` | An owned image, with samples `T` of type `u8` or `f32` and a layout `L`, `Hwc` (height × width × channels, the default) or `Chw`. It is never empty, and has 1 to 4 channels. |
| `Decode` | Decodes PNG, JPEG, WebP, TIFF or BMP `Bytes` into an `Image<u8>`, with pure-Rust decoders. The format is detected from the bytes. |
| `DecodeLimits` | The largest image `Decode` accepts: 16384 × 16384 pixels and 256 MiB by default. |
| `ToRgb`, `ToRgba`, `ToGray` | Channel conversions of an `Image<u8>`, as Pillow's `convert` does them. |
| `Crop`, `CenterCrop` | A region, with its bounds checked; the central region, padded with zeros if it is larger than the image, as in torchvision. |
| `Pad`, `PadMode`, `Fill` | Padding of each side: constant (one value per channel), edge, reflect or symmetric. |
| `PadTo`, `Anchor` | Constant padding up to a fixed size, never cropping. With the default top-left anchor, the padding goes to the right and bottom. |
| `Resize`, `Filter` | Resizing with `fast_image_resize`, to an exact size or by the shorter side (with an optional `max_size`), with the nearest, bilinear, bicubic or Lanczos3 filter, antialiased by default. |
| `HFlip`, `VFlip`, `Rotate90` | Mirror images and lossless quarter turns (as `numpy.rot90`). |
| `ToFloat` | Converts an `Image<u8, L>` into an `Image<f32, L>`, divided by 255 by default (the values of `ToTensor`). |
| `Normalize` | Computes `(x - mean) / std` per channel on `f32` images, in either layout; `Normalize::imagenet()` has ImageNet's values. |
| `ToChw`, `ToHwc` | Layout changes (`ToTensor` gives `Chw`). |
| `ImageError` | The error of these operations (`Decode`, `Limits` or `Invalid`), boxed in the crate's `Error`. |

A typical chain for image tiles: decode, convert to RGB, pad the smaller tiles at the right and
bottom up to 224 × 224, then the values of torchvision's `ToTensor` and `Normalize`, bit for bit:

```rust
use dataloader_rs::Transform;
use dataloader_rs::image::{Decode, Normalize, PadTo, ToChw, ToFloat, ToRgb};

let tiles = Decode::new()
    .then(ToRgb)
    .then(PadTo::new(224, 224, 0.0))
    .then(ToFloat::default())
    .then(Normalize::imagenet())
    .then(ToChw);
# let _ = tiles;
```

Geometric transforms take an `Image<T>` of either sample type, except `Resize`, which takes `u8`
images only: resize before `ToFloat`. When the *first* transform of a chain accepts both types,
the input type must be written out (`Transform::<Image<u8>>::then(HFlip, Crop::new(0, 0, 8, 8))`);
a chain that starts with `Decode` needs nothing.

## Parity

The operations are checked against reference files made with Pillow, NumPy and torchvision
(`tests/fixtures/image/make_fixtures.py`; the tests are in `tests/image.rs`):

| Transform | Against | Result |
|---|---|---|
| decode PNG (8/16-bit, gray, alpha, palette), BMP, TIFF, lossless WebP | Pillow | exact |
| decode JPEG, lossy WebP | Pillow (libjpeg-turbo, libwebp) | JPEG max 3, mean 0.36; WebP exact today; tests allow max 4 / 1 |
| `ToRgb`, `ToRgba`, `ToGray` | Pillow `convert` | exact |
| `Crop`, `CenterCrop`, `Pad` (4 modes), `HFlip`, `VFlip`, `Rotate90` | torchvision v2, `numpy.pad`, `numpy.rot90` | exact |
| `PadTo` | torchvision `F.pad` on the right and bottom edges | exact |
| `Resize` | Pillow (antialiased), torch (`antialias=False`, bilinear) | nearest exact; bilinear max 1, mean ≤ 0.2; bicubic max 4; Lanczos3 max 7 (means ≤ 0.3) |
| `ToFloat`, `Normalize`, `ToChw` | torchvision `ToTensor`, `Normalize` | exact (v2 `ToDtype(scale=True)` multiplies by 1/255: 1 ulp off on half the values) |
| The chain `Decode`, `ToRgb`, `PadTo`, `ToFloat`, `Normalize` | Pillow's `convert("RGB")`, padding at the right and bottom, then torchvision | exact, on RGB, RGBA, gray, palette and undersized tiles |

## Decoding

- Codecs: the `image` crate's pure-Rust decoders (`png`, `zune-jpeg`, `image-webp`, `tiff`,
  BMP); no C library.
- The decoded image keeps its channels: 1 for gray, 2 for gray with alpha, 3 for RGB and 4 for
  RGBA. Palette PNGs are expanded to RGB or RGBA. 16-bit images keep their 8 high bits (Pillow does the same for RGB; for a
  16-bit gray PNG Pillow clips to 255 instead).
- Untrusted input is expected. The header is checked against `DecodeLimits` before any pixel is
  decoded, every size is computed with checked arithmetic, and every buffer is reserved with a
  fallible allocation, so a decompression bomb or a huge declared size is an error for its
  batch, not an abort. A JPEG
  without its end marker is refused (the decoder would fill the rest with gray; Pillow raises).
