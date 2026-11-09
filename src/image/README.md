# Image transforms (feature `image`)

Per-item image operations, each a [`Transform`](crate::Transform): the loader runs them on its
pool, item by item, in parallel. Chains are typed, so a wrong order does not compile:

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
| `Image<T, L>` | an owned image: `T` = `u8` or `f32`, layout `L` = `Hwc` (default) or `Chw`; never empty, 1 to 4 channels |
| `Decode` | PNG, JPEG, WebP, TIFF, BMP `Bytes` → `Image<u8>`, format sniffed from the bytes, pure-Rust decoders |
| `DecodeLimits` | the largest image `Decode` accepts: 16384 × 16384 and 256 MiB by default |
| `ToRgb`, `ToRgba`, `ToGray` | channel conversions of `Image<u8>`, as Pillow's `convert` |
| `Crop`, `CenterCrop` | a region (bounds checked), the central one (padded with zeros if larger, as torchvision) |
| `Pad`, `PadMode`, `Fill` | padding of each side: constant (per channel), edge, reflect, symmetric |
| `PadTo`, `Anchor` | constant padding up to a fixed size, never cropping; top-left anchor = prime_dx's `ConstantFill2d` |
| `Resize`, `Filter` | `fast_image_resize`: exact size or shorter side (+ `max_size`), nearest / bilinear / bicubic / Lanczos3, antialiased by default |
| `HFlip`, `VFlip`, `Rotate90` | mirrors and lossless quarter turns (`numpy.rot90`) |
| `ToFloat` | `Image<u8, L>` → `Image<f32, L>`, divided by 255 by default (`ToTensor`'s values) |
| `Normalize` | `(x - mean) / std` per channel on `f32` images, either layout; `Normalize::imagenet()` |
| `ToChw`, `ToHwc` | layout changes (`ToTensor`'s is `ToChw`) |
| `ImageError` | `Decode`, `Limits` or `Invalid`, boxed in the crate's `Error` |

msintuit-ec's tile transform (prime_dx's `ConstantFill2d` + `ToTensor` + `Normalize`), the same
values bit for bit:

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

Geometric transforms take `Image<T>` for both sample types, except `Resize` (`u8` only: resize
before `ToFloat`). A chain whose *first* transform takes both types needs the input type
written out (`Transform::<Image<u8>>::then(HFlip, Crop::new(0, 0, 8, 8))`); starting from
`Decode` it is inferred.

## Parity

Checked against golden files from Pillow, numpy, torchvision and prime_dx
(`tests/fixtures/image/make_fixtures.py`, tests in `tests/image.rs`):

| Transform | Against | Result |
|---|---|---|
| decode PNG (8/16-bit, gray, alpha, palette), BMP, TIFF, lossless WebP | Pillow | exact |
| decode JPEG, lossy WebP | Pillow (libjpeg-turbo, libwebp) | JPEG max 3, mean 0.36; WebP exact today; tests allow max 4 / 1 |
| `ToRgb`, `ToRgba`, `ToGray` | Pillow `convert` | exact |
| `Crop`, `CenterCrop`, `Pad` (4 modes), `HFlip`, `VFlip`, `Rotate90` | torchvision v2, `numpy.pad`, `numpy.rot90` | exact |
| `PadTo` | prime_dx `ConstantFill2d` | exact |
| `Resize` | Pillow (antialiased), torch (`antialias=False`, bilinear) | nearest exact; bilinear max 1, mean ≤ 0.2; bicubic max 4; Lanczos3 max 7 (means ≤ 0.3) |
| `ToFloat`, `Normalize`, `ToChw` | torchvision `ToTensor`, `Normalize` | exact (v2 `ToDtype(scale=True)` multiplies by 1/255: 1 ulp off on half the values) |
| `Decode → ToRgb → PadTo → ToFloat → Normalize` | prime_dx + torchvision, PIL `convert("RGB")` first | exact, on RGB, RGBA, gray, palette and undersized tiles |

## Decoding

- Codecs: the `image` crate's pure-Rust decoders (`png`, `zune-jpeg`, `image-webp`, `tiff`,
  BMP); no C library.
- The decoded image keeps its channels: gray 1, gray + alpha 2, RGB 3, RGBA 4. Palette PNGs are
  expanded to RGB(A). 16-bit images keep their 8 high bits (Pillow does the same for RGB; for a
  16-bit gray PNG Pillow clips to 255 instead).
- Untrusted input: the header is checked against `DecodeLimits` before any pixel is decoded;
  every size is computed with checked arithmetic and every buffer is reserved fallibly, so a
  decompression bomb or a huge declared size is an error for its batch, not an abort. A JPEG
  without its end marker is refused (the decoder would fill the rest with gray; Pillow raises).
