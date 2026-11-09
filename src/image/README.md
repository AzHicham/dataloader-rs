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
| `ImageError` | `Decode`, `Limits` or `Invalid`, boxed in the crate's `Error` |

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
