//! Image operations: decode PNG bytes, convert, pad, resize, flip, normalise, and batch the
//! images into one buffer, all on the loader's threads. One step is a closure of our own, to
//! show that your operations chain with the built-in ones.
//!
//! Each operation is a `Transform`; `then` chains them, and the chain is checked by the
//! compiler: putting `Normalize` (which needs float pixels) before `ToFloat` does not compile.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example image_ops --features image
//! ```

use std::io::Cursor;

use bytes::Bytes;
use dataloader_rs::collator::ImageCollator;
use dataloader_rs::image::{Decode, HFlip, Image, Normalize, PadTo, Resize, ToChw, ToFloat, ToRgb};
use dataloader_rs::{DataLoader, Dataset, Transform, error::Result};

/// Nine encoded images held in memory: eight PNG tiles (the last column narrower, as at the
/// edge of a large image) and, last, a file that is not an image at all.
struct EncodedTiles(Vec<Bytes>);

impl Dataset for EncodedTiles {
    type Item = Bytes;

    fn get(&self, index: usize) -> Result<Bytes> {
        // Cloning `Bytes` shares the buffer: no copy.
        Ok(self.0[index].clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// A 160-pixel-high RGB PNG, `width` pixels wide, filled with a gradient.
fn png(width: u32, seed: u32) -> Result<Bytes> {
    let pixels = image::RgbImage::from_fn(width, 160, |x, y| {
        image::Rgb([(x + seed) as u8, (y + seed) as u8, (x + y) as u8])
    });
    let mut encoded = Vec::new();
    pixels.write_to(&mut Cursor::new(&mut encoded), image::ImageFormat::Png)?;
    Ok(Bytes::from(encoded))
}

fn main() -> Result<()> {
    let mut tiles = (0..8)
        .map(|i| png(if i % 4 == 3 { 100 } else { 160 }, i))
        .collect::<Result<Vec<_>>>()?;
    tiles.push(Bytes::from_static(b"not an image"));

    // Our own operation: invert the colours of an 8-bit image, in place. Any closure
    // `Fn(usize, Input) -> Result<Output>` is a transform; its types must be written out.
    let invert = |_index: usize, mut image: Image<u8>| -> Result<Image<u8>> {
        for value in image.data_mut() {
            *value = 255 - *value;
        }
        Ok(image)
    };

    let transform = Decode::new() // PNG, JPEG, WebP, TIFF or BMP bytes -> 8-bit image, height x width x channels
        .then(ToRgb) // gray or RGBA images -> 3 channels
        .then(invert) // our own step
        .then(PadTo::new(160, 160, 0.0)) // the narrow edge tiles are padded with black, at the right
        .then(Resize::new(224, 224)) // bilinear, antialiased
        .then(HFlip)
        .then(ToFloat::default()) // 8-bit values -> floats in [0, 1]
        .then(Normalize::imagenet()) // (x - mean) / std per channel
        .then(ToChw); // channels first, as PyTorch models expect

    let mut loader = DataLoader::builder(EncodedTiles(tiles))
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(2) // the images of a batch are decoded on 2 threads at once
        .transform(transform)
        // One buffer per batch, in batch x channels x height x width order. Up to 4 buffers of
        // dropped batches are reused, which saves the cost of touching fresh memory.
        .collator(ImageCollator::default().reuse_buffers(4))
        .build();

    for batch in &mut loader {
        match batch {
            Ok(batch) => {
                let mean = batch.data().iter().sum::<f32>() / batch.data().len() as f32;
                println!("batch of shape {:?}, mean value {mean:+.3}", batch.shape());
            }
            // The last item is not an image: its batch fails, and only that batch.
            Err(error) => println!("a batch failed: {error}"),
        }
    }

    // A transform can also be called directly, outside a loader.
    let one = Decode::new().then(ToRgb).apply(0, png(64, 0)?)?;
    println!(
        "one image decoded directly: height, width, channels = {:?}",
        one.shape()
    );
    Ok(())
}
