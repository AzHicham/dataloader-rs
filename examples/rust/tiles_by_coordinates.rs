//! Tiles of a large image, fetched over HTTP by their (x, y) coordinates, decoded and batched
//! together with their coordinates: each batch is `(images, coordinates)`.
//!
//! This is the usual way to run a model over a very large image (a satellite image, a
//! scanned microscope slide): a tile server cuts it into tiles on request, and the results are
//! put back in place with the coordinates. The example starts its own small tile server, so it
//! runs offline.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example tiles_by_coordinates --features http,image
//! ```

#[path = "local_server.rs"]
mod local_server;

use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use dataloader_rs::collator::{ImageBatch, ImageCollator};
use dataloader_rs::dataset::{Column, HttpDataset};
use dataloader_rs::image::{Chw, Decode, Image, Normalize, PadTo, ToChw, ToFloat, ToRgb};
use dataloader_rs::{Collator, DataLoader, Transform, error::Result};

const TILE: u32 = 224;
/// The large image: 560 x 448 pixels, so 3 x 2 tiles, the last column only 112 pixels wide.
const WIDTH: u32 = 560;
const HEIGHT: u32 = 448;

type Coords = (u64, u64);

/// The tile server: `/tile?x=<x>&y=<y>` returns the PNG tile whose top-left corner is (x, y).
fn tile(path: &str) -> Option<Vec<u8>> {
    let query = path.strip_prefix("/tile?")?;
    let mut x = None;
    let mut y = None;
    for pair in query.split('&') {
        match pair.split_once('=')? {
            ("x", value) => x = value.parse::<u32>().ok(),
            ("y", value) => y = value.parse::<u32>().ok(),
            _ => {}
        }
    }
    let (x, y) = (x?, y?);
    let width = TILE.min(WIDTH.checked_sub(x)?);
    let height = TILE.min(HEIGHT.checked_sub(y)?);
    let pixels = image::RgbImage::from_fn(width, height, |dx, dy| {
        image::Rgb([((x + dx) / 3) as u8, ((y + dy) / 2) as u8, 128])
    });
    let mut png = Vec::new();
    pixels
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .ok()?;
    Some(png)
}

/// Collates `(image, coordinates)` items: the images into one `ImageBatch` buffer, the
/// coordinates into a `Vec`, in the same order.
struct TilesWithCoords(ImageCollator);

impl Collator<(Image<f32, Chw>, Coords)> for TilesWithCoords {
    type Batch = (ImageBatch<f32>, Vec<Coords>);

    fn collate(&self, items: Vec<(Image<f32, Chw>, Coords)>) -> Result<Self::Batch> {
        let (images, coords): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((self.0.collate(images)?, coords))
    }
}

fn main() -> Result<()> {
    // The tile server wants a token, as most do.
    let addr = local_server::start(|request| {
        if request.header("Authorization") != Some("Bearer example-token") {
            return (401, b"missing token".to_vec());
        }
        match tile(&request.path) {
            Some(png) => (200, png),
            None => (404, b"no such tile".to_vec()),
        }
    });

    // The top-left corner of every tile, row by row.
    let coords: Vec<Coords> = (0..HEIGHT)
        .step_by(TILE as usize)
        .flat_map(|y| {
            (0..WIDTH)
                .step_by(TILE as usize)
                .map(move |x| (x as u64, y as u64))
        })
        .collect();
    let xs = Column::from(coords.iter().map(|c| c.0).collect::<Vec<_>>());
    let ys = Column::from(coords.iter().map(|c| c.1).collect::<Vec<_>>());

    // Item `i` is the tile at `coords[i]`. The URLs are made from the template, one at a
    // time, when each item is fetched: no list of URLs is kept in memory.
    let dataset = HttpDataset::builder()
        .header("Authorization", "Bearer example-token")
        .url_template(
            &format!("http://{addr}/tile?x={{x}}&y={{y}}"),
            [("x", xs), ("y", ys)],
        )?;

    // The transform gets each item with its index, whatever order the sampler chose, so it
    // can look up the item's coordinates and return them with the image.
    let decode = Decode::new()
        .then(ToRgb)
        .then(PadTo::new(TILE as usize, TILE as usize, 0.0)) // edge tiles padded at the right
        .then(ToFloat::default())
        .then(Normalize::imagenet())
        .then(ToChw);
    let coords = Arc::new(coords);
    let lookup = Arc::clone(&coords);
    let transform = move |index: usize, png: Bytes| -> Result<(Image<f32, Chw>, Coords)> {
        Ok((decode.apply(index, png)?, lookup[index]))
    };

    let mut loader = DataLoader::builder_async(dataset)
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(4) // 8 tile requests in flight
        .transform(transform)
        .collator(TilesWithCoords(ImageCollator::default()))
        .build();

    for batch in &mut loader {
        let (images, tile_coords) = batch?;
        // A model would run on `images` here; the coordinates say where each result belongs.
        println!("images {:?} at {tile_coords:?}", images.shape());
    }
    println!("{} tiles in total", coords.len());
    Ok(())
}
