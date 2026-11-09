//! Tiles of a large image, fetched over HTTP by their (x, y) coordinates, decoded and batched;
//! each batch comes with the indices of its tiles, which give their coordinates.
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

use dataloader_rs::collator::ImageCollator;
use dataloader_rs::dataset::{Column, HttpDataset};
use dataloader_rs::image::{Decode, Normalize, PadTo, ToChw, ToFloat, ToRgb};
use dataloader_rs::{DataLoader, Transform, error::Result};

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

    let decode = Decode::new()
        .then(ToRgb)
        .then(PadTo::new(TILE as usize, TILE as usize, 0.0)) // edge tiles padded at the right
        .then(ToFloat::default())
        .then(Normalize::imagenet())
        .then(ToChw);

    let mut loader = DataLoader::builder_async(dataset)
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(4) // 8 tile requests in flight
        .transform(decode)
        .collator(ImageCollator::default())
        .build();

    // Each batch comes with the dataset indices of its tiles, in the order the sampler drew
    // them, so the coordinates are looked up here: nothing is carried through the transform.
    for batch in loader.iter_with_indices() {
        let (indices, images) = batch?;
        let tile_coords: Vec<Coords> = indices.iter().map(|&i| coords[i]).collect();
        // A model would run on `images` here; the coordinates say where each result belongs.
        println!("images {:?} at {tile_coords:?}", images.shape());
    }
    println!("{} tiles in total", coords.len());
    Ok(())
}
