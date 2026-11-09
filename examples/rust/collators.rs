//! Collators: how the items of a batch become one batch value.
//!
//! - `VecCollator` (the default) keeps the items as a `Vec`.
//! - `DefaultCollator` works like PyTorch's `default_collate`: it turns a batch of tuples into
//!   a tuple of columns, recursively (arrays, vectors and maps too).
//! - A custom `Collator` can build any batch type: here, a small struct of flat columns.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example collators
//! ```

use dataloader_rs::collator::{Collator, DefaultCollator};
use dataloader_rs::{DataLoader, Dataset, error::Result};

/// A labelled point: `(label, [x, y])`.
type Point = (u32, [f32; 2]);

/// Ten points; the label says whether the index is odd.
struct Points;

impl Dataset for Points {
    type Item = Point;

    fn get(&self, index: usize) -> Result<Point> {
        let x = index as f32;
        Ok((index as u32 % 2, [x, x * x]))
    }

    fn len(&self) -> usize {
        10
    }
}

/// A batch in the shape a model wants: one vector of labels, and the features of every point
/// one after the other (`[x0, y0, x1, y1, ...]`), with their shape.
#[derive(Debug)]
struct Features {
    labels: Vec<u32>,
    values: Vec<f32>,
    shape: [usize; 2],
}

/// Builds a `Features` batch. A collator is shared by every worker thread, so `collate`
/// takes `&self`; an error it returns fails only its own batch.
struct ToFeatures;

impl Collator<Point> for ToFeatures {
    type Batch = Features;

    fn collate(&self, items: Vec<Point>) -> Result<Features> {
        let shape = [items.len(), 2];
        let mut labels = Vec::with_capacity(items.len());
        let mut values = Vec::with_capacity(items.len() * 2);
        for (label, point) in items {
            labels.push(label);
            values.extend(point);
        }
        Ok(Features {
            labels,
            values,
            shape,
        })
    }
}

fn main() -> Result<()> {
    // The default collator: a batch is the `Vec` of its items.
    let mut loader = DataLoader::builder(Points).batch_size(4).build();
    let batch: Vec<Point> = loader.iter().next().expect("a batch")?;
    println!("VecCollator:     {batch:?}");

    // DefaultCollator: a batch of `(label, [x, y])` tuples becomes `(labels, [xs, ys])`.
    let mut loader = DataLoader::builder(Points)
        .batch_size(4)
        .collator(DefaultCollator)
        .build();
    let (labels, [xs, ys]) = loader.iter().next().expect("a batch")?;
    println!("DefaultCollator: labels {labels:?}, xs {xs:?}, ys {ys:?}");

    // A custom collator, with two worker threads: each batch is collated on the thread that
    // loaded it, and batches still come out in order.
    let mut loader = DataLoader::builder(Points)
        .batch_size(4)
        .num_workers(2)
        .collator(ToFeatures)
        .build();
    for batch in &mut loader {
        let batch = batch?;
        println!(
            "ToFeatures:      shape {:?}, labels {:?}, values {:?}",
            batch.shape, batch.labels, batch.values
        );
    }
    Ok(())
}
