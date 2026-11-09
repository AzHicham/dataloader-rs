//! The basics: a small in-memory dataset, a shuffled loader with worker threads, a custom
//! collator, the number of batches, and stopping an epoch early.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example basic
//! ```

use dataloader_rs::collator::Collator;
use dataloader_rs::{DataLoader, Dataset, RandomSampler, SequentialSampler, error::Result};

/// A dataset of 100 numbers: item `i` is `i²`. `get` is called from the loader's worker
/// threads, so a dataset must be `Send + Sync`; this one holds no state at all.
struct Squares {
    len: usize,
}

impl Dataset for Squares {
    type Item = u64;

    fn get(&self, index: usize) -> Result<u64> {
        Ok((index as u64).pow(2))
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// A collator decides what a batch is. This one adds up the items of the batch.
struct Sum;

impl Collator<u64> for Sum {
    type Batch = u64;

    fn collate(&self, items: Vec<u64>) -> Result<u64> {
        Ok(items.into_iter().sum())
    }
}

fn main() -> Result<()> {
    // 1. Shuffled batches, loaded by 4 worker threads. The default collator, `VecCollator`,
    //    returns each batch as a `Vec` of its items.
    println!("Shuffled batches of 10, 4 workers:");
    let mut loader = DataLoader::builder(Squares { len: 100 })
        .batch_size(10)
        .sampler(RandomSampler::new(42)) // a new order every epoch, the same for every run
        .num_workers(4) // 4 batches are loaded in parallel
        .prefetch_depth(4) // and up to 4 more may wait, ready, for the loop below
        .drop_last(true) // a last batch smaller than 10 would be dropped (there is none here)
        .build();
    for epoch in 0..2 {
        // Iterating over the loader again starts a new epoch, in a new order.
        let first: Vec<u64> = loader.iter().next().expect("at least one batch")?;
        println!("  epoch {epoch}: first batch {first:?}");
    }

    // 2. Batches in order, summed by the custom collator, without worker threads: with
    //    `num_workers(0)` (the default) each batch is loaded inside the loop's own `next()`.
    println!("Sums of each batch of 10, in order:");
    let mut sums = DataLoader::builder(Squares { len: 100 })
        .batch_size(10)
        .sampler(SequentialSampler)
        .collator(Sum)
        .build();
    let all: Vec<u64> = sums.iter().collect::<Result<_>>()?;
    println!("  {all:?}");

    // 3. The number of batches is known before the epoch starts.
    let mut uneven = DataLoader::builder(Squares { len: 95 })
        .batch_size(10)
        .build();
    let iter = uneven.iter();
    println!(
        "95 items in batches of 10: {} batches, the last one has 5 items",
        iter.len()
    );
    drop(iter);

    // 4. Leaving an epoch early is fine: the batches still being loaded are discarded in the
    //    background, and the loader can start the next epoch at once.
    let mut early = DataLoader::builder(Squares { len: 100 })
        .batch_size(10)
        .num_workers(2)
        .build();
    for (count, batch) in early.iter().enumerate() {
        batch?;
        if count == 2 {
            println!("Stopped after 3 batches; the next epoch starts at once:");
            break;
        }
    }
    println!("  {} batches in the next epoch", early.iter().count());
    Ok(())
}
