//! Basic usage example: square numbers dataset with random batching.
//!
//! Run with:
//!   cargo run --example basic

use dataloader_rs::{DataLoader, Dataset, RandomSampler, SequentialSampler, error::Result};

// ── Dataset implementation ────────────────────────────────────────────────────

/// A trivial dataset that returns index² for each index.
struct SquaresDataset {
    len: usize,
}

impl Dataset for SquaresDataset {
    type Item = u64;

    fn get(&self, index: usize) -> Result<u64> {
        Ok((index as u64).pow(2))
    }

    fn len(&self) -> usize {
        self.len
    }
}

// ── Custom collator example ───────────────────────────────────────────────────

use dataloader_rs::collator::Collator;

/// Sums all values in a batch into a single u64.
struct SumCollator;

impl Collator<u64> for SumCollator {
    type Batch = u64;

    fn collate(&self, items: Vec<u64>) -> Result<u64> {
        Ok(items.into_iter().sum())
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let n = 100;
    let batch_size = 10;

    // ── Example 1: random batches, VecCollator (default) ─────────────────────
    println!("=== Random batches (VecCollator) ===");

    let mut loader = DataLoader::builder(SquaresDataset { len: n })
        .batch_size(batch_size)
        .sampler(RandomSampler::new(42))
        .num_workers(4)
        .prefetch_depth(4)
        .drop_last(true)
        .build();

    for (i, batch) in (&mut loader).into_iter().enumerate() {
        let batch: Vec<u64> = batch.expect("batch error");
        println!(
            "  batch {:02}: len={} first={} last={}",
            i,
            batch.len(),
            batch[0],
            batch.last().unwrap()
        );
    }

    // ── Example 2: sequential batches, custom SumCollator ────────────────────
    println!("\n=== Sequential batches (SumCollator) ===");

    let mut loader2 = DataLoader::builder(SquaresDataset { len: n })
        .batch_size(batch_size)
        .sampler(SequentialSampler)
        .collator(SumCollator)
        .num_workers(0) // sequential loading
        .build();

    // Expected: sum of i² for i in [0,10), [10,20), … [90,100)
    // Batch k covers [10k, 10k+10): Σ i² = Σ (10k+j)² for j in 0..10
    for (i, batch) in (&mut loader2).into_iter().enumerate() {
        let sum: u64 = batch.expect("batch error");
        println!("  batch {:02}: sum of squares = {}", i, sum);
    }

    // ── Example 3: ExactSizeIterator ─────────────────────────────────────────
    println!("\n=== ExactSizeIterator ===");

    let mut loader3 = DataLoader::builder(SquaresDataset { len: n })
        .batch_size(batch_size)
        .drop_last(false)
        .build();

    let iter = loader3.iter();
    println!("  expecting {} batches", iter.len());
    let collected: Vec<_> = iter.collect();
    println!("  received  {} batches", collected.len());

    // ── Example 4: early stop (drop mid-epoch) ────────────────────────────────
    println!("\n=== Early stop (prefetch thread exits cleanly) ===");

    let mut loader4 = DataLoader::builder(SquaresDataset { len: n })
        .batch_size(batch_size)
        .sampler(RandomSampler::from_entropy())
        .num_workers(2)
        .prefetch_depth(4)
        .build();

    let mut count = 0;
    for batch in &mut loader4 {
        let _ = batch.unwrap();
        count += 1;
        if count == 3 {
            println!("  broke after 3 batches — no hang, no leak");
            break;
        }
    }
}
