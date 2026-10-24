//! Shared datasets and collators used by all bench files.
#![allow(dead_code)]

use dataloader_rs::{Dataset, collator::Collator, error::Result};

// ─────────────────────────────────────────────────────────────────────────────
// Datasets
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-cost in-memory dataset: returns the index as u64. O(1) per item.
pub struct InMemoryDs(pub usize);

impl Dataset for InMemoryDs {
    type Item = u64;
    fn get(&self, index: usize) -> Result<u64> {
        Ok(index as u64)
    }
    fn len(&self) -> usize {
        self.0
    }
}

/// Work units per `HeavyCpuDs` item: ≈ 110 µs on the reference laptop.
pub const ITEM_WORK: u64 = 100_000;

/// CPU-bound dataset: [`spin`]s `ITEM_WORK` steps per item (≈ 110 µs), like
/// decoding or augmenting a small sample. Returns a 256-byte Vec<u8>.
pub struct HeavyCpuDs(pub usize);

impl Dataset for HeavyCpuDs {
    type Item = Vec<u8>;
    fn get(&self, index: usize) -> Result<Vec<u8>> {
        let acc = spin(ITEM_WORK) ^ index as u64;
        Ok(vec![(acc & 0xff) as u8; 256])
    }
    fn len(&self) -> usize {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collators
// ─────────────────────────────────────────────────────────────────────────────

/// Sums all u64 items into a single value — trivially cheap.
pub struct SumCollator;

impl Collator<u64> for SumCollator {
    type Batch = u64;
    fn collate(&self, items: Vec<u64>) -> Result<u64> {
        Ok(items.iter().sum())
    }
}

/// Concatenates Vec<u8> items into one flat buffer.
pub struct CatCollator;

impl Collator<Vec<u8>> for CatCollator {
    type Batch = Vec<u8>;
    fn collate(&self, items: Vec<Vec<u8>>) -> Result<Vec<u8>> {
        Ok(items.into_iter().flatten().collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Consumer work
// ─────────────────────────────────────────────────────────────────────────────

/// Burn CPU on the calling thread: `iterations` steps of a xorshift-multiply
/// chain, ≈ 1.1 ns each on the reference laptop. Each step depends on the
/// previous one through operations that do not reassociate, so the compiler
/// can neither shorten nor vectorise the chain (it does both to an LCG).
/// Also stands in for the work a training loop does per batch.
pub fn spin(iterations: u64) -> u64 {
    let mut acc = std::hint::black_box(iterations) | 1;
    for _ in 0..iterations {
        acc ^= acc >> 29;
        acc = acc.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    std::hint::black_box(acc)
}
