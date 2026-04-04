//! Shared datasets and collators used by all bench files.

use dataloader_rs::{collator::Collator, error::Result, Dataset};

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

/// CPU-heavy dataset that burns ~50_000 LCG iterations per item (~1ms range on
/// typical hardware). Returns a 256-byte Vec<u8>.
pub struct HeavyCpuDs(pub usize);

impl Dataset for HeavyCpuDs {
    type Item = Vec<u8>;
    fn get(&self, index: usize) -> Result<Vec<u8>> {
        let mut acc: u64 = index as u64;
        for i in 0..50_000u64 {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        Ok(vec![(acc & 0xff) as u8; 256])
    }
    fn len(&self) -> usize {
        self.0
    }
}

/// Lightweight CPU dataset: 1_000 LCG iterations per item (~µs range).
/// Used in prefetch-depth benchmarks where pipeline fill matters more than
/// per-item cost.
pub struct LightCpuDs(pub usize);

impl Dataset for LightCpuDs {
    type Item = Vec<u8>;
    fn get(&self, index: usize) -> Result<Vec<u8>> {
        let mut acc: u64 = index as u64;
        for i in 0..1_000u64 {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
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
