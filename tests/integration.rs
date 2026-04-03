//! Integration tests — exercise the full public API end-to-end.
//!
//! These tests are compiled as a separate crate, so they can only access items
//! that are exported from the library's public surface.

use dataloader_rs::{
    collator::Collator,
    error::Result,
    sampler::{DistributedSampler, RandomSampler, SequentialSampler},
    DataLoader, Dataset,
};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

// ── Shared test datasets ──────────────────────────────────────────────────────

struct RangeDs(usize);

impl Dataset for RangeDs {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        Ok(index)
    }
    fn len(&self) -> usize {
        self.0
    }
}

struct SquareDs(usize);

impl Dataset for SquareDs {
    type Item = u64;
    fn get(&self, index: usize) -> Result<u64> {
        Ok((index as u64).pow(2))
    }
    fn len(&self) -> usize {
        self.0
    }
}

/// Signals a flag before sleeping, letting tests detect concurrency.
struct SlowDs {
    len: usize,
    was_concurrent: Arc<AtomicBool>,
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
}

impl Dataset for SlowDs {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        let prev = self.in_flight.fetch_add(1, Ordering::SeqCst);
        if prev > 0 {
            self.was_concurrent.store(true, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_micros(200));
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(index)
    }
    fn len(&self) -> usize {
        self.len
    }
}

// ── Basic pipeline ────────────────────────────────────────────────────────────

#[test]
fn full_pipeline_sequential() {
    let mut loader = DataLoader::builder(RangeDs(10))
        .batch_size(3)
        .build();

    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();

    assert_eq!(batches.len(), 4);
    assert_eq!(batches[0], vec![0, 1, 2]);
    assert_eq!(batches[1], vec![3, 4, 5]);
    assert_eq!(batches[2], vec![6, 7, 8]);
    assert_eq!(batches[3], vec![9]); // partial last batch
}

#[test]
fn full_pipeline_random_sampler() {
    let n = 30;
    let mut loader = DataLoader::builder(RangeDs(n))
        .batch_size(5)
        .sampler(RandomSampler::new(0))
        .build();

    let items: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    // Every index is seen exactly once.
    assert_eq!(items.len(), n);
    assert_eq!(*items.iter().max().unwrap(), n - 1);
}

#[test]
fn for_loop_sugar() {
    let mut loader = DataLoader::builder(RangeDs(8))
        .batch_size(4)
        .build();

    let mut count = 0usize;
    for batch in &mut loader {
        let batch = batch.unwrap();
        assert_eq!(batch.len(), 4);
        count += 1;
    }
    assert_eq!(count, 2);
}

// ── drop_last semantics ───────────────────────────────────────────────────────

#[test]
fn drop_last_removes_partial_batch() {
    let mut loader = DataLoader::builder(RangeDs(11))
        .batch_size(4)
        .drop_last(true)
        .build();

    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    // floor(11/4) = 2 complete batches
    assert_eq!(batches.len(), 2);
    for b in &batches {
        assert_eq!(b.len(), 4);
    }
}

#[test]
fn drop_last_exact_divisor_unchanged() {
    let mut loader = DataLoader::builder(RangeDs(12))
        .batch_size(4)
        .drop_last(true)
        .build();
    assert_eq!(loader.iter().count(), 3);
}

// ── ExactSizeIterator ─────────────────────────────────────────────────────────

#[test]
fn exact_size_iterator() {
    let mut loader = DataLoader::builder(RangeDs(13))
        .batch_size(5)
        .build();

    let iter = loader.iter();
    // ceil(13/5) = 3
    assert_eq!(iter.len(), 3);
    let collected: Vec<_> = iter.collect();
    assert_eq!(collected.len(), 3);
}

// ── Multi-epoch ───────────────────────────────────────────────────────────────

#[test]
fn sequential_sampler_identical_across_epochs() {
    let mut loader = DataLoader::builder(RangeDs(12))
        .batch_size(4)
        .build();

    let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e3: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(e1, e2);
    assert_eq!(e2, e3);
}

#[test]
fn random_sampler_varies_across_epochs() {
    let mut loader = DataLoader::builder(RangeDs(40))
        .batch_size(8)
        .sampler(RandomSampler::new(99))
        .build();

    let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_ne!(e1, e2);

    // But both epochs must cover all indices.
    let seen1: HashSet<usize> = e1.into_iter().flatten().collect();
    let seen2: HashSet<usize> = e2.into_iter().flatten().collect();
    assert_eq!(seen1.len(), 40);
    assert_eq!(seen2.len(), 40);
}

// ── Parallel prefetch ─────────────────────────────────────────────────────────

#[test]
fn parallel_prefetch_same_results_as_sequential() {
    let n = 24;
    let bs = 6;

    let mut seq = DataLoader::builder(SquareDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(42))
        .build();

    let mut par = DataLoader::builder(SquareDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(42)) // same seed → same index order
        .num_workers(4)
        .prefetch_depth(4)
        .build();

    let seq_out: Vec<Vec<u64>> = seq.iter().map(|b| b.unwrap()).collect();
    let par_out: Vec<Vec<u64>> = par.iter().map(|b| b.unwrap()).collect();
    assert_eq!(seq_out, par_out);
}

#[test]
fn parallel_prefetch_is_actually_concurrent() {
    let was_concurrent = Arc::new(AtomicBool::new(false));
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let ds = SlowDs {
        len: 20,
        was_concurrent: Arc::clone(&was_concurrent),
        in_flight: Arc::clone(&in_flight),
    };

    let mut loader = DataLoader::builder(ds)
        .batch_size(10) // 10 items per batch → 10 concurrent get() possible
        .num_workers(4)
        .prefetch_depth(2)
        .build();

    for batch in &mut loader {
        batch.unwrap();
    }

    assert!(
        was_concurrent.load(Ordering::Relaxed),
        "expected parallel get() calls but got sequential execution"
    );
}

// ── Early drop / cancellation ─────────────────────────────────────────────────

#[test]
fn early_drop_no_hang() {
    // If there's a deadlock in the shutdown path, this test hangs.
    // The test runner will eventually time it out, which is also a clear signal.
    let mut loader = DataLoader::builder(RangeDs(100))
        .batch_size(5)
        .prefetch_depth(2)
        .num_workers(2)
        .build();

    {
        let mut iter = loader.iter();
        iter.next(); // prime the prefetch buffer
        // Drop iter with the prefetch thread potentially blocked on send.
    }

    // Must be able to start a new epoch after early drop.
    let count = loader.iter().count();
    assert_eq!(count, 20); // 100/5
}

#[test]
fn break_then_new_epoch_correct() {
    let n = 30;
    let bs = 5;
    let mut loader = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(7))
        .num_workers(2)
        .prefetch_depth(3)
        .build();

    // First epoch: consume only 2 batches.
    for (i, b) in (&mut loader).into_iter().enumerate() {
        b.unwrap();
        if i == 1 {
            break;
        }
    }

    // Second epoch: must yield all n items across all batches.
    let items: HashSet<usize> = loader.iter()
        .flat_map(|b| b.unwrap())
        .collect();
    assert_eq!(items.len(), n);
}

// ── Custom collator ───────────────────────────────────────────────────────────

/// Flattens Vec<Vec<u8>> into a single Vec<u8>.
struct FlattenCollator;

impl Collator<Vec<u8>> for FlattenCollator {
    type Batch = Vec<u8>;
    fn collate(&self, items: Vec<Vec<u8>>) -> Result<Vec<u8>> {
        Ok(items.into_iter().flatten().collect())
    }
}

struct ChunkDs(usize);

impl Dataset for ChunkDs {
    type Item = Vec<u8>;
    fn get(&self, index: usize) -> Result<Vec<u8>> {
        Ok(vec![index as u8; 4])
    }
    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn custom_collator_integration() {
    let mut loader = DataLoader::builder(ChunkDs(4))
        .batch_size(2)
        .collator(FlattenCollator)
        .build();

    let batches: Vec<Vec<u8>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 2);
    // Each batch = 2 items × 4 bytes = 8 bytes.
    assert_eq!(batches[0].len(), 8);
    assert_eq!(batches[1].len(), 8);
}

// ── DistributedSampler ────────────────────────────────────────────────────────

#[test]
fn distributed_sampler_disjoint_ranks() {
    let n = 20;
    let world_size = 4;

    // Collect all items seen by all ranks in one epoch.
    let mut all_items: Vec<usize> = Vec::new();

    for rank in 0..world_size {
        let mut loader = DataLoader::builder(RangeDs(n))
            .batch_size(2)
            .sampler(DistributedSampler::new(SequentialSampler, rank, world_size))
            .build();

        let epoch_items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        // Each rank gets n / world_size = 5 items.
        assert_eq!(epoch_items.len(), n / world_size);
        all_items.extend(epoch_items);
    }

    // Together all ranks cover the full range (20 items, no duplicates in the
    // non-padded case where n is divisible by world_size).
    assert_eq!(all_items.len(), n);
    let unique: HashSet<usize> = all_items.into_iter().collect();
    assert_eq!(unique.len(), n);
}

// ── prefetch_depth backpressure ───────────────────────────────────────────────

#[test]
fn prefetch_depth_one_does_not_over_produce() {
    // With prefetch_depth=1 the prefetch thread is never more than 1 batch
    // ahead. Verify iteration still completes correctly.
    let mut loader = DataLoader::builder(RangeDs(20))
        .batch_size(4)
        .prefetch_depth(1)
        .build();

    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 5);
}

// ── Error propagation ─────────────────────────────────────────────────────────

struct PartialErrDs {
    fail_at: usize,
    len: usize,
}

impl Dataset for PartialErrDs {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        if index == self.fail_at {
            Err(format!("fail at {index}").into())
        } else {
            Ok(index)
        }
    }
    fn len(&self) -> usize {
        self.len
    }
}

#[test]
fn error_in_batch_propagates_as_item() {
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
        .batch_size(4)
        .build();

    let results: Vec<_> = loader.iter().collect();
    // First batch: indices [0,1,2,3] — index 3 fails → Err
    // Second batch: indices [4,5,6,7] — all ok → Ok
    assert!(results[0].is_err(), "expected first batch to fail");
    assert!(results[1].is_ok(), "expected second batch to succeed");
}

#[test]
fn loader_reusable_after_error_epoch() {
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 0, len: 6 })
        .batch_size(3)
        .build();

    // First epoch: consume (with errors).
    let _ = loader.iter().count();

    // Second epoch: must also complete without panic.
    let _ = loader.iter().count();
}
