//! Integration tests — exhaustive coverage of the full public API.
//!
//! Tests are compiled as a separate crate so they can only access items that
//! are exported from the library's public surface.
//!
//! Organisation:
//!   - Shared dataset / collator definitions
//!   - Sequential path (num_workers=0, intra_workers=0)
//!   - Parallel inter-batch path (num_workers > 0)
//!   - Intra-batch parallelism (intra_workers > 0)
//!   - Sampler tests
//!   - Error propagation
//!   - Edge cases

use dataloader_rs::{
    collator::Collator,
    error::Result,
    sampler::{DistributedSampler, RandomSampler, SequentialSampler},
    DataLoader, Dataset,
};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

// ── Shared datasets ───────────────────────────────────────────────────────────

/// Returns `index` as its value; never fails.
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

/// Every call to `get` returns an error.
struct AlwaysErrDs(usize);

impl Dataset for AlwaysErrDs {
    type Item = usize;
    fn get(&self, _: usize) -> Result<usize> {
        Err("always fails".into())
    }
    fn len(&self) -> usize {
        self.0
    }
}

/// Fails at a specific index; succeeds everywhere else.
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

/// Tracks the peak number of concurrent `get` calls so tests can assert
/// that multiple items within a batch are actually fetched in parallel.
struct SlowDs {
    len: usize,
    in_flight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl Dataset for SlowDs {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        // Increment BEFORE the sleep so that concurrent calls overlap.
        let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(current, Ordering::SeqCst);
        thread::sleep(Duration::from_micros(200));
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(index)
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// Signals `was_concurrent` as soon as more than one `get` call overlaps.
/// Lighter alternative to `SlowDs` when we only need a boolean signal.
struct ConcurrentDetectorDs {
    len: usize,
    in_flight: Arc<AtomicUsize>,
    was_concurrent: Arc<AtomicBool>,
}

impl Dataset for ConcurrentDetectorDs {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        let prev = self.in_flight.fetch_add(1, Ordering::SeqCst);
        if prev > 0 {
            self.was_concurrent.store(true, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_micros(300));
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(index)
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// Counts calls to `get_batch` (the batch-level override) separately from
/// `get` (the item-level default) to allow asserting which code path is taken.
struct BatchOverrideDs {
    len: usize,
    get_calls: Arc<AtomicUsize>,
    get_batch_calls: Arc<AtomicUsize>,
}

impl Dataset for BatchOverrideDs {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        Ok(index)
    }
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<usize>> {
        self.get_batch_calls.fetch_add(1, Ordering::Relaxed);
        indices.iter().map(|&i| self.get(i)).collect()
    }
    fn len(&self) -> usize {
        self.len
    }
}

// ── Shared collators ──────────────────────────────────────────────────────────

/// Sums all items in a batch into a single `usize`.
struct SumCollator;

impl Collator<usize> for SumCollator {
    type Batch = usize;
    fn collate(&self, items: Vec<usize>) -> Result<usize> {
        Ok(items.iter().sum())
    }
}

/// Always returns an error from `collate`.
struct ErrCollator;

impl Collator<usize> for ErrCollator {
    type Batch = usize;
    fn collate(&self, _items: Vec<usize>) -> Result<usize> {
        Err("collate always fails".into())
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Sequential path (num_workers=0, intra_workers=0)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn seq_all_items_covered_small() {
    // Verifies every index [0, N) appears exactly once.  N=10, bs=3.
    let mut loader = DataLoader::builder(RangeDs(10)).batch_size(3).build();
    let mut items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    items.sort_unstable();
    assert_eq!(items, (0..10).collect::<Vec<_>>());
}

#[test]
fn seq_all_items_covered_large() {
    // Large N with a prime batch size to exercise every remainder case.
    // N=1000, bs=17 → ceil(1000/17)=59 batches (last batch has 15 items).
    let n = 1000usize;
    let mut loader = DataLoader::builder(RangeDs(n)).batch_size(17).build();
    let mut items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    items.sort_unstable();
    assert_eq!(items, (0..n).collect::<Vec<_>>());
}

#[test]
fn seq_exact_batches_no_drop_last() {
    // N=10, bs=3 → floor(10/3)=3 full batches + 1 partial → 4 total.
    let mut loader = DataLoader::builder(RangeDs(10)).batch_size(3).build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 4);
    assert_eq!(batches[3], vec![9]); // partial last batch
}

#[test]
fn seq_exact_batches_drop_last() {
    // N=10, bs=3, drop_last=true → only 3 complete batches.
    let mut loader = DataLoader::builder(RangeDs(10))
        .batch_size(3)
        .drop_last(true)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 3);
    for b in &batches {
        assert_eq!(b.len(), 3, "every batch must be full when drop_last=true");
    }
}

#[test]
fn seq_single_batch_when_bs_equals_n() {
    // Edge case: bs == N → exactly one batch containing all items.
    let mut loader = DataLoader::builder(RangeDs(6)).batch_size(6).build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0], vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn seq_bs_larger_than_n() {
    // Edge case: bs > N, drop_last=false → 1 partial batch with all N items.
    let mut loader = DataLoader::builder(RangeDs(3))
        .batch_size(10)
        .drop_last(false)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 1, "should get exactly one partial batch");
    assert_eq!(batches[0], vec![0, 1, 2]);
}

#[test]
fn seq_bs_larger_than_n_drop_last() {
    // Edge case: bs > N, drop_last=true → 0 batches because the only batch
    // would be partial and is therefore dropped.
    let mut loader = DataLoader::builder(RangeDs(3))
        .batch_size(10)
        .drop_last(true)
        .build();
    assert_eq!(
        loader.iter().count(),
        0,
        "drop_last=true with bs>N must yield no batches"
    );
}

#[test]
fn seq_empty_dataset() {
    // Edge case: N=0 → iterator is immediately exhausted, 0 batches.
    let mut loader = DataLoader::builder(RangeDs(0)).batch_size(4).build();
    assert_eq!(loader.iter().count(), 0);
}

#[test]
fn seq_single_item_dataset() {
    // N=1, bs=1 → exactly 1 batch containing [0].
    let mut loader = DataLoader::builder(RangeDs(1)).batch_size(1).build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0], vec![0]);
}

#[test]
fn seq_batch_size_one() {
    // N=5, bs=1 → 5 batches, each of length 1, in order [0]...[4].
    let mut loader = DataLoader::builder(RangeDs(5)).batch_size(1).build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 5);
    for (i, batch) in batches.iter().enumerate() {
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], i);
    }
}

#[test]
fn seq_for_loop_sugar() {
    // Verifies `for batch in &mut loader` syntax works correctly.
    let mut loader = DataLoader::builder(RangeDs(8)).batch_size(4).build();
    let mut count = 0usize;
    for batch in &mut loader {
        let batch = batch.unwrap();
        assert_eq!(batch.len(), 4);
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn seq_reusable_across_epochs() {
    // The same loader must produce identical results across multiple epochs.
    let mut loader = DataLoader::builder(RangeDs(12)).batch_size(4).build();
    let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    for _ in 0..4 {
        let en: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(e1, en, "every epoch must be identical for SequentialSampler");
    }
}

#[test]
fn seq_exact_size_iterator() {
    // ExactSizeIterator::len() must report the correct remaining batch count
    // and must decrement by 1 for each consumed batch.
    let mut loader = DataLoader::builder(RangeDs(13)).batch_size(5).build();
    // ceil(13/5) = 3
    let mut iter = loader.iter();
    assert_eq!(iter.len(), 3);
    iter.next().unwrap().unwrap();
    assert_eq!(iter.len(), 2);
    iter.next().unwrap().unwrap();
    assert_eq!(iter.len(), 1);
    iter.next().unwrap().unwrap();
    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none(), "iterator should be exhausted");
}

#[test]
fn seq_custom_collator_sum() {
    // A custom collator that sums items must produce the correct aggregated
    // values: batch 0 = 0+1+2=3, batch 1 = 3+4+5=12.
    let mut loader = DataLoader::builder(RangeDs(6))
        .batch_size(3)
        .collator(SumCollator)
        .build();
    let sums: Vec<usize> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(sums, vec![3, 12]);
}

#[test]
fn seq_collator_error_propagates() {
    // When the collator returns Err the batch must surface as Err in iteration.
    let mut loader = DataLoader::builder(RangeDs(6))
        .batch_size(3)
        .collator(ErrCollator)
        .build();
    for result in loader.iter() {
        assert!(result.is_err(), "collator error must propagate as Err batch");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Parallel inter-batch path (num_workers > 0)
// ═════════════════════════════════════════════════════════════════════════════

/// Build a sequential loader with the given random seed.
macro_rules! seq_loader {
    ($n:expr, $bs:expr, $seed:expr) => {
        DataLoader::builder(RangeDs($n))
            .batch_size($bs)
            .sampler(RandomSampler::new($seed))
            .build()
    };
}

/// Build a parallel loader with the given random seed and worker count.
macro_rules! par_loader {
    ($n:expr, $bs:expr, $seed:expr, $workers:expr) => {
        DataLoader::builder(RangeDs($n))
            .batch_size($bs)
            .sampler(RandomSampler::new($seed))
            .num_workers($workers)
            .prefetch_depth($workers.max(1))
            .build()
    };
}

#[test]
fn par_same_items_as_seq_2w() {
    // Parallel loader with 2 workers must cover the exact same items as the
    // sequential path when given the same random seed.
    let seed = 42;
    let mut seq = seq_loader!(20, 4, seed);
    let mut par = par_loader!(20, 4, seed, 2);

    let seq_items: HashSet<usize> = seq.iter().flat_map(|b| b.unwrap()).collect();
    let par_items: HashSet<usize> = par.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seq_items, par_items);
}

#[test]
fn par_same_items_as_seq_4w() {
    let seed = 7;
    let mut seq = seq_loader!(24, 4, seed);
    let mut par = par_loader!(24, 4, seed, 4);

    let seq_items: HashSet<usize> = seq.iter().flat_map(|b| b.unwrap()).collect();
    let par_items: HashSet<usize> = par.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seq_items, par_items);
}

#[test]
fn par_same_items_as_seq_8w() {
    let seed = 99;
    let mut seq = seq_loader!(40, 5, seed);
    let mut par = par_loader!(40, 5, seed, 8);

    let seq_items: HashSet<usize> = seq.iter().flat_map(|b| b.unwrap()).collect();
    let par_items: HashSet<usize> = par.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seq_items, par_items);
}

#[test]
fn par_same_items_as_seq_more_workers_than_batches() {
    // Edge case: num_workers=20, N=10, bs=2 → 5 batches, 15 workers sit idle.
    // Result must still be all 10 items.
    let seed = 1;
    let mut seq = seq_loader!(10, 2, seed);
    let mut par = par_loader!(10, 2, seed, 20);

    let seq_items: HashSet<usize> = seq.iter().flat_map(|b| b.unwrap()).collect();
    let par_items: HashSet<usize> = par.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seq_items, par_items);
    assert_eq!(par_items.len(), 10);
}

#[test]
fn par_empty_dataset() {
    // Edge case: parallel loader on N=0 must yield 0 batches, no hang.
    let mut loader = DataLoader::builder(RangeDs(0))
        .batch_size(4)
        .num_workers(4)
        .build();
    assert_eq!(loader.iter().count(), 0);
}

#[test]
fn par_single_batch_4w() {
    // N=8, bs=8, num_workers=4 → exactly 1 batch, all 8 items.
    let mut loader = DataLoader::builder(RangeDs(8))
        .batch_size(8)
        .num_workers(4)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 1);
    let mut items = batches.into_iter().flatten().collect::<Vec<_>>();
    items.sort_unstable();
    assert_eq!(items, (0..8).collect::<Vec<_>>());
}

#[test]
fn par_drop_last() {
    // N=11, bs=3, num_workers=4, drop_last=true → floor(11/3)=3 full batches.
    let mut loader = DataLoader::builder(RangeDs(11))
        .batch_size(3)
        .num_workers(4)
        .drop_last(true)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 3);
    for b in &batches {
        assert_eq!(b.len(), 3);
    }
}

#[test]
fn par_reusable_across_epochs() {
    // A parallel loader must be safely re-usable across multiple epochs.
    let mut loader = DataLoader::builder(RangeDs(12))
        .batch_size(4)
        .num_workers(2)
        .build();
    let e1: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    for _ in 0..4 {
        let en: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(e1, en);
    }
}

#[test]
fn par_early_drop_no_hang() {
    // Drop the iterator after consuming just 1 batch.  The shutdown path must
    // not deadlock.  If this test hangs, there is a deadlock in Drop.
    let mut loader = DataLoader::builder(RangeDs(100))
        .batch_size(5)
        .num_workers(2)
        .prefetch_depth(4)
        .build();

    {
        let mut iter = loader.iter();
        iter.next().unwrap().unwrap(); // prime the prefetch buffer
        // iter dropped here with prefetch thread potentially blocking on send
    }

    // Must start a new epoch successfully after early drop.
    let count = loader.iter().count();
    assert_eq!(count, 20); // 100/5
}

#[test]
fn par_early_drop_zero_batches_no_hang() {
    // Drop the iterator immediately without consuming anything.
    let mut loader = DataLoader::builder(RangeDs(50))
        .batch_size(5)
        .num_workers(4)
        .prefetch_depth(4)
        .build();

    {
        let _iter = loader.iter();
        // dropped immediately
    }

    let count = loader.iter().count();
    assert_eq!(count, 10);
}

#[test]
fn par_break_then_full_epoch() {
    // Break after consuming 2 batches, then run a complete epoch.
    let n = 30;
    let bs = 5;
    let mut loader = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .num_workers(2)
        .prefetch_depth(3)
        .build();

    // Partial epoch
    for (i, b) in (&mut loader).into_iter().enumerate() {
        b.unwrap();
        if i == 1 {
            break;
        }
    }

    // Full epoch must still yield all items.
    let items: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(items.len(), n);
}

#[test]
fn par_actually_concurrent() {
    // Verifies that multiple `get()` calls genuinely overlap in time.
    let in_flight = Arc::new(AtomicUsize::new(0));
    let was_concurrent = Arc::new(AtomicBool::new(false));

    let ds = ConcurrentDetectorDs {
        len: 20,
        in_flight: Arc::clone(&in_flight),
        was_concurrent: Arc::clone(&was_concurrent),
    };

    let mut loader = DataLoader::builder(ds)
        .batch_size(10) // 10 items/batch → 10 concurrent get() possible
        .num_workers(4)
        .prefetch_depth(2)
        .build();

    for batch in &mut loader {
        batch.unwrap();
    }

    assert!(
        was_concurrent.load(Ordering::Relaxed),
        "expected concurrent get() calls but execution appeared sequential"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Intra-batch path (intra_workers > 0)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn intra_same_items_as_seq() {
    // inter=2, intra=4 must produce the same item set as the sequential path.
    let seed = 55u64;
    let n = 24;
    let bs = 6;

    let mut seq = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(seed))
        .build();

    let mut par = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(seed))
        .num_workers(2)
        .intra_workers(4)
        .build();

    let seq_items: HashSet<usize> = seq.iter().flat_map(|b| b.unwrap()).collect();
    let par_items: HashSet<usize> = par.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seq_items, par_items);
}

#[test]
fn intra_actually_parallel() {
    // Verify that items within a batch are fetched concurrently by the rayon
    // pool when intra_workers > 0.
    let peak = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));

    let ds = SlowDs {
        len: 16,
        in_flight: Arc::clone(&in_flight),
        peak: Arc::clone(&peak),
    };

    let mut loader = DataLoader::builder(ds)
        .batch_size(8) // 8 items/batch that each sleep briefly
        .num_workers(1)
        .intra_workers(4)
        .build();

    for batch in &mut loader {
        batch.unwrap();
    }

    assert!(
        peak.load(Ordering::SeqCst) > 1,
        "expected concurrent intra-batch get() calls, but peak concurrency was 1"
    );
}

#[test]
fn intra_with_large_batch() {
    // bs=64, intra=8 — verify correctness with a large batch size.
    let n = 128;
    let mut loader = DataLoader::builder(RangeDs(n))
        .batch_size(64)
        .num_workers(2)
        .intra_workers(8)
        .build();
    let mut items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    items.sort_unstable();
    assert_eq!(items, (0..n).collect::<Vec<_>>());
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampler tests
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn sampler_sequential_is_deterministic() {
    // SequentialSampler must produce the same indices every epoch.
    let mut loader = DataLoader::builder(RangeDs(15)).batch_size(5).build();
    let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e3: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(e1, e2);
    assert_eq!(e2, e3);
}

#[test]
fn sampler_random_differs_across_epochs() {
    // RandomSampler must shuffle differently each epoch (statistically certain
    // for N=40 with a decent PRNG).
    let mut loader = DataLoader::builder(RangeDs(40))
        .batch_size(8)
        .sampler(RandomSampler::new(999))
        .build();

    let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    let e3: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    // All three epochs differ from each other.
    assert_ne!(e1, e2, "epoch 1 and 2 should differ");
    assert_ne!(e2, e3, "epoch 2 and 3 should differ");
}

#[test]
fn sampler_random_covers_all_indices() {
    // Every index in [0, N) must appear exactly once per epoch.
    let n = 100;
    let mut loader = DataLoader::builder(RangeDs(n))
        .batch_size(7)
        .sampler(RandomSampler::new(42))
        .build();

    let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
    assert_eq!(seen.len(), n);
    assert_eq!(*seen.iter().min().unwrap(), 0);
    assert_eq!(*seen.iter().max().unwrap(), n - 1);
}

#[test]
fn sampler_random_reproducible_with_same_seed() {
    // Two loaders with the same seed must produce the same first epoch.
    let n = 30;
    let bs = 5;
    let seed = 12345;

    let mut l1 = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(seed))
        .build();
    let mut l2 = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .sampler(RandomSampler::new(seed))
        .build();

    let out1: Vec<Vec<usize>> = l1.iter().map(|b| b.unwrap()).collect();
    let out2: Vec<Vec<usize>> = l2.iter().map(|b| b.unwrap()).collect();
    assert_eq!(out1, out2, "same seed → same first-epoch order");
}

#[test]
fn sampler_distributed_disjoint_coverage() {
    // 4 ranks together must cover all N items with no duplicates.
    let n = 20;
    let world_size = 4;
    let mut all_items: Vec<usize> = Vec::new();

    for rank in 0..world_size {
        let mut loader = DataLoader::builder(RangeDs(n))
            .batch_size(2)
            .sampler(DistributedSampler::new(SequentialSampler, rank, world_size))
            .build();
        let epoch_items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        all_items.extend(epoch_items);
    }

    assert_eq!(all_items.len(), n, "all items across all ranks = N");
    let unique: HashSet<usize> = all_items.into_iter().collect();
    assert_eq!(unique.len(), n, "no index should appear more than once");
}

#[test]
fn sampler_distributed_each_rank_gets_equal_share() {
    // N=20, ws=4 → each rank gets exactly 5 items.
    let n = 20;
    let world_size = 4;
    let expected = n / world_size;

    for rank in 0..world_size {
        let mut loader = DataLoader::builder(RangeDs(n))
            .batch_size(1)
            .sampler(DistributedSampler::new(SequentialSampler, rank, world_size))
            .build();
        let count: usize = loader.iter().flat_map(|b| b.unwrap()).count();
        assert_eq!(count, expected, "rank {rank} must receive exactly {expected} items");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Error propagation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn error_in_first_item_fails_batch() {
    // fail_at=0, bs=4: the very first item in the first batch fails →
    // first batch is Err, second batch ([4..7]) is Ok.
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 0, len: 8 })
        .batch_size(4)
        .build();
    let results: Vec<_> = loader.iter().collect();
    assert!(results[0].is_err(), "first batch must be Err (fail_at=0)");
    assert!(results[1].is_ok(), "second batch must succeed");
}

#[test]
fn error_in_last_item_fails_batch() {
    // fail_at=3, bs=4: the last item of the first batch (index 3) fails →
    // first batch is Err, second batch is Ok.
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
        .batch_size(4)
        .build();
    let results: Vec<_> = loader.iter().collect();
    assert!(results[0].is_err(), "first batch must be Err (fail_at=3)");
    assert!(results[1].is_ok(), "second batch must succeed");
}

#[test]
fn error_propagates_parallel() {
    // With parallel workers, at least one batch must surface as Err.
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 5, len: 16 })
        .batch_size(4)
        .num_workers(4)
        .build();
    let results: Vec<_> = loader.iter().collect();
    let has_err = results.iter().any(|r| r.is_err());
    assert!(has_err, "parallel loader must propagate dataset errors");
}

#[test]
fn error_all_items_fail() {
    // AlwaysErrDs → every batch must be Err.
    let mut loader = DataLoader::builder(AlwaysErrDs(8))
        .batch_size(4)
        .build();
    for result in loader.iter() {
        assert!(result.is_err(), "every batch must be Err when all items fail");
    }
}

#[test]
fn error_loader_reusable_after_error_epoch() {
    // A loader that produced errors in epoch 1 must still work in epoch 2.
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 0, len: 6 })
        .batch_size(3)
        .build();

    // Consume epoch 1 (with errors).
    let _ = loader.iter().count();

    // Epoch 2 must complete without panic and produce the same results.
    let count = loader.iter().count();
    assert_eq!(count, 2, "second epoch must produce the same number of batches");
}

#[test]
fn collator_error_skips_collation() {
    // When dataset.get() fails, the collator must NOT be called for that batch.
    // We count collator calls to verify this invariant.
    use std::sync::atomic::AtomicUsize;

    struct CountingCollator {
        calls: Arc<AtomicUsize>,
    }
    impl Collator<usize> for CountingCollator {
        type Batch = Vec<usize>;
        fn collate(&self, items: Vec<usize>) -> Result<Vec<usize>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(items)
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let collator = CountingCollator { calls: Arc::clone(&calls) };

    // fail_at=3 → first batch (0-3) fails; second batch (4-7) succeeds.
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
        .batch_size(4)
        .collator(collator)
        .build();

    let results: Vec<_> = loader.iter().collect();
    assert_eq!(results.len(), 2);
    assert!(results[0].is_err(), "first batch should be Err");
    assert!(results[1].is_ok(), "second batch should be Ok");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "collate must be called exactly once (only for the successful batch)"
    );
}

#[test]
fn error_middle_batch_others_succeed() {
    // N=12, bs=4, fail_at=5 → batch indices [0-3] ok, [4-7] contains 5 → Err,
    // [8-11] ok.  Exactly 1 error, 2 successes.
    let mut loader = DataLoader::builder(PartialErrDs { fail_at: 5, len: 12 })
        .batch_size(4)
        .build();
    let results: Vec<_> = loader.iter().collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok(), "batch 0 should succeed");
    assert!(results[1].is_err(), "batch 1 (indices 4-7) should fail at index 5");
    assert!(results[2].is_ok(), "batch 2 should succeed");
}

// ═════════════════════════════════════════════════════════════════════════════
// Edge cases
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn prefetch_depth_one() {
    // With prefetch_depth=1 the buffer is tight; iteration must still complete.
    let mut loader = DataLoader::builder(RangeDs(20))
        .batch_size(4)
        .prefetch_depth(1)
        .num_workers(4)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 5);
    let mut items: Vec<usize> = batches.into_iter().flatten().collect();
    items.sort_unstable();
    assert_eq!(items, (0..20).collect::<Vec<_>>());
}

#[test]
fn prefetch_depth_larger_than_batches() {
    // Edge case: depth=1000 >> number of batches.  Results must still be correct
    // and the loader must not panic or hang.
    let mut loader = DataLoader::builder(RangeDs(10))
        .batch_size(2)
        .prefetch_depth(1000)
        .num_workers(4)
        .build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 5);
    let mut items: Vec<usize> = batches.into_iter().flatten().collect();
    items.sort_unstable();
    assert_eq!(items, (0..10).collect::<Vec<_>>());
}

#[test]
fn loader_len_and_batch_len() {
    // `DataLoader::len()` returns the number of samples; `batch_len()` returns
    // the number of batches for one epoch.
    let n = 17;
    let bs = 5;
    let loader = DataLoader::builder(RangeDs(n)).batch_size(bs).build();
    assert_eq!(loader.len(), n, "loader.len() must equal dataset size");
    // ceil(17/5) = 4
    assert_eq!(loader.batch_len(), 4, "batch_len() must equal ceil(N/bs)");

    // With drop_last=true: floor(17/5) = 3
    let loader_dl = DataLoader::builder(RangeDs(n))
        .batch_size(bs)
        .drop_last(true)
        .build();
    assert_eq!(loader_dl.batch_len(), 3);
}

#[test]
fn loader_is_empty() {
    // `is_empty()` must return true iff N=0.
    let empty = DataLoader::builder(RangeDs(0)).build();
    assert!(empty.is_empty());

    let non_empty = DataLoader::builder(RangeDs(1)).build();
    assert!(!non_empty.is_empty());
}

#[test]
fn get_batch_override_is_called() {
    // When the dataset overrides `get_batch`, that method must be called once
    // per batch (not `get` N times from the batch-level logic), while `get` is
    // still called by `get_batch`'s default delegation per-item.
    let get_batch_calls = Arc::new(AtomicUsize::new(0));
    let get_calls = Arc::new(AtomicUsize::new(0));

    let ds = BatchOverrideDs {
        len: 9,
        get_calls: Arc::clone(&get_calls),
        get_batch_calls: Arc::clone(&get_batch_calls),
    };

    let mut loader = DataLoader::builder(ds).batch_size(3).build();
    let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 3);

    // get_batch should have been called once per batch (3 batches).
    assert_eq!(
        get_batch_calls.load(Ordering::Relaxed),
        3,
        "get_batch must be called exactly once per batch"
    );
    // get is called by our get_batch implementation → 9 total item fetches.
    assert_eq!(
        get_calls.load(Ordering::Relaxed),
        9,
        "get must be called once per item inside get_batch"
    );
}
