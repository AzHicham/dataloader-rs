// Stress-test the safety invariants:
// 1. Early drop with channel FULL (workers blocked on send)
// 2. Early drop with prefetch_depth=1 (maximum backpressure)
// 3. Multiple sequential early drops
// 4. Drop during active parallel processing
// 5. Reuse after panic-like early drop

use dataloader_rs::{DataLoader, Dataset, error::Result};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::Duration;

struct SlowDs { len: usize, delay_us: u64 }
impl Dataset for SlowDs {
    type Item = usize;
    fn get(&self, i: usize) -> Result<usize> {
        // Spin to simulate work
        let target = std::time::Instant::now() + Duration::from_micros(self.delay_us);
        while std::time::Instant::now() < target { std::hint::spin_loop(); }
        Ok(i)
    }
    fn len(&self) -> usize { self.len }
}

fn main() {
    // Test 1: drop with channel full (depth=1, slow consumer never drains it)
    println!("Test 1: early drop with prefetch_depth=1...");
    for _ in 0..100 {
        let mut loader = DataLoader::builder(SlowDs { len: 100, delay_us: 0 })
            .batch_size(5).num_workers(4).prefetch_depth(1).build();
        let mut iter = loader.iter();
        let _ = iter.next(); // consume 1, now channel is full
        drop(iter); // must not deadlock
        // verify reuse
        let count = loader.iter().count();
        assert_eq!(count, 20);
    }
    println!("  OK");

    // Test 2: drop with 0 batches consumed (all workers potentially blocked on send)
    println!("Test 2: zero-consume drop with deep prefetch...");
    for _ in 0..100 {
        let mut loader = DataLoader::builder(SlowDs { len: 200, delay_us: 0 })
            .batch_size(10).num_workers(8).prefetch_depth(2).build();
        let iter = loader.iter();
        drop(iter); // drop before consuming anything
        let count = loader.iter().count();
        assert_eq!(count, 20);
    }
    println!("  OK");

    // Test 3: multiple sequential early drops
    println!("Test 3: 50 sequential early drops...");
    let mut loader = DataLoader::builder(SlowDs { len: 100, delay_us: 10 })
        .batch_size(5).num_workers(4).prefetch_depth(4).build();
    for i in 0..50 {
        let mut iter = loader.iter();
        // Alternate: drop with 0 consumed or 1 consumed
        if i % 2 == 0 { let _ = iter.next(); }
        drop(iter);
    }
    // Final full epoch
    let count = loader.iter().count();
    assert_eq!(count, 20);
    println!("  OK");

    // Test 4: concurrent stress — 8 workers racing on 8 batches
    println!("Test 4: 8 workers on 8 batches (1 batch/worker)...");
    for _ in 0..50 {
        let mut loader = DataLoader::builder(SlowDs { len: 64, delay_us: 100 })
            .batch_size(8).num_workers(8).prefetch_depth(8).build();
        let items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(items.len(), 64);
        let mut sorted = items.clone(); sorted.sort();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
    }
    println!("  OK");

    // Test 5: intra_workers + early drop
    println!("Test 5: intra_workers drop safety...");
    for _ in 0..50 {
        let mut loader = DataLoader::builder(SlowDs { len: 100, delay_us: 50 })
            .batch_size(16).num_workers(2).intra_workers(4).prefetch_depth(4).build();
        let mut iter = loader.iter();
        let _ = iter.next();
        drop(iter);
        let count = loader.iter().count();
        assert_eq!(count, 7); // ceil(100/16)
    }
    println!("  OK");

    println!("\nAll safety stress tests passed!");
}
