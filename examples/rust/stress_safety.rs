//! A stress test of early exits: iterators dropped before the end of an epoch, many times
//! over, with workers, with a shallow prefetch and with item-level parallelism. Each case
//! checks that dropping an iterator never hangs and that the loader still runs a full epoch
//! afterwards.
//!
//! Run it with (in release mode, as it repeats each case many times):
//!
//! ```text
//! cargo run --release --example stress_safety
//! ```

use dataloader_rs::{DataLoader, Dataset, error::Result};
use std::time::Duration;

struct SlowDs {
    len: usize,
    delay_us: u64,
}
impl Dataset for SlowDs {
    type Item = usize;
    fn get(&self, i: usize) -> Result<usize> {
        // Busy-wait to stand for real work.
        let target = std::time::Instant::now() + Duration::from_micros(self.delay_us);
        while std::time::Instant::now() < target {
            std::hint::spin_loop();
        }
        Ok(i)
    }
    fn len(&self) -> usize {
        self.len
    }
}

fn main() {
    // Test 1: drop an iterator while finished batches wait for the consumer
    // (`prefetch_depth(1)`, and the consumer reads only one batch).
    println!("Test 1: early drop with prefetch_depth=1...");
    for _ in 0..100 {
        let mut loader = DataLoader::builder(SlowDs {
            len: 100,
            delay_us: 0,
        })
        .batch_size(5)
        .num_workers(4)
        .prefetch_depth(1)
        .build();
        let mut iter = loader.iter();
        let _ = iter.next(); // read one batch; the next ones are ready and waiting
        drop(iter); // must not hang
        // The loader is still usable: a full epoch follows.
        let count = loader.iter().count();
        assert_eq!(count, 20);
    }
    println!("  OK");

    // Test 2: drop an iterator before reading any batch, with 8 workers busy.
    println!("Test 2: zero-consume drop with deep prefetch...");
    for _ in 0..100 {
        let mut loader = DataLoader::builder(SlowDs {
            len: 200,
            delay_us: 0,
        })
        .batch_size(10)
        .num_workers(8)
        .prefetch_depth(2)
        .build();
        let iter = loader.iter();
        drop(iter); // dropped before reading anything
        let count = loader.iter().count();
        assert_eq!(count, 20);
    }
    println!("  OK");

    // Test 3: 50 early drops in a row on the same loader.
    println!("Test 3: 50 sequential early drops...");
    let mut loader = DataLoader::builder(SlowDs {
        len: 100,
        delay_us: 10,
    })
    .batch_size(5)
    .num_workers(4)
    .prefetch_depth(4)
    .build();
    for i in 0..50 {
        let mut iter = loader.iter();
        // Every other time, read one batch before dropping the iterator.
        if i % 2 == 0 {
            let _ = iter.next();
        }
        drop(iter);
    }
    // A full epoch still follows.
    let count = loader.iter().count();
    assert_eq!(count, 20);
    println!("  OK");

    // Test 4: 8 workers on 8 batches, one batch each: every item comes out exactly once.
    println!("Test 4: 8 workers on 8 batches (1 batch/worker)...");
    for _ in 0..50 {
        let mut loader = DataLoader::builder(SlowDs {
            len: 64,
            delay_us: 100,
        })
        .batch_size(8)
        .num_workers(8)
        .prefetch_depth(8)
        .build();
        let items: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(items.len(), 64);
        let mut sorted = items.clone();
        sorted.sort();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
    }
    println!("  OK");

    // Test 5: early drops while the items of each batch are fetched in parallel.
    println!("Test 5: item_concurrency drop safety...");
    for _ in 0..50 {
        let mut loader = DataLoader::builder(SlowDs {
            len: 100,
            delay_us: 50,
        })
        .batch_size(16)
        .num_workers(2)
        .item_concurrency(4)
        .prefetch_depth(4)
        .build();
        let mut iter = loader.iter();
        let _ = iter.next();
        drop(iter);
        let count = loader.iter().count();
        assert_eq!(count, 7); // 100 items in batches of 16: 7 batches
    }
    println!("  OK");

    println!("\nAll safety stress tests passed!");
}
