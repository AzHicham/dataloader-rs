//! A hand-written async dataset: each item waits 20 ms, as a network request would, and the
//! loader keeps many of them in flight at once without a thread per request.
//!
//! An `AsyncDataset` returns a future from `get`. `DataLoader::builder_async` runs those futures
//! on a tokio runtime that the loader creates, `num_workers × item_concurrency` at a time, and
//! runs the transform and the collator on its own thread pool, never on the runtime's threads.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example async_dataset --features async
//! ```

use std::time::{Duration, Instant};

use dataloader_rs::{AsyncDataset, DataLoader, error::Result};

/// 64 items that each take 20 ms to arrive. Item `i` is the text `"record i"`.
struct SlowRecords;

impl AsyncDataset for SlowRecords {
    type Item = String;

    async fn get(&self, index: usize) -> Result<String> {
        // A real dataset would await a request here. Awaiting does not block a thread: while
        // this future waits, the runtime's threads drive the other ones.
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok(format!("record {index}"))
    }

    fn len(&self) -> usize {
        64
    }
}

fn main() -> Result<()> {
    let mut loader = DataLoader::builder_async(SlowRecords)
        .batch_size(16)
        .num_workers(2) // 2 batches are fetched at the same time,
        .item_concurrency(8) // with 8 items of each in flight: 16 requests at once
        .runtime_threads(1) // one runtime thread is enough to drive them all
        // CPU work belongs in the transform: it runs on the loader's thread pool as each item
        // arrives, while the other items of the batch are still being fetched.
        .transform(|_index: usize, record: String| -> Result<usize> { Ok(record.len()) })
        .build();

    let start = Instant::now();
    let mut items = 0;
    for batch in &mut loader {
        let lengths: Vec<usize> = batch?;
        items += lengths.len();
    }
    let elapsed = start.elapsed();
    println!(
        "{items} items of 20 ms each in {} ms with 16 in flight (one at a time: {} ms)",
        elapsed.as_millis(),
        items * 20
    );
    Ok(())
}
