//! `DataLoaderBuilder::collate_into`: batches collated into reusable buffers
//! from a bounded pool, through both execution paths.

use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use dataloader_rs::{
    DataLoader, Dataset,
    collator::{CollateInto, ConcatCollator, HeapAllocator, PooledBatch},
    error::Result,
};

const ITEM_LEN: usize = 4;

/// Sample `i` is `[i; ITEM_LEN]`; optionally fails or panics at one index,
/// optionally sleeps per item.
struct RowDs {
    len: usize,
    fail_at: Option<usize>,
    panic_at: Option<usize>,
    sleep: Duration,
}

impl RowDs {
    fn new(len: usize) -> Self {
        Self {
            len,
            fail_at: None,
            panic_at: None,
            sleep: Duration::ZERO,
        }
    }
}

impl Dataset for RowDs {
    type Item = Vec<u32>;

    fn get(&self, index: usize) -> Result<Vec<u32>> {
        if !self.sleep.is_zero() {
            thread::sleep(self.sleep);
        }
        if self.fail_at == Some(index) {
            return Err(format!("fail at {index}").into());
        }
        if self.panic_at == Some(index) {
            panic!("panic at {index}");
        }
        Ok(vec![index as u32; ITEM_LEN])
    }

    fn len(&self) -> usize {
        self.len
    }
}

type Batch = PooledBatch<Vec<u32>>;

fn alloc(batch_size: usize) -> HeapAllocator<u32> {
    HeapAllocator::for_shape(&[batch_size, ITEM_LEN]).unwrap()
}

/// The sample indices of a batch, checking each row is uniform.
fn indices(batch: &Batch) -> Vec<u32> {
    batch[..batch.num_samples() * ITEM_LEN]
        .as_chunks::<ITEM_LEN>()
        .0
        .iter()
        .map(|row| {
            assert!(row.iter().all(|&v| v == row[0]), "torn row {row:?}");
            row[0]
        })
        .collect()
}

#[test]
fn direct_path_reuses_one_buffer() {
    let mut loader = DataLoader::builder(RowDs::new(10))
        .batch_size(4)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    assert_eq!(loader.collator().pool().capacity(), 2);

    let mut pointers = HashSet::new();
    let mut seen = Vec::new();
    for batch in &mut loader {
        let batch = batch.unwrap();
        pointers.insert(batch.as_ptr() as usize);
        seen.extend(indices(&batch));
    }
    assert_eq!(seen, (0..10).collect::<Vec<_>>());
    assert_eq!(pointers.len(), 1, "one buffer for the whole epoch");
    assert_eq!(loader.collator().pool().allocated(), 1);
}

#[test]
fn last_partial_batch_reports_its_size() {
    let mut loader = DataLoader::builder(RowDs::new(10))
        .batch_size(4)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    let sizes: Vec<usize> = loader.iter().map(|b| b.unwrap().num_samples()).collect();
    assert_eq!(sizes, vec![4, 4, 2]);
}

#[test]
fn workers_reuse_a_bounded_set_of_buffers_in_order() {
    for (workers, depth, items) in [(1, 1, 1), (2, 2, 1), (4, 2, 2), (3, 5, 3)] {
        let mut ds = RowDs::new(203);
        ds.sleep = Duration::from_micros(50);
        let mut loader = DataLoader::builder(ds)
            .batch_size(8)
            .num_workers(workers)
            .prefetch_depth(depth)
            .items_in_parallel(items)
            .collate_into(ConcatCollator::new(ITEM_LEN), alloc(8))
            .build();
        let capacity = workers + depth + 1;
        assert_eq!(loader.collator().pool().capacity(), capacity);

        for _epoch in 0..3 {
            let mut pointers = HashSet::new();
            let mut seen = Vec::new();
            // The iterator does not borrow the loader: the pool can be read
            // while batches are in flight.
            for batch in loader.iter() {
                let batch = batch.unwrap();
                pointers.insert(batch.as_ptr() as usize);
                seen.extend(indices(&batch));
                assert!(loader.collator().pool().allocated() <= capacity);
            }
            assert_eq!(seen, (0..203).collect::<Vec<_>>());
            assert!(pointers.len() <= capacity, "{} buffers", pointers.len());
        }
        let pool = loader.collator().pool();
        assert!(pool.allocated() <= capacity);
        // Every batch was dropped: every buffer is back.
        assert_eq!(pool.available(), pool.allocated());
    }
}

#[test]
fn holding_max_held_batches_does_not_block() {
    let held = 3;
    let mut loader = DataLoader::builder(RowDs::new(64))
        .batch_size(4)
        .num_workers(2)
        .prefetch_depth(2)
        .max_held_batches(held)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    let mut window = VecDeque::new();
    let mut seen = Vec::new();
    for batch in &mut loader {
        let batch = batch.unwrap();
        seen.extend(indices(&batch));
        window.push_back(batch);
        if window.len() == held {
            window.pop_front();
        }
    }
    drop(window);
    assert_eq!(seen, (0..64).collect::<Vec<_>>());
    assert!(loader.collator().pool().allocated() <= 2 + 2 + held);
}

#[test]
fn slow_consumer_thread_backpressures_the_workers() {
    // The consumer hands each batch to a thread that keeps it a while: the
    // pool must never grow past its capacity; workers wait instead.
    let mut loader = DataLoader::builder(RowDs::new(80))
        .batch_size(4)
        .num_workers(4)
        .prefetch_depth(4)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    let capacity = loader.collator().pool().capacity();
    assert_eq!(capacity, 9);

    let (tx, rx) = mpsc::sync_channel::<Batch>(64);
    let peak_in_sink = Arc::new(AtomicUsize::new(0));
    let sink = {
        let peak = Arc::clone(&peak_in_sink);
        thread::spawn(move || {
            let mut seen = Vec::new();
            let mut kept = VecDeque::new();
            for batch in rx {
                seen.extend(indices(&batch));
                kept.push_back(batch);
                peak.fetch_max(kept.len(), Ordering::SeqCst);
                thread::sleep(Duration::from_millis(2));
                if kept.len() > 4 {
                    kept.pop_front();
                }
            }
            seen
        })
    };
    for batch in loader.iter() {
        tx.send(batch.unwrap()).unwrap();
        assert!(loader.collator().pool().allocated() <= capacity);
    }
    drop(tx);
    assert_eq!(sink.join().unwrap(), (0..80).collect::<Vec<_>>());
    assert!(loader.collator().pool().allocated() <= capacity);
    assert!(peak_in_sink.load(Ordering::SeqCst) >= 4);
}

#[test]
fn dataset_error_is_returned_and_buffer_kept() {
    for workers in [0, 2] {
        let mut ds = RowDs::new(12);
        ds.fail_at = Some(5);
        let mut loader = DataLoader::builder(ds)
            .batch_size(4)
            .num_workers(workers)
            .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
            .build();
        for _epoch in 0..2 {
            let results: Vec<_> = loader.iter().map(|b| b.map(|b| indices(&b))).collect();
            assert_eq!(results.len(), 3);
            assert_eq!(results[0].as_ref().unwrap(), &[0, 1, 2, 3]);
            assert!(results[1].is_err());
            assert_eq!(results[2].as_ref().unwrap(), &[8, 9, 10, 11]);
        }
        let pool = loader.collator().pool();
        assert_eq!(pool.available(), pool.allocated());
    }
}

#[test]
fn collate_error_returns_the_buffer() {
    /// Fails on batches that contain sample 4.
    struct Picky;
    impl CollateInto<Vec<u32>> for Picky {
        type Buffer = Vec<u32>;
        type Output = ();
        fn collate_into(&self, items: Vec<Vec<u32>>, buffer: &mut Vec<u32>) -> Result<()> {
            if items.iter().any(|item| item[0] == 4) {
                return Err("sample 4 rejected".into());
            }
            ConcatCollator::new(ITEM_LEN).collate_into(items, buffer)
        }
    }

    let mut loader = DataLoader::builder(RowDs::new(12))
        .batch_size(4)
        .num_workers(2)
        .collate_into(Picky, alloc(4))
        .build();
    let ok: Vec<bool> = loader.iter().map(|b| b.is_ok()).collect();
    assert_eq!(ok, vec![true, false, true]);
    let pool = loader.collator().pool();
    assert_eq!(pool.available(), pool.allocated());
}

#[test]
#[should_panic(expected = "panic at 9")]
fn worker_panic_reaches_the_consumer() {
    let mut ds = RowDs::new(40);
    ds.panic_at = Some(9);
    let mut loader = DataLoader::builder(ds)
        .batch_size(4)
        .num_workers(2)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    for batch in &mut loader {
        drop(batch);
    }
}

#[test]
fn loader_is_reusable_after_a_panic_and_early_break() {
    let mut ds = RowDs::new(40);
    ds.panic_at = Some(9);
    let mut loader = DataLoader::builder(ds)
        .batch_size(4)
        .num_workers(2)
        .prefetch_depth(3)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    let capacity = loader.collator().pool().capacity();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loader.iter().for_each(drop);
    }));
    assert!(caught.is_err());

    // Break early, keep the first batch alive across the next epoch.
    let mut iter = loader.iter();
    let first = iter.next().unwrap().unwrap();
    drop(iter);
    let again: Vec<Vec<u32>> = loader
        .iter()
        .take(2)
        .map(|b| indices(&b.unwrap()))
        .collect();
    assert_eq!(again, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    assert_eq!(indices(&first), vec![0, 1, 2, 3]);
    drop(first);

    // Batches of the abandoned epochs come back once their tasks finish.
    let pool = loader.collator().pool();
    for _ in 0..200 {
        if pool.available() == pool.allocated() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(pool.available(), pool.allocated());
    assert!(pool.allocated() <= capacity);
}

#[test]
fn detached_batch_leaves_the_pool() {
    let mut loader = DataLoader::builder(RowDs::new(8))
        .batch_size(4)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(4))
        .build();
    let kept: Vec<Vec<u32>> = loader.iter().map(|b| b.unwrap().detach().0).collect();
    assert_eq!(kept.len(), 2);
    assert_ne!(kept[0].as_ptr(), kept[1].as_ptr());
    assert_eq!(loader.collator().pool().allocated(), 0);
}

#[test]
fn closure_allocator_and_per_batch_output() {
    /// Records the batch's first index as its output.
    struct FirstIndex;
    impl CollateInto<Vec<u32>> for FirstIndex {
        type Buffer = Box<[u32]>;
        type Output = u32;
        fn collate_into(&self, items: Vec<Vec<u32>>, buffer: &mut Box<[u32]>) -> Result<u32> {
            let first = items[0][0];
            ConcatCollator::<u32, Box<[u32]>>::for_buffer(ITEM_LEN).collate_into(items, buffer)?;
            Ok(first)
        }
    }

    let allocations = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&allocations);
    let allocator = move || -> Result<Box<[u32]>> {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0; 4 * ITEM_LEN].into_boxed_slice())
    };
    let mut loader = DataLoader::builder(RowDs::new(16))
        .batch_size(4)
        .num_workers(2)
        .prefetch_depth(1)
        .collate_into(FirstIndex, allocator)
        .build();
    for _epoch in 0..3 {
        let firsts: Vec<u32> = loader.iter().map(|b| *b.unwrap().output()).collect();
        assert_eq!(firsts, vec![0, 4, 8, 12]);
    }
    assert!(allocations.load(Ordering::SeqCst) <= 2 + 1 + 1);
}

#[test]
fn hostile_buffer_sizes_are_errors() {
    assert!(HeapAllocator::<f32>::for_shape(&[usize::MAX, 3, 224, 224]).is_err());
    // A buffer too small for the batch is an error per batch, not a panic.
    let mut loader = DataLoader::builder(RowDs::new(8))
        .batch_size(4)
        .collate_into(ConcatCollator::new(ITEM_LEN), alloc(2))
        .build();
    assert!(loader.iter().all(|b| b.is_err()));
}
