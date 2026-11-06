use std::sync::Arc;

#[cfg(feature = "torch-rs")]
use crate::collator::TorchPinnedCollator;
use crate::{
    collator::{
        Collator, VecCollator,
        pooled::{BufferAllocator, PoolSizer, PooledCollator},
    },
    dataset::Dataset,
    loader::core::{DataLoader, LoaderPool, ThreadMain},
    sampler::{BatchSampler, Sampler, SequentialSampler},
};

/// Builder for [`DataLoader`].
pub struct DataLoaderBuilder<D, S, C> {
    dataset: D,
    sampler: S,
    collator: C,
    settings: Settings,
}

/// The settings that do not depend on the builder's type parameters, carried
/// over as they are when [`sampler`](DataLoaderBuilder::sampler),
/// [`collator`](DataLoaderBuilder::collator) or `pin_memory` change a type.
struct Settings {
    batch_size: usize,
    drop_last: bool,
    prefetch_depth: usize,
    /// Batches processed in parallel.
    num_workers: usize,
    /// Items of one batch fetched in parallel.
    items_in_parallel: usize,
    /// Main function of the pool's threads; `None` runs rayon's loop as is.
    thread_main: Option<ThreadMain>,
    /// Batches the consumer keeps alive at once, for sizing a buffer pool.
    max_held_batches: usize,
    /// The buffer pool of a [`PooledCollator`] set with `collate_into`, sized
    /// by `build` once every setting is known.
    buffer_pool: Option<Arc<dyn PoolSizer>>,
}

impl<D: Dataset> DataLoaderBuilder<D, SequentialSampler, VecCollator> {
    pub(super) fn new(dataset: D) -> Self {
        DataLoaderBuilder {
            dataset,
            sampler: SequentialSampler,
            collator: VecCollator,
            settings: Settings {
                batch_size: 1,
                drop_last: false,
                prefetch_depth: 2,
                num_workers: 0,
                items_in_parallel: 1,
                thread_main: None,
                max_held_batches: 1,
                buffer_pool: None,
            },
        }
    }
}

impl<D, S, C> DataLoaderBuilder<D, S, C> {
    /// Number of items per batch. Default: `1`.
    pub fn batch_size(mut self, n: usize) -> Self {
        assert!(n > 0, "batch_size must be > 0");
        self.settings.batch_size = n;
        self
    }

    /// Number of batches workers may run ahead of the consumer. At most
    /// `num_workers + prefetch_depth` batches are in flight or buffered.
    /// Default: `2`.
    pub fn prefetch_depth(mut self, n: usize) -> Self {
        assert!(n > 0, "prefetch_depth must be > 0");
        self.settings.prefetch_depth = n;
        self
    }

    /// Whether to drop a non-full final batch. Default: `false`.
    pub fn drop_last(mut self, v: bool) -> Self {
        self.settings.drop_last = v;
        self
    }

    /// Number of batches processed in parallel — PyTorch's `num_workers`.
    ///
    /// Batches run on the loader's thread pool, built once and reused by every
    /// epoch. `0` (default) processes batches directly in `Iterator::next`.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.settings.num_workers = n;
        self
    }

    /// Number of items of one batch fetched in parallel. Default: `1`.
    ///
    /// With `1`, a batch is fetched with one [`Dataset::get_batch`] call. With
    /// `n > 1`, each batch is fetched by `n` concurrent lanes, each calling
    /// [`Dataset::get`] for the batch's next item, so at most `n` items of a
    /// batch are in flight. Useful for I/O-bound datasets (network, object
    /// storage) and for few, large CPU-bound batches.
    ///
    /// The loader's pool has `max(num_workers, 1) × items_in_parallel`
    /// threads; sizing it for the machine is the caller's choice.
    pub fn items_in_parallel(mut self, n: usize) -> Self {
        assert!(n > 0, "items_in_parallel must be > 0");
        self.settings.items_in_parallel = n;
        self
    }

    /// Run each pool thread through `main`, which must call
    /// `ThreadBuilder::run` (the Python binding keeps the thread attached to
    /// the interpreter for its whole life).
    #[cfg(feature = "python")]
    pub(crate) fn thread_main(mut self, main: ThreadMain) -> Self {
        self.settings.thread_main = Some(main);
        self
    }

    /// Pin Tensor batches in host memory before they are sent to the consumer.
    ///
    /// Available only for `tch::Tensor` batches under `torch-rs` feature.
    #[cfg(feature = "torch-rs")]
    pub fn pin_memory(self, enabled: bool) -> DataLoaderBuilder<D, S, TorchPinnedCollator<C>>
    where
        D: Dataset,
        C: Collator<D::Item, Batch = tch::Tensor>,
    {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collator: TorchPinnedCollator::new(self.collator, enabled),
            settings: self.settings,
        }
    }

    /// Replace the sampler and preserve all other settings.
    pub fn sampler<S2: Sampler>(self, sampler: S2) -> DataLoaderBuilder<D, S2, C> {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler,
            collator: self.collator,
            settings: self.settings,
        }
    }

    /// Replace the collator and preserve all other settings.
    pub fn collator<C2>(mut self, collator: C2) -> DataLoaderBuilder<D, S, C2> {
        self.settings.buffer_pool = None;
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collator,
            settings: self.settings,
        }
    }

    /// Collate each batch into a reusable buffer from `allocator`, with
    /// `collator` (a [`CollateInto`](crate::collator::CollateInto)), instead
    /// of allocating every batch.
    ///
    /// Batches are [`PooledBatch`](crate::collator::PooledBatch)es: dropping
    /// one returns its buffer to the loader's pool. The pool allocates
    /// buffers on demand, up to `num_workers + prefetch_depth` (`1` with
    /// `num_workers = 0`) plus [`max_held_batches`](Self::max_held_batches),
    /// and never more: when every buffer is taken, workers wait for the
    /// consumer to drop a batch. Keeping more than `max_held_batches` batches
    /// alive on the thread that iterates (e.g. `collect()`) therefore
    /// deadlocks; drop them, or hand them to another thread that does.
    ///
    /// The allocator decides where the memory lives: [`HeapAllocator`] for
    /// plain `Vec`s, `TorchPinnedAllocator` for pinned tensors, or a closure
    /// returning memory from another library (e.g. pinned host memory from
    /// ONNX Runtime or CUDA).
    ///
    /// [`HeapAllocator`]: crate::collator::HeapAllocator
    pub fn collate_into<C2, A>(
        mut self,
        collator: C2,
        allocator: A,
    ) -> DataLoaderBuilder<D, S, PooledCollator<C2, A>>
    where
        A: BufferAllocator,
    {
        let collator = PooledCollator::new(collator, allocator, 1);
        self.settings.buffer_pool = Some(collator.sizer());
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collator,
            settings: self.settings,
        }
    }

    /// Batches the consumer keeps alive at once, the one it is processing
    /// included, when batches come from [`collate_into`](Self::collate_into).
    /// Default: `1`, for a loop that drops each batch before asking for the
    /// next; `2` to process one batch while the previous one is still in use
    /// (e.g. an asynchronous copy to the GPU). Sizes the buffer pool only.
    pub fn max_held_batches(mut self, n: usize) -> Self {
        self.settings.max_held_batches = n;
        self
    }

    /// Finalize configuration and construct a [`DataLoader`].
    pub fn build(self) -> DataLoader<D, S, C>
    where
        D: Dataset,
        S: Sampler,
        C: Collator<D::Item>,
    {
        let Settings {
            batch_size,
            drop_last,
            prefetch_depth,
            num_workers,
            items_in_parallel,
            thread_main,
            max_held_batches,
            buffer_pool,
        } = self.settings;
        if let Some(pool) = buffer_pool {
            // Batches in flight or waiting (see `ParallelCore`), plus those
            // the consumer holds.
            let in_flight = match num_workers {
                0 => 1,
                n => n.saturating_add(prefetch_depth),
            };
            pool.set_capacity(in_flight.saturating_add(max_held_batches));
        }
        // One pool for batch-level and item-level work: num_workers batches,
        // each with items_in_parallel items in flight.
        let threads = num_workers
            .max(1)
            .checked_mul(items_in_parallel)
            .expect("num_workers × items_in_parallel overflows usize");
        let needs_pool = num_workers > 0 || items_in_parallel > 1;
        let pool = needs_pool.then(|| LoaderPool::new(threads, thread_main));

        DataLoader {
            dataset: Arc::new(self.dataset),
            batch_sampler: BatchSampler::new(self.sampler, batch_size, drop_last),
            collator: Arc::new(self.collator),
            prefetch_depth,
            num_workers,
            items_in_parallel,
            pool,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collator::Collator, dataset::Dataset, error::Result, loader::DataLoader, sampler::Sampler,
    };

    struct TinyDs(usize);

    impl Dataset for TinyDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<Self::Item> {
            Ok(index)
        }

        fn len(&self) -> usize {
            self.0
        }
    }

    struct ReverseSampler;

    impl Sampler for ReverseSampler {
        fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
            (0..dataset_len).rev().collect()
        }
    }

    struct SumCollator;

    impl Collator<usize> for SumCollator {
        type Batch = usize;

        fn collate(&self, items: Vec<usize>) -> Result<Self::Batch> {
            Ok(items.into_iter().sum())
        }
    }

    #[test]
    fn builder_defaults_create_single_item_batches() {
        let mut loader = DataLoader::builder(TinyDs(4)).build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn builder_replaces_sampler_and_collator() {
        let mut loader = DataLoader::builder(TinyDs(4))
            .batch_size(2)
            .sampler(ReverseSampler)
            .collator(SumCollator)
            .build();

        let sums: Vec<usize> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(sums, vec![5, 1]); // [3,2] and [1,0]
    }

    #[test]
    #[should_panic]
    fn builder_rejects_zero_batch_size() {
        let _ = DataLoader::builder(TinyDs(4)).batch_size(0);
    }

    #[test]
    #[should_panic]
    fn builder_rejects_zero_prefetch_depth() {
        let _ = DataLoader::builder(TinyDs(4)).prefetch_depth(0);
    }
}
