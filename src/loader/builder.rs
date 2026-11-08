use std::sync::Arc;

#[cfg(feature = "torch-rs")]
use crate::collator::TorchPinnedCollator;
use crate::{
    collator::{Collator, VecCollator},
    dataset::Dataset,
    loader::core::{DataLoader, LoaderPool, ThreadMain},
    sampler::{BatchSampler, Sampler, SequentialSampler},
    transform::{Identity, Transform},
};

/// Builder for [`DataLoader`].
pub struct DataLoaderBuilder<D, S, C, T = Identity> {
    dataset: D,
    sampler: S,
    transform: T,
    collator: C,
    settings: Settings,
}

/// The settings that do not depend on the builder's type parameters, carried
/// over as they are when [`sampler`](DataLoaderBuilder::sampler),
/// [`transform`](DataLoaderBuilder::transform),
/// [`collator`](DataLoaderBuilder::collator) or `pin_memory` change a type.
struct Settings {
    batch_size: usize,
    drop_last: bool,
    prefetch_depth: usize,
    /// Batches processed in parallel.
    num_workers: usize,
    /// Items of one batch fetched in parallel.
    item_concurrency: usize,
    /// Main function of the pool's threads; `None` runs rayon's loop as is.
    thread_main: Option<ThreadMain>,
}

impl<D: Dataset> DataLoaderBuilder<D, SequentialSampler, VecCollator> {
    pub(super) fn new(dataset: D) -> Self {
        DataLoaderBuilder {
            dataset,
            sampler: SequentialSampler,
            transform: Identity,
            collator: VecCollator,
            settings: Settings {
                batch_size: 1,
                drop_last: false,
                prefetch_depth: 2,
                num_workers: 0,
                item_concurrency: 1,
                thread_main: None,
            },
        }
    }
}

impl<D, S, C, T> DataLoaderBuilder<D, S, C, T> {
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
    /// The loader's pool has `max(num_workers, 1) × item_concurrency`
    /// threads; sizing it for the machine is the caller's choice.
    pub fn item_concurrency(mut self, n: usize) -> Self {
        assert!(n > 0, "item_concurrency must be > 0");
        self.settings.item_concurrency = n;
        self
    }

    /// Former name of [`item_concurrency`](Self::item_concurrency).
    #[deprecated(since = "0.3.0", note = "renamed to `item_concurrency`")]
    pub fn items_in_parallel(self, n: usize) -> Self {
        self.item_concurrency(n)
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
    pub fn pin_memory(self, enabled: bool) -> DataLoaderBuilder<D, S, TorchPinnedCollator<C>, T>
    where
        D: Dataset,
        T: Transform<D::Item>,
        C: Collator<T::Output, Batch = tch::Tensor>,
    {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            transform: self.transform,
            collator: TorchPinnedCollator::new(self.collator, enabled),
            settings: self.settings,
        }
    }

    /// Replace the sampler and preserve all other settings.
    pub fn sampler<S2: Sampler>(self, sampler: S2) -> DataLoaderBuilder<D, S2, C, T> {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler,
            transform: self.transform,
            collator: self.collator,
            settings: self.settings,
        }
    }

    /// Apply `transform` to every item, with its index, before collation; it
    /// replaces any previous one (chain them with
    /// [`Transform::then`]). Default: [`Identity`], at no cost.
    ///
    /// It runs where the item is loaded: right after `get` / `get_batch` on the
    /// worker (in each item lane when
    /// [`item_concurrency`](Self::item_concurrency) is above 1, so items are
    /// transformed in parallel).
    /// The collator receives the transformed items. An error fails the item's
    /// batch; a panic is re-raised by the iterator.
    ///
    /// ```rust
    /// use dataloader_rs::{DataLoader, Dataset, error::Result};
    ///
    /// struct Numbers;
    ///
    /// impl Dataset for Numbers {
    ///     type Item = u64;
    ///     fn get(&self, index: usize) -> Result<u64> { Ok(index as u64) }
    ///     fn len(&self) -> usize { 4 }
    /// }
    ///
    /// let mut loader = DataLoader::builder(Numbers)
    ///     .batch_size(2)
    ///     .transform(|index: usize, x: u64| -> Result<String> { Ok(format!("{index}:{}", x * 10)) })
    ///     .build();
    /// let batches: Vec<Vec<String>> = loader.iter().map(|b| b.unwrap()).collect();
    /// assert_eq!(batches, [["0:0", "1:10"], ["2:20", "3:30"]]);
    /// ```
    pub fn transform<T2>(self, transform: T2) -> DataLoaderBuilder<D, S, C, T2>
    where
        D: Dataset,
        T2: Transform<D::Item>,
    {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            transform,
            collator: self.collator,
            settings: self.settings,
        }
    }

    /// Replace the collator and preserve all other settings.
    pub fn collator<C2>(self, collator: C2) -> DataLoaderBuilder<D, S, C2, T> {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            transform: self.transform,
            collator,
            settings: self.settings,
        }
    }

    /// Finalize configuration and construct a [`DataLoader`].
    pub fn build(self) -> DataLoader<D, S, C, T>
    where
        D: Dataset,
        S: Sampler,
        T: Transform<D::Item>,
        C: Collator<T::Output>,
    {
        let Settings {
            batch_size,
            drop_last,
            prefetch_depth,
            num_workers,
            item_concurrency,
            thread_main,
        } = self.settings;
        // One pool for batch-level and item-level work: num_workers batches,
        // each with item_concurrency items in flight.
        let threads = num_workers
            .max(1)
            .checked_mul(item_concurrency)
            .expect("num_workers × item_concurrency overflows usize");
        let needs_pool = num_workers > 0 || item_concurrency > 1;
        let pool = needs_pool.then(|| LoaderPool::new(threads, thread_main));

        DataLoader {
            dataset: Arc::new(self.dataset),
            batch_sampler: BatchSampler::new(self.sampler, batch_size, drop_last),
            transform: Arc::new(self.transform),
            collator: Arc::new(self.collator),
            prefetch_depth,
            num_workers,
            item_concurrency,
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
