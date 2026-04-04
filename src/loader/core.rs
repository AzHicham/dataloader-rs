use std::sync::Arc;

#[cfg(feature = "python")]
use crate::loader::iter::OwnedDataLoaderIter;
use crate::{
    collator::{Collator, VecCollator},
    dataset::Dataset,
    error::Result,
    loader::{builder::DataLoaderBuilder, iter::DataLoaderIter},
    sampler::{BatchSampler, Sampler, SequentialSampler},
};

/// High-performance DataLoader with a PyTorch-like interface.
pub struct DataLoader<D, S: Sampler, C> {
    pub(super) dataset: D,
    pub(super) batch_sampler: BatchSampler<S>,
    pub(super) collator: C,
    pub(super) prefetch_depth: usize,
    /// Number of independent worker threads (inter-batch concurrency).
    /// `0` means the direct path: batches are processed in `Iterator::next`.
    pub(super) inter_workers: usize,
    /// Optional rayon pool for intra-batch item-level parallelism.
    /// `None` when `intra_workers = 0`.
    pub(super) pool: Option<Arc<rayon::ThreadPool>>,
}

impl<D: Dataset> DataLoader<D, SequentialSampler, VecCollator> {
    /// Create a [`DataLoaderBuilder`] starting with [`SequentialSampler`] and
    /// [`VecCollator`] as defaults.
    pub fn builder(dataset: D) -> DataLoaderBuilder<D, SequentialSampler, VecCollator> {
        DataLoaderBuilder::new(dataset)
    }
}

impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collator<D::Item>,
{
    /// Start one epoch of iteration.
    pub fn iter(&mut self) -> DataLoaderIter<'_, D, C> {
        DataLoaderIter::new(self)
    }

    /// Start one epoch of iteration without borrowing `self` in the returned
    /// iterator.
    #[cfg(feature = "python")]
    pub(crate) fn iter_owned(&mut self) -> OwnedDataLoaderIter<C::Batch> {
        OwnedDataLoaderIter::new(self)
    }

    /// Reference to the underlying dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Reference to the collator.
    pub fn collator(&self) -> &C {
        &self.collator
    }

    /// Returns `true` when inter-batch worker threads are active.
    pub(crate) fn has_workers(&self) -> bool {
        self.inter_workers > 0
    }

    /// Generate batch index chunks for one epoch, advancing the sampler.
    pub(crate) fn epoch_chunks(&mut self) -> Vec<Vec<usize>> {
        self.batch_sampler.batch_indices(self.dataset.len())
    }

    /// Total number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Total number of batches for one epoch with current batch settings.
    pub fn batch_len(&self) -> usize {
        let n = self.dataset.len();
        let bs = self.batch_sampler.batch_size();
        if self.batch_sampler.drop_last() {
            n / bs
        } else {
            n.div_ceil(bs)
        }
    }

    /// Returns `true` when the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

impl<'a, D, S, C> IntoIterator for &'a mut DataLoader<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collator<D::Item>,
    C::Batch: Send + 'static,
{
    type Item = Result<C::Batch>;
    type IntoIter = DataLoaderIter<'a, D, C>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::Arc as StdArc,
        sync::atomic::{AtomicUsize, Ordering},
        thread,
        time::Duration,
    };

    use crate::{
        collator::{Collator, VecCollator},
        dataset::Dataset,
        error::Result,
        loader::DataLoader,
        sampler::{RandomSampler, SequentialSampler},
    };

    /// Returns `index` as its item. Never fails.
    struct CountingDs {
        len: usize,
    }

    impl Dataset for CountingDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<usize> {
            Ok(index)
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    /// Tracks concurrent calls to `get` so tests can assert parallel execution.
    struct ConcurrentDs {
        len: usize,
        peak: StdArc<AtomicUsize>,
        active: StdArc<AtomicUsize>,
    }

    impl Dataset for ConcurrentDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<usize> {
            let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(current, Ordering::SeqCst);
            thread::sleep(Duration::from_micros(100));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(index)
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    /// Always returns an error.
    struct ErrDs;

    impl Dataset for ErrDs {
        type Item = u8;

        fn get(&self, _: usize) -> Result<u8> {
            Err("dataset failure".into())
        }

        fn len(&self) -> usize {
            4
        }
    }

    struct PartialErrDs {
        fail_at: usize,
        len: usize,
    }

    impl Dataset for PartialErrDs {
        type Item = usize;

        fn get(&self, index: usize) -> Result<Self::Item> {
            if index == self.fail_at {
                Err(format!("fail at index {index}").into())
            } else {
                Ok(index)
            }
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    fn counting(len: usize) -> DataLoader<CountingDs, SequentialSampler, VecCollator> {
        DataLoader::builder(CountingDs { len }).build()
    }

    #[test]
    fn builder_defaults() {
        let mut loader = counting(5);
        assert_eq!(loader.iter().count(), 5);
    }

    #[test]
    fn builder_batch_size() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn builder_drop_last() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .drop_last(true)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.len(), 3);
        }
    }

    #[test]
    fn builder_no_drop_last_exact_divisor() {
        let mut loader = DataLoader::builder(CountingDs { len: 9 })
            .batch_size(3)
            .drop_last(true)
            .build();
        assert_eq!(loader.iter().count(), 3);
    }

    #[test]
    #[should_panic]
    fn builder_zero_batch_size_panics() {
        DataLoader::builder(CountingDs { len: 5 })
            .batch_size(0)
            .build();
    }

    #[test]
    #[should_panic]
    fn builder_zero_prefetch_depth_panics() {
        DataLoader::builder(CountingDs { len: 5 })
            .prefetch_depth(0)
            .build();
    }

    #[test]
    fn sequential_batch_values() {
        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(2)
            .build();
        let batches: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
    }

    #[test]
    fn sequential_single_item_batches() {
        let mut loader = counting(4);
        let items: Vec<usize> = loader
            .iter()
            .map(|b| b.unwrap().into_iter().next().unwrap())
            .collect();
        assert_eq!(items, vec![0, 1, 2, 3]);
    }

    #[test]
    fn empty_dataset_yields_no_batches() {
        let mut loader = counting(0);
        assert_eq!(loader.iter().count(), 0);
    }

    #[test]
    fn single_item_dataset() {
        let mut loader = counting(1);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].as_ref().unwrap(), &[0]);
    }

    #[test]
    fn all_indices_covered_sequential() {
        let n = 25;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(4)
            .build();
        let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn all_indices_covered_drop_last() {
        let n = 7;
        let bs = 3;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(bs)
            .drop_last(true)
            .build();
        let seen: Vec<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn exact_size_no_drop_last() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .build();
        let iter = loader.iter();
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn exact_size_with_drop_last() {
        let mut loader = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(3)
            .drop_last(true)
            .build();
        let iter = loader.iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn exact_size_decrements_as_consumed() {
        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(2)
            .build();
        let mut iter = loader.iter();
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
        iter.next();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn loader_is_reusable_across_epochs() {
        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(2)
            .build();
        let epoch1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        let epoch2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(epoch1, epoch2);
    }

    #[test]
    fn random_sampler_differs_across_epochs() {
        let mut loader = DataLoader::builder(CountingDs { len: 20 })
            .batch_size(4)
            .sampler(RandomSampler::new(0))
            .build();
        let e1: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        let e2: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_ne!(e1, e2);
    }

    #[test]
    fn all_indices_covered_random() {
        let n = 50;
        let mut loader = DataLoader::builder(CountingDs { len: n })
            .batch_size(5)
            .sampler(RandomSampler::new(1))
            .build();
        let seen: HashSet<usize> = loader.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn parallel_fetch_produces_same_items() {
        let n = 30;
        let bs = 5;

        let mut seq = DataLoader::builder(CountingDs { len: n })
            .batch_size(bs)
            .build();
        let mut par = DataLoader::builder(CountingDs { len: n })
            .batch_size(bs)
            .num_workers(4)
            .build();

        let seq_out: Vec<Vec<usize>> = seq.iter().map(|b| b.unwrap()).collect();
        let par_out: Vec<Vec<usize>> = par.iter().map(|b| b.unwrap()).collect();
        assert_eq!(seq_out, par_out);
    }

    #[test]
    fn parallel_fetch_actually_concurrent() {
        let peak = StdArc::new(AtomicUsize::new(0));
        let active = StdArc::new(AtomicUsize::new(0));

        let ds = ConcurrentDs {
            len: 16,
            peak: StdArc::clone(&peak),
            active: StdArc::clone(&active),
        };

        let mut loader = DataLoader::builder(ds)
            .batch_size(8)
            .num_workers(4)
            .prefetch_depth(2)
            .build();

        loader.iter().for_each(|b| {
            b.unwrap();
        });

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "expected concurrent get() calls, but peak concurrency was 1"
        );
    }

    #[test]
    fn dataset_error_propagates() {
        let mut loader = DataLoader::builder(ErrDs).batch_size(2).build();
        let first = loader.iter().next().unwrap();
        assert!(first.is_err());
    }

    #[test]
    fn dataset_error_does_not_panic_loader() {
        let mut loader = DataLoader::builder(ErrDs).batch_size(2).build();
        for batch in loader.iter() {
            let _ = batch;
        }
        let count = loader.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn iter_catches_error_from_dataset_get() {
        let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
            .batch_size(4)
            .build();

        let first = loader.iter().next().expect("missing first batch");
        assert!(
            first.is_err(),
            "first batch should surface dataset get() error"
        );
    }

    #[test]
    fn collate_is_skipped_when_get_fails_and_error_is_iterated() {
        struct CountingCollator {
            calls: StdArc<AtomicUsize>,
        }

        impl Collator<usize> for CountingCollator {
            type Batch = Vec<usize>;

            fn collate(&self, items: Vec<usize>) -> Result<Self::Batch> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(items)
            }
        }

        let calls = StdArc::new(AtomicUsize::new(0));
        let collator = CountingCollator {
            calls: StdArc::clone(&calls),
        };

        let mut loader = DataLoader::builder(PartialErrDs { fail_at: 3, len: 8 })
            .batch_size(4)
            .collator(collator)
            .build();

        let results: Vec<_> = loader.iter().collect();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_err(), "first batch should be an error");
        assert!(results[1].is_ok(), "second batch should succeed");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "collate should be called only for successful batches"
        );
    }

    #[test]
    fn custom_collator_sum() {
        struct SumCollator;
        impl Collator<usize> for SumCollator {
            type Batch = usize;
            fn collate(&self, items: Vec<usize>) -> Result<usize> {
                Ok(items.iter().sum())
            }
        }

        let mut loader = DataLoader::builder(CountingDs { len: 6 })
            .batch_size(3)
            .collator(SumCollator)
            .build();

        let sums: Vec<usize> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(sums, vec![3, 12]);
    }

    #[test]
    fn early_drop_does_not_hang() {
        let mut loader = DataLoader::builder(CountingDs { len: 100 })
            .batch_size(5)
            .prefetch_depth(2)
            .num_workers(2)
            .build();

        let mut iter = loader.iter();
        iter.next();
        drop(iter);

        let count = DataLoader::builder(CountingDs { len: 10 })
            .batch_size(2)
            .build()
            .iter()
            .count();
        assert_eq!(count, 5);
    }

    #[test]
    fn early_break_then_new_epoch() {
        let mut loader = DataLoader::builder(CountingDs { len: 20 })
            .batch_size(4)
            .prefetch_depth(4)
            .num_workers(2)
            .build();

        for (i, b) in (&mut loader).into_iter().enumerate() {
            b.unwrap();
            if i == 1 {
                break;
            }
        }

        let second: Vec<Vec<usize>> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(second.len(), 5);

        let all_items: HashSet<usize> = second.into_iter().flatten().collect();
        assert_eq!(all_items.len(), 20);
    }

    #[test]
    fn loader_len() {
        let loader = counting(42);
        assert_eq!(loader.len(), 42);
        assert!(!loader.is_empty());
    }

    #[test]
    fn loader_is_empty() {
        let loader = counting(0);
        assert!(loader.is_empty());
        assert_eq!(loader.len(), 0);
    }

    #[test]
    fn dataset_accessor() {
        let loader = counting(7);
        assert_eq!(loader.dataset().len(), 7);
    }

    #[cfg(feature = "torch-rs")]
    #[test]
    fn pin_memory_true_keeps_tensor_batches_valid() {
        use crate::collator::DefaultCollator;
        use tch::{Device, Kind, Tensor};

        struct TensorDs(usize);
        impl Dataset for TensorDs {
            type Item = Tensor;

            fn get(&self, _index: usize) -> Result<Self::Item> {
                Tensor::f_zeros([2, 3], (Kind::Float, Device::Cpu)).map_err(|e| e.into())
            }

            fn len(&self) -> usize {
                self.0
            }
        }

        let mut loader = DataLoader::builder(TensorDs(4))
            .batch_size(2)
            .collator(DefaultCollator)
            .pin_memory(true)
            .build();

        for batch in loader.iter() {
            let batch = batch.unwrap();
            assert_eq!(batch.size(), vec![2, 2, 3]);
        }
    }
}
