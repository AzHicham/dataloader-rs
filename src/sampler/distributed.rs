use crate::sampler::Sampler;

/// Shards indices across `world_size` workers for distributed training.
///
/// Mirrors `torch.utils.data.DistributedSampler`: every rank draws the same
/// order from `inner`, then takes every `world_size`-th index starting at
/// `rank`. Without `drop_last` the order is padded by wrapping around so
/// every rank gets the same number of indices; with it, the tail is dropped.
///
/// Before each epoch, `inner` is set to the current epoch, so a randomized
/// inner sampler such as [`RandomSampler`](crate::RandomSampler) yields an
/// order that depends only on `(seed, epoch)`. As in PyTorch, call
/// [`set_epoch`](Sampler::set_epoch) at the start of each epoch, or every
/// epoch reuses the same order. All ranks must use the same seed.
pub struct DistributedSampler<S: Sampler> {
    inner: S,
    rank: usize,
    world_size: usize,
    drop_last: bool,
    epoch: u64,
}

impl<S: Sampler> DistributedSampler<S> {
    /// Create a new `DistributedSampler`.
    ///
    /// # Panics
    ///
    /// Panics if `rank >= world_size` or `world_size == 0`.
    pub fn new(inner: S, rank: usize, world_size: usize) -> Self {
        assert!(world_size > 0, "world_size must be > 0");
        assert!(rank < world_size, "rank must be < world_size");
        Self {
            inner,
            rank,
            world_size,
            drop_last: false,
            epoch: 0,
        }
    }

    /// Drop the tail so each rank gets `len / world_size` indices, instead
    /// of padding up to `ceil(len / world_size)`. Default: `false`.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl<S: Sampler> DistributedSampler<S> {
    fn per_rank(&self, total: usize) -> usize {
        if self.drop_last {
            total / self.world_size
        } else {
            total.div_ceil(self.world_size)
        }
    }
}

impl<S: Sampler> Sampler for DistributedSampler<S> {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        self.inner.set_epoch(self.epoch);
        let all = self.inner.indices(dataset_len);

        let per_rank = self.per_rank(all.len());
        all.iter()
            .copied()
            .cycle()
            .take(per_rank * self.world_size)
            .skip(self.rank)
            .step_by(self.world_size)
            .collect()
    }

    fn len(&self, dataset_len: usize) -> usize {
        self.per_rank(self.inner.len(dataset_len))
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }
}

#[cfg(test)]
mod tests {
    use crate::sampler::{DistributedSampler, RandomSampler, Sampler, SequentialSampler};

    #[test]
    fn distributed_partitions_have_expected_total_len() {
        let world_size = 3;
        let n = 10;
        let mut all_indices: Vec<usize> = Vec::new();
        for rank in 0..world_size {
            let mut ds = DistributedSampler::new(SequentialSampler, rank, world_size);
            all_indices.extend(ds.indices(n));
        }
        assert_eq!(all_indices.len(), 12);
    }

    #[test]
    fn distributed_equal_length_per_rank() {
        let world_size = 4;
        let n = 10;
        let lengths: Vec<usize> = (0..world_size)
            .map(|rank| {
                let mut ds = DistributedSampler::new(SequentialSampler, rank, world_size);
                ds.indices(n).len()
            })
            .collect();
        assert!(lengths.windows(2).all(|w| w[0] == w[1]));
        assert_eq!(lengths[0], 3);
    }

    #[test]
    fn distributed_len_matches_indices() {
        for (n, world_size) in [(10, 3), (10, 4), (8, 4), (1, 4), (0, 2)] {
            for rank in 0..world_size {
                let mut ds = DistributedSampler::new(SequentialSampler, rank, world_size);
                assert_eq!(ds.len(n), ds.indices(n).len(), "n={n} world={world_size}");
            }
        }
    }

    #[test]
    fn distributed_drop_last_truncates_instead_of_padding() {
        let (n, world_size) = (10, 4);
        let mut seen = Vec::new();
        for rank in 0..world_size {
            let mut ds =
                DistributedSampler::new(SequentialSampler, rank, world_size).drop_last(true);
            let indices = ds.indices(n);
            assert_eq!(indices.len(), 2);
            assert_eq!(ds.len(n), 2);
            seen.extend(indices);
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..8).collect::<Vec<_>>(),
            "no index repeated, tail dropped"
        );
    }

    #[test]
    fn distributed_shuffle_is_consistent_across_ranks() {
        // Same seed on every rank: shards are disjoint and cover the dataset.
        let (n, world_size) = (12, 3);
        for epoch in 0..3 {
            let mut seen = Vec::new();
            for rank in 0..world_size {
                let mut ds = DistributedSampler::new(RandomSampler::new(7), rank, world_size);
                ds.set_epoch(epoch);
                seen.extend(ds.indices(n));
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn distributed_order_depends_only_on_epoch() {
        let mut ds = DistributedSampler::new(RandomSampler::new(7), 0, 2);
        let first = ds.indices(20);
        assert_eq!(ds.indices(20), first, "same epoch, same order");
        ds.set_epoch(1);
        let second = ds.indices(20);
        assert_ne!(second, first, "new epoch, new order");
        ds.set_epoch(0);
        assert_eq!(ds.indices(20), first, "order is reproducible per epoch");
    }

    #[test]
    fn distributed_exact_divisor() {
        let world_size = 4;
        let n = 8;
        for rank in 0..world_size {
            let mut ds = DistributedSampler::new(SequentialSampler, rank, world_size);
            assert_eq!(ds.indices(n).len(), 2);
        }
    }

    #[test]
    #[should_panic]
    fn distributed_rank_out_of_bounds_panics() {
        DistributedSampler::new(SequentialSampler, 4, 4);
    }

    #[test]
    #[should_panic]
    fn distributed_zero_world_size_panics() {
        DistributedSampler::new(SequentialSampler, 0, 0);
    }

    #[test]
    fn distributed_wraps_random_sampler() {
        let world_size = 2;
        let n = 6;
        let mut ds = DistributedSampler::new(RandomSampler::new(99), 0, world_size);
        let indices = ds.indices(n);
        assert_eq!(indices.len(), n / world_size);
        assert!(indices.iter().all(|&i| i < n));
    }
}
