use crate::sampler::Sampler;

/// Shards indices across `world_size` workers for distributed training.
pub struct DistributedSampler<S: Sampler> {
    inner: S,
    rank: usize,
    world_size: usize,
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
        }
    }
}

impl<S: Sampler> Sampler for DistributedSampler<S> {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        let all = self.inner.indices(dataset_len);

        let padded_len = all.len().div_ceil(self.world_size) * self.world_size;
        let padded: Vec<usize> = all.iter().copied().cycle().take(padded_len).collect();

        padded
            .into_iter()
            .skip(self.rank)
            .step_by(self.world_size)
            .collect()
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
