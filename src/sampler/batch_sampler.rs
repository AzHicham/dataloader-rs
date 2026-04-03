use crate::sampler::Sampler;

/// Batches indices produced by an inner [`Sampler`].
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Build a batch sampler from an index sampler.
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn drop_last(&self) -> bool {
        self.drop_last
    }

    pub fn set_batch_size(&mut self, batch_size: usize) {
        assert!(batch_size > 0, "batch_size must be > 0");
        self.batch_size = batch_size;
    }

    pub fn set_drop_last(&mut self, drop_last: bool) {
        self.drop_last = drop_last;
    }

    /// Return one epoch of grouped batch indices.
    pub fn batch_indices(&mut self, dataset_len: usize) -> Vec<Vec<usize>> {
        let indices = self.sampler.indices(dataset_len);
        let raw = indices.chunks(self.batch_size);
        if self.drop_last {
            raw.filter(|chunk| chunk.len() == self.batch_size)
                .map(<[usize]>::to_vec)
                .collect()
        } else {
            raw.map(<[usize]>::to_vec).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sampler::{BatchSampler, Sampler, SequentialSampler};

    #[test]
    fn batch_sampler_respects_batch_size() {
        let mut bs = BatchSampler::new(SequentialSampler, 3, false);
        let batches = bs.batch_indices(8);
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7]]);
    }

    #[test]
    fn batch_sampler_drop_last() {
        let mut bs = BatchSampler::new(SequentialSampler, 3, true);
        let batches = bs.batch_indices(8);
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn batch_sampler_uses_inner_sampler_order() {
        struct ReverseSampler;
        impl Sampler for ReverseSampler {
            fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
                (0..dataset_len).rev().collect()
            }
        }

        let mut bs = BatchSampler::new(ReverseSampler, 2, false);
        let batches = bs.batch_indices(5);
        assert_eq!(batches, vec![vec![4, 3], vec![2, 1], vec![0]]);
    }

    #[test]
    fn batch_sampler_setters_work() {
        let mut bs = BatchSampler::new(SequentialSampler, 2, false);
        assert_eq!(bs.batch_indices(5), vec![vec![0, 1], vec![2, 3], vec![4]]);
        bs.set_drop_last(true);
        assert_eq!(bs.batch_indices(5), vec![vec![0, 1], vec![2, 3]]);
        bs.set_batch_size(3);
        assert_eq!(bs.batch_size(), 3);
    }
}
