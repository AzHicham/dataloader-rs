use crate::sampler::Sampler;

/// Yields indices `0, 1, ..., N-1` in order every epoch.
#[derive(Clone, Copy, Debug, Default)]
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        (0..dataset_len).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::sampler::{Sampler, SequentialSampler};

    #[test]
    fn sequential_returns_0_to_n() {
        let mut s = SequentialSampler;
        assert_eq!(s.indices(5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn sequential_empty_dataset() {
        let mut s = SequentialSampler;
        assert_eq!(s.indices(0), Vec::<usize>::new());
    }

    #[test]
    fn sequential_is_repeatable() {
        let mut s = SequentialSampler;
        assert_eq!(s.indices(3), s.indices(3));
    }

    #[test]
    fn sequential_len_matches() {
        let mut s = SequentialSampler;
        let n = 100;
        assert_eq!(s.indices(n).len(), n);
    }
}
