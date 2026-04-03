use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use crate::sampler::Sampler;

/// Yields a uniformly shuffled permutation of indices every epoch.
#[derive(Debug)]
pub struct RandomSampler {
    rng: SmallRng,
}

impl RandomSampler {
    /// Deterministic seed. Use this for reproducible experiments.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Seed from OS entropy. Use this when reproducibility is not required.
    pub fn from_entropy() -> Self {
        Self {
            rng: SmallRng::from_entropy(),
        }
    }
}

impl Sampler for RandomSampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        indices.shuffle(&mut self.rng);
        indices
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::sampler::{RandomSampler, Sampler};

    #[test]
    fn random_same_seed_same_order() {
        let mut a = RandomSampler::new(42);
        let mut b = RandomSampler::new(42);
        assert_eq!(a.indices(20), b.indices(20));
    }

    #[test]
    fn random_different_seeds_different_order() {
        let mut a = RandomSampler::new(1);
        let mut b = RandomSampler::new(2);
        assert_ne!(a.indices(20), b.indices(20));
    }

    #[test]
    fn random_is_permutation() {
        let mut s = RandomSampler::new(0);
        let n = 50;
        let indices = s.indices(n);
        assert_eq!(indices.len(), n);
        let set: HashSet<usize> = indices.into_iter().collect();
        assert_eq!(set.len(), n);
        assert!(set.iter().all(|&i| i < n));
    }

    #[test]
    fn random_advances_rng_across_epochs() {
        let mut s = RandomSampler::new(7);
        let epoch1 = s.indices(30);
        let epoch2 = s.indices(30);
        assert_ne!(epoch1, epoch2);
    }

    #[test]
    fn random_empty_dataset() {
        let mut s = RandomSampler::new(0);
        assert_eq!(s.indices(0), Vec::<usize>::new());
    }
}
