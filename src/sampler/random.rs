use crate::sampler::Sampler;

/// Yields a uniformly shuffled permutation of indices every epoch.
///
/// Uses the inside-out Fisher-Yates algorithm with `fastrand` (wyrand):
/// a single-multiply PRNG that is ~30% faster than Xoshiro256++ for this
/// workload. The inside-out variant builds and shuffles in a single forward
/// pass — no pre-fill, and random accesses stay within the already-written
/// prefix `[0..i]` for better cache behaviour.
#[derive(Debug)]
pub struct RandomSampler {
    rng: fastrand::Rng,
}

impl RandomSampler {
    /// Deterministic seed. Use this for reproducible experiments.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
        }
    }

    /// Seed from OS entropy. Use this when reproducibility is not required.
    pub fn from_entropy() -> Self {
        Self {
            rng: fastrand::Rng::new(),
        }
    }
}

impl Sampler for RandomSampler {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(dataset_len);
        for i in 0..dataset_len {
            // Inside-out Fisher-Yates: pick j in [0, i].
            // If j == i: push i.  Otherwise: copy out[j] to end, write i to out[j].
            let j = self.rng.usize(0..=i);
            if j == out.len() {
                out.push(i);
            } else {
                out.push(out[j]);
                out[j] = i;
            }
        }
        out
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
