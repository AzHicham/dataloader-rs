mod batch_sampler;
mod distributed;
mod random;
mod sequential;

pub use batch_sampler::BatchSampler;
pub use distributed::DistributedSampler;
pub use random::RandomSampler;
pub use sequential::SequentialSampler;

/// Produces an ordered sequence of indices for one full pass over a dataset.
pub trait Sampler: Send + 'static {
    /// Return the index sequence for one epoch over a dataset of `dataset_len`
    /// items.
    fn indices(&mut self, dataset_len: usize) -> Vec<usize>;

    /// Number of indices [`indices`](Self::indices) returns for a dataset of
    /// `dataset_len` items, without advancing the sampler.
    ///
    /// Defaults to `dataset_len`; override it for samplers that subsample,
    /// shard, or oversample.
    fn len(&self, dataset_len: usize) -> usize {
        dataset_len
    }
}
