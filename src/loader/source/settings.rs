/// How an epoch runs: batches in flight, items in flight per batch, batches buffered.
pub struct EpochSettings {
    pub(in crate::loader) num_workers: usize,
    pub(in crate::loader) item_concurrency: usize,
    pub(in crate::loader) prefetch_depth: usize,
}
