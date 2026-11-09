//! Sizing of the loader's rayon pool, shared by both sources.

#[cfg(feature = "async")]
use std::thread;

/// `max(num_workers, 1) × item_concurrency`.
pub(super) fn pool_threads(num_workers: usize, item_concurrency: usize) -> usize {
    num_workers
        .max(1)
        .checked_mul(item_concurrency)
        .expect("num_workers × item_concurrency overflows usize")
}

/// The CPUs available to the process (affinity and cgroup quota included), 1 if unknown.
#[cfg(feature = "async")]
pub(super) fn available_cpus() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}
