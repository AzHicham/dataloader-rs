//! PyTorch's `DataLoader` model — dataset, sampler, collator, loader — as a typed Rust library.
//!
//! A [`DataLoader`] owns a [`Dataset`], a [`Sampler`] and a [`Collator`]; each
//! [`iter`](DataLoader::iter) runs one epoch and yields `Result<Batch>` in epoch order, loading
//! batches inline (`num_workers = 0`) or as tasks on the loader's rayon pool, optionally with the
//! items of each batch fetched in parallel on the same pool (`parallel_items`).
//!
//! ```rust
//! use dataloader_rs::{DataLoader, Dataset, RandomSampler, error::Result};
//!
//! struct Squares(usize);
//!
//! impl Dataset for Squares {
//!     type Item = u64;
//!     fn get(&self, index: usize) -> Result<u64> {
//!         Ok((index as u64).pow(2))
//!     }
//!     fn len(&self) -> usize {
//!         self.0
//!     }
//! }
//!
//! let mut loader = DataLoader::builder(Squares(1000))
//!     .batch_size(32)
//!     .sampler(RandomSampler::new(42))
//!     .num_workers(4)
//!     .prefetch_depth(8)
//!     .drop_last(true)
//!     .build();
//!
//! for batch in &mut loader {
//!     let batch: Vec<u64> = batch.unwrap();
//!     assert_eq!(batch.len(), 32);
//! }
//! ```
//!
//! | Module | Content |
//! |---|---|
//! | [`dataset`] | the [`Dataset`] contract |
//! | [`sampler`] | index order: sequential, random, distributed, batching |
//! | [`collator`] | batch assembly: `Vec`, PyTorch-like `DefaultCollator`, pinned tensors |
//! | [`loader`] | [`DataLoader`], its builder and iterator, the worker threads |
//! | [`error`] | [`Error`] and [`Result`] |
//!
//! Features: `ndarray` and `torch-rs` add collation of `ndarray::Array` and `tch::Tensor`;
//! `python` builds the PyO3 module behind the `dataloader-rs` Python package.
//!
//! Design, threads and failure handling:
//! [ARCHITECTURE.md](https://github.com/AzHicham/dataloader-rs/blob/dev/ARCHITECTURE.md).
//! Benchmarks against PyTorch:
//! [BENCH.md](https://github.com/AzHicham/dataloader-rs/blob/dev/BENCH.md).

// The dataset and collator are shared with worker threads through `Arc`, so
// the crate needs no unsafe code at all.
#![forbid(unsafe_code)]

pub mod collator;
pub mod dataset;
pub mod error;
pub mod loader;
#[cfg(feature = "python")]
pub mod python;
pub mod sampler;

// ── Top-level re-exports ──────────────────────────────────────────────────────

pub use collator::{Collator, VecCollator};
pub use dataset::{Dataset, IterableDataset};
pub use error::{Error, Result};
pub use loader::{DataLoader, DataLoaderBuilder, DataLoaderIter};
pub use sampler::{BatchSampler, DistributedSampler, RandomSampler, Sampler, SequentialSampler};
