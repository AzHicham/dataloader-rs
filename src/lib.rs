//! High-performance DataLoader for Rust with a PyTorch-like interface.
//!
//! # Overview
//!
//! This crate provides a typed, streaming batch pipeline with four independent
//! stages connected by a bounded async channel:
//!
//! ```text
//! Sampler ──▶ index chunks ──▶ prefetch thread (rayon pool)
//!                                    │
//!                              collate batch
//!                                    │
//!                            bounded channel
//!                                    │
//!                            DataLoaderIter (consumer)
//! ```
//!
//! ## Advantages over PyTorch's DataLoader
//!
//! | PyTorch problem | Rust solution |
//! |---|---|
//! | GIL blocks parallel transforms | Rayon: true parallelism, no GIL |
//! | `pickle` IPC serialization | Shared `Arc<Dataset>` — zero-copy |
//! | `fork`-based worker spawning | Long-lived thread pool |
//! | Fixed `prefetch_factor` granularity | Bounded channel (exact depth) |
//! | Python overhead per `__getitem__` | Monomorphized `Dataset::get` |
//!
//! # Quick start
//!
//! ```rust
//! use dataloader_rs::{Dataset, DataLoader, RandomSampler, error::Result};
//!
//! struct MyDataset;
//!
//! impl Dataset for MyDataset {
//!     type Item = u32;
//!     fn get(&self, index: usize) -> Result<u32> { Ok(index as u32) }
//!     fn len(&self) -> usize { 1000 }
//! }
//!
//! let mut loader = DataLoader::builder(MyDataset)
//!     .batch_size(32)
//!     .sampler(RandomSampler::new(42))
//!     .num_workers(4)
//!     .prefetch_depth(8)
//!     .drop_last(true)
//!     .build();
//!
//! for batch in &mut loader {
//!     let batch: Vec<u32> = batch.unwrap();
//!     // pass to model, write to disk, etc.
//!     let _ = batch;
//! }
//! ```
//!
//! # Future Python bindings
//!
//! The generic `DataLoader<D, S, C>` is the Rust-facing API.  Future PyO3
//! bindings will introduce a type-erased `PyDataLoader` that wraps a
//! `Box<dyn Iterator<Item = Result<PyObject>> + Send>`.  The design here
//! deliberately avoids lifetime parameters on public structs and requires
//! `'static` bounds everywhere so that wrapping is straightforward.

// All modules other than `loader` must be unsafe-free.
// `loader` uses intentional, documented unsafe (raw-pointer sharing with the
// prefetch thread) and opts in with `#[allow(unsafe_code)]` at the top of
// that file.
#![deny(unsafe_code)]

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
