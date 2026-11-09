#![doc = include_str!("README.md")]

#[cfg(feature = "async")]
mod async_epoch;
mod builder;
mod core;
mod iter;
mod source;
mod worker;

#[cfg(feature = "async")]
pub use async_epoch::MAX_RUNTIME_THREADS;
pub use builder::DataLoaderBuilder;
pub use core::DataLoader;
pub use iter::DataLoaderIter;
#[cfg(feature = "async")]
pub use source::AsyncSource;
pub use source::{LoaderSource, SyncSource};
