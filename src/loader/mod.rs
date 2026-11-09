#![doc = include_str!("README.md")]

#[cfg(feature = "async")]
mod asynchronous;
mod builder;
mod dataloader;
mod iter;
mod pool;
mod source;
mod sync;

#[cfg(feature = "async")]
pub use asynchronous::MAX_RUNTIME_THREADS;
pub use builder::DataLoaderBuilder;
pub use dataloader::DataLoader;
pub use iter::{DataLoaderIter, IndexedIter};
#[cfg(feature = "async")]
pub use source::AsyncSource;
pub use source::{LoaderSource, SyncSource};
