// This module contains intentional unsafe code for zero-overhead dataset
// sharing with the prefetch thread.
#![allow(unsafe_code)]

mod builder;
mod core;
mod iter;
mod prefetch;

pub use builder::DataLoaderBuilder;
pub use core::DataLoader;
pub use iter::DataLoaderIter;
#[cfg(feature = "python")]
pub(crate) use iter::OwnedDataLoaderIter;
