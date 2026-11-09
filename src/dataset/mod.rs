#[cfg(feature = "async")]
mod asynchronous;
#[cfg(feature = "async")]
mod io;
mod sync;

#[cfg(feature = "async")]
pub use asynchronous::AsyncDataset;
#[cfg(feature = "async")]
pub use io::{Column, DEFAULT_MAX_BYTES, FetchError, FileDataset, FileDatasetBuilder, Retry};
#[cfg(feature = "http")]
pub use io::{HttpDataset, HttpDatasetBuilder};
#[cfg(feature = "object-store")]
pub use io::{ObjectStoreDataset, ObjectStoreDatasetBuilder};
pub use sync::{Dataset, IterableDataset};
