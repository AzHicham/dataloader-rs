//! Where a [`DataLoader`](crate::DataLoader) gets its items: a [`Dataset`](crate::Dataset) in a
//! [`SyncSource`], or an `AsyncDataset` and the tokio runtime that fetches its items in an
//! `AsyncSource` (feature `async`).

#[cfg(feature = "async")]
mod asynchronous;
mod loader_source;
mod sealed;
mod settings;
mod sync;
mod threads;

#[cfg(feature = "async")]
pub use asynchronous::AsyncSource;
pub use loader_source::LoaderSource;
pub use settings::EpochSettings;
pub use sync::SyncSource;
