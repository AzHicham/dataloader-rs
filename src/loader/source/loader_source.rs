use super::sealed::Sealed;

/// The dataset of a [`DataLoader`](crate::DataLoader), as the loader holds it: a
/// [`SyncSource`](super::SyncSource) from [`DataLoader::builder`](crate::DataLoader::builder),
/// or an `AsyncSource` from `DataLoader::builder_async` (feature `async`). Sealed: these two
/// are the only sources.
pub trait LoaderSource: Sealed {
    /// The dataset: a [`Dataset`](crate::Dataset) or an `AsyncDataset`.
    type Dataset;

    /// The dataset.
    fn dataset(&self) -> &Self::Dataset;
}
