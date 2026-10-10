use crate::error::Result;

/// Map-style dataset: random access by index.
///
/// The `Send + Sync + 'static` bounds are required so that an `Arc<dyn Dataset>`
/// (or `Arc<MyDataset>`) can be shared safely across rayon worker threads without
/// copying the dataset.  Implementations typically wrap their data in an `Arc`
/// internally or store a path and open files per-call.
///
/// # Example
///
/// ```rust
/// use dataloader_rs::{Dataset, error::Result};
///
/// struct RangeDataset(usize);
///
/// impl Dataset for RangeDataset {
///     type Item = u64;
///     fn get(&self, index: usize) -> Result<u64> { Ok(index as u64) }
///     fn len(&self) -> usize { self.0 }
/// }
/// ```
pub trait Dataset: Send + Sync + 'static {
    /// The type of a single sample.  Must be `Send + 'static` so it can be
    /// moved across thread boundaries inside the prefetch channel.
    type Item: Send + 'static;

    /// Return the sample at `index`.
    ///
    /// This method may be called concurrently from multiple rayon threads, so
    /// implementations must be re-entrant (shared `&self`).
    fn get(&self, index: usize) -> Result<Self::Item>;

    /// Fetch a batch of samples by index slice.
    ///
    /// The default implementation calls [`get`](Self::get) sequentially.
    /// Override this to amortize per-batch setup costs (e.g. acquiring a
    /// Python GIL token once for all items in a batch rather than per item).
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Item>> {
        indices.iter().map(|&i| self.get(i)).collect()
    }

    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns `true` when the dataset contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Map-style dataset whose samples are fetched asynchronously — for data
/// behind network calls (object stores, HTTP, databases) where many requests
/// should be in flight at once without one thread per request.
///
/// The loader only composes futures (it never spawns tasks or uses timers),
/// so any executor can drive it: tokio, smol, async-std or
/// `futures::executor`. Futures that need a specific runtime (e.g. reqwest or
/// the AWS SDK need tokio) must be polled inside that runtime, i.e. through
/// [`AsyncDataLoader::stream`](crate::AsyncDataLoader::stream) from a task of
/// that runtime.
///
/// # Example
///
/// ```rust
/// use dataloader_rs::{AsyncDataset, error::Result};
///
/// struct Remote;
///
/// impl AsyncDataset for Remote {
///     type Item = u64;
///     async fn get(&self, index: usize) -> Result<u64> {
///         // e.g. `client.get(url(index)).send().await?` in a real dataset
///         Ok(index as u64)
///     }
///     fn len(&self) -> usize { 100 }
/// }
/// ```
#[cfg(feature = "async")]
pub trait AsyncDataset: Send + Sync + 'static {
    /// The type of a single sample.
    type Item: Send + 'static;

    /// Fetch the sample at `index`.
    fn get(&self, index: usize) -> impl Future<Output = Result<Self::Item>> + Send;

    /// Fetch a batch of samples. The default runs every [`get`](Self::get)
    /// of the batch concurrently; override it to use a batched request.
    fn get_batch(&self, indices: &[usize]) -> impl Future<Output = Result<Vec<Self::Item>>> + Send {
        futures::future::try_join_all(indices.iter().map(|&i| self.get(i)))
    }

    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns `true` when the dataset contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Streaming (iterable-style) dataset for sources without random access,
/// such as network streams, stdin, or sharded files read sequentially.
///
/// # Note
///
/// [`DataLoader`](crate::loader::DataLoader) does not yet support
/// `IterableDataset` directly.  This trait is provided for forward
/// compatibility and for users building custom streaming pipelines on top of
/// the sampler/collator primitives.
pub trait IterableDataset: Send + 'static {
    /// The type of a single sample.
    type Item: Send + 'static;

    /// Return an iterator over every sample in the dataset.
    ///
    /// Each call should produce a fresh iterator starting from the beginning.
    fn iter(&self) -> impl Iterator<Item = Result<Self::Item>> + Send + '_;
}
