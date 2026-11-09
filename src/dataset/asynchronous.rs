//! [`AsyncDataset`]: a map-style dataset whose items are fetched by futures.

use crate::error::Result;

/// Map-style dataset whose items are fetched asynchronously: network, object storage, files.
///
/// [`DataLoader::builder_async`](crate::DataLoader::builder_async) runs `get` on a tokio
/// runtime, with up to `num_workers × item_concurrency` items in flight, instead of tying up
/// one thread per item as a blocking [`Dataset`](crate::Dataset) does.
///
/// ```rust
/// use dataloader_rs::{AsyncDataset, error::Result};
///
/// struct Doubled(usize);
///
/// impl AsyncDataset for Doubled {
///     type Item = u64;
///
///     async fn get(&self, index: usize) -> Result<u64> {
///         // e.g. an HTTP request, awaited without blocking a thread
///         Ok(index as u64 * 2)
///     }
///
///     fn len(&self) -> usize {
///         self.0
///     }
/// }
/// ```
pub trait AsyncDataset: Send + Sync + 'static {
    /// The type of a single sample.
    type Item: Send + 'static;

    /// Fetch the sample at `index`. Many calls are in flight at once; dropping the future
    /// cancels the fetch. Return an error, not a panic, for an index out of range.
    fn get(&self, index: usize) -> impl Future<Output = Result<Self::Item>> + Send;

    /// Fetch a batch of samples, used when `item_concurrency` is `1`.
    ///
    /// The default awaits [`get`](Self::get) for each index in turn. Override it to fetch a
    /// whole batch in one request.
    fn get_batch(&self, indices: &[usize]) -> impl Future<Output = Result<Vec<Self::Item>>> + Send {
        async move {
            let mut items = Vec::with_capacity(indices.len());
            for &index in indices {
                items.push(self.get(index).await?);
            }
            Ok(items)
        }
    }

    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns `true` when the dataset contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::AsyncDataset;
    use crate::error::Result;

    struct Numbers(usize);

    impl AsyncDataset for Numbers {
        type Item = u64;

        async fn get(&self, index: usize) -> Result<u64> {
            tokio::task::yield_now().await;
            if index < self.0 {
                Ok(index as u64)
            } else {
                Err(format!("index {index} out of range").into())
            }
        }

        fn len(&self) -> usize {
            self.0
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn default_get_batch_awaits_get_in_order() {
        assert_eq!(block_on(Numbers(4).get_batch(&[3, 0])).unwrap(), vec![3, 0]);
        assert!(block_on(Numbers(4).get_batch(&[1, 7])).is_err());
    }
}
