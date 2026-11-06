//! [`MemorySource`]: items already in memory.

use std::sync::Arc;

use super::{Source, SourceError};

/// Items already in memory, e.g. downloaded or packed beforehand. `fetch` clones the item, so
/// use a cheaply cloned type: `Arc<[u8]>` (the default), `bytes::Bytes`, `&'static [u8]`.
#[derive(Clone, Debug)]
pub struct MemorySource<B = Arc<[u8]>> {
    items: Vec<B>,
}

impl<B> MemorySource<B> {
    /// A source of `items`, in order.
    pub fn new(items: impl IntoIterator<Item = B>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }

    /// The items.
    pub fn items(&self) -> &[B] {
        &self.items
    }
}

impl<B> Source for MemorySource<B>
where
    B: AsRef<[u8]> + Clone + Send + Sync + 'static,
{
    type Bytes = B;

    fn fetch(&self, index: usize) -> Result<B, SourceError> {
        self.items
            .get(index)
            .cloned()
            .ok_or(SourceError::OutOfRange {
                index,
                len: self.items.len(),
            })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
