//! [`SourceDataset`]: a [`Source`] with per-item metadata, as a [`Dataset`].

use crate::{Dataset, error::Result};

use super::{Source, SourceError};

/// A [`Dataset`] over a [`Source`]: item `i` is `(bytes, meta)`, the bytes fetched from the
/// source and a clone of the metadata of item `i` (a label, an id, coordinates), passed through
/// untouched.
///
/// With [`ImageCollator`], the metadata ends up in the batch's `SampleInfo::meta`.
///
/// Items are fetched one per call: to have several requests or reads in flight per batch, set
/// the loader's `items_in_parallel`.
///
/// [`ImageCollator`]: https://docs.rs/dataloader-rs/latest/dataloader_rs/image/struct.ImageCollator.html
#[derive(Clone, Debug)]
pub struct SourceDataset<S, M = ()> {
    source: S,
    meta: Vec<M>,
}

impl<S: Source> SourceDataset<S> {
    /// A dataset of `source`, with no metadata (`()`).
    pub fn new(source: S) -> Self {
        let meta = vec![(); source.len()];
        Self { source, meta }
    }
}

impl<S: Source, M> SourceDataset<S, M> {
    /// A dataset of `source` with `meta[i]` for item `i`; both must have the same length.
    pub fn with_meta(source: S, meta: Vec<M>) -> std::result::Result<Self, SourceError> {
        if meta.len() != source.len() {
            return Err(SourceError::MetaLength {
                items: source.len(),
                meta: meta.len(),
            });
        }
        Ok(Self { source, meta })
    }

    /// The source.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// The metadata, one entry per item.
    pub fn meta(&self) -> &[M] {
        &self.meta
    }
}

impl<S, M> Dataset for SourceDataset<S, M>
where
    S: Source,
    M: Clone + Send + Sync + 'static,
{
    type Item = (S::Bytes, M);

    fn get(&self, index: usize) -> Result<Self::Item> {
        let meta = self.meta.get(index).ok_or(SourceError::OutOfRange {
            index,
            len: self.meta.len(),
        })?;
        let bytes = self.source.fetch(index)?;
        Ok((bytes, meta.clone()))
    }

    fn len(&self) -> usize {
        self.meta.len()
    }
}
