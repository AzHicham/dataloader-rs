#![doc = include_str!("README.md")]

mod dataset;
mod error;
mod file;
mod http;
mod memory;

pub use dataset::SourceDataset;
pub use error::SourceError;
pub use file::FileSource;
pub use http::{HttpSource, HttpSourceBuilder, Retry};
pub use memory::MemorySource;

use std::io::Read;

/// Default cap on the size of one item, for [`FileSource`] and [`HttpSource`]: 64 MiB.
pub const DEFAULT_MAX_BYTES: u64 = 64 << 20;

/// Where the bytes of each item come from: memory, files, HTTP, or your own.
///
/// A source is indexed like a [`Dataset`](crate::Dataset); [`SourceDataset`] turns it into
/// one that yields `(bytes, meta)` items, which [`ImageCollator`] accepts.
///
/// `fetch` may be called from several worker threads at once.
///
/// [`ImageCollator`]: https://docs.rs/dataloader-rs/latest/dataloader_rs/image/struct.ImageCollator.html
pub trait Source: Send + Sync + 'static {
    /// The bytes of one item.
    type Bytes: AsRef<[u8]> + Send + 'static;

    /// The bytes of item `index`.
    fn fetch(&self, index: usize) -> Result<Self::Bytes, SourceError>;

    /// Number of items.
    fn len(&self) -> usize;

    /// Whether the source has no item.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Why [`read_capped`] stopped.
#[derive(Debug)]
pub(crate) enum ReadError {
    /// More than the cap, or memory for it cannot be reserved; the announced size if any.
    TooLarge(Option<u64>),
    Io(std::io::Error),
}

/// Read `reader` to its end into a new `Vec`, failing as soon as it has more than `limit`
/// bytes. `size_hint` (a file length or a `Content-Length`) is checked first, and memory for
/// it, at most `limit`, is reserved fallibly up front.
pub(crate) fn read_capped(
    reader: impl Read,
    size_hint: Option<u64>,
    limit: u64,
) -> Result<Vec<u8>, ReadError> {
    if let Some(size) = size_hint
        && size > limit
    {
        return Err(ReadError::TooLarge(Some(size)));
    }
    let mut bytes = Vec::new();
    let reserve = usize::try_from(size_hint.unwrap_or(0).min(limit))
        .map_err(|_| ReadError::TooLarge(size_hint))?;
    bytes
        .try_reserve_exact(reserve)
        .map_err(|_| ReadError::TooLarge(size_hint))?;
    reader
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(ReadError::Io)?;
    if bytes.len() as u64 > limit {
        return Err(ReadError::TooLarge(size_hint));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_capped_checks_hint_and_actual_size() {
        assert_eq!(read_capped(&b"abcd"[..], Some(4), 4).unwrap(), b"abcd");
        assert_eq!(read_capped(&b"abcd"[..], None, 4).unwrap(), b"abcd");
        assert!(matches!(
            read_capped(&b"abcd"[..], Some(5), 4),
            Err(ReadError::TooLarge(Some(5)))
        ));
        // The hint lies: the reader has more.
        assert!(matches!(
            read_capped(&b"abcde"[..], Some(1), 4),
            Err(ReadError::TooLarge(Some(1)))
        ));
        assert!(matches!(
            read_capped(std::io::repeat(0), None, 1 << 10),
            Err(ReadError::TooLarge(None))
        ));
        // A huge hint under a huge limit cannot be reserved: an error, not an abort.
        assert!(matches!(
            read_capped(&b""[..], Some(u64::MAX / 2), u64::MAX),
            Err(ReadError::TooLarge(_))
        ));
    }
}
