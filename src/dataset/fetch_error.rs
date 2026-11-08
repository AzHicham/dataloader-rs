//! [`FetchError`]: why a built-in dataset could not fetch an item.

use std::error::Error as StdError;
use std::fmt::{self, Display};

/// Why [`FileDataset`](super::FileDataset), `HttpDataset` or `ObjectStoreDataset` could not
/// fetch an item. Locations in messages never include a URL's query or credentials.
#[derive(Debug)]
#[non_exhaustive]
pub enum FetchError {
    /// The index is not below the dataset's length.
    OutOfRange { index: usize, len: usize },
    /// The item is larger than the dataset's `max_bytes`; `size` is the announced size, when
    /// known before reading.
    TooLarge {
        location: String,
        limit: u64,
        size: Option<u64>,
    },
    /// The server answered with a status other than 2xx, after `attempts` attempts.
    Status {
        url: String,
        status: u16,
        attempts: u32,
    },
    /// The request or read failed (connection, timeout, I/O), after `attempts` attempts.
    Io {
        location: String,
        attempts: u32,
        timeout: bool,
        source: Box<dyn StdError + Send + Sync>,
    },
    /// An object store request failed after the store's own retries: at most `max_attempts`
    /// attempts, when known (a store opened by `ObjectStoreDataset::from_url`).
    Store {
        location: String,
        max_attempts: Option<u32>,
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl FetchError {
    /// Whether the last attempt ran out of time.
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Io { timeout: true, .. })
    }
}

impl Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { index, len } => {
                write!(
                    f,
                    "index {index} is out of range for a dataset of {len} items"
                )
            }
            Self::TooLarge {
                location,
                limit,
                size: Some(size),
            } => write!(
                f,
                "{location}: {size} bytes, more than the limit of {limit}"
            ),
            Self::TooLarge {
                location, limit, ..
            } => write!(f, "{location}: more than the limit of {limit} bytes"),
            Self::Status {
                url,
                status,
                attempts,
            } => write!(f, "{url}: HTTP status {status} after {attempts} attempt(s)"),
            Self::Io {
                location,
                attempts,
                timeout,
                source,
            } => {
                let what = if *timeout { "timed out" } else { "failed" };
                write!(
                    f,
                    "{location}: {what} after {attempts} attempt(s): {source}"
                )
            }
            Self::Store {
                location,
                max_attempts: Some(max_attempts),
                source,
            } => write!(
                f,
                "{location}: failed after up to {max_attempts} attempt(s), \
                 retried by object_store: {source}"
            ),
            Self::Store {
                location, source, ..
            } => write!(
                f,
                "{location}: failed, retried by the store's own settings: {source}"
            ),
        }
    }
}

impl StdError for FetchError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } | Self::Store { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
