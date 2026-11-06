//! [`SourceError`]: every error of the data sources.

use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// An error of a [`Source`](super::Source) or a [`SourceDataset`](super::SourceDataset): an
/// invalid configuration when it is built, or an item that cannot be fetched.
///
/// Errors about an item name its index in the source and where it was read from. URLs are shown
/// without their user info, query and fragment, which may hold credentials. The loader reports
/// the error boxed in [`crate::Error`], from which it can be downcast.
#[derive(Debug)]
#[non_exhaustive]
pub enum SourceError {
    /// The configuration is invalid (a URL, a template, a header, a limit): reported when the
    /// source is built, before anything is fetched.
    Invalid(String),
    /// `index` is not below the number of items.
    OutOfRange {
        /// The index asked for.
        index: usize,
        /// The number of items.
        len: usize,
    },
    /// The metadata list and the source have different lengths.
    MetaLength {
        /// Items in the source.
        items: usize,
        /// Entries in the metadata list.
        meta: usize,
    },
    /// The item is larger than the source's `max_bytes`, or its memory cannot be reserved.
    TooLarge {
        /// Index of the item.
        index: usize,
        /// The path or URL of the item.
        location: String,
        /// The cap, in bytes.
        limit: u64,
        /// The size the file or the server announced, if any.
        size: Option<u64>,
    },
    /// A file cannot be read.
    File {
        /// Index of the item.
        index: usize,
        /// The file.
        path: PathBuf,
        /// What went wrong.
        source: io::Error,
    },
    /// The server answered with a status other than 2xx, after `attempts` attempts.
    Status {
        /// Index of the item.
        index: usize,
        /// The URL, without user info, query or fragment.
        url: String,
        /// The last HTTP status.
        status: u16,
        /// Requests sent.
        attempts: u32,
    },
    /// The request failed (connection, timeout, malformed or truncated response), after
    /// `attempts` attempts.
    Request {
        /// Index of the item.
        index: usize,
        /// The URL, without user info, query or fragment.
        url: String,
        /// Requests sent.
        attempts: u32,
        /// Whether the last attempt ran out of time.
        timeout: bool,
        /// The last error.
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl SourceError {
    /// Index of the item the error is about, if any.
    pub fn index(&self) -> Option<usize> {
        match self {
            Self::OutOfRange { index, .. }
            | Self::TooLarge { index, .. }
            | Self::File { index, .. }
            | Self::Status { index, .. }
            | Self::Request { index, .. } => Some(*index),
            Self::Invalid(_) | Self::MetaLength { .. } => None,
        }
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid source: {message}"),
            Self::OutOfRange { index, len } => {
                write!(
                    f,
                    "item {index} is out of range: the source has {len} items"
                )
            }
            Self::MetaLength { items, meta } => write!(
                f,
                "the source has {items} items but the metadata list has {meta} entries"
            ),
            Self::TooLarge {
                index,
                location,
                limit,
                size,
            } => match size {
                Some(size) => write!(
                    f,
                    "item {index} ({location}): {size} bytes, above the limit of {limit} bytes"
                ),
                None => write!(
                    f,
                    "item {index} ({location}): above the limit of {limit} bytes"
                ),
            },
            Self::File {
                index,
                path,
                source,
            } => write!(f, "item {index} ({}): {source}", path.display()),
            Self::Status {
                index,
                url,
                status,
                attempts,
            } => write!(
                f,
                "item {index} ({url}): HTTP status {status} after {attempts} attempt(s)"
            ),
            Self::Request {
                index,
                url,
                attempts,
                timeout,
                source,
            } => {
                let what = if *timeout {
                    "timed out"
                } else {
                    "request failed"
                };
                write!(
                    f,
                    "item {index} ({url}): {what} after {attempts} attempt(s): {source}"
                )
            }
        }
    }
}

impl StdError for SourceError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::File { source, .. } => Some(source),
            Self::Request { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
