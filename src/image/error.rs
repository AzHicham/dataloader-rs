//! [`ImageError`]: why an image transform or the image collator refused an image.

use std::fmt;

/// Why an image transform or [`ImageCollator`](crate::collator::ImageCollator) refused an image.
/// It reaches the caller boxed in [`Error`](crate::Error), for the batch of that image only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// Bytes that are not an image of a supported format, or a corrupt or truncated one.
    Decode(String),
    /// An image above the configured limits, or too large to address or allocate.
    Limits(String),
    /// An image the transform cannot take: wrong type or channel count, a crop out of bounds,
    /// images of different shapes in one batch.
    Invalid(String),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(message) => write!(f, "cannot decode image: {message}"),
            Self::Limits(message) => write!(f, "image too large: {message}"),
            Self::Invalid(message) => write!(f, "invalid image: {message}"),
        }
    }
}

impl std::error::Error for ImageError {}
