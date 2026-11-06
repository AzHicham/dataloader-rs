//! [`ImageError`]: every error of the image pipeline.

use std::collections::TryReserveError;
use std::fmt;

use std::io;

use ::image::ImageError as CodecError;
use ::image::error::LimitError;
use fast_image_resize::{ImageBufferError, ResizeError};

/// An error of the image pipeline: an invalid pipeline when it is built, or a sample that cannot
/// be decoded or transformed when a batch is collated.
///
/// In a batch, the error of a sample is wrapped in [`Sample`](Self::Sample), which names it;
/// [`kind`](Self::kind) looks through it. The loader reports it boxed in
/// [`crate::Error`], from which it can be downcast.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageError {
    /// The pipeline is invalid: reported when it is built, before any image is read.
    InvalidPipeline(String),
    /// The bytes are not a supported image (PNG, JPEG, WebP), or are corrupt or truncated.
    Decode(String),
    /// The image, or a working image of the pipeline, is above the [`ImageLimits`], or its
    /// memory cannot be allocated.
    ///
    /// [`ImageLimits`]: super::ImageLimits
    Limits(String),
    /// The image does not fit the pipeline: another size than `input_size`, a crop that leaves
    /// it, a pad smaller than it.
    Size(String),
    /// The batch buffer cannot hold the batch: too small, or of another dtype.
    Buffer(String),
    /// The error of sample `index` of a batch.
    Sample {
        /// Position of the sample in its batch.
        index: usize,
        /// What went wrong.
        source: Box<ImageError>,
    },
}

impl ImageError {
    /// The error, through any [`Sample`](Self::Sample) wrapper.
    pub fn kind(&self) -> &Self {
        match self {
            Self::Sample { source, .. } => source.kind(),
            other => other,
        }
    }

    /// Position in its batch of the sample that failed, if the error is about one.
    pub fn sample(&self) -> Option<usize> {
        match self {
            Self::Sample { index, .. } => Some(*index),
            _ => None,
        }
    }

    pub(crate) fn in_sample(self, index: usize) -> Self {
        Self::Sample {
            index,
            source: Box::new(self),
        }
    }
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPipeline(m) => write!(f, "invalid image pipeline: {m}"),
            Self::Decode(m) => write!(f, "cannot decode image: {m}"),
            Self::Limits(m) | Self::Size(m) | Self::Buffer(m) => f.write_str(m),
            Self::Sample { index, source } => write!(f, "image {index} of the batch: {source}"),
        }
    }
}

impl std::error::Error for ImageError {}

impl From<CodecError> for ImageError {
    fn from(err: CodecError) -> Self {
        match err {
            CodecError::Limits(e) => Self::Limits(limit_message(&e)),
            other => Self::Decode(other.to_string()),
        }
    }
}

fn limit_message(e: &LimitError) -> String {
    format!("image decoder: {e}")
}

impl From<io::Error> for ImageError {
    fn from(err: io::Error) -> Self {
        Self::Decode(err.to_string())
    }
}

impl From<TryReserveError> for ImageError {
    fn from(err: TryReserveError) -> Self {
        Self::Limits(format!("cannot allocate the image: {err}"))
    }
}

impl From<ImageBufferError> for ImageError {
    fn from(err: ImageBufferError) -> Self {
        Self::Size(format!("resize: {err}"))
    }
}

impl From<ResizeError> for ImageError {
    fn from(err: ResizeError) -> Self {
        Self::Size(format!("resize: {err}"))
    }
}

/// `Result` of the image pipeline.
pub type ImageResult<T> = std::result::Result<T, ImageError>;
