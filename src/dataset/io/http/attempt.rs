//! [`Attempt`]: how one request of an [`HttpDataset`](super::HttpDataset) failed, and whether
//! another attempt may succeed.

use std::time::Duration;

use reqwest::header::RETRY_AFTER;
use reqwest::{Error as ReqwestError, Response, StatusCode};

use crate::dataset::{FetchError, io::retry::Transient};

/// Statuses worth another attempt: timeout, rate limit and the transient server errors.
const TRANSIENT_STATUS: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// How one attempt failed.
pub(super) enum Attempt {
    Status {
        status: StatusCode,
        retry_after: Option<Duration>,
    },
    TooLarge(Option<u64>),
    Request(ReqwestError),
}

impl Attempt {
    /// A response with a status other than 2xx.
    pub(super) fn status(response: &Response) -> Self {
        Self::Status {
            status: response.status(),
            retry_after: retry_after(response),
        }
    }

    /// The error of the item, at `location` (redacted), after `attempts` attempts.
    pub(super) fn into_error(self, location: String, attempts: u32, limit: u64) -> FetchError {
        match self {
            Self::Status { status, .. } => FetchError::Status {
                url: location,
                status: status.as_u16(),
                attempts,
            },
            Self::TooLarge(size) => FetchError::TooLarge {
                location,
                limit,
                size,
            },
            Self::Request(error) => FetchError::Io {
                location,
                attempts,
                timeout: error.is_timeout(),
                // reqwest's message repeats the full URL: keep its cause only.
                source: error.without_url().into(),
            },
        }
    }
}

impl Transient for Attempt {
    /// A connection error, a timeout, a body cut short, or status 408, 429, 500, 502, 503
    /// or 504.
    fn is_transient(&self) -> bool {
        match self {
            Self::Status { status, .. } => TRANSIENT_STATUS.contains(&status.as_u16()),
            Self::TooLarge(_) => false,
            Self::Request(error) => {
                error.is_timeout() || error.is_connect() || error.is_request() || error.is_body()
            }
        }
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// A `Retry-After` in seconds, if the response has one.
fn retry_after(response: &Response) -> Option<Duration> {
    let value = response.headers().get(RETRY_AFTER)?.to_str().ok()?;
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}
