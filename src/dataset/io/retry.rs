//! [`Retry`]: how often and after how long a built-in dataset tries an item again. Which
//! failures are worth another attempt is up to each dataset ([`Transient`]).

use std::result::Result as StdResult;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::sleep;

use crate::error::Result;

/// Most retries [`Retry::max_retries`] may ask for.
const MAX_RETRIES: u32 = 32;

/// How often and after how long to try an item again after a transient failure.
///
/// What is transient depends on the dataset:
/// - `HttpDataset`: a connection error, a timeout, a body cut short, or status 408, 429, 500,
///   502, 503 or 504; other statuses (404, 403, …) fail at once. A `Retry-After` in seconds
///   is honoured, up to `max_backoff`.
/// - [`FileDataset`](super::FileDataset): an I/O error of kind `TimedOut`, `Interrupted` or
///   `WouldBlock`, or an attempt over the dataset's `timeout`. Default: no retry.
/// - `ObjectStoreDataset`: the store retries, with these settings (see its docs).
///
/// The wait before retry `n` (from 0) is `initial_backoff × 2ⁿ`, at most `max_backoff`, of
/// which the second half is random (jitter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retry {
    /// Retries after the first attempt, at most 32; `0` disables retrying.
    pub max_retries: u32,
    /// The wait before the first retry.
    pub initial_backoff: Duration,
    /// The longest wait between two attempts.
    pub max_backoff: Duration,
}

impl Retry {
    /// No retry: one attempt per item.
    pub const NONE: Self = Self {
        max_retries: 0,
        initial_backoff: Duration::ZERO,
        max_backoff: Duration::ZERO,
    };

    /// Refuses more than 32 retries.
    pub(crate) fn check(&self) -> Result<()> {
        if self.max_retries > MAX_RETRIES {
            return Err(format!("at most {MAX_RETRIES} retries, not {}", self.max_retries).into());
        }
        Ok(())
    }

    /// The wait before retry `retry` (from 0); `server` is a delay the server asked for.
    pub(crate) fn backoff(&self, retry: u32, server: Option<Duration>) -> Duration {
        let exponential = self
            .initial_backoff
            .saturating_mul(2u32.saturating_pow(retry))
            .min(self.max_backoff);
        let half = exponential / 2;
        let nanos = u64::try_from(half.as_nanos()).unwrap_or(u64::MAX);
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| u64::from(elapsed.subsec_nanos()));
        let jitter = Duration::from_nanos(fastrand::Rng::with_seed(seed).u64(0..=nanos));
        let wait = half.saturating_add(jitter);
        match server {
            Some(server) => wait.max(server.min(self.max_backoff)),
            None => wait,
        }
    }

    /// Runs `attempt` until it succeeds, fails for good, or the retries are spent; on failure,
    /// the last one and the number of attempts.
    pub(crate) async fn run<T, F, A>(
        &self,
        mut attempt: impl FnMut() -> A,
    ) -> StdResult<T, (F, u32)>
    where
        F: Transient,
        A: Future<Output = StdResult<T, F>>,
    {
        let mut attempts = 0;
        loop {
            attempts += 1;
            let failure = match attempt().await {
                Ok(value) => return Ok(value),
                Err(failure) => failure,
            };
            let retry = attempts - 1;
            if !failure.is_transient() || retry >= self.max_retries {
                return Err((failure, attempts));
            }
            sleep(self.backoff(retry, failure.retry_after())).await;
        }
    }
}

impl Default for Retry {
    /// 3 retries, waiting about 100 ms, 200 ms then 400 ms; at most 5 s.
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
        }
    }
}

/// A failed attempt, as its dataset classifies it.
pub(crate) trait Transient {
    /// Whether another attempt may succeed.
    fn is_transient(&self) -> bool;

    /// The wait the server asked for, if any.
    fn retry_after(&self) -> Option<Duration> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Retry, Transient};

    /// A failure that is transient or not.
    struct Failure(bool);

    impl Transient for Failure {
        fn is_transient(&self) -> bool {
            self.0
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(future)
    }

    const FAST: Retry = Retry {
        max_retries: 3,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(2),
    };

    #[test]
    fn backoff_grows_and_is_capped() {
        let retry = Retry {
            max_retries: 10,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(300),
        };
        for _ in 0..20 {
            let first = retry.backoff(0, None);
            assert!((Duration::from_millis(50)..=Duration::from_millis(100)).contains(&first));
            assert!(retry.backoff(9, None) <= Duration::from_millis(300));
            assert!(retry.backoff(9, None) >= Duration::from_millis(150));
            // A server's Retry-After is honoured, up to the cap.
            assert_eq!(
                retry.backoff(0, Some(Duration::from_secs(60))),
                Duration::from_millis(300)
            );
        }
        // No overflow at the largest exponent and durations.
        let huge = Retry {
            max_retries: 32,
            initial_backoff: Duration::MAX,
            max_backoff: Duration::MAX,
        };
        assert!(huge.backoff(u32::MAX, Some(Duration::MAX)) >= Duration::MAX / 2);
    }

    #[test]
    fn retries_transient_failures_until_success() {
        let mut calls = 0;
        let result = block_on(FAST.run(|| {
            calls += 1;
            let result = if calls < 3 {
                Err(Failure(true))
            } else {
                Ok(calls)
            };
            async move { result }
        }));
        assert!(matches!(result, Ok(3)));
    }

    #[test]
    fn gives_up_on_a_lasting_failure_or_a_permanent_one() {
        let mut calls = 0;
        let result = block_on(FAST.run(|| {
            calls += 1;
            async { Err::<(), _>(Failure(true)) }
        }));
        assert!(matches!(result, Err((Failure(true), 4))));
        assert_eq!(calls, 4);

        let mut calls = 0;
        let result = block_on(FAST.run(|| {
            calls += 1;
            async { Err::<(), _>(Failure(false)) }
        }));
        assert!(matches!(result, Err((Failure(false), 1))));
        assert_eq!(calls, 1);

        let result = block_on(Retry::NONE.run(|| async { Err::<(), _>(Failure(true)) }));
        assert!(matches!(result, Err((_, 1))));
    }

    #[test]
    fn refuses_more_than_32_retries() {
        let retry = |max_retries| Retry {
            max_retries,
            ..Retry::default()
        };
        assert!(retry(32).check().is_ok());
        assert!(retry(33).check().is_err());
    }
}
