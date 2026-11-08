//! [`HttpDataset`]: one HTTP GET per item, on a shared async client.

use std::fmt::{self, Debug};
use std::result::Result as StdResult;
use std::sync::{Arc, Once};
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, RETRY_AFTER, USER_AGENT};
use reqwest::redirect::Policy;
use reqwest::tls::Certificate;
use reqwest::{Client, Error as ReqwestError, Response, StatusCode, Url};
use rustls::crypto::ring;

use crate::{
    dataset::{
        AsyncDataset, Column, DEFAULT_MAX_BYTES, FetchError, Retry,
        location::Locations,
        retry::Transient,
        template::{Escape, Template},
    },
    error::Result,
};

/// Statuses worth another attempt: timeout, rate limit and the transient server errors.
const TRANSIENT_STATUS: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// Capacity reserved for a body before its bytes arrive: an announced length is trusted only
/// up to this, so a lying `Content-Length` cannot make the dataset allocate.
const INITIAL_BODY_CAPACITY: u64 = 1 << 20;

/// One HTTP GET per item: item `i` is the body of the response for URL `i`.
///
/// The URLs are a list ([`urls`](HttpDatasetBuilder::urls)), a function of the index
/// ([`url_fn`](HttpDatasetBuilder::url_fn)) or a template with per-item values
/// ([`url_template`](HttpDatasetBuilder::url_template)), as for the other built-in datasets.
/// All items share one [`reqwest`] client: connections are kept alive and reused across items
/// and batches. Each request sends the configured headers, must complete (body included)
/// within `timeout`, and is retried on transient failures ([`Retry`]). A body larger than
/// `max_bytes` fails the item, checked against `Content-Length` before reading and again while
/// reading. Responses are not decompressed.
///
/// TLS is rustls with the *ring* provider, checking certificates against the bundled Mozilla
/// roots (`webpki-root-certs`), not the system store. If no rustls provider is installed for
/// the process yet, building the first client installs *ring* as the default.
///
/// ```no_run
/// use std::time::Duration;
/// use dataloader_rs::{DataLoader, dataset::{HttpDataset, Retry}};
///
/// let dataset = HttpDataset::builder()
///     .header("Authorization", "Bearer <token>")
///     .timeout(Duration::from_secs(10))
///     .retry(Retry { max_retries: 5, ..Retry::default() })
///     .url_fn(1000, |i| format!("http://tiles.internal/slide/{i}.png"))?;
/// let mut loader = DataLoader::builder_async(dataset)
///     .batch_size(64)
///     .num_workers(2)
///     .item_concurrency(32)
///     .build();
/// # Ok::<(), dataloader_rs::Error>(())
/// ```
#[derive(Clone)]
pub struct HttpDataset {
    client: Client,
    urls: Locations<Url>,
    retry: Retry,
    max_bytes: u64,
}

/// Settings of an [`HttpDataset`]; [`urls`](Self::urls), [`url_fn`](Self::url_fn) or
/// [`url_template`](Self::url_template) builds it. Its `Debug` shows header names, not their
/// values.
#[derive(Clone)]
pub struct HttpDatasetBuilder {
    headers: Vec<(String, String)>,
    timeout: Duration,
    connect_timeout: Duration,
    retry: Retry,
    max_bytes: u64,
    max_idle_per_host: usize,
    /// Tests only: trust this PEM certificate instead of the bundled roots.
    #[cfg(test)]
    test_root: Option<&'static [u8]>,
}

impl Default for HttpDatasetBuilder {
    fn default() -> Self {
        Self {
            headers: Vec::new(),
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            retry: Retry::default(),
            max_bytes: DEFAULT_MAX_BYTES,
            max_idle_per_host: 64,
            #[cfg(test)]
            test_root: None,
        }
    }
}

impl Debug for HttpDatasetBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header values may be secrets (`Authorization`): names only.
        let headers: Vec<(&str, &str)> = self
            .headers
            .iter()
            .map(|(name, _)| (name.as_str(), "***"))
            .collect();
        f.debug_struct("HttpDatasetBuilder")
            .field("headers", &headers)
            .field("timeout", &self.timeout)
            .field("connect_timeout", &self.connect_timeout)
            .field("retry", &self.retry)
            .field("max_bytes", &self.max_bytes)
            .field("max_idle_per_host", &self.max_idle_per_host)
            .finish_non_exhaustive()
    }
}

impl HttpDatasetBuilder {
    /// Add a header to every request, e.g. `Authorization`. Its value never appears in errors
    /// or in `Debug` output.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Time allowed for one attempt, from connecting to the last byte of the body. Default
    /// 30 s.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Time allowed to open a connection. Default 10 s.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// When to retry. Default [`Retry::default`]; [`Retry::NONE`] for one attempt.
    pub fn retry(mut self, retry: Retry) -> Self {
        self.retry = retry;
        self
    }

    /// Cap on the size of one response body, in bytes. Default [`DEFAULT_MAX_BYTES`].
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Idle connections kept open per host for reuse. Default 64; set it to at least the
    /// requests in flight (`num_workers × item_concurrency`).
    pub fn max_idle_per_host(mut self, connections: usize) -> Self {
        self.max_idle_per_host = connections;
        self
    }

    /// A dataset of `urls` (`http://` or `https://`); an invalid URL is an error here.
    pub fn urls<U: AsRef<str>>(self, urls: impl IntoIterator<Item = U>) -> Result<HttpDataset> {
        let urls = urls
            .into_iter()
            .map(|url| parse_url(url.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        self.build(Locations::List(urls))
    }

    /// A dataset of `len` items whose URLs `url(index)` makes on demand, for datasets too big
    /// to list (tiles of a slide). An invalid URL fails its item.
    pub fn url_fn(
        self,
        len: usize,
        url: impl Fn(usize) -> String + Send + Sync + 'static,
    ) -> Result<HttpDataset> {
        self.build(Locations::Fn {
            len,
            location: Arc::new(url),
        })
    }

    /// A dataset whose URLs are made on demand from `template` and one value per item in each
    /// of `columns`, without a string per item in memory. The number of items is the length
    /// of the columns, which must all have the same.
    ///
    /// The template is parsed here: `{name}` is the value of column `name` for the item,
    /// `{index}` the index of the item, `{{` and `}}` literal braces. Integers are written in
    /// decimal, strings percent-encoded as a URL component (see [`Column`]); the rest of the
    /// template is used as it is. A placeholder without a column, a column the template does
    /// not use (a typo, likely), unequal lengths, unbalanced braces, an empty template or a
    /// value `.` or `..` are errors here, as is an invalid URL for the first item; an invalid
    /// URL for another item fails that item.
    ///
    /// ```no_run
    /// use dataloader_rs::dataset::{Column, HttpDataset};
    ///
    /// // 34 000 tiles of 224 px, 200 per row.
    /// let xs: Vec<u64> = (0..34_000).map(|i| i % 200 * 224).collect();
    /// let ys: Vec<u64> = (0..34_000).map(|i| i / 200 * 224).collect();
    /// let dataset = HttpDataset::builder().url_template(
    ///     "http://imgsrv.internal/slides/{slide}/tile?x={x}&y={y}&w=224&h=224&f=png",
    ///     [
    ///         ("slide", Column::from(vec!["s 1.svs"; xs.len()])),
    ///         ("x", Column::from(xs)),
    ///         ("y", Column::from(ys)),
    ///     ],
    /// )?;
    /// # Ok::<(), dataloader_rs::Error>(())
    /// ```
    pub fn url_template<'a>(
        self,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<HttpDataset> {
        self.templated(None, template, columns)
    }

    /// [`url_template`](Self::url_template) with `len` items: for a template using only
    /// `{index}`, with no column, or to check the length of the columns.
    pub fn url_template_len<'a>(
        self,
        len: usize,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<HttpDataset> {
        self.templated(Some(len), template, columns)
    }

    fn templated<'a>(
        self,
        len: Option<usize>,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<HttpDataset> {
        let columns = columns.into_iter().collect();
        let template = Template::new(template, columns, len, Escape::Url)?;
        self.build(Locations::template(template, parse_url)?)
    }

    fn build(self, urls: Locations<Url>) -> Result<HttpDataset> {
        self.retry.check()?;
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static(concat!("dataloader-rs/", env!("CARGO_PKG_VERSION"))),
        );
        for (name, value) in &self.headers {
            let name = HeaderName::try_from(name.as_str())?;
            let mut value = HeaderValue::try_from(value.as_str())?;
            value.set_sensitive(true);
            headers.insert(name, value);
        }
        install_ring_provider();
        let builder = Client::builder()
            .default_headers(headers)
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout)
            .pool_max_idle_per_host(self.max_idle_per_host)
            .redirect(Policy::limited(5));
        #[cfg(test)]
        let builder = match self.test_root {
            Some(pem) => builder.tls_certs_only([Certificate::from_pem(pem)?]),
            None => builder.tls_certs_only(mozilla_roots()?),
        };
        #[cfg(not(test))]
        let builder = builder.tls_certs_only(mozilla_roots()?);
        Ok(HttpDataset {
            client: builder.build()?,
            urls,
            retry: self.retry,
            max_bytes: self.max_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn test_root(mut self, pem: &'static [u8]) -> Self {
        self.test_root = Some(pem);
        self
    }
}

/// Make *ring* the process's rustls provider, unless one is installed already.
pub(crate) fn install_ring_provider() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Err: another provider was installed first, which rustls then uses.
        let _ = ring::default_provider().install_default();
    });
}

/// The bundled Mozilla root certificates.
fn mozilla_roots() -> Result<Vec<Certificate>> {
    webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .map(|der| Ok(Certificate::from_der(der)?))
        .collect()
}

fn parse_url(url: &str) -> Result<Url> {
    let parsed = Url::parse(url).map_err(|e| format!("invalid URL {:?}: {e}", redact_str(url)))?;
    match parsed.scheme() {
        "http" | "https" => Ok(parsed),
        scheme => Err(format!("unsupported URL scheme {scheme:?}, use http or https").into()),
    }
}

/// The URL for messages: scheme, host, port and path; no credentials, query or fragment.
fn redact(url: &Url) -> String {
    let mut out = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
    if let Some(port) = url.port() {
        out.push_str(&format!(":{port}"));
    }
    out.push_str(url.path());
    out
}

fn redact_str(url: &str) -> String {
    Url::parse(url).map_or_else(|_| "<unparsable>".to_owned(), |url| redact(&url))
}

/// How one attempt failed.
enum Attempt {
    Status {
        status: StatusCode,
        retry_after: Option<Duration>,
    },
    TooLarge(Option<u64>),
    Request(ReqwestError),
}

impl Attempt {
    /// A response with a status other than 2xx.
    fn status(response: &Response) -> Self {
        Self::Status {
            status: response.status(),
            retry_after: retry_after(response),
        }
    }

    /// The error of the item, at `location` (redacted), after `attempts` attempts.
    fn into_error(self, location: String, attempts: u32, limit: u64) -> FetchError {
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

impl HttpDataset {
    /// Settings with their defaults: no header, 30 s per attempt, 10 s to connect,
    /// [`Retry::default`], [`DEFAULT_MAX_BYTES`].
    pub fn builder() -> HttpDatasetBuilder {
        HttpDatasetBuilder::default()
    }

    /// A dataset of `urls` with the default settings.
    pub fn new<U: AsRef<str>>(urls: impl IntoIterator<Item = U>) -> Result<Self> {
        Self::builder().urls(urls)
    }

    async fn get_once(&self, url: &Url) -> StdResult<Bytes, Attempt> {
        let response = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(Attempt::Request)?;
        if !response.status().is_success() {
            return Err(Attempt::status(&response));
        }
        self.read_capped(response).await
    }

    /// The body, failing as soon as it is known to exceed `max_bytes`.
    async fn read_capped(&self, mut response: Response) -> StdResult<Bytes, Attempt> {
        let announced = response.content_length();
        if announced.is_some_and(|size| size > self.max_bytes) {
            return Err(Attempt::TooLarge(announced));
        }
        let capacity = announced.unwrap_or(0).min(INITIAL_BODY_CAPACITY);
        let mut body = BytesMut::with_capacity(usize::try_from(capacity).unwrap_or(0));
        while let Some(chunk) = response.chunk().await.map_err(Attempt::Request)? {
            let size = (body.len() as u64).saturating_add(chunk.len() as u64);
            if size > self.max_bytes {
                return Err(Attempt::TooLarge(None));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body.freeze())
    }
}

impl AsyncDataset for HttpDataset {
    type Item = Bytes;

    async fn get(&self, index: usize) -> Result<Bytes> {
        let url = self.urls.get(index, parse_url)?;
        let url = url.as_ref();
        self.retry
            .run(|| self.get_once(url))
            .await
            .map_err(|(failure, attempts)| {
                failure
                    .into_error(redact(url), attempts, self.max_bytes)
                    .into()
            })
    }

    fn len(&self) -> usize {
        self.urls.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{HttpDataset, Retry};
    use crate::{
        DataLoader,
        dataset::{
            AsyncDataset, Column, FetchError,
            test_server::{CA_PEM, TestServer},
        },
        error::{Error, Result},
    };

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    /// Short backoff so that retry tests run fast.
    fn fast_retry(max_retries: u32) -> Retry {
        Retry {
            max_retries,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
        }
    }

    fn fetch(error: &Error) -> &FetchError {
        error.downcast_ref().expect("a FetchError")
    }

    #[test]
    fn fetches_urls_through_the_loader_in_order() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .url_fn(40, move |i| server.url(&format!("/ok/{i}")))
            .unwrap();
        let mut loader = DataLoader::builder_async(ds)
            .batch_size(8)
            .num_workers(2)
            .item_concurrency(4)
            .build();
        let items: Vec<String> = loader
            .iter()
            .flat_map(|batch| batch.unwrap())
            .map(|body| String::from_utf8(body.to_vec()).unwrap())
            .collect();
        let expected: Vec<String> = (0..40).map(|i| format!("item-{i}")).collect();
        assert_eq!(items, expected);
    }

    #[test]
    fn retries_503_and_429_then_succeeds() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .retry(fast_retry(3))
            .urls([server.url("/flaky/503/2/a"), server.url("/flaky/429/3/b")])
            .unwrap();
        assert_eq!(&block_on(ds.get(0)).unwrap()[..], b"ok");
        assert_eq!(&block_on(ds.get(1)).unwrap()[..], b"ok");
        assert_eq!(server.hits("/flaky/503/2/a"), 3);
        assert_eq!(server.hits("/flaky/429/3/b"), 4);
    }

    #[test]
    fn gives_up_after_the_last_retry() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .retry(fast_retry(2))
            .urls([server.url("/status/503")])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::Status {
                status: 503,
                attempts: 3,
                ..
            }
        ));
        assert_eq!(server.hits("/status/503"), 3);
    }

    #[test]
    fn not_found_fails_at_once() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .retry(fast_retry(5))
            .urls([server.url("/status/404")])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::Status {
                status: 404,
                attempts: 1,
                ..
            }
        ));
        assert_eq!(server.hits("/status/404"), 1);
    }

    #[test]
    fn a_huge_content_length_fails_before_reading() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .max_bytes(1 << 20)
            .urls([server.url("/huge")])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::TooLarge {
                size: Some(1_000_000_000_000),
                ..
            }
        ));
        assert_eq!(server.hits("/huge"), 1, "too large is not retried");
    }

    #[test]
    fn a_body_over_the_cap_fails_while_reading() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .max_bytes(10_000)
            .urls([server.url("/chunked/10000"), server.url("/chunked/10001")])
            .unwrap();
        assert_eq!(block_on(ds.get(0)).unwrap().len(), 10_000);
        let err = block_on(ds.get(1)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::TooLarge { size: None, .. }
        ));
    }

    #[test]
    fn slow_loris_and_silent_servers_time_out() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .timeout(Duration::from_millis(300))
            .retry(fast_retry(1))
            .urls([server.url("/slow"), server.url("/hang")])
            .unwrap();
        for index in 0..2 {
            let start = Instant::now();
            let err = block_on(ds.get(index)).unwrap_err();
            let elapsed = start.elapsed();
            assert!(fetch(&err).is_timeout(), "{err}");
            assert!(matches!(fetch(&err), FetchError::Io { attempts: 2, .. }));
            // Two attempts of 300 ms, and not the 5 s the slow body would take.
            assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
        }
    }

    #[test]
    fn connection_refused_is_retried_then_reported() {
        let ds = HttpDataset::builder()
            .retry(fast_retry(2))
            .urls(["http://127.0.0.1:1/x"])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(
            matches!(fetch(&err), FetchError::Io { attempts: 3, .. }),
            "{err}"
        );
    }

    #[test]
    fn errors_hide_query_and_credentials() {
        let server = TestServer::start();
        let url = server
            .url("/status/404?token=secret")
            .replace("http://", "http://user:pw@");
        let ds = HttpDataset::builder()
            .header("Authorization", "Bearer secret")
            .urls([url])
            .unwrap();
        let message = block_on(ds.get(0)).unwrap_err().to_string();
        assert!(
            !message.contains("secret") && !message.contains("pw"),
            "{message}"
        );
        assert!(message.contains("/status/404"), "{message}");
    }

    #[test]
    fn debug_shows_header_names_not_values() {
        let builder = HttpDataset::builder()
            .header("Authorization", "Bearer secret")
            .header("X-Api-Key", "key-value");
        let debug = format!("{builder:?}");
        assert!(
            debug.contains("Authorization") && debug.contains("X-Api-Key"),
            "{debug}"
        );
        assert!(
            !debug.contains("secret") && !debug.contains("key-value"),
            "{debug}"
        );
    }

    #[test]
    fn rejects_bad_urls_and_out_of_range_indices() {
        assert!(HttpDataset::new(["not a url"]).is_err());
        assert!(HttpDataset::new(["file:///etc/passwd"]).is_err());
        assert!(
            HttpDataset::builder()
                .retry(Retry {
                    max_retries: 33,
                    ..Retry::default()
                })
                .urls(["http://x"])
                .is_err()
        );
        let ds = HttpDataset::builder()
            .url_fn(2, |i| {
                if i == 0 {
                    "nope".into()
                } else {
                    "http://x".into()
                }
            })
            .unwrap();
        assert!(block_on(ds.get(0)).is_err());
        let err = block_on(ds.get(2)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::OutOfRange { index: 2, len: 2 }
        ));
    }

    #[test]
    fn tls_with_a_trusted_root_and_not_without() {
        let server = TestServer::start_tls();
        let trusted = HttpDataset::builder()
            .test_root(CA_PEM)
            .urls([server.url("/ok/1")])
            .unwrap();
        assert_eq!(&block_on(trusted.get(0)).unwrap()[..], b"item-1");
        // The bundled Mozilla roots do not include the test CA.
        let untrusted = HttpDataset::builder()
            .retry(Retry::NONE)
            .urls([server.url("/ok/1")])
            .unwrap();
        let err = block_on(untrusted.get(0)).unwrap_err();
        let mut chain = err.to_string();
        let mut source = err.source();
        while let Some(cause) = source {
            chain.push_str(&format!(" / {cause}"));
            source = cause.source();
        }
        assert!(chain.contains("certificate"), "{chain}");
    }

    #[test]
    fn a_url_template_requests_its_paths_and_queries_in_order() {
        let server = TestServer::start();
        let base = server.url("");
        let xs: Vec<u64> = (0..12).map(|i| i % 4 * 224).collect();
        let ys: Vec<i64> = (0..12).map(|i| i / 4 * 224).collect();
        let names: Vec<String> = (0..12).map(|i| format!("s {i}&")).collect();
        let ds = HttpDataset::builder()
            .url_template(
                &format!("{base}/ok/{{index}}?x={{x}}&y={{y}}&s={{s}}&l=12"),
                [
                    ("x", Column::from(xs)),
                    ("y", Column::from(ys)),
                    ("s", Column::from(names)),
                ],
            )
            .unwrap();
        assert_eq!(AsyncDataset::len(&ds), 12);
        let mut loader = DataLoader::builder_async(ds)
            .batch_size(4)
            .num_workers(2)
            .item_concurrency(4)
            .build();
        let items: Vec<String> = loader
            .iter()
            .flat_map(|batch| batch.unwrap())
            .map(|body| String::from_utf8(body.to_vec()).unwrap())
            .collect();
        let expected: Vec<String> = (0..12)
            .map(|i| {
                format!(
                    "item-{i}?x={}&y={}&s=s%20{i}%26&l=12",
                    i % 4 * 224,
                    i / 4 * 224
                )
            })
            .collect();
        assert_eq!(items, expected);
        assert_eq!(server.hits("/ok/5?x=224&y=224&s=s%205%26&l=12"), 1);
    }

    #[test]
    fn a_url_template_checks_its_first_url_and_length() {
        let ds = HttpDataset::builder()
            .url_template_len(3, "http://h/{index}.png", [])
            .unwrap();
        assert_eq!(AsyncDataset::len(&ds), 3);
        let err = block_on(ds.get(3)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::OutOfRange { index: 3, len: 3 }
        ));
        let error = HttpDataset::builder()
            .url_template("{s}/x", [("s", Column::from(vec!["http"]))])
            .err()
            .unwrap();
        assert!(error.to_string().contains("invalid URL"), "{error}");
        assert!(
            HttpDataset::builder()
                .url_template("ftp://h/{x}", [("x", Column::from(vec![1i64]))])
                .is_err()
        );
        // No item, no URL to check.
        let empty = HttpDataset::builder()
            .url_template("{s}", [("s", Column::from(Vec::<String>::new()))])
            .unwrap();
        assert_eq!(AsyncDataset::len(&empty), 0);
    }

    #[test]
    fn a_failed_item_fails_its_batch_in_the_loader() {
        let server = TestServer::start();
        let ds = HttpDataset::builder()
            .url_fn(8, move |i| {
                if i == 5 {
                    server.url("/status/404")
                } else {
                    server.url(&format!("/ok/{i}"))
                }
            })
            .unwrap();
        let mut loader = DataLoader::builder_async(ds)
            .batch_size(4)
            .num_workers(2)
            .item_concurrency(4)
            .build();
        let results: Vec<Result<Vec<_>>> = loader.iter().collect();
        assert!(results[0].is_ok());
        assert!(results[1].as_ref().unwrap_err().to_string().contains("404"));
    }
}
