//! [`HttpSource`]: one HTTP GET per item, with a blocking, pooled client.

use std::error::Error as StdError;
use std::fmt::{self, Display, Write as _};
use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ureq::Agent;
use ureq::http::{HeaderName, HeaderValue, Uri};

use super::{DEFAULT_MAX_BYTES, ReadError, Source, SourceError, read_capped};

/// Most retries [`Retry::max_retries`] accepts.
const MAX_RETRIES: u32 = 32;

/// Statuses worth another attempt: timeout, rate limit and the transient server errors.
const TRANSIENT_STATUS: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// When to send a request again after a transient failure: a connection error, a timeout, a
/// truncated body, or status 408, 429, 500, 502, 503 or 504. Other statuses (404, 403, …),
/// malformed responses and TLS failures (such as an untrusted certificate) fail at once.
///
/// The wait before retry `n` (from 0) is `initial_backoff × 2ⁿ`, at most `max_backoff`, of which
/// the second half is random (jitter). A `Retry-After` of a few seconds on 429 or 503 is
/// honoured, also up to `max_backoff`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retry {
    /// Retries after the first attempt (at most 32); `0` disables retrying.
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

    /// The wait before retry `retry` (from 0), `server` being a `Retry-After` delay.
    fn backoff(&self, retry: u32, server: Option<Duration>, seed: u64) -> Duration {
        let exponential = self
            .initial_backoff
            .saturating_mul(2u32.saturating_pow(retry))
            .min(self.max_backoff);
        let half = exponential / 2;
        let nanos = u64::try_from(half.as_nanos()).unwrap_or(u64::MAX);
        let jitter = Duration::from_nanos(fastrand::Rng::with_seed(seed).u64(0..=nanos));
        let wait = half.saturating_add(jitter);
        match server {
            Some(server) => wait.max(server.min(self.max_backoff)),
            None => wait,
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

/// One HTTP GET per item, from a list of URLs or a URL template with per-item parameters.
///
/// The client is [`ureq`]: blocking, one connection per request in flight, connections kept
/// alive and reused across items and batches. Each request sends the configured headers, must
/// complete within `timeout`, and is retried on transient failures ([`Retry`]). A body larger
/// than `max_bytes` fails the item, checked against `Content-Length` before reading, and again
/// while reading. Responses are not decompressed.
///
/// Plain `http://` only, unless the crate is built with the `sources-tls` feature, which adds
/// `https://` through rustls. Certificates are checked against the bundled Mozilla roots
/// (`webpki-roots`), not the system store; a TLS failure is not retried.
///
/// ```no_run
/// use std::time::Duration;
/// use dataloader_rs::sources::{HttpSource, Retry, SourceDataset};
///
/// let source = HttpSource::builder()
///     .header("Authorization", "Bearer <token>")
///     .timeout(Duration::from_secs(10))
///     .retry(Retry::default())
///     .template(
///         "http://images.internal/{split}/{id}.jpeg",
///         (0..1000).map(|id| [("split", "train".to_string()), ("id", id.to_string())]),
///     )?;
/// let dataset = SourceDataset::new(source);
/// # Ok::<(), dataloader_rs::sources::SourceError>(())
/// ```
#[derive(Clone)]
pub struct HttpSource {
    agent: Agent,
    urls: Vec<Uri>,
    headers: Vec<(HeaderName, HeaderValue)>,
    retry: Retry,
    max_bytes: u64,
}

impl fmt::Debug for HttpSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header values may be credentials: names only.
        f.debug_struct("HttpSource")
            .field("urls", &self.urls.len())
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("retry", &self.retry)
            .field("max_bytes", &self.max_bytes)
            .finish_non_exhaustive()
    }
}

/// Settings of an [`HttpSource`]; [`urls`](Self::urls) or [`template`](Self::template) builds
/// it.
#[derive(Clone, Debug)]
pub struct HttpSourceBuilder {
    headers: Vec<(String, String)>,
    timeout: Duration,
    connect_timeout: Duration,
    retry: Retry,
    max_bytes: u64,
    max_redirects: u32,
    max_idle_connections: usize,
    /// Tests only: trust this PEM certificate instead of the bundled roots.
    #[cfg(all(test, feature = "sources-tls"))]
    test_root: Option<&'static [u8]>,
}

impl Default for HttpSourceBuilder {
    fn default() -> Self {
        Self {
            headers: Vec::new(),
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            retry: Retry::default(),
            max_bytes: DEFAULT_MAX_BYTES,
            max_redirects: 5,
            max_idle_connections: 64,
            #[cfg(all(test, feature = "sources-tls"))]
            test_root: None,
        }
    }
}

impl HttpSource {
    /// Settings with their defaults: no header, 30 s per attempt, 10 s to connect,
    /// [`Retry::default`], [`DEFAULT_MAX_BYTES`], 5 redirects, 64 idle connections kept.
    pub fn builder() -> HttpSourceBuilder {
        HttpSourceBuilder::default()
    }

    /// A source of `urls` with the default settings.
    pub fn urls<U: AsRef<str>>(urls: impl IntoIterator<Item = U>) -> Result<Self, SourceError> {
        Self::builder().urls(urls)
    }

    /// The URL of item `index`, as requested (with its query).
    pub fn url(&self, index: usize) -> Option<&Uri> {
        self.urls.get(index)
    }

    fn get_once(&self, uri: &Uri) -> Result<Vec<u8>, Attempt> {
        let mut request = self.agent.get(uri);
        for (name, value) in &self.headers {
            request = request.header(name, value);
        }
        let response = request.call().map_err(Attempt::from_ureq)?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse::<u64>().ok())
                .map(Duration::from_secs);
            return Err(Attempt::Status {
                status,
                retry_after,
            });
        }
        let body = response.into_body();
        let size = body.content_length();
        read_capped(body.into_reader(), size, self.max_bytes).map_err(|error| match error {
            ReadError::TooLarge(size) => Attempt::TooLarge(size),
            ReadError::Io(error) => Attempt::from_ureq(ureq::Error::from(error)),
        })
    }
}

impl Source for HttpSource {
    type Bytes = Vec<u8>;

    fn fetch(&self, index: usize) -> Result<Vec<u8>, SourceError> {
        let uri = self.urls.get(index).ok_or(SourceError::OutOfRange {
            index,
            len: self.urls.len(),
        })?;
        let mut attempts = 0;
        loop {
            attempts += 1;
            let failure = match self.get_once(uri) {
                Ok(bytes) => return Ok(bytes),
                Err(failure) => failure,
            };
            let retry = attempts - 1;
            if !failure.is_transient() || retry >= self.retry.max_retries {
                return Err(failure.into_error(index, redact(uri), attempts, self.max_bytes));
            }
            let seed = seed(index, attempts);
            std::thread::sleep(self.retry.backoff(retry, failure.retry_after(), seed));
        }
    }

    fn len(&self) -> usize {
        self.urls.len()
    }
}

/// How one attempt failed.
enum Attempt {
    Status {
        status: u16,
        retry_after: Option<Duration>,
    },
    TooLarge(Option<u64>),
    Request {
        transient: bool,
        timeout: bool,
        error: ureq::Error,
    },
}

impl Attempt {
    fn from_ureq(error: ureq::Error) -> Self {
        let timeout = matches!(error, ureq::Error::Timeout(_));
        // rustls reports a failed handshake (untrusted or expired certificate, wrong host name,
        // no common protocol) as an I/O error of kind `InvalidData`: another attempt would fail
        // the same way.
        let transient = timeout
            || matches!(&error, ureq::Error::Io(error) if error.kind() != io::ErrorKind::InvalidData)
            || matches!(error, ureq::Error::ConnectionFailed);
        Self::Request {
            transient,
            timeout,
            error,
        }
    }

    fn is_transient(&self) -> bool {
        match self {
            Self::Status { status, .. } => TRANSIENT_STATUS.contains(status),
            Self::TooLarge(_) => false,
            Self::Request { transient, .. } => *transient,
        }
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    fn into_error(self, index: usize, url: String, attempts: u32, limit: u64) -> SourceError {
        match self {
            Self::Status { status, .. } => SourceError::Status {
                index,
                url,
                status,
                attempts,
            },
            Self::TooLarge(size) => SourceError::TooLarge {
                index,
                location: url,
                limit,
                size,
            },
            Self::Request { timeout, error, .. } => SourceError::Request {
                index,
                url,
                attempts,
                timeout,
                source: Box::new(error) as Box<dyn StdError + Send + Sync>,
            },
        }
    }
}

/// A seed for the jitter of one retry: differs per item, attempt and run.
fn seed(index: usize, attempts: u32) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.subsec_nanos());
    (index as u64).rotate_left(32) ^ u64::from(attempts) ^ u64::from(now)
}

/// The URL for error messages: no user info, query or fragment, which may hold credentials.
fn redact(uri: &Uri) -> String {
    let mut out = String::new();
    if let Some(scheme) = uri.scheme_str() {
        out.push_str(scheme);
        out.push_str("://");
    }
    if let Some(host) = uri.host() {
        out.push_str(host);
    }
    if let Some(port) = uri.port_u16() {
        let _ = write!(out, ":{port}");
    }
    out.push_str(uri.path());
    out
}

impl HttpSourceBuilder {
    /// Add a header to every request, e.g. `Authorization`. Its value is never shown in errors
    /// or `Debug`.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Time allowed for one attempt, from connecting to the last byte of the body. Default 30 s.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Time allowed to open a connection, within [`timeout`](Self::timeout). Default 10 s.
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

    /// Redirects followed per request; more is an error. Default 5; `0` follows none.
    pub fn max_redirects(mut self, max_redirects: u32) -> Self {
        self.max_redirects = max_redirects;
        self
    }

    /// Idle connections kept open for reuse, per host and in total. Default 64; set it to at
    /// least the requests in flight (`num_workers × items_in_parallel`).
    pub fn max_idle_connections(mut self, connections: usize) -> Self {
        self.max_idle_connections = connections;
        self
    }

    /// A source of `urls`, in order. Each must be an absolute `http://` URL (or `https://` with
    /// the `sources-tls` feature).
    pub fn urls<U: AsRef<str>>(
        self,
        urls: impl IntoIterator<Item = U>,
    ) -> Result<HttpSource, SourceError> {
        let urls = urls
            .into_iter()
            .enumerate()
            .map(|(index, url)| parse_url(index, url.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.build(urls)
    }

    /// A source with one URL per entry of `params`, made from `template` by replacing each
    /// `{name}` with the value of `name` in the entry (`{{` and `}}` are literal braces).
    ///
    /// Values are percent-encoded, except the unreserved characters `A-Z a-z 0-9 - . _ ~`, so a
    /// value cannot add path segments or a query. Every entry must give each name of the
    /// template exactly once, and no other name.
    pub fn template<P, K, V>(
        self,
        template: &str,
        params: impl IntoIterator<Item = P>,
    ) -> Result<HttpSource, SourceError>
    where
        P: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Display,
    {
        let template = Template::parse(template)?;
        let urls = params
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                let url = template.render(index, entry)?;
                parse_url(index, &url)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.build(urls)
    }

    fn build(self, urls: Vec<Uri>) -> Result<HttpSource, SourceError> {
        let invalid = |message: String| Err(SourceError::Invalid(message));
        if self.timeout.is_zero() || self.connect_timeout.is_zero() {
            return invalid("timeouts must be above zero".into());
        }
        if self.max_bytes == 0 {
            return invalid("max_bytes must be above zero".into());
        }
        if self.retry.max_retries > MAX_RETRIES {
            return invalid(format!(
                "max_retries is {}, above {MAX_RETRIES}",
                self.retry.max_retries
            ));
        }
        let headers = self
            .headers
            .iter()
            .map(|(name, value)| {
                let parsed_name = HeaderName::try_from(name.as_str());
                let parsed_value = HeaderValue::try_from(value.as_str());
                match (parsed_name, parsed_value) {
                    (Ok(name), Ok(mut value)) => {
                        value.set_sensitive(true);
                        Ok((name, value))
                    }
                    (Err(_), _) => Err(SourceError::Invalid(format!(
                        "invalid header name {name:?}"
                    ))),
                    // The value may be a credential: not shown.
                    (_, Err(_)) => Err(SourceError::Invalid(format!(
                        "invalid value for header {name:?}"
                    ))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let config = Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(self.timeout))
            .timeout_connect(Some(self.connect_timeout))
            .max_redirects(self.max_redirects)
            .max_idle_connections(self.max_idle_connections)
            .max_idle_connections_per_host(self.max_idle_connections)
            .user_agent(concat!("dataloader-rs/", env!("CARGO_PKG_VERSION")));
        #[cfg(all(test, feature = "sources-tls"))]
        let config = match self.test_root {
            Some(pem) => {
                use ureq::tls::{Certificate, RootCerts, TlsConfig};
                let root = Certificate::from_pem(pem).expect("test root certificate");
                config.tls_config(
                    TlsConfig::builder()
                        .root_certs(RootCerts::new_with_certs(&[root]))
                        .build(),
                )
            }
            None => config,
        };
        let config = config.build();
        Ok(HttpSource {
            agent: Agent::new_with_config(config),
            urls,
            headers,
            retry: self.retry,
            max_bytes: self.max_bytes,
        })
    }
}

/// The schemes this build can fetch.
#[cfg(feature = "sources-tls")]
const SCHEMES: &[&str] = &["http", "https"];
#[cfg(not(feature = "sources-tls"))]
const SCHEMES: &[&str] = &["http"];

fn parse_url(index: usize, url: &str) -> Result<Uri, SourceError> {
    let invalid = |reason: &str| {
        // Shown up to the query, which may hold a credential.
        let shown = url.split(['?', '#']).next().unwrap_or_default();
        SourceError::Invalid(format!("URL {index} ({shown}): {reason}"))
    };
    let uri = Uri::try_from(url).map_err(|_| invalid("not a valid URL"))?;
    match uri.scheme_str() {
        Some(scheme) if SCHEMES.contains(&scheme) => {}
        Some("https") => {
            return Err(invalid(
                "https needs the `sources-tls` feature of dataloader-rs",
            ));
        }
        Some(_) => return Err(invalid("only http and https URLs are supported")),
        None => return Err(invalid("not an absolute URL")),
    }
    if uri.host().is_none_or(str::is_empty) {
        return Err(invalid("no host"));
    }
    Ok(uri)
}

/// A URL template: literal text and `{name}` placeholders.
#[derive(Debug)]
struct Template {
    pieces: Vec<Piece>,
    names: Vec<String>,
}

#[derive(Debug)]
enum Piece {
    Text(String),
    Name(usize),
}

impl Template {
    fn parse(template: &str) -> Result<Self, SourceError> {
        let invalid = |reason: String| SourceError::Invalid(format!("URL template: {reason}"));
        let mut pieces = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut text = String::new();
        let mut chars = template.chars();
        while let Some(c) = chars.next() {
            match c {
                '{' if chars.as_str().starts_with('{') => {
                    chars.next();
                    text.push('{');
                }
                '}' if chars.as_str().starts_with('}') => {
                    chars.next();
                    text.push('}');
                }
                '{' => {
                    let rest = chars.as_str();
                    let end = rest
                        .find('}')
                        .ok_or_else(|| invalid("unclosed `{`".into()))?;
                    let name = &rest[..end];
                    if name.is_empty()
                        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                    {
                        return Err(invalid(format!(
                            "invalid name {name:?}: use letters, digits and `_`"
                        )));
                    }
                    if !text.is_empty() {
                        pieces.push(Piece::Text(std::mem::take(&mut text)));
                    }
                    let slot = match names.iter().position(|known| known == name) {
                        Some(slot) => slot,
                        None => {
                            names.push(name.to_string());
                            names.len() - 1
                        }
                    };
                    pieces.push(Piece::Name(slot));
                    chars = rest[end + 1..].chars();
                }
                '}' => return Err(invalid("unmatched `}`".into())),
                c => text.push(c),
            }
        }
        if !text.is_empty() {
            pieces.push(Piece::Text(text));
        }
        Ok(Self { pieces, names })
    }

    fn render<K: AsRef<str>, V: Display>(
        &self,
        index: usize,
        entry: impl IntoIterator<Item = (K, V)>,
    ) -> Result<String, SourceError> {
        let invalid = |reason: String| SourceError::Invalid(format!("URL {index}: {reason}"));
        let mut values: Vec<Option<String>> = vec![None; self.names.len()];
        for (name, value) in entry {
            let name = name.as_ref();
            let slot = self
                .names
                .iter()
                .position(|known| known == name)
                .ok_or_else(|| invalid(format!("{name:?} is not in the template")))?;
            if values[slot].is_some() {
                return Err(invalid(format!("{name:?} is given twice")));
            }
            values[slot] = Some(percent_encode(&value.to_string()));
        }
        let mut url = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Text(text) => url.push_str(text),
                Piece::Name(slot) => match &values[*slot] {
                    Some(value) => url.push_str(value),
                    None => {
                        return Err(invalid(format!("no value for {:?}", self.names[*slot])));
                    }
                },
            }
        }
        Ok(url)
    }
}

/// Percent-encode all but the unreserved characters of RFC 3986.
fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(template: &str, entry: &[(&str, &str)]) -> Result<String, SourceError> {
        Template::parse(template)?.render(0, entry.iter().copied())
    }

    #[test]
    fn template_renders_and_encodes() {
        assert_eq!(
            render("http://h/{a}/{b}_{a}.png", &[("a", "x"), ("b", "1")]).unwrap(),
            "http://h/x/1_x.png"
        );
        assert_eq!(
            render("http://h/{a}?q={b}", &[("a", "../etc"), ("b", "1&c=2 /")]).unwrap(),
            "http://h/..%2Fetc?q=1%26c%3D2%20%2F"
        );
        assert_eq!(
            render("http://h/{{lit}}/{a}", &[("a", "é")]).unwrap(),
            "http://h/{lit}/%C3%A9"
        );
    }

    #[test]
    fn template_errors() {
        for bad in [
            "http://h/{a",
            "http://h/a}",
            "http://h/{}",
            "http://h/{a-b}",
        ] {
            assert!(Template::parse(bad).is_err(), "{bad}");
        }
        let error = |entry: &[(&str, &str)]| render("http://h/{a}", entry).unwrap_err().to_string();
        assert!(error(&[]).contains("no value for \"a\""));
        assert!(error(&[("a", "1"), ("b", "2")]).contains("\"b\" is not in the template"));
        assert!(error(&[("a", "1"), ("a", "2")]).contains("given twice"));
    }

    #[test]
    fn urls_are_checked() {
        assert!(parse_url(0, "http://localhost:8080/a?b=c").is_ok());
        for bad in ["localhost/a", "ftp://h/a", "/a", "http:///a", "http://h a/"] {
            assert!(parse_url(0, bad).is_err(), "{bad}");
        }
        #[cfg(not(feature = "sources-tls"))]
        assert!(
            parse_url(0, "https://h/a")
                .unwrap_err()
                .to_string()
                .contains("sources-tls")
        );
    }

    #[test]
    fn redact_drops_credentials() {
        let uri = Uri::try_from("http://user:pw@host:81/p/q.png?token=secret#f").unwrap();
        assert_eq!(redact(&uri), "http://host:81/p/q.png");
    }

    #[test]
    fn backoff_grows_and_is_capped() {
        let retry = Retry {
            max_retries: 10,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        };
        for seed in 0..50 {
            let first = retry.backoff(0, None, seed);
            assert!(first >= Duration::from_millis(50) && first <= Duration::from_millis(100));
            let late = retry.backoff(9, None, seed);
            assert!(late >= Duration::from_millis(500) && late <= Duration::from_secs(1));
            // Huge exponents saturate instead of overflowing.
            assert!(retry.backoff(u32::MAX, None, seed) <= Duration::from_secs(1));
            // Retry-After is honoured up to max_backoff.
            assert!(
                retry.backoff(0, Some(Duration::from_secs(60)), seed) == Duration::from_secs(1)
            );
        }
    }

    /// HTTPS against a local rustls server with the certificate of `tests/fixtures/tls`
    /// (localhost and 127.0.0.1, signed by a test CA that no store trusts).
    #[cfg(feature = "sources-tls")]
    mod tls {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use rustls::pki_types::pem::PemObject;
        use rustls::pki_types::{CertificateDer, PrivateKeyDer};
        use rustls::{ServerConfig, ServerConnection, StreamOwned};

        use super::*;

        const CA: &[u8] = include_bytes!("../../tests/fixtures/tls/ca.pem");
        const CERT: &[u8] = include_bytes!("../../tests/fixtures/tls/localhost.pem");
        const KEY: &[u8] = include_bytes!("../../tests/fixtures/tls/localhost.key");

        /// Serves `body` to every request; returns its address and the connection count.
        fn serve(body: &'static [u8]) -> (String, Arc<AtomicUsize>) {
            let certs = CertificateDer::pem_slice_iter(CERT)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let key = PrivateKeyDer::from_pem_slice(KEY).unwrap();
            let config = Arc::new(
                ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(certs, key)
                    .unwrap(),
            );
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap().to_string();
            let connections = Arc::new(AtomicUsize::new(0));
            let counter = Arc::clone(&connections);
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { return };
                    counter.fetch_add(1, Ordering::SeqCst);
                    let connection = ServerConnection::new(Arc::clone(&config)).unwrap();
                    let mut tls = StreamOwned::new(connection, stream);
                    let mut request = Vec::new();
                    let mut buf = [0; 1024];
                    // A client that rejects the certificate aborts the handshake here.
                    while !request.ends_with(b"\r\n\r\n") {
                        match tls.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => request.extend_from_slice(&buf[..n]),
                        }
                    }
                    if request.ends_with(b"\r\n\r\n") {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = tls.write_all(head.as_bytes());
                        let _ = tls.write_all(body);
                        tls.conn.send_close_notify();
                        let _ = tls.flush();
                    }
                }
            });
            (address, connections)
        }

        #[test]
        fn https_fetches_with_a_trusted_root() {
            let (address, _) = serve(b"over tls");
            let builder = HttpSourceBuilder {
                test_root: Some(CA),
                ..HttpSource::builder()
            };
            let source = builder.urls([format!("https://{address}/a")]).unwrap();
            assert_eq!(source.fetch(0).unwrap(), b"over tls");
        }

        #[test]
        fn untrusted_certificate_fails_without_retry() {
            let (address, connections) = serve(b"never sent");
            let source = HttpSource::builder()
                .retry(Retry {
                    max_retries: 3,
                    initial_backoff: Duration::from_millis(1),
                    max_backoff: Duration::from_millis(1),
                })
                .urls([format!("https://{address}/a?token=secret")])
                .unwrap();
            let error = source.fetch(0).unwrap_err();
            let message = error.to_string();
            assert!(
                message.starts_with(&format!(
                    "item 0 (https://{address}/a): request failed after 1 attempt(s)"
                )),
                "{message}"
            );
            assert!(message.contains("certificate"), "{message}");
            assert!(!message.contains("secret"), "{message}");
            assert_eq!(connections.load(Ordering::SeqCst), 1);
        }
    }
}
