//! [`ObjectStoreDataset`] and its builder: one object (or byte range of one) per item, from S3,
//! GCS, Azure or an HTTP store, through the `object_store` crate.

use std::fmt::{self, Debug};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::http::HttpBuilder;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    BackoffConfig, Certificate, ClientOptions, Error as StoreError, GetOptions, ObjectStore,
    ObjectStoreScheme, RetryConfig,
};
use reqwest::Url;

use super::{
    location::Locations,
    template::{Escape, Template},
    tls::{install_ring_provider, mozilla_roots},
};
use crate::{
    dataset::{AsyncDataset, Column, DEFAULT_MAX_BYTES, FetchError, Retry},
    error::Result,
};

/// Capacity reserved for an object before its bytes arrive, whatever size the store
/// announces.
const INITIAL_CAPACITY: u64 = 1 << 20;

/// The key of an item, and its byte range if it is not the whole object.
type Key = (Path, Option<Range<u64>>);

/// One object per item, or one byte range of an object: item `i` is the content of key `i`.
///
/// Built on the [`object_store`] crate: [`from_url`](Self::from_url) opens an S3 (`s3://`),
/// GCS (`gs://`), Azure (`az://`, `abfs://`) or HTTP (`https://`) store, with credentials and
/// settings from the environment (`AWS_*`, `GOOGLE_*`, `AZURE_*`) and from `options`, and TLS
/// through rustls with the bundled Mozilla roots; [`builder`](Self::builder) takes any store.
/// The keys are a list ([`keys`](ObjectStoreDatasetBuilder::keys), or
/// [`ranges`](ObjectStoreDatasetBuilder::ranges) of objects), a function of the index
/// ([`key_fn`](ObjectStoreDatasetBuilder::key_fn)) or a template with per-item values
/// ([`key_template`](ObjectStoreDatasetBuilder::key_template)), as for the other built-in
/// datasets. An object larger than `max_bytes` fails its item.
///
/// The store retries failed requests itself, which this dataset does not repeat: a store
/// opened by [`from_url`](Self::from_url) gets the dataset's [`Retry`] (default
/// [`Retry::default`]) and `timeout` per request (default 30 s). It retries 5xx statuses,
/// 429 and 408, connection errors and timeouts, with exponential backoff and jitter, within
/// 3 minutes from the first attempt; other statuses (404, 403, …) fail at once.
///
/// ```no_run
/// use dataloader_rs::{DataLoader, dataset::{ObjectStoreDataset, Retry}};
///
/// let keys = (0..1000).map(|i| format!("tiles/{i}.png"));
/// let dataset = ObjectStoreDataset::from_url("s3://bucket", [("aws_region", "eu-west-3")])
///     .retry(Retry { max_retries: 5, ..Retry::default() })
///     .keys(keys)?;
/// let mut loader = DataLoader::builder_async(dataset)
///     .batch_size(64)
///     .item_concurrency(32)
///     .build();
/// # Ok::<(), dataloader_rs::Error>(())
/// ```
///
/// Its `Debug` shows the store as it displays itself (`AmazonS3(bucket)`, …), never the store's
/// credentials.
#[derive(Clone)]
pub struct ObjectStoreDataset {
    store: Arc<dyn ObjectStore>,
    keys: Locations<Key>,
    max_bytes: u64,
    /// Attempts the store makes per request at most, if it was opened by `from_url`.
    max_attempts: Option<u32>,
}

impl Debug for ObjectStoreDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The store's own `Debug` may show credentials (an Azure key or SAS token).
        f.debug_struct("ObjectStoreDataset")
            .field("store", &format_args!("{}", self.store))
            .field("keys", &self.keys)
            .field("max_bytes", &self.max_bytes)
            .field("max_attempts", &self.max_attempts)
            .finish()
    }
}

/// Where the store of an [`ObjectStoreDatasetBuilder`] comes from.
enum Store {
    /// Built by the caller, with its own settings.
    Built(Arc<dyn ObjectStore>),
    /// Opened from a URL and `object_store` options when the dataset is built.
    Url {
        url: String,
        options: Vec<(String, String)>,
    },
}

impl Debug for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Built(store) => write!(f, "Built({store})"),
            Self::Url { url, options } => {
                // Option values may be secrets (`aws_secret_access_key`): keys only.
                let options: Vec<(&str, &str)> = options
                    .iter()
                    .map(|(key, _)| (key.as_str(), "***"))
                    .collect();
                f.debug_struct("Url")
                    .field("url", &redact(url))
                    .field("options", &options)
                    .finish()
            }
        }
    }
}

/// Settings of an [`ObjectStoreDataset`]; [`keys`](Self::keys), [`ranges`](Self::ranges),
/// [`key_fn`](Self::key_fn) or [`key_template`](Self::key_template) builds it. Its `Debug`
/// shows option keys, not their values, and the URL without password, query or fragment.
#[derive(Debug)]
pub struct ObjectStoreDatasetBuilder {
    store: Store,
    retry: Option<Retry>,
    timeout: Option<Duration>,
    max_bytes: u64,
}

impl ObjectStoreDataset {
    /// Settings for a dataset over `store`, which keeps its own retry and timeout settings.
    pub fn builder(store: Arc<dyn ObjectStore>) -> ObjectStoreDatasetBuilder {
        ObjectStoreDatasetBuilder::new(Store::Built(store))
    }

    /// Settings for a dataset over the store and prefix of `url`, opened when the dataset is
    /// built. `options` are `object_store` configuration keys (`aws_region`, `allow_http`, …);
    /// an unknown key is an error, and so is `timeout`, set with
    /// [`timeout`](ObjectStoreDatasetBuilder::timeout). Credentials go in `options`: a URL with
    /// a password, a query or a fragment is an error. Keys and templates are relative to the
    /// prefix.
    pub fn from_url<K, V>(
        url: &str,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> ObjectStoreDatasetBuilder
    where
        K: Into<String>,
        V: Into<String>,
    {
        let options = options
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        ObjectStoreDatasetBuilder::new(Store::Url {
            url: url.to_owned(),
            options,
        })
    }

    /// A dataset of whole objects `keys` in `store`.
    pub fn new<K: AsRef<str>>(
        store: Arc<dyn ObjectStore>,
        keys: impl IntoIterator<Item = K>,
    ) -> Result<Self> {
        Self::builder(store).keys(keys)
    }
}

impl ObjectStoreDatasetBuilder {
    fn new(store: Store) -> Self {
        Self {
            store,
            retry: None,
            timeout: None,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// When the store retries a request. Default [`Retry::default`]; [`Retry::NONE`] for one
    /// attempt. Only for a store opened by [`from_url`](ObjectStoreDataset::from_url).
    pub fn retry(mut self, retry: Retry) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Time allowed for one request, from connecting to the last byte. Default 30 s. Only for
    /// a store opened by [`from_url`](ObjectStoreDataset::from_url).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Cap on the size of one item, in bytes. Default [`DEFAULT_MAX_BYTES`].
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// A dataset of whole objects `keys`.
    pub fn keys<K: AsRef<str>>(
        self,
        keys: impl IntoIterator<Item = K>,
    ) -> Result<ObjectStoreDataset> {
        self.build(|prefix| {
            let keys = keys
                .into_iter()
                .map(|key| parse_key(&join(prefix, key.as_ref())));
            Ok(Locations::List(keys.collect::<Result<_>>()?))
        })
    }

    /// A dataset of byte ranges: item `i` is bytes `range` of object `key`.
    pub fn ranges<K: AsRef<str>>(
        self,
        keys: impl IntoIterator<Item = (K, Range<u64>)>,
    ) -> Result<ObjectStoreDataset> {
        self.build(|prefix| {
            let keys = keys.into_iter().map(|(key, range)| {
                if range.start >= range.end {
                    return Err(format!("empty byte range {range:?}").into());
                }
                Ok((Path::parse(join(prefix, key.as_ref()))?, Some(range)))
            });
            Ok(Locations::List(keys.collect::<Result<_>>()?))
        })
    }

    /// A dataset of `len` whole objects whose keys `key(index)` makes on demand.
    pub fn key_fn(
        self,
        len: usize,
        key: impl Fn(usize) -> String + Send + Sync + 'static,
    ) -> Result<ObjectStoreDataset> {
        self.build(|prefix| {
            let prefix = prefix.to_owned();
            Ok(Locations::Fn {
                len,
                location: Arc::new(move |index| join(&prefix, &key(index))),
            })
        })
    }

    /// A dataset of whole objects whose keys are made on demand from `template` and one value
    /// per item in each of `columns`, without a string per item in memory. The number of
    /// items is the length of the columns, which must all have the same.
    ///
    /// The template is parsed here: `{name}` is the value of column `name` for the item,
    /// `{index}` the index of the item, `{{` and `}}` literal braces; the checks are those of
    /// `HttpDataset`'s `url_template`. Integers are written in decimal, strings as they are,
    /// but a value must be a valid key part (not empty, `.` or `..`, without `/` or control
    /// characters) and an item whose values make a `.`, `..` or empty part with the text
    /// around them fails.
    ///
    /// ```no_run
    /// use dataloader_rs::dataset::{Column, ObjectStoreDataset};
    ///
    /// let slides = Column::from(vec!["a.svs", "a.svs", "b.svs"]);
    /// let dataset = ObjectStoreDataset::from_url("s3://bucket/tiles", [("aws_region", "eu-west-3")])
    ///     .key_template("{slide}/{index}.png", [("slide", slides)])?;
    /// # Ok::<(), dataloader_rs::Error>(())
    /// ```
    pub fn key_template<'a>(
        self,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<ObjectStoreDataset> {
        self.templated(None, template, columns)
    }

    /// [`key_template`](Self::key_template) with `len` items: for a template using only
    /// `{index}`, with no column, or to check the length of the columns.
    pub fn key_template_len<'a>(
        self,
        len: usize,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<ObjectStoreDataset> {
        self.templated(Some(len), template, columns)
    }

    fn templated<'a>(
        self,
        len: Option<usize>,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<ObjectStoreDataset> {
        let columns = columns.into_iter().collect();
        self.build(|prefix| {
            // The prefix is literal text in front of the template.
            let escaped = prefix.replace('{', "{{").replace('}', "}}");
            let template = Template::new(&join(&escaped, template), columns, len, Escape::Key)?;
            Locations::template(template, parse_key)
        })
    }

    /// Opens the store, then makes the keys relative to its prefix.
    fn build(
        self,
        keys: impl FnOnce(&str) -> Result<Locations<Key>>,
    ) -> Result<ObjectStoreDataset> {
        let (store, prefix, max_attempts) = match self.store {
            Store::Built(store) => {
                if self.retry.is_some() || self.timeout.is_some() {
                    return Err("retry and timeout apply to a store opened by from_url; \
                                set them on the client of your own store"
                        .into());
                }
                (store, Path::default(), None)
            }
            Store::Url { url, options } => {
                let retry = self.retry.unwrap_or_default();
                retry.check()?;
                let (store, prefix) = open_store(&url, options, retry, self.timeout)?;
                (store, prefix, Some(retry.max_retries.saturating_add(1)))
            }
        };
        Ok(ObjectStoreDataset {
            store,
            keys: keys(prefix.as_ref())?,
            max_bytes: self.max_bytes,
            max_attempts,
        })
    }
}

/// `key` under `prefix`.
fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_owned()
    } else {
        format!("{prefix}/{key}")
    }
}

/// `url` for messages: without password, query or fragment; the user name stays (the
/// container of an `abfs://container@account…` URL).
fn redact(url: &str) -> String {
    match Url::parse(url) {
        Ok(mut url) => {
            // Err: the URL cannot have a password, so it has none.
            let _ = url.set_password(None);
            url.set_query(None);
            url.set_fragment(None);
            url.to_string()
        }
        Err(_) => "<unparsable>".to_owned(),
    }
}

fn parse_key(key: &str) -> Result<Key> {
    Ok((Path::parse(key)?, None))
}

/// Our retry settings in `object_store`'s: the same retries and backoff (base 2), within its
/// default 3 minutes from the first attempt.
fn retry_config(retry: Retry) -> RetryConfig {
    RetryConfig {
        max_retries: usize::try_from(retry.max_retries).unwrap_or(usize::MAX),
        backoff: BackoffConfig {
            init_backoff: retry.initial_backoff,
            max_backoff: retry.max_backoff,
            base: 2.0,
        },
        ..RetryConfig::default()
    }
}

/// Apply `options` to an `object_store` builder, rejecting unknown keys.
macro_rules! configure {
    ($builder:expr, $options:expr) => {{
        let mut builder = $builder;
        for (key, value) in $options {
            let parsed = key
                .to_ascii_lowercase()
                .parse()
                .map_err(|_| format!("unknown object store option {key:?}"))?;
            builder = builder.with_config(parsed, value);
        }
        builder
    }};
}

/// The store of `url`, with `retry` and `timeout`, and the path prefix it names.
fn open_store(
    url: &str,
    options: Vec<(String, String)>,
    retry: Retry,
    timeout: Option<Duration>,
) -> Result<(Arc<dyn ObjectStore>, Path)> {
    if options
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case("timeout"))
    {
        return Err("set the timeout with the dataset's timeout, not in options".into());
    }
    let url = Url::parse(url)?;
    // object_store's errors repeat the URL: keep secrets out of it.
    if url.password().is_some() || url.query().is_some() || url.fragment().is_some() {
        return Err(format!(
            "the store URL {} may not have a password, a query or a fragment; \
             pass credentials in options",
            redact(url.as_str())
        )
        .into());
    }
    let (scheme, prefix) = ObjectStoreScheme::parse(&url)?;
    let prefix = Path::parse(prefix)?;
    install_ring_provider();
    let roots = mozilla_roots(Certificate::from_der)?;
    let mut client = roots.into_iter().fold(
        ClientOptions::new().with_no_system_certificates(true),
        ClientOptions::with_root_certificate,
    );
    if let Some(timeout) = timeout {
        client = client.with_timeout(timeout);
    }
    let retry = retry_config(retry);
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::AmazonS3 => {
            let builder = AmazonS3Builder::from_env()
                .with_url(url.as_str())
                .with_client_options(client)
                .with_retry(retry);
            Arc::new(configure!(builder, options).build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            let builder = GoogleCloudStorageBuilder::from_env()
                .with_url(url.as_str())
                .with_client_options(client)
                .with_retry(retry);
            Arc::new(configure!(builder, options).build()?)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            let builder = MicrosoftAzureBuilder::from_env()
                .with_url(url.as_str())
                .with_client_options(client)
                .with_retry(retry);
            Arc::new(configure!(builder, options).build()?)
        }
        ObjectStoreScheme::Http => {
            let builder = HttpBuilder::new()
                .with_url(url.origin().ascii_serialization())
                .with_client_options(client)
                .with_retry(retry);
            Arc::new(configure!(builder, options).build()?)
        }
        ObjectStoreScheme::Memory => Arc::new(InMemory::new()),
        scheme => return Err(format!("unsupported object store {scheme:?}").into()),
    };
    Ok((store, prefix))
}

impl ObjectStoreDataset {
    /// A request the store failed, after its own retries.
    fn failed(&self, path: &Path, error: StoreError) -> FetchError {
        FetchError::Store {
            location: path.to_string(),
            max_attempts: self.max_attempts,
            source: error.into(),
        }
    }
}

impl AsyncDataset for ObjectStoreDataset {
    type Item = Bytes;

    async fn get(&self, index: usize) -> Result<Bytes> {
        let key = self.keys.get(index, parse_key)?;
        let (path, range) = key.as_ref();
        let too_large = |size| FetchError::TooLarge {
            location: path.to_string(),
            limit: self.max_bytes,
            size,
        };
        if let Some(range) = range
            && range.end - range.start > self.max_bytes
        {
            return Err(too_large(Some(range.end - range.start)).into());
        }
        let options = GetOptions::new().with_range(range.clone());
        let result = self
            .store
            .get_opts(path, options)
            .await
            .map_err(|error| self.failed(path, error))?;
        let announced = result.range.end.saturating_sub(result.range.start);
        if announced > self.max_bytes {
            return Err(too_large(Some(announced)).into());
        }
        let capacity = usize::try_from(announced.min(INITIAL_CAPACITY))?;
        let mut content = BytesMut::with_capacity(capacity);
        let mut chunks = result.into_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|error| self.failed(path, error))?;
            let size = (content.len() as u64).saturating_add(chunk.len() as u64);
            if size > self.max_bytes {
                return Err(too_large(None).into());
            }
            content.extend_from_slice(&chunk);
        }
        Ok(content.freeze())
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

    use super::{ObjectStoreDataset, retry_config};
    use crate::{
        DataLoader,
        dataset::{AsyncDataset, Column, FetchError, Retry, io::test_server::TestServer},
    };

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    fn memory_store(objects: &[(&str, &[u8])]) -> Arc<dyn ObjectStore> {
        let store = InMemory::new();
        block_on(async {
            for (key, content) in objects {
                let payload = PutPayload::from(content.to_vec());
                store.put(&Path::from(*key), payload).await.unwrap();
            }
        });
        Arc::new(store)
    }

    /// Short backoff so that retry tests run fast.
    fn fast_retry(max_retries: u32) -> Retry {
        Retry {
            max_retries,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
        }
    }

    #[test]
    fn whole_objects_and_ranges_through_the_loader() {
        let store = memory_store(&[("a/0", b"zero"), ("a/1", b"one"), ("b", b"0123456789")]);
        let ds = ObjectStoreDataset::new(Arc::clone(&store), ["a/0", "a/1"]).unwrap();
        let mut loader = DataLoader::builder_async(ds)
            .batch_size(2)
            .item_concurrency(2)
            .build();
        let batch = loader.iter().next().unwrap().unwrap();
        assert_eq!(batch, vec![&b"zero"[..], &b"one"[..]]);

        let ranges = ObjectStoreDataset::builder(store)
            .ranges([("b", 2..5), ("b", 8..10)])
            .unwrap();
        assert_eq!(&block_on(ranges.get(0)).unwrap()[..], b"234");
        assert_eq!(&block_on(ranges.get(1)).unwrap()[..], b"89");
    }

    #[test]
    fn keys_from_a_template_or_a_function() {
        let store = memory_store(&[("s 1/0.png", b"a"), ("s 1/1.png", b"b"), ("t/2.png", b"c")]);
        let slides = Column::from(vec!["s 1", "s 1", "t"]);
        let ds = ObjectStoreDataset::builder(Arc::clone(&store))
            .key_template("{slide}/{index}.png", [("slide", slides)])
            .unwrap();
        let items: Vec<_> = (0..3).map(|i| block_on(ds.get(i)).unwrap()).collect();
        assert_eq!(items, [&b"a"[..], b"b", b"c"]);

        let ds = ObjectStoreDataset::builder(Arc::clone(&store))
            .key_fn(2, |i| format!("s 1/{i}.png"))
            .unwrap();
        assert_eq!(&block_on(ds.get(1)).unwrap()[..], b"b");
        let ds = ObjectStoreDataset::builder(store)
            .key_fn(1, |_| "a/../b".to_owned())
            .unwrap();
        assert!(block_on(ds.get(0)).is_err());
    }

    #[test]
    fn a_template_value_must_be_a_valid_key_part() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        for value in ["", ".", "..", "a/b", "a\nb"] {
            let names = Column::from(vec!["ok", value]);
            let error = ObjectStoreDataset::builder(Arc::clone(&store))
                .key_template("tiles/{name}", [("name", names)])
                .unwrap_err();
            assert!(error.to_string().contains("item 1"), "{value:?}: {error}");
        }
    }

    #[test]
    fn missing_objects_out_of_range_and_oversized_items_fail() {
        let store = memory_store(&[("big", &[1u8; 100])]);
        let ds = ObjectStoreDataset::builder(Arc::clone(&store))
            .max_bytes(99)
            .keys(["missing", "big"])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref(),
                Some(FetchError::Store {
                    max_attempts: None,
                    ..
                })
            ),
            "{err}"
        );
        assert!(
            err.to_string()
                .starts_with("missing: failed, retried by the store's own settings: "),
            "{err}"
        );
        let err = block_on(ds.get(1)).unwrap_err();
        assert!(matches!(
            err.downcast_ref(),
            Some(FetchError::TooLarge {
                size: Some(100),
                ..
            })
        ));
        let err = block_on(ds.get(2)).unwrap_err();
        assert!(matches!(
            err.downcast_ref(),
            Some(FetchError::OutOfRange { index: 2, len: 2 })
        ));
        let ranges = ObjectStoreDataset::builder(store)
            .max_bytes(10)
            .ranges([("big", 0..100)])
            .unwrap();
        assert!(block_on(ranges.get(0)).is_err());
    }

    #[test]
    fn rejects_bad_keys_ranges_urls_and_options() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        assert!(ObjectStoreDataset::new(Arc::clone(&store), ["a/../b"]).is_err());
        let builder = || ObjectStoreDataset::builder(Arc::clone(&store));
        assert!(builder().ranges([("a", 5..5)]).is_err());
        // A store built by the caller keeps its own retry and timeout settings.
        assert!(builder().retry(Retry::NONE).keys(["a"]).is_err());
        assert!(
            builder()
                .timeout(Duration::from_secs(1))
                .keys(["a"])
                .is_err()
        );
        let s3 = |options: &[(&str, &str)]| {
            ObjectStoreDataset::from_url("s3://bucket/prefix", options.to_vec())
        };
        assert!(
            ObjectStoreDataset::from_url("ftp://host/x", [("a", "b")])
                .keys(["k"])
                .is_err()
        );
        let unknown = s3(&[("aws_region", "eu-west-3"), ("no_such_option", "1")]).keys(["k"]);
        assert!(unknown.unwrap_err().to_string().contains("no_such_option"));
        let timeout = s3(&[("aws_region", "eu-west-3"), ("TIMEOUT", "5s")]).keys(["k"]);
        assert!(timeout.unwrap_err().to_string().contains("timeout"));
        let too_many = Retry {
            max_retries: 33,
            ..Retry::default()
        };
        assert!(
            s3(&[("aws_region", "eu-west-3")])
                .retry(too_many)
                .keys(["k"])
                .is_err()
        );
        let ds = s3(&[("aws_region", "eu-west-3")]).keys(["k"]).unwrap();
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn debug_and_errors_hide_secrets() {
        let secrets = [
            ("aws_access_key_id", "AKIDEXAMPLE"),
            ("aws_secret_access_key", "s3cr3t-key"),
            ("aws_session_token", "s3ssion-token"),
            ("aws_region", "eu-west-3"),
        ];
        let builder = ObjectStoreDataset::from_url("s3://bucket/prefix", secrets);
        let debug = format!("{builder:?}");
        for (key, value) in secrets {
            assert!(debug.contains(key), "{debug}");
            assert!(!debug.contains(value), "{debug}");
        }
        let ds = builder.keys(["k"]).unwrap();
        let debug = format!("{ds:?}");
        assert!(debug.contains("AmazonS3(bucket)"), "{debug}");
        assert!(
            !debug.contains("s3cr3t") && !debug.contains("s3ssion"),
            "{debug}"
        );

        // An Azure store's own Debug shows its key (as bytes); the dataset's does not.
        let key = "c2VjcmV0LWF6dXJlLWtleQ==";
        let ds = ObjectStoreDataset::from_url(
            "az://container",
            [
                ("azure_storage_account_name", "account"),
                ("azure_storage_account_key", key),
            ],
        )
        .keys(["k"])
        .unwrap();
        assert!(format!("{:?}", ds.store).contains("AzureAccessKey(["));
        let debug = format!("{ds:?}");
        assert!(
            debug.contains("MicrosoftAzure { account: account, container: container }"),
            "{debug}"
        );
        assert!(!debug.contains("AccessKey"), "{debug}");

        // A password or a query in the URL: refused, and not repeated.
        let url = "https://user:hunter2@host.example/prefix?sig=s1gnature";
        let builder = ObjectStoreDataset::from_url(url, [("allow_http", "true")]);
        let debug = format!("{builder:?}");
        assert!(
            debug.contains("https://user@host.example/prefix"),
            "{debug}"
        );
        assert!(
            !debug.contains("hunter2") && !debug.contains("s1gnature"),
            "{debug}"
        );
        let message = builder.keys(["k"]).unwrap_err().to_string();
        assert!(message.contains("host.example/prefix"), "{message}");
        assert!(
            !message.contains("hunter2") && !message.contains("s1gnature"),
            "{message}"
        );
    }

    #[test]
    fn our_retry_settings_become_the_store_s() {
        let config = retry_config(Retry {
            max_retries: 5,
            initial_backoff: Duration::from_millis(20),
            max_backoff: Duration::from_secs(2),
        });
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.backoff.init_backoff, Duration::from_millis(20));
        assert_eq!(config.backoff.max_backoff, Duration::from_secs(2));
        assert_eq!(config.backoff.base, 2.0);
        assert_eq!(config.retry_timeout, Duration::from_secs(180));
        assert_eq!(retry_config(Retry::NONE).max_retries, 0);
    }

    #[test]
    fn http_store_from_url_with_a_prefix_and_a_template() {
        let server = TestServer::start();
        let http = || ObjectStoreDataset::from_url(&server.url("/ok"), [("allow_http", "true")]);
        let ds = http().keys(["3", "4"]).unwrap();
        assert_eq!(&block_on(ds.get(0)).unwrap()[..], b"item-3");
        assert_eq!(&block_on(ds.get(1)).unwrap()[..], b"item-4");
        let ds = http().key_template_len(2, "{index}", []).unwrap();
        assert_eq!(&block_on(ds.get(1)).unwrap()[..], b"item-1");
    }

    #[test]
    fn the_store_retries_with_our_settings() {
        let server = TestServer::start();
        let flaky = |retry| {
            ObjectStoreDataset::from_url(&server.url("/flaky/503/2"), [("allow_http", "true")])
                .retry(retry)
        };
        let ds = flaky(fast_retry(3)).keys(["a"]).unwrap();
        assert_eq!(&block_on(ds.get(0)).unwrap()[..], b"ok");
        assert_eq!(server.hits("/flaky/503/2/a"), 3);
        let ds = flaky(Retry::NONE).keys(["b"]).unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert_eq!(server.hits("/flaky/503/2/b"), 1);
        assert!(
            err.to_string()
                .contains("failed after up to 1 attempt(s), retried by object_store: "),
            "{err}"
        );
        // The store does not retry a 404; the error gives the most attempts it may make.
        let ds = ObjectStoreDataset::from_url(&server.url("/none"), [("allow_http", "true")])
            .retry(fast_retry(2))
            .keys(["c"])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref(),
                Some(FetchError::Store {
                    max_attempts: Some(3),
                    ..
                })
            ),
            "{err}"
        );
    }

    #[test]
    fn a_silent_server_times_out() {
        let server = TestServer::start();
        let ds = ObjectStoreDataset::from_url(&server.url(""), [("allow_http", "true")])
            .timeout(Duration::from_millis(300))
            .retry(Retry::NONE)
            .keys(["hang"])
            .unwrap();
        let start = Instant::now();
        assert!(block_on(ds.get(0)).is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
