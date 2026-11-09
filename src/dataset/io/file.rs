//! [`FileDataset`] and its builder: one local file per item, read with `tokio::fs`.

use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::time::timeout;

use super::{
    location::Locations,
    retry::Transient,
    template::{Escape, Template},
};
use crate::{
    dataset::{AsyncDataset, Column, FetchError, Retry},
    error::Result,
};

/// Default cap on the size of one item, for every built-in dataset: 64 MiB.
pub const DEFAULT_MAX_BYTES: u64 = 64 << 20;

/// One file per item: item `i` is the content of path `i`, read by tokio's blocking pool.
///
/// The paths are a list ([`paths`](FileDatasetBuilder::paths)), a function of the index
/// ([`path_fn`](FileDatasetBuilder::path_fn)) or a template with per-item values
/// ([`path_template`](FileDatasetBuilder::path_template)), as for the other built-in datasets.
/// A file larger than `max_bytes` fails its item ([`FetchError::TooLarge`]), checked before
/// reading and again while reading, so a file that grows cannot exceed it.
///
/// By default an item is read once and may take as long as the file system does. On a network
/// file system, [`timeout`](FileDatasetBuilder::timeout) bounds each attempt and
/// [`retry`](FileDatasetBuilder::retry) tries again after a timeout or an I/O error of kind
/// `TimedOut`, `Interrupted` or `WouldBlock`.
///
/// ```no_run
/// use dataloader_rs::{DataLoader, dataset::FileDataset};
///
/// let dataset = FileDataset::new(["a.png", "b.png"]);
/// let mut loader = DataLoader::builder_async(dataset)
///     .batch_size(2)
///     .item_concurrency(2)
///     .build();
/// for batch in &mut loader {
///     let files: Vec<bytes::Bytes> = batch.unwrap();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct FileDataset {
    paths: Locations<PathBuf>,
    retry: Retry,
    timeout: Option<Duration>,
    max_bytes: u64,
}

/// Settings of a [`FileDataset`]; [`paths`](Self::paths), [`path_fn`](Self::path_fn) or
/// [`path_template`](Self::path_template) builds it.
#[derive(Clone, Debug)]
pub struct FileDatasetBuilder {
    retry: Retry,
    timeout: Option<Duration>,
    max_bytes: u64,
}

impl Default for FileDatasetBuilder {
    fn default() -> Self {
        Self {
            retry: Retry {
                max_retries: 0,
                ..Retry::default()
            },
            timeout: None,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl FileDatasetBuilder {
    /// When to retry. Default: no retry (with [`Retry::default`]'s backoff once
    /// `max_retries` is set).
    pub fn retry(mut self, retry: Retry) -> Self {
        self.retry = retry;
        self
    }

    /// Time allowed for one attempt, from opening the file to its last byte. Default: none.
    /// A read past it is abandoned, but the blocking-pool thread doing it stays busy until the
    /// file system answers.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Cap on the size of one file, in bytes. Default [`DEFAULT_MAX_BYTES`].
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// A dataset of `paths`.
    pub fn paths<P: Into<PathBuf>>(
        self,
        paths: impl IntoIterator<Item = P>,
    ) -> Result<FileDataset> {
        self.build(Locations::List(paths.into_iter().map(Into::into).collect()))
    }

    /// A dataset of `len` items whose paths `path(index)` makes on demand.
    pub fn path_fn(
        self,
        len: usize,
        path: impl Fn(usize) -> String + Send + Sync + 'static,
    ) -> Result<FileDataset> {
        self.build(Locations::Fn {
            len,
            location: Arc::new(path),
        })
    }

    /// A dataset whose paths are made on demand from `template` and one value per item in
    /// each of `columns`, without a string per item in memory. The number of items is the
    /// length of the columns, which must all have the same.
    ///
    /// The template is parsed here: `{name}` is the value of column `name` for the item,
    /// `{index}` the index of the item, `{{` and `}}` literal braces; the checks are those of
    /// `HttpDataset`'s `url_template`. Integers are written in decimal, strings as they are,
    /// but a value holding `/`, `\` or NUL, or equal to `.` or `..`, is an error here, and an
    /// item whose values make a `.` or `..` path component with the text around them fails:
    /// a value cannot move out of the template's directory.
    ///
    /// ```no_run
    /// use dataloader_rs::dataset::{Column, FileDataset};
    ///
    /// let slides = Column::from(vec!["a.svs", "a.svs", "b.svs"]);
    /// let levels = Column::from(vec![0u64, 1, 0]);
    /// let dataset = FileDataset::builder().path_template(
    ///     "/data/tiles/{slide}/{level}/{index}.png",
    ///     [("slide", slides), ("level", levels)],
    /// )?;
    /// # Ok::<(), dataloader_rs::Error>(())
    /// ```
    pub fn path_template<'a>(
        self,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<FileDataset> {
        self.templated(None, template, columns)
    }

    /// [`path_template`](Self::path_template) with `len` items: for a template using only
    /// `{index}`, with no column, or to check the length of the columns.
    pub fn path_template_len<'a>(
        self,
        len: usize,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<FileDataset> {
        self.templated(Some(len), template, columns)
    }

    fn templated<'a>(
        self,
        len: Option<usize>,
        template: &str,
        columns: impl IntoIterator<Item = (&'a str, Column)>,
    ) -> Result<FileDataset> {
        let columns = columns.into_iter().collect();
        let template = Template::new(template, columns, len, Escape::Path)?;
        self.build(Locations::template(template, parse_path)?)
    }

    fn build(self, paths: Locations<PathBuf>) -> Result<FileDataset> {
        self.retry.check()?;
        Ok(FileDataset {
            paths,
            retry: self.retry,
            timeout: self.timeout,
            max_bytes: self.max_bytes,
        })
    }
}

fn parse_path(path: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(path))
}

/// How one attempt failed.
enum Attempt {
    Io(IoError),
    TimedOut(Duration),
    TooLarge(Option<u64>),
}

impl Transient for Attempt {
    fn is_transient(&self) -> bool {
        match self {
            Self::Io(error) => matches!(
                error.kind(),
                ErrorKind::TimedOut | ErrorKind::Interrupted | ErrorKind::WouldBlock
            ),
            Self::TimedOut(_) => true,
            Self::TooLarge(_) => false,
        }
    }
}

impl Attempt {
    fn into_error(self, path: &Path, attempts: u32, limit: u64) -> FetchError {
        let location = path.display().to_string();
        match self {
            Self::Io(error) => FetchError::Io {
                location,
                attempts,
                timeout: error.kind() == ErrorKind::TimedOut,
                source: error.into(),
            },
            Self::TimedOut(timeout) => FetchError::Io {
                location,
                attempts,
                timeout: true,
                source: format!("no answer within {timeout:?}").into(),
            },
            Self::TooLarge(size) => FetchError::TooLarge {
                location,
                limit,
                size,
            },
        }
    }
}

impl FileDataset {
    /// Settings with their defaults: no retry, no timeout, [`DEFAULT_MAX_BYTES`].
    pub fn builder() -> FileDatasetBuilder {
        FileDatasetBuilder::default()
    }

    /// A dataset of `paths` with the default settings.
    pub fn new<P: Into<PathBuf>>(paths: impl IntoIterator<Item = P>) -> Self {
        let builder = Self::builder();
        Self {
            paths: Locations::List(paths.into_iter().map(Into::into).collect()),
            retry: builder.retry,
            timeout: builder.timeout,
            max_bytes: builder.max_bytes,
        }
    }

    async fn read(&self, path: &Path) -> StdResult<Bytes, Attempt> {
        match self.timeout {
            Some(limit) => timeout(limit, self.read_capped(path))
                .await
                .unwrap_or(Err(Attempt::TimedOut(limit))),
            None => self.read_capped(path).await,
        }
    }

    /// The content of `path`, failing as soon as it is known to exceed `max_bytes`.
    async fn read_capped(&self, path: &Path) -> StdResult<Bytes, Attempt> {
        let file = File::open(path).await.map_err(Attempt::Io)?;
        let size = file.metadata().await.map_err(Attempt::Io)?.len();
        if size > self.max_bytes {
            return Err(Attempt::TooLarge(Some(size)));
        }
        // `size` is at most `max_bytes`; read one byte past the cap to notice a file that grew.
        let capacity = usize::try_from(size).map_err(|_| Attempt::TooLarge(Some(size)))?;
        let mut content = Vec::with_capacity(capacity);
        file.take(self.max_bytes.saturating_add(1))
            .read_to_end(&mut content)
            .await
            .map_err(Attempt::Io)?;
        if content.len() as u64 > self.max_bytes {
            return Err(Attempt::TooLarge(None));
        }
        Ok(Bytes::from(content))
    }
}

impl AsyncDataset for FileDataset {
    type Item = Bytes;

    async fn get(&self, index: usize) -> Result<Bytes> {
        let path = self.paths.get(index, parse_path)?;
        let path = path.as_path();
        self.retry
            .run(|| self.read(path))
            .await
            .map_err(|(failure, attempts)| {
                failure.into_error(path, attempts, self.max_bytes).into()
            })
    }

    fn len(&self) -> usize {
        self.paths.len()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Error as IoError, ErrorKind};
    use std::path::PathBuf;
    use std::process;
    use std::time::{Duration, Instant};

    use super::{Attempt, FileDataset, Transient};
    use crate::{
        dataset::{AsyncDataset, Column, FetchError, Retry},
        error::Error,
    };

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(future)
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dlrs-file-{name}-{}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fetch(error: &Error) -> &FetchError {
        error.downcast_ref().expect("a FetchError")
    }

    #[test]
    fn reads_files_and_reports_missing_ones() {
        let dir = temp_dir("read");
        fs::write(dir.join("a"), b"alpha").unwrap();
        fs::write(dir.join("b"), b"").unwrap();
        let ds = FileDataset::new([dir.join("a"), dir.join("b"), dir.join("missing")]);
        assert_eq!(ds.len(), 3);
        assert_eq!(&block_on(ds.get(0)).unwrap()[..], b"alpha");
        assert!(block_on(ds.get(1)).unwrap().is_empty());
        let err = block_on(ds.get(2)).unwrap_err();
        assert!(
            matches!(fetch(&err), FetchError::Io { attempts: 1, .. }),
            "{err}"
        );
        let err = block_on(ds.get(3)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::OutOfRange { index: 3, len: 3 }
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_file_over_the_cap_fails() {
        let dir = temp_dir("cap");
        fs::write(dir.join("big"), vec![7u8; 100]).unwrap();
        let ds = FileDataset::builder()
            .max_bytes(99)
            .paths([dir.join("big")])
            .unwrap();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(matches!(
            fetch(&err),
            FetchError::TooLarge {
                limit: 99,
                size: Some(100),
                ..
            }
        ));
        let ds = FileDataset::builder()
            .max_bytes(100)
            .paths([dir.join("big")])
            .unwrap();
        assert_eq!(block_on(ds.get(0)).unwrap().len(), 100);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn paths_from_a_template_or_a_function() {
        let dir = temp_dir("template");
        for (slide, level) in [("a b", 0), ("a b", 1), ("c", 0)] {
            fs::create_dir_all(dir.join(slide)).unwrap();
            fs::write(dir.join(slide).join(format!("{level}.bin")), slide).unwrap();
        }
        let template = format!("{}/{{slide}}/{{level}}.bin", dir.display());
        let ds = FileDataset::builder()
            .path_template(
                &template,
                [
                    ("slide", Column::from(vec!["a b", "a b", "c"])),
                    ("level", Column::from(vec![0u64, 1, 0])),
                ],
            )
            .unwrap();
        assert_eq!(ds.len(), 3);
        let items: Vec<_> = (0..3).map(|i| block_on(ds.get(i)).unwrap()).collect();
        assert_eq!(items, [&b"a b"[..], b"a b", b"c"]);

        let base = dir.clone();
        let ds = FileDataset::builder()
            .path_fn(2, move |i| format!("{}/c/{i}.bin", base.display()))
            .unwrap();
        assert_eq!(&block_on(ds.get(0)).unwrap()[..], b"c");
        assert!(matches!(
            fetch(&block_on(ds.get(1)).unwrap_err()),
            FetchError::Io { .. }
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_template_value_cannot_leave_the_directory() {
        let template = "/data/{name}.bin";
        for value in ["..", ".", "../etc/passwd", "a/b", "a\\b", "a\0b"] {
            let names = Column::from(vec!["ok", value]);
            let error = FileDataset::builder()
                .path_template(template, [("name", names)])
                .unwrap_err();
            assert!(error.to_string().contains("item 1"), "{value:?}: {error}");
        }
        // Next to the template's dots, an empty value would make "..": its item fails.
        let ds = FileDataset::builder()
            .path_template("/data/..{name}/x", [("name", Column::from(vec!["a", ""]))])
            .unwrap();
        let error = block_on(ds.get(1)).unwrap_err().to_string();
        assert!(error.contains("\"..\""), "{error}");
    }

    #[test]
    fn transient_failures_are_io_timeouts_interruptions_and_would_block() {
        for kind in [
            ErrorKind::TimedOut,
            ErrorKind::Interrupted,
            ErrorKind::WouldBlock,
        ] {
            assert!(Attempt::Io(IoError::from(kind)).is_transient(), "{kind:?}");
        }
        for kind in [
            ErrorKind::NotFound,
            ErrorKind::PermissionDenied,
            ErrorKind::InvalidData,
        ] {
            assert!(!Attempt::Io(IoError::from(kind)).is_transient(), "{kind:?}");
        }
        assert!(Attempt::TimedOut(Duration::ZERO).is_transient());
        assert!(!Attempt::TooLarge(None).is_transient());
    }

    #[test]
    fn a_missing_file_is_not_retried() {
        let ds = FileDataset::builder()
            .retry(Retry {
                max_retries: 3,
                initial_backoff: Duration::from_secs(10),
                max_backoff: Duration::from_secs(10),
            })
            .paths(["/no/such/file"])
            .unwrap();
        let start = Instant::now();
        let err = block_on(ds.get(0)).unwrap_err();
        assert!(matches!(fetch(&err), FetchError::Io { attempts: 1, .. }));
        assert!(start.elapsed() < Duration::from_secs(5));
        let too_many = Retry {
            max_retries: 33,
            ..Retry::default()
        };
        assert!(FileDataset::builder().retry(too_many).paths(["x"]).is_err());
    }

    /// A FIFO without a writer blocks `open` like a hung network file system.
    #[cfg(unix)]
    #[test]
    fn a_hung_read_times_out_and_is_retried() {
        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;
        use std::process::Command;

        let dir = temp_dir("fifo");
        let fifo = dir.join("hung");
        let made = Command::new("mkfifo").arg(&fifo).status().unwrap();
        assert!(made.success());
        let ds = FileDataset::builder()
            .timeout(Duration::from_millis(100))
            .retry(Retry {
                max_retries: 1,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(1),
            })
            .paths([&fifo])
            .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let start = Instant::now();
        let err = runtime.block_on(ds.get(0)).unwrap_err();
        assert!(fetch(&err).is_timeout(), "{err}");
        assert!(matches!(fetch(&err), FetchError::Io { attempts: 2, .. }));
        assert!(start.elapsed() < Duration::from_secs(2));
        // Release the blocked opens: a writer lets them return (and read an empty file).
        let _writer = OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&fifo);
        runtime.shutdown_background();
        fs::remove_dir_all(dir).unwrap();
    }
}
