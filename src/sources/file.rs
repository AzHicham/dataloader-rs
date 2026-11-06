//! [`FileSource`]: one file per item.

use std::fs::File;
use std::path::{Path, PathBuf};

use super::{DEFAULT_MAX_BYTES, ReadError, Source, SourceError, read_capped};

/// One file per item, read whole on each `fetch`.
///
/// A file larger than [`max_bytes`](Self::max_bytes) is an error, checked against its length
/// before reading and again while reading (a file can grow, and special files report no
/// length). Paths are used as given: confine them yourself if they come from untrusted input.
#[derive(Clone, Debug)]
pub struct FileSource {
    paths: Vec<PathBuf>,
    max_bytes: u64,
}

impl FileSource {
    /// A source of the files at `paths`, in order, capped at [`DEFAULT_MAX_BYTES`] each.
    pub fn new<P: Into<PathBuf>>(paths: impl IntoIterator<Item = P>) -> Self {
        Self {
            paths: paths.into_iter().map(Into::into).collect(),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Cap on the size of one file, in bytes (at least 1).
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes.max(1);
        self
    }

    /// The paths.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn read(&self, index: usize, path: &Path) -> Result<Vec<u8>, SourceError> {
        let file_error = |source| SourceError::File {
            index,
            path: path.to_path_buf(),
            source,
        };
        let file = File::open(path).map_err(file_error)?;
        let size = file.metadata().map_err(file_error)?.len();
        // A zero length is also what special files report: rely on the read's cap then.
        let hint = (size > 0).then_some(size);
        read_capped(file, hint, self.max_bytes).map_err(|error| match error {
            ReadError::TooLarge(size) => SourceError::TooLarge {
                index,
                location: path.display().to_string(),
                limit: self.max_bytes,
                size,
            },
            ReadError::Io(source) => file_error(source),
        })
    }
}

impl Source for FileSource {
    type Bytes = Vec<u8>;

    fn fetch(&self, index: usize) -> Result<Vec<u8>, SourceError> {
        let path = self.paths.get(index).ok_or(SourceError::OutOfRange {
            index,
            len: self.paths.len(),
        })?;
        self.read(index, path)
    }

    fn len(&self) -> usize {
        self.paths.len()
    }
}
