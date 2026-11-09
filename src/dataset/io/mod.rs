mod error;
mod file;
#[cfg(feature = "http")]
mod http;
mod location;
#[cfg(feature = "object-store")]
mod object_store;
mod retry;
mod template;
#[cfg(all(test, any(feature = "http", feature = "object-store")))]
#[cfg_attr(not(feature = "http"), allow(dead_code))]
mod test_server;
#[cfg(any(feature = "http", feature = "object-store"))]
mod tls;

#[cfg(feature = "object-store")]
pub use self::object_store::{ObjectStoreDataset, ObjectStoreDatasetBuilder};
pub use error::FetchError;
pub use file::{DEFAULT_MAX_BYTES, FileDataset, FileDatasetBuilder};
#[cfg(feature = "http")]
pub use http::{HttpDataset, HttpDatasetBuilder};
pub use retry::Retry;
pub use template::Column;
