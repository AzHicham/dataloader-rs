//! TLS setup shared by the network datasets: rustls with the *ring* provider, and the bundled
//! Mozilla roots instead of the system store, so that the same certificates are trusted
//! everywhere the crate (or the Python wheel) runs.

use std::result::Result as StdResult;
use std::sync::Once;

use rustls::crypto::ring;

use crate::error::{Error, Result};

/// Make *ring* the process's rustls provider, unless one is installed already (reqwest built
/// without a provider uses the process's).
pub(crate) fn install_ring_provider() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Err: another provider was installed first, which rustls then uses.
        let _ = ring::default_provider().install_default();
    });
}

/// The bundled Mozilla root certificates, each made by `from_der`.
pub(crate) fn mozilla_roots<C, E>(from_der: impl Fn(&[u8]) -> StdResult<C, E>) -> Result<Vec<C>>
where
    E: Into<Error>,
{
    webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .map(|der| from_der(der).map_err(Into::into))
        .collect()
}
