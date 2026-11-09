//! A raw HTTP/1.1 server for the dataset tests, plain or TLS, that can misbehave on purpose.
//!
//! Routes (one connection per request, closed after the response):
//! - `/ok/<n>`: 200, body `item-<n>`;
//! - `/status/<code>`: that status, always;
//! - `/flaky/<code>/<k>/<id>`: `code` with `Retry-After: 0` for the first `k` hits, then 200;
//! - `/huge`: announces a `Content-Length` of 1 TB, sends 16 bytes;
//! - `/chunked/<n>`: `n` bytes, chunked, no `Content-Length`;
//! - `/slow`: announces 100 bytes, sends one every 50 ms (slow loris);
//! - `/hang`: reads the request and never answers.

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use rustls::ServerConfig;
use rustls::crypto::ring;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::runtime::Builder;
use tokio::time::sleep;
use tokio_rustls::TlsAcceptor;

pub(crate) const CA_PEM: &[u8] = include_bytes!("../../../tests/fixtures/tls/ca.pem");
const CERT_PEM: &[u8] = include_bytes!("../../../tests/fixtures/tls/localhost.pem");
const KEY_PEM: &[u8] = include_bytes!("../../../tests/fixtures/tls/localhost.key");

pub(crate) struct TestServer {
    pub(crate) addr: SocketAddr,
    hits: Arc<Mutex<HashMap<String, u32>>>,
    tls: bool,
}

impl TestServer {
    /// A plain HTTP server on a thread of its own, alive until the test process ends.
    pub(crate) fn start() -> Self {
        Self::spawn(None)
    }

    /// The same server behind TLS, with the `localhost` certificate of `tests/fixtures/tls`.
    pub(crate) fn start_tls() -> Self {
        let certs = vec![CertificateDer::from_pem_slice(CERT_PEM).unwrap()];
        let key = PrivateKeyDer::from_pem_slice(KEY_PEM).unwrap();
        let config = ServerConfig::builder_with_provider(Arc::new(ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .unwrap();
        Self::spawn(Some(TlsAcceptor::from(Arc::new(config))))
    }

    fn spawn(tls: Option<TlsAcceptor>) -> Self {
        let hits = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = mpsc::channel();
        let server_hits = Arc::clone(&hits);
        let is_tls = tls.is_some();
        thread::spawn(move || {
            let runtime = Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                tx.send(listener.local_addr().unwrap()).unwrap();
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    let hits = Arc::clone(&server_hits);
                    let tls = tls.clone();
                    tokio::spawn(async move {
                        match tls {
                            Some(tls) => {
                                if let Ok(stream) = tls.accept(stream).await {
                                    serve(stream, hits).await;
                                }
                            }
                            None => serve(stream, hits).await,
                        }
                    });
                }
            });
        });
        Self {
            addr: rx.recv().unwrap(),
            hits,
            tls: is_tls,
        }
    }

    pub(crate) fn url(&self, path: &str) -> String {
        let scheme = if self.tls { "https" } else { "http" };
        let host = if self.tls { "localhost" } else { "127.0.0.1" };
        format!("{scheme}://{host}:{}{path}", self.addr.port())
    }

    /// Requests received for `path`.
    pub(crate) fn hits(&self, path: &str) -> u32 {
        let hits = self.hits.lock().unwrap_or_else(PoisonError::into_inner);
        hits.get(path).copied().unwrap_or(0)
    }
}

async fn serve<S: AsyncRead + AsyncWrite + Unpin>(
    mut stream: S,
    hits: Arc<Mutex<HashMap<String, u32>>>,
) {
    let mut request = Vec::new();
    let mut buf = [0u8; 1024];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    }
    let head = String::from_utf8_lossy(&request);
    let path = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let hit = {
        let mut hits = hits.lock().unwrap_or_else(PoisonError::into_inner);
        let hit = hits.entry(path.clone()).or_insert(0);
        *hit += 1;
        *hit
    };
    let parts: Vec<&str> = path.split('/').skip(1).collect();
    let _ = match parts.as_slice() {
        ["ok", n] => reply(&mut stream, 200, &format!("item-{n}")).await,
        ["status", code] => reply(&mut stream, code.parse().unwrap(), "no").await,
        ["flaky", code, k, _] if hit <= k.parse().unwrap() => {
            let head = format!(
                "HTTP/1.1 {code} X\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(head.as_bytes()).await
        }
        ["flaky", ..] => reply(&mut stream, 200, "ok").await,
        ["huge"] => {
            let head =
                "HTTP/1.1 200 OK\r\nContent-Length: 1000000000000\r\nConnection: close\r\n\r\n";
            stream.write_all(head.as_bytes()).await.ok();
            stream.write_all(&[0u8; 16]).await
        }
        ["chunked", n] => chunked(&mut stream, n.parse().unwrap()).await,
        ["slow"] => {
            let head = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n";
            stream.write_all(head.as_bytes()).await.ok();
            for _ in 0..100 {
                stream.flush().await.ok();
                sleep(Duration::from_millis(50)).await;
                if stream.write_all(b"x").await.is_err() {
                    break;
                }
            }
            Ok(())
        }
        ["hang"] => {
            sleep(Duration::from_secs(3600)).await;
            Ok(())
        }
        _ => reply(&mut stream, 404, "not found").await,
    };
    let _ = stream.shutdown().await;
}

async fn reply<S: AsyncWrite + Unpin>(stream: &mut S, status: u16, body: &str) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await
}

async fn chunked<S: AsyncWrite + Unpin>(stream: &mut S, n: usize) -> io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    let mut left = n;
    while left > 0 {
        let size = left.min(1024);
        stream.write_all(format!("{size:x}\r\n").as_bytes()).await?;
        stream.write_all(&vec![b'c'; size]).await?;
        stream.write_all(b"\r\n").await?;
        left -= size;
    }
    stream.write_all(b"0\r\n\r\n").await
}
