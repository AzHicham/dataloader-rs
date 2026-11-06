//! A tiny HTTP/1.1 server on `127.0.0.1` for the `sources` tests and bench: keep-alive, one
//! thread per connection, a handler deciding each reply, and counters.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// A request as the handler sees it.
pub struct Request {
    /// Path and query, e.g. `/a/1.png?x=2`.
    pub target: String,
    /// Header lines, names lowercased.
    pub headers: Vec<(String, String)>,
    /// Requests the server has received so far, this one included.
    pub number: usize,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// What the server answers.
pub enum Reply {
    /// A response with a `Content-Length`; the connection stays open.
    Body {
        status: u16,
        headers: Vec<(String, String)>,
        body: Arc<[u8]>,
    },
    /// These bytes as they are, then close the connection.
    Raw(Vec<u8>),
    /// A 200 with a chunked body that never ends.
    Endless,
}

impl Reply {
    pub fn ok(body: impl Into<Arc<[u8]>>) -> Self {
        Self::Body {
            status: 200,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn status(status: u16) -> Self {
        Self::Body {
            status,
            headers: Vec::new(),
            body: Arc::from(&b""[..]),
        }
    }
}

pub struct Server {
    pub addr: SocketAddr,
    requests: Arc<AtomicUsize>,
    connections: Arc<AtomicUsize>,
}

impl Server {
    pub fn start(handler: impl Fn(&Request) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let connections = Arc::new(AtomicUsize::new(0));
        let handler = Arc::new(handler);
        {
            let requests = requests.clone();
            let connections = connections.clone();
            thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    connections.fetch_add(1, Ordering::SeqCst);
                    let handler = handler.clone();
                    let requests = requests.clone();
                    thread::spawn(move || serve(stream, &*handler, &requests));
                }
            });
        }
        Self {
            addr,
            requests,
            connections,
        }
    }

    pub fn url(&self, target: &str) -> String {
        format!("http://{}{target}", self.addr)
    }

    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    pub fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }
}

fn serve(
    stream: TcpStream,
    handler: &(dyn Fn(&Request) -> Reply + Send + Sync),
    requests: &AtomicUsize,
) {
    let _ = stream.set_nodelay(true);
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(read_half);
    let mut stream = stream;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let target = line.split(' ').nth(1).unwrap_or("/").to_string();
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
            }
        }
        let number = requests.fetch_add(1, Ordering::SeqCst) + 1;
        let reply = handler(&Request {
            target,
            headers,
            number,
        });
        match reply {
            Reply::Body {
                status,
                headers,
                body,
            } => {
                let mut head = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\n", body.len());
                for (name, value) in headers {
                    head.push_str(&format!("{name}: {value}\r\n"));
                }
                head.push_str("\r\n");
                if stream.write_all(head.as_bytes()).is_err() || stream.write_all(&body).is_err() {
                    return;
                }
            }
            Reply::Raw(bytes) => {
                let _ = stream.write_all(&bytes);
                return;
            }
            Reply::Endless => {
                if stream
                    .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                    .is_err()
                {
                    return;
                }
                let chunk = [b'x'; 1 << 16];
                loop {
                    let ok = stream.write_all(b"10000\r\n").is_ok()
                        && stream.write_all(&chunk).is_ok()
                        && stream.write_all(b"\r\n").is_ok();
                    if !ok {
                        return;
                    }
                }
            }
        }
    }
}
