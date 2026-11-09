//! A tiny HTTP/1.1 server for the examples, so that they run without a network.
//!
//! It is not part of the library and only good enough for a demo: it reads one request per
//! connection, answers it and closes the connection.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

/// What the server receives: the request's path (with its query string) and its headers.
pub struct Request {
    pub path: String,
    pub headers: Vec<(String, String)>,
}

impl Request {
    /// The value of header `name`, if the request has it (names are case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// Serve requests on `127.0.0.1`, on a free port, from background threads that live until the
/// program exits; returns the address. `handler` answers each request with a status code and
/// a body.
pub fn start<H>(handler: H) -> SocketAddr
where
    H: Fn(&Request) -> (u16, Vec<u8>) + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
    let addr = listener.local_addr().expect("local address");
    let handler = Arc::new(handler);
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let handler = Arc::clone(&handler);
            thread::spawn(move || serve(stream, &*handler));
        }
    });
    addr
}

fn serve(stream: TcpStream, handler: &dyn Fn(&Request) -> (u16, Vec<u8>)) {
    let mut reader = BufReader::new(&stream);
    // The request line: "GET /path?query HTTP/1.1".
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
    // The headers, up to the empty line that ends them.
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) if line.trim().is_empty() => break,
            Ok(_) => {
                if let Some((name, value)) = line.split_once(':') {
                    headers.push((name.trim().to_owned(), value.trim().to_owned()));
                }
            }
        }
    }
    let (status, body) = handler(&Request { path, headers });
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        if status < 400 { "OK" } else { "Error" },
        body.len()
    );
    let mut stream = &stream;
    // A client that went away is not an error worth reporting in a demo.
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
}
