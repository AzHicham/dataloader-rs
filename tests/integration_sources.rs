//! Feature `sources`: `MemorySource`, `FileSource` and `HttpSource` through `SourceDataset` and a
//! loader, against a local HTTP server (`support/http_server.rs`): retries, timeouts, size caps,
//! hostile responses, connection reuse, file errors; and, with `image`, an `ImageCollator` fed
//! over HTTP with metadata.
#![cfg(feature = "sources")]

#[path = "support/http_server.rs"]
mod http_server;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use dataloader_rs::{
    DataLoader, Dataset,
    sources::{FileSource, HttpSource, MemorySource, Retry, Source, SourceDataset, SourceError},
};
use http_server::{Reply, Server};

/// Fast retries for tests.
const QUICK: Retry = Retry {
    max_retries: 3,
    initial_backoff: Duration::from_millis(1),
    max_backoff: Duration::from_millis(5),
};

fn source_error(error: &dataloader_rs::Error) -> &SourceError {
    error
        .downcast_ref::<SourceError>()
        .unwrap_or_else(|| panic!("not a SourceError: {error}"))
}

/// A server answering `/<n>` with `n` as text.
fn echo_server() -> Server {
    Server::start(|request| Reply::ok(request.target.trim_start_matches('/').as_bytes().to_vec()))
}

#[test]
fn memory_source_with_meta() {
    let items: Vec<Arc<[u8]>> = (0..5u8).map(|i| Arc::from(vec![i; 3])).collect();
    let dataset =
        SourceDataset::with_meta(MemorySource::new(items), vec!["a", "b", "c", "d", "e"]).unwrap();
    assert_eq!(dataset.len(), 5);
    let (bytes, meta) = dataset.get(3).unwrap();
    assert_eq!((&bytes[..], meta), (&[3u8, 3, 3][..], "d"));
    let error = dataset.get(5).unwrap_err();
    assert!(matches!(
        source_error(&error),
        SourceError::OutOfRange { index: 5, len: 5 }
    ));
    assert!(matches!(
        SourceDataset::with_meta(MemorySource::new([&b"x"[..]]), vec![1, 2]),
        Err(SourceError::MetaLength { items: 1, meta: 2 })
    ));
}

#[test]
fn file_source_reads_caps_and_reports() {
    let dir = std::env::temp_dir().join(format!("dataloader_rs_sources_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let small = dir.join("small.bin");
    let big = dir.join("big.bin");
    std::fs::write(&small, b"hello").unwrap();
    std::fs::write(&big, vec![0u8; 100]).unwrap();
    let source =
        FileSource::new([small.clone(), big.clone(), dir.join("missing.bin")]).max_bytes(10);
    assert_eq!(source.fetch(0).unwrap(), b"hello");
    match source.fetch(1).unwrap_err() {
        SourceError::TooLarge {
            index: 1,
            limit: 10,
            size: Some(100),
            location,
        } => assert!(location.ends_with("big.bin")),
        other => panic!("{other}"),
    }
    match source.fetch(2).unwrap_err() {
        SourceError::File {
            index: 2, source, ..
        } => {
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound)
        }
        other => panic!("{other}"),
    }
    // A directory is an error, not a hang or a panic.
    assert!(FileSource::new([&dir]).fetch(0).is_err());
    // A special file of length 0 that never ends is capped while reading.
    #[cfg(target_os = "linux")]
    assert!(matches!(
        FileSource::new(["/dev/zero"]).max_bytes(1 << 12).fetch(0),
        Err(SourceError::TooLarge { size: None, .. })
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn http_urls_through_a_loader_with_meta() {
    let server = echo_server();
    let urls: Vec<String> = (0..37).map(|i| server.url(&format!("/{i}"))).collect();
    let meta: Vec<usize> = (0..37).map(|i| i * 10).collect();
    let source = HttpSource::urls(&urls).unwrap();
    let dataset = SourceDataset::with_meta(source, meta).unwrap();
    for (workers, in_parallel) in [(0, 1), (2, 1), (3, 4)] {
        let mut loader = DataLoader::builder(dataset.clone())
            .batch_size(8)
            .num_workers(workers)
            .items_in_parallel(in_parallel)
            .build();
        let items: Vec<(Vec<u8>, usize)> = loader.iter().flat_map(|batch| batch.unwrap()).collect();
        assert_eq!(items.len(), 37);
        for (i, (bytes, meta)) in items.iter().enumerate() {
            assert_eq!(bytes, i.to_string().as_bytes());
            assert_eq!(*meta, i * 10);
        }
    }
}

#[test]
fn http_template_encodes_values_and_sends_headers() {
    let server = Server::start(|request| {
        let auth = request
            .header("authorization")
            .unwrap_or("none")
            .to_string();
        Reply::ok(format!("{} {auth}", request.target).into_bytes())
    });
    let base = server.url("");
    let source = HttpSource::builder()
        .header("Authorization", "Bearer s3cret")
        .template(
            &format!("{base}/{{slide}}/{{x}}_{{y}}.png?v={{{{1}}}}"),
            [
                [
                    ("slide", "a b".to_string()),
                    ("x", "0".into()),
                    ("y", "1".into()),
                ],
                [
                    ("slide", "../etc".to_string()),
                    ("x", "2".into()),
                    ("y", "?q".into()),
                ],
            ],
        )
        .unwrap();
    assert_eq!(
        source.fetch(0).unwrap(),
        b"/a%20b/0_1.png?v={1} Bearer s3cret"
    );
    assert_eq!(
        source.fetch(1).unwrap(),
        b"/..%2Fetc/2_%3Fq.png?v={1} Bearer s3cret"
    );
    // The header value is a credential: not in Debug.
    assert!(!format!("{source:?}").contains("s3cret"));
}

#[test]
fn http_reuses_connections() {
    let server = echo_server();
    let source = HttpSource::urls((0..50).map(|i| server.url(&format!("/{i}")))).unwrap();
    for i in 0..50 {
        source.fetch(i).unwrap();
    }
    assert_eq!(server.requests(), 50);
    assert_eq!(server.connections(), 1, "keep-alive connection not reused");
}

#[test]
fn http_retries_transient_failures() {
    // 503, 500 then 200: two retries.
    let server = Server::start(|request| match request.number {
        1 => Reply::status(503),
        2 => Reply::status(500),
        _ => Reply::ok(&b"ok"[..]),
    });
    let source = HttpSource::builder()
        .retry(QUICK)
        .urls([server.url("/a")])
        .unwrap();
    assert_eq!(source.fetch(0).unwrap(), b"ok");
    assert_eq!(server.requests(), 3);

    // A connection closed before any response, then success.
    let server = Server::start(|request| match request.number {
        1 => Reply::Raw(Vec::new()),
        _ => Reply::ok(&b"ok"[..]),
    });
    let source = HttpSource::builder()
        .retry(QUICK)
        .urls([server.url("/a")])
        .unwrap();
    assert_eq!(source.fetch(0).unwrap(), b"ok");

    // A truncated body (Content-Length 100, 3 bytes sent), then success.
    let server = Server::start(|request| match request.number {
        1 => Reply::Raw(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nabc".to_vec()),
        _ => Reply::ok(&b"ok"[..]),
    });
    let source = HttpSource::builder()
        .retry(QUICK)
        .urls([server.url("/a")])
        .unwrap();
    assert_eq!(source.fetch(0).unwrap(), b"ok");
    assert_eq!(server.requests(), 2);
}

#[test]
fn http_gives_up_after_max_retries() {
    let server = Server::start(|_| Reply::status(502));
    let source = HttpSource::builder()
        .retry(QUICK)
        .urls([server.url("/a?token=secret")])
        .unwrap();
    let error = source.fetch(0).unwrap_err();
    match &error {
        SourceError::Status {
            index: 0,
            status: 502,
            attempts: 4,
            url,
        } => assert!(url.ends_with("/a"), "{url}"),
        other => panic!("{other}"),
    }
    assert!(!error.to_string().contains("secret"));
    assert_eq!(server.requests(), 4);
}

#[test]
fn http_does_not_retry_permanent_failures() {
    for status in [400, 403, 404, 410] {
        let server = Server::start(move |_| Reply::status(status));
        let source = HttpSource::builder()
            .retry(QUICK)
            .urls([server.url("/a")])
            .unwrap();
        assert!(matches!(
            source.fetch(0),
            Err(SourceError::Status { attempts: 1, .. })
        ));
        assert_eq!(server.requests(), 1);
    }
    // Garbage instead of HTTP.
    let server = Server::start(|_| Reply::Raw(b"SSH-2.0-OpenSSH\r\n\r\n".to_vec()));
    let source = HttpSource::builder()
        .retry(QUICK)
        .urls([server.url("/a")])
        .unwrap();
    assert!(matches!(source.fetch(0), Err(SourceError::Request { .. })));
}

#[test]
fn http_honours_retry_after_up_to_max_backoff() {
    let server = Server::start(|request| match request.number {
        1 => Reply::Body {
            status: 429,
            headers: vec![("Retry-After".into(), "3600".into())],
            body: Arc::from(&b""[..]),
        },
        _ => Reply::ok(&b"ok"[..]),
    });
    let retry = Retry {
        max_retries: 1,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(200),
    };
    let source = HttpSource::builder()
        .retry(retry)
        .urls([server.url("/a")])
        .unwrap();
    let start = Instant::now();
    assert_eq!(source.fetch(0).unwrap(), b"ok");
    let waited = start.elapsed();
    assert!(waited >= Duration::from_millis(200), "{waited:?}");
    assert!(waited < Duration::from_secs(5), "{waited:?}");
}

#[test]
fn http_times_out() {
    let server = Server::start(|_| {
        thread::sleep(Duration::from_secs(3));
        Reply::ok(&b"late"[..])
    });
    let source = HttpSource::builder()
        .timeout(Duration::from_millis(200))
        .retry(Retry {
            max_retries: 1,
            ..QUICK
        })
        .urls([server.url("/slow")])
        .unwrap();
    let start = Instant::now();
    let error = source.fetch(0).unwrap_err();
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "{:?}",
        start.elapsed()
    );
    match &error {
        SourceError::Request {
            timeout: true,
            attempts: 2,
            ..
        } => {}
        other => panic!("{other}"),
    }
    assert!(error.to_string().contains("timed out"));
}

#[test]
fn http_caps_response_size() {
    // Announced too large: fails before reading, no retry.
    let server = Server::start(|_| Reply::ok(vec![0u8; 2048]));
    let source = HttpSource::builder()
        .retry(QUICK)
        .max_bytes(1024)
        .urls([server.url("/big")])
        .unwrap();
    assert!(matches!(
        source.fetch(0),
        Err(SourceError::TooLarge {
            limit: 1024,
            size: Some(2048),
            ..
        })
    ));
    assert_eq!(server.requests(), 1);

    // No Content-Length and a body that never ends: stopped at the cap.
    let server = Server::start(|_| Reply::Endless);
    let source = HttpSource::builder()
        .max_bytes(1 << 20)
        .urls([server.url("/endless")])
        .unwrap();
    assert!(matches!(
        source.fetch(0),
        Err(SourceError::TooLarge { size: None, .. })
    ));

    // A lying, huge Content-Length is rejected without reserving it.
    let server = Server::start(|_| {
        Reply::Raw(b"HTTP/1.1 200 OK\r\nContent-Length: 18446744073709551615\r\n\r\nabc".to_vec())
    });
    let source = HttpSource::urls([server.url("/a")]).unwrap();
    assert!(source.fetch(0).is_err());
}

#[test]
fn http_redirects_are_followed_and_capped() {
    let server = Server::start(|request| {
        let hop: usize = request.target.trim_start_matches('/').parse().unwrap_or(0);
        if hop < 3 {
            Reply::Body {
                status: 302,
                headers: vec![("Location".into(), format!("/{}", hop + 1))],
                body: Arc::from(&b""[..]),
            }
        } else {
            Reply::ok(&b"done"[..])
        }
    });
    let source = HttpSource::urls([server.url("/0")]).unwrap();
    assert_eq!(source.fetch(0).unwrap(), b"done");
    let source = HttpSource::builder()
        .max_redirects(2)
        .retry(QUICK)
        .urls([server.url("/0")])
        .unwrap();
    assert!(source.fetch(0).is_err());
}

#[test]
fn http_invalid_configuration() {
    let invalid = |result: Result<HttpSource, SourceError>| match result {
        Err(SourceError::Invalid(message)) => message,
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("accepted"),
    };
    assert!(invalid(HttpSource::urls(["http://h/a", "nope"])).contains("URL 1"));
    invalid(HttpSource::urls(["file:///etc/passwd"]));
    assert!(invalid(HttpSource::urls(["ftp://h/a"])).contains("only http"));
    #[cfg(not(feature = "sources-tls"))]
    assert!(invalid(HttpSource::urls(["https://h/a?key=secret"])).contains("sources-tls"));
    assert!(!invalid(HttpSource::urls(["ftp://h/a?key=secret"])).contains("secret"));
    assert!(
        invalid(
            HttpSource::builder()
                .header("Bad Name", "v")
                .urls(["http://h/"])
        )
        .contains("header name")
    );
    let message = invalid(
        HttpSource::builder()
            .header("X-Key", "s3cret\n")
            .urls(["http://h/"]),
    );
    assert!(!message.contains("s3cret"), "{message}");
    assert!(invalid(HttpSource::builder().max_bytes(0).urls(["http://h/"])).contains("max_bytes"));
    assert!(
        invalid(
            HttpSource::builder()
                .timeout(Duration::ZERO)
                .urls(["http://h/"])
        )
        .contains("timeouts")
    );
    assert!(
        invalid(
            HttpSource::builder()
                .retry(Retry {
                    max_retries: 1000,
                    ..Retry::default()
                })
                .urls(["http://h/"])
        )
        .contains("max_retries")
    );
    assert!(
        invalid(HttpSource::builder().template("http://h/{a}", [[("b", 1)]])).contains("\"b\"")
    );
}

#[test]
fn http_failed_item_fails_its_batch_only() {
    let server = Server::start(|request| {
        if request.target == "/5" {
            Reply::status(404)
        } else {
            Reply::ok(request.target.clone().into_bytes())
        }
    });
    let source = HttpSource::builder()
        .retry(QUICK)
        .urls((0..16).map(|i| server.url(&format!("/{i}"))))
        .unwrap();
    let mut loader = DataLoader::builder(SourceDataset::new(source))
        .batch_size(4)
        .num_workers(2)
        .items_in_parallel(2)
        .build();
    let results: Vec<_> = loader.iter().collect();
    assert_eq!(results.len(), 4);
    let error = results[1].as_ref().unwrap_err();
    assert!(matches!(
        source_error(error),
        SourceError::Status {
            index: 5,
            status: 404,
            ..
        }
    ));
    for batch in [&results[0], &results[2], &results[3]] {
        assert_eq!(batch.as_ref().unwrap().len(), 4);
    }
}

#[test]
fn http_concurrent_requests_from_workers() {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let server = {
        let (in_flight, peak) = (in_flight.clone(), peak.clone());
        Server::start(move |request| {
            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(20));
            in_flight.fetch_sub(1, Ordering::SeqCst);
            Reply::ok(request.target.clone().into_bytes())
        })
    };
    let source = HttpSource::urls((0..32).map(|i| server.url(&format!("/{i}")))).unwrap();
    let mut loader = DataLoader::builder(SourceDataset::new(source))
        .batch_size(8)
        .num_workers(2)
        .items_in_parallel(4)
        .build();
    let total: usize = loader.iter().map(|batch| batch.unwrap().len()).sum();
    assert_eq!(total, 32);
    assert!(peak.load(Ordering::SeqCst) > 1, "requests never overlapped");
}

#[cfg(feature = "image")]
#[test]
fn image_collator_over_http_with_meta() {
    use dataloader_rs::{
        collator::HeapAllocator,
        image::{Channels, ImageCollator, ImageTransform, Layout},
    };

    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image");
    let png: Arc<[u8]> = std::fs::read(fixtures.join("rgb.png")).unwrap().into();
    let expected = std::fs::read(fixtures.join("rgb.u8")).unwrap();
    let server = Server::start(move |_| Reply::ok(png.clone()));
    let source = HttpSource::urls((0..10).map(|i| server.url(&format!("/{i}.png")))).unwrap();
    let labels: Vec<String> = (0..10).map(|i| format!("label-{i}")).collect();
    let dataset = SourceDataset::with_meta(source, labels).unwrap();

    let collator = ImageCollator::<u8>::new(
        ImageTransform::new(Channels::Rgb)
            .input_size(64, 48)
            .layout(Layout::Nhwc),
    )
    .unwrap();
    let shape = collator.batch_shape(4);
    let alloc = HeapAllocator::<u8>::for_shape(&shape).unwrap();
    let mut loader = DataLoader::builder(dataset)
        .batch_size(4)
        .num_workers(2)
        .items_in_parallel(2)
        .collate_into(collator, alloc)
        .build();
    let mut seen = Vec::new();
    for batch in &mut loader {
        let batch = batch.unwrap();
        let row = expected.len();
        for (i, info) in batch.output().iter().enumerate() {
            assert_eq!((info.width, info.height), (64, 48));
            assert_eq!(&batch[i * row..(i + 1) * row], &expected[..]);
            seen.push(info.meta.clone());
        }
    }
    let labels: Vec<String> = (0..10).map(|i| format!("label-{i}")).collect();
    assert_eq!(seen, labels);
}
