//! `HttpDataset`: one HTTP GET per item, with the URLs made from a template and columns of
//! values, a header sent with every request, and retries of transient failures. It talks to
//! a small local server started by the example, so it runs offline.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example http_dataset --features http
//! ```

#[path = "local_server.rs"]
mod local_server;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use bytes::Bytes;
use dataloader_rs::dataset::{Column, HttpDataset, Retry};
use dataloader_rs::{DataLoader, error::Result};

const TOKEN: &str = "Bearer example-token";

fn main() -> Result<()> {
    // The server answers `/shards/<shard>/items/<id>.txt` with a line of text. It refuses
    // requests without the right `Authorization` header, and item 5 fails twice with
    // "503 Service Unavailable" before it succeeds, as an overloaded server might.
    let item_5_requests = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&item_5_requests);
    let addr = local_server::start(move |request| {
        if request.header("Authorization") != Some(TOKEN) {
            return (401, b"missing token".to_vec());
        }
        let parts: Vec<&str> = request.path.split('/').collect();
        match parts.as_slice() {
            ["", "shards", shard, "items", file] => {
                let id = file.trim_end_matches(".txt");
                if id == "5" && counter.fetch_add(1, Ordering::SeqCst) < 2 {
                    return (503, b"busy, try again".to_vec());
                }
                (200, format!("item {id} of shard {shard}").into_bytes())
            }
            _ => (404, b"not found".to_vec()),
        }
    });

    // Eight items: item `i` is fetched from the URL made with the `i`-th value of each column.
    // Strings are percent-encoded in a URL ("shard a" becomes "shard%20a"); numbers are written
    // in decimal. The template is parsed once; each URL is made only when its item is fetched.
    let shards = Column::from(
        vec!["shard a"; 4]
            .into_iter()
            .chain(vec!["shard b"; 4])
            .collect::<Vec<_>>(),
    );
    let ids = Column::from((0..8).collect::<Vec<u64>>());
    let dataset = HttpDataset::builder()
        .header("Authorization", TOKEN)
        .timeout(Duration::from_secs(5)) // for each attempt, the body included
        .retry(Retry {
            max_retries: 3, // up to 3 more attempts after a 503, a timeout or a lost connection
            initial_backoff: Duration::from_millis(10), // waiting 10 ms, then 20 ms, then 40 ms
            max_backoff: Duration::from_millis(100),
        })
        .url_template(
            &format!("http://{addr}/shards/{{shard}}/items/{{id}}.txt"),
            [("shard", shards), ("id", ids)],
        )?;

    // The loader fetches 2 batches at a time, with all 4 items of each in flight. Items arrive
    // as `Bytes`; the transform turns them into text on the loader's thread pool.
    let mut loader = DataLoader::builder_async(dataset)
        .batch_size(4)
        .num_workers(2)
        .item_concurrency(4)
        .transform(|_index: usize, body: Bytes| -> Result<String> {
            Ok(String::from_utf8(body.to_vec())?)
        })
        .build();
    for batch in &mut loader {
        println!("{:?}", batch?);
    }
    println!(
        "item 5 took {} requests: two 503 answers, then a success",
        item_5_requests.load(Ordering::SeqCst)
    );

    // A failure that is not transient (here a 404) fails at once, and only its own batch:
    // the iterator returns an error for that batch and goes on with the next one.
    let urls = [
        format!("http://{addr}/no/such/item"),
        format!("http://{addr}/shards/c/items/0.txt"),
    ];
    let dataset = HttpDataset::builder()
        .header("Authorization", TOKEN)
        .urls(&urls)?;
    let mut loader = DataLoader::builder_async(dataset).batch_size(1).build();
    for (url, batch) in urls.iter().zip(&mut loader) {
        match batch {
            Ok(items) => println!("{url}: {} bytes", items[0].len()),
            Err(error) => println!("{url}: error: {error}"),
        }
    }
    Ok(())
}
