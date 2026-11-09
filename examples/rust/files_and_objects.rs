//! `FileDataset` and `ObjectStoreDataset`: items read from local files, and from an object
//! store (S3, GCS, Azure or HTTP in real use; here an in-memory store, so it runs offline).
//!
//! Both return each item as `Bytes`, and both take their locations as a list, a template with
//! columns of values, or a function of the item's index.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example files_and_objects --features object-store
//! ```

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use dataloader_rs::dataset::{Column, FileDataset, ObjectStoreDataset, Retry};
use dataloader_rs::{DataLoader, error::Result};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStoreExt, PutPayload};

/// The loader's transform for both datasets: the bytes of an item, as text.
fn to_text(_index: usize, bytes: Bytes) -> Result<String> {
    Ok(String::from_utf8(bytes.to_vec())?)
}

fn main() -> Result<()> {
    files()?;
    objects()
}

/// Six files in two folders, read through a path template.
fn files() -> Result<()> {
    let root = std::env::temp_dir().join(format!("dataloader-rs-example-{}", std::process::id()));
    for (index, folder) in ["cats", "cats", "cats", "dogs", "dogs", "dogs"]
        .iter()
        .enumerate()
    {
        std::fs::create_dir_all(root.join(folder))?;
        std::fs::write(
            root.join(folder).join(format!("{index}.txt")),
            format!("{folder} #{index}"),
        )?;
    }

    // `{folder}` takes the item's value in the "folder" column, and `{index}` is the item's
    // index: item 4 is `<root>/dogs/4.txt`. Braces in the root path itself are doubled so that
    // they are not read as placeholders. A value cannot contain `/` or be `..`, so it can never
    // point outside the template's directory.
    let root_text = root
        .display()
        .to_string()
        .replace('{', "{{")
        .replace('}', "}}");
    let folders = Column::from(vec!["cats", "cats", "cats", "dogs", "dogs", "dogs"]);
    let dataset = FileDataset::builder()
        // On a network file system, give each read a time limit and retry it if it fails.
        .timeout(Duration::from_secs(10))
        .retry(Retry {
            max_retries: 2,
            ..Retry::default()
        })
        .path_template(
            &format!("{root_text}/{{folder}}/{{index}}.txt"),
            [("folder", folders)],
        )?;

    let mut loader = DataLoader::builder_async(dataset)
        .batch_size(3)
        .item_concurrency(3) // the 3 files of a batch are read at the same time
        .transform(to_text)
        .build();
    for batch in &mut loader {
        println!("files:   {:?}", batch?);
    }
    std::fs::remove_dir_all(&root)?;
    Ok(())
}

/// Objects in an in-memory store: whole objects by key, then byte ranges of one object.
fn objects() -> Result<()> {
    // Fill the store. Its API is async, so a small tokio runtime runs the uploads.
    let store = Arc::new(InMemory::new());
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(async {
            for i in 0..4 {
                let payload = PutPayload::from(format!("object number {i}").into_bytes());
                store
                    .put(&Path::from(format!("tiles/{i}.txt")), payload)
                    .await?;
            }
            let row = PutPayload::from(b"AAAABBBBCCCCDDDD".to_vec());
            store.put(&Path::from("rows/packed.bin"), row).await?;
            Ok::<_, object_store::Error>(())
        })?;

    // In real use, `ObjectStoreDataset::from_url("s3://bucket/prefix", options)` opens the
    // store, with its credentials from the environment. Here we hand over our own store.
    let dataset =
        ObjectStoreDataset::builder(store.clone()).key_template_len(4, "tiles/{index}.txt", [])?;
    let mut loader = DataLoader::builder_async(dataset)
        .batch_size(2)
        .num_workers(2)
        .transform(to_text)
        .build();
    for batch in &mut loader {
        println!("objects: {:?}", batch?);
    }

    // Item `i` can also be a byte range of an object: four 4-byte records packed in one object.
    let ranges = (0..4u64).map(|i| ("rows/packed.bin", i * 4..i * 4 + 4));
    let dataset = ObjectStoreDataset::builder(store).ranges(ranges)?;
    let mut loader = DataLoader::builder_async(dataset)
        .batch_size(4)
        .item_concurrency(4)
        .transform(to_text)
        .build();
    for batch in &mut loader {
        println!("ranges:  {:?}", batch?);
    }
    Ok(())
}
