use std::collections::HashMap;

use dataloader_rs::{
    Collator, DataLoader, Dataset,
    collator::{DefaultCollate, DefaultCollator},
    error::Result,
    sampler::RandomSampler,
};

#[derive(Debug, Clone, PartialEq)]
struct Record {
    id: u32,
    values: Vec<u8>,
}

#[derive(Debug, PartialEq)]
struct RecordBatch {
    id: Vec<u32>,
    values: Vec<Vec<u8>>,
}

impl DefaultCollate for Record {
    type Batch = RecordBatch;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let ids: Vec<u32> = items.iter().map(|r| r.id).collect();
        let vals: Vec<Vec<u8>> = items.into_iter().map(|r| r.values).collect();
        let c = DefaultCollator;
        Ok(RecordBatch {
            id: c.collate(ids)?,
            values: c.collate(vals)?,
        })
    }
}

struct RecordDs(usize);

impl Dataset for RecordDs {
    type Item = Record;

    fn get(&self, index: usize) -> Result<Self::Item> {
        Ok(Record {
            id: index as u32,
            values: vec![index as u8, (index as u8).wrapping_add(1)],
        })
    }

    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn dataloader_collates_custom_struct_end_to_end() {
    let mut loader = DataLoader::builder(RecordDs(10))
        .batch_size(3)
        .drop_last(true)
        .sampler(RandomSampler::new(7))
        .collator(DefaultCollator)
        .num_workers(2)
        .prefetch_depth(2)
        .build();

    let batches: Vec<RecordBatch> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 3); // floor(10/3)
    for b in &batches {
        assert_eq!(b.id.len(), 3);
        assert_eq!(b.values.len(), 2); // transposed dimension of `values`
        assert_eq!(b.values[0].len(), 3);
        assert_eq!(b.values[1].len(), 3);
    }
}

struct MapDs(usize);

impl Dataset for MapDs {
    type Item = HashMap<&'static str, u32>;

    fn get(&self, index: usize) -> Result<Self::Item> {
        Ok(HashMap::from([
            ("x", index as u32),
            ("y", (index as u32) * 10),
        ]))
    }

    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn dataloader_collates_maps_end_to_end() {
    let mut loader = DataLoader::builder(MapDs(6))
        .batch_size(2)
        .collator(DefaultCollator)
        .prefetch_depth(1)
        .build();

    let batches: Vec<HashMap<&'static str, Vec<u32>>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 3);
    for b in &batches {
        assert_eq!(b["x"].len(), 2);
        assert_eq!(b["y"].len(), 2);
    }
}
