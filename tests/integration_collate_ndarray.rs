#![cfg(feature = "ndarray")]

use dataloader_rs::{collator::DefaultCollator, error::Result, DataLoader, Dataset};

struct NdDs(usize);

impl Dataset for NdDs {
    type Item = ndarray::Array1<i32>;

    fn get(&self, index: usize) -> Result<Self::Item> {
        Ok(ndarray::arr1(&[index as i32, index as i32 + 1, index as i32 + 2]))
    }

    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn dataloader_collates_ndarray_end_to_end() {
    let mut loader = DataLoader::builder(NdDs(8))
        .batch_size(4)
        .collator(DefaultCollator)
        .num_workers(2)
        .prefetch_depth(2)
        .build();

    let batches: Vec<ndarray::Array2<i32>> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 2);
    for b in &batches {
        assert_eq!(b.shape(), &[4, 3]);
    }
}
