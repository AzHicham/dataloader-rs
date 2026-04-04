#![cfg(feature = "torch-rs")]

use dataloader_rs::{DataLoader, Dataset, collator::DefaultCollator, error::Result};

struct TensorDs(usize);

impl Dataset for TensorDs {
    type Item = tch::Tensor;

    fn get(&self, _index: usize) -> Result<Self::Item> {
        tch::Tensor::f_zeros([2, 3], (tch::Kind::Float, tch::Device::Cpu)).map_err(|e| e.into())
    }

    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn dataloader_collates_tensors_end_to_end() {
    let mut loader = DataLoader::builder(TensorDs(6))
        .batch_size(3)
        .collator(DefaultCollator)
        .num_workers(2)
        .prefetch_depth(2)
        .build();

    let batches: Vec<tch::Tensor> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 2);
    for b in &batches {
        assert_eq!(b.size(), vec![3, 2, 3]);
    }
}

#[test]
fn dataloader_tensor_collate_with_pin_memory_flag() {
    let mut loader = DataLoader::builder(TensorDs(4))
        .batch_size(2)
        .collator(DefaultCollator)
        .pin_memory(true)
        .build();

    let batches: Vec<tch::Tensor> = loader.iter().map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 2);
    for b in &batches {
        assert_eq!(b.size(), vec![2, 2, 3]);
    }
}
