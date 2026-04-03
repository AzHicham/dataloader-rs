//! Torch-rs specific collation implementations.
//!
//! Keep third-party integrations isolated in feature-gated files so the core
//! collator stays dependency-free.
//!
//! Feature: `torch-rs`

use crate::collator::default_collator::DefaultCollate;
use crate::error::Result;

impl DefaultCollate for tch::Tensor {
    type Batch = tch::Tensor;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        tch::Tensor::f_stack(&items, 0).map_err(|e| e.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, DefaultCollator};
    use tch::{Device, Kind, Tensor};

    fn zeros(shape: &[i64], kind: Kind) -> Tensor {
        Tensor::f_zeros(shape, (kind, Device::Cpu)).unwrap()
    }

    #[test]
    fn torch_collate_trait_impl_compiles() {
        fn uses_torch_collate<C: Collator<tch::Tensor>>(c: &C) {
            let _ = c;
        }
        uses_torch_collate(&DefaultCollator);
    }

    #[test]
    fn torch_collate_ok_cpu() {
        let c = DefaultCollator;

        let a = zeros(&[2, 3], Kind::Float);
        let b = zeros(&[2, 3], Kind::Float);

        let out = c.collate(vec![a, b]).unwrap();
        assert_eq!(out.size(), vec![2, 2, 3]);
    }

    #[test]
    fn torch_collate_ok_single_item_adds_batch_dim() {
        let c = DefaultCollator;

        let a = zeros(&[2, 3], Kind::Float);
        let out = c.collate(vec![a]).unwrap();
        assert_eq!(out.size(), vec![1, 2, 3]);
    }

    #[test]
    fn torch_collate_ok_scalar_tensors() {
        let c = DefaultCollator;

        let a = zeros(&[], Kind::Float);
        let b = zeros(&[], Kind::Float);
        let out = c.collate(vec![a, b]).unwrap();
        assert_eq!(out.size(), vec![2]);
    }

    #[test]
    fn torch_collate_faulty_shape_cpu() {
        let c = DefaultCollator;

        let a = zeros(&[2, 3], Kind::Float);
        let b = zeros(&[2, 4], Kind::Float);

        let out = c.collate(vec![a, b]);
        assert!(out.is_err(), "expected f_stack to fail on mismatched shape");
    }

    #[test]
    fn torch_collate_faulty_rank_cpu() {
        let c = DefaultCollator;

        let a = zeros(&[2, 3], Kind::Float);
        let b = zeros(&[2, 3, 1], Kind::Float);

        let out = c.collate(vec![a, b]);
        assert!(out.is_err(), "expected f_stack to fail on mismatched rank");
    }

    #[test]
    fn torch_collate_faulty_empty_batch() {
        let c = DefaultCollator;
        let out = c.collate(Vec::<Tensor>::new());
        assert!(out.is_err(), "expected f_stack to fail on empty batch");
    }
}
