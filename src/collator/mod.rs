use crate::error::Result;

mod default_collator;
#[cfg(feature = "torch-rs")]
mod torch;
mod vec_collator;

pub use default_collator::{DefaultCollate, DefaultCollator};
#[cfg(feature = "torch-rs")]
pub use torch::TorchPinnedCollator;
pub use vec_collator::VecCollator;

/// Merges a `Vec<Item>` produced by one batch of `Dataset::get` calls into a
/// single typed batch value.
///
/// Making this an explicit generic parameter on `DataLoader` (rather than a
/// runtime closure) means collation is monomorphized.
pub trait Collator<Item>: Send + Sync + 'static {
    /// The batched representation of multiple `Item`s.
    ///
    /// Must be `Send + 'static` so it can be moved through the prefetch
    /// channel to the consumer thread.
    type Batch: Send + 'static;

    /// Merge `items` into a single batch.
    fn collate(&self, items: Vec<Item>) -> Result<Self::Batch>;
}

#[cfg(test)]
mod tests {
    use super::Collator;
    use crate::error::Result;

    #[test]
    fn custom_collator_implements_trait() {
        struct SumCollator;

        impl Collator<i64> for SumCollator {
            type Batch = i64;

            fn collate(&self, items: Vec<i64>) -> Result<Self::Batch> {
                Ok(items.into_iter().sum())
            }
        }

        let collator = SumCollator;
        let batch = collator.collate(vec![1, 2, 3, 4]).unwrap();
        assert_eq!(batch, 10);
    }

    #[test]
    fn custom_collator_can_return_error() {
        struct FailingCollator;

        impl Collator<u8> for FailingCollator {
            type Batch = u8;

            fn collate(&self, _items: Vec<u8>) -> Result<Self::Batch> {
                Err("intentional collator failure".into())
            }
        }

        let collator = FailingCollator;
        let result = collator.collate(vec![1, 2, 3]);
        assert!(result.is_err());
    }
}
