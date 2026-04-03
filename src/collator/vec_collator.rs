use crate::{collator::Collator, error::Result};

/// The simplest collator: the batch is just a `Vec` of items.
///
/// This is the default collator used by [`DataLoader::builder`].
///
/// [`DataLoader::builder`]: crate::loader::DataLoader::builder
#[derive(Clone, Copy, Debug, Default)]
pub struct VecCollator;

impl<Item: Send + 'static> Collator<Item> for VecCollator {
    type Batch = Vec<Item>;

    #[inline]
    fn collate(&self, items: Vec<Item>) -> Result<Self::Batch> {
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, VecCollator};
    use crate::error::Result;

    #[test]
    fn vec_collator_preserves_order() {
        let c = VecCollator;
        let items = vec![3u32, 1, 4, 1, 5, 9];
        let batch = c.collate(items.clone()).unwrap();
        assert_eq!(batch, items);
    }

    #[test]
    fn vec_collator_empty_input() {
        let c = VecCollator;
        let batch = c.collate(Vec::<u64>::new()).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn vec_collator_single_item() {
        let c = VecCollator;
        let batch = c.collate(vec![42u8]).unwrap();
        assert_eq!(batch, vec![42u8]);
    }

    #[test]
    fn custom_collator_sum() {
        struct SumCollator;
        impl Collator<i64> for SumCollator {
            type Batch = i64;
            fn collate(&self, items: Vec<i64>) -> Result<i64> {
                Ok(items.iter().sum())
            }
        }

        let c = SumCollator;
        assert_eq!(c.collate(vec![1, 2, 3, 4]).unwrap(), 10);
    }

    #[test]
    fn custom_collator_can_fail() {
        struct FailingCollator;
        impl Collator<u8> for FailingCollator {
            type Batch = u8;
            fn collate(&self, _items: Vec<u8>) -> Result<u8> {
                Err("intentional failure".into())
            }
        }

        let c = FailingCollator;
        assert!(c.collate(vec![1, 2]).is_err());
    }
}
