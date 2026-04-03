use crate::{collator::Collator, error::Result};

mod array;
mod map;
#[cfg(feature = "ndarray")]
mod ndarray;
mod primitive;
mod sequence;
#[cfg(feature = "torch-rs")]
mod torch;
mod tuple;

/// Type-level collation for nested structures.
pub trait DefaultCollate: Sized + Send + 'static {
    type Batch: Send + 'static;
    fn collate_items(items: Vec<Self>) -> Result<Self::Batch>;
}

/// PyTorch-like default collator for common Rust structures.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultCollator;

impl<Item> Collator<Item> for DefaultCollator
where
    Item: DefaultCollate,
{
    type Batch = Item::Batch;

    fn collate(&self, items: Vec<Item>) -> Result<Self::Batch> {
        Item::collate_items(items)
    }
}

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, DefaultCollator};

    #[test]
    fn collates_tuple_pairs() {
        let c = DefaultCollator;
        let out = c.collate(vec![(1u32, 10i32), (2, 20), (3, 30)]).unwrap();
        assert_eq!(out, (vec![1, 2, 3], vec![10, 20, 30]));
    }
}
