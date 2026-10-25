use crate::{collator::default_collator::DefaultCollate, error::Result};

impl<T, const N: usize> DefaultCollate for [T; N]
where
    T: DefaultCollate,
{
    type Batch = [T::Batch; N];

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let mut columns: [Vec<T>; N] = std::array::from_fn(|_| Vec::with_capacity(items.len()));

        for row in items {
            for (column, value) in columns.iter_mut().zip(row) {
                column.push(value);
            }
        }

        columns
            .into_iter()
            .map(T::collate_items)
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| "internal error while building array batch".into())
    }
}

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, DefaultCollator};

    #[test]
    fn collates_arrays_by_position() {
        let out = DefaultCollator
            .collate(vec![[1u8, 2, 3], [4, 5, 6]])
            .unwrap();
        assert_eq!(out, [vec![1, 4], vec![2, 5], vec![3, 6]]);
    }
}
