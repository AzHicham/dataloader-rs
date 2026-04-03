use crate::{collator::default_collator::DefaultCollate, error::Result};

impl<T> DefaultCollate for Vec<T>
where
    T: DefaultCollate,
{
    type Batch = Vec<T::Batch>;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let expected_len = items[0].len();
        if items.iter().any(|v| v.len() != expected_len) {
            return Err("sequence items must all have the same length".into());
        }

        let mut columns: Vec<Vec<T>> = (0..expected_len)
            .map(|_| Vec::with_capacity(items.len()))
            .collect();

        for row in items {
            for (i, value) in row.into_iter().enumerate() {
                columns[i].push(value);
            }
        }

        columns
            .into_iter()
            .map(T::collate_items)
            .collect::<Result<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, DefaultCollator};

    #[test]
    fn collates_nested_vecs_by_transpose() {
        let c = DefaultCollator;
        let out = c.collate(vec![vec![1u8, 2u8], vec![3u8, 4u8]]).unwrap();
        assert_eq!(out, vec![vec![1u8, 3u8], vec![2u8, 4u8]]);
    }

    #[test]
    fn nested_vec_mismatch_len_fails() {
        let c = DefaultCollator;
        let err = c.collate(vec![vec![1u8, 2u8], vec![3u8]]).unwrap_err();
        assert!(err.to_string().contains("same length"));
    }
}
