use crate::{collator::default_collator::DefaultCollate, error::Result};

impl<T, const N: usize> DefaultCollate for [T; N]
where
    T: DefaultCollate,
{
    type Batch = [T::Batch; N];

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let mut columns: [Vec<T>; N] = std::array::from_fn(|_| Vec::with_capacity(items.len()));

        for row in items {
            for (i, value) in row.into_iter().enumerate() {
                columns[i].push(value);
            }
        }

        let mut out: Vec<T::Batch> = Vec::with_capacity(N);
        for col in columns {
            out.push(T::collate_items(col)?);
        }

        out.try_into()
            .map_err(|_| "internal error while building array batch".into())
    }
}
