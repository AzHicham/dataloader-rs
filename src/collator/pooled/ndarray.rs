//! `DefaultCollator` writing `ndarray::Array` samples into an array buffer.
//!
//! Feature: `ndarray`

use ndarray::{Array, Axis, Dimension, RemoveAxis};

use crate::{
    collator::{DefaultCollator, pooled::CollateInto},
    error::Result,
};

/// Assigns sample `i` to row `i` of the buffer (axis 0), in place. Every
/// sample must have the shape of a buffer row.
impl<A, D> CollateInto<Array<A, D>> for DefaultCollator
where
    A: Clone + Send + Sync + 'static,
    D: Dimension + Send + 'static,
    D::Larger: RemoveAxis + Send + 'static,
{
    type Buffer = Array<A, D::Larger>;
    type Output = ();

    fn collate_into(&self, items: Vec<Array<A, D>>, buffer: &mut Self::Buffer) -> Result<()> {
        let Some((&rows, row_shape)) = buffer.shape().split_first() else {
            return Err("buffer array has no batch axis".into());
        };
        if items.len() > rows {
            return Err(format!("{} samples for a buffer of {rows} rows", items.len()).into());
        }
        if let Some(item) = items.iter().find(|item| item.shape() != row_shape) {
            return Err(
                format!("sample of shape {:?}, expected {row_shape:?}", item.shape()).into(),
            );
        }
        for (item, mut row) in items.iter().zip(buffer.axis_iter_mut(Axis(0))) {
            row.assign(item);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, array};

    use crate::collator::{DefaultCollator, pooled::CollateInto};

    #[test]
    fn assigns_rows_in_place() {
        let mut buffer = Array3::<f32>::zeros((3, 2, 2));
        let ptr = buffer.as_ptr();
        DefaultCollator
            .collate_into(vec![array![[1.0, 2.0], [3.0, 4.0]]], &mut buffer)
            .unwrap();
        assert_eq!(buffer.as_ptr(), ptr);
        assert_eq!(buffer[[0, 1, 1]], 4.0);
        assert_eq!(buffer[[1, 0, 0]], 0.0);
    }

    #[test]
    fn rejects_bad_samples() {
        let mut buffer = Array2::<i32>::zeros((2, 3));
        let wrong = array![1, 2];
        assert!(
            DefaultCollator
                .collate_into(vec![wrong], &mut buffer)
                .is_err()
        );
        let row = || array![1, 2, 3];
        assert!(
            DefaultCollator
                .collate_into(vec![row(), row(), row()], &mut buffer)
                .is_err()
        );
    }
}
