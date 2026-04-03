use crate::collator::default_collator::DefaultCollate;
use crate::error::Result;

impl<A, D> DefaultCollate for ndarray::Array<A, D>
where
    A: Clone + Send + Sync + 'static,
    D: ndarray::Dimension + Send + 'static,
    D::Larger: ndarray::RemoveAxis + Send + 'static,
{
    type Batch = ndarray::Array<A, D::Larger>;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let views: Vec<ndarray::ArrayView<'_, A, D>> = items.iter().map(|a| a.view()).collect();
        ndarray::stack(ndarray::Axis(0), views.as_slice())
            .map_err(|e| format!("failed to stack ndarray batch: {e}").into())
    }
}

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, DefaultCollator};

    #[test]
    fn collates_ndarray_by_stacking_axis_zero() {
        use ndarray::array;

        let c = DefaultCollator;
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let out = c.collate(vec![a, b]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }
}
