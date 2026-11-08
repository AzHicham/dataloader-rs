//! Per-item CPU work the loader applies to a dataset's items: [`Transform`].

use crate::error::Result;

/// A synchronous, CPU-side step the loader applies to every item, such as decoding bytes into
/// an image. Set with [`DataLoaderBuilder::transform`](crate::DataLoaderBuilder::transform).
///
/// A dataset only fetches; the transform runs on the loader's threads, with the index of the
/// item: right after `get` on the worker for a [`Dataset`](crate::Dataset).
///
/// Closures `Fn(usize, In) -> Result<Out>` are transforms; [`then`](Self::then) chains two. A
/// closure passed where a transform is expected needs its types written out, as Rust does not
/// infer them through this trait:
///
/// ```rust
/// use dataloader_rs::{Transform, error::Result};
///
/// let double = |_: usize, x: u64| -> Result<u64> { Ok(x * 2) };
/// let label = |index: usize, x: u64| -> Result<String> { Ok(format!("{index}:{x}")) };
/// assert_eq!(double.then(label).apply(7, 21).unwrap(), "7:42");
/// ```
pub trait Transform<In>: Send + Sync + 'static {
    /// The transformed item.
    type Output: Send + 'static;

    /// Transform `item`, the item at `index`. Called concurrently from several threads.
    fn apply(&self, index: usize, item: In) -> Result<Self::Output>;

    /// Apply `self`, then `next` to its output.
    fn then<T>(self, next: T) -> Then<Self, T>
    where
        Self: Sized,
        T: Transform<Self::Output>,
    {
        Then {
            first: self,
            second: next,
        }
    }

    /// Transform the items of a batch, `items[i]` being the item at `indices[i]`; both have
    /// the same length. Default: [`apply`](Self::apply) to each item in order, up to the first
    /// error. [`Identity`] returns the items as they are, without a pass over them.
    fn apply_batch(&self, indices: &[usize], items: Vec<In>) -> Result<Vec<Self::Output>> {
        items
            .into_iter()
            .zip(indices)
            .map(|(item, &index)| self.apply(index, item))
            .collect()
    }
}

impl<In, Out, F> Transform<In> for F
where
    F: Fn(usize, In) -> Result<Out> + Send + Sync + 'static,
    Out: Send + 'static,
{
    type Output = Out;

    fn apply(&self, index: usize, item: In) -> Result<Out> {
        self(index, item)
    }
}

/// The loader's default transform: items are left as they are.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl<In: Send + 'static> Transform<In> for Identity {
    type Output = In;

    fn apply_batch(&self, _indices: &[usize], items: Vec<In>) -> Result<Vec<In>> {
        Ok(items)
    }

    fn apply(&self, _index: usize, item: In) -> Result<In> {
        Ok(item)
    }
}

/// Two transforms applied one after the other; see [`Transform::then`].
#[derive(Clone, Debug)]
pub struct Then<A, B> {
    first: A,
    second: B,
}

impl<In, A, B> Transform<In> for Then<A, B>
where
    A: Transform<In>,
    B: Transform<A::Output>,
{
    type Output = B::Output;

    fn apply(&self, index: usize, item: In) -> Result<Self::Output> {
        self.second.apply(index, self.first.apply(index, item)?)
    }
}

/// Apply `transform` to the items a `get_batch` returned for `indices`, in order.
pub(crate) fn apply_all<In, T: Transform<In>>(
    transform: &T,
    indices: &[usize],
    items: Vec<In>,
) -> Result<Vec<T::Output>> {
    // A short or long batch would silently pair items with the wrong indices.
    if items.len() != indices.len() {
        return Err(format!(
            "get_batch returned {} items for {} indices",
            items.len(),
            indices.len()
        )
        .into());
    }
    transform.apply_batch(indices, items)
}

#[cfg(test)]
mod tests {
    use super::{Identity, Transform, apply_all};
    use crate::error::Result;

    struct AddOne;

    impl Transform<u32> for AddOne {
        type Output = u32;

        fn apply(&self, _index: usize, item: u32) -> Result<u32> {
            item.checked_add(1).ok_or_else(|| "overflow".into())
        }
    }

    #[test]
    fn struct_and_closure_chain() {
        let t =
            AddOne.then(|index: usize, x: u32| -> Result<String> { Ok(format!("{index}<{x}>")) });
        assert_eq!(t.apply(3, 1).unwrap(), "3<2>");
        assert_eq!(Transform::<u32>::apply(&Identity, 0, 7).unwrap(), 7);
    }

    #[test]
    fn first_error_stops_the_chain() {
        let t = AddOne.then(|_: usize, _: u32| -> Result<u32> { panic!("not reached") });
        assert_eq!(t.apply(0, u32::MAX).unwrap_err().to_string(), "overflow");
    }

    #[test]
    fn apply_all_pairs_items_with_indices_and_checks_the_count() {
        let tag = |index: usize, x: u32| -> Result<(usize, u32)> { Ok((index, x)) };
        assert_eq!(
            apply_all(&tag, &[5, 2], vec![10, 20]).unwrap(),
            vec![(5, 10), (2, 20)]
        );
        let err = apply_all(&tag, &[5, 2], vec![10]).unwrap_err();
        assert_eq!(err.to_string(), "get_batch returned 1 items for 2 indices");
    }

    #[test]
    fn identity_hands_the_batch_back_as_is() {
        let items = vec![10u32, 20];
        let buffer = items.as_ptr();
        let out = apply_all(&Identity, &[5, 2], items).unwrap();
        assert_eq!(out, vec![10, 20]);
        assert_eq!(out.as_ptr(), buffer);
        assert!(apply_all(&Identity, &[5, 2], vec![10u32]).is_err());
    }
}
