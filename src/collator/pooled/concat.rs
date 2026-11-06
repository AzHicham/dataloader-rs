use std::marker::PhantomData;

use crate::{collator::pooled::CollateInto, error::Result};

/// Concatenates samples of `item_len` elements each into one flat buffer:
/// sample `i` lands at `i * item_len`. For samples that are already laid out
/// as their batch row, e.g. a `Vec<f32>` CHW image into an NCHW batch.
///
/// The buffer is any `AsMut<[T]>` (default `Vec<T>`, from
/// [`HeapAllocator`](super::HeapAllocator)); a sample of another length, or
/// more samples than the buffer holds, is an error for the batch.
pub struct ConcatCollator<T, B = Vec<T>> {
    item_len: usize,
    marker: PhantomData<fn() -> (T, B)>,
}

impl<T> ConcatCollator<T> {
    /// Samples of `item_len` elements, into a `Vec<T>`.
    pub fn new(item_len: usize) -> Self {
        Self::for_buffer(item_len)
    }
}

impl<T, B> ConcatCollator<T, B> {
    /// Samples of `item_len` elements, into buffers of type `B`.
    pub fn for_buffer(item_len: usize) -> Self {
        Self {
            item_len,
            marker: PhantomData,
        }
    }

    /// Elements per sample.
    pub fn item_len(&self) -> usize {
        self.item_len
    }
}

impl<T, B, I> CollateInto<I> for ConcatCollator<T, B>
where
    T: Copy + Send + 'static,
    B: AsMut<[T]> + Send + 'static,
    I: AsRef<[T]>,
{
    type Buffer = B;
    type Output = ();

    fn collate_into(&self, items: Vec<I>, buffer: &mut B) -> Result<()> {
        let buffer = buffer.as_mut();
        let needed = items
            .len()
            .checked_mul(self.item_len)
            .filter(|&n| n <= buffer.len())
            .ok_or_else(|| {
                format!(
                    "{} samples of {} elements do not fit a buffer of {}",
                    items.len(),
                    self.item_len,
                    buffer.len()
                )
            })?;
        // `item_len` is non-zero here unless `needed` is 0; chunks_exact(0)
        // would panic.
        if needed == 0 {
            return match items
                .iter()
                .find(|item| item.as_ref().len() != self.item_len)
            {
                Some(item) => Err(length_error(item.as_ref().len(), self.item_len)),
                None => Ok(()),
            };
        }
        for (item, row) in items
            .iter()
            .zip(buffer[..needed].chunks_exact_mut(self.item_len))
        {
            let item = item.as_ref();
            if item.len() != self.item_len {
                return Err(length_error(item.len(), self.item_len));
            }
            row.copy_from_slice(item);
        }
        Ok(())
    }
}

fn length_error(got: usize, expected: usize) -> crate::error::Error {
    format!("sample of {got} elements, expected {expected}").into()
}

#[cfg(test)]
mod tests {
    use super::ConcatCollator;
    use crate::collator::pooled::CollateInto;

    #[test]
    fn concatenates_in_order() {
        let mut buffer = vec![0u8; 6];
        ConcatCollator::new(3)
            .collate_into(vec![vec![1, 2, 3], vec![4, 5, 6]], &mut buffer)
            .unwrap();
        assert_eq!(buffer, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn partial_batch_leaves_the_tail() {
        let mut buffer = vec![9u8; 6];
        ConcatCollator::new(3)
            .collate_into(vec![[1, 2, 3]], &mut buffer)
            .unwrap();
        assert_eq!(buffer, [1, 2, 3, 9, 9, 9]);
    }

    #[test]
    fn rejects_wrong_lengths_and_overflow() {
        let c = ConcatCollator::<u8>::new(3);
        let mut buffer = vec![0u8; 6];
        assert!(c.collate_into(vec![vec![1, 2]], &mut buffer).is_err());
        assert!(c.collate_into(vec![vec![1, 2, 3, 4]], &mut buffer).is_err());
        assert!(c.collate_into(vec![[0u8; 3]; 3], &mut buffer).is_err());

        let huge = ConcatCollator::<u8>::new(usize::MAX);
        assert!(huge.collate_into(vec![[0u8; 1]; 2], &mut buffer).is_err());

        let empty = ConcatCollator::<u8>::new(0);
        assert!(empty.collate_into(vec![Vec::new(); 4], &mut buffer).is_ok());
        assert!(empty.collate_into(vec![vec![1]], &mut buffer).is_err());
    }

    #[test]
    fn writes_into_any_slice_buffer() {
        let mut buffer = [0f32; 4];
        ConcatCollator::<f32, [f32; 4]>::for_buffer(2)
            .collate_into(vec![[1.0, 2.0], [3.0, 4.0]], &mut buffer)
            .unwrap();
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0]);
    }
}
