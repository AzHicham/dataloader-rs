//! Collation into reusable, optionally pinned, batch buffers.
//!
//! See the "Reusable batch buffers" section of the collator guide.

use std::sync::Arc;

use crate::{collator::Collator, error::Result};

mod concat;
#[cfg(feature = "ndarray")]
mod ndarray;
mod pool;
#[cfg(feature = "torch-rs")]
mod torch;

pub use concat::ConcatCollator;
pub use pool::{BufferAllocator, BufferPool, HeapAllocator, PooledBatch};
#[cfg(feature = "torch-rs")]
pub use torch::TorchPinnedAllocator;

/// Writes the samples of one batch into a buffer the caller provides, instead
/// of allocating the batch.
///
/// Used through [`PooledCollator`] (or `DataLoaderBuilder::collate_into`),
/// which takes the buffer from a [`BufferPool`] and hands the batch to the
/// consumer as a [`PooledBatch`].
///
/// The buffer comes back with the content of an earlier batch: write every
/// sample of `items`, and do not rely on what is there. It is sized for a
/// full batch; `items` may be shorter (the last batch of an epoch).
pub trait CollateInto<Item>: Send + Sync + 'static {
    /// The buffer a batch is written into.
    type Buffer: Send + 'static;

    /// Per-batch values returned along with the buffer (e.g. timings, labels);
    /// `()` if none.
    type Output: Send + 'static;

    /// Write `items` into `buffer`.
    fn collate_into(&self, items: Vec<Item>, buffer: &mut Self::Buffer) -> Result<Self::Output>;
}

/// A [`Collator`] that collates each batch into a buffer of a bounded
/// [`BufferPool`], with a [`CollateInto`].
///
/// The batch is a [`PooledBatch`]; dropping it returns its buffer to the pool,
/// so the loader allocates at most `capacity` buffers in total, and a buffer
/// is reused by many batches. When all of them are taken, the next `collate`
/// waits for the consumer to drop a batch.
pub struct PooledCollator<C, A: BufferAllocator> {
    inner: C,
    pool: Arc<BufferPool<A>>,
}

impl<C, A: BufferAllocator> PooledCollator<C, A> {
    /// Collate with `inner` into at most `capacity` buffers from `allocator`.
    ///
    /// `DataLoaderBuilder::collate_into` sizes the pool for the loader's
    /// settings; with this constructor, `capacity` must be at least the
    /// batches that can exist at once: `num_workers + prefetch_depth` (`1`
    /// with `num_workers = 0`) plus the batches the consumer keeps while it
    /// asks for the next one.
    pub fn new(inner: C, allocator: A, capacity: usize) -> Self {
        Self {
            inner,
            pool: Arc::new(BufferPool::new(allocator, capacity)),
        }
    }

    /// The wrapped collator.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// The buffer pool: capacity, buffers allocated and available.
    pub fn pool(&self) -> &BufferPool<A> {
        &self.pool
    }

    /// A handle that resizes the pool once the loader's settings are known.
    pub(crate) fn sizer(&self) -> Arc<dyn PoolSizer> {
        Arc::clone(&self.pool) as Arc<dyn PoolSizer>
    }
}

/// Sets a pool's capacity, whatever its buffer type.
pub(crate) trait PoolSizer: Send + Sync {
    fn set_capacity(&self, capacity: usize);
}

impl<A: BufferAllocator> PoolSizer for BufferPool<A> {
    fn set_capacity(&self, capacity: usize) {
        BufferPool::set_capacity(self, capacity);
    }
}

impl<Item, C, A> Collator<Item> for PooledCollator<C, A>
where
    C: CollateInto<Item, Buffer = A::Buffer>,
    A: BufferAllocator,
{
    type Batch = PooledBatch<A::Buffer, C::Output>;

    fn collate(&self, items: Vec<Item>) -> Result<Self::Batch> {
        let num_samples = items.len();
        let mut lease = self.pool.acquire()?;
        // On an error or a panic the lease is dropped and the buffer returns
        // to the pool.
        let output = self.inner.collate_into(items, &mut lease)?;
        Ok(PooledBatch::new(lease, output, num_samples))
    }
}

#[cfg(test)]
mod tests {
    use super::{CollateInto, ConcatCollator, HeapAllocator, PooledCollator};
    use crate::{collator::Collator, error::Result};

    #[test]
    fn collates_into_a_reused_buffer() {
        let collator = PooledCollator::new(
            ConcatCollator::new(2),
            HeapAllocator::<u32>::new(6).unwrap(),
            2,
        );
        let first = collator.collate(vec![[1, 2], [3, 4], [5, 6]]).unwrap();
        assert_eq!(first.num_samples(), 3);
        assert_eq!(&first[..], &[1, 2, 3, 4, 5, 6]);
        let ptr = first.as_ptr();
        drop(first);

        let last = collator.collate(vec![[7, 8]]).unwrap();
        assert_eq!(last.as_ptr(), ptr);
        assert_eq!(last.num_samples(), 1);
        assert_eq!(&last[..2], &[7, 8]);
        assert_eq!(collator.pool().allocated(), 1);
    }

    #[test]
    fn error_returns_the_buffer() {
        struct Failing;
        impl CollateInto<u8> for Failing {
            type Buffer = Vec<u8>;
            type Output = ();
            fn collate_into(&self, _: Vec<u8>, _: &mut Vec<u8>) -> Result<()> {
                Err("collate failed".into())
            }
        }

        let collator = PooledCollator::new(Failing, HeapAllocator::<u8>::new(4).unwrap(), 1);
        assert!(collator.collate(vec![1]).is_err());
        assert_eq!(collator.pool().available(), 1);
        // The only buffer is free again, so this does not block.
        assert!(collator.collate(vec![1]).is_err());
    }

    #[test]
    fn output_travels_with_the_batch() {
        struct WithSum;
        impl CollateInto<u64> for WithSum {
            type Buffer = Vec<u64>;
            type Output = u64;
            fn collate_into(&self, items: Vec<u64>, buffer: &mut Vec<u64>) -> Result<u64> {
                buffer[..items.len()].copy_from_slice(&items);
                Ok(items.iter().sum())
            }
        }

        let collator = PooledCollator::new(WithSum, HeapAllocator::<u64>::new(3).unwrap(), 1);
        let batch = collator.collate(vec![1, 2, 3]).unwrap();
        assert_eq!(*batch.output(), 6);
    }
}
