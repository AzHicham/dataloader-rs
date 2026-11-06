use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use crate::error::{Error, Result};

/// Creates the batch buffers of a [`BufferPool`].
///
/// The allocator decides the buffer's type and size, and where its memory
/// lives: [`HeapAllocator`] for a plain `Vec<T>`, `TorchPinnedAllocator`
/// (feature `torch-rs`) for page-locked `tch::Tensor`s, or any closure
/// `Fn() -> Result<B>` for memory from another library (e.g. ONNX Runtime's or
/// cudarc's pinned host allocator).
///
/// `allocate` is called at most [`BufferPool::capacity`] times while the pool
/// lives (more only after [`PooledBatch::detach`]); every buffer it returns
/// must have room for a full batch.
pub trait BufferAllocator: Send + Sync + 'static {
    /// The buffer one batch is collated into.
    type Buffer: Send + 'static;

    /// Allocate one buffer, large enough for a full batch.
    fn allocate(&self) -> Result<Self::Buffer>;
}

impl<F, B> BufferAllocator for F
where
    F: Fn() -> Result<B> + Send + Sync + 'static,
    B: Send + 'static,
{
    type Buffer = B;

    fn allocate(&self) -> Result<B> {
        self()
    }
}

/// Allocates `Vec<T>` buffers of a fixed length on the heap, filled with
/// `T::default()`.
#[derive(Clone, Copy, Debug)]
pub struct HeapAllocator<T> {
    len: usize,
    marker: std::marker::PhantomData<fn() -> T>,
}

impl<T> HeapAllocator<T> {
    /// Buffers of `len` elements. Fails if `len` elements of `T` exceed
    /// `isize::MAX` bytes, the most a `Vec` can hold.
    pub fn new(len: usize) -> Result<Self> {
        len.checked_mul(size_of::<T>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| -> Error { format!("buffer of {len} elements is too large").into() })?;
        Ok(Self {
            len,
            marker: std::marker::PhantomData,
        })
    }

    /// Buffers of `shape.iter().product()` elements, e.g. `[64, 3, 224, 224]`
    /// for a batch of 64 RGB images in NCHW. Fails on overflow.
    pub fn for_shape(shape: &[usize]) -> Result<Self> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| -> Error { format!("buffer shape {shape:?} overflows").into() })?;
        Self::new(len)
    }

    /// Elements per buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether buffers have no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> BufferAllocator for HeapAllocator<T>
where
    T: Clone + Default + Send + 'static,
{
    type Buffer = Vec<T>;

    fn allocate(&self) -> Result<Vec<T>> {
        // `try_reserve_exact` turns an allocation failure into an error
        // instead of aborting the process.
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(self.len)?;
        buffer.resize(self.len, T::default());
        Ok(buffer)
    }
}

struct PoolState<B> {
    /// Buffers ready for a batch; the last one returned is reused first, while
    /// its memory is still in cache.
    free: Vec<B>,
    /// Buffers alive: free, collating, in flight, or held by the consumer.
    allocated: usize,
    capacity: usize,
}

/// A bounded pool of batch buffers, shared by the collator and the batches.
///
/// [`acquire`](Self::acquire) hands out a free buffer, allocates a new one
/// while fewer than `capacity` exist, and otherwise **blocks** until a batch
/// is dropped: memory never grows beyond `capacity` buffers. Buffers are
/// allocated lazily, so a large capacity costs nothing until it is used.
pub struct BufferPool<A: BufferAllocator> {
    allocator: A,
    state: Mutex<PoolState<A::Buffer>>,
    returned: Condvar,
}

impl<A: BufferAllocator> BufferPool<A> {
    /// A pool of at most `capacity` buffers (at least 1).
    pub fn new(allocator: A, capacity: usize) -> Self {
        Self {
            allocator,
            state: Mutex::new(PoolState {
                free: Vec::new(),
                allocated: 0,
                capacity: capacity.max(1),
            }),
            returned: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, PoolState<A::Buffer>> {
        // The state is consistent between statements, so a panic elsewhere
        // while it was locked leaves nothing to repair.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The allocator buffers come from.
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    /// The most buffers that can exist at once.
    pub fn capacity(&self) -> usize {
        self.lock().capacity
    }

    /// Change the capacity (at least 1). Buffers above a reduced capacity are
    /// freed as they come back.
    pub fn set_capacity(&self, capacity: usize) {
        self.lock().capacity = capacity.max(1);
        self.returned.notify_all();
    }

    /// Buffers alive: free, being collated, in flight, or held as batches.
    pub fn allocated(&self) -> usize {
        self.lock().allocated
    }

    /// Buffers ready to be reused.
    pub fn available(&self) -> usize {
        self.lock().free.len()
    }

    /// Take a buffer: a free one, else a new one if the pool is below its
    /// capacity, else wait until one comes back. The buffer returns to the
    /// pool when the [`Lease`] is dropped.
    pub(crate) fn acquire(self: &Arc<Self>) -> Result<Lease<A::Buffer>> {
        let mut state = self.lock();
        loop {
            if let Some(buffer) = state.free.pop() {
                return Ok(self.lease(buffer));
            }
            if state.allocated < state.capacity {
                state.allocated += 1;
                drop(state);
                // Allocate unlocked: it may be slow (page-locking memory).
                // The reservation is undone if allocation fails or panics.
                let reservation = Reservation(self);
                let buffer = self.allocator.allocate()?;
                std::mem::forget(reservation);
                return Ok(self.lease(buffer));
            }
            state = self
                .returned
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn lease(self: &Arc<Self>, buffer: A::Buffer) -> Lease<A::Buffer> {
        Lease {
            buffer: Some(buffer),
            pool: Arc::clone(self) as Arc<dyn Recycle<A::Buffer>>,
        }
    }

    /// Forget one allocated buffer, making room for a new one.
    fn release_slot(&self) {
        self.lock().allocated -= 1;
        self.returned.notify_one();
    }
}

/// Undoes a capacity reservation if the allocation it was made for fails.
struct Reservation<'a, A: BufferAllocator>(&'a BufferPool<A>);

impl<A: BufferAllocator> Drop for Reservation<'_, A> {
    fn drop(&mut self) {
        self.0.release_slot();
    }
}

/// The pool, as seen by a buffer that goes back to it.
trait Recycle<B>: Send + Sync {
    fn recycle(&self, buffer: B);
    fn forget(&self);
}

impl<A: BufferAllocator> Recycle<A::Buffer> for BufferPool<A> {
    fn recycle(&self, buffer: A::Buffer) {
        let mut state = self.lock();
        if state.allocated > state.capacity {
            // The capacity was reduced: free this one.
            state.allocated -= 1;
            drop(state);
            drop(buffer);
        } else {
            state.free.push(buffer);
            drop(state);
        }
        self.returned.notify_one();
    }

    fn forget(&self) {
        self.release_slot();
    }
}

/// A buffer taken from a pool, returned to it on drop.
pub(crate) struct Lease<B: Send + 'static> {
    /// `None` only once detached or dropped.
    buffer: Option<B>,
    pool: Arc<dyn Recycle<B>>,
}

impl<B: Send + 'static> Lease<B> {
    fn take(&mut self) -> B {
        self.buffer.take().expect("lease holds a buffer")
    }
}

impl<B: Send + 'static> Deref for Lease<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.buffer.as_ref().expect("lease holds a buffer")
    }
}

impl<B: Send + 'static> DerefMut for Lease<B> {
    fn deref_mut(&mut self) -> &mut B {
        self.buffer.as_mut().expect("lease holds a buffer")
    }
}

impl<B: Send + 'static> Drop for Lease<B> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.recycle(buffer);
        }
    }
}

/// A batch collated into a pooled buffer; the buffer goes back to its pool
/// when the batch is dropped.
///
/// Dereferences to the buffer. The buffer is sized for a full batch: a last,
/// partial batch fills only the first [`num_samples`](Self::num_samples)
/// samples of it, and the rest holds data of an earlier batch.
pub struct PooledBatch<B: Send + 'static, O = ()> {
    lease: Lease<B>,
    output: O,
    num_samples: usize,
}

impl<B: Send + 'static, O> PooledBatch<B, O> {
    pub(crate) fn new(lease: Lease<B>, output: O, num_samples: usize) -> Self {
        Self {
            lease,
            output,
            num_samples,
        }
    }

    /// Samples collated into this batch: the batch size, or fewer for the
    /// last batch of an epoch.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// The buffer holding the batch.
    pub fn buffer(&self) -> &B {
        &self.lease
    }

    /// The buffer holding the batch, mutably.
    pub fn buffer_mut(&mut self) -> &mut B {
        &mut self.lease
    }

    /// What [`CollateInto::collate_into`](super::CollateInto::collate_into)
    /// returned along with the buffer (per-batch metadata).
    pub fn output(&self) -> &O {
        &self.output
    }

    /// Keep the buffer for good: it leaves the pool, which may allocate a new
    /// one in its place.
    pub fn detach(mut self) -> (B, O) {
        let buffer = self.lease.take();
        self.lease.pool.forget();
        (buffer, self.output)
    }
}

impl<B: Send + 'static, O> Deref for PooledBatch<B, O> {
    type Target = B;

    fn deref(&self) -> &B {
        &self.lease
    }
}

impl<B: Send + 'static, O> DerefMut for PooledBatch<B, O> {
    fn deref_mut(&mut self) -> &mut B {
        &mut self.lease
    }
}

impl<B: Send + 'static, O> std::fmt::Debug for PooledBatch<B, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBatch")
            .field("num_samples", &self.num_samples)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use super::{BufferAllocator, BufferPool, HeapAllocator, PooledBatch};
    use crate::error::Result;

    fn counting_pool(
        capacity: usize,
    ) -> (
        Arc<BufferPool<impl BufferAllocator<Buffer = Vec<u8>>>>,
        Arc<AtomicUsize>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let alloc = move || -> Result<Vec<u8>> {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0u8; 16])
        };
        (Arc::new(BufferPool::new(alloc, capacity)), calls)
    }

    #[test]
    fn reuses_the_returned_buffer() {
        let (pool, calls) = counting_pool(4);
        let first = pool.acquire().unwrap();
        let ptr = first.as_ptr();
        drop(first);
        let second = pool.acquire().unwrap();
        assert_eq!(second.as_ptr(), ptr);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(pool.allocated(), 1);
    }

    #[test]
    fn allocates_lazily_up_to_capacity() {
        let (pool, calls) = counting_pool(3);
        let leases: Vec<_> = (0..3).map(|_| pool.acquire().unwrap()).collect();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(pool.available(), 0);
        drop(leases);
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.allocated(), 3);
    }

    #[test]
    fn blocks_at_capacity_until_a_buffer_returns() {
        let (pool, calls) = counting_pool(1);
        let held = pool.acquire().unwrap();
        let waiter = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || pool.acquire().map(|lease| lease.as_ptr() as usize))
        };
        thread::sleep(Duration::from_millis(50));
        assert!(!waiter.is_finished(), "acquire must wait at capacity");
        let ptr = held.as_ptr() as usize;
        drop(held);
        assert_eq!(waiter.join().unwrap().unwrap(), ptr);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_allocation_frees_its_slot() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let alloc = move || -> Result<Vec<u8>> {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                Err("no memory".into())
            } else {
                Ok(vec![1])
            }
        };
        let pool = Arc::new(BufferPool::new(alloc, 1));
        assert!(pool.acquire().is_err());
        assert_eq!(pool.allocated(), 0);
        assert!(pool.acquire().is_ok());
    }

    #[test]
    fn panicking_allocation_frees_its_slot() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let alloc = move || -> Result<Vec<u8>> {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("allocator panicked");
            }
            Ok(vec![1])
        };
        let pool = Arc::new(BufferPool::new(alloc, 1));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.acquire()));
        assert!(caught.is_err());
        assert_eq!(pool.allocated(), 0);
        assert!(pool.acquire().is_ok());
    }

    #[test]
    fn detach_leaves_the_pool() {
        let (pool, calls) = counting_pool(1);
        let batch = PooledBatch::new(pool.acquire().unwrap(), 7u32, 1);
        let (buffer, output) = batch.detach();
        assert_eq!((buffer.len(), output), (16, 7));
        assert_eq!(pool.allocated(), 0);
        let _again = pool.acquire().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn reduced_capacity_frees_returned_buffers() {
        let (pool, _) = counting_pool(3);
        let leases: Vec<_> = (0..3).map(|_| pool.acquire().unwrap()).collect();
        pool.set_capacity(1);
        drop(leases);
        assert_eq!(pool.allocated(), 1);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn zero_capacity_is_one() {
        let (pool, _) = counting_pool(0);
        assert_eq!(pool.capacity(), 1);
        assert!(pool.acquire().is_ok());
    }

    #[test]
    fn heap_allocator_sizes() {
        let a = HeapAllocator::<f32>::for_shape(&[2, 3, 4]).unwrap();
        assert_eq!(a.len(), 24);
        let buffer = a.allocate().unwrap();
        assert_eq!(buffer, vec![0.0; 24]);
    }

    #[test]
    fn heap_allocator_rejects_hostile_sizes() {
        assert!(HeapAllocator::<f32>::for_shape(&[usize::MAX, 2]).is_err());
        assert!(HeapAllocator::<f32>::for_shape(&[1 << 32, 1 << 32]).is_err());
        assert!(HeapAllocator::<f32>::new(usize::MAX / 2).is_err());
        assert!(HeapAllocator::<u8>::new(isize::MAX as usize + 1).is_err());
        // Within `isize::MAX` bytes but beyond any real memory: an error from
        // `allocate`, not an abort.
        let huge = HeapAllocator::<u8>::new(isize::MAX as usize).unwrap();
        assert!(huge.allocate().is_err());
        // Zero-sized types and empty shapes are fine.
        assert!(HeapAllocator::<()>::new(usize::MAX).is_ok());
        assert_eq!(HeapAllocator::<u8>::for_shape(&[]).unwrap().len(), 1);
        assert!(
            HeapAllocator::<u8>::for_shape(&[0, usize::MAX])
                .unwrap()
                .is_empty()
        );
    }
}
