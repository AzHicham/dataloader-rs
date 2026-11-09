//! [`BufferPool`]: batch buffers kept for reuse, so that a new batch is written to memory
//! that is already mapped instead of fresh pages.

use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use crate::error::Result;
use crate::image::with_capacity;

/// Up to `max` buffers of dropped batches, of any sample type. Taking from an empty pool
/// allocates; nothing ever waits for a buffer, so a consumer holding many batches costs
/// allocations, never a deadlock.
#[derive(Debug)]
pub(crate) struct BufferPool {
    free: Mutex<Vec<Box<dyn Any + Send>>>,
    max: usize,
}

impl BufferPool {
    pub(crate) fn new(max: usize) -> Arc<Self> {
        Arc::new(Self {
            free: Mutex::new(Vec::with_capacity(max)),
            max,
        })
    }

    /// An empty `Vec<T>` with room for `len` values: a kept one if one is large enough.
    pub(crate) fn take<T: Send + 'static>(&self, len: usize) -> Result<Vec<T>> {
        let kept = {
            let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
            let fits = free.iter().position(|buffer| {
                buffer
                    .downcast_ref::<Vec<T>>()
                    .is_some_and(|b| b.capacity() >= len)
            });
            fits.map(|at| free.swap_remove(at))
        };
        match kept.map(|buffer| buffer.downcast::<Vec<T>>()) {
            Some(Ok(mut buffer)) => {
                buffer.clear();
                Ok(*buffer)
            }
            _ => with_capacity(len),
        }
    }

    /// Keep `buffer` for a next batch, unless the pool is full.
    pub(crate) fn give<T: Send + 'static>(&self, buffer: Vec<T>) {
        if buffer.capacity() == 0 {
            return;
        }
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        if free.len() < self.max {
            free.push(Box::new(buffer));
        }
    }

    /// Buffers kept now.
    #[cfg(test)]
    pub(crate) fn kept(&self) -> usize {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::BufferPool;

    #[test]
    fn keeps_at_most_max_buffers_by_type() {
        let pool = BufferPool::new(2);
        let a: Vec<f32> = pool.take(10).unwrap();
        let ptr = a.as_ptr();
        pool.give(a);
        pool.give(vec![0u8; 4]);
        pool.give(vec![0u8; 4]);
        assert_eq!(pool.kept(), 2);
        // The f32 one comes back, empty; the kept u8 one is too small for 100.
        let b: Vec<f32> = pool.take(10).unwrap();
        assert_eq!((b.as_ptr(), b.len()), (ptr, 0));
        let c: Vec<u8> = pool.take(100).unwrap();
        assert!(c.capacity() >= 100);
        assert_eq!(pool.kept(), 1);
    }
}
