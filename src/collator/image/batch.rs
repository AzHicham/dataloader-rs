//! [`ImageBatch`]: the images of a batch in one contiguous buffer, and its [`BatchLayout`].

use std::fmt;
use std::mem;
use std::sync::Arc;

use super::buffer_pool::BufferPool;
use crate::image::Sample;

/// The order of an [`ImageBatch`]'s values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchLayout {
    /// `[batch, channels, height, width]`: PyTorch's.
    #[default]
    Nchw,
    /// `[batch, height, width, channels]`: numpy's and TensorFlow's.
    Nhwc,
}

impl BatchLayout {
    /// `"NCHW"` or `"NHWC"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Nchw => "NCHW",
            Self::Nhwc => "NHWC",
        }
    }
}

/// The images of a batch, of one shape, in one contiguous row-major buffer: what
/// [`ImageCollator`](super::ImageCollator) makes. With
/// [`reuse_buffers`](super::ImageCollator::reuse_buffers), dropping it gives its buffer back to
/// the collator.
pub struct ImageBatch<T: Sample> {
    shape: [usize; 4],
    layout: BatchLayout,
    data: Vec<T>,
    /// Where the buffer goes back when the batch is dropped.
    pool: Option<Arc<BufferPool>>,
}

impl<T: Sample> ImageBatch<T> {
    pub(crate) fn new(
        shape: [usize; 4],
        layout: BatchLayout,
        data: Vec<T>,
        pool: Option<Arc<BufferPool>>,
    ) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            shape,
            layout,
            data,
            pool,
        }
    }

    /// `[batch, channels, height, width]` or `[batch, height, width, channels]`, per
    /// [`layout`](Self::layout).
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// The order of the values.
    pub fn layout(&self) -> BatchLayout {
        self.layout
    }

    /// The values, row-major in [`shape`](Self::shape).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The values, mutable.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The values; the buffer is then the caller's, not reused.
    pub fn into_vec(mut self) -> Vec<T> {
        mem::take(&mut self.data)
    }
}

impl<T: Sample> Drop for ImageBatch<T> {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.give(mem::take(&mut self.data));
        }
    }
}

/// The shape, type and layout; not the values.
impl<T: Sample> fmt::Debug for ImageBatch<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [n, a, b, c] = self.shape;
        let layout = self.layout.name();
        write!(f, "ImageBatch<{}, {layout}>({n}x{a}x{b}x{c})", T::NAME)
    }
}
