//! [`ImageCollator`]: images of one shape into one [`ImageBatch`].

use std::sync::Arc;

use super::buffer_pool::BufferPool;
use super::{BatchLayout, ImageBatch};
use crate::collator::Collator;
use crate::error::Result;
use crate::image::{Image, Layout, Sample, invalid, sample_count, with_capacity};

/// Collates images of one shape into an [`ImageBatch`]: one contiguous buffer, `NCHW` (the
/// default) or `NHWC`, of the images' sample type (`u8` or `f32`). Items in either layout are
/// copied once, transposed if needed; images of different shapes are an error for the batch
/// (pad, crop or resize them first).
///
/// [`reuse_buffers(n)`](Self::reuse_buffers) keeps the buffers of up to `n` dropped batches and
/// writes new batches into them: fresh memory costs a page fault per 4 KiB on first write,
/// which for a 64 × 3 × 224 × 224 `f32` batch (38.5 MB) is three quarters of the collation
/// (see `benches/bench_image.rs`). When none is free a buffer is allocated: the pool never
/// waits, so holding batches costs memory, not a deadlock.
///
/// ```rust
/// use dataloader_rs::Collator;
/// use dataloader_rs::collator::{BatchLayout, ImageCollator};
/// use dataloader_rs::image::Image;
///
/// let pixel = |v: u8| Image::<u8>::new(1, 1, 3, vec![v, v + 1, v + 2]).unwrap();
/// let batch = ImageCollator::default().collate(vec![pixel(0), pixel(10)]).unwrap();
/// assert_eq!(batch.shape(), [2, 3, 1, 1]);
/// assert_eq!(batch.data(), [0, 1, 2, 10, 11, 12]);
/// let nhwc = ImageCollator::new(BatchLayout::Nhwc);
/// assert_eq!(nhwc.collate(vec![pixel(0)]).unwrap().shape(), [1, 1, 1, 3]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ImageCollator {
    layout: BatchLayout,
    pool: Option<Arc<BufferPool>>,
}

impl ImageCollator {
    /// A collator making batches in `layout`.
    pub fn new(layout: BatchLayout) -> Self {
        Self { layout, pool: None }
    }

    /// Keep the buffers of up to `max` dropped batches for the next ones; `0` keeps none (the
    /// default).
    pub fn reuse_buffers(mut self, max: usize) -> Self {
        self.pool = (max > 0).then(|| BufferPool::new(max));
        self
    }

    /// The layout of its batches.
    pub fn layout(&self) -> BatchLayout {
        self.layout
    }

    /// Collate borrowed images, as [`collate`](Collator::collate) does owned ones.
    pub fn collate_refs<T: Sample, L: Layout>(
        &self,
        images: &[&Image<T, L>],
    ) -> Result<ImageBatch<T>> {
        let Some(first) = images.first() else {
            return Err(invalid("an empty batch has no shape".into()));
        };
        let (w, h, c) = (first.width(), first.height(), first.channels());
        if let Some((i, other)) = images
            .iter()
            .enumerate()
            .find(|(_, i)| i.shape() != first.shape())
        {
            return Err(invalid(format!(
                "image {i} of the batch is {:?}, image 0 is {:?}: all images of a batch must \
                 have the same shape (pad, crop or resize them)",
                other.shape(),
                first.shape()
            )));
        }
        let per_image = sample_count(w, h, c)?;
        let len = per_image
            .checked_mul(images.len())
            .ok_or_else(|| invalid(format!("{} images of {per_image} values", images.len())))?;
        let mut data = match &self.pool {
            Some(pool) => pool.take(len)?,
            None => with_capacity(len)?,
        };
        let planar = self.layout == BatchLayout::Nchw;
        for image in images {
            write(&mut data, image, planar);
        }
        let n = images.len();
        let shape = match self.layout {
            BatchLayout::Nchw => [n, c, h, w],
            BatchLayout::Nhwc => [n, h, w, c],
        };
        Ok(ImageBatch::new(shape, self.layout, data, self.pool.clone()))
    }
}

impl<T: Sample, L: Layout> Collator<Image<T, L>> for ImageCollator {
    type Batch = ImageBatch<T>;

    fn collate(&self, images: Vec<Image<T, L>>) -> Result<ImageBatch<T>> {
        let refs: Vec<&Image<T, L>> = images.iter().collect();
        self.collate_refs(&refs)
    }
}

/// Append `image` to `out`, channel planes after one another if `planar`, interleaved if not.
fn write<T: Sample, L: Layout>(out: &mut Vec<T>, image: &Image<T, L>, planar: bool) {
    let src = image.data();
    if L::PLANAR == planar {
        out.extend_from_slice(src);
        return;
    }
    let (c, plane) = (image.channels(), image.width() * image.height());
    let start = out.len();
    out.resize(start + src.len(), T::default());
    let dst = &mut out[start..];
    for channel in 0..c {
        match planar {
            // Interleaved to planar: gather channel `channel` of every pixel.
            true => {
                let dst = &mut dst[channel * plane..(channel + 1) * plane];
                for (d, pixel) in dst.iter_mut().zip(src.chunks_exact(c)) {
                    *d = pixel[channel];
                }
            }
            // Planar to interleaved: scatter plane `channel` to every pixel.
            false => {
                let src = &src[channel * plane..(channel + 1) * plane];
                for (pixel, &s) in dst.chunks_exact_mut(c).zip(src) {
                    pixel[channel] = s;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchLayout, ImageCollator};
    use crate::collator::Collator;
    use crate::image::{Chw, Image};

    /// 1 row of 2 RGB pixels, values from `v`.
    fn image(v: u8) -> Image<u8> {
        Image::new(2, 1, 3, (v..v + 6).collect()).unwrap()
    }

    #[test]
    fn hwc_items_in_both_layouts() {
        let nchw = ImageCollator::default()
            .collate(vec![image(0), image(10)])
            .unwrap();
        assert_eq!(nchw.shape(), [2, 3, 1, 2]);
        assert_eq!(nchw.data(), [0, 3, 1, 4, 2, 5, 10, 13, 11, 14, 12, 15]);
        let nhwc = ImageCollator::new(BatchLayout::Nhwc)
            .collate(vec![image(0), image(10)])
            .unwrap();
        assert_eq!(nhwc.shape(), [2, 1, 2, 3]);
        assert_eq!(nhwc.data(), [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn chw_items_in_both_layouts() {
        let planar = Image::<f32, Chw>::new(2, 1, 3, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]).unwrap();
        let nchw = ImageCollator::default()
            .collate(vec![planar.clone()])
            .unwrap();
        assert_eq!(nchw.data(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let nhwc = ImageCollator::new(BatchLayout::Nhwc)
            .collate(vec![planar])
            .unwrap();
        assert_eq!(nhwc.data(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn dropped_batches_give_their_buffer_back() {
        let collator = ImageCollator::default().reuse_buffers(1);
        let first = collator.collate(vec![image(0)]).unwrap();
        let ptr = first.data().as_ptr();
        let second = collator.collate(vec![image(1)]).unwrap();
        assert_ne!(second.data().as_ptr(), ptr, "nothing free: a new buffer");
        drop(first);
        drop(second);
        let third = collator.collate(vec![image(2)]).unwrap();
        assert_eq!(third.data().as_ptr(), ptr, "kept: the first one");
        assert_eq!(third.data(), [2, 5, 3, 6, 4, 7]);
        let owned = third.into_vec();
        let fourth = collator.collate(vec![image(3)]).unwrap();
        assert_ne!(
            fourth.data().as_ptr(),
            owned.as_ptr(),
            "taken out with into_vec"
        );
    }

    #[test]
    fn shapes_must_match() {
        let other = Image::new(1, 2, 3, vec![0u8; 6]).unwrap();
        let err = ImageCollator::default()
            .collate(vec![image(0), image(1), other])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("image 2 of the batch is [2, 1, 3], image 0 is [1, 2, 3]"),
            "{err}"
        );
        assert!(
            ImageCollator::default()
                .collate(Vec::<Image<u8>>::new())
                .is_err()
        );
    }
}
