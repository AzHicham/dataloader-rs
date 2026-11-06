//! [`ImageCollator`]: decode, transform and write each sample into its row of the batch buffer.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::mem;
use std::sync::{Arc, Mutex, PoisonError};

use fast_image_resize::Resizer;
use half::f16;

use crate::collator::CollateInto;
use crate::error::Result;

use super::decode::decode;
use super::resize::resize;
use super::transform::{Op, Plan};
use super::{Channels, ImageError, ImageResult, ImageTransform, Layout};

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
}

/// A value type of the batch buffer: `u8`, `f32` or `half::f16`.
pub trait Element: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// Whether the type holds the result of `to_float` / `normalize`.
    const FLOAT: bool;

    /// The element closest to `x` (round to nearest even for `f16`, as `Tensor.half()`).
    fn from_f32(x: f32) -> Self;
}

impl Element for u8 {
    const FLOAT: bool = false;

    fn from_f32(x: f32) -> Self {
        // Only called with the values 0..=255 of a pipeline without value operations.
        x as u8
    }
}

impl Element for f32 {
    const FLOAT: bool = true;

    fn from_f32(x: f32) -> Self {
        x
    }
}

impl Element for f16 {
    const FLOAT: bool = true;

    fn from_f32(x: f32) -> Self {
        f16::from_f32(x)
    }
}

/// A sample [`ImageCollator`] accepts: encoded image bytes, and metadata passed through to the
/// batch's [`SampleInfo`].
///
/// Implemented for `Vec<u8>`, `Box<[u8]>`, `Arc<[u8]>`, `&'static [u8]` and `Cow<'static, [u8]>`
/// (no metadata), and for `(bytes, meta)` pairs whose bytes are `AsRef<[u8]>` (metadata `meta`:
/// a label, a path, coordinates).
pub trait ImageItem {
    /// Metadata of the sample, returned with the batch.
    type Meta: Send + 'static;

    /// The encoded image (PNG, JPEG or WebP).
    fn bytes(&self) -> &[u8];

    /// The metadata, once the image is written.
    fn into_meta(self) -> Self::Meta;
}

macro_rules! bytes_item {
    ($($t:ty),*) => {$(
        impl ImageItem for $t {
            type Meta = ();

            fn bytes(&self) -> &[u8] {
                self
            }

            fn into_meta(self) {}
        }
    )*};
}

bytes_item!(
    Vec<u8>,
    Box<[u8]>,
    Arc<[u8]>,
    &'static [u8],
    Cow<'static, [u8]>
);

impl<B: AsRef<[u8]>, M: Send + 'static> ImageItem for (B, M) {
    type Meta = M;

    fn bytes(&self) -> &[u8] {
        self.0.as_ref()
    }

    fn into_meta(self) -> M {
        self.1
    }
}

/// What [`ImageCollator`] reports for each sample of a batch, in order: the decoded size before
/// any operation, and the sample's metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleInfo<M = ()> {
    /// Width of the decoded image.
    pub width: u32,
    /// Height of the decoded image.
    pub height: u32,
    /// The sample's metadata ([`ImageItem::Meta`]).
    pub meta: M,
}

/// Working memory of one sample in flight: reused, one per concurrent batch.
#[derive(Default)]
struct Scratch {
    raw: Vec<u8>,
    image: Vec<u8>,
    spare: Vec<u8>,
    resizer: Resizer,
}

/// Decodes encoded images and writes each one, transformed by an [`ImageTransform`], straight
/// into its row of a batch buffer of `T` (`u8`, `f32` or `f16`): sample `i` fills
/// `i * sample_len .. (i + 1) * sample_len`. No tensor is allocated per sample.
///
/// A [`CollateInto`] for [`ImageItem`]s, used with `DataLoaderBuilder::collate_into` and a
/// [`HeapAllocator`](crate::collator::HeapAllocator) of `T` (or any allocator whose buffer is
/// `AsMut<[T]>`) sized with [`batch_shape`](Self::batch_shape). The batch's output is one
/// [`SampleInfo`] per sample.
///
/// An image that cannot be decoded, is above the limits or does not fit an operation is an error
/// for its batch, which names the sample.
pub struct ImageCollator<T, B = Vec<T>> {
    transform: ImageTransform,
    plan: Plan,
    /// `256` output values per channel: the value operations, precomputed for every 8-bit input.
    table: Vec<[T; 256]>,
    /// Working memory, reused across batches; one entry per batch collated at once.
    scratch: Mutex<Vec<Scratch>>,
    marker: PhantomData<fn() -> B>,
}

impl<T: Element> ImageCollator<T> {
    /// Collate with `transform` into `Vec<T>` buffers; fails if the pipeline is invalid or
    /// produces floats and `T` is `u8`.
    pub fn new(transform: ImageTransform) -> ImageResult<Self> {
        Self::for_buffer(transform)
    }
}

impl<T: Element, B> ImageCollator<T, B> {
    /// Collate with `transform` into buffers of type `B`.
    pub fn for_buffer(transform: ImageTransform) -> ImageResult<Self> {
        let plan = transform.plan()?;
        if !T::FLOAT && !plan.values.is_empty() {
            return Err(ImageError::InvalidPipeline(
                "to_float and normalize produce floats: use an f32 or f16 buffer, not u8".into(),
            ));
        }
        let table = (0..plan.channels.count())
            .map(|c| std::array::from_fn(|v| T::from_f32(plan.value(c, v as u8))))
            .collect();
        Ok(Self {
            transform,
            plan,
            table,
            scratch: Mutex::new(Vec::new()),
            marker: PhantomData,
        })
    }

    /// The pipeline.
    pub fn transform(&self) -> &ImageTransform {
        &self.transform
    }

    /// The shape of one sample: `[C, H, W]` (NCHW) or `[H, W, C]` (NHWC).
    pub fn sample_shape(&self) -> [usize; 3] {
        self.plan.sample_shape()
    }

    /// Values per sample.
    pub fn sample_len(&self) -> usize {
        self.plan.sample_len()
    }

    /// The shape of a batch of `batch_size` samples, for
    /// [`HeapAllocator::for_shape`](crate::collator::HeapAllocator::for_shape).
    pub fn batch_shape(&self, batch_size: usize) -> [usize; 4] {
        let [a, b, c] = self.sample_shape();
        [batch_size, a, b, c]
    }

    /// Decode and transform one image into `row`, which must hold exactly
    /// [`sample_len`](Self::sample_len) values. Returns the decoded size.
    pub fn write_sample(&self, bytes: &[u8], row: &mut [T]) -> ImageResult<(u32, u32)> {
        if row.len() != self.sample_len() {
            return Err(ImageError::Buffer(format!(
                "row of {} values, expected {}",
                row.len(),
                self.sample_len()
            )));
        }
        let mut scratch = self.take_scratch();
        let result = self.write_with(bytes, row, &mut scratch);
        self.put_scratch(scratch);
        result
    }

    fn take_scratch(&self) -> Scratch {
        self.scratch
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_default()
    }

    fn put_scratch(&self, scratch: Scratch) {
        self.scratch
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    }

    fn write_with(&self, bytes: &[u8], row: &mut [T], s: &mut Scratch) -> ImageResult<(u32, u32)> {
        let plan = &self.plan;
        let channels = plan.channels;
        let decoded = decode(bytes, channels, plan.limits, &mut s.raw, &mut s.image)?;
        if let Some((w, h)) = plan.input_size
            && (w as usize, h as usize) != decoded
        {
            return Err(ImageError::Size(format!(
                "image of {}x{}, expected {w}x{h} (input_size)",
                decoded.0, decoded.1
            )));
        }
        let mut size = decoded;
        for op in &plan.geometry {
            size = apply(op, &s.image, size, channels, &mut s.spare, &mut s.resizer)?;
            mem::swap(&mut s.image, &mut s.spare);
        }
        if size != (plan.width, plan.height) || s.image.len() != row.len() {
            // Unreachable for a valid plan; checked rather than trusted.
            return Err(ImageError::Size(format!(
                "transformed image of {}x{}, expected {}x{}",
                size.0, size.1, plan.width, plan.height
            )));
        }
        write_row(&s.image, channels, plan.layout, &self.table, row);
        // Within the limits, so the sides fit `u32`.
        Ok((decoded.0 as u32, decoded.1 as u32))
    }
}

/// Apply one geometric operation to `src` (`size`), into `dst`; returns the new size.
fn apply(
    op: &Op,
    src: &[u8],
    (w, h): (usize, usize),
    channels: Channels,
    dst: &mut Vec<u8>,
    resizer: &mut Resizer,
) -> ImageResult<(usize, usize)> {
    let c = channels.count();
    match *op {
        Op::Resize {
            width,
            height,
            filter,
        } => {
            let to = (width as usize, height as usize);
            if to == (w, h) {
                copy_into(dst, src)?;
            } else {
                resize(resizer, src, (w, h), dst, to, channels, filter)?;
            }
            Ok(to)
        }
        Op::Crop {
            left,
            top,
            width,
            height,
        } => {
            let (left, top, width, height) =
                (left as usize, top as usize, width as usize, height as usize);
            // The sides are below 2^32, so these sums do not overflow `usize` on 64-bit targets;
            // checked anyway for 32-bit ones.
            let fits = left.checked_add(width).is_some_and(|r| r <= w)
                && top.checked_add(height).is_some_and(|b| b <= h);
            if !fits {
                return Err(ImageError::Size(format!(
                    "crop of {width}x{height} at ({left}, {top}) leaves the {w}x{h} image"
                )));
            }
            dst.clear();
            dst.try_reserve_exact(width * height * c)?;
            for row in src.chunks_exact(w * c).skip(top).take(height) {
                dst.extend_from_slice(&row[left * c..(left + width) * c]);
            }
            Ok((width, height))
        }
        Op::Pad {
            width,
            height,
            fill,
        } => {
            let (width, height) = (width as usize, height as usize);
            if w > width || h > height {
                return Err(ImageError::Size(format!(
                    "image of {w}x{h} is larger than the pad of {width}x{height}"
                )));
            }
            dst.clear();
            dst.try_reserve_exact(width * height * c)?;
            dst.resize(width * height * c, fill);
            for (out, row) in dst.chunks_exact_mut(width * c).zip(src.chunks_exact(w * c)) {
                out[..w * c].copy_from_slice(row);
            }
            Ok((width, height))
        }
        Op::ToFloat | Op::Normalize { .. } => Err(ImageError::InvalidPipeline(
            "not a geometric operation".into(),
        )),
    }
}

fn copy_into(dst: &mut Vec<u8>, src: &[u8]) -> ImageResult<()> {
    dst.clear();
    dst.try_reserve_exact(src.len())?;
    dst.extend_from_slice(src);
    Ok(())
}

/// Write interleaved 8-bit pixels into `row` in `layout`, through the per-channel `table`.
/// `row` and `src` have the same length.
fn write_row<T: Copy>(
    src: &[u8],
    channels: Channels,
    layout: Layout,
    table: &[[T; 256]],
    row: &mut [T],
) {
    match (channels, layout) {
        (Channels::Gray, _) => {
            let t = &table[0];
            for (o, &v) in row.iter_mut().zip(src) {
                *o = t[usize::from(v)];
            }
        }
        (Channels::Rgb, Layout::Nhwc) => {
            let (t0, t1, t2) = (&table[0], &table[1], &table[2]);
            for (o, px) in row
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0)
            {
                o[0] = t0[usize::from(px[0])];
                o[1] = t1[usize::from(px[1])];
                o[2] = t2[usize::from(px[2])];
            }
        }
        (Channels::Rgb, Layout::Nchw) => {
            let plane = row.len() / 3;
            let (r, rest) = row.split_at_mut(plane);
            let (g, b) = rest.split_at_mut(plane);
            let (t0, t1, t2) = (&table[0], &table[1], &table[2]);
            for (((r, g), b), px) in r.iter_mut().zip(g).zip(b).zip(src.as_chunks::<3>().0) {
                *r = t0[usize::from(px[0])];
                *g = t1[usize::from(px[1])];
                *b = t2[usize::from(px[2])];
            }
        }
    }
}

impl<T, B, I> CollateInto<I> for ImageCollator<T, B>
where
    T: Element,
    B: AsMut<[T]> + Send + 'static,
    I: ImageItem,
{
    type Buffer = B;
    type Output = Vec<SampleInfo<I::Meta>>;

    fn collate_into(&self, items: Vec<I>, buffer: &mut B) -> Result<Self::Output> {
        let buffer = buffer.as_mut();
        let len = self.sample_len();
        let needed = items
            .len()
            .checked_mul(len)
            .filter(|&n| n <= buffer.len())
            .ok_or_else(|| {
                ImageError::Buffer(format!(
                    "{} samples of {len} values do not fit a buffer of {}",
                    items.len(),
                    buffer.len()
                ))
            })?;
        let mut infos = Vec::with_capacity(items.len());
        let mut scratch = self.take_scratch();
        // `len` is non-zero: the plan's sizes are.
        let result = items
            .into_iter()
            .zip(buffer[..needed].chunks_exact_mut(len))
            .enumerate()
            .try_for_each(|(i, (item, row))| -> ImageResult<()> {
                let (width, height) = self
                    .write_with(item.bytes(), row, &mut scratch)
                    .map_err(|e| e.in_sample(i))?;
                infos.push(SampleInfo {
                    width,
                    height,
                    meta: item.into_meta(),
                });
                Ok(())
            });
        self.put_scratch(scratch);
        result?;
        Ok(infos)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use half::f16;
    use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

    use super::{ImageCollator, SampleInfo};
    use crate::collator::CollateInto;
    use crate::image::{Channels, ImageTransform, Layout, ResizeFilter};

    fn png(pixels: &[u8], w: u32, h: u32, color: ExtendedColorType) -> Vec<u8> {
        let mut out = Vec::new();
        PngEncoder::new(Cursor::new(&mut out))
            .write_image(pixels, w, h, color)
            .unwrap();
        out
    }

    /// A 2x2 RGB image: pixel `p` channel `c` is `10 * p + c`.
    fn rgb2x2() -> Vec<u8> {
        let px: Vec<u8> = (0..4)
            .flat_map(|p| (0..3).map(move |c| 10 * p + c))
            .collect();
        png(&px, 2, 2, ExtendedColorType::Rgb8)
    }

    #[test]
    fn layouts() {
        let t = ImageTransform::new(Channels::Rgb).input_size(2, 2);
        let nhwc = ImageCollator::<u8>::new(t.clone().layout(Layout::Nhwc)).unwrap();
        let mut row = vec![0u8; 12];
        nhwc.write_sample(&rgb2x2(), &mut row).unwrap();
        assert_eq!(row, [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]);
        let nchw = ImageCollator::<u8>::new(t).unwrap();
        nchw.write_sample(&rgb2x2(), &mut row).unwrap();
        assert_eq!(row, [0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32]);
        assert_eq!(nchw.sample_shape(), [3, 2, 2]);
        assert_eq!(nchw.batch_shape(8), [8, 3, 2, 2]);
    }

    #[test]
    fn float_types() {
        let t = ImageTransform::new(Channels::Rgb)
            .input_size(2, 2)
            .to_float()
            .normalize([0.5], [0.5]);
        let mut f = vec![0f32; 12];
        ImageCollator::<f32>::new(t.clone())
            .unwrap()
            .write_sample(&rgb2x2(), &mut f)
            .unwrap();
        assert_eq!(f[0], (0.0f32 / 255.0 - 0.5) / 0.5);
        assert_eq!(f[5], (11.0f32 / 255.0 - 0.5) / 0.5);
        let mut h = vec![f16::ZERO; 12];
        ImageCollator::<f16>::new(t)
            .unwrap()
            .write_sample(&rgb2x2(), &mut h)
            .unwrap();
        assert!(h.iter().zip(&f).all(|(h, f)| *h == f16::from_f32(*f)));
        // Floats without value operations: the 8-bit values.
        let mut raw = vec![0f32; 12];
        ImageCollator::<f32>::new(ImageTransform::new(Channels::Rgb).input_size(2, 2))
            .unwrap()
            .write_sample(&rgb2x2(), &mut raw)
            .unwrap();
        assert_eq!(raw[4], 1.0);
    }

    #[test]
    fn u8_rejects_value_operations() {
        let t = ImageTransform::new(Channels::Rgb)
            .input_size(2, 2)
            .to_float();
        let err = ImageCollator::<u8>::new(t).err().unwrap().to_string();
        assert!(err.contains("f32 or f16"), "{err}");
    }

    #[test]
    fn crop_and_pad() {
        let t = ImageTransform::new(Channels::Rgb)
            .crop(1, 0, 1, 2)
            .pad(2, 3, 99)
            .layout(Layout::Nhwc);
        let c = ImageCollator::<u8>::new(t).unwrap();
        let mut row = vec![0u8; 18];
        c.write_sample(&rgb2x2(), &mut row).unwrap();
        assert_eq!(
            row,
            [
                10, 11, 12, 99, 99, 99, //
                30, 31, 32, 99, 99, 99, //
                99, 99, 99, 99, 99, 99,
            ]
        );
        // Without input_size the crop is checked per image.
        let far = ImageTransform::new(Channels::Rgb).crop(2, 0, 1, 1);
        let err = ImageCollator::<u8>::new(far)
            .unwrap()
            .write_sample(&rgb2x2(), &mut [0u8; 3])
            .unwrap_err();
        assert!(err.to_string().contains("leaves"), "{err}");
        let small = ImageTransform::new(Channels::Rgb).pad(1, 4, 0);
        assert!(
            ImageCollator::<u8>::new(small)
                .unwrap()
                .write_sample(&rgb2x2(), &mut [0u8; 12])
                .is_err()
        );
    }

    #[test]
    fn resize_same_size_is_a_copy() {
        let t = ImageTransform::new(Channels::Rgb)
            .resize(2, 2, ResizeFilter::Lanczos3)
            .layout(Layout::Nhwc);
        let mut row = vec![0u8; 12];
        ImageCollator::<u8>::new(t)
            .unwrap()
            .write_sample(&rgb2x2(), &mut row)
            .unwrap();
        assert_eq!(row, [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]);
    }

    #[test]
    fn collates_with_metadata_and_names_the_bad_sample() {
        let c =
            ImageCollator::<u8>::new(ImageTransform::new(Channels::Gray).input_size(2, 2)).unwrap();
        let mut buffer = vec![0u8; 12];
        let infos = c
            .collate_into(vec![(rgb2x2(), "a"), (rgb2x2(), "b")], &mut buffer)
            .unwrap();
        assert_eq!(
            infos,
            [
                SampleInfo {
                    width: 2,
                    height: 2,
                    meta: "a"
                },
                SampleInfo {
                    width: 2,
                    height: 2,
                    meta: "b"
                }
            ]
        );
        let err = c
            .collate_into(vec![rgb2x2(), b"junk".to_vec()], &mut buffer)
            .unwrap_err();
        assert!(err.to_string().starts_with("image 1 of the batch"), "{err}");
        let err = c.collate_into(vec![rgb2x2(); 4], &mut buffer).unwrap_err();
        assert!(err.to_string().contains("do not fit"), "{err}");
        assert!(c.write_sample(&rgb2x2(), &mut [0u8; 3]).is_err());
    }
}
