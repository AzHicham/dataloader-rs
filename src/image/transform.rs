//! The declarative pipeline: channels, operations, layout and limits, validated once.

use std::fmt;

use super::{ImageError, ImageLimits, ImageResult, ResizeFilter};

/// The channels of the decoded image, and of every sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Channels {
    /// Three channels; gray images are replicated, alpha is dropped (Pillow's `convert("RGB")`).
    #[default]
    Rgb,
    /// One channel; color images are converted with Pillow's ITU-R 601-2 luma (`convert("L")`).
    Gray,
}

impl Channels {
    /// Values per pixel: 3 or 1.
    pub fn count(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Gray => 1,
        }
    }
}

/// How a sample's values are laid out in its row of the batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Channel planes, `C × H × W` per sample: PyTorch's layout (`ToTensor`).
    #[default]
    Nchw,
    /// Interleaved pixels, `H × W × C` per sample: the decoded layout (TensorFlow, ONNX models
    /// that expect channels last).
    Nhwc,
}

/// One operation of an [`ImageTransform`], applied in order.
///
/// Geometric operations (`Resize`, `Crop`, `Pad`) work on 8-bit pixels and come first; value
/// operations (`ToFloat`, `Normalize`) follow, and are applied as the sample is written to the
/// batch.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Resize to `width × height` pixels.
    Resize {
        /// Output width.
        width: u32,
        /// Output height.
        height: u32,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Keep the `width × height` region whose top-left pixel is (`left`, `top`); a region that
    /// leaves the image is an error.
    Crop {
        /// First column.
        left: u32,
        /// First row.
        top: u32,
        /// Columns kept.
        width: u32,
        /// Rows kept.
        height: u32,
    },
    /// Grow the image to `width × height`, anchored top-left, filling the new pixels with `fill`
    /// in every channel. An image larger than the target is an error.
    Pad {
        /// Output width, at least the image's.
        width: u32,
        /// Output height, at least the image's.
        height: u32,
        /// Value of the added pixels.
        fill: u8,
    },
    /// Divide by 255: values in `[0, 1]`, as torchvision's `ToTensor`.
    ToFloat,
    /// `(x - mean[c]) / std[c]` per channel `c`, as torchvision's `Normalize`; one value applies
    /// to every channel.
    Normalize {
        /// Per-channel mean.
        mean: Vec<f32>,
        /// Per-channel standard deviation; non-zero.
        std: Vec<f32>,
    },
}

/// A declarative image pipeline: decode, geometric operations, value operations, and the layout
/// of the sample in its batch row.
///
/// Built with chained calls (or [`ops`](Self::ops) from a list) and validated by
/// [`ImageCollator::new`](super::ImageCollator::new), which reports the first invalid operation.
///
/// ```rust
/// use dataloader_rs::image::{Channels, ImageTransform, Layout, ResizeFilter};
///
/// let transform = ImageTransform::new(Channels::Rgb)
///     .resize(256, 256, ResizeFilter::Bilinear)
///     .crop(16, 16, 224, 224)
///     .to_float()
///     .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
///     .layout(Layout::Nchw);
/// assert_eq!(transform.sample_shape().unwrap(), [3, 224, 224]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTransform {
    channels: Channels,
    input_size: Option<(u32, u32)>,
    ops: Vec<Op>,
    layout: Layout,
    limits: ImageLimits,
}

impl ImageTransform {
    /// A pipeline that decodes to `channels`, with no operation, NCHW and default limits.
    pub fn new(channels: Channels) -> Self {
        Self {
            channels,
            input_size: None,
            ops: Vec::new(),
            layout: Layout::default(),
            limits: ImageLimits::default(),
        }
    }

    /// Require every decoded image to be `width × height`: another size is an error for its
    /// batch. Without a geometric operation, this sets the sample size.
    pub fn input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = Some((width, height));
        self
    }

    /// Append operations.
    pub fn ops(mut self, ops: impl IntoIterator<Item = Op>) -> Self {
        self.ops.extend(ops);
        self
    }

    /// Append [`Op::Resize`].
    pub fn resize(self, width: u32, height: u32, filter: ResizeFilter) -> Self {
        self.ops([Op::Resize {
            width,
            height,
            filter,
        }])
    }

    /// Append [`Op::Crop`].
    pub fn crop(self, left: u32, top: u32, width: u32, height: u32) -> Self {
        self.ops([Op::Crop {
            left,
            top,
            width,
            height,
        }])
    }

    /// Append [`Op::Pad`].
    pub fn pad(self, width: u32, height: u32, fill: u8) -> Self {
        self.ops([Op::Pad {
            width,
            height,
            fill,
        }])
    }

    /// Append [`Op::ToFloat`].
    pub fn to_float(self) -> Self {
        self.ops([Op::ToFloat])
    }

    /// Append [`Op::Normalize`].
    pub fn normalize(
        self,
        mean: impl IntoIterator<Item = f32>,
        std: impl IntoIterator<Item = f32>,
    ) -> Self {
        self.ops([Op::Normalize {
            mean: mean.into_iter().collect(),
            std: std.into_iter().collect(),
        }])
    }

    /// Lay samples out as `layout` (default NCHW).
    pub fn layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Bound the images accepted (default [`ImageLimits::default`]).
    pub fn limits(mut self, limits: ImageLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The channels of every sample.
    pub fn channels(&self) -> Channels {
        self.channels
    }

    /// The operations, in order.
    pub fn operations(&self) -> &[Op] {
        &self.ops
    }

    /// The sample layout.
    pub fn sample_layout(&self) -> Layout {
        self.layout
    }

    /// The image limits.
    pub fn image_limits(&self) -> ImageLimits {
        self.limits
    }

    /// The shape of one sample, `[C, H, W]` (NCHW) or `[H, W, C]` (NHWC); an error if the
    /// pipeline is invalid.
    pub fn sample_shape(&self) -> ImageResult<[usize; 3]> {
        Ok(self.plan()?.sample_shape())
    }

    /// Validate the pipeline into the plan the collator runs.
    pub(crate) fn plan(&self) -> ImageResult<Plan> {
        let channels = self.channels.count();
        let check = |step: &dyn fmt::Display, w: u32, h: u32| -> ImageResult<()> {
            if w == 0 || h == 0 {
                return Err(invalid(format!("{step}: empty size {w}x{h}")));
            }
            self.limits
                .check(w as usize, h as usize, channels)
                .map_err(|e| invalid(format!("{step}: {e}")))
        };
        let mut size = self.input_size;
        if let Some((w, h)) = size {
            check(&"input_size", w, h)?;
        }
        let mut geometry = Vec::new();
        let mut values = Vec::new();
        let mut has_to_float = false;
        for (index, op) in self.ops.iter().enumerate() {
            let step = Step { index, op };
            let geometric = matches!(op, Op::Resize { .. } | Op::Crop { .. } | Op::Pad { .. });
            if geometric && !values.is_empty() {
                return Err(invalid(format!(
                    "{step}: geometric operations must come before to_float and normalize"
                )));
            }
            match *op {
                Op::Resize { width, height, .. } => {
                    check(&step, width, height)?;
                    size = Some((width, height));
                }
                Op::Crop {
                    left,
                    top,
                    width,
                    height,
                } => {
                    check(&step, width, height)?;
                    if let Some((w, h)) = size {
                        let fits = left.checked_add(width).is_some_and(|r| r <= w)
                            && top.checked_add(height).is_some_and(|b| b <= h);
                        if !fits {
                            return Err(invalid(format!(
                                "{step}: region leaves the {w}x{h} image"
                            )));
                        }
                    }
                    size = Some((width, height));
                }
                Op::Pad { width, height, .. } => {
                    check(&step, width, height)?;
                    if let Some((w, h)) = size
                        && (w > width || h > height)
                    {
                        return Err(invalid(format!("{step}: the image is already {w}x{h}")));
                    }
                    size = Some((width, height));
                }
                Op::ToFloat => {
                    if has_to_float {
                        return Err(invalid(format!("{step}: to_float is applied twice")));
                    }
                    has_to_float = true;
                    values.push(ValueOp::Scale);
                }
                Op::Normalize { ref mean, ref std } => {
                    let per_channel = |v: &[f32], name: &str| -> ImageResult<Vec<f32>> {
                        match v.len() {
                            1 => Ok(vec![v[0]; channels]),
                            n if n == channels => Ok(v.to_vec()),
                            n => Err(invalid(format!(
                                "{step}: {n} {name} values for {channels} channel(s)"
                            ))),
                        }
                    };
                    let (mean, std) = (per_channel(mean, "mean")?, per_channel(std, "std")?);
                    if mean.iter().any(|m| !m.is_finite()) {
                        return Err(invalid(format!("{step}: mean must be finite")));
                    }
                    if std.iter().any(|s| !s.is_finite() || *s == 0.0) {
                        return Err(invalid(format!("{step}: std must be finite and non-zero")));
                    }
                    values.push(ValueOp::Normalize { mean, std });
                }
            }
            if geometric {
                geometry.push(op.clone());
            }
        }
        let (width, height) = size.ok_or_else(|| {
            invalid(
                "the sample size depends on the input: set input_size, or end the geometric \
                 operations with resize, crop or pad"
                    .into(),
            )
        })?;
        Ok(Plan {
            channels: self.channels,
            input_size: self.input_size,
            geometry,
            values,
            layout: self.layout,
            limits: self.limits,
            width: width as usize,
            height: height as usize,
        })
    }
}

fn invalid(message: String) -> ImageError {
    ImageError::InvalidPipeline(message)
}

/// An operation and its position, for error messages.
struct Step<'a> {
    index: usize,
    op: &'a Op,
}

impl fmt::Display for Step<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.op {
            Op::Resize { .. } => "resize",
            Op::Crop { .. } => "crop",
            Op::Pad { .. } => "pad",
            Op::ToFloat => "to_float",
            Op::Normalize { .. } => "normalize",
        };
        write!(f, "operation {} ({name})", self.index)
    }
}

/// An operation on values, per channel.
#[derive(Clone, Debug)]
pub(crate) enum ValueOp {
    /// `x / 255`.
    Scale,
    /// `(x - mean) / std`.
    Normalize { mean: Vec<f32>, std: Vec<f32> },
}

/// A validated pipeline.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub(crate) channels: Channels,
    pub(crate) input_size: Option<(u32, u32)>,
    /// `Resize`, `Crop` and `Pad` only.
    pub(crate) geometry: Vec<Op>,
    pub(crate) values: Vec<ValueOp>,
    pub(crate) layout: Layout,
    pub(crate) limits: ImageLimits,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Plan {
    pub(crate) fn sample_shape(&self) -> [usize; 3] {
        let c = self.channels.count();
        match self.layout {
            Layout::Nchw => [c, self.height, self.width],
            Layout::Nhwc => [self.height, self.width, c],
        }
    }

    /// Values per sample. Within the limits, so it does not overflow.
    pub(crate) fn sample_len(&self) -> usize {
        self.width * self.height * self.channels.count()
    }

    /// The value of 8-bit input `v` in channel `channel` after the value operations, computed in
    /// `f32` step by step as torchvision does (`x / 255`, then `(x - mean) / std`).
    pub(crate) fn value(&self, channel: usize, v: u8) -> f32 {
        let mut x = f32::from(v);
        for op in &self.values {
            x = match op {
                ValueOp::Scale => x / 255.0,
                ValueOp::Normalize { mean, std } => (x - mean[channel]) / std[channel],
            };
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::{Channels, ImageTransform, Layout, Op};
    use crate::image::{ImageLimits, ResizeFilter};

    fn err(t: ImageTransform) -> String {
        t.plan().unwrap_err().to_string()
    }

    #[test]
    fn shapes() {
        let t = ImageTransform::new(Channels::Rgb).resize(32, 16, ResizeFilter::Bilinear);
        assert_eq!(t.sample_shape().unwrap(), [3, 16, 32]);
        let t = t.layout(Layout::Nhwc);
        assert_eq!(t.sample_shape().unwrap(), [16, 32, 3]);
        let t = ImageTransform::new(Channels::Gray).input_size(5, 4);
        assert_eq!(t.sample_shape().unwrap(), [1, 4, 5]);
        assert_eq!(t.plan().unwrap().sample_len(), 20);
        let t = ImageTransform::new(Channels::Rgb)
            .input_size(10, 10)
            .crop(2, 2, 6, 4)
            .pad(8, 8, 0);
        assert_eq!(t.sample_shape().unwrap(), [3, 8, 8]);
    }

    #[test]
    fn declarative_list_equals_chained_calls() {
        let chained = ImageTransform::new(Channels::Rgb)
            .pad(64, 64, 255)
            .to_float()
            .normalize([0.5], [0.25]);
        let listed = ImageTransform::new(Channels::Rgb).ops([
            Op::Pad {
                width: 64,
                height: 64,
                fill: 255,
            },
            Op::ToFloat,
            Op::Normalize {
                mean: vec![0.5],
                std: vec![0.25],
            },
        ]);
        assert_eq!(chained, listed);
    }

    #[test]
    fn values_follow_torchvision() {
        let plan = ImageTransform::new(Channels::Rgb)
            .input_size(1, 1)
            .to_float()
            .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            .plan()
            .unwrap();
        assert_eq!(plan.value(0, 255), (255.0f32 / 255.0 - 0.485) / 0.229);
        assert_eq!(plan.value(2, 17), (17.0f32 / 255.0 - 0.406) / 0.225);
        let raw = ImageTransform::new(Channels::Gray)
            .input_size(1, 1)
            .plan()
            .unwrap();
        assert_eq!(raw.value(0, 200), 200.0);
    }

    #[test]
    fn invalid_pipelines_are_reported() {
        let rgb = || ImageTransform::new(Channels::Rgb);
        assert!(err(rgb()).contains("sample size depends on the input"));
        assert!(err(rgb().to_float()).contains("sample size"));
        assert!(err(rgb().resize(0, 4, ResizeFilter::Nearest)).contains("empty size"));
        assert!(
            err(rgb().resize(4, 4, ResizeFilter::Bilinear).crop(2, 0, 3, 3))
                .contains("operation 1 (crop): region leaves the 4x4 image")
        );
        assert!(err(rgb().input_size(8, 8).crop(u32::MAX, 0, 2, 2)).contains("region leaves"));
        assert!(err(rgb().input_size(8, 8).pad(4, 9, 0)).contains("already 8x8"));
        assert!(
            err(rgb()
                .input_size(8, 8)
                .to_float()
                .resize(4, 4, ResizeFilter::Box))
            .contains("must come before")
        );
        assert!(err(rgb().input_size(8, 8).to_float().to_float()).contains("twice"));
        assert!(err(rgb().input_size(8, 8).normalize([0.5, 0.5], [1.0])).contains("2 mean"));
        assert!(err(rgb().input_size(8, 8).normalize([0.5], [0.0])).contains("non-zero"));
        assert!(err(rgb().input_size(8, 8).normalize([f32::NAN], [1.0])).contains("finite"));
        assert!(err(rgb().input_size(8, 8).normalize([0.0], [f32::INFINITY])).contains("finite"));
    }

    #[test]
    fn sizes_above_the_limits_are_rejected() {
        let rgb = || ImageTransform::new(Channels::Rgb);
        assert!(err(rgb().resize(u32::MAX, u32::MAX, ResizeFilter::Bilinear)).contains("limits"));
        assert!(err(rgb().pad(16_384, 16_384, 0)).contains("limits"));
        assert!(err(rgb().input_size(16_385, 1)).contains("limits"));
        let small = ImageLimits {
            max_width: 100,
            max_height: 100,
            max_bytes: 300,
        };
        assert!(rgb().limits(small).input_size(10, 10).plan().is_ok());
        assert!(err(rgb().limits(small).input_size(10, 11)).contains("limits"));
    }
}
