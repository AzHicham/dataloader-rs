//! The image transforms of Python, without static types: a [`Value`] (bytes or an image of
//! some type and layout) through a list of [`Op`]s, each the Rust transform of the same name.

use std::fmt;

use bytes::Bytes;

use crate::error::Result;
use crate::image::{
    CenterCrop, Chw, Crop, Decode, HFlip, Image, ImageError, Normalize, Pad, PadTo, Resize,
    Rotate90, ToChw, ToFloat, ToGray, ToHwc, ToRgb, ToRgba, VFlip,
};
use crate::transform::Transform;

/// What flows through Python's image transforms.
#[derive(Clone)]
pub(crate) enum Value {
    /// Encoded bytes, before `Decode`.
    Bytes(Bytes),
    U8(Image<u8>),
    U8Chw(Image<u8, Chw>),
    F32(Image<f32>),
    F32Chw(Image<f32, Chw>),
}

impl Value {
    /// `"bytes"`, or the type and layout of the image, for error messages.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Bytes(_) => "bytes",
            Self::U8(_) => "a uint8 HWC image",
            Self::U8Chw(_) => "a uint8 CHW image",
            Self::F32(_) => "a float32 HWC image",
            Self::F32Chw(_) => "a float32 CHW image",
        }
    }
}

/// One of the Rust image transforms.
#[derive(Clone, Debug)]
pub(crate) enum Op {
    Decode(Decode),
    ToRgb,
    ToRgba,
    ToGray,
    Crop(Crop),
    CenterCrop(CenterCrop),
    Pad(Pad),
    PadTo(PadTo),
    Resize(Resize),
    HFlip,
    VFlip,
    Rotate90(Rotate90),
    ToFloat(ToFloat),
    Normalize(Normalize),
    ToChw,
    ToHwc,
}

impl fmt::Display for Op {
    /// The transform's Python name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Decode(_) => "Decode",
            Self::ToRgb => "ToRgb",
            Self::ToRgba => "ToRgba",
            Self::ToGray => "ToGray",
            Self::Crop(_) => "Crop",
            Self::CenterCrop(_) => "CenterCrop",
            Self::Pad(_) => "Pad",
            Self::PadTo(_) => "PadTo",
            Self::Resize(_) => "Resize",
            Self::HFlip => "HFlip",
            Self::VFlip => "VFlip",
            Self::Rotate90(_) => "Rotate90",
            Self::ToFloat(_) => "ToFloat",
            Self::Normalize(_) => "Normalize",
            Self::ToChw => "ToChw",
            Self::ToHwc => "ToHwc",
        };
        f.write_str(name)
    }
}

/// A transform of HWC images of both sample types.
trait Geometric:
    Transform<Image<u8>, Output = Image<u8>> + Transform<Image<f32>, Output = Image<f32>>
{
}

impl<T> Geometric for T where
    T: Transform<Image<u8>, Output = Image<u8>> + Transform<Image<f32>, Output = Image<f32>>
{
}

impl Op {
    /// Apply this transform to `value`, the item at `index`.
    pub(crate) fn apply(&self, index: usize, value: Value) -> Result<Value> {
        use Value::{Bytes, F32, F32Chw, U8, U8Chw};
        Ok(match (self, value) {
            (Self::Decode(t), Bytes(b)) => U8(t.apply(index, b)?),
            (Self::ToRgb, U8(i)) => U8(ToRgb.apply(index, i)?),
            (Self::ToRgba, U8(i)) => U8(ToRgba.apply(index, i)?),
            (Self::ToGray, U8(i)) => U8(ToGray.apply(index, i)?),
            (Self::Resize(t), U8(i)) => U8(t.apply(index, i)?),
            (Self::Crop(t), v) => geometric(t, index, v, self)?,
            (Self::CenterCrop(t), v) => geometric(t, index, v, self)?,
            (Self::Pad(t), v) => geometric(t, index, v, self)?,
            (Self::PadTo(t), v) => geometric(t, index, v, self)?,
            (Self::HFlip, v) => geometric(&HFlip, index, v, self)?,
            (Self::VFlip, v) => geometric(&VFlip, index, v, self)?,
            (Self::Rotate90(t), v) => geometric(t, index, v, self)?,
            (Self::ToFloat(t), U8(i)) => F32(t.apply(index, i)?),
            (Self::ToFloat(t), U8Chw(i)) => F32Chw(t.apply(index, i)?),
            (Self::Normalize(t), F32(i)) => F32(t.apply(index, i)?),
            (Self::Normalize(t), F32Chw(i)) => F32Chw(t.apply(index, i)?),
            (Self::ToChw, U8(i)) => U8Chw(ToChw.apply(index, i)?),
            (Self::ToChw, F32(i)) => F32Chw(ToChw.apply(index, i)?),
            (Self::ToHwc, U8Chw(i)) => U8(ToHwc.apply(index, i)?),
            (Self::ToHwc, F32Chw(i)) => F32(ToHwc.apply(index, i)?),
            (op, value) => return Err(wrong_input(op, &value)),
        })
    }
}

/// Apply `t` to an HWC image of either type.
fn geometric<T: Geometric>(t: &T, index: usize, value: Value, op: &Op) -> Result<Value> {
    match value {
        Value::U8(i) => Ok(Value::U8(t.apply(index, i)?)),
        Value::F32(i) => Ok(Value::F32(t.apply(index, i)?)),
        other => Err(wrong_input(op, &other)),
    }
}

/// `op` cannot take `value`: say what it takes.
fn wrong_input(op: &Op, value: &Value) -> crate::Error {
    let takes = match op {
        Op::Decode(_) => "bytes",
        Op::ToRgb | Op::ToRgba | Op::ToGray | Op::Resize(_) => "a uint8 HWC image",
        Op::ToFloat(_) => "a uint8 image",
        Op::Normalize(_) => "a float32 image (add ToFloat before it)",
        Op::ToChw => "an HWC image",
        Op::ToHwc => "a CHW image",
        _ => "an HWC image (apply ToChw last)",
    };
    ImageError::Invalid(format!("{op} takes {takes}, not {}", value.kind())).into()
}

/// Apply `ops` in turn to `value`, the item at `index`.
pub(crate) fn apply_all(ops: &[Op], index: usize, value: Value) -> Result<Value> {
    ops.iter()
        .try_fold(value, |value, op| op.apply(index, value))
}

#[cfg(test)]
mod tests {
    use super::{Op, Value, apply_all};
    use crate::image::{Image, ToFloat};

    #[test]
    fn a_wrong_order_says_what_the_op_takes() {
        let image = Value::U8(Image::new(1, 1, 3, vec![0, 0, 0]).unwrap());
        let ops = [Op::ToChw, Op::HFlip];
        let err = apply_all(&ops, 0, image).err().unwrap();
        assert_eq!(
            err.to_string(),
            "invalid image: HFlip takes an HWC image (apply ToChw last), not a uint8 CHW image"
        );
        let float = [Op::ToFloat(ToFloat::default()), Op::ToRgb];
        let image = Value::U8(Image::new(1, 1, 3, vec![0, 0, 0]).unwrap());
        let err = apply_all(&float, 0, image).err().unwrap();
        assert!(
            err.to_string()
                .contains("ToRgb takes a uint8 HWC image, not a float32 HWC image")
        );
    }
}
