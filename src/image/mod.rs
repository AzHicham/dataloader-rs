#![doc = include_str!("README.md")]

mod buffer;
mod color;
mod crop;
mod decode;
mod dtype;
mod error;
mod flip;
mod layout;
mod normalize;
mod pad;
mod resize;
mod rotate;

pub use buffer::{Image, MAX_CHANNELS, Sample};
pub use color::{ToGray, ToRgb, ToRgba};
pub use crop::{CenterCrop, Crop};
pub use decode::{Decode, DecodeLimits};
pub use dtype::ToFloat;
pub use error::ImageError;
pub use flip::{HFlip, VFlip};
pub use layout::{Chw, Hwc, Layout, ToChw, ToHwc};
pub use normalize::Normalize;
pub use pad::{Anchor, Fill, Pad, PadMode, PadTo};
pub use resize::{Filter, Resize};
pub use rotate::Rotate90;
