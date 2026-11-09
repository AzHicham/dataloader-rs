#![doc = include_str!("README.md")]

mod buffer;
mod color;
mod crop;
mod decode;
mod error;
mod flip;
mod layout;
mod pad;
mod resize;
mod rotate;

pub use buffer::{Image, MAX_CHANNELS, Sample};
pub use color::{ToGray, ToRgb, ToRgba};
pub use crop::{CenterCrop, Crop};
pub use decode::{Decode, DecodeLimits};
pub use error::ImageError;
pub use flip::{HFlip, VFlip};
pub use layout::{Chw, Hwc, Layout};
pub use pad::{Anchor, Fill, Pad, PadMode, PadTo};
pub use resize::{Filter, Resize};
pub use rotate::Rotate90;
