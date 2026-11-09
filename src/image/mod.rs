#![doc = include_str!("README.md")]

mod buffer;
mod color;
mod decode;
mod error;
mod layout;

pub use buffer::{Image, MAX_CHANNELS, Sample};
pub use color::{ToGray, ToRgb, ToRgba};
pub use decode::{Decode, DecodeLimits};
pub use error::ImageError;
pub use layout::{Chw, Hwc, Layout};
