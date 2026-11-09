#![doc = include_str!("README.md")]

mod default_collator;
#[cfg(feature = "torch-rs")]
mod torch;
mod traits;
mod vec_collator;

pub use default_collator::{DefaultCollate, DefaultCollator};
#[cfg(feature = "torch-rs")]
pub use torch::TorchPinnedCollator;
pub use traits::Collator;
pub use vec_collator::VecCollator;
