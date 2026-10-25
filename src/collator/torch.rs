use tch::{Cuda, Device, Tensor};

use crate::collator::Collator;

/// Torch-only wrapper collator that optionally pins Tensor batches.
pub struct TorchPinnedCollator<C> {
    inner: C,
    /// The accelerator batches are pinned for, when pinning was asked for and
    /// a CUDA or MPS device exists. libtorch's `pin_memory` takes the device
    /// the page-locked host memory will be copied to, not the tensor's own
    /// (always the CPU): pinning "for the CPU" is an error.
    pin_for: Option<Device>,
}

impl<C> TorchPinnedCollator<C> {
    pub fn new(inner: C, enabled: bool) -> Self {
        Self {
            inner,
            pin_for: enabled.then(accelerator).flatten(),
        }
    }
}

/// The accelerator pinned memory is for: CUDA (any index; the pinned
/// allocator is per device type), else MPS.
fn accelerator() -> Option<Device> {
    if Cuda::is_available() {
        Some(Device::Cuda(0))
    } else if tch::utils::has_mps() {
        Some(Device::Mps)
    } else {
        None
    }
}

impl<Item, C> Collator<Item> for TorchPinnedCollator<C>
where
    C: Collator<Item, Batch = Tensor>,
{
    type Batch = Tensor;

    fn collate(&self, items: Vec<Item>) -> crate::error::Result<Self::Batch> {
        let batch = self.inner.collate(items)?;
        match self.pin_for {
            Some(device) => batch.f_pin_memory(device).map_err(Into::into),
            None => Ok(batch),
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    #[test]
    fn libtorch_rejects_pinning_for_the_cpu() {
        // Why the collator pins for the accelerator: the device argument is
        // where the memory will be copied to, and the CPU is not one.
        let batch = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        assert!(batch.f_pin_memory(Device::Cpu).is_err());
    }
}
