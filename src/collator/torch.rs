use crate::collator::Collator;

/// Torch-only wrapper collator that optionally pins Tensor batches.
pub struct TorchPinnedCollator<C> {
    inner: C,
    /// Pinning was asked for, and a CUDA or MPS device can use pinned memory.
    pin: bool,
}

impl<C> TorchPinnedCollator<C> {
    pub fn new(inner: C, enabled: bool) -> Self {
        let pin = enabled && (tch::Cuda::is_available() || tch::utils::has_mps());
        Self { inner, pin }
    }
}

impl<Item, C> Collator<Item> for TorchPinnedCollator<C>
where
    C: Collator<Item, Batch = tch::Tensor>,
{
    type Batch = tch::Tensor;

    fn collate(&self, items: Vec<Item>) -> crate::error::Result<Self::Batch> {
        let batch = self.inner.collate(items)?;
        if self.pin {
            batch.f_pin_memory(tch::Device::Cpu).map_err(|e| e.into())
        } else {
            Ok(batch)
        }
    }
}
