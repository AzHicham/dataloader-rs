use crate::collator::Collator;

/// Torch-only wrapper collator that optionally pins Tensor batches.
pub struct TorchPinnedCollator<C> {
    inner: C,
    enabled: bool,
    can_pin: bool,
}

impl<C> TorchPinnedCollator<C> {
    pub fn new(inner: C, enabled: bool) -> Self {
        let can_pin = enabled && (tch::Cuda::is_available() || tch::utils::has_mps());
        Self {
            inner,
            enabled,
            can_pin,
        }
    }
}

impl<Item, C> Collator<Item> for TorchPinnedCollator<C>
where
    C: Collator<Item, Batch = tch::Tensor>,
{
    type Batch = tch::Tensor;

    fn collate(&self, items: Vec<Item>) -> crate::error::Result<Self::Batch> {
        let batch = self.inner.collate(items)?;
        if self.enabled && self.can_pin {
            batch.f_pin_memory(tch::Device::Cpu).map_err(|e| e.into())
        } else {
            Ok(batch)
        }
    }
}
