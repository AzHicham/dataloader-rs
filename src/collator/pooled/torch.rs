//! Pinned `tch::Tensor` batch buffers, and `DefaultCollator` writing tensors
//! into them.
//!
//! Feature: `torch-rs`

use tch::{Device, Kind, Tensor};

use crate::{
    collator::{
        DefaultCollator,
        pooled::{BufferAllocator, CollateInto},
        torch::accelerator,
    },
    error::{Error, Result},
};

/// Allocates CPU `tch::Tensor` buffers of a fixed shape, page-locked (pinned)
/// for the accelerator when one is available, so that their copy to the GPU
/// is fast and can be asynchronous.
///
/// Pinning is skipped when it is disabled or neither CUDA nor MPS is
/// available, like `DataLoaderBuilder::pin_memory`.
#[derive(Debug)]
pub struct TorchPinnedAllocator {
    shape: Vec<i64>,
    kind: Kind,
    pin_for: Option<Device>,
}

impl TorchPinnedAllocator {
    /// Buffers of `shape` and `kind`, e.g. `[64, 3, 224, 224]` and
    /// `Kind::Float`; pinned if `pin` and an accelerator exists. Fails on a
    /// negative dimension or a size beyond `isize::MAX` bytes.
    pub fn new(shape: &[i64], kind: Kind, pin: bool) -> Result<Self> {
        let elements = shape
            .iter()
            .try_fold(1i64, |acc, &dim| {
                (dim >= 0).then_some(())?;
                acc.checked_mul(dim)
            })
            .ok_or_else(|| -> Error { format!("invalid buffer shape {shape:?}").into() })?;
        let bytes = usize::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(kind.elt_size_in_bytes()))
            .filter(|&bytes| bytes <= isize::MAX as usize);
        if bytes.is_none() {
            return Err(format!("buffer shape {shape:?} of {kind:?} is too large").into());
        }
        Ok(Self {
            shape: shape.to_vec(),
            kind,
            pin_for: pin.then(accelerator).flatten(),
        })
    }

    /// Whether buffers are pinned (pinning was asked for and an accelerator
    /// exists).
    pub fn is_pinned(&self) -> bool {
        self.pin_for.is_some()
    }
}

impl BufferAllocator for TorchPinnedAllocator {
    type Buffer = Tensor;

    fn allocate(&self) -> Result<Tensor> {
        let buffer = Tensor::f_empty(self.shape.as_slice(), (self.kind, Device::Cpu))?;
        Ok(match self.pin_for {
            Some(device) => buffer.f_pin_memory(device)?,
            None => buffer,
        })
    }
}

/// Stacks tensor samples into rows `0..n` of the buffer, in place. Every
/// sample must have the shape of a buffer row.
impl CollateInto<Tensor> for DefaultCollator {
    type Buffer = Tensor;
    type Output = ();

    fn collate_into(&self, items: Vec<Tensor>, buffer: &mut Tensor) -> Result<()> {
        let size = buffer.size();
        let Some((&rows, row_shape)) = size.split_first() else {
            return Err("buffer tensor has no batch dimension".into());
        };
        let n = i64::try_from(items.len())?;
        if n == 0 || n > rows {
            return Err(format!("{n} samples for a buffer of {rows} rows").into());
        }
        if let Some(item) = items.iter().find(|item| item.size() != row_shape) {
            return Err(
                format!("sample of shape {:?}, expected {row_shape:?}", item.size()).into(),
            );
        }
        // A view of the first `n` rows: with the right shape, `stack_out`
        // writes into it without reallocating.
        let out = buffer.f_narrow(0, 0, n)?;
        // Returns `out` itself.
        let _ = Tensor::f_stack_out(&out, &items, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::TorchPinnedAllocator;
    use crate::collator::{
        DefaultCollator,
        pooled::{BufferAllocator, CollateInto},
    };

    #[test]
    fn stacks_into_the_buffer_in_place() {
        let alloc = TorchPinnedAllocator::new(&[4, 2, 3], Kind::Float, true).unwrap();
        let mut buffer = alloc.allocate().unwrap();
        assert_eq!(buffer.size(), vec![4, 2, 3]);
        let ptr = buffer.data_ptr();

        let items = vec![
            Tensor::ones([2, 3], (Kind::Float, Device::Cpu)),
            Tensor::full([2, 3], 2.0, (Kind::Float, Device::Cpu)),
        ];
        DefaultCollator.collate_into(items, &mut buffer).unwrap();
        assert_eq!(buffer.data_ptr(), ptr);
        assert_eq!(buffer.size(), vec![4, 2, 3]);
        assert_eq!(buffer.get(0).sum(Kind::Float).double_value(&[]), 6.0);
        assert_eq!(buffer.get(1).sum(Kind::Float).double_value(&[]), 12.0);
    }

    #[test]
    fn rejects_bad_samples() {
        let alloc = TorchPinnedAllocator::new(&[2, 3], Kind::Float, false).unwrap();
        let mut buffer = alloc.allocate().unwrap();
        let row = || Tensor::zeros([3], (Kind::Float, Device::Cpu));
        assert!(
            DefaultCollator
                .collate_into(Vec::<Tensor>::new(), &mut buffer)
                .is_err()
        );
        assert!(
            DefaultCollator
                .collate_into(vec![row(), row(), row()], &mut buffer)
                .is_err()
        );
        let wrong = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        assert!(
            DefaultCollator
                .collate_into(vec![wrong], &mut buffer)
                .is_err()
        );
    }

    #[test]
    fn rejects_hostile_shapes() {
        assert!(TorchPinnedAllocator::new(&[-1, 3], Kind::Float, false).is_err());
        assert!(TorchPinnedAllocator::new(&[i64::MAX, 2], Kind::Float, false).is_err());
        assert!(TorchPinnedAllocator::new(&[1 << 62], Kind::Float, false).is_err());
    }
}
