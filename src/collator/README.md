# Collators

A collator turns the samples of one batch, `Vec<Item>`, into the batch value the iterator yields.
It runs on the thread that built the batch (a worker thread with `num_workers > 0`). Choose one
with `DataLoaderBuilder::collator`; the default is `VecCollator`.

| Collator | Batch | Feature |
|---|---|---|
| `VecCollator` | `Vec<Item>`, unchanged | |
| `DefaultCollator` | column-wise, like PyTorch's `default_collate` (below) | `ndarray` / `torch-rs` for arrays and tensors |
| `TorchPinnedCollator<C>` | `C`'s `tch::Tensor` batch, pinned in host memory when CUDA or MPS is available | `torch-rs` |

## The `Collator` trait

```rust
use dataloader_rs::{Collator, error::Result};

/// Sum of the batch's samples.
struct Sum;

impl Collator<u64> for Sum {
    type Batch = u64;

    fn collate(&self, items: Vec<u64>) -> Result<u64> {
        Ok(items.into_iter().sum())
    }
}

assert_eq!(Sum.collate(vec![1, 2, 3]).unwrap(), 6);
```

`Collator<Item>: Send + Sync + 'static` and `Batch: Send + 'static`: one collator is shared by
every worker, and batches cross threads. A `collate` error is reported for its batch, like a
dataset error.

## `DefaultCollator`

Collates by structure, recursively, through the `DefaultCollate` trait:

| Sample type | Batch type | How |
|---|---|---|
| `bool`, `char`, integers, `f32`, `f64`, `String` | `Vec<T>` | collected |
| `(A, B, …)`, up to 6 elements | `(A::Batch, B::Batch, …)` | per position |
| `[T; N]` | `[T::Batch; N]` | per position |
| `Vec<T>` | `Vec<T::Batch>` | per position; every sample must have the same length |
| `HashMap<K, V>`, `BTreeMap<K, V>` | the same map of `V::Batch` | per key; every sample must have the same keys |
| `ndarray::Array<A, D>` (`ndarray`) | `Array<A, D::Larger>` | stacked on a new axis 0 |
| `tch::Tensor` (`torch-rs`) | `Tensor` | `Tensor::stack` on dimension 0 |

A sample that breaks a rule (different lengths, different keys, arrays of different shapes)
makes `collate` return an error for that batch.

```rust
use dataloader_rs::{DataLoader, Dataset, collator::DefaultCollator, error::Result};

/// (label, features)
struct Points;

impl Dataset for Points {
    type Item = (u32, [f32; 2]);

    fn get(&self, index: usize) -> Result<(u32, [f32; 2])> {
        let x = index as f32;
        Ok((index as u32 % 2, [x, x * 10.0]))
    }

    fn len(&self) -> usize {
        4
    }
}

let mut loader = DataLoader::builder(Points)
    .batch_size(2)
    .collator(DefaultCollator)
    .build();
let (labels, [xs, ys]) = loader.iter().next().unwrap().unwrap();
assert_eq!(labels, vec![0, 1]);
assert_eq!(xs, vec![0.0, 1.0]);
assert_eq!(ys, vec![0.0, 10.0]);
```

Implement `DefaultCollate` for your own sample type to make it collatable the same way.

## `TorchPinnedCollator`

`DataLoaderBuilder::pin_memory(true)` (feature `torch-rs`) wraps the current collator, whose
batch must be a `tch::Tensor`, and pins each batch in page-locked host memory, which speeds up
the copy to a GPU. Pinning is skipped when neither CUDA nor MPS is available.

## From Python

`collate_fn=` takes any callable `list[sample] -> batch`; without it a batch is a `list`. It runs
on the worker thread that fetched the batch ([python/README.md](../../python/README.md)).
