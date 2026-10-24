# Samplers

A sampler decides the **order of indices** in one epoch; `BatchSampler` then cuts that order into
batches. Choose one with `DataLoaderBuilder::sampler`; the default is `SequentialSampler`.

| Sampler | Order every epoch | Length |
|---|---|---|
| `SequentialSampler` | `0, 1, …, len - 1` | `len` |
| `RandomSampler::new(seed)` | a new uniform permutation, reproducible from `seed` | `len` |
| `RandomSampler::from_entropy()` | a new uniform permutation, seeded from the OS | `len` |
| `DistributedSampler::new(inner, rank, world_size)` | the `rank`-th of `world_size` strided shards of `inner`'s order | `ceil(len / world_size)` |

## The `Sampler` trait

```rust
pub trait Sampler: Send + 'static {
    /// The index order of one epoch over `dataset_len` items.
    fn indices(&mut self, dataset_len: usize) -> Vec<usize>;

    /// How many indices `indices` returns, without advancing the sampler.
    fn len(&self, dataset_len: usize) -> usize {
        dataset_len
    }
}
```

- `indices` is called once per epoch, at `DataLoader::iter()`, and takes `&mut self`: a sampler
  may keep state between epochs (`RandomSampler` advances its generator, so each epoch gets a new
  order while the sequence of orders stays reproducible).
- `len` sizes the epoch for `DataLoader::batch_len` and Python's `len(loader)`. Override it when
  a sampler returns more or fewer indices than the dataset holds.

## `RandomSampler`

Inside-out Fisher–Yates with `fastrand` (wyrand): one forward pass, no pre-filled array. It costs
1.6–2× the sequential sampler, about 4 ns per index ([BENCH.md](../../BENCH.md)).

## `DistributedSampler`

Shards an inner sampler's order across `world_size` ranks, like PyTorch's: the order is padded by
repeating it from the start up to a multiple of `world_size`, then rank `r` takes every
`world_size`-th index starting at `r`. Every rank gets the same number of indices, so batches stay
in step across ranks; a few indices appear twice in an epoch when `len` is not a multiple of
`world_size`.

```rust
use dataloader_rs::{DistributedSampler, Sampler, SequentialSampler};

// 10 items over 4 ranks: padded to 12, 3 indices per rank.
let mut rank1 = DistributedSampler::new(SequentialSampler, 1, 4);
assert_eq!(rank1.indices(10), vec![1, 5, 9]);
assert_eq!(rank1.len(10), 3);

let mut rank3 = DistributedSampler::new(SequentialSampler, 3, 4);
assert_eq!(rank3.indices(10), vec![3, 7, 1]); // index 1 repeated as padding
```

With a `RandomSampler` inside, give every rank the **same seed**: the ranks then shard the same
permutation.

## `BatchSampler`

Built by the loader from `batch_size` and `drop_last`; you do not create it yourself.

```rust
use dataloader_rs::{BatchSampler, SequentialSampler};

let mut batches = BatchSampler::new(SequentialSampler, 4, false);
assert_eq!(batches.batch_indices(10), vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]);
assert_eq!(batches.len(10), 3);

let full_only = BatchSampler::new(SequentialSampler, 4, true); // drop_last
assert_eq!(full_only.len(10), 2);
```

## A custom sampler

```rust
use dataloader_rs::{DataLoader, Dataset, Sampler, error::Result};

/// Every index, newest first.
struct Reverse;

impl Sampler for Reverse {
    fn indices(&mut self, dataset_len: usize) -> Vec<usize> {
        (0..dataset_len).rev().collect()
    }
}

struct Numbers(usize);

impl Dataset for Numbers {
    type Item = usize;
    fn get(&self, index: usize) -> Result<usize> {
        Ok(index)
    }
    fn len(&self) -> usize {
        self.0
    }
}

let mut loader = DataLoader::builder(Numbers(5)).batch_size(2).sampler(Reverse).build();
let batches: Vec<Vec<usize>> = loader.iter().map(|batch| batch.unwrap()).collect();
assert_eq!(batches, vec![vec![4, 3], vec![2, 1], vec![0]]);
```

From Python, `sampler=` accepts any iterable of `int`, and `shuffle=True` / `seed=` use
`RandomSampler` ([python/README.md](../../python/README.md#samplers)).
