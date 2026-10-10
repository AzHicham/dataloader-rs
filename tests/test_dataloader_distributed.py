"""DistributedSampler: the Rust sampler exposed with torch's Python API."""

import pytest

from dataloader_rs import DistributedSampler
from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset


def _flat(loader):
    return [x for batch in loader for x in batch]


def _rank_loaders(ds, world, num_workers=0, **sampler_kwargs):
    samplers = [
        DistributedSampler(ds, num_replicas=world, rank=r, **sampler_kwargs) for r in range(world)
    ]
    loaders = [
        DataLoader(ds, batch_size=2, sampler=s, num_workers=num_workers) for s in samplers
    ]
    return samplers, loaders


@pytest.mark.parametrize("num_workers", [0, 2])
def test_ranks_get_disjoint_shards_covering_dataset(num_workers):
    ds = ListDataset(range(12))
    _, loaders = _rank_loaders(ds, 3, num_workers=num_workers, seed=7)
    shards = [_flat(loader) for loader in loaders]
    assert all(len(shard) == 4 for shard in shards)
    assert sorted(x for shard in shards for x in shard) == list(range(12))


def test_set_epoch_changes_order_identically_on_all_ranks():
    ds = ListDataset(range(40))
    samplers, loaders = _rank_loaders(ds, 2, seed=3)

    epoch0 = [_flat(loader) for loader in loaders]
    for s in samplers:
        s.set_epoch(1)
    epoch1 = [_flat(loader) for loader in loaders]

    assert epoch0 != epoch1
    for epoch in (epoch0, epoch1):
        assert sorted(epoch[0] + epoch[1]) == list(range(40)), "ranks must not overlap"


def test_same_epoch_gives_same_order():
    """Like torch: without set_epoch, every epoch reuses the same order."""
    ds = ListDataset(range(20))
    sampler = DistributedSampler(ds, num_replicas=2, rank=0, seed=1)
    loader = DataLoader(ds, sampler=sampler)
    assert list(loader) == list(loader)
    sampler.set_epoch(5)
    order5 = list(loader)
    sampler.set_epoch(0)
    sampler.set_epoch(5)
    assert list(loader) == order5


def test_shuffle_false_is_strided_order():
    ds = ListDataset(range(10))
    sampler = DistributedSampler(ds, num_replicas=3, rank=1, shuffle=False)
    # Padded to 12 by wrapping: [0..9, 0, 1]; rank 1 takes 1, 4, 7, 0.
    assert list(sampler) == [1, 4, 7, 0]
    assert len(sampler) == 4


def test_drop_last_truncates():
    ds = ListDataset(range(10))
    sampler = DistributedSampler(ds, num_replicas=3, rank=2, shuffle=False, drop_last=True)
    assert list(sampler) == [2, 5, 8]
    assert len(sampler) == 3


def test_loader_len_counts_this_rank_only():
    ds = ListDataset(range(10))
    sampler = DistributedSampler(ds, num_replicas=4, rank=0)
    loader = DataLoader(ds, batch_size=2, sampler=sampler)
    assert len(loader) == 2  # 3 indices on this rank -> 2 batches
    assert len(list(loader)) == 2


def test_same_seed_matches_across_processes_emulated():
    """Two independent sampler objects (as on two machines) agree."""
    ds = ListDataset(range(30))
    a = DistributedSampler(ds, num_replicas=2, rank=0, seed=11)
    b = DistributedSampler(ds, num_replicas=2, rank=0, seed=11)
    a.set_epoch(2)
    b.set_epoch(2)
    assert list(a) == list(b)


def test_properties():
    s = DistributedSampler(ListDataset(range(4)), num_replicas=2, rank=1)
    s.set_epoch(9)
    assert (s.num_replicas, s.rank, s.epoch) == (2, 1, 9)


def test_invalid_rank_raises():
    with pytest.raises(ValueError, match="Invalid rank 2"):
        DistributedSampler(ListDataset(range(4)), num_replicas=2, rank=2)


def test_zero_replicas_raises():
    with pytest.raises(ValueError, match="num_replicas must be > 0"):
        DistributedSampler(ListDataset(range(4)), num_replicas=0, rank=0)


def test_requires_world_without_torch_distributed():
    with pytest.raises(ValueError, match="num_replicas and rank are required"):
        DistributedSampler(ListDataset(range(4)))
