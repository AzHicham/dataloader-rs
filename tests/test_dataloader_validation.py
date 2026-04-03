import pytest

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset


def test_rejects_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        DataLoader(ListDataset([1, 2, 3]), batch_size=0)


def test_rejects_zero_prefetch_depth():
    with pytest.raises(ValueError, match="prefetch_depth must be > 0"):
        DataLoader(ListDataset([1, 2, 3]), prefetch_depth=0)


def test_rejects_shuffle_and_sampler_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DataLoader(ListDataset([1, 2, 3]), shuffle=True, sampler=[0, 1, 2])


def test_rejects_dataset_not_subclassing_pydataset():
    class PlainDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    with pytest.raises(TypeError, match="PyDatasetBase"):
        DataLoader(PlainDataset(), batch_size=1)


def test_len_matches_drop_last_false():
    loader = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=False)
    assert len(loader) == 4


def test_len_matches_drop_last_true():
    loader = DataLoader(ListDataset(range(10)), batch_size=3, drop_last=True)
    assert len(loader) == 3


def test_accepts_num_workers_argument():
    loader = DataLoader(ListDataset(range(10)), batch_size=3, num_workers=0)
    assert len(loader) == 4


def test_accepts_prefetch_depth_argument():
    loader = DataLoader(ListDataset(range(10)), batch_size=3, prefetch_depth=4)
    assert len(loader) == 4
