from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset


def test_iter_len_decrements_as_consumed():
    loader = DataLoader(ListDataset(range(7)), batch_size=3, drop_last=False)
    it = iter(loader)
    assert len(it) == 3
    next(it)
    assert len(it) == 2
    next(it)
    assert len(it) == 1
    next(it)
    assert len(it) == 0


def test_loader_reusable_across_epochs():
    loader = DataLoader(ListDataset(range(8)), batch_size=3)
    epoch1 = list(loader)
    epoch2 = list(loader)
    assert epoch1 == epoch2


def test_iter_returns_self():
    loader = DataLoader(ListDataset(range(4)), batch_size=2)
    it = iter(loader)
    assert iter(it) is it
