import pytest

from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset


class FailingDataset(ListDataset):
    def __init__(self, values, fail_index: int):
        super().__init__(values)
        self.fail_index = fail_index

    def __getitem__(self, idx):
        if idx == self.fail_index:
            raise RuntimeError(f"boom at index {idx}")
        return super().__getitem__(idx)


def test_dataset_error_surfaces_in_iteration():
    loader = DataLoader(FailingDataset(range(8), fail_index=3), batch_size=4)
    it = iter(loader)
    with pytest.raises(RuntimeError, match="boom at index 3"):
        next(it)


def test_collate_error_surfaces_in_iteration():
    def bad_collate(_items):
        raise RuntimeError("collate exploded")

    loader = DataLoader(ListDataset(range(8)), batch_size=4, collate_fn=bad_collate)
    it = iter(loader)
    with pytest.raises(RuntimeError, match="collate exploded"):
        next(it)
