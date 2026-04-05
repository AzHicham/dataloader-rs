from dataloader_rs import PyDataloader as DataLoader
from tests.py_dataloader_test_utils import ListDataset, ToyDataset, materialize


def test_iteration_without_collate_matches_torch_like_batches():
    loader = DataLoader(ToyDataset(5), batch_size=2, drop_last=False)
    batches = materialize(loader)
    assert len(batches) == 3
    assert batches[0] == [{"x": 0, "y": 0}, {"x": 1, "y": 2}]
    assert batches[1] == [{"x": 2, "y": 4}, {"x": 3, "y": 6}]
    assert batches[2] == [{"x": 4, "y": 8}]


def test_drop_last_true_discards_short_final_batch():
    loader = DataLoader(ToyDataset(5), batch_size=2, drop_last=True)
    batches = materialize(loader)
    assert len(batches) == 2
    assert batches[-1] == [{"x": 2, "y": 4}, {"x": 3, "y": 6}]


def test_custom_collate_fn_is_applied():
    def collate_fn(items):
        return {
            "x": [it["x"] for it in items],
            "y": [it["y"] for it in items],
        }

    loader = DataLoader(ToyDataset(4), batch_size=2, collate_fn=collate_fn)
    out = materialize(loader)
    assert out == [
        {"x": [0, 1], "y": [0, 2]},
        {"x": [2, 3], "y": [4, 6]},
    ]


def test_sum_collate_sequential_exact():
    """N=12, bs=4, sum collate → exact expected values [6, 22, 38]."""
    # batch 0: 0+1+2+3=6, batch 1: 4+5+6+7=22, batch 2: 8+9+10+11=38
    loader = DataLoader(ListDataset(range(12)), batch_size=4, collate_fn=sum)
    assert list(loader) == [6, 22, 38]


def test_sum_collate_parallel_exact():
    """Same expected sums in the parallel path — SequentialSampler is deterministic."""
    loader = DataLoader(ListDataset(range(12)), batch_size=4, num_workers=4, collate_fn=sum)
    assert list(loader) == [6, 22, 38]


def test_python_sampler_order_is_respected():
    sampler = [4, 0, 2, 1, 3]
    loader = DataLoader(ToyDataset(5), batch_size=2, sampler=sampler)
    batches = materialize(loader)
    assert batches[0][0]["x"] == 4
    assert batches[0][1]["x"] == 0
    assert batches[1][0]["x"] == 2
