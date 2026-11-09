"""Batches made by ``collate_fn``: without one a batch is a list of items; with one it can be
anything, here a dict of NumPy arrays. The dataset also fetches whole batches at once with
``__getitems__``, PyTorch's batched-fetch method. Needs NumPy.

Run it with ``python examples/python/collate_fn.py``.
"""

import numpy as np

from dataloader_rs import PyDataloader, PyDataset


class Points(PyDataset):
    """Twelve labelled points: item ``i`` is ``{"features": [x, x * x], "label": i % 3}``."""

    def __init__(self) -> None:
        super().__init__()
        x = np.arange(12, dtype=np.float32)
        self.features = np.stack([x, x * x], axis=1)  # shape (12, 2)
        self.labels = np.arange(12) % 3

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"features": self.features[index], "label": int(self.labels[index])}

    def __getitems__(self, indices: list[int]) -> list[dict]:
        # Optional. When a dataset defines it, the loader calls it once per batch instead of
        # __getitem__ once per item: one NumPy indexing for the whole batch. It must return
        # one item per index, in the same order.
        features = self.features[indices]
        labels = self.labels[indices]
        return [
            {"features": f, "label": int(label)} for f, label in zip(features, labels, strict=True)
        ]


def to_arrays(items: list[dict]) -> dict[str, np.ndarray]:
    """Stack a batch's items key by key: ``{"features": (n, 2), "label": (n,)}``."""
    return {key: np.stack([item[key] for item in items]) for key in items[0]}


def main() -> None:
    # Without collate_fn: a list of the items.
    plain = PyDataloader(Points(), batch_size=4)
    print("a list:", next(iter(plain))[:2], "...")

    # With collate_fn: it runs on the worker thread that loaded the batch, so heavy
    # collation is parallel too.
    loader = PyDataloader(
        Points(), batch_size=4, shuffle=True, seed=0, num_workers=2, collate_fn=to_arrays
    )
    for batch in loader:
        print(f"features {batch['features'].shape}, labels {batch['label']}")

    # Any callable that takes a list works, such as np.stack for items that are arrays.
    stacked = PyDataloader(
        Points(), batch_size=6, collate_fn=lambda items: np.stack([i["features"] for i in items])
    )
    print("np.stack:", [batch.shape for batch in stacked])


if __name__ == "__main__":
    main()
