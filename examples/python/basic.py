"""The basics: a small in-memory dataset, a shuffled loader with worker threads, the number of
batches, and stopping an epoch early. Needs nothing but dataloader-rs.

Run it with ``python examples/python/basic.py``.
"""

from dataloader_rs import PyDataloader, PyDataset


class Squares(PyDataset):
    """Item ``i`` is ``i * i``. ``__getitem__`` runs on the loader's worker threads, so a
    dataset should only read its own state (or protect what it changes with a lock)."""

    def __init__(self, length: int) -> None:
        super().__init__()  # required when a dataset defines __init__
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index * index


def main() -> None:
    loader = PyDataloader(
        Squares(100),
        batch_size=10,
        shuffle=True,  # a new order every epoch...
        seed=42,  # ...the same sequence of orders on every run
        num_workers=4,  # 4 batches are loaded in parallel, on threads
        prefetch_depth=4,  # and up to 4 more may wait, ready, for the loop below
        drop_last=True,  # a last batch smaller than 10 would be dropped (there is none here)
    )
    print(f"{len(loader)} batches per epoch")
    for epoch in range(2):
        # Iterating over the loader again starts a new epoch, in a new order. Without a
        # collate_fn, a batch is a list of items.
        first = next(iter(loader))
        print(f"epoch {epoch}: first batch {first}")

    # Batches come out in the sampler's order, whatever the number of workers.
    in_order = PyDataloader(Squares(25), batch_size=10, num_workers=2)
    print("in order:", list(in_order))

    # Leaving an epoch early is fine: the batches still being loaded are discarded in the
    # background, and the next epoch can start at once.
    for count, _batch in enumerate(loader):
        if count == 2:
            break
    iterator = iter(loader)
    next(iterator)
    print(f"stopped after 3 batches; in the next epoch, {len(iterator)} batches are left after one")


if __name__ == "__main__":
    main()
