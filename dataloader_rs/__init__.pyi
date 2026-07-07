"""
dataloader-rs — high-performance DataLoader backed by a Rust core.

Public surface
--------------
PyDataset        Base class for map-style datasets.  Subclass it and implement
                 ``__len__`` and ``__getitem__``.
PyDataloader     Main entry point.  Mirrors the ``torch.utils.data.DataLoader``
                 constructor signature for the parameters it supports.
PyDataloaderIter Iterator returned by ``PyDataloader.__iter__``.  Supports
                 ``len()`` (number of batches remaining in the current epoch).
"""

from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "PyDataset",
    "PyDataloader",
    "PyDataloaderIter",
]

class PyDataset:
    """Base class for map-style datasets.

    Subclass this and implement :meth:`__len__` and :meth:`__getitem__`.
    Instances are passed directly to :class:`PyDataloader`.

    Example
    -------
    >>> class MyDataset(PyDataset):
    ...     def __init__(self, data):
    ...         super().__init__()
    ...         self.data = data
    ...     def __len__(self) -> int:
    ...         return len(self.data)
    ...     def __getitem__(self, index: int):
    ...         return self.data[index]
    """

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        ...

    def __getitem__(self, index: int) -> Any:
        """Return the sample at *index*.

        Parameters
        ----------
        index:
            Zero-based sample index in ``[0, len(self))``.
        """
        ...

class PyDataloaderIter:
    """Iterator over one epoch of a :class:`PyDataloader`.

    Returned by :meth:`PyDataloader.__iter__`.  Do not instantiate directly.
    Supports ``len()`` — the number of batches **remaining** in the current epoch.
    """

    def __iter__(self) -> PyDataloaderIter: ...
    def __next__(self) -> Any: ...
    def __len__(self) -> int:
        """Return the number of batches not yet consumed in this epoch."""
        ...

class PyDataloader:
    """High-performance DataLoader with a PyTorch-like interface.

    Parameters
    ----------
    dataset:
        A :class:`PyDataset` instance (map-style, random-access).
    batch_size:
        How many samples to include in each batch. Default: ``1``.
    shuffle:
        If ``True`` the sample indices are shuffled at the start of every epoch.
        Mutually exclusive with *sampler*. Default: ``False``.
    sampler:
        An iterable of ``int`` indices that determines the draw order.
        Mutually exclusive with *shuffle*. Default: ``None`` (sequential).
    num_workers:
        Number of background threads used to prefetch batches in parallel.
        ``0`` (default) processes batches on the calling thread.
    collate_fn:
        A callable ``(list[sample]) -> batch`` that merges individual samples
        into a single batch object.  When ``None`` (default) each batch is a
        plain ``list`` of samples.
    drop_last:
        If ``True``, drop the last batch when the dataset size is not divisible
        by *batch_size*. Default: ``False``.
    prefetch_depth:
        Maximum number of pre-computed batches kept in the internal buffer.
        Only meaningful when ``num_workers > 0``. Default: ``1``.
    generator:
        Integer seed for the shuffle RNG, enabling reproducible ordering across
        epochs and runs.  Requires ``shuffle=True``; raises ``ValueError``
        otherwise.  When ``None`` (default) a random seed is drawn from the OS
        entropy pool.

    Example
    -------
    >>> loader = PyDataloader(dataset, batch_size=32, shuffle=True, num_workers=4)
    >>> # Reproducible shuffle — same order every run:
    >>> loader = PyDataloader(dataset, batch_size=32, shuffle=True, generator=42)
    >>> for epoch in range(10):
    ...     for batch in loader:
    ...         train(batch)
    """

    def __init__(
        self,
        dataset: PyDataset,
        batch_size: int = 1,
        prefetch_depth: int = 1,
        shuffle: bool = False,
        sampler: Iterable[int] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        drop_last: bool = False,
        generator: int | None = None,
    ) -> None: ...
    def __iter__(self) -> PyDataloaderIter:
        """Start a new epoch and return an iterator over batches."""
        ...

    def __len__(self) -> int:
        """Return the number of batches per epoch (respecting *drop_last*)."""
        ...
