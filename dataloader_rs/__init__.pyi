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
ImageCollator    ``collate_fn`` that decodes, transforms and collates encoded
                 images in Rust, without the GIL; batches are ``ImageBatch``.
ImageBatch       A batch in a pooled buffer: ``numpy.asarray(batch)`` and
                 ``torch.from_dlpack(batch)`` view it without copying.
Resize, Crop, Pad, ToFloat, Normalize
                 The operations of an ``ImageCollator``.
Source           A ``PyDataset`` whose items Rust fetches itself: bytes from
                 memory, files or HTTP, with per-item metadata.
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

__all__ = [
    "PyDataset",
    "PyDataloader",
    "PyDataloaderIter",
    "ImageCollator",
    "ImageBatch",
    "Resize",
    "Crop",
    "Pad",
    "ToFloat",
    "Normalize",
    "Source",
]

ResizeFilter = Literal["nearest", "box", "bilinear", "hamming", "bicubic", "mitchell", "lanczos3"]

class Resize:
    """Resize to ``width x height`` (fast_image_resize convolution; antialiased).

    ``filter`` defaults to ``"bilinear"`` (Pillow's ``BILINEAR``, torchvision's
    default with ``antialias=True``); ``"lanczos"`` is accepted for ``"lanczos3"``.
    """

    width: int
    height: int
    filter: ResizeFilter
    def __init__(self, width: int, height: int, filter: ResizeFilter = "bilinear") -> None: ...

class Crop:
    """Keep the ``width x height`` region whose top-left pixel is (``left``, ``top``).

    A region that leaves the image is an error for its batch.
    """

    left: int
    top: int
    width: int
    height: int
    def __init__(self, left: int, top: int, width: int, height: int) -> None: ...

class Pad:
    """Grow the image to ``width x height``, image at the top left, ``fill`` elsewhere.

    An image larger than the target is an error for its batch.
    """

    width: int
    height: int
    fill: int
    def __init__(self, width: int, height: int, fill: int = 0) -> None: ...

class ToFloat:
    """Divide by 255: values in ``[0, 1]``, as torchvision's ``ToTensor``."""

    def __init__(self) -> None: ...

class Normalize:
    """``(x - mean[c]) / std[c]`` per channel, as torchvision's ``Normalize``.

    One value applies to every channel.
    """

    mean: list[float]
    std: list[float]
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None: ...

ImageOp = Resize | Crop | Pad | ToFloat | Normalize

class ImageCollator:
    """Decode, transform and collate encoded images in Rust.

    Pass it as ``collate_fn`` to :class:`PyDataloader`. The dataset is a
    ``list`` of ``bytes`` (or of ``(bytes, meta)`` pairs), read without any
    Python per item, or a :class:`PyDataset` whose ``__getitem__`` returns
    ``bytes`` or ``(bytes, meta)``. PNG, JPEG and WebP are detected from the
    bytes. Decoding, the operations and collation run on the loader's threads
    with the GIL released (on GIL builds too); Python only runs to fetch the
    bytes. Every batch is an :class:`ImageBatch`.

    Parameters
    ----------
    ops:
        Operations in order. Geometric ones (``Resize``, ``Crop``, ``Pad``)
        come before ``ToFloat`` and ``Normalize``. The sample size must be
        known: end the geometric operations with a resize, crop or pad, or
        set *input_size*.
    channels:
        ``"rgb"`` (gray replicated, alpha dropped, as Pillow's
        ``convert("RGB")``) or ``"gray"`` (Pillow's ``convert("L")``).
    dtype:
        ``"float32"``, ``"float16"`` or ``"uint8"`` (no value operation).
    layout:
        ``"nchw"`` (PyTorch) or ``"nhwc"``.
    input_size:
        ``(width, height)`` every image must have; another size fails its batch.
    max_width, max_height, max_bytes:
        Bounds checked against the image header before decoding, and against
        every size of the pipeline.

    Raises
    ------
    ValueError
        If the pipeline is invalid; the message names the operation.
    TypeError
        If an element of *ops* is not an operation.
    """

    def __init__(
        self,
        ops: Sequence[ImageOp] = (),
        *,
        channels: Literal["rgb", "gray"] = "rgb",
        dtype: Literal["float32", "float16", "uint8"] = "float32",
        layout: Literal["nchw", "nhwc"] = "nchw",
        input_size: tuple[int, int] | None = None,
        max_width: int = 16384,
        max_height: int = 16384,
        max_bytes: int = 268435456,
    ) -> None: ...
    @property
    def sample_shape(self) -> tuple[int, int, int]:
        """``(C, H, W)`` for ``"nchw"``, ``(H, W, C)`` for ``"nhwc"``."""
        ...

    @property
    def dtype(self) -> str:
        """``"float32"``, ``"float16"`` or ``"uint8"``."""
        ...

class ImageBatch:
    """A batch an :class:`ImageCollator` wrote into a buffer of the loader's pool.

    ``numpy.asarray(batch)`` (``__array_interface__``) and
    ``torch.from_dlpack(batch)`` are views of the buffer: no copy, writable.
    A view keeps the batch alive; the buffer returns to the pool once the
    batch and all its views are gone. ``PyDataloader`` raises
    ``RuntimeError`` rather than wait when *max_held_batches* batches are
    still alive: copy what you keep (``numpy.array(batch)``).
    """

    def __len__(self) -> int:
        """Samples in this batch (fewer than ``batch_size`` for a last, partial one)."""
        ...

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(n, C, H, W)`` or ``(n, H, W, C)``."""
        ...

    @property
    def dtype(self) -> str: ...
    @property
    def meta(self) -> list[Any]:
        """The ``meta`` of ``(bytes, meta)`` samples, else ``None``, in batch order."""
        ...

    @property
    def sizes(self) -> list[tuple[int, int]]:
        """The decoded ``(width, height)`` of each sample, before any operation."""
        ...

    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    def numpy(self) -> Any:
        """``numpy.asarray(self)``: a ``numpy.ndarray`` view, no copy (needs NumPy)."""
        ...

    def __dlpack__(self, **kwargs: Any) -> Any:
        """DLPack capsule (through NumPy's export) viewing the buffer."""
        ...

    def __dlpack_device__(self) -> tuple[int, int]: ...

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

    # Optional: define ``__getitems__(self, indices: list[int]) -> Sequence[Any]``
    # to fetch a whole batch in one call. When present, the loader calls it once
    # per batch instead of ``__getitem__`` per index; it must return one sample
    # per index, in order. Setting it to ``None`` disables it, as in PyTorch.
    # It is not declared here so that ``hasattr`` stays ``False`` for datasets
    # that do not implement it.

class Source(PyDataset):
    """A dataset of encoded items that Rust fetches itself, each with optional metadata.

    Item ``i`` is ``(bytes, meta[i])`` (``meta[i]`` is ``None`` without ``meta``). With
    ``collate_fn=ImageCollator(...)``, ``PyDataloader`` fetches, decodes, transforms and
    collates every item on its worker threads without the GIL, and ``ImageBatch.meta`` holds
    the metadata. As a plain ``PyDataset``, ``source[i]`` fetches item ``i`` with the GIL
    released.

    Build one with :meth:`from_bytes`, :meth:`from_files`, :meth:`from_urls` or
    :meth:`from_template`; invalid settings raise ``ValueError``.

    Errors of an item, in ``source[i]`` or in the loader (failing that batch only):
    ``TimeoutError``, ``FileNotFoundError``, ``PermissionError``, else ``OSError``; messages
    name the item and its path or URL (without user info, query or fragment).

    HTTP: blocking HTTP/1.1, connections kept alive and reused. ``timeout`` (seconds) bounds
    one attempt from connecting to the last byte. Connection errors, timeouts, truncated
    bodies and statuses 408, 429, 500, 502, 503, 504 are retried up to ``max_retries`` times,
    waiting ``backoff * 2**n`` seconds (at most ``max_backoff``, half of it random, and a
    ``Retry-After`` honoured up to ``max_backoff``); other statuses fail at once. A body above
    ``max_bytes`` fails. Header values are never shown. This build fetches ``http://`` URLs
    only: ``https://`` raises ``ValueError``.
    """

    @staticmethod
    def from_bytes(items: Sequence[bytes | bytearray], meta: Sequence[Any] | None = None) -> Source:
        """Items already in memory, read once (``bytearray`` is copied)."""
        ...

    @staticmethod
    def from_files(
        paths: Sequence[str | os.PathLike[str]],
        meta: Sequence[Any] | None = None,
        *,
        max_bytes: int = 64 << 20,
    ) -> Source:
        """One file per item, read whole; a file above ``max_bytes`` fails its item."""
        ...

    @staticmethod
    def from_urls(
        urls: Sequence[str],
        meta: Sequence[Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
        max_redirects: int = 5,
        max_idle_connections: int = 64,
    ) -> Source:
        """One HTTP ``GET`` per URL."""
        ...

    @staticmethod
    def from_template(
        template: str,
        params: Sequence[dict[str, Any]],
        meta: Sequence[Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
        max_redirects: int = 5,
        max_idle_connections: int = 64,
    ) -> Source:
        """One HTTP ``GET`` per entry of ``params``: ``template`` with each ``{name}``
        replaced by ``str(params[i][name])``, percent-encoded (``{{`` / ``}}`` are literal
        braces). Each entry gives every name of the template once, and no other."""
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[bytes, Any]:
        """``(bytes, meta)`` of item ``index``, fetched with the GIL released."""
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
        Number of batches processed in parallel on the loader's thread pool
        (``num_workers`` threads, created once and reused across epochs).
        ``0`` (default) processes batches on the calling thread.
    collate_fn:
        A callable ``(list[sample]) -> batch`` that merges individual samples
        into a single batch object.  When ``None`` (default) each batch is a
        plain ``list`` of samples. An :class:`ImageCollator` instead decodes
        and collates encoded images in Rust, and *dataset* may then also be a
        ``list`` of ``bytes``; batches are :class:`ImageBatch`.
    drop_last:
        If ``True``, drop the last batch when the dataset size is not divisible
        by *batch_size*. Default: ``False``.
    prefetch_depth:
        How many batches workers may run ahead of the consumer: at most
        ``num_workers + prefetch_depth`` batches are in flight or buffered.
        Only meaningful when ``num_workers > 0``. Default: ``2``.
    seed:
        Seed for ``shuffle=True``. Two loaders built with the same seed yield
        the same sequence of epoch orders; each epoch still differs from the
        previous one. Requires ``shuffle=True``. Default: ``None`` (seeded
        from OS entropy, not reproducible).
    items_in_parallel:
        Number of items of one batch fetched in parallel, each with its own
        ``__getitem__`` call (``__getitems__`` is then not used). For I/O-bound
        datasets (network, object storage) whose items would otherwise be
        fetched one after another. At most ``max(num_workers, 1) *
        items_in_parallel`` items are in flight. Default: ``1``.

    Example
    -------
    >>> loader = PyDataloader(dataset, batch_size=32, shuffle=True, num_workers=4)
    >>> for epoch in range(10):
    ...     for batch in loader:
    ...         train(batch)
    """

    def __init__(
        self,
        dataset: PyDataset | Sequence[bytes | tuple[bytes, Any]],
        batch_size: int = 1,
        prefetch_depth: int = 2,
        shuffle: bool = False,
        sampler: Iterable[int] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[Any]], Any] | ImageCollator | None = None,
        drop_last: bool = False,
        seed: int | None = None,
        items_in_parallel: int = 1,
        max_held_batches: int = 2,
    ) -> None: ...
    def __iter__(self) -> PyDataloaderIter:
        """Start a new epoch and return an iterator over batches."""
        ...

    def __len__(self) -> int:
        """Return the number of batches per epoch (respecting *drop_last*).

        With a custom *sampler* this is based on ``len(sampler)``: a sampler
        without ``__len__`` raises :class:`TypeError`, and an exception raised
        by its ``__len__`` propagates.
        """
        ...
