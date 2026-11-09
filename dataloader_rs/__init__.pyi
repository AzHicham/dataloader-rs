"""
dataloader-rs — high-performance DataLoader backed by a Rust core.

Public surface
--------------
A dataset is a source: it only fetches items, synchronously (``PyDataset``)
or asynchronously (``PyAsyncDataset`` and the Rust datasets). Per-item work
(decoding, augmentation) is the loader's ``transform``, run on its threads.

PyDataset        Base class for map-style datasets.  Subclass it and implement
                 ``__len__`` and ``__getitem__``.
PyAsyncDataset   Base class for datasets whose ``__getitem__`` is ``async def``.
HttpDataset, FileDataset, ObjectStoreDataset
                 Datasets implemented in Rust (items are ``bytes``), fetched
                 without the interpreter.
PyDataloader     Main entry point.  Mirrors the ``torch.utils.data.DataLoader``
                 constructor signature for the parameters it supports, plus
                 ``transform``.
PyDataloaderIter Iterator returned by ``PyDataloader.__iter__``.  Supports
                 ``len()`` (number of batches remaining in the current epoch).
transforms       Image transforms implemented in Rust (``Decode``, ``PadTo``,
                 ``Normalize``, ...); a ``Compose`` of them passed as
                 ``transform=`` runs without the interpreter.
ImageCollator    ``collate_fn`` for their images: one ``ImageBatch`` buffer,
                 viewed by numpy and torch without a copy.
"""

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, Literal

from . import transforms as transforms

__all__ = [
    "FileDataset",
    "HttpDataset",
    "Image",
    "ImageBatch",
    "ImageCollator",
    "ObjectStoreDataset",
    "PyAsyncDataset",
    "PyDataset",
    "PyDataloader",
    "PyDataloaderIter",
    "RustDataset",
    "transforms",
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

    # Optional: define ``__getitems__(self, indices: list[int]) -> Sequence[Any]``
    # to fetch a whole batch in one call. When present, the loader calls it once
    # per batch instead of ``__getitem__`` per index; it must return one sample
    # per index, in order. Setting it to ``None`` disables it, as in PyTorch.
    # It is not declared here so that ``hasattr`` stays ``False`` for datasets
    # that do not implement it.

class PyAsyncDataset:
    """Base class for map-style datasets fetched by coroutines.

    Subclass it and implement :meth:`__len__` and ``async def __getitem__``
    (optionally ``async def __getitems__(indices)``, used when
    ``item_concurrency=1``). The loader runs the coroutines on an asyncio event
    loop thread of its own, with up to ``num_workers * item_concurrency`` of
    them in flight; dropping an iterator cancels the coroutines in flight.

    A subclass may also define ``async def aclose(self)`` to release what is
    bound to that loop (an ``aiohttp`` session, pooled connections). When the
    loader is dropped, it awaits ``aclose()`` once on its loop, then stops the
    loop. An error, or ``aclose()`` still running after 5 seconds, is logged
    to the ``dataloader_rs`` logger and the loop stops anyway. A plain ``def
    aclose`` is a ``TypeError`` when the loader is built.

    Example
    -------
    >>> class Tiles(PyAsyncDataset):
    ...     def __init__(self, session, urls):
    ...         super().__init__()
    ...         self.session, self.urls = session, urls
    ...     def __len__(self) -> int:
    ...         return len(self.urls)
    ...     async def __getitem__(self, index: int):
    ...         async with self.session.get(self.urls[index]) as response:
    ...             return await response.read()
    ...     async def aclose(self):
    ...         await self.session.close()
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Awaitable[Any]: ...

class RustDataset:
    """A dataset implemented in Rust: :class:`HttpDataset`, :class:`FileDataset`
    or :class:`ObjectStoreDataset`.

    Items are fetched on the loader's async runtime without the interpreter, and
    reach Python as ``bytes`` (or what the loader's ``transform`` makes of them).
    """

    def __len__(self) -> int: ...

class HttpDataset(RustDataset):
    """One HTTP GET per item: the body of the response to ``urls[i]``.

    One shared client keeps connections alive. Each attempt must complete within
    ``timeout`` seconds (body included). Connection errors, timeouts and statuses
    408, 429, 500, 502, 503, 504 are retried up to ``retries`` times, waiting
    ``backoff * 2**n`` seconds (with jitter, at most ``max_backoff``; a
    ``Retry-After`` is honoured up to it). Other statuses fail at once with
    ``OSError``; a body over ``max_bytes`` raises ``ValueError``. TLS uses the
    bundled Mozilla root certificates.
    """

    def __init__(
        self,
        urls: Iterable[str],
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        retries: int = 3,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
    ) -> None: ...
    @staticmethod
    def from_template(
        template: str,
        columns: Mapping[str, Sequence[int] | Sequence[str] | Any] | None = None,
        *,
        length: int | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        retries: int = 3,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
    ) -> HttpDataset:
        """A dataset whose URLs are made in Rust, item by item, from ``template``
        and one value per item in each of ``columns``; no URL string is kept per
        item.

        ``{name}`` is the value of column ``name`` for the item, ``{index}`` the
        index of the item, ``{{`` and ``}}`` literal braces. A column is a sequence
        (or numpy array of any integer dtype) of ints, written in decimal, or of
        strs, percent-encoded as a URL component (``.`` and ``..`` are refused).
        The number of items is the length of the columns; ``length`` is required
        when there is no column, and must match them otherwise.

        Raises ``ValueError`` for a placeholder without a column, a column the
        template does not use, unequal lengths, unbalanced braces, an empty
        template or an invalid URL for the first item (an invalid URL for another
        item fails that item), and ``TypeError`` for a column that is not ints or
        strs. Other options are those of :class:`HttpDataset`.
        """
        ...

class FileDataset(RustDataset):
    """One local file per item: the content of ``paths[i]``. A file over
    ``max_bytes`` raises ``ValueError``; a missing one ``OSError``.

    By default a file is read once, for as long as the file system takes. For a
    network file system, ``timeout`` (seconds) bounds each attempt, and attempts
    that time out or fail with ``TimedOut``, ``Interrupted`` or ``WouldBlock``
    I/O errors are retried up to ``retries`` times (default 0), waiting
    ``backoff * 2**n`` seconds (with jitter, at most ``max_backoff``).
    """

    def __init__(
        self,
        paths: Iterable[str | PathLike[str]],
        *,
        timeout: float | None = None,
        retries: int = 0,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
    ) -> None: ...
    @staticmethod
    def from_template(
        template: str,
        columns: Mapping[str, Sequence[int] | Sequence[str] | Any] | None = None,
        *,
        length: int | None = None,
        timeout: float | None = None,
        retries: int = 0,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
    ) -> FileDataset:
        """A dataset whose paths are made in Rust, item by item, from
        ``template`` and one value per item in each of ``columns``, as
        :meth:`HttpDataset.from_template` makes URLs.

        Strs are written as they are, but a value holding ``/``, ``\\`` or NUL,
        or equal to ``.`` or ``..``, raises ``ValueError``, and an item whose
        values make a ``.`` or ``..`` path component with the text around them
        fails: a value cannot leave the template's directory. Other options are
        those of :class:`FileDataset`.
        """
        ...

class ObjectStoreDataset(RustDataset):
    """One object per item: key ``keys[i]`` under the store and prefix of ``url``
    (``s3://bucket/prefix``, ``gs://``, ``az://``, ``https://``).

    Credentials and settings come from the environment (``AWS_*``, ``GOOGLE_*``,
    ``AZURE_*``), then from ``options`` (``object_store`` keys such as
    ``aws_region`` or ``allow_http``; an unknown key raises ``ValueError``, and
    so does ``timeout``: use the ``timeout`` argument). A ``url`` with a
    password, a query or a fragment raises ``ValueError``: credentials go in
    ``options``.

    The store retries a request itself: 5xx statuses, 429, 408, connection
    errors and timeouts, up to ``retries`` times, waiting about ``backoff *
    2**n`` seconds (with jitter, at most ``max_backoff``), within 3 minutes of
    the first attempt. Each request must complete within ``timeout`` seconds
    (``None``: the store's default, 30 s).
    """

    def __init__(
        self,
        url: str,
        keys: Iterable[str],
        *,
        options: Mapping[str, str] | None = None,
        timeout: float | None = None,
        retries: int = 3,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
    ) -> None: ...
    @staticmethod
    def from_template(
        url: str,
        template: str,
        columns: Mapping[str, Sequence[int] | Sequence[str] | Any] | None = None,
        *,
        length: int | None = None,
        options: Mapping[str, str] | None = None,
        timeout: float | None = None,
        retries: int = 3,
        backoff: float = 0.1,
        max_backoff: float = 5.0,
        max_bytes: int = 64 << 20,
    ) -> ObjectStoreDataset:
        """A dataset whose keys, relative to the prefix of ``url``, are made in
        Rust, item by item, from ``template`` and one value per item in each of
        ``columns``, as :meth:`HttpDataset.from_template` makes URLs.

        Strs are written as they are, but a value must be a valid key part (not
        empty, ``.`` or ``..``, without ``/`` or control characters), else
        ``ValueError``. Other options are those of :class:`ObjectStoreDataset`.
        """
        ...

class Image:
    """One image from :mod:`dataloader_rs.transforms`, in Rust's buffer.

    ``np.asarray(image)`` is a read-only view of it, without a copy;
    :class:`ImageCollator` batches images.
    """

    @staticmethod
    def from_array(obj: Any, layout: Literal["hwc", "chw", "HWC", "CHW"] = "hwc") -> Image:
        """An image of ``obj``, back into Rust after a numpy or torch step.

        ``obj`` is a numpy array, a CPU torch tensor or anything ``np.asarray``
        takes (without numpy: an object with the buffer protocol and its own
        shape) of ``uint8`` or ``float32`` values; other types are a
        ``TypeError``, never converted. Its shape is ``(height, width)`` or,
        in ``layout``, ``(height, width, channels)`` (``"hwc"``, default) or
        ``(channels, height, width)`` (``"chw"``, torch's), with 1, 3 or 4
        channels and ``Decode``'s limits (16384 x 16384, 256 MiB); else
        ``ValueError``. A GPU tensor is a ``TypeError``: move it first
        (``tensor.cpu()``).

        The image is HWC, so every transform and :class:`ImageCollator` take
        it. A contiguous HWC array is copied once; a CHW or strided one is
        made contiguous by numpy first (one more copy).
        """
        ...
    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height, width, channels)`` or ``(channels, height, width)``."""
        ...
    @property
    def dtype(self) -> Literal["uint8", "float32"]: ...
    @property
    def layout(self) -> Literal["HWC", "CHW"]: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    def __dlpack__(self, **kwargs: Any) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class ImageBatch:
    """A batch from :class:`ImageCollator`: images of one shape in one buffer.

    ``np.asarray(batch)`` and ``torch.from_dlpack(batch)`` are writable views
    of it, without a copy; the buffer lives as long as the batch or any view.
    """

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(batch, channels, height, width)`` or ``(batch, height, width, channels)``."""
        ...
    @property
    def dtype(self) -> Literal["uint8", "float32"]: ...
    @property
    def layout(self) -> Literal["NCHW", "NHWC"]: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...
    def __dlpack__(self, **kwargs: Any) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class ImageCollator:
    """``collate_fn`` for :class:`Image` items: one :class:`ImageBatch`.

    As a loader's ``collate_fn`` with a :mod:`dataloader_rs.transforms`
    ``transform``, batches are collated in Rust, without the interpreter.
    Items may also be tuples ``(Image, a, ...)`` (from a Python transform):
    the batch is then ``(ImageBatch, [a, ...], ...)``.

    Parameters
    ----------
    layout:
        ``"NCHW"`` (default) or ``"NHWC"``; items of either layout are
        transposed as needed.
    dtype:
        ``"uint8"`` or ``"float32"``: the type items must have (``ValueError``
        otherwise). Default: the items' own.
    buffers:
        Keep the buffers of up to this many dropped batches and write the
        next batches into them: no page faults on fresh memory (a 64 x 3 x
        224 x 224 float32 batch collates about 4x faster). Never waits for a
        buffer. Default: ``0``.
    """

    def __init__(
        self,
        layout: Literal["NCHW", "NHWC"] = "NCHW",
        dtype: Literal["uint8", "float32"] | None = None,
        buffers: int = 0,
    ) -> None: ...
    def __call__(self, items: list[Any]) -> Any: ...

class PyDataloaderIter:
    """Iterator over one epoch of a :class:`PyDataloader`.

    Returned by :meth:`PyDataloader.__iter__`.  Do not instantiate directly.
    Supports ``len()`` — the number of batches **remaining** in the current epoch.
    """

    def __iter__(self) -> PyDataloaderIter: ...
    def __next__(self) -> Any:
        """The next batch; ``(batch, indices)`` with ``return_indices=True``."""
        ...
    def __len__(self) -> int:
        """Return the number of batches not yet consumed in this epoch."""
        ...

class PyDataloader:
    """High-performance DataLoader with a PyTorch-like interface.

    Parameters
    ----------
    dataset:
        The source of the items: a :class:`PyDataset`, a :class:`PyAsyncDataset`
        or a Rust dataset (:class:`HttpDataset`, :class:`FileDataset`,
        :class:`ObjectStoreDataset`). Async and Rust datasets always load
        ``num_workers`` (at least one) batches at once.
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
        (after *transform*) into a single batch object, or an
        :class:`ImageCollator` (collates in Rust).  When ``None``
        (default) each batch is a plain ``list`` of samples.
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
    item_concurrency:
        Number of items of one batch fetched in parallel, each with its own
        ``__getitem__`` call (``__getitems__`` is then not used). For I/O-bound
        datasets (network, object storage) whose items would otherwise be
        fetched one after another. At most ``max(num_workers, 1) *
        item_concurrency`` items are in flight. Default: ``1``.
    items_in_parallel:
        Deprecated name of *item_concurrency*; emits ``DeprecationWarning``.
    runtime_threads:
        Async and Rust datasets only: worker threads of the loader's tokio
        runtime, which receives responses, runs TLS and copies bodies. Between
        ``1`` and ``1024``; one thread can drive many requests in flight.
        Default: ``None``, one per CPU available to the process (affinity and
        cgroup quota included). ``ValueError`` with a :class:`PyDataset`.
    transform:
        A callable ``(index, sample) -> sample`` applied to every item once
        fetched, before *collate_fn*: decoding, augmentation. ``index`` is the
        item's dataset index, as the sampler drew it; wrap a one-argument
        callable such as a torchvision ``Compose``: ``lambda i, x: compose(x)``.
        It runs on the loader's threads (on the calling thread
        for a :class:`PyDataset` with ``num_workers=0`` and
        ``item_concurrency=1``), holding the
        interpreter only for the call; items of a batch are transformed in
        parallel with ``item_concurrency > 1``, and as they arrive for async
        and Rust datasets. An exception it raises fails its batch and keeps
        its type. A :mod:`dataloader_rs.transforms` ``Transform`` (or
        ``Compose``) instead runs in Rust, without the interpreter, on
        ``bytes`` items. Default: ``None``, items as fetched.
    return_indices:
        If ``True``, each batch comes as the tuple ``(batch, indices)``:
        ``indices`` is a ``list`` of the dataset indices of the batch's items,
        in the order the sampler drew them (``indices[k]`` is the index of the
        batch's ``k``-th item), so ``coords[indices]`` gives their coordinates
        with NumPy. Works with every dataset and *collate_fn*; the list is made
        once per batch, when it is returned. Default: ``False``.

    Example
    -------
    >>> loader = PyDataloader(dataset, batch_size=32, shuffle=True, num_workers=4)
    >>> for epoch in range(10):
    ...     for batch in loader:
    ...         train(batch)

    Tiles fetched, decoded and batched in Rust, with the indices of each batch:

    >>> tiles = HttpDataset.from_template(
    ...     "https://images.example/slide/tile?x={x}&y={y}&size=224", {"x": xs, "y": ys}
    ... )
    >>> coords = np.stack([xs, ys], axis=1)
    >>> loader = PyDataloader(
    ...     tiles,
    ...     batch_size=64,
    ...     num_workers=4,
    ...     item_concurrency=16,
    ...     transform=transforms.Compose([transforms.Decode(), transforms.ToRgb()]),
    ...     collate_fn=ImageCollator(),
    ...     return_indices=True,
    ... )
    >>> for images, indices in loader:
    ...     predict(images, coords[indices])
    """

    def __init__(
        self,
        dataset: PyDataset | PyAsyncDataset | RustDataset,
        batch_size: int = 1,
        prefetch_depth: int = 2,
        shuffle: bool = False,
        sampler: Iterable[int] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[Any]], Any] | ImageCollator | None = None,
        drop_last: bool = False,
        seed: int | None = None,
        item_concurrency: int = 1,
        items_in_parallel: int | None = None,
        runtime_threads: int | None = None,
        transform: Callable[[int, Any], Any] | transforms.Transform | None = None,
        return_indices: bool = False,
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
