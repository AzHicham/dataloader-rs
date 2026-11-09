import atexit

from . import transforms
from .dataloader_rs import (
    FileDataset,
    HttpDataset,
    Image,
    ImageBatch,
    ImageCollator,
    ObjectStoreDataset,
    PyDataloader,
    PyDataloaderIter,
    RustDataset,
    _keep_pools_alive,
)
from .dataloader_rs import PyAsyncDatasetBase as PyAsyncDataset
from .dataloader_rs import PyDatasetBase as PyDataset

# Keep the loaders' thread pools alive while the interpreter shuts down: a
# pool freed then would wake threads into a finalizing interpreter, which
# aborts the process on CPython 3.11 to 3.13 (see src/python/pool.rs).
atexit.register(_keep_pools_alive)

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
