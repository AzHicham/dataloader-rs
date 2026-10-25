import atexit

from .dataloader_rs import PyDataloader, PyDataloaderIter, _keep_pools_alive
from .dataloader_rs import PyDatasetBase as PyDataset

# Keep the loaders' thread pools alive while the interpreter shuts down: a
# pool freed then would wake threads into a finalizing interpreter, which
# aborts the process on CPython 3.13 (see src/python/pool.rs).
atexit.register(_keep_pools_alive)

__all__ = [
    "PyDataset",
    "PyDataloader",
    "PyDataloaderIter",
]
