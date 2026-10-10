"""Real-world workloads shared by the real-world benchmarks.

Each workload is the per-sample work a real dataset does — reading files,
augmenting images, waiting on the network, running Python code — written
once as a mixin and combined with both ``dataloader_rs.PyDataset`` and
``torch.utils.data.Dataset``, so both loaders run identical work and only
the loading (scheduling, parallelism, prefetching, collation) differs.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from dataloader_rs import PyDataset

# ── local_files: read a file per sample, decode bytes to a normalised array ───

FILE_BYTES = 128 * 1024


def make_file_tree(n: int, root: Path | None = None) -> Path:
    """Write *n* random 128 KB files (once) and return their directory."""
    root = root or Path(tempfile.gettempdir()) / f"dataloader_rs_bench_{n}"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        path = root / f"{i:06d}.bin"
        if not path.exists() or path.stat().st_size != FILE_BYTES:
            path.write_bytes(rng.integers(0, 256, FILE_BYTES, dtype=np.uint8).tobytes())
    return root


class LocalFilesWork:
    """Read one file per sample and normalise it (I/O + numpy, GIL released)."""

    def __init__(self, root: Path, n: int):
        self.paths = [str(root / f"{i:06d}.bin") for i in range(n)]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        with open(self.paths[index], "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        return (raw.astype(np.float32) / 255.0 - 0.5) / 0.25


# ── image_augment: torchvision-style transforms on an already-decoded image ──

IMAGE_SHAPE = (3, 256, 256)
CROP = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class ImageAugmentWork:
    """Random crop + horizontal flip + float conversion + normalisation."""

    def __init__(self, n: int):
        rng = np.random.default_rng(0)
        # A pool of decoded images, reused cyclically to bound memory.
        self.images = rng.integers(0, 256, (64, *IMAGE_SHAPE), dtype=np.uint8)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(index)
        image = self.images[index % len(self.images)]
        top, left = rng.integers(0, IMAGE_SHAPE[1] - CROP + 1, size=2)
        crop = image[:, top : top + CROP, left : left + CROP]
        if rng.random() < 0.5:
            crop = crop[:, :, ::-1]
        return (crop.astype(np.float32) / 255.0 - _MEAN) / _STD


# ── remote: each sample waits on a (simulated) network round trip ────────────

LATENCY_S = 0.010


class RemoteWork:
    """10 ms per sample, like an object-store GET; sleeping releases the GIL."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> bytes:
        time.sleep(LATENCY_S)
        return index.to_bytes(8, "little") * 128


class AsyncRemoteWork(RemoteWork):
    """The same round trip with an async client (dataloader_rs only)."""

    async def __getitem__(self, index: int) -> bytes:  # type: ignore[override]
        await asyncio.sleep(LATENCY_S)
        return index.to_bytes(8, "little") * 128


# ── python_cpu: pure-Python preprocessing that holds the GIL ─────────────────

_WORDS = ("the quick brown fox jumps over the lazy dog " * 40).split()
_VOCAB = {word: i for i, word in enumerate(sorted(set(_WORDS)))}


class PythonCpuWork:
    """Tokenise-and-pad in pure Python: threads cannot run this in parallel."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> list[int]:
        ids = []
        for _ in range(3):
            ids = [_VOCAB[word] + index % 7 for word in _WORDS]
        return ids[:256] + [0] * max(0, 256 - len(ids))


# ── Loader-specific dataset classes ──────────────────────────────────────────


class OursLocalFiles(LocalFilesWork, PyDataset):
    pass


class OursImageAugment(ImageAugmentWork, PyDataset):
    pass


class OursRemote(RemoteWork, PyDataset):
    pass


class OursAsyncRemote(AsyncRemoteWork, PyDataset):
    pass


class OursPythonCpu(PythonCpuWork, PyDataset):
    pass


def torch_datasets():
    """Torch counterparts, built lazily so torch stays optional."""
    from torch.utils.data import Dataset

    class TorchLocalFiles(LocalFilesWork, Dataset):
        pass

    class TorchImageAugment(ImageAugmentWork, Dataset):
        pass

    class TorchRemote(RemoteWork, Dataset):
        pass

    class TorchPythonCpu(PythonCpuWork, Dataset):
        pass

    return {
        "local_files": TorchLocalFiles,
        "image_augment": TorchImageAugment,
        "remote": TorchRemote,
        "python_cpu": TorchPythonCpu,
    }


def stack_collate(items):
    """Stack arrays into one batch array; other samples stay a list."""
    if isinstance(items[0], np.ndarray):
        return np.stack(items)
    return items


def cpu_count() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
