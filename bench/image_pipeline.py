#!/usr/bin/env python3
"""
Decode, transform and collate images: dataloader_rs's ImageCollator against
torch.utils.data.DataLoader with Pillow and torchvision transforms.

Every batch is 64 lossless WebP images of 224x224 (64 distinct ones, cycled),
decoded to RGB, then ToTensor + Normalize (ImageNet) into a float32
64x3x224x224 batch. The outputs of both loaders are compared once, then
throughput is measured over whole epochs:

  rs-list      PyDataloader(list of bytes, collate_fn=ImageCollator(...)):
               no Python runs per item; decode, transforms and collation run
               in Rust with the GIL released
  rs-dataset   the same with a PyDataset whose __getitem__ returns bytes:
               Python fetches the bytes, Rust does the rest without the GIL
  torch        torch DataLoader, Dataset doing Image.open(...).convert("RGB")
               + ToTensor + Normalize, default collation, persistent workers
               (worker processes)

Reported: items per second (median of --repeats epochs, after one warm-up
epoch), per loader and num_workers.

Needs numpy, Pillow, torch and torchvision. Pin the cores to compare runs:
  taskset -c 12-19 python bench/image_pipeline.py --workers 0 2 4 8
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import sysconfig
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataloader_rs import ImageCollator, Normalize, PyDataloader, PyDataset, ToFloat

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIDE = 224
BATCH = 64


def scene(seed: int) -> np.ndarray:
    """Gradients, a checkerboard and noise: not trivially compressible."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    img = np.stack([255 * x / SIDE, 255 * y / SIDE, 127.5 + 127.5 * np.sin((x + seed) / 9)], -1)
    img[(x // 8 + y // 8) % 2 == 0] *= 0.6
    img += rng.normal(0, 12, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def webp_images(n: int) -> list[bytes]:
    out = []
    for seed in range(n):
        buf = io.BytesIO()
        Image.fromarray(scene(seed), "RGB").save(buf, format="WEBP", lossless=True)
        out.append(buf.getvalue())
    return out


class BytesDataset(PyDataset):
    def __init__(self, images: list[bytes], length: int):
        super().__init__()
        self.images = images
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> bytes:
        return self.images[index % len(self.images)]


class TorchImages(Dataset):
    def __init__(self, images: list[bytes], length: int):
        self.images = images
        self.length = length
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(io.BytesIO(self.images[index % len(self.images)])).convert("RGB")
        return self.transform(img)


def collator() -> ImageCollator:
    return ImageCollator([ToFloat(), Normalize(MEAN, STD)], input_size=(SIDE, SIDE))


def make_loader(kind: str, images: list[bytes], length: int, workers: int):
    if kind == "rs-list":
        data = [images[i % len(images)] for i in range(length)]
        return PyDataloader(data, batch_size=BATCH, num_workers=workers, collate_fn=collator())
    if kind == "rs-dataset":
        return PyDataloader(
            BytesDataset(images, length),
            batch_size=BATCH,
            num_workers=workers,
            collate_fn=collator(),
        )
    return DataLoader(
        TorchImages(images, length),
        batch_size=BATCH,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def epoch_rate(loader, length: int) -> float:
    start = time.perf_counter()
    n = 0
    for batch in loader:
        n += len(batch)
    assert n == length, (n, length)
    return length / (time.perf_counter() - start)


def check_outputs(images: list[bytes]) -> float:
    ours = np.array(next(iter(make_loader("rs-list", images, BATCH, 0))))
    theirs = next(iter(make_loader("torch", images, BATCH, 0))).numpy()
    return float(np.abs(ours - theirs).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--batches", type=int, default=8, help="batches per epoch")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loaders", nargs="+", default=["rs-list", "rs-dataset", "torch"])
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()

    torch.set_num_threads(1)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    build = f"{sys.version.split()[0]}{'' if gil else 't'}"
    print(f"Python {build} ({'GIL' if gil else 'free-threaded'}), {sysconfig.get_platform()}")
    images = webp_images(64)
    size = sum(map(len, images)) / len(images) / 1024
    print(f"64 lossless WebP of {SIDE}x{SIDE}, {size:.0f} KiB each on average")
    print(f"max |rs - torchvision| on one batch: {check_outputs(images):.3g}")

    length = BATCH * args.batches
    results = []
    print(f"\n{'loader':<11} {'workers':>7} {'items/s':>9}")
    for kind in args.loaders:
        for workers in args.workers:
            loader = make_loader(kind, images, length, workers)
            epoch_rate(loader, length)  # warm-up: worker start, buffer pool, caches
            rate = statistics.median(epoch_rate(loader, length) for _ in range(args.repeats))
            results.append(
                {"python": build, "loader": kind, "workers": workers, "items_per_s": rate}
            )
            print(f"{kind:<11} {workers:>7} {rate:>9.0f}", flush=True)
            del loader

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
