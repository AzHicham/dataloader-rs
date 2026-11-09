#!/usr/bin/env python3
"""
Image tiles from local files: torch.utils.data.DataLoader with Pillow and
torchvision against dataloader_rs with its Rust transforms and ImageCollator.

The transform is msintuit-ec's feature pass (prime_dx): PNG decode, RGB,
ConstantFill2d to 224 x 224, ToTensor, Normalize(ImageNet); batches are
64 x 3 x 224 x 224 float32. The tiles are 224 x 224 RGB PNGs of gradients and
noise, every eighth one 224 x 150 (an undersized edge tile), written once to
--dir. An epoch is --items tiles. Each measurement builds a fresh loader, so
worker start-up counts, then reads one epoch and turns every batch into a
torch tensor (torch.from_dlpack for dataloader_rs: no copy):

  epoch   loader creation + one epoch, seconds   (median, min-max over rounds)
  cpu     CPU seconds of the process and its worker processes   (median)

Loaders:
  torch_w4 / torch_w8   torch, 4 / 8 worker processes, Pillow + torchvision
  dlrs_pil_w4_c4        FileDataset, the same Pillow + torchvision code as a
                        Python transform (holds the interpreter per item)
  dlrs_w4_c4            FileDataset + transforms.Compose (Rust, no interpreter)
                        + ImageCollator(buffers=8)
  dlrs_w4_c4_nobuf      the same, ImageCollator() without buffer reuse

The first batch of each dataloader_rs loader is checked bit for bit against
torch's. Loaders needing Pillow / torchvision are skipped without them.
Rounds run every loader once, in a rotated order, after one warm-up epoch each.

Run (4 CPUs, as the CI runners):
  taskset -c 0,2,4,6 python bench/compare_image.py --dir /path/to/tiles [--rounds 5]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

TILE = 224
BATCH = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def write_tiles(directory: Path, items: int) -> list[Path]:
    """`items` PNG tiles in `directory`, written once (two distinct files, copied)."""
    import zlib

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def png(width: int, height: int) -> bytes:
        y, x = np.mgrid[0:height, 0:width]
        noise = rng.integers(0, 32, (height, width), dtype=np.uint8)
        rgb = np.stack([(x + noise) % 256, (y + noise) % 256, noise], axis=-1).astype(np.uint8)
        raw = b"".join(b"\0" + row.tobytes() for row in rgb)

        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            return len(data).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

        header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b"")
        )

    full, edge = png(TILE, TILE), png(TILE, 150)
    paths = []
    for i in range(items):
        path = directory / f"{i}.png"
        if not path.exists():
            path.write_bytes(edge if i % 8 == 7 else full)
        paths.append(path)
    return paths


# ── torch ─────────────────────────────────────────────────────────────────────


def pil_pipeline():
    """prime_dx's transform, as msintuit-ec runs it."""
    from PIL import Image
    from torchvision import transforms

    def constant_fill_2d(img):
        right, bottom = TILE - img.size[0], TILE - img.size[1]
        if right > 0 or bottom > 0:
            array = np.pad(np.array(img), ((0, bottom), (0, right), (0, 0)))
            return Image.fromarray(array)
        return img

    compose = transforms.Compose(
        [constant_fill_2d, transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    return lambda path: compose(Image.open(path).convert("RGB"))


class TorchTiles:
    """A map-style torch dataset: path → tensor."""

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        self.transform = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.transform is None:
            self.transform = pil_pipeline()
        return self.transform(self.paths[index])


def torch_loader(paths, workers):
    import torch

    return torch.utils.data.DataLoader(TorchTiles(paths), batch_size=BATCH, num_workers=workers)


# ── dataloader_rs ─────────────────────────────────────────────────────────────


def dlrs_loader(paths, workers, concurrency, buffers):
    from dataloader_rs import FileDataset, ImageCollator, PyDataloader
    from dataloader_rs import transforms as T

    chain = T.Compose(
        [T.Decode(), T.ToRgb(), T.PadTo(TILE, TILE), T.ToFloat(), T.Normalize(MEAN, STD), T.ToChw()]
    )
    return PyDataloader(
        FileDataset(paths),
        batch_size=BATCH,
        num_workers=workers,
        item_concurrency=concurrency,
        transform=chain,
        collate_fn=ImageCollator(buffers=buffers),
    )


def dlrs_pil_loader(paths, workers, concurrency):
    import io

    import torch

    from dataloader_rs import FileDataset, PyDataloader

    pipeline = pil_pipeline()
    return PyDataloader(
        FileDataset(paths),
        batch_size=BATCH,
        num_workers=workers,
        item_concurrency=concurrency,
        transform=lambda i, data: pipeline(io.BytesIO(data)),
        collate_fn=torch.stack,
    )


def loaders(paths, have_pil):
    out = {}
    if have_pil:
        out["torch_w4"] = lambda: torch_loader(paths, 4)
        out["torch_w8"] = lambda: torch_loader(paths, 8)
        out["dlrs_pil_w4_c4"] = lambda: dlrs_pil_loader(paths, 4, 4)
    out["dlrs_w4_c4"] = lambda: dlrs_loader(paths, 4, 4, 8)
    out["dlrs_w4_c4_nobuf"] = lambda: dlrs_loader(paths, 4, 4, 0)
    return out


# ── measurement ───────────────────────────────────────────────────────────────


def as_tensor(batch):
    import torch

    return batch if isinstance(batch, torch.Tensor) else torch.from_dlpack(batch)


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def epoch(make) -> tuple[float, float, int]:
    start, cpu = time.perf_counter(), cpu_seconds()
    loader = make()
    count = 0
    for batch in loader:
        count += as_tensor(batch).shape[0]
    del loader
    return time.perf_counter() - start, cpu_seconds() - cpu, count


def first_batch(make):
    batch = next(iter(make()))
    return as_tensor(batch).numpy().copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="where the tiles are written")
    parser.add_argument("--items", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    # torch_w8 on 4 CPUs is deliberate.
    warnings.filterwarnings("ignore", message="This DataLoader will create")

    try:
        import PIL  # noqa: F401
        import torchvision  # noqa: F401

        have_pil = True
    except ImportError:
        have_pil = False
    gil = "GIL" if getattr(sys, "_is_gil_enabled", lambda: True)() else "free-threaded"
    print(f"Python {sys.version.split()[0]} ({gil}), {len(os.sched_getaffinity(0))} CPUs")

    paths = write_tiles(args.dir, args.items)
    candidates = loaders(paths, have_pil)
    if have_pil:
        reference = first_batch(candidates["torch_w4"])
        for name in [n for n in candidates if n.startswith("dlrs")]:
            same = first_batch(candidates[name]).tobytes() == reference.tobytes()
            print(f"{name}: first batch {'equal to' if same else 'DIFFERENT from'} torch's")

    names = list(candidates)
    for name in names:  # warm-up
        epoch(candidates[name])
    results: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
    for r in range(args.rounds):
        order = names[r % len(names) :] + names[: r % len(names)]
        for name in order:
            seconds, cpu, count = epoch(candidates[name])
            assert count == args.items, (name, count)
            results[name].append((seconds, cpu))

    print(f"\n{args.items} tiles, batch {BATCH}, {args.rounds} rounds")
    print(f"{'loader':<20} {'epoch s':>8} {'min-max':>13} {'tiles/s':>8} {'cpu s':>7}")
    summary = {}
    for name in names:
        times = [t for t, _ in results[name]]
        cpus = [c for _, c in results[name]]
        median = statistics.median(times)
        summary[name] = {"epoch": times, "cpu": cpus}
        print(
            f"{name:<20} {median:>8.3f} {min(times):>6.3f}-{max(times):<6.3f}"
            f" {args.items / median:>8.0f} {statistics.median(cpus):>7.2f}"
        )
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
