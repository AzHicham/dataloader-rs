"""Rust fetches, torch transforms: FileDataset reads the tiles on the loader's threads and
a plain torchvision pipeline decodes and normalizes them, as in a torch DataLoader but
with threads instead of processes.

The transform is Python, so each item holds the interpreter while its Python lines run
(torch and torchvision release it inside their ops); ``collate_fn=torch.stack`` batches.
With several loader threads running torch ops, call ``torch.set_num_threads(1)`` so the
ops do not each fan out to all cores (oversubscription).
"""

import tempfile
from pathlib import Path

import torch
from tiles import TILE, write_tiles
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from dataloader_rs import FileDataset, PyDataloader

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main() -> None:
    torch.set_num_threads(1)
    pipeline = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def transform(index: int, data: bytes) -> torch.Tensor:
        # bytes are read-only; torch wants a writable buffer: copy the small PNG.
        encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        image = decode_image(encoded, mode=ImageReadMode.RGB)  # uint8 CHW
        height, width = image.shape[1:]
        image = v2.functional.pad(image, [0, 0, TILE - width, TILE - height])  # edge tiles
        return pipeline(image)

    with tempfile.TemporaryDirectory() as tmp:
        paths, _ = write_tiles(Path(tmp))
        loader = PyDataloader(
            FileDataset(paths),
            batch_size=4,
            num_workers=2,
            transform=transform,
            collate_fn=torch.stack,
        )
        for images in loader:
            print(tuple(images.shape), images.dtype, f"mean {images.mean():+.3f}")


if __name__ == "__main__":
    main()
