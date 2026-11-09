"""All in Rust: read PNG tiles, decode, pad, normalize and batch them on the loader's
threads, without the Python interpreter.

FileDataset reads the files, the ``transforms.Compose`` runs on each item and
ImageCollator writes the images into one batch buffer: no Python code runs per item,
so the GIL (or the free-threaded interpreter) is never taken until a batch is handed
over. ``torch.from_dlpack(batch)`` is a view of that buffer: no copy.
"""

import tempfile
from pathlib import Path

import torch
from tiles import TILE, write_tiles

from dataloader_rs import FileDataset, ImageCollator, PyDataloader
from dataloader_rs import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, _ = write_tiles(Path(tmp))
        # Over HTTP, the same tiles would be:
        #   HttpDataset.from_template("https://host/slide/tile?x={x}&y={y}", {"x": xs, "y": ys})
        dataset = FileDataset(paths)
        transform = T.Compose(
            [
                T.Decode(),  # PNG, JPEG, WebP, TIFF or BMP bytes -> uint8 HWC
                T.ToRgb(),  # gray, RGBA or palette tiles -> 3 channels
                T.PadTo(TILE, TILE),  # edge tiles padded with 0, at the bottom and right
                T.ToFloat(),  # / 255, as ToTensor
                T.Normalize(MEAN, STD),
                T.ToChw(),  # torch's layout
            ]
        )
        loader = PyDataloader(
            dataset,
            batch_size=4,
            num_workers=2,
            transform=transform,
            collate_fn=ImageCollator(),  # -> ImageBatch, NCHW float32
        )
        for batch in loader:
            images = torch.from_dlpack(batch)  # no copy: the batch's own memory
            print(tuple(images.shape), images.dtype, f"mean {images.mean():+.3f}")


if __name__ == "__main__":
    main()
