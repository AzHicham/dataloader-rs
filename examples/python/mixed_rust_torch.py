"""Rust and torch in one transform: Rust decodes, a torch op augments, Rust normalizes
and batches, and each tile keeps its coordinates.

Per item, in ``transform`` (run on the loader's threads):

1. ``decode(data)``: Rust decodes the PNG bytes into an ``Image``; the interpreter is
   released during the call.
2. ``torch.from_dlpack(image)``: a tensor viewing the image's memory, no copy.
3. ``torch.rot90``: a torch op; it makes a new tensor and releases the GIL inside, the
   Python lines around it hold the interpreter.
4. ``Image.from_array(tensor, layout="chw")``: back into Rust. A contiguous HWC tensor is
   copied once; this CHW one is made HWC by numpy first (one more copy).
5. ``normalize(image)``: Rust again, interpreter released.

The transform returns ``(Image, coords[i])``: ImageCollator batches the images in Rust and
lists the rest, so each batch is ``(ImageBatch, [coords, ...])``.

torch ops on several loader threads: call ``torch.set_num_threads(1)`` so each op stays on
its thread instead of every op fanning out to all cores.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from tiles import TILE, write_tiles

from dataloader_rs import FileDataset, Image, ImageCollator, PyDataloader
from dataloader_rs import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main() -> None:
    torch.set_num_threads(1)
    decode = T.Compose([T.Decode(), T.ToRgb(), T.PadTo(TILE, TILE)])
    normalize = T.Compose([T.ToFloat(), T.Normalize(MEAN, STD)])

    with tempfile.TemporaryDirectory() as tmp:
        paths, coords = write_tiles(Path(tmp))

        def transform(index: int, data: bytes) -> tuple[Image, np.ndarray]:
            image = decode(data)  # 1. Rust, uint8 HWC
            chw = torch.from_dlpack(image).permute(2, 0, 1)  # 2. a view, as torch's CHW
            chw = torch.rot90(chw, k=index % 4, dims=(1, 2))  # 3. torch
            image = Image.from_array(chw, layout="chw")  # 4. back into Rust, HWC
            return normalize(image), coords[index]  # 5. Rust, float32

        loader = PyDataloader(
            FileDataset(paths),
            batch_size=4,
            num_workers=2,
            item_concurrency=2,
            transform=transform,
            collate_fn=ImageCollator(),
        )
        for batch, tile_coords in loader:
            images = torch.from_dlpack(batch)  # NCHW float32, no copy
            print(tuple(images.shape), [tuple(int(v) for v in c) for c in tile_coords])


if __name__ == "__main__":
    main()
