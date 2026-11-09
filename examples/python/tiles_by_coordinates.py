"""Tiles of a large image, fetched over HTTP by their (x, y) coordinates, decoded and batched in
Rust; each batch comes with the indices of its tiles, which give their coordinates.

This is the usual way to run a model over a very large image (a satellite image, a scanned
microscope slide): a tile server cuts it into tiles on request, and the coordinates say where
each result belongs. The example starts its own small tile server, so it runs offline. Needs
NumPy.

Run it with ``python examples/python/tiles_by_coordinates.py``.
"""

from urllib.parse import parse_qs, urlsplit

import numpy as np
from local_server import serve
from tiles import TILE, png

from dataloader_rs import HttpDataset, ImageCollator, PyDataloader
from dataloader_rs import transforms as T

WIDTH, HEIGHT = 2 * TILE + TILE // 2, 2 * TILE  # 3 x 2 tiles, the last column half as wide
TOKEN = "Bearer example-token"


def tile_server(path: str, headers) -> tuple[int, bytes]:
    """``/tile?x=<x>&y=<y>`` returns the PNG tile whose top-left corner is (x, y)."""
    if headers.get("Authorization") != TOKEN:
        return 401, b"missing token"
    url = urlsplit(path)
    query = parse_qs(url.query)
    if url.path != "/tile" or "x" not in query or "y" not in query:
        return 404, b"not found"
    x, y = int(query["x"][0]), int(query["y"][0])
    width, height = min(TILE, WIDTH - x), min(TILE, HEIGHT - y)
    if width <= 0 or height <= 0:
        return 404, b"outside the image"
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = x // 3
    pixels[..., 1] = y // 2
    return 200, png(pixels)


def main() -> None:
    # The top-left corner of every tile, row by row: shape (6, 2).
    coords = np.array([(x, y) for y in range(0, HEIGHT, TILE) for x in range(0, WIDTH, TILE)])
    decode = T.Compose(
        [
            T.Decode(),
            T.ToRgb(),
            T.PadTo(TILE, TILE),  # the narrow edge tiles are padded with black, at the right
            T.ToFloat(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.ToChw(),
        ]
    )

    with serve(tile_server) as base:
        # Item i is the tile at coords[i]. The URLs are made in Rust from the template, one at
        # a time, when each item is fetched: no list of URLs is kept. NumPy columns work.
        tiles = HttpDataset.from_template(
            base + "/tile?x={x}&y={y}",
            {"x": coords[:, 0], "y": coords[:, 1]},
            headers={"Authorization": TOKEN},
        )
        loader = PyDataloader(
            tiles,
            batch_size=4,
            num_workers=2,
            item_concurrency=4,  # 8 tile requests in flight
            # Fetching, decoding and batching all run in Rust, without the GIL.
            transform=decode,
            collate_fn=ImageCollator(),
            # Each batch comes as (images, indices): the dataset indices of its tiles, in the
            # order the sampler drew them, so coords[indices] are their coordinates.
            return_indices=True,
        )
        for images, indices in loader:
            batch = np.asarray(images)  # a view of the batch's buffer, no copy
            tile_coords = coords[indices]  # shape (len(indices), 2)
            # A model would run on the batch here (torch.from_dlpack(images) is a view too).
            print(batch.shape, batch.dtype, "at", [tuple(int(v) for v in c) for c in tile_coords])


if __name__ == "__main__":
    main()
