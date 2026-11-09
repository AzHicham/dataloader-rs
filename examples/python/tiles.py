"""Small PNG tiles for the examples, written with numpy and the standard library only."""

import struct
import zlib
from pathlib import Path

import numpy as np

TILE = 224


def png(pixels: np.ndarray) -> bytes:
    """An 8-bit RGB PNG of ``pixels``, shaped ``(height, width, 3)``."""
    height, width, _ = pixels.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)  # filter 0: raw rows
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def write_tiles(directory: Path, columns: int = 3, rows: int = 3) -> tuple[list[Path], np.ndarray]:
    """A ``columns x rows`` grid of tiles of a slide, as ``tile_{x}_{y}.png``: the paths and
    the ``(x, y)`` coordinates of each. The last column is narrower, as at a slide's edge."""
    rng = np.random.default_rng(0)
    paths, coords = [], []
    for y in range(0, rows * TILE, TILE):
        for column in range(columns):
            x = column * TILE
            width = TILE if column < columns - 1 else TILE // 2
            pixels = rng.integers(0, 256, (TILE, width, 3), dtype=np.uint8)
            path = directory / f"tile_{x}_{y}.png"
            path.write_bytes(png(pixels))
            paths.append(path)
            coords.append((x, y))
    return paths, np.array(coords)
