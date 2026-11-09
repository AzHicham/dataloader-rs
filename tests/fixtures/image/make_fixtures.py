"""Golden fixtures for tests/image.rs, from Pillow, numpy and torchvision.

Run from the repository root with Pillow, numpy, torch and torchvision installed:

    python tests/fixtures/image/make_fixtures.py

Inputs are encoded images; expected outputs are raw arrays (`.u8`, little-endian `.f32`) whose
shapes are in the tests. Re-running rewrites every file: the inputs are deterministic, the lossy
JPEG and WebP files depend on the installed Pillow's encoders.
"""

import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).parent
# Odd sizes, so that off-by-one errors in rows, centers and padding show.
W, H = 37, 29


def scene(width: int, height: int, seed: int) -> np.ndarray:
    """Gradients, hard edges and noise: something every operation can get wrong."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 255 * x / max(width - 1, 1)
    g = 255 * y / max(height - 1, 1)
    b = 127.5 + 127.5 * np.sin(x / 3.0) * np.cos(y / 5.0)
    img = np.stack([r, g, b], axis=-1)
    img[(x // 8 + y // 8) % 2 == 0] *= 0.6
    img += rng.normal(0, 12, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def save(name: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    if array.dtype == np.float32:
        array = array.astype("<f4")
    (OUT / name).write_bytes(array.tobytes())


def png16(name: str, array: np.ndarray) -> None:
    """A 16-bit PNG (gray or RGB), which Pillow cannot write."""
    height, width = array.shape[:2]
    color = 0 if array.ndim == 2 else 2
    rows = array.astype(">u2").reshape(height, -1)
    raw = b"".join(b"\0" + row.tobytes() for row in rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 16, color, 0, 0, 0)
    (OUT / name).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def decode_fixtures() -> None:
    rgb = scene(W, H, seed=0)
    img = Image.fromarray(rgb, "RGB")
    img.save(OUT / "rgb.png")
    img.save(OUT / "rgb.bmp")
    img.save(OUT / "rgb.tiff", compression="tiff_deflate")
    img.save(OUT / "rgb_lossless.webp", lossless=True, quality=100, method=6)
    img.save(OUT / "rgb_q90.jpg", quality=90)
    img.save(OUT / "rgb_q90.webp", quality=90)
    save("rgb.u8", rgb)
    save("rgb_q90_jpg.u8", np.asarray(Image.open(OUT / "rgb_q90.jpg").convert("RGB")))
    save("rgb_q90_webp.u8", np.asarray(Image.open(OUT / "rgb_q90.webp").convert("RGB")))
    save("rgb_to_gray.u8", np.asarray(img.convert("L")))

    alpha = (np.arange(W * H) % 256).astype(np.uint8).reshape(H, W, 1)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    Image.fromarray(rgba, "RGBA").save(OUT / "rgba.png")
    save("rgba.u8", rgba)
    save("rgba_to_gray.u8", np.asarray(Image.fromarray(rgba, "RGBA").convert("L")))

    gray = Image.fromarray(rgb[:, :, 1], "L")
    gray.save(OUT / "gray.png")
    save("gray_to_rgb.u8", np.asarray(gray.convert("RGB")))
    la = Image.fromarray(np.concatenate([rgb[:, :, 1:2], alpha], axis=-1), "LA")
    la.save(OUT / "la.png")
    save("la_to_rgba.u8", np.asarray(la.convert("RGBA")))

    palette = img.quantize(colors=16)
    palette.save(OUT / "palette.png")
    save("palette_to_rgb.u8", np.asarray(palette.convert("RGB")))
    palette.save(OUT / "palette_trns.png", transparency=3)

    wide = (rgb.astype(np.uint16) << 8) | 0xA5
    png16("rgb16.png", wide)
    png16("gray16.png", wide[:, :, 1])
    # The high bytes; Pillow agrees for RGB (checked here), not for gray (it clips).
    assert (np.asarray(Image.open(OUT / "rgb16.png").convert("RGB")) == rgb).all()


def main() -> None:
    decode_fixtures()
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.suffix != ".py")
    print(f"{total / 1024:.0f} KiB of fixtures in {OUT}")


if __name__ == "__main__":
    main()
