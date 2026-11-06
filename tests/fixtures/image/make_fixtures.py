"""Golden fixtures for tests/integration_image.rs, from Pillow and torchvision.

Run from the repository root with Pillow, NumPy, torch and torchvision installed:

    python tests/fixtures/image/make_fixtures.py

Inputs are encoded images; expected outputs are raw little-endian arrays (`.u8`, `.f32`,
`.f16`) whose shapes are in the test. Re-running rewrites every file; the inputs are
deterministic, the JPEG and WebP encoders are the installed Pillow's.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

OUT = Path(__file__).parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos3": Image.Resampling.LANCZOS,
}


def scene(width: int, height: int, seed: int) -> np.ndarray:
    """Smooth gradients, hard edges and noise: something a resize filter can get wrong."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 255 * x / max(width - 1, 1)
    g = 255 * y / max(height - 1, 1)
    b = 127.5 + 127.5 * np.sin(x / 3.0) * np.cos(y / 5.0)
    img = np.stack([r, g, b], axis=-1)
    img[(x // 8 + y // 8) % 2 == 0] *= 0.6  # checkerboard edges
    img += rng.normal(0, 12, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def save(name: str, array: np.ndarray) -> None:
    (OUT / name).write_bytes(
        np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
    )


def to_tensor_normalize(img: Image.Image) -> torch.Tensor:
    return transforms.Normalize(MEAN, STD)(transforms.ToTensor()(img))


def main() -> None:
    rgb = scene(64, 48, seed=0)
    img = Image.fromarray(rgb, "RGB")
    img.save(OUT / "rgb.png")
    img.save(OUT / "rgb_lossless.webp", lossless=True, quality=100, method=6)
    img.save(OUT / "rgb_q90.jpg", quality=90)
    rgba = np.concatenate([rgb, np.arange(64 * 48, dtype=np.uint8).reshape(48, 64, 1)], axis=-1)
    Image.fromarray(rgba, "RGBA").save(OUT / "rgba.png")
    save("rgb.u8", rgb)
    save("rgb_q90_pil.u8", np.asarray(Image.open(OUT / "rgb_q90.jpg").convert("RGB")))
    save("rgb_to_gray_pil.u8", np.asarray(img.convert("L")))

    big = Image.fromarray(scene(160, 120, seed=1), "RGB")
    big.save(OUT / "big.png")
    for name, f in FILTERS.items():
        save(f"down_{name}.u8", np.asarray(big.resize((64, 48), f)))
        save(f"up_{name}.u8", np.asarray(img.resize((96, 72), f)))

    normalized = to_tensor_normalize(img)
    save("rgb_normalize.f32", normalized.numpy())
    save("rgb_normalize.f16", normalized.half().numpy())
    save("rgb_to_tensor.f32", transforms.ToTensor()(img).numpy())
    save(
        "big_pipeline.f32",
        to_tensor_normalize(big.resize((64, 48), Image.Resampling.BILINEAR)).numpy(),
    )

    total = sum(p.stat().st_size for p in OUT.iterdir() if p.suffix != ".py")
    print(f"{total / 1024:.0f} KiB of fixtures in {OUT}")


if __name__ == "__main__":
    main()
