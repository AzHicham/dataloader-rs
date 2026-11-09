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
import torch
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision import transforms

OUT = Path(__file__).parent
# Odd sizes, so that off-by-one errors in rows, centers and padding show.
W, H = 37, 29
RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos3": Image.Resampling.LANCZOS,
}


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


def chw(image: np.ndarray) -> "torch.Tensor":
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def hwc(tensor: "torch.Tensor") -> np.ndarray:
    return tensor.permute(1, 2, 0).numpy()


def constant_fill_2d(img: Image.Image, shape: tuple[int, int], value: float) -> Image.Image:
    """prime_dx 2.5.0 `ConstantFill2d.__call__` (transforms/transforms.py), without its log."""
    right_padding = shape[0] - img.size[0]
    bottom_padding = shape[1] - img.size[1]
    if right_padding > 0 or bottom_padding > 0:
        return Image.fromarray(
            np.pad(
                np.array(img),
                pad_width=((0, bottom_padding), (0, right_padding), (0, 0)),
                mode="constant",
                constant_values=value,
            )
        )
    return img


def geometry_fixtures() -> None:
    rgb = scene(W, H, seed=0)
    img = Image.fromarray(rgb, "RGB")
    tensor = chw(rgb)
    save("crop_3_5_20_30.u8", hwc(F.crop(tensor, 3, 5, 20, 30)))
    for h, w in [(20, 30), (40, 45), (31, 20)]:
        save(f"center_crop_{h}_{w}.u8", hwc(F.center_crop(tensor, [h, w])))
        assert (np.asarray(F.center_crop(img, [h, w])) == hwc(F.center_crop(tensor, [h, w]))).all()
    save("hflip.u8", hwc(F.horizontal_flip(tensor)))
    save("vflip.u8", hwc(F.vertical_flip(tensor)))
    for k in (1, 2, 3):
        save(f"rot90_{k}.u8", np.rot90(rgb, k))
        assert (hwc(torch.rot90(tensor, k, dims=(-2, -1))) == np.rot90(rgb, k)).all()

    sides = [3, 1, 4, 2]  # left, top, right, bottom
    save("pad_constant.u8", hwc(F.pad(tensor, sides, fill=[10, 20, 30], padding_mode="constant")))
    for mode in ("edge", "reflect", "symmetric"):
        save(f"pad_{mode}.u8", hwc(F.pad(tensor, sides, padding_mode=mode)))
        # Wider than the image: numpy.pad, as torchvision on PIL images.
        wide = np.pad(rgb, ((30, 50), (40, 2), (0, 0)), mode=mode)
        assert (np.asarray(F.pad(img, [40, 30, 2, 50], padding_mode=mode)) == wide).all()
        save(f"pad_wide_{mode}.u8", wide)

    # ConstantFill2d: an undersized tile, an exact one, one undersized on one side only.
    save("pad_to_64.u8", np.asarray(constant_fill_2d(img, (64, 64), 0)))
    save("pad_to_37_40.u8", np.asarray(constant_fill_2d(img, (37, 40), 255)))

    # Resize: Pillow (what torchvision does on PIL images, antialiased) and torch
    # without antialias.
    for name, f in RESAMPLE.items():
        save(f"resize_down_{name}.u8", np.asarray(img.resize((17, 13), f)))
        save(f"resize_up_{name}.u8", np.asarray(img.resize((53, 41), f)))
    # torch's bicubic is a = -0.75, Pillow's -0.5: only bilinear compares.
    small = F.resize(tensor, [13, 17], interpolation=F.InterpolationMode.BILINEAR, antialias=False)
    save("resize_down_bilinear_no_aa.u8", hwc(small))


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TILE = 48


def float_fixtures() -> None:
    img = Image.open(OUT / "rgb.png")
    to_tensor = transforms.ToTensor()
    save("to_tensor.f32", to_tensor(img).numpy())
    save("to_tensor_normalize.f32", transforms.Normalize(MEAN, STD)(to_tensor(img)).numpy())
    # torchvision v2's ToDtype multiplies by 1/255 where ToTensor divides by 255: 126 of the
    # 256 values differ by one ulp. ToFloat divides.
    v2 = transforms_v2.Compose(
        [transforms_v2.PILToTensor(), transforms_v2.ToDtype(torch.float32, scale=True)]
    )
    assert torch.allclose(v2(img), to_tensor(img), rtol=1e-7, atol=0)

    # msintuit-ec's feature pass (prime_dx): ConstantFill2d + ToTensor + Normalize, on a full
    # tile, an undersized edge tile and tiles that are not RGB.
    full = Image.fromarray(scene(TILE, TILE, seed=2), "RGB")
    full.save(OUT / "tile_full.png")
    pipeline = transforms.Compose(
        [
            lambda x: constant_fill_2d(x.convert("RGB"), (TILE, TILE), 0.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    for name in ["tile_full.png", "rgb.png", "rgba.png", "gray.png", "palette.png"]:
        expected = pipeline(Image.open(OUT / name))
        save(f"msintuit_{Path(name).stem}.f32", expected.numpy())


def main() -> None:
    decode_fixtures()
    geometry_fixtures()
    float_fixtures()
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.suffix != ".py")
    print(f"{total / 1024:.0f} KiB of fixtures in {OUT}")


if __name__ == "__main__":
    main()
