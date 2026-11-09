"""``dataloader_rs.transforms`` and ``ImageCollator``: the Rust image transforms from
Python, against the golden files of ``tests/fixtures/image`` (made with Pillow, numpy
and torchvision), in the loader on every kind of dataset, and zero-copy batches."""

import asyncio
import gc
from pathlib import Path

import numpy as np
import pytest

from dataloader_rs import (
    FileDataset,
    Image,
    ImageBatch,
    ImageCollator,
    PyAsyncDataset,
    PyDataset,
)
from dataloader_rs import PyDataloader as DataLoader
from dataloader_rs import transforms as T

FIXTURES = Path(__file__).parent / "fixtures" / "image"
W, H = 37, 29
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# (num_workers, item_concurrency): direct, item lanes, workers, workers with lanes.
MODES = [(0, 1), (0, 4), (2, 1), (2, 4)]
TILES = ["tile_full", "rgb", "rgba", "gray", "palette"]


def png(name: str) -> bytes:
    return (FIXTURES / f"{name}.png").read_bytes()


def expected(name: str, shape, dtype=np.uint8) -> np.ndarray:
    return np.fromfile(FIXTURES / name, dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape)


def msintuit() -> T.Transform:
    """prime_dx's ConstantFill2d(48) + ToTensor + Normalize, in Rust."""
    return T.Compose(
        [T.Decode(), T.ToRgb(), T.PadTo(48, 48), T.ToFloat(), T.Normalize(MEAN, STD), T.ToChw()]
    )


# ── Each transform, against the golden files ─────────────────────────────────


@pytest.mark.parametrize(
    ("transform", "name", "shape"),
    [
        (T.Decode(), "rgb.u8", (H, W, 3)),
        (T.Compose([T.Decode(), T.ToGray()]), "rgb_to_gray.u8", (H, W, 1)),
        (T.Compose([T.Decode(), T.Crop(3, 5, 20, 30)]), "crop_3_5_20_30.u8", (20, 30, 3)),
        (T.Compose([T.Decode(), T.CenterCrop((40, 45))]), "center_crop_40_45.u8", (40, 45, 3)),
        (T.Compose([T.Decode(), T.HFlip()]), "hflip.u8", (H, W, 3)),
        (T.Compose([T.Decode(), T.VFlip()]), "vflip.u8", (H, W, 3)),
        (T.Compose([T.Decode(), T.Rotate90(3)]), "rot90_3.u8", (W, H, 3)),
        (
            T.Compose([T.Decode(), T.Pad((3, 1, 4, 2), fill=[10, 20, 30])]),
            "pad_constant.u8",
            (H + 3, W + 7, 3),
        ),
        (
            T.Compose([T.Decode(), T.Pad((40, 30, 2, 50), padding_mode="symmetric")]),
            "pad_wide_symmetric.u8",
            (H + 80, W + 42, 3),
        ),
        (T.Compose([T.Decode(), T.PadTo(37, 40, fill=255)]), "pad_to_37_40.u8", (40, 37, 3)),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_transforms_are_exact(transform, name, shape):
    image = transform(png("rgb"))
    assert isinstance(image, Image)
    assert image.shape == shape
    np.testing.assert_array_equal(np.asarray(image), expected(name, shape))


def test_float_chain_is_torchvision_bit_for_bit():
    to_tensor = T.Compose([T.Decode(), T.ToChw(), T.ToFloat()])
    out = np.asarray(to_tensor(png("rgb")))
    assert out.dtype == np.float32
    assert out.tobytes() == expected("to_tensor.f32", (3, H, W), np.float32).tobytes()
    for name in TILES:
        out = np.asarray(msintuit()(png(name)))
        ref = expected(f"msintuit_{name}.f32", (3, 48, 48), np.float32)
        assert out.tobytes() == ref.tobytes(), name


def test_resize_is_close_to_pillow():
    resize = T.Compose([T.Decode(), T.Resize((13, 17))])
    out = np.asarray(resize(png("rgb"))).astype(int)
    ref = expected("resize_down_bilinear.u8", (13, 17, 3)).astype(int)
    assert np.abs(out - ref).max() <= 1
    shorter = T.Compose([T.Decode(), T.Resize(10, interpolation="nearest")])
    assert shorter(png("rgb")).shape == (10, 12, 3)


def test_transforms_chain_on_images_too():
    image = T.Decode()(png("rgb"))
    flipped = T.HFlip()(image)
    np.testing.assert_array_equal(np.asarray(flipped), expected("hflip.u8", (H, W, 3)))
    # The input image is unchanged: transforms copy.
    np.testing.assert_array_equal(np.asarray(image), expected("rgb.u8", (H, W, 3)))


def test_reprs():
    assert repr(T.Compose([T.ToRgb(), T.PadTo(224, 224)])) == (
        "Compose([ToRgb(), PadTo(224, 224, fill=0, anchor='top_left')])"
    )
    assert repr(ImageCollator()) == "ImageCollator(layout='NCHW', dtype=None)"
    assert repr(T.Decode()(png("gray"))) == "Image(29x37x1, uint8, HWC)"


# ── Errors: clean exceptions, never a crash ──────────────────────────────────


@pytest.mark.parametrize(
    ("transform", "item", "match"),
    [
        (T.Decode(), b"not an image", "unknown format"),
        (T.Decode(), png("rgb")[:100], "cannot decode"),
        (T.Decode(max_width=36), png("rgb"), "too large"),
        (T.Compose([T.Decode(), T.Normalize([0.5], [0.5])]), png("rgb"), "add ToFloat"),
        (T.Compose([T.Decode(), T.ToChw(), T.HFlip()]), png("rgb"), "apply ToChw last"),
        (T.Compose([T.Decode(), T.Crop(20, 0, 10, 10)]), png("rgb"), "outside"),
        (T.Compose([T.Decode(), T.PadTo(64, 64, fill=300)]), png("rgb"), "not a uint8"),
        (T.ToRgb(), png("rgb"), "takes a uint8 HWC image, not bytes"),
    ],
    ids=["garbage", "truncated", "limits", "order", "chw", "crop", "fill", "bytes"],
)
def test_bad_images_and_orders_raise_value_error(transform, item, match):
    with pytest.raises(ValueError, match=match):
        transform(item)


def test_bad_arguments():
    with pytest.raises(TypeError, match="bytes or an Image"):
        T.Decode()(3)
    with pytest.raises(ValueError, match="padding_mode"):
        T.Pad(1, padding_mode="wrap")
    with pytest.raises(ValueError, match="fill only applies"):
        T.Pad(1, fill=3, padding_mode="edge")
    with pytest.raises(ValueError, match="interpolation"):
        T.Resize(8, interpolation="area")
    with pytest.raises(ValueError, match="anchor"):
        T.PadTo(8, 8, anchor="bottom")
    with pytest.raises(ValueError, match="non-zero stds"):
        T.Normalize([0.0], [0.0])
    with pytest.raises(TypeError):
        T.Compose([T.ToRgb(), lambda x: x])
    with pytest.raises(ValueError, match="layout"):
        ImageCollator(layout="CHW")
    with pytest.raises(ValueError, match="dtype"):
        ImageCollator(dtype="float64")


# ── In the loader: natively, on every kind of dataset ────────────────────────


class Pngs(PyDataset):
    def __init__(self, names):
        super().__init__()
        self.items = [png(n) for n in names]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AsyncPngs(PyAsyncDataset):
    def __init__(self, names):
        super().__init__()
        self.items = [png(n) for n in names]

    def __len__(self):
        return len(self.items)

    async def __getitem__(self, index):
        await asyncio.sleep(0.001 * (index % 2))
        return self.items[index]


def datasets(names):
    return {
        "py": Pngs(names),
        "async": AsyncPngs(names),
        "file": FileDataset([FIXTURES / f"{n}.png" for n in names]),
    }


@pytest.mark.parametrize("kind", ["py", "async", "file"])
@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_the_loader_runs_the_transforms_and_collates(kind, num_workers, item_concurrency):
    names = TILES * 2
    loader = DataLoader(
        datasets(names)[kind],
        batch_size=4,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
        transform=msintuit(),
        collate_fn=ImageCollator(),
    )
    for _epoch in range(2):
        batches = list(loader)
        assert all(isinstance(b, ImageBatch) for b in batches)
        assert [b.shape for b in batches] == [(4, 3, 48, 48), (4, 3, 48, 48), (2, 3, 48, 48)]
        out = np.concatenate([np.asarray(b) for b in batches])
        ref = np.stack([expected(f"msintuit_{n}.f32", (3, 48, 48), np.float32) for n in names])
        assert out.tobytes() == ref.tobytes()


@pytest.mark.parametrize("kind", ["py", "async", "file"])
def test_without_a_collator_items_are_images(kind):
    loader = DataLoader(datasets(["rgb", "gray"])[kind], batch_size=2, transform=T.Decode())
    (batch,) = list(loader)
    assert [type(i) for i in batch] == [Image, Image]
    assert [i.shape for i in batch] == [(H, W, 3), (H, W, 1)]
    stacked = DataLoader(
        datasets(["rgb", "gray"])[kind],
        batch_size=2,
        transform=T.Compose([T.Decode(), T.ToRgb()]),
        collate_fn=lambda items: np.stack([np.asarray(i) for i in items]),
    )
    (array,) = list(stacked)
    assert array.shape == (2, H, W, 3)


def test_python_transforms_pair_images_with_coordinates():
    coords = np.arange(10).reshape(5, 2)
    tiles = msintuit()
    loader = DataLoader(
        FileDataset([FIXTURES / f"{n}.png" for n in TILES]),
        batch_size=3,
        num_workers=2,
        item_concurrency=2,
        transform=lambda i, data: (tiles(data), coords[i]),
        collate_fn=ImageCollator(),
    )
    (images, first), (more, second) = list(loader)
    assert images.shape == (3, 3, 48, 48) and more.shape == (2, 3, 48, 48)
    np.testing.assert_array_equal(np.stack(first + second), coords)


def test_a_bad_image_fails_its_batch_only():
    names = ["rgb", "rgb", "rgb", "rgb"]
    dataset = Pngs(names)
    dataset.items[2] = b"garbage"
    loader = DataLoader(
        dataset, batch_size=2, num_workers=2, transform=T.Decode(), collate_fn=ImageCollator()
    )
    iterator = iter(loader)
    assert next(iterator).shape == (2, 3, H, W)
    with pytest.raises(ValueError, match="unknown format"):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_collator_errors():
    shapes = Pngs(["rgb", "tile_full"])
    loader = DataLoader(shapes, batch_size=2, transform=T.Decode(), collate_fn=ImageCollator())
    with pytest.raises(ValueError, match="same shape"):
        list(loader)
    floats = DataLoader(
        Pngs(["rgb"]), transform=T.Decode(), collate_fn=ImageCollator(dtype="float32")
    )
    with pytest.raises(ValueError, match="ToFloat"):
        list(floats)
    with pytest.raises(TypeError, match="collates dataloader_rs Images"):
        ImageCollator()([np.zeros((2, 2, 3), np.uint8)])
    with pytest.raises(ValueError, match="takes images, not bytes"):
        list(DataLoader(Pngs(["rgb"]), transform=T.Compose([]), collate_fn=ImageCollator()))


# ── Zero-copy batches ────────────────────────────────────────────────────────


def address(obj) -> int:
    return obj.__array_interface__["data"][0]


def test_numpy_views_the_batch_without_a_copy_and_keeps_it_alive():
    images = [T.Decode()(png("rgb")) for _ in range(2)]
    batch = ImageCollator(layout="NHWC")(images)
    assert (batch.shape, batch.dtype, batch.layout, len(batch)) == (
        (2, H, W, 3),
        "uint8",
        "NHWC",
        2,
    )
    array = np.asarray(batch)
    assert array.ctypes.data == address(batch)
    assert array.flags.c_contiguous and array.flags.writeable
    del batch, images
    gc.collect()
    np.testing.assert_array_equal(array[1], expected("rgb.u8", (H, W, 3)))
    # Images are read-only views.
    image = T.Decode()(png("rgb"))
    view = np.asarray(image)
    assert view.ctypes.data == address(image) and not view.flags.writeable


def test_torch_takes_the_batch_through_dlpack_without_a_copy():
    torch = pytest.importorskip("torch")
    batch = ImageCollator()([msintuit()(png(n)) for n in TILES])
    tensor = torch.from_dlpack(batch)
    assert tensor.data_ptr() == address(batch)
    assert tensor.shape == (5, 3, 48, 48) and tensor.dtype == torch.float32
    assert batch.__dlpack_device__() == (1, 0)
    ref = np.stack([expected(f"msintuit_{n}.f32", (3, 48, 48), np.float32) for n in TILES])
    del batch
    gc.collect()
    assert tensor.numpy().tobytes() == ref.tobytes()


def test_dropped_batches_lend_their_buffer_to_the_next():
    collate = ImageCollator(buffers=1)
    images = [T.Decode()(png("rgb"))]
    first = collate(images)
    kept = address(first)
    second = collate(images)
    assert address(second) != kept
    del first, second
    gc.collect()
    assert address(collate(images)) == kept
    # A batch still viewed by numpy keeps its buffer.
    held = np.asarray(collate(images))
    assert held.ctypes.data == kept
    assert address(collate(images)) != kept


# ── Image.from_array: arrays and tensors back into Rust ──────────────────────


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape", [(H, W), (H, W, 1), (H, W, 3), (H, W, 4)])
def test_from_array_round_trips(dtype, shape):
    array = np.random.default_rng(0).integers(0, 256, shape).astype(dtype)
    image = Image.from_array(array)
    channels = shape[2] if len(shape) == 3 else 1
    assert (image.shape, image.dtype, image.layout) == ((H, W, channels), dtype.__name__, "HWC")
    view = np.asarray(image)
    assert view.ctypes.data == address(image) != array.ctypes.data
    assert view.tobytes() == array.tobytes()


def test_from_array_reads_strided_arrays_and_chw():
    array = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    for strided in [array[::-1], array[:, ::2], np.asfortranarray(array)]:
        np.testing.assert_array_equal(np.asarray(Image.from_array(strided)), strided)
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    np.testing.assert_array_equal(np.asarray(Image.from_array(chw, layout="chw")), array)
    np.testing.assert_array_equal(np.asarray(Image.from_array(chw, layout="CHW")), array)
    gray = np.asarray(Image.from_array(array[..., 0], layout="chw"))
    np.testing.assert_array_equal(gray, array[..., :1])


def test_from_array_makes_no_numpy_copy_of_contiguous_arrays():
    tracemalloc = pytest.importorskip("tracemalloc")
    array = np.zeros((512, 512, 3), np.float32)
    tracemalloc.start()
    try:
        Image.from_array(array)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The copy is Rust's (not traced); numpy allocates nothing.
    assert peak < array.nbytes // 100


def test_from_array_takes_torch_tensors():
    torch = pytest.importorskip("torch")
    tensor = torch.rand(3, H, W)
    image = Image.from_array(tensor, layout="chw")
    assert image.shape == (H, W, 3)
    assert torch.equal(torch.from_dlpack(image), tensor.permute(1, 2, 0))
    hwc = (tensor * 255).to(torch.uint8).permute(1, 2, 0)  # a strided view
    np.testing.assert_array_equal(np.asarray(Image.from_array(hwc)), hwc.numpy())


def test_from_array_refuses_gpu_tensors():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    with pytest.raises(TypeError, match=r"not cuda:0: move the tensor first \(tensor.cpu\(\)\)"):
        Image.from_array(torch.zeros(3, 4, 4, device="cuda"), layout="chw")


@pytest.mark.parametrize(
    ("array", "error", "match"),
    [
        (np.zeros((4, 4), np.int16), TypeError, "uint8 or float32 values, not int16"),
        (np.zeros((4, 4), np.float64), TypeError, "not float64"),
        (np.zeros((4, 4), ">f4"), TypeError, "not >f4"),
        (np.zeros((4, 4), bool), TypeError, "not bool"),
        ([[1, 2], [3, 4]], TypeError, "not int64"),
        (np.zeros(4, np.uint8), ValueError, r"shape \[4\]: expected 2 dimensions"),
        (np.zeros((1, 2, 2, 3), np.uint8), ValueError, "expected 2 dimensions"),
        (np.zeros((4, 4, 2), np.uint8), ValueError, "2 channels, expected 1, 3 or 4"),
        (np.zeros((4, 4, 5), np.uint8), ValueError, "5 channels"),
        (np.zeros((0, 4, 3), np.uint8), ValueError, "empty array"),
        (np.uint8(3), TypeError, r"cannot read a uint8 \(an array of shape \(\)\)"),
        (b"\x00" * 12, TypeError, "cannot read a bytes"),
    ],
    ids=str,
)
def test_from_array_refuses_bad_arrays(array, error, match):
    with pytest.raises(error, match=match):
        Image.from_array(array)


def test_from_array_refuses_a_wrong_layout_or_shape_for_it():
    with pytest.raises(ValueError, match='layout must be "hwc" or "chw"'):
        Image.from_array(np.zeros((4, 4), np.uint8), layout="nchw")
    # An HWC array read as CHW: 224 "channels" of 224x3.
    with pytest.raises(ValueError, match="Chw layout has 224 channels"):
        Image.from_array(np.zeros((224, 224, 3), np.uint8), layout="chw")


def test_from_array_refuses_huge_arrays_before_any_copy():
    # Views of one value: numpy allocates nothing, so only the checks stand in the way.
    one = np.zeros(1, np.uint8)
    for shape in [(16_385, 1, 3), (1, 100_000, 3), (100_000, 100_000, 3)]:
        with pytest.raises(ValueError, match="image too large"):
            Image.from_array(np.broadcast_to(one, shape))
    # Within 16384 x 16384, but 1 GiB of float32: above Decode's 256 MiB.
    big = np.broadcast_to(np.zeros(1, np.float32), (16_384, 16_384, 4))
    with pytest.raises(ValueError, match="above the limit of 268435456"):
        Image.from_array(big, layout="hwc")
    with pytest.raises(ValueError, match="image too large"):
        Image.from_array(np.broadcast_to(one, (3, 16_384, 16_385)), layout="chw")


def test_from_array_without_numpy_reads_buffers(monkeypatch):
    data = bytes(range(24))
    expected_array = np.frombuffer(data, np.uint8).reshape(2, 4, 3)
    monkeypatch.setitem(__import__("sys").modules, "numpy", None)
    image = Image.from_array(memoryview(data).cast("B", (2, 4, 3)))
    gray = Image.from_array(memoryview(data).cast("B", (4, 6)), layout="chw")
    with pytest.raises(TypeError, match="needs numpy for layout"):
        Image.from_array(memoryview(data).cast("B", (3, 2, 4)), layout="chw")
    with pytest.raises(TypeError, match="without numpy, only objects with the buffer protocol"):
        Image.from_array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="expected 2 dimensions"):
        Image.from_array(data)
    monkeypatch.undo()
    np.testing.assert_array_equal(np.asarray(image), expected_array)
    assert gray.shape == (4, 6, 1)


def test_from_array_images_take_every_transform_and_the_collator():
    decoded = T.Compose([T.Decode(), T.ToRgb()])(png("rgba"))
    image = Image.from_array(np.asarray(decoded).copy())
    chain = T.Compose([T.PadTo(48, 48), T.ToFloat(), T.Normalize(MEAN, STD), T.ToChw()])
    native = chain(T.Compose([T.Decode(), T.ToRgb()])(png("rgba")))
    assert np.asarray(chain(image)).tobytes() == np.asarray(native).tobytes()
    assert (
        np.asarray(chain(image)).tobytes()
        == expected("msintuit_rgba.f32", (3, 48, 48), np.float32).tobytes()
    )
    batch = ImageCollator()(
        [
            Image.from_array(np.zeros((4, 5), np.uint8)),
            Image.from_array(np.ones((4, 5, 1), np.uint8)),
        ]
    )
    assert batch.shape == (2, 1, 4, 5)
    np.testing.assert_array_equal(np.asarray(batch)[1], 1)


@pytest.mark.parametrize(("num_workers", "item_concurrency"), MODES)
def test_loader_mixes_rust_and_torch_transforms(num_workers, item_concurrency):
    """Rust decode, a torch op, back into Rust for Normalize, and collated in Rust,
    with each item's coordinates beside it."""
    torch = pytest.importorskip("torch")
    decode = T.Compose([T.Decode(), T.ToRgb(), T.PadTo(48, 48)])
    normalize = T.Compose([T.ToFloat(), T.Normalize(MEAN, STD)])
    coords = np.arange(2 * len(TILES)).reshape(len(TILES), 2)

    def transform(index, data):
        chw = torch.from_dlpack(decode(data)).permute(2, 0, 1)
        flipped = torch.flip(chw, dims=[2])
        return normalize(Image.from_array(flipped, layout="chw")), coords[index]

    loader = DataLoader(
        FileDataset([FIXTURES / f"{n}.png" for n in TILES]),
        batch_size=3,
        num_workers=num_workers,
        item_concurrency=item_concurrency,
        transform=transform,
        collate_fn=ImageCollator(),
    )
    (first, first_coords), (second, second_coords) = list(loader)
    assert isinstance(first, ImageBatch) and first.shape == (3, 3, 48, 48)
    np.testing.assert_array_equal(np.stack(first_coords + second_coords), coords)
    out = np.concatenate([np.asarray(first), np.asarray(second)])
    ref = np.stack([expected(f"msintuit_{n}.f32", (3, 48, 48), np.float32) for n in TILES])
    assert out.tobytes() == ref[..., ::-1].tobytes()
