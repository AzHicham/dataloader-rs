"""The Rust image pipeline from Python: ImageCollator with PyDataloader.

Encoded images and expected outputs are the golden fixtures of the Rust tests
(tests/fixtures/image, from Pillow and torchvision), so these tests need
neither Pillow nor torchvision.
"""

import gc
from pathlib import Path

import numpy as np
import pytest

from dataloader_rs import (
    Crop,
    ImageBatch,
    ImageCollator,
    Normalize,
    Pad,
    PyDataset,
    Resize,
    ToFloat,
)
from dataloader_rs import PyDataloader as DataLoader

FIXTURES = Path(__file__).parent / "fixtures" / "image"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def golden(name: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(fixture(name), dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape)


# 64x48 RGB scene in four encodings, and a 160x120 one.
RGB_PNG = fixture("rgb.png")
RGB_WEBP = fixture("rgb_lossless.webp")
RGB_JPEG = fixture("rgb_q90.jpg")
BIG_PNG = fixture("big.png")
SOURCES = [RGB_PNG, BIG_PNG, RGB_JPEG, RGB_WEBP]


def normalized_224(**kwargs) -> ImageCollator:
    """Images of any size to an f32 NCHW 3x30x40 sample."""
    return ImageCollator([Resize(40, 30), ToFloat(), Normalize(MEAN, STD)], **kwargs)


def reference(collator: ImageCollator, images: list[bytes]) -> np.ndarray:
    """The samples of `images`, one batch, sequential: the expected values."""
    loader = DataLoader(images, batch_size=len(images), collate_fn=collator)
    (batch,) = list(loader)
    return np.array(batch)


class ImageDataset(PyDataset):
    """Encoded images with their index as metadata; `bad` indices return junk."""

    def __init__(self, n: int, bad=(), plain: bool = False):
        super().__init__()
        self.n = n
        self.bad = set(bad)
        self.plain = plain

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        data = b"junk" if index in self.bad else SOURCES[index % len(SOURCES)]
        return data if self.plain else (data, index)


# ── Configuration ────────────────────────────────────────────────────────────


def test_collator_describes_its_samples():
    c = ImageCollator(
        [Resize(256, 256, "bicubic"), Crop(16, 16, 224, 224), ToFloat(), Normalize(MEAN, STD)],
        dtype="float16",
    )
    assert c.sample_shape == (3, 224, 224)
    assert c.dtype == "float16"
    assert 'Resize(256, 256, filter="bicubic")' in repr(c)
    nhwc = ImageCollator([Pad(8, 8, fill=255)], channels="gray", dtype="uint8", layout="nhwc")
    assert nhwc.sample_shape == (8, 8, 1)
    assert Resize(4, 2).filter == "bilinear"
    assert (Crop(1, 2, 3, 4).left, Pad(5, 6).fill, Normalize([0.5], [2.0]).std) == (1, 0, [2.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ops": []}, "sample size depends on the input"),
        ({"ops": [Resize(0, 4)]}, "empty size"),
        ({"ops": [Resize(4, 4), Crop(2, 0, 3, 3)]}, "operation 1 \\(crop\\)"),
        ({"ops": [ToFloat(), Resize(4, 4)], "input_size": (8, 8)}, "must come before"),
        ({"ops": [Normalize([0.5], [0.0])], "input_size": (8, 8)}, "non-zero"),
        ({"ops": [ToFloat()], "input_size": (8, 8), "dtype": "uint8"}, "float"),
        ({"ops": [Resize(4, 4)], "dtype": "int64"}, "unknown dtype"),
        ({"ops": [Resize(4, 4)], "layout": "chw"}, "unknown layout"),
        ({"ops": [Resize(4, 4)], "channels": "cmyk"}, "unknown channels"),
        ({"ops": [Resize(20_000, 4)]}, "limits"),
    ],
)
def test_invalid_pipelines_raise_value_error(kwargs, message):
    ops = kwargs.pop("ops")
    with pytest.raises(ValueError, match=message):
        ImageCollator(ops, **kwargs)


def test_invalid_operations_raise_type_error():
    with pytest.raises(TypeError, match="not an image operation"):
        ImageCollator([("resize", 4, 4)])
    with pytest.raises(ValueError, match="unknown filter"):
        Resize(4, 4, "sinc")


def test_dataset_types():
    c = normalized_224()
    with pytest.raises(TypeError, match="PyDataset or a list of bytes"):
        DataLoader({"a": RGB_PNG}, collate_fn=c)
    with pytest.raises(TypeError, match="bytes or \\(bytes, meta\\)"):
        DataLoader([RGB_PNG, "not bytes"], collate_fn=c)
    # A list without ImageCollator is not a dataset.
    with pytest.raises(TypeError, match="PyDatasetBase"):
        DataLoader([RGB_PNG])
    with pytest.raises(ValueError, match="max_held_batches"):
        DataLoader([RGB_PNG], collate_fn=c, max_held_batches=0)


# ── Values ───────────────────────────────────────────────────────────────────


def test_png_and_lossless_webp_decode_like_pillow():
    c = ImageCollator(dtype="uint8", layout="nhwc", input_size=(64, 48))
    (batch,) = DataLoader([RGB_PNG, bytearray(RGB_WEBP)], batch_size=2, collate_fn=c)
    expected = golden("rgb.u8", "uint8", (48, 64, 3))
    assert batch.shape == (2, 48, 64, 3)
    np.testing.assert_array_equal(np.asarray(batch), np.stack([expected, expected]))


def test_to_tensor_normalize_match_torchvision():
    for dtype, name in [("float32", "rgb_normalize.f32"), ("float16", "rgb_normalize.f16")]:
        c = ImageCollator([ToFloat(), Normalize(MEAN, STD)], dtype=dtype, input_size=(64, 48))
        (batch,) = DataLoader([RGB_PNG], collate_fn=c)
        assert batch.dtype == dtype
        expected = golden(name, dtype, (1, 3, 48, 64))
        np.testing.assert_array_equal(np.asarray(batch), expected)


def test_meta_sizes_and_partial_last_batch():
    loader = DataLoader(ImageDataset(10), batch_size=4, num_workers=2, collate_fn=normalized_224())
    assert len(loader) == 3
    batches = []
    for batch in loader:
        batches.append((len(batch), batch.shape, batch.meta, batch.sizes))
        del batch
    assert [b[0] for b in batches] == [4, 4, 2]
    assert batches[2][1] == (2, 3, 30, 40)
    assert sum((b[2] for b in batches), []) == list(range(10))
    assert batches[0][3] == [(64, 48), (160, 120), (64, 48), (64, 48)]


def test_list_dataset_with_meta_and_without():
    c = normalized_224()
    items = [(img, f"label-{i}") for i, img in enumerate(SOURCES)]
    (batch,) = DataLoader(items, batch_size=4, collate_fn=c)
    assert batch.meta == ["label-0", "label-1", "label-2", "label-3"]
    (plain,) = DataLoader(SOURCES, batch_size=4, collate_fn=c)
    assert plain.meta == [None] * 4
    np.testing.assert_array_equal(np.asarray(batch), np.asarray(plain))


@pytest.mark.parametrize(
    ("num_workers", "items_in_parallel"), [(0, 1), (0, 3), (1, 1), (3, 1), (4, 2)]
)
def test_same_batches_with_any_threads(num_workers, items_in_parallel):
    c = normalized_224(dtype="float32")
    expected = reference(c, [SOURCES[i % 4] for i in range(23)])
    for dataset in [ImageDataset(23), [SOURCES[i % 4] for i in range(23)]]:
        loader = DataLoader(
            dataset,
            batch_size=5,
            num_workers=num_workers,
            items_in_parallel=items_in_parallel,
            collate_fn=c,
        )
        got = []
        for batch in loader:
            got.append(np.array(batch))
            del batch
        np.testing.assert_array_equal(np.concatenate(got), expected)


def test_shuffle_follows_the_sampler():
    loader = DataLoader(
        ImageDataset(12),
        batch_size=5,
        shuffle=True,
        seed=3,
        num_workers=2,
        collate_fn=normalized_224(),
    )
    expected = reference(normalized_224(), [SOURCES[i % 4] for i in range(12)])
    order = []
    for batch in loader:
        for row, index in zip(np.asarray(batch), batch.meta, strict=True):
            np.testing.assert_array_equal(row, expected[index])
            order.append(index)
        del batch
    assert sorted(order) == list(range(12))
    assert order != list(range(12))


# ── Zero-copy hand-off and the buffer pool ───────────────────────────────────


def test_numpy_views_the_buffer_without_copy():
    (batch,) = DataLoader(SOURCES, batch_size=4, collate_fn=normalized_224())
    assert isinstance(batch, ImageBatch)
    array = np.asarray(batch)
    again = batch.numpy()
    address = batch.__array_interface__["data"][0]
    assert array.ctypes.data == again.ctypes.data == address
    assert array.base is batch
    assert array.flags.c_contiguous and array.flags.writeable
    array[0, 0, 0, 0] = 42.0
    assert again[0, 0, 0, 0] == 42.0


def test_dlpack_views_the_buffer_without_copy():
    torch = pytest.importorskip("torch")
    (batch,) = DataLoader(SOURCES, batch_size=4, collate_fn=normalized_224(dtype="float16"))
    assert batch.__dlpack_device__() == (1, 0)
    tensor = torch.from_dlpack(batch)
    assert tensor.dtype == torch.float16
    assert tuple(tensor.shape) == batch.shape
    assert tensor.data_ptr() == np.asarray(batch).ctypes.data
    np.testing.assert_array_equal(tensor.numpy(), np.asarray(batch))


def test_buffers_are_reused_across_batches_and_epochs():
    loader = DataLoader(
        [SOURCES[i % 4] for i in range(40)],
        batch_size=4,
        num_workers=2,
        prefetch_depth=1,
        collate_fn=normalized_224(),
    )
    addresses = set()
    for _ in range(3):
        for batch in loader:
            addresses.add(batch.__array_interface__["data"][0])
            del batch
    # At most num_workers + prefetch_depth + max_held_batches (2) buffers.
    assert 1 <= len(addresses) <= 5


def test_a_view_keeps_its_buffer_until_released():
    torch = pytest.importorskip("torch")
    loader = DataLoader(
        [SOURCES[i % 4] for i in range(32)],
        batch_size=4,
        num_workers=2,
        collate_fn=normalized_224(),
        max_held_batches=3,
    )
    it = iter(loader)
    kept = torch.from_dlpack(next(it))
    copy = kept.clone()
    # The batch object is gone; the tensor still holds its buffer, which no later batch reuses.
    gc.collect()
    for batch in it:
        assert batch.__array_interface__["data"][0] != kept.data_ptr()
        del batch
    assert torch.equal(kept, copy)


def test_holding_too_many_batches_raises_instead_of_hanging():
    loader = DataLoader(
        [SOURCES[i % 4] for i in range(40)],
        batch_size=4,
        num_workers=2,
        collate_fn=normalized_224(),
    )
    with pytest.raises(RuntimeError, match="max_held_batches is 2"):
        list(loader)
    # The held batches are gone with the failed list: the loader works again.
    gc.collect()
    total = 0
    for batch in loader:
        total += len(batch)
        del batch
    assert total == 40
    # Copies can be kept.
    assert len([np.array(b) for b in loader]) == 10


# ── Errors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_bad_image_fails_its_batch_only(num_workers):
    loader = DataLoader(
        ImageDataset(8, bad={5}), batch_size=2, num_workers=num_workers, collate_fn=normalized_224()
    )
    results = []
    it = iter(loader)
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        except RuntimeError as err:
            results.append(str(err))
            continue
        results.append(batch.meta)
        del batch
    assert results[0] == [0, 1] and results[3] == [6, 7]
    assert "image 1 of the batch: cannot decode image" in results[2]


def test_dataset_errors_keep_their_type():
    class Failing(PyDataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index == 2:
                raise KeyError("missing")
            return 42 if index == 3 else RGB_PNG

    for num_workers in [0, 2]:
        loader = DataLoader(
            Failing(), batch_size=1, num_workers=num_workers, collate_fn=normalized_224()
        )
        it = iter(loader)
        next(it)
        next(it)
        with pytest.raises(KeyError, match="missing"):
            next(it)
        with pytest.raises(TypeError, match="bytes or \\(bytes, meta\\)"):
            next(it)


def test_sampler_index_out_of_range():
    loader = DataLoader([RGB_PNG], sampler=[0, 3], collate_fn=normalized_224())
    it = iter(loader)
    next(it)
    with pytest.raises(IndexError, match="out of range"):
        next(it)
