# dataloader-rs for Python

A `DataLoader` for Python with the `torch.utils.data.DataLoader` interface and a Rust core. Workers
are threads, not processes: samples are handed over by reference instead of being pickled, an
epoch starts in milliseconds, and against PyTorch it is **4.9–6.6× faster on the loader's own
overhead, 3.2–3.9× on network-bound datasets and 2.5–3.5× on array samples** with free-threaded
Python — with no PyTorch dependency.

- **PyTorch's arguments**: `batch_size`, `shuffle`, `sampler`, `num_workers`, `collate_fn`,
  `drop_last`, plus `seed` for reproducible shuffling, `prefetch_depth`, and `item_concurrency`
  to fetch the items of a batch concurrently.
- **Any batch type**: lists by default, NumPy arrays or tensors from your `collate_fn`.
- **Batched fetching**: `__getitems__(indices)` is used when your dataset defines it.
- **Async datasets**: `async def __getitem__` on a `PyAsyncDataset`, or the built-in
  `HttpDataset`, `FileDataset` and `ObjectStoreDataset`, which fetch in Rust without the GIL:
  64 requests in flight from two worker threads.
- **Image transforms in Rust**: `dataloader_rs.transforms` (`Decode`, `PadTo`, `Resize`,
  `Normalize`, …, `Compose`) decode and transform images on the loader's threads without the
  GIL, bit for bit like Pillow + torchvision where it matters; `ImageCollator` batches them in
  one buffer that `torch.from_dlpack` takes without a copy.
- **Exceptions keep their type**: a `KeyError` in `__getitem__` reaches you as a `KeyError`.
- **Free-threaded Python**: on CPython 3.14t and 3.15t, worker threads run Python code in
  parallel. Wheels for the GIL builds (3.11 to 3.15) too: there, threads help work that releases
  the GIL (see below).

> **When PyTorch is faster.** The workers are threads in one process, not processes. On regular
> CPython (the GIL build, such as 3.13), only one thread runs Python code at a time: a
> `__getitem__` that spends its time in pure Python (tokenising text, parsing records, Python
> loops) gets no faster with more workers, while PyTorch's worker processes each run their own
> interpreter and are 2.4–2.6× faster on such datasets. Threads win when the work releases the
> GIL (file and network I/O, NumPy, image decoding), and on free-threaded CPython 3.14t / 3.15t.
> Even there, threads that all read one shared Python dict or list contend on its reference
> count, and PyTorch is 2.5–2.9× faster: give each thread its own copy (`threading.local`, see
> [Workers, threads and the GIL](#workers-threads-and-the-gil)).

Requires Python ≥ 3.11.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [PyDataloader arguments](#pydataloader-arguments)
- [Batches and `collate_fn`](#batches-and-collate_fn)
- [Transforms](#transforms)
- [Image transforms in Rust](#image-transforms-in-rust)
- [Samplers](#samplers)
- [Workers, threads and the GIL](#workers-threads-and-the-gil)
- [I/O-bound datasets](#io-bound-datasets)
- [Async datasets](#async-datasets)
- [Errors](#errors)
- [Coming from `torch.utils.data.DataLoader`](#coming-from-torchutilsdatadataloader)
- [Performance](#performance)

## Installation

```bash
pip install dataloader-rs
```

| Wheels | Platforms |
|---|---|
| CPython 3.11 to 3.15 and free-threaded 3.14t, 3.15t | Linux x86_64 (x86-64-v3 CPU: AVX2, Intel 2013+ / AMD 2017+) and arm64 (glibc ≥ 2.28), macOS arm64 (11.0+) |

Other interpreters and platforms build from the source distribution, which needs a Rust
toolchain; from a checkout: `pip install maturin && maturin develop --release --features python`.

## Quick start

```python
import numpy as np
from dataloader_rs import PyDataloader, PyDataset


class Images(PyDataset):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return np.load(self.paths[index])


loader = PyDataloader(
    Images(paths),
    batch_size=32,
    shuffle=True,
    seed=0,  # reproducible order, a new permutation every epoch
    num_workers=4,  # 4 worker threads
    prefetch_depth=4,  # load up to 4 batches ahead of the 4 in progress
    collate_fn=np.stack,  # one (32, ...) array per batch
    drop_last=True,
)

for epoch in range(10):
    for batch in loader:  # iterating again starts a new epoch
        ...
```

## Datasets

A dataset is a source: it only fetches items, synchronously (`PyDataset`) or asynchronously
(`PyAsyncDataset` and the Rust datasets, see [Async datasets](#async-datasets)). What is done to
each item afterwards (decoding, augmentation) is the loader's `transform` (see
[Transforms](#transforms)).

Subclass `PyDataset` and implement `__len__` and `__getitem__(index)`. Call `super().__init__()`
if you define `__init__`.

Optionally implement `__getitems__(indices)` — PyTorch's batched-fetch protocol — to fetch a whole
batch in one call: one database query, one NumPy fancy-indexing, one Python call instead of
`batch_size`. It receives a `list[int]` and must return one sample per index, in order (any
iterable); a wrong count raises `ValueError`. Setting `__getitems__ = None` in a subclass turns it
off, as in PyTorch.

```python
class Rows(PyDataset):
    def __init__(self, table: np.ndarray) -> None:
        super().__init__()
        self.table = table

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.table[index]

    def __getitems__(self, indices: list[int]) -> list[np.ndarray]:
        return list(self.table[indices])  # one vectorised read per batch
```

With `num_workers > 0`, `__getitem__` / `__getitems__` run on several threads at once: keep the
dataset read-only, or protect shared state with a lock.

## PyDataloader arguments

| Argument | Default | Meaning |
|---|---|---|
| `dataset` | — | a `PyDataset`, a `PyAsyncDataset`, or a Rust dataset (`HttpDataset`, `FileDataset`, `ObjectStoreDataset`; see [Async datasets](#async-datasets)) |
| `batch_size` | `1` | samples per batch (> 0) |
| `shuffle` | `False` | a new random order every epoch; exclusive with `sampler` |
| `seed` | `None` | seed for `shuffle=True` (requires it): same seed, same sequence of epoch orders |
| `sampler` | `None` | any iterable of `int` indices, iterated at the start of each epoch |
| `num_workers` | `0` | batches loaded in parallel; `0` loads each batch in the calling thread |
| `prefetch_depth` | `2` | how far ahead workers may load: at most `num_workers + prefetch_depth` batches in progress or waiting (with `num_workers > 0`, > 0) |
| `transform` | `None` | `(index, sample) -> sample`, applied to every item before `collate_fn` (see [Transforms](#transforms)) |
| `collate_fn` | `None` | `list[sample] -> batch`; without it a batch is a `list` |
| `drop_last` | `False` | drop a final batch smaller than `batch_size` |
| `item_concurrency` | `1` | items of one batch fetched in parallel, one `__getitem__` call each (see [I/O-bound datasets](#io-bound-datasets)); formerly `items_in_parallel=`, still accepted with a `DeprecationWarning` |
| `runtime_threads` | `None` | async and Rust datasets only: worker threads of the loader's tokio runtime, 1 to 1024; `None` = one per CPU available to the process. `ValueError` with a `PyDataset` |

`len(loader)` is the number of batches per epoch; `len(iterator)` the number left in the current
one.

## Batches and `collate_fn`

Without `collate_fn` a batch is the `list` of its samples. `collate_fn` receives that list and
returns anything: `np.stack`, `torch.stack`, a dict of arrays. It runs on the worker thread that
loaded the batch (or the calling thread with `num_workers=0`), so heavy collation is parallel too.

## Transforms

`transform=f` calls `f(index, sample)` for every item once the dataset has fetched it, before
`collate_fn`: `index` is the item's dataset index, as the sampler drew it (shuffled or not), the
same order as Rust's `Transform::apply(index, item)`. It runs on the loader's threads, holding
the interpreter only for the call: right after `__getitem__` on the thread that fetched the item
(the calling thread for a `PyDataset` with `num_workers=0` and `item_concurrency=1`), or on the
loader's pool as each item of an async or Rust dataset arrives. With `item_concurrency > 1` the
items of a batch are transformed in parallel. An exception it raises fails its batch and keeps
its type.

A one-argument callable, such as a torchvision `Compose`, is wrapped:

```python
from torchvision import transforms

compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
loader = PyDataloader(dataset, batch_size=32, transform=lambda i, x: compose(x))
```

The index finds what belongs to each item without carrying it through the dataset. Tiles of a
slide, fetched in Rust, decoded on the loader's threads, batched with their coordinates:

```python
import io

import numpy as np
from PIL import Image
from dataloader_rs import HttpDataset, PyDataloader

coords = np.array(tile_coords)  # (n, 2): x, y of each tile at level 0
tiles = HttpDataset.from_template(
    "https://imgsrv.example/slides/slide-1/tile?x={x}&y={y}&w=224&h=224&f=png",
    {"x": coords[:, 0], "y": coords[:, 1]},
    headers={"Authorization": f"Bearer {token}"},
)


def decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def stack(items):
    images, tile_coords = zip(*items)
    return np.stack(images), np.stack(tile_coords)


loader = PyDataloader(
    tiles,
    batch_size=64,
    num_workers=2,
    item_concurrency=32,
    transform=lambda i, png: (decode(png), coords[i]),
    collate_fn=stack,
)
for images, tile_coords in loader:  # (64, 224, 224, 3) and (64, 2)
    save(model(images), tile_coords)
```

## Image transforms in Rust

`dataloader_rs.transforms` has torchvision-like image transforms implemented in Rust (pure-Rust
decoders, no C library). Pass a `Compose` of them as `transform=`: the loader runs it on its
threads **without the interpreter**, so decoding scales with threads even on the GIL build, and
with `collate_fn=ImageCollator()` the batch is assembled in Rust too.

```python
from dataloader_rs import FileDataset, ImageCollator, PyDataloader
from dataloader_rs import transforms as T

tiles = T.Compose(
    [
        T.Decode(),  # PNG, JPEG, WebP, TIFF, BMP bytes → uint8 HWC
        T.ToRgb(),  # Image.convert("RGB")
        T.PadTo(224, 224),  # prime_dx ConstantFill2d((224, 224), 0)
        T.ToFloat(),  # ToTensor's values (/ 255)
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.ToChw(),  # ToTensor's layout
    ]
)
loader = PyDataloader(
    FileDataset(paths),
    batch_size=64,
    num_workers=4,
    item_concurrency=4,
    transform=tiles,
    collate_fn=ImageCollator(buffers=8),  # one float32 buffer per batch, reused
)
for batch in loader:  # ImageBatch (64, 3, 224, 224)
    x = torch.from_dlpack(batch)  # no copy; .to("cuda") as usual
```

| `dataloader_rs.transforms` | torchvision / Pillow | Takes → gives |
|---|---|---|
| `Decode(max_width=16384, max_height=16384, max_alloc=256 MiB)` | `Image.open(...)`; `decode_image` | bytes → uint8 HWC, the image's channels (1, 2, 3, 4) |
| `ToRgb()`, `ToRgba()`, `ToGray()` | `convert("RGB" / "RGBA" / "L")` | uint8 HWC |
| `Crop(top, left, height, width)` | `F.crop` | HWC, either type |
| `CenterCrop(size)` | `CenterCrop` | HWC, either type |
| `Pad(padding, fill=0, padding_mode="constant")` | `Pad` (constant, edge, reflect, symmetric) | HWC, either type |
| `PadTo(width, height, fill=0, anchor="top_left")` | prime_dx `ConstantFill2d`; `"center"`: even padding | HWC, either type; never crops |
| `Resize(size, interpolation="bilinear", max_size=None, antialias=True)` | `Resize` (nearest, bilinear, bicubic, lanczos3) | uint8 HWC |
| `HFlip()`, `VFlip()`, `Rotate90(k=1)` | `F.hflip`, `F.vflip`, `numpy.rot90` | HWC, either type |
| `ToFloat(scale=True)` | `ToTensor` values, `ToDtype(float32, scale=True)` | uint8 → float32 |
| `Normalize(mean, std)` | `Normalize` | float32, HWC or CHW |
| `ToChw()`, `ToHwc()` | `ToTensor`'s permute | layout |
| `Compose(transforms)` | `Compose` | — |

- **Same values**: decoding (PNG, BMP, TIFF, lossless WebP), color conversions, crops, padding,
  flips, rotations, `ToFloat` and `Normalize` give Pillow's / torchvision's values bit for bit;
  `Decode → ToRgb → PadTo → ToFloat → Normalize` equals prime_dx's `ConstantFill2d + ToTensor +
  Normalize` on RGB, RGBA, gray, palette and undersized tiles. JPEG decoding differs by at most
  a few levels (another IDCT); `Resize` is within 1 level of Pillow for bilinear, a few for
  bicubic and Lanczos. torchvision v2's `ToDtype(scale=True)` multiplies by 1/255 and differs
  from `ToTensor` (and `ToFloat`) by one ulp on half the values.
- **Errors**: a corrupt or truncated image, one above `Decode`'s limits (checked from the header,
  before decoding: decompression bombs are refused), or a transform given an image it does not
  take (`Normalize` before `ToFloat`) raises `ValueError` for its batch, saying why.
- **Called directly**, `tiles(png_bytes)` returns an `Image`; `np.asarray(image)` is a read-only
  view of it. A Python `transform` can call them, with the interpreter released while they run.
- **Without `ImageCollator`**, a batch is a `list` of `Image`s, or whatever `collate_fn` makes of
  them.

`ImageCollator(layout="NCHW", dtype=None, buffers=0)`: one `ImageBatch` per batch, `"NCHW"` or
`"NHWC"`; `dtype` (`"uint8"` or `"float32"`) checks the items' type; images of different shapes
fail the batch. `np.asarray(batch)` and `torch.from_dlpack(batch)` are writable views of its
buffer, without a copy, and keep it alive. `buffers=n` keeps the buffers of up to `n` dropped
batches and writes the next ones into them: a fresh 38.5 MB float32 batch spends three quarters
of its collation on page faults. It never waits for a buffer, so holding batches only costs
memory.

**Per-item metadata.** Batches follow the sampler's order (sequential by default), so batch `k`
holds items `k * batch_size` and on: slice your coordinates the same way. Or pair each image
with its metadata in a Python transform; `ImageCollator` then returns `(batch, [metadata])`:

```python
coords = np.array(tile_coords)  # (n, 2)
tiles_http = HttpDataset.from_template(
    "https://imgsrv.example/slides/slide-1/tile?x={x}&y={y}&w=224&h=224&f=png",
    {"x": coords[:, 0], "y": coords[:, 1]},
)
loader = PyDataloader(
    tiles_http,
    batch_size=64,
    num_workers=2,
    item_concurrency=32,
    transform=lambda i, png: (tiles(png), coords[i]),  # Rust work, GIL released inside
    collate_fn=ImageCollator(),
)
for batch, batch_coords in loader:  # ImageBatch, list of 64 coordinate rows
    save(model(torch.from_dlpack(batch)), np.stack(batch_coords))
```

The first form keeps the interpreter out entirely; the second attaches once per item for the
lambda.

Runnable versions, offline (they write their own tiles): [`examples/python/`](../examples/python/)
— all Rust, Rust and torch mixed, and torchvision transforms on Rust-fetched bytes.

**Back from numpy or torch.** `Image.from_array(obj, layout="hwc")` makes an `Image` of a numpy
array, a CPU torch tensor or anything `np.asarray` takes, so a Python transform can mix Rust and
torch steps: Rust decode, a torch op on `torch.from_dlpack(image)` (a view), `Image.from_array`,
then Rust `Normalize` and `ImageCollator`.

```python
decode = T.Compose([T.Decode(), T.ToRgb(), T.PadTo(224, 224)])
normalize = T.Compose([T.ToFloat(), T.Normalize(MEAN, STD)])


def transform(i, png):
    chw = torch.from_dlpack(decode(png)).permute(2, 0, 1)  # no copy
    chw = augment(chw)  # any torch / torchvision op on uint8 CHW
    return normalize(Image.from_array(chw, layout="chw")), coords[i]
```

- Values are `uint8` or `float32` (others raise `TypeError`, never converted); the shape is
  `(height, width)` or, in `layout`, `(height, width, channels)` (`"hwc"`) or `(channels, height,
  width)` (`"chw"`), with 1, 3 or 4 channels and `Decode`'s limits (16384 × 16384, 256 MiB), else
  `ValueError`. A GPU tensor raises `TypeError`: `.cpu()` it first.
- The image is HWC, so every transform and `ImageCollator` take it. A contiguous HWC array is
  copied once, into Rust's buffer; a CHW or strided one is made contiguous by numpy first (one
  more copy, several times faster than reading it strided). The interpreter is held during the
  copy.
- Without numpy, objects with the buffer protocol and their own shape (a `memoryview` cast to
  `(h, w, c)`) are read, `"hwc"` only.

## Samplers

- `shuffle=True`: a Rust permutation, new every epoch; with `seed=` two loaders produce the same
  sequence of orders.
- `sampler=`: any finite iterable of indices — a list, a generator-producing object, a weighted
  sampler. It is read whole at the start of each epoch (`iter(loader)`), so an endless sampler
  (`itertools.cycle`, the infinite samplers of iteration-based training) never starts one: give
  each epoch a finite sampler instead (Ctrl-C stops one that never ends). `len(loader)` uses
  `len(sampler)`, so a sampler without `__len__` makes `len(loader)` raise `TypeError`, as in
  PyTorch; iteration still works.

```python
import random


class Weighted:
    def __init__(self, weights: list[float]) -> None:
        self.weights = weights

    def __len__(self) -> int:
        return len(self.weights)

    def __iter__(self):
        population = range(len(self.weights))
        yield from random.choices(population, weights=self.weights, k=len(self.weights))


loader = PyDataloader(dataset, batch_size=8, sampler=Weighted(weights))
```

## Workers, threads and the GIL

`num_workers > 0` loads that many batches in parallel on a thread pool of
`num_workers × item_concurrency` threads, created once and reused by every epoch; they load
batches ahead while your loop runs and the loader returns them in order.

- **Regular CPython**: the threads share the GIL. Work that releases it — file reads, NumPy,
  image decoding, hashing, most C extensions — runs in parallel; pure-Python code in
  `__getitem__` does not, and for such datasets PyTorch's worker processes are faster (2.4–2.6×
  in the benchmarks). Use free-threaded CPython, or keep the per-item work in code that releases
  the GIL.
- **Free-threaded CPython 3.14t / 3.15t**: no GIL, so pure-Python code runs in parallel too.

  ```bash
  uv python install 3.14t
  uv run --python 3.14t python -X gil=0 train.py
  ```

- **Shared Python objects on free-threaded CPython**: an object's reference count is cheap only for
  the thread that created it; other threads update it atomically. Threads that all read the same
  dict or list millions of times per second (a vocabulary, a lookup table) contend on it and stop
  scaling: a pure-Python tokeniser over one shared vocabulary runs *slower* on 8 threads than on 1
  with 3.14t. Give each thread its own copy of hot lookup tables (`threading.local`), or on 3.15t
  key them with interned strings (`sys.intern`), which no longer carry this cost. Processes
  (PyTorch's workers) never share objects, so they do not see this.

- **Array samples on Linux: tune the allocator NumPy uses.** With default settings, the memory of
  every batch goes back to the system and is faulted back in, which serialises worker threads.
  The settings are process-wide, so they are yours to make, at start-up:

  ```bash
  # CPython 3.13 and 3.14t: NumPy allocates with glibc's malloc
  GLIBC_TUNABLES=glibc.malloc.trim_threshold=67108864:glibc.malloc.mmap_threshold=33554432 python train.py
  # CPython 3.15t: NumPy's buffers come from CPython's mimalloc
  MIMALLOC_PURGE_DELAY=-1 python train.py
  ```

  On 3.14t this is worth up to 2× for array samples with 4 workers, and the process keeps some
  free memory for reuse (80–100 MB in the benchmarks). With glibc, raise both thresholds: either
  one alone turns off glibc's adjustment of the other and is slower than the defaults. Batches of
  16 MiB and more hit a mimalloc path no setting avoids on 3.15t: smaller batches are faster
  there.
- **Breaking out of a loop** returns immediately; batches already being loaded finish in the
  background and are discarded.
- **Ctrl-C** interrupts `next()` within 100 ms while it waits for workers, and the iterator stays
  usable. A batch fetched on your own thread (`item_concurrency > 1` without workers) runs to the
  end first.
- **Any thread** can create, iterate and drop a loader, one thread at a time per object: a second
  thread calling `next()` on an iterator that another is advancing gets `RuntimeError`.
- **Worker threads have an 8 MiB stack**, a main thread's usual size, so a `__getitem__` recurses
  as deep as it would on your thread.
- **`multiprocessing` with `fork`**: a child process can use a loader its parent already
  iterated; the loader rebuilds its threads there.

## I/O-bound datasets

By default a worker fetches the items of its batch one after another, so a dataset that waits
on the network (an object store, an HTTP API, a database) keeps only `num_workers` requests in
flight. With `item_concurrency=k`, up to `k` items of each batch are fetched concurrently, one
`__getitem__` call each, so `num_workers × item_concurrency` requests are in flight:

```python
loader = PyDataloader(
    S3Images(keys),
    batch_size=32,
    num_workers=4,  # 4 batches in parallel
    item_concurrency=16,  # 16 items of each batch at once: 64 requests in flight
)
```

Waiting on I/O releases the GIL, so this works on regular CPython too. `__getitems__` is not used
when `item_concurrency > 1`: each item is its own call.

Each request in flight here holds a thread. For many requests, an [async dataset](#async-datasets)
holds a future instead.

## Async datasets

Two kinds of datasets are fetched by futures on the loader's async runtime instead of threads.
For both, `num_workers` is the number of batches in flight (at least one), `item_concurrency`
the number of items of each batch in flight, so `num_workers × item_concurrency` requests run at
once, from a few threads. The other arguments, the order of batches and the errors are the same
as for `PyDataset`.

| Setting | Sizes | Rule |
|---|---|---|
| `num_workers × item_concurrency` | requests in flight | latency × wanted items/s; far more than cores |
| `runtime_threads` | CPU for network receive, TLS, body copy (tokio) | default (CPUs available) is enough; lower it to leave cores to training |
| the loader's pool (`num_workers × item_concurrency` threads, at most the CPUs available, with a `transform` or a Rust dataset; else `num_workers`) | `transform` and `collate_fn` | the CPU-bound part |

**`PyAsyncDataset`**: subclass it and write `async def __getitem__` (and optionally
`async def __getitems__(indices)`, used when `item_concurrency=1`). The loader runs the
coroutines on an asyncio event loop of its own, on a background thread; dropping an iterator
cancels the coroutines in flight. Objects tied to an event loop (a client session, open
connections) must be created inside the coroutines, on the loader's loop, and released by an
optional `async def aclose(self)`: when the loader is dropped, it awaits `aclose()` once on its
loop, then stops the loop. An exception from `aclose()`, or an `aclose()` still running after
5 seconds, is logged (logger `dataloader_rs`) and the loop stops anyway.

```python
import aiohttp
from dataloader_rs import PyAsyncDataset, PyDataloader


class Tiles(PyAsyncDataset):
    def __init__(self, urls: list[str]) -> None:
        super().__init__()
        self.urls = urls
        self.session = None

    def __len__(self) -> int:
        return len(self.urls)

    async def __getitem__(self, index: int) -> bytes:
        if self.session is None:  # created on the loader's event loop
            self.session = aiohttp.ClientSession()
        async with self.session.get(self.urls[index]) as response:
            return await response.read()

    async def aclose(self) -> None:  # awaited on the loader's loop when it is dropped
        if self.session is not None:
            await self.session.close()


loader = PyDataloader(Tiles(urls), batch_size=64, num_workers=2, item_concurrency=32)
```

**Rust datasets**: items are `bytes`, fetched in Rust without holding the interpreter; the
loader's `transform` turns them into what you need.

| Dataset | Item `i` | Options |
|---|---|---|
| `HttpDataset(urls)` | body of a GET of `urls[i]`; one client, keep-alive | `headers`, `timeout` (s, per attempt, body included; 30), `connect_timeout` (10), `retries` (3), `backoff`, `max_backoff`, `max_bytes` |
| `HttpDataset.from_template(template, columns, *, length)` | body of a GET of the URL made from `template` with the values of item `i` | the options of `HttpDataset(urls)` |
| `FileDataset(paths)` | content of `paths[i]` | `timeout` (s, per attempt; `None`), `retries` (0), `backoff`, `max_backoff`, `max_bytes` |
| `FileDataset.from_template(template, columns, *, length)` | content of the path made from `template` with the values of item `i` | the options of `FileDataset(paths)` |
| `ObjectStoreDataset(url, keys)` | object `keys[i]` under the store and prefix of `url` (`s3://`, `gs://`, `az://`, `https://`) | `options` (`object_store` keys such as `aws_region`; credentials also from `AWS_*`, `GOOGLE_*`, `AZURE_*`), `timeout` (s, per request; `None`: 30), `retries` (3), `backoff`, `max_backoff`, `max_bytes` |
| `ObjectStoreDataset.from_template(url, template, columns, *, length)` | object whose key, under the prefix of `url`, is made from `template` with the values of item `i` | the options of `ObjectStoreDataset(url, keys)` |

Every `from_template` makes the locations in Rust, item by item: no string per item is kept.
`length` is required without `columns`. The retry options are the same everywhere: up to
`retries` more attempts after a transient failure, waiting `backoff × 2ⁿ` seconds with jitter,
at most `max_backoff` (0.1 and 5 by default).

```python
import io

import numpy as np
from PIL import Image
from dataloader_rs import HttpDataset, PyDataloader

tiles = HttpDataset(
    [f"https://tiles.example/slide/{x}_{y}.png" for x, y in coords],
    headers={"Authorization": f"Bearer {token}"},
    retries=5,  # 408, 429, 5xx and connection errors, with exponential backoff
)
loader = PyDataloader(
    tiles,
    batch_size=64,
    num_workers=2,
    item_concurrency=32,
    transform=lambda i, png: np.asarray(Image.open(io.BytesIO(png))),
    collate_fn=np.stack,
)
```

For many URLs that differ only by a few values (34 000 tiles of a slide, say), give a template
and the values instead of a list of URLs. The template is parsed once; each URL is made in Rust
when its item is fetched:

```python
cols, rows = np.meshgrid(np.arange(200), np.arange(170))  # 34 000 (col, row) coordinates
tiles = HttpDataset.from_template(
    "https://imgsrv.example/slides/{slide}/tile?x={x}&y={y}&w=224&h=224&l=12&f=png&q=100",
    {"slide": ["slide 1.svs"] * cols.size, "x": cols.ravel() * 224, "y": rows.ravel() * 224},
    headers={"Authorization": f"Bearer {token}"},
)
```

`{name}` is the value of column `name` for the item, `{index}` the item index, `{{` and `}}`
literal braces. A column is a sequence or a numpy array (any integer dtype) of ints, written in
decimal, or of strs, percent-encoded as a URL component (`"slide 1.svs"` → `slide%201.svs`; `.`
and `..` are refused). All columns have the same length, the number of items; with `{index}`
only, pass `length=`. A placeholder without a column, a column the template does not use (a
typo, likely), unbalanced braces, or an invalid first URL raise `ValueError`; a column that is
not ints or strs raises `TypeError`.

Files and objects take templates the same way. In a path or a key a str is written as it is,
but a value cannot hold `/` (nor `\` or NUL in a path) or be `.` or `..`, so it cannot leave
the template's directory (`ValueError`); a key part cannot be empty either:

```python
from dataloader_rs import FileDataset, ObjectStoreDataset

slides = ["a.svs", "a.svs", "b.svs"]
# /mnt/tiles/a.svs/0.png, /mnt/tiles/a.svs/1.png, /mnt/tiles/b.svs/2.png
files = FileDataset.from_template(
    "/mnt/tiles/{slide}/{index}.png",
    {"slide": slides},
    timeout=10.0,  # a hung network file system fails the attempt...
    retries=2,  # ...which is tried again
)
# s3://bucket/tiles/a.svs/0.png, ...: the template is relative to the prefix of the URL
objects = ObjectStoreDataset.from_template(
    "s3://bucket/tiles",
    "{slide}/{index}.png",
    {"slide": slides},
    options={"aws_region": "eu-west-3"},
)
```

What is retried depends on the dataset. `HttpDataset`: connection errors, timeouts and the
statuses 408, 429, 500, 502, 503 and 504, honouring a `Retry-After` up to `max_backoff`; other
statuses fail at once. `FileDataset`: attempts over `timeout` and the I/O errors `TimedOut`,
`Interrupted` and `WouldBlock`; none by default. `ObjectStoreDataset`: the `object_store` crate
retries by itself (5xx, 429, 408, connection errors, timeouts, within 3 minutes of the first
attempt) with these settings; set its timeout with `timeout=`, not in `options`. Every item is
capped at `max_bytes` (64 MiB by default), checked before and while reading. TLS uses the
bundled Mozilla root certificates, not the system store.

The loader checks a dataset when it is built: `async def` methods on a `PyDataset`, plain ones on
a `PyAsyncDataset` (`aclose` included), a class with both bases, or a missing `__len__` raise
`TypeError`. A plain
`__getitem__` that returns a coroutine (an `async def` helper called without `await`) raises
`TypeError` on the first item.

## Errors

- An exception raised by `__getitem__`, `__getitems__` (sync or async), `transform` or
  `collate_fn` reaches you with its own type and message, and the next `next()` continues with
  the following batch, with or without workers, as in PyTorch.
- The Rust datasets raise `OSError` for a failed request or read (with the status for HTTP),
  `TimeoutError` when the last attempt ran out of time, `ValueError` for an item over
  `max_bytes`, `IndexError` for an index out of range. Messages show URLs without their query
  or credentials.
- A `__len__` or a sampler that raises, or a sampler that yields something other than a
  non-negative `int`, makes `iter(loader)` raise — an epoch never silently comes out empty or
  short.
- Invalid arguments raise `ValueError` (`batch_size=0`, `shuffle` with `sampler`, `seed` without
  `shuffle`) or `TypeError` (a non-iterable `sampler`).

## Coming from `torch.utils.data.DataLoader`

| PyTorch | dataloader-rs |
|---|---|
| `torch.utils.data.Dataset` | subclass `PyDataset` (same `__len__` / `__getitem__` / `__getitems__`) |
| `batch_size`, `shuffle`, `sampler`, `num_workers`, `collate_fn`, `drop_last` | same |
| `generator=` for a reproducible shuffle | `seed=` |
| `prefetch_factor` (per worker) | `prefetch_depth` (total, beyond the `num_workers` batches in progress) |
| worker processes, `persistent_workers`, `worker_init_fn`, `multiprocessing_context` | threads: nothing to start, copy or initialise per worker |
| a `Dataset` doing blocking I/O in worker processes | `PyAsyncDataset`, or `HttpDataset` / `FileDataset` / `ObjectStoreDataset` |
| transforms applied inside `__getitem__` | `transform=f`, applied by the loader to any dataset as `f(index, item)` |
| Pillow + torchvision image transforms | `dataloader_rs.transforms.Compose([...])`, in Rust, without the GIL |
| `pin_memory`, `timeout`, `batch_sampler`, `IterableDataset` | not supported |
| default collation into tensors | a `list`; pass `collate_fn` (e.g. `np.stack`, `torch.stack`) |

## Performance

Against `torch.utils.data.DataLoader` with the same dataset code, on 4 cores of an Intel Core
i9-13900H, with the allocator settings of [Workers, threads and the GIL](#workers-threads-and-the-gil)
for both: 4.9–6.6× faster on the loader's own overhead, 3.2–3.9× on network-bound datasets
(`item_concurrency`), first batches in milliseconds against 11–334 ms for torch's fresh worker
processes, and 2.5–3.5× on array samples (file reads, NumPy augmentation) with free-threaded
3.14t / 3.15t, 1.0–1.5× on 3.13. Pure-Python CPU work with per-thread tables is about 1.2×
faster on 3.14t and 3.15t; on shared objects or under the GIL, torch's processes are 2.4–2.9×
faster. Fetching 2 048 tiles of 64 KiB over HTTP (10 ms each, batches of 64), a fresh loader per
epoch: torch with 8 / 16 worker processes (`requests`) takes 3.0 / 1.6 s, `HttpDataset` with 64
requests in flight 0.38 s (`PyAsyncDataset` with aiohttp 0.45 s, `requests` threads 0.48 s on
3.14t and 1.04 s under the GIL), first batch 0.03 s against 0.8 s, on 3.13 and 3.14t alike. Every number, the method and the caveats:
[BENCH.md](https://github.com/AzHicham/dataloader-rs/blob/dev/BENCH.md).

Source, issues and the Rust crate: [github.com/AzHicham/dataloader-rs](https://github.com/AzHicham/dataloader-rs).
