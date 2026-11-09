"""Image transforms implemented in Rust, under torchvision-like names.

Each returns a :class:`Transform`; :func:`Compose` chains them. Passed as a
loader's ``transform=``, a ``Transform`` runs in Rust on the loader's threads,
without the interpreter; called directly, ``transform(item)`` takes ``bytes``
or an :class:`~dataloader_rs.Image` and returns an ``Image``.

Images are uint8 after :func:`Decode` and float32 after :func:`ToFloat`, in
HWC layout until :func:`ToChw`. A transform given an image it does not take
(``Normalize`` before ``ToFloat``, ``HFlip`` after ``ToChw``) raises
``ValueError`` saying what it takes; so does a corrupt, truncated or too
large image.
"""

from collections.abc import Sequence

from . import Image

__all__ = [
    "CenterCrop",
    "Compose",
    "Crop",
    "Decode",
    "HFlip",
    "Normalize",
    "Pad",
    "PadTo",
    "Resize",
    "Rotate90",
    "ToChw",
    "ToFloat",
    "ToGray",
    "ToHwc",
    "ToRgb",
    "ToRgba",
    "Transform",
    "VFlip",
]

class Transform:
    """Image transforms implemented in Rust, applied in order."""

    def __call__(self, item: bytes | bytearray | Image) -> Image:
        """Apply the transforms to ``item``, with the interpreter released."""
        ...

def Decode(
    *, max_width: int = 16384, max_height: int = 16384, max_alloc: int = 268435456
) -> Transform:
    """Decode PNG, JPEG, WebP, TIFF or BMP bytes (the format told by their
    content) into a uint8 HWC image with the image's own channels (1 gray, 2
    gray + alpha, 3 RGB, 4 RGBA). Pure-Rust decoders. An image wider or taller
    than the limits, or needing more than ``max_alloc`` bytes, is refused from
    its header."""
    ...

def ToRgb() -> Transform:
    """To 3 channels, as Pillow's ``convert("RGB")``."""
    ...

def ToRgba() -> Transform:
    """To 4 channels, as Pillow's ``convert("RGBA")``."""
    ...

def ToGray() -> Transform:
    """To 1 channel, as Pillow's ``convert("L")``."""
    ...

def Crop(top: int, left: int, height: int, width: int) -> Transform:
    """The ``height x width`` region at ``(top, left)``, as torchvision's
    ``F.crop``; a region outside the image raises ``ValueError``."""
    ...

def CenterCrop(size: int | tuple[int, int]) -> Transform:
    """The central region, as torchvision's ``CenterCrop`` (zero-padded if
    the image is smaller)."""
    ...

def Pad(
    padding: int | tuple[int, int] | tuple[int, int, int, int],
    fill: float | Sequence[float] | None = None,
    padding_mode: str = "constant",
) -> Transform:
    """Padding, as torchvision's ``Pad``: ``padding_mode`` is ``"constant"``
    (``fill``, 0 by default, one value or one per channel), ``"edge"``,
    ``"reflect"`` or ``"symmetric"``."""
    ...

def PadTo(
    width: int,
    height: int,
    fill: float | Sequence[float] | None = None,
    anchor: str = "top_left",
) -> Transform:
    """Pad with ``fill`` (default 0) up to ``width x height``, never cropping.
    ``anchor="top_left"`` pads at the right and bottom; ``"center"`` pads
    evenly."""
    ...

def Resize(
    size: int | tuple[int, int],
    interpolation: str = "bilinear",
    max_size: int | None = None,
    antialias: bool = True,
) -> Transform:
    """Resize a uint8 image, as torchvision's ``Resize``: ``size`` is
    ``(height, width)``, or an int for the shorter side. ``interpolation``:
    ``"nearest"``, ``"bilinear"``, ``"bicubic"`` or ``"lanczos3"``. Close to
    Pillow's values, not bit-exact."""
    ...

def HFlip() -> Transform:
    """Mirror left to right."""
    ...

def VFlip() -> Transform:
    """Mirror top to bottom."""
    ...

def Rotate90(k: int = 1) -> Transform:
    """Rotate by ``k`` quarter turns counter-clockwise, as ``numpy.rot90``."""
    ...

def ToFloat(scale: bool = True) -> Transform:
    """uint8 to float32, divided by 255 if ``scale``: torchvision's
    ``ToTensor`` values, bit for bit."""
    ...

def Normalize(mean: Sequence[float], std: Sequence[float]) -> Transform:
    """``(x - mean) / std`` per channel of a float32 image, as torchvision's
    ``Normalize``, bit for bit."""
    ...

def ToChw() -> Transform:
    """HWC to CHW: the layout change of torchvision's ``ToTensor``."""
    ...

def ToHwc() -> Transform:
    """CHW to HWC."""
    ...

def Compose(transforms: Sequence[Transform]) -> Transform:
    """The ``transforms`` in order, as one ``Transform``."""
    ...
