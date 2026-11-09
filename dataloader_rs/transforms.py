"""Image transforms implemented in Rust, under torchvision-like names.

Each returns a :class:`Transform`; :func:`Compose` chains them. Passed as a
loader's ``transform=``, a ``Transform`` runs in Rust on the loader's threads,
without the interpreter; called directly, ``transform(item)`` takes ``bytes``
or an :class:`~dataloader_rs.Image` and returns an ``Image``.
"""

from .dataloader_rs import (
    CenterCrop,
    Compose,
    Crop,
    Decode,
    HFlip,
    Normalize,
    Pad,
    PadTo,
    Resize,
    Rotate90,
    ToChw,
    ToFloat,
    ToGray,
    ToHwc,
    ToRgb,
    ToRgba,
    Transform,
    VFlip,
)

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
