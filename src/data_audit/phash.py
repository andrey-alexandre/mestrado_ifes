"""Perceptual hash via 2D DCT (pHash), the second-stage filter of the task 8
staged pipeline.

Why a second hash and not just a stricter dHash threshold: task 7's dHash
(9x8 grayscale gradient, 64 bits) found a huge real candidate count
(421,196 across the two priority pairs) because coarse 9x8 gradients are not
discriminative enough for dermoscopy images, which share strong acquisition
structure (centered lesion, consistent framing/lighting). Tightening dHash's
own threshold would not fix this — it is a resolution problem, not a
threshold problem: dHash's 9x8 grid simply does not encode enough
information to tell two different centered round lesions apart. pHash uses
far more information per image (a 32x32 resize, keeping a 16x16 low-frequency
DCT block = 256 bits, vs dHash's 64) and captures global frequency-domain
structure (smooth intensity variation across the whole image) rather than a
single row of local gradients, which is a different and more discriminative
signal — but still cheap (no training, no model weights, scipy's DCT is
already available in this environment).

Detection only, same as ``visual_duplicates.py``: this module computes and
compares hashes. No decision about exclusion is made or implied here.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn

PHASH_RESIZE_SIZE = 32  # image resized to this x this before the DCT
PHASH_LOW_FREQ_SIZE = 16  # top-left low-frequency DCT block kept -> 16*16 = 256-bit hash

_READ_ERRORS = (UnidentifiedImageError, OSError, zipfile.BadZipFile, RuntimeError)


def phash_bytes(
    data: bytes, resize_size: int = PHASH_RESIZE_SIZE, low_freq_size: int = PHASH_LOW_FREQ_SIZE
) -> int:
    """Compute the pHash of an image given its raw encoded bytes.

    Decodes to grayscale, resizes to ``resize_size x resize_size``, takes a
    2D DCT (``scipy.fft.dctn``, orthonormal), keeps the top-left
    ``low_freq_size x low_freq_size`` low-frequency block (coarse structure;
    the rest is high-frequency detail/noise that reprocessing most disturbs),
    and sets each bit to 1 where that block's value is above the block's own
    median. The DC term ([0, 0]) is deliberately left in when computing the
    median rather than excluded: with ``low_freq_size=16`` the block holds
    256 values, so one outlier term has negligible effect on the median
    (unlike the classic tiny e.g. 2x2 block case, where excluding it matters).

    Raises ``PIL.UnidentifiedImageError``/``OSError`` on undecodable input —
    same convention as ``visual_duplicates.dhash_bytes``.
    """
    image = Image.open(io.BytesIO(data)).convert("L").resize(
        (resize_size, resize_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(image, dtype=np.float64)
    dct = dctn(pixels, norm="ortho")
    low_freq = dct[:low_freq_size, :low_freq_size]
    median = np.median(low_freq)
    bits = (low_freq > median).astype(np.uint8)
    packed = np.packbits(bits.flatten())
    return int.from_bytes(packed.tobytes(), byteorder="big")


def phash_disk_images(
    file_paths: Iterable[str | Path], base_dir: str | Path, **phash_kwargs: Any
) -> dict[str, Any]:
    """pHash manifest for image files already extracted on disk (HAM10000).

    Same path/coverage convention as ``visual_duplicates.dhash_disk_images``.
    """
    base = Path(base_dir)
    paths = list(file_paths)
    manifest: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for fp in paths:
        fp = Path(fp)
        rel_path = str(fp.relative_to(base)).replace("\\", "/")
        try:
            data = fp.read_bytes()
            h = phash_bytes(data, **phash_kwargs)
        except _READ_ERRORS as exc:
            unreadable.append({"path": rel_path, "error": str(exc)})
            continue
        manifest.append({"path": rel_path, "phash": h})
    return {
        "expected": len(paths),
        "processed": len(manifest),
        "unreadable": unreadable,
        "manifest": manifest,
    }


def phash_zip_entries(
    zip_path: str | Path, entry_names: Iterable[str], **phash_kwargs: Any
) -> dict[str, Any]:
    """pHash manifest for image entries read directly from a zip (ISIC2018
    Task1, BCN20000 train/test), without extracting anything to disk.

    Same single-open-handle / tolerate-unreadable-entry pattern as
    ``visual_duplicates.dhash_zip_entries``.
    """
    entry_names = list(entry_names)
    manifest: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in entry_names:
            try:
                data = zf.read(name)
                h = phash_bytes(data, **phash_kwargs)
            except _READ_ERRORS as exc:
                unreadable.append({"path": name, "error": str(exc)})
                continue
            manifest.append({"path": name, "phash": h})
    return {
        "expected": len(entry_names),
        "processed": len(manifest),
        "unreadable": unreadable,
        "manifest": manifest,
    }
