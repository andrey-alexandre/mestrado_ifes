"""Perceptual-hash (dHash) candidate detection for visual duplicates that are
NOT byte-identical (task 7 of the reconstruction; extends task 6's sha256
content-hash comparisons, which only detect byte-identical files).

Why dHash (difference hash): it needs no training, no model weights and no
dependency beyond Pillow+numpy (both already present), it is deterministic,
and it is specifically tolerant of the kind of reprocessing expected between
dataset releases of the same lesion photo (recompression, resizing, small
crop) because it encodes coarse brightness *gradients* across a heavily
downsampled (9x8) grayscale version of the image rather than exact pixel
values. It is NOT tolerant of rotation, mirroring or a large crop, because
those change which gradients fall next to which others — see
``tests/data_audit/test_visual_duplicates.py`` for measured Hamming
distances that document this limit in code, not just in prose.

Detection only: this module finds and scores *candidate* pairs below a
chosen distance threshold. No decision about exclusion is made or implied
here — that is a separate researcher decision informed by the report this
feeds.
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

DHASH_SIZE = 8  # (DHASH_SIZE+1) x DHASH_SIZE pixels -> DHASH_SIZE*DHASH_SIZE = 64-bit hash

# Chosen from measured distances in test_visual_duplicates.py: genuine
# reprocessing of the same image (identical copy, JPEG recompression,
# resize, small/5% crop) measured at distance <= 6; structural changes
# (90-degree rotation, horizontal mirror, large/30% crop) measured at
# distance >= 16. 10 sits with margin on both sides of that gap.
DEFAULT_CANDIDATE_THRESHOLD = 10

_READ_ERRORS = (UnidentifiedImageError, OSError, zipfile.BadZipFile, RuntimeError)


def dhash_bytes(data: bytes, hash_size: int = DHASH_SIZE) -> int:
    """Compute the dHash of an image given its raw encoded bytes.

    Decodes to grayscale, resizes to ``(hash_size + 1, hash_size)`` pixels,
    and sets bit ``i`` to 1 when pixel ``i`` is brighter than its immediate
    right neighbor — a ``hash_size * hash_size``-bit integer (64 bits for
    the default size).

    Raises ``PIL.UnidentifiedImageError``/``OSError`` on undecodable input;
    callers in this module catch that and record it as unreadable instead of
    aborting a batch (same convention as ``core.sha256_zip_entries``).
    """
    image = Image.open(io.BytesIO(data)).convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(image, dtype=np.int16)
    diff_bits = (pixels[:, :-1] > pixels[:, 1:]).astype(np.uint8)
    packed = np.packbits(diff_bits.flatten())
    return int.from_bytes(packed.tobytes(), byteorder="big")


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Number of differing bits between two hashes of the same width."""
    return bin(hash_a ^ hash_b).count("1")


def dhash_disk_images(
    file_paths: Iterable[str | Path], base_dir: str | Path, hash_size: int = DHASH_SIZE
) -> dict[str, Any]:
    """dHash manifest for image files already extracted on disk (HAM10000).

    Mirrors ``core.build_file_manifest``'s path convention (path relative to
    ``base_dir``, forward slashes) but stores ``dhash`` instead of ``sha256``.
    An unreadable file does not abort the remaining ones (``unreadable``).
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
            h = dhash_bytes(data, hash_size)
        except _READ_ERRORS as exc:
            unreadable.append({"path": rel_path, "error": str(exc)})
            continue
        manifest.append({"path": rel_path, "dhash": h})
    return {
        "expected": len(paths),
        "processed": len(manifest),
        "unreadable": unreadable,
        "manifest": manifest,
    }


def dhash_zip_entries(
    zip_path: str | Path, entry_names: Iterable[str], hash_size: int = DHASH_SIZE
) -> dict[str, Any]:
    """dHash manifest for image entries read directly from a zip (ISIC2018
    Task1, BCN20000 train/test), without extracting anything to disk.

    Same single-open-handle / tolerate-unreadable-entry pattern as
    ``core.sha256_zip_entries``.
    """
    entry_names = list(entry_names)
    manifest: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in entry_names:
            try:
                data = zf.read(name)
                h = dhash_bytes(data, hash_size)
            except _READ_ERRORS as exc:
                unreadable.append({"path": name, "error": str(exc)})
                continue
            manifest.append({"path": name, "dhash": h})
    return {
        "expected": len(entry_names),
        "processed": len(manifest),
        "unreadable": unreadable,
        "manifest": manifest,
    }


def find_candidate_pairs(
    manifest_a: list[dict[str, Any]],
    manifest_b: list[dict[str, Any]],
    threshold: int = DEFAULT_CANDIDATE_THRESHOLD,
    block_size: int = 1000,
) -> list[dict[str, Any]]:
    """Find every ``(path_a, path_b)`` pair whose dHash Hamming distance is
    <= ``threshold``, comparing all of ``manifest_a`` against all of
    ``manifest_b``.

    Vectorized with numpy instead of a nested Python loop: hashes are held
    as ``uint64`` arrays; for each block of ``manifest_a`` (default 1000
    rows), the full XOR against every hash in ``manifest_b`` is computed at
    once and popcounted via ``numpy.bitwise_count`` (a vectorized,
    built-in popcount — equivalent in effect to a hand-rolled 8-bit
    lookup-table popcount, but native to this numpy version and needs no
    lookup table to maintain). Blocking over ``manifest_a`` bounds peak
    memory to ``block_size * len(manifest_b)`` elements instead of
    materializing the full ``len(a) x len(b)`` matrix at once, which matters
    at the ~10,015 x ~18,914 (~189M pairs) scale this is run at in practice.
    """
    if not manifest_a or not manifest_b:
        return []

    paths_a = [m["path"] for m in manifest_a]
    paths_b = [m["path"] for m in manifest_b]
    hashes_a = np.array([m["dhash"] for m in manifest_a], dtype=np.uint64)
    hashes_b = np.array([m["dhash"] for m in manifest_b], dtype=np.uint64)

    candidates: list[dict[str, Any]] = []
    for start in range(0, len(hashes_a), block_size):
        block = hashes_a[start : start + block_size]
        xor = block[:, None] ^ hashes_b[None, :]
        distances = np.bitwise_count(xor)
        rows, cols = np.nonzero(distances <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            candidates.append(
                {
                    "path_a": paths_a[start + r],
                    "path_b": paths_b[c],
                    "distance": int(distances[r, c]),
                }
            )
    return candidates


def expected_random_candidate_count(
    n_a: int, n_b: int, threshold: int = DEFAULT_CANDIDATE_THRESHOLD, hash_bits: int = 64
) -> float:
    """Expected number of candidate pairs among ``n_a * n_b`` comparisons
    IF hashes were uniformly random and independent (i.e. the images shared
    no structure at all): ``n_a * n_b`` times the probability that two
    independent uniform ``hash_bits``-bit values differ in <= ``threshold``
    bits.

    This is a null-model baseline, not a prediction — real dermoscopy images
    are not uniformly random in hash space (they share strong coarse
    structure: a centered pigmented lesion on a roughly uniform skin
    background, consistent dermatoscope framing/lighting). Comparing this
    baseline against the actually observed candidate count is how the
    report quantifies just how much that shared structure inflates the
    real candidate count — see the report's real-data section.
    """
    prob_within_threshold = sum(math.comb(hash_bits, k) for k in range(threshold + 1)) / (2**hash_bits)
    return n_a * n_b * prob_within_threshold


def _make_synthetic_base_image() -> Image.Image:
    """A small synthetic (not patient) image with enough structure (a
    gradient background plus two solid shapes) to produce a non-degenerate
    dHash — a flat/solid image would hash to all-zero and make every
    comparison meaningless.
    """
    size = 256
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    arr = np.stack([xs, ys, (xs + ys) % 256], axis=-1).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 60, 180, 180), fill=(200, 30, 30))
    draw.rectangle((20, 180, 100, 240), fill=(10, 200, 10))
    return image


def _image_to_bytes(image: Image.Image, fmt: str = "PNG", **save_kwargs: Any) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def generate_controlled_variants() -> dict[str, bytes]:
    """Build one synthetic base image plus a fixed set of named, encoded
    byte-variants of it, covering both kinds of transformation this task
    cares about: genuine reprocessing (identical copy, JPEG recompression,
    50% resize, small 5%-border crop) and structural changes (90-degree
    rotation, horizontal mirror, large 30%-border crop).

    Used by both the characterization tests
    (``tests/data_audit/test_visual_duplicates.py``) and the generated
    report (``scripts/run_visual_duplicate_audit.py``) so the measured
    Hamming distances shown in the report come from the exact same code
    path the tests assert on, instead of being copied/invented numbers.
    """
    base = _make_synthetic_base_image()
    width, height = base.size
    small_margin_w, small_margin_h = int(width * 0.05), int(height * 0.05)
    large_left, large_top = int(width * 0.3), int(height * 0.3)
    large_right, large_bottom = int(width * 0.7), int(height * 0.7)

    return {
        "base": _image_to_bytes(base, "PNG"),
        "identical_copy": _image_to_bytes(base, "PNG"),
        "jpeg_recompress_q40": _image_to_bytes(base, "JPEG", quality=40),
        "resize_50pct": _image_to_bytes(
            base.resize((width // 2, height // 2), Image.Resampling.LANCZOS), "PNG"
        ),
        "crop_5pct_border": _image_to_bytes(
            base.crop(
                (small_margin_w, small_margin_h, width - small_margin_w, height - small_margin_h)
            ),
            "PNG",
        ),
        "rotate_90": _image_to_bytes(base.rotate(90, expand=True), "PNG"),
        "mirror_horizontal": _image_to_bytes(base.transpose(Image.FLIP_LEFT_RIGHT), "PNG"),
        "crop_30pct_border": _image_to_bytes(
            base.crop((large_left, large_top, large_right, large_bottom)), "PNG"
        ),
    }


def measure_controlled_variant_distances() -> list[dict[str, Any]]:
    """Compute, for each named variant from :func:`generate_controlled_variants`
    (excluding ``"base"`` itself), its dHash Hamming distance to the base
    image and whether that distance falls within
    :data:`DEFAULT_CANDIDATE_THRESHOLD`.

    This is the single source of truth for the "controlled validation"
    numbers shown in the generated report — it runs the real algorithm, it
    does not hardcode measured values anywhere.
    """
    variants = generate_controlled_variants()
    base_hash = dhash_bytes(variants["base"])
    results = []
    for name, data in variants.items():
        if name == "base":
            continue
        distance = hamming_distance(base_hash, dhash_bytes(data))
        results.append(
            {
                "variant": name,
                "distance": distance,
                "is_candidate": distance <= DEFAULT_CANDIDATE_THRESHOLD,
            }
        )
    return results
