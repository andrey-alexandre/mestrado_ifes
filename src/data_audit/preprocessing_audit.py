"""Directed preprocessing-pipeline audit (task 8 item 1) — verification, not
bug-hunting: checks specific, plausible causes of task 7's dHash candidate
inflation (421,196 candidates across HAM10000 x ISIC2018 Task1 and
HAM10000 x BCN20000) using real sampled data and the already-computed dHash
manifests, before trusting a second filter (pHash) to fix the problem.

Kept in its own module, separate from visual_duplicates.py/phash.py: those
two compute hashes; this one evaluates *why* the hashes behave the way they
do (EXIF orientation, native dimensions, degenerate images, hub hashes). A
different concern, read by a different script section, so a separate file.

Four checks:
1. EXIF orientation: PIL.Image.open does NOT auto-apply EXIF orientation
   (Pillow's documented behavior) -- confirmed here on real sampled bytes,
   not assumed, and measures how many real images actually carry a non-1
   orientation tag (which would mean the pixels dHash/pHash operate on are
   NOT upright for those images, independent of this task).
2. Native dimensions: width/height/aspect-ratio distribution per source, to
   see whether the 9x8 (dHash) / 32x32 (pHash) resize distorts proportion
   systematically differently between datasets.
3. Degenerate images: near-uniform (low grayscale std) or near-black/white
   (extreme grayscale mean) images, which hash close to many other
   low-information images almost by construction.
4. Hub hashes: among the 421,196 dHash candidates, whether a small number
   of *distinct* dHash values account for a disproportionate share of
   candidate edges (a few generic hashes connecting many unrelated images),
   using only the already-written manifests -- no image re-read needed.
"""
from __future__ import annotations

import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

_READ_ERRORS = (UnidentifiedImageError, OSError, zipfile.BadZipFile, RuntimeError)

# Below this grayscale standard deviation (0-255 scale), an image is
# considered near-uniform/low-texture. Chosen as a conservative, clearly
# "almost no visible structure" cutoff (e.g. a std of 5 means pixels vary by
# only a few shades around the mean across the whole image) -- not fit to
# this data, flagged explicitly as a convention in the report.
DEGENERATE_STD_THRESHOLD = 5.0
# Grayscale mean within this distance of 0 or 255 is considered near-black /
# near-white.
DEGENERATE_MEAN_EXTREME_MARGIN = 10.0


def sample_paths(manifest_entries: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic random sample of ``n`` manifest entries (or all of them
    if fewer than ``n`` exist), via a seeded ``random.Random`` so the sample
    is reproducible across reruns."""
    rng = random.Random(seed)
    if n >= len(manifest_entries):
        return list(manifest_entries)
    return rng.sample(manifest_entries, n)


def _read_bytes_disk(base_dir: Path, rel_path: str) -> bytes:
    return (base_dir / rel_path).read_bytes()


def _read_bytes_zip(zf: zipfile.ZipFile, entry_name: str) -> bytes:
    return zf.read(entry_name)


def verify_open_does_not_apply_exif_orientation() -> dict[str, Any]:
    """One-time behavioral verification, not a per-sample statistic: builds
    a synthetic image with known raw pixel dimensions and an EXIF
    orientation tag that implies a 90-degree reorientation (6), encodes it
    to real JPEG bytes, and checks whether ``PIL.Image.open()`` alone
    reports the RAW (un-rotated) size or the EXIF-corrected (swapped) size.

    If ``.open()`` does not apply orientation (Pillow's documented
    behavior), the reported size equals the raw pre-rotation size, because
    ``.open()`` decodes pixels as stored and leaves the orientation tag
    unconsumed. This is measured on real encoded bytes here, not assumed
    from documentation.
    """
    import io

    raw_width, raw_height = 100, 60
    image = Image.new("RGB", (raw_width, raw_height), (255, 255, 255))
    exif = image.getexif()
    exif[0x0112] = 6  # implies a 90-degree CW rotation needed to display upright
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    data = buf.getvalue()

    opened = Image.open(_as_bytesio(data))
    size_from_open = opened.size
    applies_automatically = size_from_open != (raw_width, raw_height)

    return {
        "raw_size": (raw_width, raw_height),
        "size_from_open": size_from_open,
        "open_applies_orientation_automatically": applies_automatically,
    }


def exif_orientation_audit(samples: list[tuple[str, bytes]]) -> dict[str, Any]:
    """For each ``(path, data)`` sample, read the EXIF orientation tag
    (``0x0112``; absent/1 means "normal") and tally its distribution. Does
    NOT itself determine whether ``.open()`` applies orientation -- see
    :func:`verify_open_does_not_apply_exif_orientation` for that one-time
    behavioral check; this function only measures how many *real* sampled
    images carry a non-default orientation tag, which matters regardless of
    that behavior (a non-1 tag means the pixel array dHash/pHash operate on
    is not upright for that image, for anyone who doesn't separately apply
    ``ImageOps.exif_transpose``).
    """
    orientation_counts: Counter[int] = Counter()
    non_default_examples: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []

    for path, data in samples:
        try:
            image = Image.open(_as_bytesio(data))
            orientation = image.getexif().get(0x0112, 1)
        except _READ_ERRORS as exc:
            unreadable.append({"path": path, "error": str(exc)})
            continue

        orientation_counts[orientation] += 1
        if orientation != 1:
            non_default_examples.append({"path": path, "orientation": orientation})

    return {
        "n_sampled": len(samples),
        "n_processed": len(samples) - len(unreadable),
        "unreadable": unreadable,
        "orientation_value_counts": dict(sorted(orientation_counts.items())),
        "n_non_default_orientation": sum(c for o, c in orientation_counts.items() if o != 1),
        "non_default_examples": non_default_examples[:20],
    }


def dimension_audit(samples: list[tuple[str, bytes]], dhash_target_size: tuple[int, int] = (9, 8)) -> dict[str, Any]:
    """Native width/height/aspect-ratio distribution for a sample, plus how
    far each image's native aspect ratio deviates from the target resize
    grid's aspect ratio (``dhash_target_size`` = width x height before the
    dHash gradient step; e.g. 9/8 = 1.125) -- a systematic difference in
    native aspect ratio between datasets would mean the forced resize
    distorts proportion differently per source, independent of content.
    """
    target_w, target_h = dhash_target_size
    target_aspect = target_w / target_h

    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    unreadable: list[dict[str, Any]] = []

    for path, data in samples:
        try:
            image = Image.open(_as_bytesio(data))
            w, h = image.size
        except _READ_ERRORS as exc:
            unreadable.append({"path": path, "error": str(exc)})
            continue
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

    if not widths:
        return {"n_sampled": len(samples), "n_processed": 0, "unreadable": unreadable}

    widths_arr, heights_arr, aspects_arr = np.array(widths), np.array(heights), np.array(aspect_ratios)
    return {
        "n_sampled": len(samples),
        "n_processed": len(widths),
        "unreadable": unreadable,
        "width": {"min": int(widths_arr.min()), "mean": float(widths_arr.mean()), "max": int(widths_arr.max())},
        "height": {"min": int(heights_arr.min()), "mean": float(heights_arr.mean()), "max": int(heights_arr.max())},
        "aspect_ratio": {
            "min": float(aspects_arr.min()),
            "mean": float(aspects_arr.mean()),
            "max": float(aspects_arr.max()),
        },
        "dhash_target_aspect_ratio": target_aspect,
        "mean_abs_deviation_from_target_aspect": float(np.mean(np.abs(aspects_arr - target_aspect))),
    }


def degenerate_image_audit(samples: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Flag images whose grayscale pixel statistics suggest little usable
    structure: near-uniform (``std < DEGENERATE_STD_THRESHOLD``) or
    near-black/near-white (``mean`` within ``DEGENERATE_MEAN_EXTREME_MARGIN``
    of 0 or 255). These are exactly the kind of image whose dHash collides
    broadly with other low-information images almost by construction,
    regardless of actual lesion content.
    """
    flagged: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    n_processed = 0

    for path, data in samples:
        try:
            image = Image.open(_as_bytesio(data)).convert("L")
            arr = np.asarray(image, dtype=np.float64)
        except _READ_ERRORS as exc:
            unreadable.append({"path": path, "error": str(exc)})
            continue
        n_processed += 1
        std = float(arr.std())
        mean = float(arr.mean())
        is_near_uniform = std < DEGENERATE_STD_THRESHOLD
        is_near_extreme = mean < DEGENERATE_MEAN_EXTREME_MARGIN or mean > 255 - DEGENERATE_MEAN_EXTREME_MARGIN
        if is_near_uniform or is_near_extreme:
            flagged.append(
                {
                    "path": path,
                    "std": std,
                    "mean": mean,
                    "near_uniform": is_near_uniform,
                    "near_extreme_mean": is_near_extreme,
                }
            )

    return {
        "n_sampled": len(samples),
        "n_processed": n_processed,
        "unreadable": unreadable,
        "n_flagged": len(flagged),
        "flagged": flagged,
        "std_threshold": DEGENERATE_STD_THRESHOLD,
        "mean_extreme_margin": DEGENERATE_MEAN_EXTREME_MARGIN,
    }


def hub_hash_audit(
    candidates: list[dict[str, Any]],
    path_to_dhash_a: dict[str, int],
    path_to_dhash_b: dict[str, int],
    top_n: int = 20,
) -> dict[str, Any]:
    """Among the dHash candidate pairs, count how many candidate edges each
    *distinct* dHash value participates in (as either side), using only the
    already-written candidate index and dHash manifests -- no image bytes
    read. A "hub" pattern (a handful of hash values responsible for a large
    fraction of the 421,196 candidates) would directly explain the
    inflation: those images' dHash is generic enough to sit within the
    threshold of many unrelated images, not necessarily duplicated content.
    """
    hash_edge_count: Counter[int] = Counter()
    for cand in candidates:
        ha = path_to_dhash_a.get(cand["path_a_ham10000"])
        hb = path_to_dhash_b.get(cand["path_b"])
        if ha is not None:
            hash_edge_count[ha] += 1
        if hb is not None:
            hash_edge_count[hb] += 1

    total_candidates = len(candidates)
    top = hash_edge_count.most_common(top_n)
    top_edge_sum = sum(count for _, count in top)
    top_hash_set = {h for h, _ in top}

    # top_edge_sum double-counts a candidate whose two endpoints are BOTH in
    # the top-N set, so it is not itself a valid "share of candidates" (can
    # exceed total_candidates). The actual, boundable answer to "how many
    # real candidates touch at least one hub hash" is this distinct count.
    n_candidates_touching_top_hashes = sum(
        1
        for cand in candidates
        if path_to_dhash_a.get(cand["path_a_ham10000"]) in top_hash_set
        or path_to_dhash_b.get(cand["path_b"]) in top_hash_set
    )

    return {
        "total_candidates": total_candidates,
        "n_distinct_hashes_involved": len(hash_edge_count),
        "top_hashes": [{"dhash": h, "edge_count": c} for h, c in top],
        "top_hashes_edge_count_sum": top_edge_sum,
        "n_candidates_touching_top_hashes": n_candidates_touching_top_hashes,
        "top_hashes_share_of_candidates": (
            (n_candidates_touching_top_hashes / total_candidates) if total_candidates else 0.0
        ),
    }


def _as_bytesio(data: bytes):
    import io

    return io.BytesIO(data)


def collect_samples_disk(
    manifest_entries: list[dict[str, Any]], base_dir: str | Path, n: int, seed: int
) -> list[tuple[str, bytes]]:
    """Sample ``n`` entries and read their bytes from disk (HAM10000)."""
    base = Path(base_dir)
    sampled = sample_paths(manifest_entries, n, seed)
    return [(e["path"], _read_bytes_disk(base, e["path"])) for e in sampled]


def collect_samples_zip(
    manifest_entries: list[dict[str, Any]], zip_path: str | Path, n: int, seed: int
) -> list[tuple[str, bytes]]:
    """Sample ``n`` entries and read their bytes from a single zip (ISIC2018
    Task1, or one already-materialized BCN20000 split)."""
    sampled = sample_paths(manifest_entries, n, seed)
    with zipfile.ZipFile(zip_path) as zf:
        return [(e["path"], _read_bytes_zip(zf, e["path"])) for e in sampled]
