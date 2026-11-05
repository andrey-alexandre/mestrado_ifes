"""Visual-duplicate CANDIDATE detection (task 7 of the reconstruction).

Extends task 6's sha256 byte-identical comparison (which only catches
byte-identical files) with a perceptual hash (dHash) comparison that also
flags recompressed / resized / lightly-cropped re-releases of the same
lesion photo. See ``src.data_audit.visual_duplicates`` for the method and
how its candidate threshold was derived from controlled tests.

Reuses the existing manifests under ``experiments/data_audit/`` (written by
``scripts/run_dataset_inventory.py`` and ``scripts/run_content_hash_audit.py``)
purely for their ``path`` lists — this script does not re-list any zip's
central directory, does not re-hash with sha256, and does not re-materialize
BCN20000's inner zips (it reuses the already-materialized files recorded in
``bcn20000_content_manifest.json["train"/"test"]["materialization"]["dest_path"]``).

This is DETECTION AND REPORTING ONLY. A low-distance candidate here is not a
confirmed duplicate and does not authorize removing anything; every
candidate requires researcher review. No exclusion is decided or
implemented here.

Usage:
    python scripts/run_visual_duplicate_audit.py
    python scripts/run_visual_duplicate_audit.py --max-rendered-per-pair 100
    python scripts/run_visual_duplicate_audit.py --force-recompute-dhash
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import visual_duplicates as vd  # noqa: E402
from src.data_audit import visual_duplicate_report  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiments" / "data_audit"
CANDIDATES_DIR = MANIFEST_DIR / "visual_candidates"
INDEX_PATH = MANIFEST_DIR / "visual_candidates_index.json"
REPORT_PATH = REPO_ROOT / "docs" / "reconstruction" / "04_visual_duplicate_candidates_report.md"

DEFAULT_MAX_RENDERED_PER_PAIR = 200
THUMBNAIL_SIZE = (200, 200)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _dhash_manifest_cached(cache_path: Path, force_recompute: bool, compute_fn) -> dict[str, Any]:
    """Reuse a previously written dHash manifest unless it's missing or
    ``force_recompute`` is set — same idempotent-cache convention already
    used by ``bcn_materialize.materialize_inner_zip``. Avoids re-decoding
    tens of thousands of images on every rerun while developing/debugging
    this script; delete the file (or pass --force-recompute-dhash) to redo
    the hashing for real.
    """
    if not force_recompute and cache_path.is_file():
        return _load_json(cache_path)
    result = compute_fn()
    _save_json(cache_path, result)
    return result


def _dhash_ham10000(images_dir: Path, ham10000_manifest: dict[str, Any]) -> dict[str, Any]:
    file_paths = [images_dir / e["path"] for e in ham10000_manifest["image_manifest"]]
    return vd.dhash_disk_images(file_paths, base_dir=images_dir)


def _dhash_isic2018_task1(images_zip: Path, isic_content_manifest: dict[str, Any]) -> dict[str, Any]:
    entry_names = [e["path"] for e in isic_content_manifest["manifest"]]
    return vd.dhash_zip_entries(images_zip, entry_names)


def _dhash_bcn20000(bcn_content_manifest: dict[str, Any]) -> dict[str, Any]:
    train_zip = Path(bcn_content_manifest["train"]["materialization"]["dest_path"])
    train_entries = [e["path"] for e in bcn_content_manifest["train"]["manifest"]]
    train_result = vd.dhash_zip_entries(train_zip, train_entries)

    test_zip = Path(bcn_content_manifest["test"]["materialization"]["dest_path"])
    test_entries = [e["path"] for e in bcn_content_manifest["test"]["manifest"]]
    test_result = vd.dhash_zip_entries(test_zip, test_entries)

    return {
        "expected": train_result["expected"] + test_result["expected"],
        "processed": train_result["processed"] + test_result["processed"],
        "unreadable": train_result["unreadable"] + test_result["unreadable"],
        "manifest": train_result["manifest"] + test_result["manifest"],
        "train": {**train_result, "source_zip": str(train_zip)},
        "test": {**test_result, "source_zip": str(test_zip)},
    }


def _load_thumbnail_from_disk(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(THUMBNAIL_SIZE)
    return image


def _load_thumbnail_from_zip(zip_path: Path, entry_name: str) -> Image.Image:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(entry_name)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail(THUMBNAIL_SIZE)
    return image


def _render_side_by_side(
    image_a: Image.Image, image_b: Image.Image, label_a: str, label_b: str, distance: int, out_path: Path
) -> None:
    caption_h = 54
    width = image_a.width + image_b.width
    height = max(image_a.height, image_b.height) + caption_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(image_a, (0, 0))
    canvas.paste(image_b, (image_a.width, 0))
    draw = ImageDraw.Draw(canvas)
    text_y = max(image_a.height, image_b.height) + 2
    draw.text((4, text_y), f"A: {label_a}", fill=(0, 0, 0))
    draw.text((4, text_y + 16), f"B: {label_b}", fill=(0, 0, 0))
    draw.text((4, text_y + 32), f"hamming distance: {distance}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _candidate_thumbnail_loaders(
    pair_name: str,
    ham10000_images_dir: Path,
    isic_images_zip: Path,
    bcn_train_zip: Path,
    bcn_test_zip: Path,
):
    """Return (load_a, load_b) functions for a given pair name, where side A
    is always HAM10000 (loaded from disk) and side B is the other dataset
    (loaded from its zip; for BCN20000 dispatched to train/test by the
    entry's directory prefix already present in its manifest path, e.g.
    'bcn_20k_train/...' vs 'bcn_20k_test/...')."""

    def load_ham10000(rel_path: str) -> Image.Image:
        return _load_thumbnail_from_disk(ham10000_images_dir / rel_path)

    if pair_name == "ham10000_x_isic2018_task1":
        def load_b(entry_name: str) -> Image.Image:
            return _load_thumbnail_from_zip(isic_images_zip, entry_name)
    elif pair_name == "ham10000_x_bcn20000":
        def load_b(entry_name: str) -> Image.Image:
            zip_path = bcn_train_zip if entry_name.startswith("bcn_20k_train/") else bcn_test_zip
            return _load_thumbnail_from_zip(zip_path, entry_name)
    else:
        raise ValueError(f"unknown pair_name: {pair_name}")

    return load_ham10000, load_b


def _run_pair(
    pair_name: str,
    pair_label: str,
    ham_manifest: dict[str, Any],
    other_manifest: dict[str, Any],
    threshold: int,
    max_rendered: int,
    ham10000_images_dir: Path,
    isic_images_zip: Path,
    bcn_train_zip: Path,
    bcn_test_zip: Path,
) -> dict[str, Any]:
    print(f"=== {pair_label}: buscando candidatos (limiar={threshold}) ===")
    candidates = vd.find_candidate_pairs(
        ham_manifest["manifest"], other_manifest["manifest"], threshold=threshold
    )
    candidates.sort(key=lambda c: c["distance"])
    total = len(candidates)
    to_render = candidates[:max_rendered]
    print(f"candidatos encontrados={total} renderizando={len(to_render)}")

    load_a, load_b = _candidate_thumbnail_loaders(
        pair_name, ham10000_images_dir, isic_images_zip, bcn_train_zip, bcn_test_zip
    )

    index_entries = []
    pair_dir = CANDIDATES_DIR / pair_name
    for i, cand in enumerate(candidates):
        entry = {
            "pair": pair_label,
            "path_a_ham10000": cand["path_a"],
            "path_b": cand["path_b"],
            "distance": cand["distance"],
            "rendered": False,
            "side_by_side_image": None,
        }
        if i < max_rendered:
            try:
                img_a = load_a(cand["path_a"])
                img_b = load_b(cand["path_b"])
            except (OSError, UnidentifiedImageError, zipfile.BadZipFile) as exc:
                entry["render_error"] = str(exc)
            else:
                out_path = pair_dir / f"{i:05d}.png"
                _render_side_by_side(
                    img_a, img_b, cand["path_a"], cand["path_b"], cand["distance"], out_path
                )
                entry["rendered"] = True
                entry["side_by_side_image"] = str(out_path.relative_to(REPO_ROOT)).replace("\\", "/")
        index_entries.append(entry)

    return {
        "pair": pair_label,
        "threshold": threshold,
        "n_a": ham_manifest["processed"],
        "n_b": other_manifest["processed"],
        "total_candidates": total,
        "rendered_count": sum(1 for e in index_entries if e["rendered"]),
        "max_rendered_per_pair": max_rendered,
        "candidates_dir": str(pair_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "index_entries": index_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "dataset_inventory.yaml"),
        help="Caminho do YAML de configuração do inventário (mesmo do task 2/6).",
    )
    parser.add_argument(
        "--max-rendered-per-pair",
        type=int,
        default=DEFAULT_MAX_RENDERED_PER_PAIR,
        help="Máximo de imagens lado a lado renderizadas por par (os de menor distância primeiro).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=vd.DEFAULT_CANDIDATE_THRESHOLD,
        help="Limiar de distância de Hamming para considerar um par candidato.",
    )
    parser.add_argument(
        "--force-recompute-dhash",
        action="store_true",
        help="Recalcula os manifestos dHash mesmo se já existirem em disco.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ham10000_images_dir = Path(config["ham10000"]["images_dir"])
    isic_images_zip = Path(config["isic2018_task1"]["images_zip"])

    ham10000_manifest = _load_json(MANIFEST_DIR / "ham10000_manifest.json")
    isic_content_manifest = _load_json(MANIFEST_DIR / "isic2018_task1_content_manifest.json")
    bcn_content_manifest = _load_json(MANIFEST_DIR / "bcn20000_content_manifest.json")

    print("=== dHash HAM10000 ===")
    ham_dhash = _dhash_manifest_cached(
        MANIFEST_DIR / "ham10000_dhash_manifest.json",
        args.force_recompute_dhash,
        lambda: _dhash_ham10000(ham10000_images_dir, ham10000_manifest),
    )
    print(
        f"expected={ham_dhash['expected']} processed={ham_dhash['processed']} "
        f"unreadable={len(ham_dhash['unreadable'])}"
    )

    print("=== dHash ISIC2018 Task1 ===")
    isic_dhash = _dhash_manifest_cached(
        MANIFEST_DIR / "isic2018_task1_dhash_manifest.json",
        args.force_recompute_dhash,
        lambda: _dhash_isic2018_task1(isic_images_zip, isic_content_manifest),
    )
    print(
        f"expected={isic_dhash['expected']} processed={isic_dhash['processed']} "
        f"unreadable={len(isic_dhash['unreadable'])}"
    )

    print("=== dHash BCN20000 (treino+teste) ===")
    bcn_dhash = _dhash_manifest_cached(
        MANIFEST_DIR / "bcn20000_dhash_manifest.json",
        args.force_recompute_dhash,
        lambda: _dhash_bcn20000(bcn_content_manifest),
    )
    print(
        f"expected={bcn_dhash['expected']} processed={bcn_dhash['processed']} "
        f"unreadable={len(bcn_dhash['unreadable'])}"
    )

    bcn_train_zip = Path(bcn_content_manifest["train"]["materialization"]["dest_path"])
    bcn_test_zip = Path(bcn_content_manifest["test"]["materialization"]["dest_path"])

    pair_results = []
    pair_results.append(
        _run_pair(
            "ham10000_x_isic2018_task1",
            "HAM10000 x ISIC2018 Task1",
            ham_dhash,
            isic_dhash,
            args.threshold,
            args.max_rendered_per_pair,
            ham10000_images_dir,
            isic_images_zip,
            bcn_train_zip,
            bcn_test_zip,
        )
    )
    pair_results.append(
        _run_pair(
            "ham10000_x_bcn20000",
            "HAM10000 x BCN20000",
            ham_dhash,
            bcn_dhash,
            args.threshold,
            args.max_rendered_per_pair,
            ham10000_images_dir,
            isic_images_zip,
            bcn_train_zip,
            bcn_test_zip,
        )
    )

    index = {
        "threshold": args.threshold,
        "pairs": [
            {
                "pair": r["pair"],
                "n_a": r["n_a"],
                "n_b": r["n_b"],
                "total_candidates": r["total_candidates"],
                "rendered_count": r["rendered_count"],
                "candidates_dir": r["candidates_dir"],
            }
            for r in pair_results
        ],
        "candidates": [entry for r in pair_results for entry in r["index_entries"]],
    }
    _save_json(INDEX_PATH, index)
    print(f"índice gravado: {INDEX_PATH}")

    controlled_variants = vd.measure_controlled_variant_distances()

    report_text = visual_duplicate_report.render_report(
        threshold=args.threshold,
        controlled_variants=controlled_variants,
        ham_dhash=ham_dhash,
        isic_dhash=isic_dhash,
        bcn_dhash=bcn_dhash,
        pair_results=pair_results,
        index_path=INDEX_PATH.relative_to(REPO_ROOT),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"relatório gravado: {REPORT_PATH}")


if __name__ == "__main__":
    main()
