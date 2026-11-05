"""Task 9 pilot: pHash 64 bits vs. 256 bits compared head-to-head on the
same protocol, dHash funnel-entry coverage (new), and ORB+RANSAC run
independent of any pHash gate.

Task 8's staged pilot (pHash 256 bits + ORB+RANSAC) found ZERO survivors
among the 2,000 sampled pHash candidates, including all 400 known hard
pairs, and could not distinguish "no real duplicates in the sample" from
"the 256-bit pHash discards real positives (crop-sensitive, see task 8's
report item 2) before the geometric stage ever runs." This script answers
that by (1) measuring pHash at 64 bits (8x8 DCT block) alongside the
existing 256-bit config on the SAME real images/variants/hard pairs --
no sweep of any other block size; (2) measuring dHash's own coverage of the
same transforms (never done in tasks 7/8); (3) running ORB+RANSAC directly
on every one of the 2,000 sampled pairs, with no pHash gate at all, so the
geometric stage gets a real chance to speak for itself.

Reuses, unchanged: the exact 2,000-pair sample from
experiments/data_audit/visual_duplicate_pilot_index.json (same seed=42 as
task 8), src.data_audit.phash.phash_bytes (parametrized by low_freq_size,
no new hash function), src.data_audit.visual_duplicates.generate_variants_of_image,
src.data_audit.geometric_verification (verify_geometric_match,
classify_geometric_support, draw_match_evidence).

No exclusion of any lesion/image is decided or implemented at any stage.
Does not overwrite task 8's index or report.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import geometric_verification as gv  # noqa: E402
from src.data_audit import phash  # noqa: E402
from src.data_audit import preprocessing_audit as pa  # noqa: E402
from src.data_audit import visual_duplicate_pilot_v2 as v2  # noqa: E402
from src.data_audit import visual_duplicate_pilot_v2_report as v2_report  # noqa: E402
from src.data_audit import visual_duplicates as vd  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiments" / "data_audit"
V2_EVIDENCE_DIR = MANIFEST_DIR / "visual_duplicate_pilot_v2_evidence"
V1_INDEX_PATH = MANIFEST_DIR / "visual_duplicate_pilot_index.json"
V2_INDEX_PATH = MANIFEST_DIR / "visual_duplicate_pilot_v2_index.json"
REPORT_PATH = REPO_ROOT / "docs" / "reconstruction" / "06_visual_duplicate_pilot_v2_report.md"

DEFAULT_SEED = 42
# 10 real images per source (HAM10000, ISIC2018 Task1, BCN20000) = 30 total:
# double task 8's 8-image single-source sample, while covering all three
# sources that actually feed task 7's candidate pool (task 8 only covered
# HAM10000), and still cheap (each image needs only 7 in-memory variants, no
# extra disk/zip reads beyond the one read per source image).
N_REAL_IMAGES_PER_SOURCE = 10
MAX_EVIDENCE_RENDERED = 300
GEOMETRIC_MAX_DIM = v2.GEOMETRIC_MAX_DIM

PHASH_VARIANT_LABELS = {
    "identical_copy": "Cópia idêntica",
    "jpeg_recompress_q40": "Recompressão JPEG (qualidade 40)",
    "resize_50pct": "Redimensionamento (50%)",
    "crop_5pct_border": "Recorte pequeno (5% da borda)",
    "rotate_90": "Rotação 90°",
    "mirror_horizontal": "Espelhamento horizontal",
    "crop_30pct_border": "Recorte significativo (30% da borda)",
}
VARIANT_ORDER = list(PHASH_VARIANT_LABELS.keys())


class ImageSourceReader:
    """Dispatches byte reads across disk (HAM10000) and the already-
    materialized zips (ISIC2018 Task1, BCN20000 train/test) -- the exact
    same files tasks 6/7/8 used; nothing is re-extracted or re-listed here.
    Deliberately a standalone copy of task 8's identical class (script-level
    plumbing, not analysis logic) rather than importing across sibling
    script modules.
    """

    def __init__(self, ham_dir: Path, isic_zip_path: Path, bcn_train_zip_path: Path, bcn_test_zip_path: Path):
        self.ham_dir = ham_dir
        self._isic_zf = zipfile.ZipFile(isic_zip_path)
        self._bcn_train_zf = zipfile.ZipFile(bcn_train_zip_path)
        self._bcn_test_zf = zipfile.ZipFile(bcn_test_zip_path)

    def read_ham10000(self, path: str) -> bytes:
        return (self.ham_dir / path).read_bytes()

    def read_other(self, path: str) -> bytes:
        if path.startswith("bcn_20k_train/"):
            return self._bcn_train_zf.read(path)
        if path.startswith("bcn_20k_test/"):
            return self._bcn_test_zf.read(path)
        return self._isic_zf.read(path)

    def close(self) -> None:
        self._isic_zf.close()
        self._bcn_train_zf.close()
        self._bcn_test_zf.close()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _variant_stat_row(name: str, distances: list[int], bits: int) -> dict[str, Any]:
    return {
        "variant": name,
        "label": PHASH_VARIANT_LABELS.get(name, name),
        "n": len(distances),
        "min": min(distances),
        "mean": sum(distances) / len(distances),
        "max": max(distances),
        "normalized_min": v2.normalized_distance(min(distances), bits),
        "normalized_mean": v2.normalized_distance(sum(distances) / len(distances), bits),
        "normalized_max": v2.normalized_distance(max(distances), bits),
    }


def _hard_pairs_stats(distances: list[int], bits: int) -> dict[str, Any]:
    return {
        "n_pairs": len(distances),
        "min": min(distances),
        "mean": statistics.mean(distances),
        "median": statistics.median(distances),
        "max": max(distances),
        "normalized_min": v2.normalized_distance(min(distances), bits),
        "normalized_mean": v2.normalized_distance(statistics.mean(distances), bits),
        "normalized_median": v2.normalized_distance(statistics.median(distances), bits),
        "normalized_max": v2.normalized_distance(max(distances), bits),
    }


# ---------------------------------------------------------------------------
# Item 1 + 2 + 3 (controlled positives): real-image sample, pHash 64/256,
# dHash coverage, ORB with no pHash gate -- all measured together per image
# since they share the same generated variants.
# ---------------------------------------------------------------------------


def run_real_image_validation(
    reader: ImageSourceReader,
    ham_manifest: list[dict[str, Any]],
    isic_manifest: list[dict[str, Any]],
    bcn_manifest: list[dict[str, Any]],
    hard_pairs: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    print("=== Itens 1+2+3 (parte controlada): amostra real ampliada ===")

    sources = {
        "HAM10000": (ham_manifest, reader.read_ham10000),
        "ISIC2018 Task1": (isic_manifest, reader.read_other),
        "BCN20000": (bcn_manifest, reader.read_other),
    }
    per_source_n: dict[str, int] = {}

    phash64_by_variant: dict[str, list[int]] = {name: [] for name in VARIANT_ORDER}
    phash256_by_variant: dict[str, list[int]] = {name: [] for name in VARIANT_ORDER}
    dhash_by_variant: dict[str, list[int]] = {name: [] for name in VARIANT_ORDER}
    orb_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANT_ORDER}

    for source_name, (manifest, read_fn) in sources.items():
        sampled_entries = pa.sample_paths(manifest, N_REAL_IMAGES_PER_SOURCE, seed)
        per_source_n[source_name] = len(sampled_entries)
        for entry in sampled_entries:
            data = read_fn(entry["path"])
            image = Image.open(io.BytesIO(data)).convert("RGB")
            variants = vd.generate_variants_of_image(image)
            base_data = variants["base"]
            for name in VARIANT_ORDER:
                variant_data = variants[name]
                phash_distances = v2.compute_phash_distances(base_data, variant_data)
                phash64_by_variant[name].append(phash_distances["phash_distance_64"])
                phash256_by_variant[name].append(phash_distances["phash_distance_256"])

                dhash_result = v2.dhash_recovery(base_data, variant_data)
                dhash_by_variant[name].append(dhash_result["distance"])

                orb_outcome = v2.verify_pair_from_bytes(base_data, variant_data)
                orb_by_variant[name].append(orb_outcome)
        print(f"[{source_name}] amostrado(s)={len(sampled_entries)}")

    n_real_images = sum(per_source_n.values())
    print(f"total de imagens reais para variantes controladas: {n_real_images}")

    # Hard pairs (dHash distance 0-2, task 7's already-confirmed real false
    # positives) at both pHash widths -- same pairs task 8 used for pHash-256.
    hard64: list[int] = []
    hard256: list[int] = []
    for cand in hard_pairs:
        data_a = reader.read_ham10000(cand["path_a_ham10000"])
        data_b = reader.read_other(cand["path_b"])
        distances = v2.compute_phash_distances(data_a, data_b)
        hard64.append(distances["phash_distance_64"])
        hard256.append(distances["phash_distance_256"])
    print(f"pares difíceis (n={len(hard64)}): pHash-64 min={min(hard64)} max={max(hard64)}; "
          f"pHash-256 min={min(hard256)} max={max(hard256)}")

    variant_stats_64 = [_variant_stat_row(name, phash64_by_variant[name], 64) for name in VARIANT_ORDER]
    variant_stats_256 = [_variant_stat_row(name, phash256_by_variant[name], 256) for name in VARIANT_ORDER]
    hard_pairs_64 = _hard_pairs_stats(hard64, 64)
    hard_pairs_256 = _hard_pairs_stats(hard256, 256)

    threshold_64 = v2.derive_threshold_from_gap(phash64_by_variant, hard64)
    threshold_64["bits"] = 64
    threshold_256 = v2.derive_threshold_from_gap(phash256_by_variant, hard256)
    threshold_256["bits"] = 256
    print(f"limiar derivado pHash-64={threshold_64['threshold']} pHash-256={threshold_256['threshold']}")

    dhash_coverage = []
    for name in VARIANT_ORDER:
        distances = dhash_by_variant[name]
        n_recovered = sum(1 for d in distances if d <= vd.DEFAULT_CANDIDATE_THRESHOLD)
        dhash_coverage.append(
            {
                "variant": name,
                "label": PHASH_VARIANT_LABELS[name],
                "n": len(distances),
                "n_recovered": n_recovered,
                "recovery_rate": n_recovered / len(distances),
                "min": min(distances),
                "max": max(distances),
            }
        )

    controlled_orb = []
    for name in VARIANT_ORDER:
        outcomes = orb_by_variant[name]
        n = len(outcomes)
        evaluated = [o for o in outcomes if o["result"]["status"] == "evaluated"]
        n_forte = sum(1 for o in outcomes if o["strength"] == "forte")
        n_fraco = sum(1 for o in outcomes if o["strength"] == "fraco")
        n_not_evaluable = sum(1 for o in outcomes if o["strength"] == "not_evaluable")
        mean_inlier_ratio = (
            sum(o["result"]["inlier_ratio"] for o in evaluated) / len(evaluated) if evaluated else None
        )
        controlled_orb.append(
            {
                "variant": name,
                "label": PHASH_VARIANT_LABELS[name],
                "n": n,
                "n_evaluated": len(evaluated),
                "n_forte": n_forte,
                "n_fraco": n_fraco,
                "n_not_evaluable": n_not_evaluable,
                "mean_inlier_ratio": mean_inlier_ratio,
            }
        )

    real_sample = {"seed": seed, "n_real_images": n_real_images, "per_source_n": per_source_n}
    phash_comparison = {
        "variant_stats_64": variant_stats_64,
        "variant_stats_256": variant_stats_256,
        "hard_pairs_64": hard_pairs_64,
        "hard_pairs_256": hard_pairs_256,
        "threshold_64": threshold_64,
        "threshold_256": threshold_256,
    }
    return {
        "real_sample": real_sample,
        "phash_comparison": phash_comparison,
        "dhash_coverage": dhash_coverage,
        "controlled_orb": controlled_orb,
    }


# ---------------------------------------------------------------------------
# Item 3 (central part): ORB+RANSAC on ALL 2,000 sampled pairs, unconditional
# ---------------------------------------------------------------------------


def run_full_sample_unconditional(
    reader: ImageSourceReader,
    v1_pairs: list[dict[str, Any]],
    phash_threshold_64: int,
    phash_threshold_256: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("=== Item 3 (central): ORB+RANSAC em TODOS os 2.000 pares, sem gate de pHash ===")

    data_cache: dict[tuple[str, str], bytes | None] = {}

    def cached_data(side: str, path: str) -> bytes | None:
        key = (side, path)
        if key in data_cache:
            return data_cache[key]
        try:
            data = reader.read_ham10000(path) if side == "ham" else reader.read_other(path)
        except (OSError, zipfile.BadZipFile):
            data_cache[key] = None
            return None
        data_cache[key] = data
        return data

    array_cache: dict[tuple[str, str], np.ndarray | None] = {}

    def cached_array(side: str, path: str, data: bytes | None) -> np.ndarray | None:
        key = (side, path)
        if key in array_cache:
            return array_cache[key]
        if data is None:
            array_cache[key] = None
            return None
        arr = v2.decode_grayscale_array(data)
        array_cache[key] = arr
        return arr

    entries: list[dict[str, Any]] = []
    geometric_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    n_read_errors = 0

    start = time.perf_counter()
    for cand in v1_pairs:
        path_a, path_b = cand["path_a_ham10000"], cand["path_b"]
        data_a = cached_data("ham", path_a)
        data_b = cached_data("other", path_b)

        entry: dict[str, Any] = {
            "pair": cand["pair"],
            "path_a_ham10000": path_a,
            "path_b": path_b,
            "dhash_distance": cand["dhash_distance"],
            "phash_distance_64": None,
            "phash_distance_256": None,
            "phash_survivor_64": False,
            "phash_survivor_256": False,
            "geometric_status": None,
            "n_keypoints_a": None,
            "n_keypoints_b": None,
            "n_matches": None,
            "n_inliers": None,
            "inlier_ratio": None,
            "geometric_strength": None,
            "evidence_image": None,
        }

        if data_a is None or data_b is None:
            entry["geometric_status"] = "image_read_error"
            entry["geometric_strength"] = "image_read_error"
            n_read_errors += 1
            entries.append(entry)
            continue

        distances = v2.compute_phash_distances(data_a, data_b)
        entry["phash_distance_64"] = distances["phash_distance_64"]
        entry["phash_distance_256"] = distances["phash_distance_256"]
        entry["phash_survivor_64"] = distances["phash_distance_64"] <= phash_threshold_64
        entry["phash_survivor_256"] = distances["phash_distance_256"] <= phash_threshold_256

        array_a = cached_array("ham", path_a, data_a)
        array_b = cached_array("other", path_b, data_b)
        result = gv.verify_geometric_match(array_a, array_b)
        strength = gv.classify_geometric_support(result)
        entry["geometric_status"] = result["status"]
        entry["n_keypoints_a"] = result["n_keypoints_a"]
        entry["n_keypoints_b"] = result["n_keypoints_b"]
        entry["n_matches"] = result["n_matches"]
        entry["n_inliers"] = result["n_inliers"]
        entry["inlier_ratio"] = result["inlier_ratio"]
        entry["geometric_strength"] = strength

        entries.append(entry)
        if result["status"] == "evaluated":
            geometric_results.append((entry, result))

    elapsed = time.perf_counter() - start
    print(f"ORB+RANSAC em {len(v1_pairs)} pares levou {elapsed:.1f}s ({n_read_errors} erros de leitura)")

    # Evidence: strongest support first ("forte" before "fraco"), then by
    # inlier_ratio descending within each group, capped at MAX_EVIDENCE_RENDERED.
    strength_rank = {"forte": 1, "fraco": 0}
    geometric_results.sort(
        key=lambda pair: (strength_rank.get(pair[0]["geometric_strength"], -1), pair[1]["inlier_ratio"]),
        reverse=True,
    )
    V2_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence_rendered = 0
    for i, (entry, result) in enumerate(geometric_results):
        if i >= MAX_EVIDENCE_RENDERED:
            break
        data_a = cached_data("ham", entry["path_a_ham10000"])
        data_b = cached_data("other", entry["path_b"])
        array_a = cached_array("ham", entry["path_a_ham10000"], data_a)
        array_b = cached_array("other", entry["path_b"], data_b)
        evidence_bgr = gv.draw_match_evidence(array_a, array_b, result)
        out_path = V2_EVIDENCE_DIR / f"{i:05d}.png"
        Image.fromarray(evidence_bgr[:, :, ::-1]).save(out_path)
        entry["evidence_image"] = str(out_path.relative_to(REPO_ROOT)).replace("\\", "/")
        evidence_rendered += 1
    print(f"evidências renderizadas: {evidence_rendered}")

    n_survivors_64 = sum(1 for e in entries if e["phash_survivor_64"])
    n_survivors_256 = sum(1 for e in entries if e["phash_survivor_256"])
    n_evaluated = sum(1 for e in entries if e["geometric_status"] == "evaluated")
    n_forte = sum(1 for e in entries if e["geometric_strength"] == "forte")
    n_fraco = sum(1 for e in entries if e["geometric_strength"] == "fraco")
    n_not_evaluable = sum(1 for e in entries if e["geometric_strength"] == "not_evaluable")

    buckets = []
    for label, lo, hi in v2.DHASH_BUCKETS:
        in_bucket = [e for e in entries if lo <= e["dhash_distance"] <= hi]
        buckets.append(
            {
                "label": label,
                "n": len(in_bucket),
                "n_forte": sum(1 for e in in_bucket if e["geometric_strength"] == "forte"),
                "n_fraco": sum(1 for e in in_bucket if e["geometric_strength"] == "fraco"),
                "n_not_evaluable": sum(1 for e in in_bucket if e["geometric_strength"] == "not_evaluable"),
                "n_survivors_64": sum(1 for e in in_bucket if e["phash_survivor_64"]),
                "n_survivors_256": sum(1 for e in in_bucket if e["phash_survivor_256"]),
            }
        )

    funnel = {
        "n_sampled": len(entries),
        "n_read_errors": n_read_errors,
        "phash_threshold_64": phash_threshold_64,
        "phash_threshold_256": phash_threshold_256,
        "n_survivors_64": n_survivors_64,
        "n_survivors_256": n_survivors_256,
        "n_geometric_evaluated": n_evaluated,
        "n_forte": n_forte,
        "n_fraco": n_fraco,
        "n_not_evaluable": n_not_evaluable,
        "buckets": buckets,
        "evidence_rendered": evidence_rendered,
        "max_evidence_rendered": MAX_EVIDENCE_RENDERED,
        "elapsed_seconds": elapsed,
    }
    return entries, funnel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "dataset_inventory.yaml"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ham_dir = Path(config["ham10000"]["images_dir"])
    isic_zip = Path(config["isic2018_task1"]["images_zip"])
    bcn_content = _load_json(MANIFEST_DIR / "bcn20000_content_manifest.json")
    bcn_train_zip = Path(bcn_content["train"]["materialization"]["dest_path"])
    bcn_test_zip = Path(bcn_content["test"]["materialization"]["dest_path"])

    ham_dhash = _load_json(MANIFEST_DIR / "ham10000_dhash_manifest.json")
    isic_dhash = _load_json(MANIFEST_DIR / "isic2018_task1_dhash_manifest.json")
    bcn_dhash = _load_json(MANIFEST_DIR / "bcn20000_dhash_manifest.json")

    v1_index = _load_json(V1_INDEX_PATH)
    v1_pairs = v1_index["pairs"]
    print(f"reaproveitando amostra exata da Tarefa 8 (somente leitura): {len(v1_pairs)} pares, "
          f"semente={v1_index['seed']}")
    hard_pairs = [p for p in v1_pairs if p["dhash_distance"] <= 2]
    print(f"pares difíceis reais (dHash 0-2): {len(hard_pairs)}")

    reader = ImageSourceReader(ham_dir, isic_zip, bcn_train_zip, bcn_test_zip)
    try:
        validation = run_real_image_validation(
            reader, ham_dhash["manifest"], isic_dhash["manifest"], bcn_dhash["manifest"], hard_pairs, args.seed
        )
        threshold_64 = validation["phash_comparison"]["threshold_64"]["threshold"]
        threshold_256 = validation["phash_comparison"]["threshold_256"]["threshold"]

        entries, funnel = run_full_sample_unconditional(reader, v1_pairs, threshold_64, threshold_256)
    finally:
        reader.close()

    cross_tab = v2.build_cross_tab(entries)
    recommendation = v2.recommend_configuration(cross_tab)
    print(f"recomendação: {recommendation['decision']}")

    total_candidates = v1_index["funnel"]["total_task7_candidates"]
    n_sampled = funnel["n_sampled"]
    full_run_estimate = {
        "total_task7_candidates": total_candidates,
        "n_sampled": n_sampled,
        "estimated_survivors_64": round(total_candidates * funnel["n_survivors_64"] / n_sampled),
        "estimated_survivors_256": round(total_candidates * funnel["n_survivors_256"] / n_sampled),
        "measured_seconds_for_sample": funnel["elapsed_seconds"],
        "estimated_seconds_for_full_orb_no_gate": funnel["elapsed_seconds"] / n_sampled * total_candidates,
    }

    v2_index = {
        "seed": args.seed,
        "reused_v1_index": str(V1_INDEX_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "phash_threshold_64": threshold_64,
        "phash_threshold_256": threshold_256,
        "funnel": funnel,
        "cross_tab": cross_tab,
        "recommendation": recommendation,
        "pairs": entries,
    }
    _save_json(V2_INDEX_PATH, v2_index)
    print(f"índice v2 gravado: {V2_INDEX_PATH}")

    forte_pairs = [e for e in entries if e["geometric_strength"] == "forte"]

    report_text = v2_report.render_report(
        real_sample=validation["real_sample"],
        phash_comparison=validation["phash_comparison"],
        dhash_coverage=validation["dhash_coverage"],
        controlled_orb=validation["controlled_orb"],
        full_funnel=funnel,
        cross_tab=cross_tab,
        recommendation=recommendation,
        full_run_estimate=full_run_estimate,
        forte_pairs=forte_pairs,
        pilot_v2_index_path=V2_INDEX_PATH.relative_to(REPO_ROOT),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"relatório gravado: {REPORT_PATH}")


if __name__ == "__main__":
    main()
