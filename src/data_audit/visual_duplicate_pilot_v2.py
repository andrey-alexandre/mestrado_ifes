"""Analysis helpers for the task 9 pilot: pHash 64-bit vs 256-bit compared
head-to-head on the same protocol, dHash funnel-entry coverage, and ORB+
RANSAC run independent of any pHash gate.

Why this module and not just more code in ``scripts/run_visual_duplicate_pilot_v2.py``:
the comparison/aggregation logic here (threshold derivation, cross-tabulation,
configuration recommendation) is exactly what task 9 asks to be measured, not
just executed once and discarded -- keeping it in ``src/data_audit`` makes it
importable from both the orchestration script and the test suite, same
separation as ``phash.py``/``geometric_verification.py`` vs
``scripts/run_visual_duplicate_pilot.py``.

Detection/comparison only: nothing here decides or implements exclusion of
any lesion/image.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from src.data_audit import geometric_verification as gv
from src.data_audit import phash
from src.data_audit import visual_duplicates as vd

# 64 bits (low_freq_size=8) vs 256 bits (low_freq_size=16, task 8's original
# config) -- the exact two widths task 9 scopes this comparison to, no sweep
# of any other block size.
PHASH_64_LOW_FREQ_SIZE = 8
PHASH_256_LOW_FREQ_SIZE = 16

GEOMETRIC_MAX_DIM = 800  # same cap as task 8's pilot script, keeps ORB fast

DHASH_BUCKETS = (("0-2", 0, 2), ("3-7", 3, 7), ("8-10", 8, 10))


def compute_phash_distances(data_a: bytes, data_b: bytes) -> dict[str, int]:
    """pHash Hamming distance between two images' encoded bytes at both bit
    widths this pilot compares. Each width decodes/DCTs independently via
    ``phash.phash_bytes`` (no new hash function, no DCT-sharing shortcut --
    task 9 explicitly scopes this to comparing the two existing
    configurations, not optimizing the computation)."""
    hash_a_64 = phash.phash_bytes(data_a, low_freq_size=PHASH_64_LOW_FREQ_SIZE)
    hash_b_64 = phash.phash_bytes(data_b, low_freq_size=PHASH_64_LOW_FREQ_SIZE)
    hash_a_256 = phash.phash_bytes(data_a, low_freq_size=PHASH_256_LOW_FREQ_SIZE)
    hash_b_256 = phash.phash_bytes(data_b, low_freq_size=PHASH_256_LOW_FREQ_SIZE)
    return {
        "phash_distance_64": vd.hamming_distance(hash_a_64, hash_b_64),
        "phash_distance_256": vd.hamming_distance(hash_a_256, hash_b_256),
    }


def normalized_distance(distance: int, bits: int) -> float:
    """Hamming distance as a fraction of total bits -- the only way to
    compare a 64-bit and a 256-bit hash's distances on the same scale."""
    return distance / bits


def dhash_recovery(
    base_data: bytes, variant_data: bytes, threshold: int = vd.DEFAULT_CANDIDATE_THRESHOLD
) -> dict[str, Any]:
    """Whether a variant's dHash distance from its base image falls within
    task 7's candidate threshold, i.e. whether this transformation would
    ever have entered task 7's 421,196-candidate pool in the first place. A
    transform that fails this is invisible to every later filter (pHash,
    ORB), no matter how good those filters are.
    """
    base_hash = vd.dhash_bytes(base_data)
    variant_hash = vd.dhash_bytes(variant_data)
    distance = vd.hamming_distance(base_hash, variant_hash)
    return {"distance": distance, "recovered": distance <= threshold}


def derive_threshold_from_gap(
    variant_distances: dict[str, list[int]], hard_pair_distances: list[int]
) -> dict[str, Any]:
    """Generalization of task 8's threshold-derivation logic
    (``scripts/run_visual_duplicate_pilot.py::run_phash_validation``), usable
    for either pHash bit width: the threshold sits at the floor of the
    midpoint between the "clean" variant cluster's max distance (every named
    variant whose max distance is strictly below the hard pairs' min
    distance) and the hard pairs' min distance -- a measured gap, not an
    assumed one. If no variant cleanly separates, that is reported
    explicitly (``has_clean_gap=False``) instead of inventing a threshold,
    same fallback convention as task 8.
    """
    hard_min = min(hard_pair_distances)
    all_variant_names = list(variant_distances.keys())
    clean_variants = [name for name in all_variant_names if max(variant_distances[name]) < hard_min]
    non_clean_variants = [name for name in all_variant_names if name not in clean_variants]

    if clean_variants:
        clean_max = max(max(variant_distances[name]) for name in clean_variants)
        threshold = (clean_max + hard_min) // 2
        gap_bits = hard_min - clean_max
    else:
        clean_max = None
        gap_bits = None
        threshold = max(variant_distances.get("identical_copy", [0])) + 1

    return {
        "hard_min": hard_min,
        "clean_variants": clean_variants,
        "non_clean_variants": non_clean_variants,
        "clean_max": clean_max,
        "gap_bits": gap_bits,
        "threshold": threshold,
        "has_clean_gap": bool(clean_variants),
    }


def _dhash_bucket(distance: int) -> str | None:
    for label, lo, hi in DHASH_BUCKETS:
        if lo <= distance <= hi:
            return label
    return None


def build_cross_tab(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-tabulate the 2,000-pair sample's dHash bucket (the only "known"
    axis available here -- bucket 0-2 contains task 7's already manually
    confirmed-different-patient hard pairs; this pilot has no per-pair ground
    truth label for genuine duplication) against both pHash survivor flags
    and the geometric support classification. Answers: do pairs with strong
    ("forte") geometric support survive pHash-64? pHash-256? Do pHash
    survivors actually coincide with strong geometric support, or is there no
    correlation? How much known hard-pair noise (bucket 0-2) does each pHash
    config still retain as a false candidate?

    Each item of ``pairs`` is expected to carry ``dhash_distance``,
    ``phash_survivor_64`` (bool), ``phash_survivor_256`` (bool), and
    ``geometric_strength`` (``"forte"``, ``"fraco"``, ``"not_evaluable"``, or
    any other string -- e.g. ``"image_read_error"`` -- treated as not
    evaluated).
    """
    bucket_rows: dict[str, dict[str, Any]] = {
        label: {
            "label": label,
            "n": 0,
            "n_phash64_survivor": 0,
            "n_phash256_survivor": 0,
            "n_forte": 0,
            "n_fraco": 0,
            "n_not_evaluable": 0,
            "n_forte_survives_phash64": 0,
            "n_forte_survives_phash256": 0,
        }
        for label, _, _ in DHASH_BUCKETS
    }

    for pair in pairs:
        bucket = _dhash_bucket(pair["dhash_distance"])
        if bucket is None:
            continue
        row = bucket_rows[bucket]
        row["n"] += 1
        if pair.get("phash_survivor_64"):
            row["n_phash64_survivor"] += 1
        if pair.get("phash_survivor_256"):
            row["n_phash256_survivor"] += 1
        strength = pair.get("geometric_strength")
        if strength == "forte":
            row["n_forte"] += 1
            if pair.get("phash_survivor_64"):
                row["n_forte_survives_phash64"] += 1
            if pair.get("phash_survivor_256"):
                row["n_forte_survives_phash256"] += 1
        elif strength == "fraco":
            row["n_fraco"] += 1
        elif strength == "not_evaluable":
            row["n_not_evaluable"] += 1

    n_forte_total = sum(r["n_forte"] for r in bucket_rows.values())
    n_forte_survives_64 = sum(r["n_forte_survives_phash64"] for r in bucket_rows.values())
    n_forte_survives_256 = sum(r["n_forte_survives_phash256"] for r in bucket_rows.values())

    return {
        "buckets": bucket_rows,
        "n_total": len(pairs),
        "n_forte_total": n_forte_total,
        "n_forte_survives_phash64": n_forte_survives_64,
        "n_forte_survives_phash256": n_forte_survives_256,
        "n_forte_rejected_by_phash64": n_forte_total - n_forte_survives_64,
        "n_forte_rejected_by_phash256": n_forte_total - n_forte_survives_256,
        "n_hardpair_total": bucket_rows["0-2"]["n"],
        "n_hardpair_retained_by_phash64": bucket_rows["0-2"]["n_phash64_survivor"],
        "n_hardpair_retained_by_phash256": bucket_rows["0-2"]["n_phash256_survivor"],
    }


def recommend_configuration(cross_tab: dict[str, Any]) -> dict[str, Any]:
    """Pick among keeping pHash-64, pHash-256, dropping pHash entirely in
    favor of dHash->ORB direct, or -- if this sample found no pair with
    strong geometric support at all, so there is nothing to measure
    separation against -- an explicit bounded manual-review fallback.

    The choice between the two pHash widths (when both are viable) is driven
    by which one retains every observed "forte" pair while shedding more of
    the known hard-pair noise (bucket 0-2); never by which one simply
    shrinks the candidate list the most.
    """
    if cross_tab["n_forte_total"] == 0:
        return {
            "decision": "manual_review_fallback",
            "reasoning": [
                "Nenhum par entre os 2.000 amostrados (incluindo os 400 pares difíceis "
                "conhecidos, distância dHash 0-2) obteve classificação de suporte geométrico "
                "'forte' nesta amostra, com ou sem filtro de pHash. Não há nenhum par com "
                "suporte forte para medir se um limiar de pHash o rejeitaria -- qualquer "
                "escolha de configuração de pHash aqui seria arbitrária, não baseada em "
                "separação observada entre recuperação de positivo e retenção de ruído.",
            ],
            "residual_risk": (
                "Procedimento de revisão manual proposto: revisar diretamente, sem qualquer "
                "filtro de pHash antes, os pares desta amostra com maior inlier_ratio/n_inliers "
                "entre os avaliáveis geometricamente (mesmo os classificados 'fraco', já que "
                "'forte' não ocorreu nenhuma vez aqui), ordenados por força do suporte ORB. "
                "Risco residual explícito e não mitigado por este procedimento: uma duplicata "
                "real cujas duas imagens produzam poucos keypoints ORB (pele lisa, baixo "
                "contraste, textura genérica) permanece 'não avaliável' por este método "
                "inteiro, independente de qualquer hash -- nenhum procedimento baseado em ORB "
                "a resgataria, e essa lacuna não foi medida nem mitigada por nenhuma tarefa "
                "até agora."
            ),
        }

    retains_64 = cross_tab["n_forte_survives_phash64"] == cross_tab["n_forte_total"]
    retains_256 = cross_tab["n_forte_survives_phash256"] == cross_tab["n_forte_total"]

    if retains_64 and not retains_256:
        decision = "keep_phash_64"
    elif retains_256 and not retains_64:
        decision = "keep_phash_256"
    elif retains_64 and retains_256:
        if cross_tab["n_hardpair_retained_by_phash64"] < cross_tab["n_hardpair_retained_by_phash256"]:
            decision = "keep_phash_64"
        elif cross_tab["n_hardpair_retained_by_phash256"] < cross_tab["n_hardpair_retained_by_phash64"]:
            decision = "keep_phash_256"
        else:
            decision = "keep_phash_64"  # tie: 64 bits is cheaper to compute/store
    else:
        decision = "drop_phash_use_dhash_orb_direct"

    reasoning = [
        f"Pares com suporte geométrico 'forte' observados na amostra de 2.000: "
        f"{cross_tab['n_forte_total']}.",
        f"Desses, sobreviveriam ao pHash-64: {cross_tab['n_forte_survives_phash64']} "
        f"({cross_tab['n_forte_rejected_by_phash64']} seriam rejeitados antes da etapa geométrica).",
        f"Desses, sobreviveriam ao pHash-256: {cross_tab['n_forte_survives_phash256']} "
        f"({cross_tab['n_forte_rejected_by_phash256']} seriam rejeitados antes da etapa geométrica).",
        f"Pares difíceis conhecidos (bucket dHash 0-2, n={cross_tab['n_hardpair_total']}) retidos "
        f"como falso candidato pelo pHash-64: {cross_tab['n_hardpair_retained_by_phash64']}; "
        f"pelo pHash-256: {cross_tab['n_hardpair_retained_by_phash256']}.",
    ]

    return {"decision": decision, "reasoning": reasoning, "residual_risk": None}


def decode_grayscale_array(data: bytes, max_dim: int = GEOMETRIC_MAX_DIM) -> np.ndarray:
    """Decode raw image bytes to a grayscale uint8 array for ORB, capping the
    longer side at ``max_dim``. Mirrors
    ``scripts/run_visual_duplicate_pilot.py::_decode_grayscale_array`` so
    task 8's pilot and this task's unconditional (no pHash gate) run decode
    identically.
    """
    image = Image.open(io.BytesIO(data)).convert("L")
    w, h = image.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return np.array(image)


def verify_pair_from_bytes(data_a: bytes, data_b: bytes) -> dict[str, Any]:
    """Decode both sides and run ORB+RANSAC verification directly -- the
    "independent of any pHash filter" entry point task 9 requires: this
    function takes raw image bytes straight from the source, never a
    pre-filtered candidate list, so nothing upstream of it can silently gate
    it behind a pHash survivor check.
    """
    array_a = decode_grayscale_array(data_a)
    array_b = decode_grayscale_array(data_b)
    result = gv.verify_geometric_match(array_a, array_b)
    strength = gv.classify_geometric_support(result)
    return {"result": result, "strength": strength}
