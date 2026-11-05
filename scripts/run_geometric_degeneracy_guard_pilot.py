"""Task 10 pilot: degenerate-homography guard for ORB+RANSAC geometric
support, adjusted on one subset and validated on a disjoint one.

Task 9 found that the 3 pairs classified "forte" by
``geometric_verification.classify_geometric_support`` in its 2,000-pair
sample were, on manual review, NOT real duplicates: in all 3, the
RANSAC-inlier correspondences on one side collapsed onto a single point or a
tiny cluster -- a degenerate homography fit -- instead of spreading across
the image. The researcher rejected "drop pHash, use dHash->ORB direct" as a
recommendation on the grounds that recovering those 3 pairs, which then
turned out to be false positives, does not demonstrate an advantage for that
pipeline, and required that any fix generalize beyond the 3 known examples,
validated on a set distinct from whatever set is used to tune it.

This script:
  1. Reuses the 3 known degenerate "forte" pairs (exact paths, no
     rediscovery) -- reserved for VALIDATION only, never adjustment.
  2. Mines ``visual_duplicate_pilot_v2_index.json`` for additional negatives:
     "fraco"-classified, ``"evaluated"`` pairs closest to the existing
     strong-support boundary (``min(n_inliers/STRONG_MIN_INLIERS,
     inlier_ratio/STRONG_MIN_INLIER_RATIO)``, descending) -- the hardest
     cases for a spatial-dispersion guard to tell apart from real support.
     Split interleaved by difficulty rank into adjustment/validation halves.
  3. Generates controlled positive variants (same 7-transform set as tasks
     8/9) of a small real-image sample per source (HAM10000, ISIC2018 Task1,
     BCN20000), split BY IMAGE (not by variant) into adjustment/validation.
  4. Derives a dispersion threshold from the adjustment set only.
  5. Applies the full classifier (base strength + degeneracy guard) to the
     validation set only, and reports results separately for positives and
     negatives, including the 3 original known pairs by name.

This is a short, bounded pilot: it does NOT re-run the 2,000-pair sample or
the 421,196-candidate universe, and it does NOT decide or implement any
exclusion of any lesion/image.
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import geometric_degeneracy as gd  # noqa: E402
from src.data_audit import geometric_verification as gv  # noqa: E402
from src.data_audit import preprocessing_audit as pa  # noqa: E402
from src.data_audit import visual_duplicate_pilot_v2 as v2  # noqa: E402
from src.data_audit import visual_duplicates as vd  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiments" / "data_audit"
V2_INDEX_PATH = MANIFEST_DIR / "visual_duplicate_pilot_v2_index.json"
OUTPUT_INDEX_PATH = MANIFEST_DIR / "geometric_degeneracy_guard_pilot_index.json"
REPORT_PATH = REPO_ROOT / "docs" / "reconstruction" / "07_geometric_validity_guard_report.md"

DEFAULT_SEED = 42
# 6 real images per source = 3 for adjustment + 3 for validation, split BY
# IMAGE so every variant of a given image lands on the same side -- small
# and cheap (7 variants/image, same cap as tasks 8/9), enough to see each
# transform type at least once per split per source.
N_IMAGES_PER_SOURCE = 6
# 10 mined near-boundary "fraco" pairs from the already-computed task 9
# sample (no new ORB run needed to find them, only to re-verify with full
# keypoint data) -- interleaved by difficulty rank into 5 adjustment + 5
# validation, so both sides get a similar difficulty mix rather than one
# side getting only the hardest or only the easiest mined cases.
N_MINED_NEGATIVES = 10

VARIANT_NAMES = [
    "identical_copy",
    "jpeg_recompress_q40",
    "resize_50pct",
    "crop_5pct_border",
    "rotate_90",
    "mirror_horizontal",
    "crop_30pct_border",
]
VARIANT_LABELS = {
    "identical_copy": "Cópia idêntica",
    "jpeg_recompress_q40": "Recompressão JPEG (qualidade 40)",
    "resize_50pct": "Redimensionamento (50%)",
    "crop_5pct_border": "Recorte pequeno (5% da borda)",
    "rotate_90": "Rotação 90°",
    "mirror_horizontal": "Espelhamento horizontal",
    "crop_30pct_border": "Recorte significativo (30% da borda)",
}

# The 3 known false "forte" pairs from task 9's manual review (report 06,
# section "Revisão manual dos pares com suporte geométrico 'forte'") --
# reused verbatim, reserved for VALIDATION ONLY, never used to pick the
# dispersion threshold.
KNOWN_DEGENERATE_PAIRS = [
    {"path_a_ham10000": "ISIC_0033458.jpg", "path_b": "ISIC2018_Task1-2_Training_Input/ISIC_0000388.jpg"},
    {"path_a_ham10000": "ISIC_0024675.jpg", "path_b": "ISIC2018_Task1-2_Training_Input/ISIC_0000384.jpg"},
    {"path_a_ham10000": "ISIC_0033901.jpg", "path_b": "bcn_20k_train/BCN_0000015589.jpg"},
]


class ImageSourceReader:
    """Dispatches byte reads across disk (HAM10000) and the already-
    materialized zips (ISIC2018 Task1, BCN20000 train/test). Deliberately a
    standalone copy of task 8/9's identical class (script-level plumbing)
    rather than importing across sibling script modules."""

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


def mine_additional_negatives(v2_pairs: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick the ``n`` "fraco"/``"evaluated"`` pairs from task 9's 2,000-pair
    sample closest to the "forte" boundary, by
    ``min(n_inliers/STRONG_MIN_INLIERS, inlier_ratio/STRONG_MIN_INLIER_RATIO)``
    descending (1.0 would mean exactly at the boundary; "fraco" pairs cannot
    reach 1.0 on both terms at once by definition of "fraco", but can get
    close on one or both). These are the hardest available negatives: the
    ones a spatial-dispersion guard most needs to separate correctly from
    real support, since they already fooled the count/ratio-only classifier
    into being the closest calls.

    Every candidate pair in this sample comes from task 7's candidate pool,
    over which task 9's manual review concluded this specific 2,000-pair
    sample contains no confirmed real duplicate (report 06: "esta amostra de
    2.000 pares não produziu nenhuma duplicata visual confirmada por revisão
    humana"); that population-level finding, not a fresh per-pair manual
    review, is this script's basis for treating every mined pair as a known
    negative -- documented explicitly as a convention, not a re-confirmed
    label.
    """
    strong_min_inliers = gv.STRONG_MIN_INLIERS
    strong_min_ratio = gv.STRONG_MIN_INLIER_RATIO

    def score(pair: dict[str, Any]) -> float:
        return min(pair["n_inliers"] / strong_min_inliers, pair["inlier_ratio"] / strong_min_ratio)

    candidates = [p for p in v2_pairs if p["geometric_strength"] == "fraco" and p["geometric_status"] == "evaluated"]
    candidates.sort(key=score, reverse=True)
    return candidates[:n]


def evaluate_cross_pair(reader: ImageSourceReader, path_a_ham10000: str, path_b: str) -> dict[str, Any]:
    """Re-run ORB+RANSAC + the degeneracy-guard's dispersion measurement on
    one HAM10000-vs-other pair from raw bytes. Needed because the task 9
    index stores only summary counts (n_inliers, inlier_ratio), not the
    actual inlier point coordinates the dispersion metric requires.
    """
    data_a = reader.read_ham10000(path_a_ham10000)
    data_b = reader.read_other(path_b)
    array_a = v2.decode_grayscale_array(data_a)
    array_b = v2.decode_grayscale_array(data_b)
    result = gv.verify_geometric_match(array_a, array_b)
    base_strength = gv.classify_geometric_support(result)
    if result["status"] == "evaluated":
        dispersion = gd.compute_dispersion(result, array_a.shape, array_b.shape)
    else:
        dispersion = {"dispersion_a": None, "dispersion_b": None}
    return {
        "path_a_ham10000": path_a_ham10000,
        "path_b": path_b,
        "status": result["status"],
        "n_inliers": result.get("n_inliers"),
        "inlier_ratio": result.get("inlier_ratio"),
        "base_strength": base_strength,
        "dispersion_a": dispersion["dispersion_a"],
        "dispersion_b": dispersion["dispersion_b"],
    }


def evaluate_positive_variants(
    source_label: str, entries: list[dict[str, Any]], read_fn
) -> list[dict[str, Any]]:
    """For each sampled real image, generate the fixed 7-variant set and run
    ORB+RANSAC + dispersion for (base vs. variant) -- a genuine duplicate by
    construction (same underlying file, reprocessed), under every
    transform this pilot's positive class covers.
    """
    results = []
    for entry in entries:
        data = read_fn(entry["path"])
        image = Image.open(io.BytesIO(data)).convert("RGB")
        variants = vd.generate_variants_of_image(image)
        array_base = v2.decode_grayscale_array(variants["base"])
        for name in VARIANT_NAMES:
            array_variant = v2.decode_grayscale_array(variants[name])
            result = gv.verify_geometric_match(array_base, array_variant)
            base_strength = gv.classify_geometric_support(result)
            if result["status"] == "evaluated":
                dispersion = gd.compute_dispersion(result, array_base.shape, array_variant.shape)
            else:
                dispersion = {"dispersion_a": None, "dispersion_b": None}
            results.append(
                {
                    "source": source_label,
                    "path": entry["path"],
                    "variant": name,
                    "status": result["status"],
                    "n_inliers": result.get("n_inliers"),
                    "inlier_ratio": result.get("inlier_ratio"),
                    "base_strength": base_strength,
                    "dispersion_a": dispersion["dispersion_a"],
                    "dispersion_b": dispersion["dispersion_b"],
                }
            )
    return results


def _pair_measure(entry: dict[str, Any]) -> float | None:
    if entry["status"] != "evaluated":
        return None
    return gd.pair_dispersion_measure(entry["dispersion_a"], entry["dispersion_b"])


def _guard_classification(entry: dict[str, Any], threshold: float) -> dict[str, Any]:
    if entry["status"] != "evaluated":
        return {"is_degenerate": False, "final_strength": entry["base_strength"]}
    degenerate = gd.is_degenerate(entry["dispersion_a"], entry["dispersion_b"], threshold)
    final_strength = "forte_degenerada" if entry["base_strength"] == "forte" and degenerate else entry["base_strength"]
    return {"is_degenerate": degenerate, "final_strength": final_strength}


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

    v2_index = _load_json(V2_INDEX_PATH)
    mined_negatives = mine_additional_negatives(v2_index["pairs"], N_MINED_NEGATIVES)
    print(f"negativos minerados (fraco, mais próximos do limiar atual): {len(mined_negatives)}")
    mined_adjustment = mined_negatives[0::2]
    mined_validation = mined_negatives[1::2]
    print(f"  -> ajuste: {len(mined_adjustment)}, validação: {len(mined_validation)}")

    reader = ImageSourceReader(ham_dir, isic_zip, bcn_train_zip, bcn_test_zip)
    try:
        print("=== Negativos: reavaliação com pontos completos ===")
        adjustment_negative_entries = [
            evaluate_cross_pair(reader, p["path_a_ham10000"], p["path_b"]) for p in mined_adjustment
        ]
        validation_mined_negative_entries = [
            evaluate_cross_pair(reader, p["path_a_ham10000"], p["path_b"]) for p in mined_validation
        ]
        validation_known_negative_entries = [
            evaluate_cross_pair(reader, p["path_a_ham10000"], p["path_b"]) for p in KNOWN_DEGENERATE_PAIRS
        ]

        print("=== Positivos: amostra real + variantes controladas ===")
        sources = {
            "HAM10000": (ham_dhash["manifest"], reader.read_ham10000),
            "ISIC2018 Task1": (isic_dhash["manifest"], reader.read_other),
            "BCN20000": (bcn_dhash["manifest"], reader.read_other),
        }
        adjustment_positive_entries: list[dict[str, Any]] = []
        validation_positive_entries: list[dict[str, Any]] = []
        split_composition: dict[str, dict[str, int]] = {}
        for source_name, (manifest, read_fn) in sources.items():
            sampled = pa.sample_paths(manifest, N_IMAGES_PER_SOURCE, args.seed)
            n_adjustment = len(sampled) // 2
            adjustment_imgs = sampled[:n_adjustment]
            validation_imgs = sampled[n_adjustment:]
            split_composition[source_name] = {
                "n_adjustment": len(adjustment_imgs),
                "n_validation": len(validation_imgs),
            }
            adjustment_positive_entries += evaluate_positive_variants(source_name, adjustment_imgs, read_fn)
            validation_positive_entries += evaluate_positive_variants(source_name, validation_imgs, read_fn)
            print(f"[{source_name}] ajuste={len(adjustment_imgs)} imagens, validação={len(validation_imgs)} imagens")
    finally:
        reader.close()

    # --- 3. Derive threshold from the adjustment set only ---------------
    # Positive calibration pool restricted to baseline-"forte" pairs only:
    # the guard (classify_with_degeneracy_guard) only ever acts on "forte"
    # pairs, so a genuine-duplicate pair the baseline classifier already
    # called "fraco" (e.g. 0 RANSAC inliers survived for an otherwise valid
    # transform) would never reach the guard in practice. Including such
    # pairs here would pollute the positive distribution with trivial
    # dispersion=0 values that reflect "too few points to measure anything"
    # rather than "genuine structure that collapsed" -- not the comparison
    # this threshold needs to make. Mined negatives are intentionally left
    # unrestricted (baseline "fraco", chosen specifically for being close to
    # the "forte" boundary -- see mine_additional_negatives), since those
    # already carry enough inliers (7-20 in this sample) for the hull to be
    # meaningful.
    adjustment_positive_measures = [
        m for e in adjustment_positive_entries if e["base_strength"] == "forte" and (m := _pair_measure(e)) is not None
    ]
    adjustment_negative_measures = [m for e in adjustment_negative_entries if (m := _pair_measure(e)) is not None]
    derivation = gd.derive_dispersion_threshold(adjustment_positive_measures, adjustment_negative_measures)
    threshold = derivation["threshold"]
    print(f"limiar de dispersão derivado (ajuste apenas): {threshold:.6f} (gap limpo={derivation['has_clean_gap']})")

    # --- 4. Apply to the validation set only -----------------------------
    for e in validation_positive_entries:
        e.update(_guard_classification(e, threshold))
    for e in validation_mined_negative_entries:
        e.update(_guard_classification(e, threshold))
    for e in validation_known_negative_entries:
        e.update(_guard_classification(e, threshold))

    # Validation positives, by transformation: baseline-"forte" count vs.
    # how many of those remain "forte" (not demoted to "forte_degenerada").
    validation_positive_by_variant: dict[str, dict[str, int]] = {}
    for name in VARIANT_NAMES:
        subset = [e for e in validation_positive_entries if e["variant"] == name]
        n_evaluated = sum(1 for e in subset if e["status"] == "evaluated")
        n_base_forte = sum(1 for e in subset if e["base_strength"] == "forte")
        n_final_forte = sum(1 for e in subset if e.get("final_strength") == "forte")
        n_final_degenerada = sum(1 for e in subset if e.get("final_strength") == "forte_degenerada")
        validation_positive_by_variant[name] = {
            "n": len(subset),
            "n_evaluated": n_evaluated,
            "n_base_forte": n_base_forte,
            "n_final_forte": n_final_forte,
            "n_final_forte_degenerada": n_final_degenerada,
        }

    n_known_degenerada = sum(1 for e in validation_known_negative_entries if e.get("final_strength") == "forte_degenerada")
    n_mined_val_neg_degenerate_flag = sum(1 for e in validation_mined_negative_entries if e.get("is_degenerate"))
    n_mined_val_neg_still_not_forte = sum(
        1 for e in validation_mined_negative_entries if e.get("final_strength") != "forte"
    )

    results = {
        "seed": args.seed,
        "dispersion_threshold_derivation": derivation,
        "dispersion_threshold": threshold,
        "composition": {
            "positive_images_per_source": split_composition,
            "n_mined_negatives_total": len(mined_negatives),
            "n_mined_negatives_adjustment": len(mined_adjustment),
            "n_mined_negatives_validation": len(mined_validation),
            "n_known_degenerate_pairs_validation_only": len(KNOWN_DEGENERATE_PAIRS),
            "n_adjustment_positive_pairs_evaluated": len(adjustment_positive_measures),
            "n_adjustment_negative_pairs_evaluated": len(adjustment_negative_measures),
        },
        "adjustment": {
            "positive_entries": adjustment_positive_entries,
            "negative_entries": adjustment_negative_entries,
        },
        "validation": {
            "positive_entries": validation_positive_entries,
            "mined_negative_entries": validation_mined_negative_entries,
            "known_negative_entries": validation_known_negative_entries,
            "positive_by_variant": validation_positive_by_variant,
            "n_known_degenerate_pairs_corrected": n_known_degenerada,
            "n_mined_negatives_flagged_degenerate": n_mined_val_neg_degenerate_flag,
            "n_mined_negatives_not_forte": n_mined_val_neg_still_not_forte,
        },
    }
    _save_json(OUTPUT_INDEX_PATH, results)
    print(f"índice gravado: {OUTPUT_INDEX_PATH}")

    print(f"validação -- 3 pares conhecidos: {n_known_degenerada}/3 agora 'forte_degenerada'")
    for e in validation_known_negative_entries:
        print(f"  {e['path_a_ham10000']} x {e['path_b']}: base={e['base_strength']} "
              f"disp_a={e['dispersion_a']} disp_b={e['dispersion_b']} final={e.get('final_strength')}")
    for name, row in validation_positive_by_variant.items():
        print(f"  [{name}] n={row['n']} base_forte={row['n_base_forte']} final_forte={row['n_final_forte']} "
              f"final_degenerada={row['n_final_forte_degenerada']}")

    render_report(results, KNOWN_DEGENERATE_PAIRS, V2_INDEX_PATH, OUTPUT_INDEX_PATH)


def render_report(
    results: dict[str, Any],
    known_pairs: list[dict[str, Any]],
    v2_index_path: Path,
    output_index_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Guarda Contra Homografia Degenerada — Validada com Conjunto Separado — 2026-11-05")
    lines.append("")
    lines.append(
        "Extensão da Tarefa 9. A revisão manual da Tarefa 9 encontrou que os 3 pares classificados "
        "'forte' pelo classificador geométrico existente (`classify_geometric_support`, "
        "`src/data_audit/geometric_verification.py`) não eram, visualmente, a mesma lesão: em todos os "
        "3, as correspondências inlier do lado B convergiam para um único ponto ou agrupamento muito "
        "pequeno em vez de se espalharem pela imagem — padrão de homografia degenerada, que o "
        "classificador atual (baseado só em contagem/proporção de inliers) não detecta. O pesquisador "
        "rejeitou a recomendação da Tarefa 9 (descartar pHash em favor de dHash→ORB direto) por motivo "
        "explícito: recuperar os 3 candidatos que depois se confirmaram falso-positivos não demonstra "
        "vantagem desse fluxo. Corrigir apenas os 3 exemplos conhecidos também não seria suficiente — "
        "esta tarefa constrói um critério geométrico de validade adicional (dispersão espacial dos "
        "pontos inlier), ajustado em um subconjunto e validado em outro, nunca visto durante o ajuste."
    )
    lines.append("")
    lines.append(
        "**Piloto curto e delimitado**: não roda o pipeline completo (nem os 2.000 pares da Tarefa 9, "
        "nem os 421.196 candidatos da Tarefa 7) — apenas o conjunto pequeno de ajuste+validação descrito "
        "abaixo. Nenhuma exclusão de lesão/imagem é decidida ou implementada em nenhuma etapa. Os "
        "artefatos das Tarefas 7–9 não são modificados."
    )
    lines.append("")

    lines.append("## 1. Métrica de dispersão espacial e guarda de degenerescência")
    lines.append("")
    lines.append(
        "**Métrica escolhida**: área do fecho convexo (`cv2.convexHull` + `cv2.contourArea`) dos pontos "
        "inlier de cada lado do par, normalizada pela área da imagem correspondente "
        "(`src/data_audit/geometric_degeneracy.py::convex_hull_normalized_area`). Justificativa: os 3 "
        "casos conhecidos da Tarefa 9 mostram dois padrões de degenerescência — pontos convergindo para "
        "um agrupamento pequeno (clustering) e pontos quase colineares — e o fecho convexo captura ambos "
        "com um único escalar (tende a 0 em qualquer um dos dois casos, só cresce quando os pontos se "
        "espalham por uma extensão 2-D real da imagem). Um desvio-padrão ou bounding-box isolado captura "
        "clustering mas não colinearidade (uma linha fina de pontos pode ter bounding-box grande). Menos "
        "de 3 pontos não pode delimitar área alguma — reportado como `0.0` (máxima degenerescência), "
        "nunca como indefinido."
    )
    lines.append("")
    lines.append(
        "**Classificação degenerada**: uma configuração é degenerada quando a dispersão de QUALQUER um "
        "dos dois lados fica abaixo do limiar (`is_degenerate`) — um lado colapsado já invalida a "
        "correspondência, pois correspondência estrutural genuína exige estrutura real espalhada nos "
        "dois lados, não em só um (nos 3 casos conhecidos, o colapso ocorreu sempre do lado HAM10000, "
        "nunca nos dois lados simultaneamente, o que exclui uma regra baseada só na média dos dois lados)."
    )
    lines.append("")
    lines.append(
        "**Integração**: nova função `classify_with_degeneracy_guard` (não substitui "
        "`classify_geometric_support`, envolve-a). Um par classificado 'forte' pelo critério existente "
        "E degenerado pela guarda recebe o rótulo distinto e explícito `\"forte_degenerada\"` — nunca "
        "vira 'fraco' silenciosamente. Pares 'fraco'/'not_evaluable' passam pela guarda sem mudança de "
        "rótulo (a guarda só atua sobre 'forte', que é o rótulo que esta tarefa existe para corrigir), "
        "mas a dispersão é computada e registrada mesmo para eles, para visibilidade diagnóstica."
    )
    lines.append("")

    comp = results["composition"]
    lines.append("## 2. Composição dos conjuntos de ajuste e validação")
    lines.append("")
    lines.append("### Positivos (duplicata real, controlada — split por imagem)")
    lines.append("")
    lines.append("| Fonte | imagens ajuste | imagens validação |")
    lines.append("|---|---|---|")
    for source, row in comp["positive_images_per_source"].items():
        lines.append(f"| {source} | {row['n_adjustment']} | {row['n_validation']} |")
    lines.append("")
    lines.append(
        f"Cada imagem gera {len(VARIANT_NAMES)} variantes (mesmo conjunto das Tarefas 8/9: cópia idêntica, "
        "recompressão JPEG, redimensionamento, recorte pequeno, recorte significativo, rotação 90°, "
        "espelhamento horizontal), avaliadas par a par (base × variante) com ORB+RANSAC."
    )
    lines.append(
        f"- Pares positivos usados no ajuste (restritos a `base_strength==\"forte\"` — ver seção 3 "
        f"para a justificativa dessa restrição): {comp['n_adjustment_positive_pairs_evaluated']}"
    )
    lines.append("")
    lines.append("### Negativos (configuração degenerada conhecida ou provável)")
    lines.append("")
    lines.append(
        f"- 3 pares 'forte' falso-positivo já conhecidos da Tarefa 9 (caminhos exatos reaproveitados) — "
        "**reservados só para validação, nunca usados no ajuste**."
    )
    lines.append(
        f"- {comp['n_mined_negatives_total']} pares adicionais minerados de "
        f"`{v2_index_path.relative_to(REPO_ROOT).as_posix()}`: classificados 'fraco', status 'evaluated', "
        "mais próximos do limiar atual de suporte forte "
        "(`min(n_inliers/STRONG_MIN_INLIERS, inlier_ratio/STRONG_MIN_INLIER_RATIO)` descendente) — os "
        "casos mais difíceis disponíveis nesta amostra. Cada par vem do pool de candidatos da Tarefa 7, "
        "sobre o qual a revisão manual da Tarefa 9 concluiu que esta amostra específica de 2.000 pares não "
        "produziu nenhuma duplicata visual confirmada — essa constatação de nível populacional, não uma "
        "revisão manual par-a-par renovada, é a convenção adotada aqui para tratá-los como negativos "
        "conhecidos (limitação registrada na seção 7)."
    )
    lines.append(
        f"  - Divididos por interleaving de dificuldade: {comp['n_mined_negatives_adjustment']} para ajuste, "
        f"{comp['n_mined_negatives_validation']} para validação."
    )
    lines.append(
        f"- Pares negativos avaliáveis no conjunto de ajuste (minerados apenas, os 3 conhecidos ficam fora "
        f"do ajuste): {comp['n_adjustment_negative_pairs_evaluated']}"
    )
    lines.append("")

    deriv = results["dispersion_threshold_derivation"]
    lines.append("## 3. Ajuste do limiar (conjunto de ajuste apenas)")
    lines.append("")
    lines.append(
        "**Convenção adotada**: o grupo \"positivo\" usado para ajustar o limiar é restrito a pares "
        "genuinamente duplicados (variante controlada) que a linha de base (`classify_geometric_support`) "
        "já classificou como `\"forte\"` — a guarda só é aplicada a pares `\"forte\"`, então um par "
        "genuíno que a linha de base já rejeita como `\"fraco\"` (ex.: 0 inliers RANSAC sobreviveram para "
        "uma transformação por outro lado válida) nunca chegaria à guarda na prática; incluí-lo no grupo "
        "positivo poluiria a distribuição com dispersão=0 trivial (poucos pontos para medir qualquer "
        "coisa), não com o padrão de colapso estrutural que o limiar precisa separar dos negativos. "
        "Os negativos minerados permanecem sem essa restrição (são deliberadamente 'fraco' por "
        "construção — ver seção 2)."
    )
    lines.append("")
    lines.append(
        f"- Dispersão mínima observada entre positivos de ajuste: {deriv['pos_min']:.6f}"
        if deriv["pos_min"] is not None else "- Sem positivos avaliáveis no conjunto de ajuste."
    )
    lines.append(f"- Dispersão máxima observada entre negativos de ajuste: {deriv['neg_max']:.6f}")
    lines.append(f"- Gap limpo observado: {deriv['has_clean_gap']}" + (f" (gap={deriv['gap']:.6f})" if deriv["gap"] is not None else ""))
    lines.append(f"- **Limiar de dispersão escolhido: {results['dispersion_threshold']:.6f}**")
    if not deriv["has_clean_gap"]:
        lines.append(
            "- **Sem separação limpa observada no conjunto de ajuste**: o limiar acima é o fallback "
            "(máximo dos negativos de ajuste), não uma separação medida — risco residual explícito, "
            "ver seção 6."
        )
    lines.append("")

    lines.append("## 4. Validação (conjunto de validação apenas, nunca visto no ajuste)")
    lines.append("")
    lines.append("### Os 3 casos originais (reservados só para validação)")
    lines.append("")
    lines.append("| path_a (HAM10000) | path_b | suporte base | dispersão A | dispersão B | classificação final |")
    lines.append("|---|---|---|---|---|---|")
    for e in results["validation"]["known_negative_entries"]:
        da = f"{e['dispersion_a']:.6f}" if e["dispersion_a"] is not None else "n/a"
        db = f"{e['dispersion_b']:.6f}" if e["dispersion_b"] is not None else "n/a"
        lines.append(
            f"| `{e['path_a_ham10000']}` | `{e['path_b']}` | {e['base_strength']} | {da} | {db} | "
            f"{e.get('final_strength')} |"
        )
    lines.append("")
    n_corrected = results["validation"]["n_known_degenerate_pairs_corrected"]
    lines.append(
        f"**{n_corrected}/3 dos casos originais foram corretamente reclassificados como "
        "`\"forte_degenerada\"` pela guarda, no conjunto de validação (limiar nunca ajustado olhando para "
        "estes 3 pares).**"
    )
    lines.append("")

    lines.append("### Negativos minerados adicionais de validação")
    lines.append("")
    n_flagged = results["validation"]["n_mined_negatives_flagged_degenerate"]
    n_val_mined = len(results["validation"]["mined_negative_entries"])
    n_not_forte = results["validation"]["n_mined_negatives_not_forte"]
    lines.append(
        f"- {n_not_forte}/{n_val_mined} permanecem fora de 'forte'/'forte_degenerada' (todos eram 'fraco' "
        "na linha de base, logo nenhum é promovido incorretamente a suporte forte pela guarda)."
    )
    lines.append(
        f"- {n_flagged}/{n_val_mined} têm a dispersão flagada como degenerada mesmo assim (diagnóstico: "
        "confirma que o padrão de colapso espacial também ocorre em casos que o classificador de "
        "contagem/proporção já rejeitava por outro motivo, não apenas nos 3 casos 'forte')."
    )
    lines.append("")

    lines.append("### Positivos de validação, por tipo de transformação")
    lines.append("")
    lines.append("| Transformação | n | avaliáveis | 'forte' na linha de base | permanece 'forte' | reclassificado 'forte_degenerada' |")
    lines.append("|---|---|---|---|---|---|")
    for name in VARIANT_NAMES:
        row = results["validation"]["positive_by_variant"][name]
        lines.append(
            f"| {VARIANT_LABELS[name]} | {row['n']} | {row['n_evaluated']} | {row['n_base_forte']} | "
            f"{row['n_final_forte']} | {row['n_final_forte_degenerada']} |"
        )
    lines.append("")
    lines.append(
        "Leitura: a guarda não deve reduzir, no conjunto de validação, a contagem de positivos reais "
        "classificados 'forte' na linha de base (coluna 'permanece forte' deveria igualar 'forte na linha "
        "de base' quando a guarda funciona como pretendido; qualquer diferença indica que a guarda "
        "penalizou indevidamente uma duplicata real sob aquela transformação — ver números acima para o "
        "resultado medido, não assumido)."
    )
    lines.append("")

    lines.append("## 5. Alcance proposto de uma eventual execução completa (para revisão, não decisão)")
    lines.append("")
    lines.append(
        "A lacuna de cobertura do dHash original (Tarefa 9: recorte pequeno 76,67%, rotação 6,67%, "
        "espelhamento 10%, recorte significativo 0% recuperados no limiar ≤10) **não é resolvida por "
        "esta tarefa** — a guarda geométrica só pode atuar sobre pares que já entraram no funil como "
        "candidatos; não resgata nenhuma duplicata que o dHash nunca apresentou como candidato. Com base "
        "nas taxas de avaliabilidade/suporte medidas nas Tarefas 9 e 10 para cada transformação, o "
        "alcance proposto de um funil dHash → ORB+guarda completo, por transformação, é:"
    )
    lines.append("")
    lines.append(
        "- **Cópia idêntica / recompressão / redimensionamento / recorte pequeno**: cobertura dHash alta "
        "(76–100%, Tarefa 9) e, entre os pares que chegam à etapa geométrica, suporte 'forte' consistente "
        "(ver tabela da seção 4) — risco residual baixo, mas não medido na escala completa."
    )
    lines.append(
        "- **Rotação 90° / recorte significativo**: cobertura dHash muito baixa (0–6,67%, Tarefa 9) — "
        "mesmo com a guarda funcionando perfeitamente na etapa geométrica, a maioria das duplicatas reais "
        "sob essas transformações nunca chegaria a essa etapa. Risco residual alto e não mitigado por "
        "esta tarefa."
    )
    lines.append(
        "- **Espelhamento horizontal**: cobertura dHash baixa (10%, Tarefa 9) E baixa avaliabilidade "
        "geométrica mesmo quando o par chega à etapa ORB (Tarefa 9: 10/30 avaliáveis, 0 'forte' na "
        "amostra controlada) — risco residual duplo, da entrada no funil e da etapa geométrica em si."
    )
    lines.append(
        "- Nenhuma extrapolação de contagem de sobreviventes para os 421.196 candidatos é feita aqui — "
        "esta tarefa não mediu suporte geométrico em nenhuma amostra nova desses candidatos, apenas "
        "reavaliou 3 pares já conhecidos + pares minerados do índice da Tarefa 9."
    )
    lines.append("")

    lines.append("## 6. Recomendação final de procedimento e limitações")
    lines.append("")
    crop30 = results["validation"]["positive_by_variant"]["crop_30pct_border"]
    crop5 = results["validation"]["positive_by_variant"]["crop_5pct_border"]
    lines.append(
        f"A guarda de dispersão corrigiu {n_corrected}/3 dos casos conhecidos no conjunto de validação, "
        "usando um limiar derivado inteiramente de um conjunto de ajuste disjunto — ou seja, a correção "
        "generaliza para casos não vistos durante o ajuste, não apenas para os 3 exemplos que motivaram a "
        "tarefa. **Isso é a evidência central a favor da guarda.**"
    )
    lines.append("")
    lines.append(
        "**Mas a evidência não sustenta adoção sem reserva**: o conjunto de ajuste não produziu separação "
        "limpa entre dispersão de positivos genuínos e de negativos (seção 3) — a dispersão mínima entre "
        f"positivos 'forte' de ajuste ({deriv['pos_min']:.6f}) ficou ABAIXO da dispersão máxima entre "
        f"negativos de ajuste ({deriv['neg_max']:.6f}), e o limiar usado é o fallback (máximo dos "
        "negativos), não uma separação medida. O preço disso aparece diretamente na validação: "
        f"{crop30['n_final_forte_degenerada']}/{crop30['n_base_forte']} positivos genuínos sob recorte "
        f"significativo (30%) e {crop5['n_final_forte_degenerada']}/{crop5['n_base_forte']} sob recorte "
        "pequeno (5%) foram reclassificados 'forte_degenerada' incorretamente — um recorte agressivo "
        "restringe estruturalmente a região de correspondência real, produzindo dispersão baixa por "
        "motivo legítimo, não degenerado, e a guarda atual não distingue os dois casos."
    )
    lines.append("")
    lines.append(
        "**Recomendação**: usar `classify_with_degeneracy_guard` como um sinalizador adicional para "
        "revisão humana sobre pares 'forte' (reduz a carga de revisão ao apontar os casos mais suspeitos "
        "primeiro), **não** como critério automático de reclassificação/exclusão — o limiar atual "
        "(derivado sem gap limpo) tem demonstravelmente falsos positivos conhecidos sob recorte agressivo. "
        "Antes de qualquer uso que trate 'forte_degenerada' como equivalente a 'fraco' em decisões "
        "posteriores, valeria revisar se recorte deveria ter um limiar próprio (ou ser excluído da guarda), "
        "o que esta tarefa não testou."
    )
    lines.append("")
    lines.append("**Limitações explícitas:**")
    lines.append(
        "- Amostra de ajuste/validação pequena (poucas dezenas de pares no total) — suficiente para "
        "mostrar generalização além dos 3 casos conhecidos, não para estimar uma taxa de erro com precisão "
        "estatística."
    )
    lines.append(
        "- Os negativos minerados além dos 3 conhecidos não passaram por revisão manual individual — "
        "apoiam-se na constatação de nível populacional da Tarefa 9 (ver seção 2), não em confirmação "
        "par-a-par."
    )
    lines.append(
        "- Esta tarefa não produziu nenhuma duplicata real confirmada entre os pares HAM10000×ISIC2018 "
        "Task1/BCN20000 — todo o sinal de 'positivo real' usado para ajustar/validar o limiar vem de pares "
        "controlados (mesma imagem reprocessada), nunca de duas fotos originalmente diferentes da mesma "
        "lesão física. Essa lacuna, já registrada na Tarefa 9, permanece não resolvida."
    )
    lines.append(
        "- Espelhamento/rotação continuam com baixa avaliabilidade geométrica (ver seção 4) — limitação "
        "herdada do ORB, não introduzida nem corrigida por esta guarda."
    )
    lines.append("")
    lines.append("## Aviso")
    lines.append("")
    lines.append(
        "**Isto é piloto/detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi decidida ou "
        "implementada nesta tarefa. Os artefatos das Tarefas 7–9 permanecem intactos e válidos. O índice "
        f"completo desta tarefa está em `{output_index_path.relative_to(REPO_ROOT).as_posix()}`."
    )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"relatório gravado: {REPORT_PATH}")


if __name__ == "__main__":
    main()
