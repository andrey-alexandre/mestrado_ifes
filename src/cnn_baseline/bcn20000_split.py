"""Lesion-level train/validation split of BCN20000 for CNN baseline
development (EfficientNetV2-S, ImageNet init, binary melanoma vs.
non-melanoma). Threshold selection (Youden) and training happen in later
tasks -- this module only builds the split manifest.

HAM10000 is never read, imported, or referenced here: it is reserved
exclusively for the final external evaluation.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from src.data_audit import core
from src.cnn_baseline.bcn20000_diagnostics import BCN_DIAGNOSIS_CATEGORIES

PARTITION_POLICIES = ("apenas_train", "train_mais_test_rotulado")


def binary_target_for_diagnosis(diagnosis: str) -> str:
    """melanoma iff diagnosis == 'MEL' (BCN20000's own code for melanoma);
    everything else in BCN_DIAGNOSIS_CATEGORIES is 'nao_melanoma'. Never a
    synonym for benign/malignant -- same convention as
    src.data_audit.sampling for HAM10000's 'mel'.
    """
    return "melanoma" if diagnosis == "MEL" else "nao_melanoma"


def select_partition_rows(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    """Apply the researcher-chosen partition policy, tagging each row with
    its official source partition (never silently merged without a trace).

    'apenas_train': only the official train rows.
    'train_mais_test_rotulado': train rows plus any test rows that DO have
    a usable label (diagnosis in BCN_DIAGNOSIS_CATEGORIES) -- written
    generically (filtering by label presence, not by a hardcoded "test has
    no labels" assumption), since a future data release could add labels.
    """
    if policy not in PARTITION_POLICIES:
        raise ValueError(f"partition_policy desconhecida: {policy!r} (válidas: {PARTITION_POLICIES})")

    selected = [{**row, "source_partition": "train"} for row in train_rows]
    if policy == "train_mais_test_rotulado":
        selected += [
            {**row, "source_partition": "test"}
            for row in test_rows
            if row.get("diagnosis") in BCN_DIAGNOSIS_CATEGORIES
        ]
    return selected


def build_eligible_lesion_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse the selected rows into one entry per distinct, unambiguous
    lesion_id: its diagnosis, binary_target, and the (bcn_filename,
    source_partition) pairs of all its images.

    Rows with missing lesion_id, empty/unexpected diagnosis, or a
    lesion_id whose diagnosis conflicts across rows are EXCLUDED here and
    reported separately -- never silently included with a guessed label.
    Reuses core.find_missing_or_conflicting verbatim.
    """
    labeled_rows = [r for r in rows if r.get("diagnosis") in BCN_DIAGNOSIS_CATEGORIES]
    missing_conflicting = core.find_missing_or_conflicting(labeled_rows, "lesion_id", "diagnosis")
    conflicting_ids = set(missing_conflicting["conflicting"])

    lesion_table: dict[str, dict[str, Any]] = {}
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        lid = row.get("lesion_id")
        diagnosis = row.get("diagnosis")
        reasons = []
        if not lid:
            reasons.append("sem lesion_id")
        if diagnosis not in BCN_DIAGNOSIS_CATEGORIES:
            reasons.append(f"diagnosis não utilizável: {diagnosis!r}")
        if lid in conflicting_ids:
            reasons.append("lesion_id com diagnosis conflitante entre suas linhas")
        if reasons:
            excluded_rows.append(
                {"bcn_filename": row.get("bcn_filename"), "lesion_id": lid, "reasons": reasons}
            )
            continue
        entry = lesion_table.setdefault(
            lid, {"diagnosis": diagnosis, "binary_target": binary_target_for_diagnosis(diagnosis), "images": []}
        )
        entry["images"].append(
            {"bcn_filename": row["bcn_filename"], "source_partition": row["source_partition"]}
        )

    return {"lesion_table": lesion_table, "excluded_rows": excluded_rows}


def split_lesions_stratified(
    lesion_table: Mapping[str, dict[str, Any]], val_fraction: float, seed: int
) -> dict[str, Any]:
    """Group-aware, class-stratified split: within each binary_target class
    independently, shuffle the lesion_ids deterministically (seeded) and
    take round(val_fraction * n_lesions_in_class) for validation, the rest
    for train.

    Why this algorithm, not a generic greedy-by-size packer: splitting
    separately per class is the standard way to keep a per-class
    proportion close to the original in both resulting partitions when
    groups (lesions) vary in size -- it is not claimed to be optimal, only
    deterministic, documented, and testable. Because group sizes (images
    per lesion) vary, the ACHIEVED image-level validation fraction will
    generally differ slightly from val_fraction; both the lesion-level and
    image-level achieved fractions are reported by the caller (see
    scripts/run_bcn20000_split.py) so that deviation is visible, not
    hidden.

    Processes classes in sorted name order and draws from a single
    seeded RNG across both classes (not a fresh RNG per class) so the
    result is sensitive to this documented order, not to arbitrary dict
    iteration order -- same determinism discipline as
    src.data_audit.sampling.sample_lesions.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction deve estar em (0, 1), recebido {val_fraction}")

    by_class: dict[str, list[str]] = {}
    for lid, entry in lesion_table.items():
        by_class.setdefault(entry["binary_target"], []).append(lid)

    rng = random.Random(seed)
    train_lesion_ids: list[str] = []
    val_lesion_ids: list[str] = []
    per_class_detail: dict[str, Any] = {}

    for cls in sorted(by_class.keys()):
        ids = sorted(by_class[cls])
        rng.shuffle(ids)
        n_val = round(val_fraction * len(ids))
        val_ids_cls = sorted(ids[:n_val])
        train_ids_cls = sorted(ids[n_val:])
        val_lesion_ids.extend(val_ids_cls)
        train_lesion_ids.extend(train_ids_cls)
        per_class_detail[cls] = {
            "n_lesions": len(ids),
            "n_val": len(val_ids_cls),
            "n_train": len(train_ids_cls),
        }

    return {
        "seed": seed,
        "val_fraction_requested": val_fraction,
        "train_lesion_ids": sorted(train_lesion_ids),
        "val_lesion_ids": sorted(val_lesion_ids),
        "per_class_detail": per_class_detail,
    }


def build_split_manifest_rows(
    lesion_table: Mapping[str, dict[str, Any]],
    train_lesion_ids: Sequence[str],
    val_lesion_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per IMAGE (all images of a lesion go to the same split --
    unlike src.data_audit.sampling, which draws exactly one image per
    lesion for the HAM10000 experimental sample; here, for CNN training,
    every available image of a lesion is kept to maximize training data).
    """
    split_by_lesion = {lid: "train" for lid in train_lesion_ids}
    split_by_lesion.update({lid: "val" for lid in val_lesion_ids})

    rows: list[dict[str, Any]] = []
    for lid, split in split_by_lesion.items():
        entry = lesion_table[lid]
        for image in entry["images"]:
            rows.append(
                {
                    "lesion_id": lid,
                    "bcn_filename": image["bcn_filename"],
                    "source_partition": image["source_partition"],
                    "diagnosis": entry["diagnosis"],
                    "binary_target": entry["binary_target"],
                    "split": split,
                }
            )
    return rows


def summarize_split(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts by image and by lesion, class proportions, per split -- the
    numbers the acceptance criteria require the report to show.
    """
    summary: dict[str, Any] = {}
    for split in ("train", "val"):
        split_rows = [r for r in manifest_rows if r["split"] == split]
        lesion_ids = {r["lesion_id"] for r in split_rows}
        n_images = len(split_rows)
        n_melanoma_images = sum(1 for r in split_rows if r["binary_target"] == "melanoma")
        lesion_targets = {r["lesion_id"]: r["binary_target"] for r in split_rows}
        n_melanoma_lesions = sum(1 for t in lesion_targets.values() if t == "melanoma")
        summary[split] = {
            "n_images": n_images,
            "n_lesions": len(lesion_ids),
            "n_melanoma_images": n_melanoma_images,
            "n_nao_melanoma_images": n_images - n_melanoma_images,
            "melanoma_image_fraction": n_melanoma_images / n_images if n_images else None,
            "n_melanoma_lesions": n_melanoma_lesions,
            "n_nao_melanoma_lesions": len(lesion_ids) - n_melanoma_lesions,
            "melanoma_lesion_fraction": n_melanoma_lesions / len(lesion_ids) if lesion_ids else None,
        }
    total_images = summary["train"]["n_images"] + summary["val"]["n_images"]
    total_lesions = summary["train"]["n_lesions"] + summary["val"]["n_lesions"]
    summary["achieved_val_image_fraction"] = (
        summary["val"]["n_images"] / total_images if total_images else None
    )
    summary["achieved_val_lesion_fraction"] = (
        summary["val"]["n_lesions"] / total_lesions if total_lesions else None
    )
    return summary


def verify_no_lesion_overlap(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Explicit check (not an assumption) that no lesion_id appears in both
    'train' and 'val' splits -- the first acceptance criterion.
    """
    train_ids = {r["lesion_id"] for r in manifest_rows if r["split"] == "train"}
    val_ids = {r["lesion_id"] for r in manifest_rows if r["split"] == "val"}
    overlap = sorted(train_ids & val_ids)
    return {"ok": not overlap, "overlapping_lesion_ids": overlap}


def build_split_report_markdown(
    *,
    partition_policy: str,
    val_fraction: float,
    seed: int,
    excluded_rows_count: int,
    split_result: Mapping[str, Any],
    summary: Mapping[str, Any],
    overlap_check: Mapping[str, Any],
) -> str:
    """Pure string formatting for the ``--mode geracao`` output."""
    lines = [
        "# Divisão Treino/Validação do BCN20000 — CNN Baseline (preparação de dados)",
        "",
        "**Isto é preparação de dados, não treino.** Nenhum modelo foi "
        "treinado, nenhum limiar foi selecionado, nenhuma avaliação no "
        "HAM10000 foi feita. O HAM10000 não foi lido nem referenciado nesta "
        "execução.",
        "",
        f"- Política de partição (`--partition-policy`): `{partition_policy}`",
        f"- Proporção de validação solicitada (`--val-fraction`): {val_fraction}",
        f"- Semente (`--seed`): {seed}",
        f"- Registros excluídos antes da divisão (sem lesion_id, diagnosis "
        f"inutilizável, ou lesion_id conflitante): {excluded_rows_count}",
        "",
        "## Verificação de não sobreposição de lesão",
        "",
        f"- Nenhum `lesion_id` em treino E validação simultaneamente: {overlap_check['ok']}",
    ]
    if not overlap_check["ok"]:
        lines.append(f"- **FALHA**: lesões sobrepostas: {overlap_check['overlapping_lesion_ids']}")
    lines += ["", "## Composição por split", "", "| Split | Imagens | Lesões | Melanoma (img) | Não-melanoma (img) | Proporção melanoma (img) | Melanoma (lesão) | Proporção melanoma (lesão) |",
              "|---|---|---|---|---|---|---|---|"]
    for split in ("train", "val"):
        s = summary[split]
        lines.append(
            f"| `{split}` | {s['n_images']} | {s['n_lesions']} | {s['n_melanoma_images']} | "
            f"{s['n_nao_melanoma_images']} | {s['melanoma_image_fraction']:.4f} | "
            f"{s['n_melanoma_lesions']} | {s['melanoma_lesion_fraction']:.4f} |"
        )
    lines += [
        "",
        f"- Fração de validação alcançada (por imagem): {summary['achieved_val_image_fraction']:.4f} "
        f"(solicitada: {val_fraction}, desvio: "
        f"{abs(summary['achieved_val_image_fraction'] - val_fraction):.4f})",
        f"- Fração de validação alcançada (por lesão): {summary['achieved_val_lesion_fraction']:.4f} "
        f"(solicitada: {val_fraction}, desvio: "
        f"{abs(summary['achieved_val_lesion_fraction'] - val_fraction):.4f})",
        "",
        "## Detalhe por classe (ajuste estratificado)",
        "",
        "| Classe | Lesões | Val | Train |",
        "|---|---|---|---|",
    ]
    for cls, detail in split_result["per_class_detail"].items():
        lines.append(f"| `{cls}` | {detail['n_lesions']} | {detail['n_val']} | {detail['n_train']} |")
    lines += [
        "",
        "## Aviso",
        "",
        "Este manifesto é preparação de dados para a etapa 3 (baseline CNN). "
        "Seleção de limiar (índice de Youden) acontece depois, exclusivamente "
        "na validação interna aqui gerada. O manifesto da amostra do HAM10000 "
        "permanece intacto e reservado à avaliação externa final.",
        "",
    ]
    return "\n".join(lines)
