"""Dataset-specific structural audits built on top of :mod:`src.data_audit.core`.

Every function here returns a JSON-serializable dict and does not: sample
data, extract large zips, call any LLM/VLM, or touch any path other than the
one(s) explicitly passed in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import core


def audit_ph2(root_dir: str | Path, metadata_txt: str | Path) -> dict[str, Any]:
    """PH2: pair image/lesion-mask/ROI per lesion id, and read the two
    diagnosis criteria from ``PH2_dataset.txt`` side by side.
    """
    root_check = core.check_path_exists(root_dir)
    meta_check = core.check_path_exists(metadata_txt)
    if not root_check["exists"] or not meta_check["exists"]:
        return {"dataset": "ph2", "status": "ausente", "root": root_check, "metadata": meta_check}

    root = Path(root_dir)
    lesion_dirs = sorted(p.name for p in root.iterdir() if p.is_dir())

    image_ids, mask_ids, roi_ids = [], [], []
    image_paths, mask_paths = [], []
    for lesion_id in lesion_dirs:
        lesion_path = root / lesion_id
        image_file = lesion_path / f"{lesion_id}_Dermoscopic_Image" / f"{lesion_id}.bmp"
        if image_file.is_file():
            image_ids.append(lesion_id)
            image_paths.append(image_file)
        mask_file = lesion_path / f"{lesion_id}_lesion" / f"{lesion_id}_lesion.bmp"
        if mask_file.is_file():
            mask_ids.append(lesion_id)
            mask_paths.append(mask_file)
        roi_dir = lesion_path / f"{lesion_id}_roi"
        if roi_dir.is_dir() and any(roi_dir.glob("*.bmp")):
            roi_ids.append(lesion_id)

    rows = _parse_ph2_metadata_txt(metadata_txt)
    metadata_ids = [r["Name"] for r in rows]

    hist_positive = [r["Name"] for r in rows if r["Histological Diagnosis"].strip() != ""]
    clinical_positive = [r["Name"] for r in rows if r["Clinical Diagnosis"].strip() == "2"]
    melanoma_text_positive = [
        r["Name"] for r in rows if "melanoma" in r["Histological Diagnosis"].strip().lower()
    ]
    hist_value_counts = core.category_counts(
        [r for r in rows if r["Histological Diagnosis"].strip() != ""],
        "Histological Diagnosis",
    )

    return {
        "dataset": "ph2",
        "status": "completo",
        "lesion_dirs_found": len(lesion_dirs),
        "metadata_rows": len(rows),
        "pairing_image_vs_metadata": core.pair_by_id(image_ids, metadata_ids),
        "pairing_mask_vs_metadata": core.pair_by_id(mask_ids, metadata_ids),
        "pairing_roi_vs_metadata": core.pair_by_id(roi_ids, metadata_ids),
        "pairing_image_vs_mask": core.pair_by_id(image_ids, mask_ids),
        "image_manifest": core.build_file_manifest(image_paths, base_dir=root),
        "mask_manifest": core.build_file_manifest(mask_paths, base_dir=root),
        "diagnosis_criteria_side_by_side": {
            "histological_diagnosis_non_empty": {
                "count": len(hist_positive),
                "ids": hist_positive,
            },
            "histological_diagnosis_melanoma_text": {
                "count": len(melanoma_text_positive),
                "ids": melanoma_text_positive,
            },
            "histological_diagnosis_values": {
                str(k): v for k, v in sorted(hist_value_counts.items())
            },
            "clinical_diagnosis_equals_2": {
                "count": len(clinical_positive),
                "ids": clinical_positive,
            },
            "note": (
                "Três leituras do critério histológico são reportadas lado a lado: "
                "'não vazio' (qualquer biópsia registrada, inclui lesões não "
                "melanoma), 'melanoma confirmado por texto' (subconjunto do "
                "anterior cujo valor contém a palavra 'melanoma') e "
                "'Clinical Diagnosis == 2'. A comparação metodologicamente "
                "relevante para avaliar o critério clínico é melanoma-por-texto "
                "× Clinical Diagnosis == 2, não não-vazio × Clinical Diagnosis == 2."
            ),
        },
    }


def _parse_ph2_metadata_txt(metadata_txt: str | Path) -> list[dict[str, str]]:
    """Parse the pipe-table rows of PH2_dataset.txt (lines starting with '||').

    Format per data line:
      || Name || Histological Diagnosis || Clinical Diagnosis || A | PN | DG | S | RA | BWV || Colors ||
    A trailing legend section (not starting with '||') follows the table and is ignored.
    """
    with open(metadata_txt, encoding="utf-8") as f:
        table_lines = [line.rstrip("\r\n") for line in f if line.startswith("||")]

    if not table_lines:
        return []

    data_lines = table_lines[1:]  # skip header
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split("||")]
        # parts: ['', Name, Histological Diagnosis, Clinical Diagnosis, "A | PN | DG | S | RA | BWV", Colors, '']
        if len(parts) < 6:
            continue
        rows.append(
            {
                "Name": parts[1],
                "Histological Diagnosis": parts[2],
                "Clinical Diagnosis": parts[3],
            }
        )
    return rows


def audit_ham10000(
    images_dir: str | Path,
    masks_dir: str | Path,
    metadata_csv: str | Path,
    hash_files: bool = True,
    previous_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """HAM10000: 3-way pairing (metadata x images x masks), dx distribution,
    lesion_id -> dx conflicts, and a sha256 manifest of every image/mask.

    When ``hash_files`` is False, no hashing is done in this call. If
    ``previous_manifest`` carries non-empty ``image_manifest``/``mask_manifest``
    from an earlier run, those are reused as-is (``hash_status`` =
    ``"reused_from_previous"``); otherwise the manifests come back empty
    (``hash_status`` = ``"skipped_no_previous"``). This avoids a caller
    silently discarding previously computed hashes just because this run
    chose to skip hashing.
    """
    images_check = core.check_path_exists(images_dir)
    masks_check = core.check_path_exists(masks_dir)
    meta_check = core.check_path_exists(metadata_csv)
    if not (images_check["exists"] and masks_check["exists"] and meta_check["exists"]):
        return {
            "dataset": "ham10000",
            "status": "ausente",
            "images": images_check,
            "masks": masks_check,
            "metadata": meta_check,
        }

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(images_dir.glob("*.jpg"))
    mask_files = sorted(masks_dir.glob("*.png"))

    image_ids = [p.stem for p in image_files]
    mask_ids = [p.stem.removesuffix("_segmentation") for p in mask_files]

    with open(metadata_csv, encoding="utf-8") as f:
        rows = core.parse_csv_text(f.read())
    metadata_image_ids = [r["image_id"] for r in rows]

    lesion_dx_check = core.find_missing_or_conflicting(rows, "lesion_id", "dx")
    dx_counts = core.category_counts(rows, "dx")
    distinct_lesions = sorted({r["lesion_id"] for r in rows if r.get("lesion_id")})

    images_per_lesion = core.category_counts(rows, "lesion_id")

    result: dict[str, Any] = {
        "dataset": "ham10000",
        "status": "completo",
        "metadata_rows": len(rows),
        "image_files_found": len(image_files),
        "mask_files_found": len(mask_files),
        "distinct_lesions": len(distinct_lesions),
        "pairing_metadata_vs_images": core.pair_by_id(metadata_image_ids, image_ids),
        "pairing_metadata_vs_masks": core.pair_by_id(metadata_image_ids, mask_ids),
        "pairing_images_vs_masks": core.pair_by_id(image_ids, mask_ids),
        "lesion_id_to_dx_missing_or_conflicting": lesion_dx_check,
        "dx_counts": dict(dx_counts),
        "images_per_lesion_distribution": _count_histogram(images_per_lesion),
    }

    if hash_files:
        result["image_manifest"] = core.build_file_manifest(image_files, base_dir=images_dir)
        result["mask_manifest"] = core.build_file_manifest(mask_files, base_dir=masks_dir)
        result["hash_status"] = "hashed"
    elif (
        previous_manifest
        and previous_manifest.get("image_manifest")
        and previous_manifest.get("mask_manifest")
    ):
        result["image_manifest"] = previous_manifest["image_manifest"]
        result["mask_manifest"] = previous_manifest["mask_manifest"]
        result["hash_status"] = "reused_from_previous"
    else:
        result["image_manifest"] = []
        result["mask_manifest"] = []
        result["hash_status"] = "skipped_no_previous"

    return result


def _count_histogram(counter) -> dict[str, int]:
    """Collapse a {lesion_id: n_images} Counter into {n_images: n_lesions_with_that_count}."""
    histogram: dict[int, int] = {}
    for n_images in counter.values():
        histogram[n_images] = histogram.get(n_images, 0) + 1
    return {str(k): v for k, v in sorted(histogram.items())}


def audit_isic2018_task1(
    images_zip: str | Path, masks_zip: str | Path
) -> dict[str, Any]:
    """ISIC2018 Task1: list both the training-input zip's and the
    ground-truth masks zip's central directories (no extraction of
    either) and pair image ids against mask ids.
    """
    images_zip_check = core.check_path_exists(images_zip)
    masks_zip_check = core.check_path_exists(masks_zip)

    if not images_zip_check["exists"] or not masks_zip_check["exists"]:
        return {
            "dataset": "isic2018_task1",
            "status": "ausente",
            "images_zip": images_zip_check,
            "masks_zip": masks_zip_check,
        }

    image_all_entries = core.zip_entry_manifest(images_zip)
    image_entries = [e for e in image_all_entries if e["name"].lower().endswith(".jpg")]
    other_image_entries = [e for e in image_all_entries if not e["name"].lower().endswith(".jpg")]

    mask_all_entries = core.zip_entry_manifest(masks_zip)
    mask_entries = [e for e in mask_all_entries if e["name"].lower().endswith(".png")]
    other_mask_entries = [e for e in mask_all_entries if not e["name"].lower().endswith(".png")]

    image_ids = [Path(e["name"]).stem for e in image_entries]
    mask_ids = [Path(e["name"]).stem.removesuffix("_segmentation") for e in mask_entries]

    pairing = core.pair_by_id(image_ids, mask_ids)
    perfect_pairing = not pairing["only_in_a"] and not pairing["only_in_b"]
    status = "completo" if perfect_pairing else "parcial"

    return {
        "dataset": "isic2018_task1",
        "status": status,
        "images_zip": images_zip_check,
        "images_zip_entry_count_total": len(image_all_entries),
        "images_zip_image_entries": len(image_entries),
        "images_zip_other_entries": other_image_entries,
        "masks_zip": masks_zip_check,
        "masks_zip_entry_count_total": len(mask_all_entries),
        "masks_zip_mask_entries": len(mask_entries),
        "masks_zip_other_entries": other_mask_entries,
        "pairing_image_vs_mask": pairing,
        "note": (
            f"Imagens e máscaras presentes; pareamento perfeito "
            f"({len(pairing['matched'])} ids em ambos os conjuntos, nenhum só de um lado)."
            if perfect_pairing
            else "Imagens e máscaras presentes, mas o pareamento por id não é perfeito "
            f"({len(pairing['only_in_a'])} id(s) só nas imagens, "
            f"{len(pairing['only_in_b'])} só nas máscaras)."
        ),
    }


def audit_bcn20000(
    outer_zip: str | Path,
    train_csv_entry: str,
    test_csv_entry: str,
    train_zip_entry: str,
    test_zip_entry: str,
) -> dict[str, Any]:
    """BCN20000: list the 4 entries of the outer zip, read the two small CSVs
    in memory (train/test), and audit lesion_id -> diagnosis within train.

    The two inner image zips (train/test) are never opened beyond being
    listed by name/size in the outer zip's central directory.
    """
    zip_check = core.check_path_exists(outer_zip)
    if not zip_check["exists"]:
        return {"dataset": "bcn20000", "status": "ausente", "outer_zip": zip_check}

    entries = core.zip_entry_manifest(outer_zip)

    train_text = core.read_zip_entry_text(outer_zip, train_csv_entry)
    test_text = core.read_zip_entry_text(outer_zip, test_csv_entry)
    train_rows = core.parse_csv_text(train_text)
    test_rows = core.parse_csv_text(test_text)

    train_lesion_ids = sorted({r["lesion_id"] for r in train_rows if r.get("lesion_id")})
    test_lesion_ids = sorted({r["lesion_id"] for r in test_rows if r.get("lesion_id")})

    train_diag_check = core.find_missing_or_conflicting(train_rows, "lesion_id", "diagnosis")
    train_diag_counts = core.category_counts(train_rows, "diagnosis")

    test_diagnosis_filled = sum(1 for r in test_rows if r.get("diagnosis", "").strip())

    return {
        "dataset": "bcn20000",
        "status": "parcial",
        "outer_zip_entries": entries,
        "train_csv_rows": len(train_rows),
        "test_csv_rows": len(test_rows),
        "train_distinct_lesions": len(train_lesion_ids),
        "test_distinct_lesions": len(test_lesion_ids),
        "train_lesion_id_to_diagnosis_missing_or_conflicting": train_diag_check,
        "train_diagnosis_counts": dict(train_diag_counts),
        "test_diagnosis_non_empty_count": test_diagnosis_filled,
        "notes": {
            "test_diagnosis_empty": (
                f"diagnosis vazio em {len(test_rows) - test_diagnosis_filled}/"
                f"{len(test_rows)} linhas de teste (rótulo retido na versão pública); "
                "não é dado ausente por erro, é retenção intencional."
            ),
            "lesion_id_format_divergence": (
                "Formato de lesion_id difere entre treino (ex.: 'BCN_0003884', "
                "com underscore e zero-padding) e teste (ex.: 'BCN4657', sem "
                "underscore/padding) — mesma fonte, formatos diferentes. "
                "Apenas documentado; não normalizado nem presumido equivalente "
                "a identificadores de outros datasets."
            ),
            "images_not_inventoried": (
                f"Imagens aninhadas em '{train_zip_entry}'/'{test_zip_entry}' "
                "(dentro do zip externo) não foram listadas nem hasheadas nesta "
                "tarefa — inventário em nível de arquivo de imagem fica para "
                "quando o treino da CNN for implementado."
            ),
        },
    }
