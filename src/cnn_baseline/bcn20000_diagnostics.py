"""Diagnostic-only inspection of BCN20000's official train/test partitions,
ahead of building a lesion-level train/validation split for the CNN
baseline (EfficientNetV2-S). Read-only: never generates a split, never
decides which partition policy to use -- see ``bcn20000_split`` for that.

Reuses ``src.data_audit.core`` primitives (zip reading, missing/conflicting
field detection) and ``src.data_audit.bcn_materialize`` (idempotent inner-
zip materialization) verbatim -- no re-implementation of zip/CSV reading.

HAM10000 is never read or referenced by this module: it is reserved
exclusively for the final external evaluation and must not influence any
development-side decision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data_audit import bcn_materialize, core

# BCN20000's own 8 diagnostic categories (distinct from HAM10000's 7 --
# e.g. BCN has "SCC", HAM10000 does not; whether BCN's "AK" corresponds to
# HAM10000's "akiec" is irrelevant here, since HAM10000 never participates
# in this split).
BCN_DIAGNOSIS_CATEGORIES = ("AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC")


def load_bcn20000_csv(outer_zip: str | Path, csv_entry: str) -> list[dict[str, Any]]:
    """Read one of BCN20000's small CSVs (``bcn_20k_train.csv`` /
    ``bcn_20k_test.csv``) directly from the outer zip, in memory -- same
    read path as ``src.data_audit.content_hash`` uses for the image zips
    (``core.read_zip_entry_text`` + ``core.parse_csv_text``), never
    extracting the CSV to disk.
    """
    text = core.read_zip_entry_text(outer_zip, csv_entry)
    return core.parse_csv_text(text)


def list_zip_image_filenames(inner_zip_path: str | Path) -> set[str]:
    """Basenames (``BCN_XXXXXXXXXX.jpg``, no directory prefix) of every
    ``.jpg`` entry in an already-materialized BCN20000 inner zip
    (``BCN_20k_train.zip``/``BCN_20k_test.zip``), so they compare directly
    against the CSV's ``bcn_filename`` column (which has no path prefix).
    """
    entries = core.zip_entry_manifest(inner_zip_path)
    return {Path(e["name"]).name for e in entries if e["name"].lower().endswith(".jpg")}


def diagnose_partition(
    rows: Sequence[Mapping[str, Any]],
    image_filenames_in_zip: set[str],
    partition_label: str,
) -> dict[str, Any]:
    """Structural diagnostic for one official BCN20000 partition (``train``
    or ``test``). Does not exclude or decide anything -- only measures and
    reports, so the researcher's partition-policy decision (see
    ``bcn20000_split``) is informed by real numbers, not assumptions.
    """
    n_csv_rows = len(rows)

    csv_filenames = {row.get("bcn_filename") for row in rows if row.get("bcn_filename")}
    missing_from_zip = sorted(csv_filenames - image_filenames_in_zip)
    extra_in_zip_not_in_csv = sorted(image_filenames_in_zip - csv_filenames)

    usable_label_rows = [
        row for row in rows if row.get("diagnosis") in BCN_DIAGNOSIS_CATEGORIES
    ]
    empty_label_rows = [row for row in rows if not row.get("diagnosis")]
    unexpected_label_rows = [
        row
        for row in rows
        if row.get("diagnosis") and row.get("diagnosis") not in BCN_DIAGNOSIS_CATEGORIES
    ]

    missing_conflicting = core.find_missing_or_conflicting(rows, "lesion_id", "diagnosis")
    distinct_lesion_ids = {row["lesion_id"] for row in rows if row.get("lesion_id")}

    category_counts_by_row: dict[str, int] = {}
    for row in usable_label_rows:
        category_counts_by_row[row["diagnosis"]] = category_counts_by_row.get(row["diagnosis"], 0) + 1

    lesion_diagnosis: dict[str, str] = {}
    for row in usable_label_rows:
        lid = row.get("lesion_id")
        if lid and lid not in missing_conflicting["conflicting"]:
            lesion_diagnosis[lid] = row["diagnosis"]
    category_counts_by_lesion: dict[str, int] = {}
    for dx in lesion_diagnosis.values():
        category_counts_by_lesion[dx] = category_counts_by_lesion.get(dx, 0) + 1

    ineligible_records = []
    for row in rows:
        reasons = []
        if not row.get("lesion_id"):
            reasons.append("sem lesion_id")
        if not row.get("diagnosis"):
            reasons.append("diagnosis vazio")
        elif row.get("diagnosis") not in BCN_DIAGNOSIS_CATEGORIES:
            reasons.append(f"diagnosis inesperado: {row.get('diagnosis')!r}")
        if row.get("lesion_id") in missing_conflicting["conflicting"]:
            reasons.append("lesion_id com diagnosis conflitante entre suas linhas")
        if row.get("bcn_filename") in missing_from_zip:
            reasons.append("imagem ausente do zip local")
        if reasons:
            ineligible_records.append(
                {
                    "bcn_filename": row.get("bcn_filename"),
                    "lesion_id": row.get("lesion_id"),
                    "reasons": reasons,
                }
            )

    return {
        "partition": partition_label,
        "n_csv_rows": n_csv_rows,
        "n_images_in_zip": len(image_filenames_in_zip),
        "n_missing_from_zip": len(missing_from_zip),
        "missing_from_zip_filenames": missing_from_zip,
        "n_extra_in_zip_not_in_csv": len(extra_in_zip_not_in_csv),
        "n_usable_label_rows": len(usable_label_rows),
        "n_empty_label_rows": len(empty_label_rows),
        "n_unexpected_label_rows": len(unexpected_label_rows),
        "unexpected_label_values": sorted({r["diagnosis"] for r in unexpected_label_rows}),
        "n_missing_lesion_id_rows": len(missing_conflicting["missing_id"]),
        "conflicting_lesion_id_to_diagnosis": missing_conflicting["conflicting"],
        "n_distinct_lesion_ids": len(distinct_lesion_ids),
        "category_counts_by_row": category_counts_by_row,
        "category_counts_by_lesion": category_counts_by_lesion,
        "n_ineligible_records": len(ineligible_records),
        "ineligible_records": ineligible_records,
    }


def compare_lesion_ids_heuristic(
    train_lesion_ids: Sequence[str], test_lesion_ids: Sequence[str]
) -> dict[str, Any]:
    """Attempt to compare ``lesion_id`` between the two official partitions,
    which use different formats (train: ``BCN_0003884``, zero-padded with
    underscore; test: ``BCN4657``, no underscore/padding). A direct string
    comparison would never match anything even if the same lesion appears
    in both, so this applies a normalization heuristic (strip every
    non-digit character, compare the resulting digit strings as integers,
    which removes both the underscore and leading zeros).

    THIS IS A HEURISTIC, NOT A CONFIRMED OVERLAP CHECK. A numeric
    coincidence after normalization does not prove the two ids refer to
    the same lesion -- the two partitions could simply use unrelated
    numbering ranges that happen to collide. Any count returned here must
    be reported as "candidate, unconfirmed", per the project's rule against
    inventing lesion-identifier compatibility across sources without
    evidence.
    """

    def _normalize(lesion_id: str) -> str:
        digits = "".join(ch for ch in lesion_id if ch.isdigit())
        return str(int(digits)) if digits else ""

    train_normalized = {_normalize(lid): lid for lid in train_lesion_ids if _normalize(lid)}
    test_normalized = {_normalize(lid): lid for lid in test_lesion_ids if _normalize(lid)}

    collisions = sorted(set(train_normalized) & set(test_normalized))

    return {
        "method": "heuristic: strip non-digit characters, compare as integer "
        "(removes underscore and zero-padding differences)",
        "is_confirmed_overlap": False,
        "n_train_lesion_ids": len(train_lesion_ids),
        "n_test_lesion_ids": len(test_lesion_ids),
        "n_normalized_collisions": len(collisions),
        "example_collisions": [
            {"normalized": c, "train_lesion_id": train_normalized[c], "test_lesion_id": test_normalized[c]}
            for c in collisions[:20]
        ],
        "warning": "Coincidência numérica após normalização NÃO confirma que seja "
        "a mesma lesão -- os dois formatos podem usar faixas de numeração "
        "não relacionadas. Não usar este resultado para decidir exclusão "
        "nem para afirmar sobreposição confirmada.",
    }


def build_diagnostic_report_markdown(
    *,
    train_diag: Mapping[str, Any],
    test_diag: Mapping[str, Any],
    overlap_heuristic: Mapping[str, Any],
) -> str:
    """Pure string formatting for the ``--mode diagnostico`` output -- no
    diagnostic logic here, same convention as
    ``src.data_audit.sampling.build_sampling_summary_markdown``.
    """
    lines = [
        "# Diagnóstico das Partições Oficiais do BCN20000 (preparação da Tarefa de divisão treino/validação)",
        "",
        "**Isto é só diagnóstico — nenhuma divisão treino/validação foi gerada "
        "nesta execução.** O HAM10000 não foi lido nem referenciado; é "
        "reservado exclusivamente para avaliação externa final.",
        "",
    ]

    for diag, title in ((train_diag, "## Partição `train`"), (test_diag, "## Partição `test`")):
        lines += [
            title,
            "",
            f"- Linhas no CSV: {diag['n_csv_rows']}",
            f"- Imagens encontradas no zip local: {diag['n_images_in_zip']}",
            f"- Imagens do CSV ausentes do zip local: {diag['n_missing_from_zip']}"
            + (f" ({diag['missing_from_zip_filenames'][:5]}{'...' if diag['n_missing_from_zip'] > 5 else ''})"
               if diag["n_missing_from_zip"] else ""),
            f"- Arquivos no zip sem linha correspondente no CSV: {diag['n_extra_in_zip_not_in_csv']}",
            f"- Linhas com rótulo utilizável (`diagnosis` válido): {diag['n_usable_label_rows']}",
            f"- Linhas com `diagnosis` vazio: {diag['n_empty_label_rows']}",
            f"- Linhas com `diagnosis` inesperado: {diag['n_unexpected_label_rows']} "
            f"(valores: {diag['unexpected_label_values']})",
            f"- Linhas sem `lesion_id`: {diag['n_missing_lesion_id_rows']}",
            f"- `lesion_id` com `diagnosis` conflitante entre suas linhas: "
            f"{len(diag['conflicting_lesion_id_to_diagnosis'])}",
            f"- Lesões distintas (com `lesion_id` válido): {diag['n_distinct_lesion_ids']}",
            "",
            "| Categoria | Linhas | Lesões |",
            "|---|---|---|",
        ]
        for cat in BCN_DIAGNOSIS_CATEGORIES:
            lines.append(
                f"| `{cat}` | {diag['category_counts_by_row'].get(cat, 0)} | "
                f"{diag['category_counts_by_lesion'].get(cat, 0)} |"
            )
        lines += [
            "",
            f"- Registros inelegíveis (qualquer motivo): {diag['n_ineligible_records']}",
            "",
        ]

    lines += [
        "## Possível sobreposição de lesão entre `train` e `test` (heurística, NÃO confirmada)",
        "",
        f"- Método: {overlap_heuristic['method']}",
        f"- `lesion_id` distintos em treino: {overlap_heuristic['n_train_lesion_ids']}",
        f"- `lesion_id` distintos em teste: {overlap_heuristic['n_test_lesion_ids']}",
        f"- Colisões após normalização: {overlap_heuristic['n_normalized_collisions']}",
        f"- **{overlap_heuristic['warning']}**",
        "",
        "## Aviso",
        "",
        "Este relatório é só diagnóstico. A recomendação de política de "
        "partições (usar só `train` ou incorporar `test` rotulado) é "
        "apresentada separadamente para decisão do pesquisador — não está "
        "implementada como padrão silencioso em nenhum lugar do código.",
        "",
    ]
    return "\n".join(lines)
