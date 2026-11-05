"""Cross-dataset overlap/duplicate detection (task 4 of the reconstruction;
extended in task 6 with content-hash comparisons for ISIC2018 Task1 and
BCN20000 against HAM10000).

Read-only: reads the JSON manifests already written by
``scripts/run_dataset_inventory.py`` and ``scripts/run_content_hash_audit.py``
under ``experiments/data_audit/`` and compares them for overlap. Does not
read any raw dataset file, does not extract any zip, does not hash anything
itself.

This is DETECTION AND REPORTING ONLY. Finding an overlap here does not
authorize removing any lesion/image from any dataset — that is a separate
researcher decision informed by this report.

Usage:
    python scripts/run_overlap_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import overlap  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiments" / "data_audit"
REPORT_PATH = REPO_ROOT / "docs" / "reconstruction" / "03_dataset_overlap_report.md"
HAM10000_METADATA_CSV = Path(
    "D:/Andrey/Mestrado/dataverse_files/HAM10000_metadata"
)

NOT_VERIFIABLE_PAIRS = [
    {
        "pair": "PH2 x ISIC2018 Task1",
        "reason": (
            "Esquemas de identificador incompatíveis: PH2 usa `IMDxxx`, "
            "ISIC2018 Task1 usa `ISIC_xxxxxxx` — não há campo em comum para "
            "comparação por id. Comparação por conteúdo (hash) não foi feita "
            "nesta tarefa por não ser um dos dois pares prioritários "
            "(ISIC2018 Task1 x HAM10000 e BCN20000 x HAM10000); o manifesto "
            "de conteúdo do ISIC2018 Task1 já existe "
            "(`isic2018_task1_content_manifest.json`, gerado na Tarefa 6 sem "
            "extrair o zip) e pode ser reaproveitado para este par numa "
            "tarefa futura, caso priorizado."
        ),
    },
    {
        "pair": "PH2 x BCN20000",
        "reason": (
            "Esquemas de identificador incompatíveis: PH2 usa `IMDxxx`, "
            "BCN20000 usa `BCN_xxx`/`BCNxxx` — não há campo em comum. "
            "Comparação por conteúdo (hash) não foi feita nesta tarefa por "
            "não ser um dos dois pares prioritários; o manifesto de conteúdo "
            "do BCN20000 já existe (`bcn20000_content_manifest.json`, gerado "
            "na Tarefa 6) e pode ser reaproveitado para este par numa tarefa "
            "futura, caso priorizado."
        ),
    },
    {
        "pair": "ISIC2018 Task1 x BCN20000",
        "reason": (
            "Esquemas de identificador incompatíveis: ISIC2018 Task1 usa "
            "`ISIC_xxxxxxx`, BCN20000 usa `BCN_xxx`/`BCNxxx`. Comparação por "
            "conteúdo (hash) não foi feita nesta tarefa por não ser um dos "
            "dois pares prioritários; os manifestos de conteúdo de ambos já "
            "existem (gerados na Tarefa 6) e podem ser reaproveitados para "
            "este par numa tarefa futura, caso priorizado."
        ),
    },
]

CONTENT_HASH_LIMITATION_NOTE = (
    "**Limitação do SHA-256 nesta comparação**: SHA-256 só detecta arquivos "
    "byte-idênticos. Duas imagens da mesma lesão distribuídas com resolução, "
    "recorte, compressão (ex.: JPEG com qualidade diferente) ou formato de "
    "arquivo diferentes produzem hashes diferentes mesmo sendo clinicamente "
    "'a mesma imagem' — isso não é uma limitação desta implementação, é uma "
    "propriedade do hashing criptográfico, e é a motivação da próxima tarefa "
    "(detecção de candidatos a duplicata visual/perceptual, fora do escopo "
    "desta)."
)


def _load_manifest(name: str) -> dict[str, Any]:
    path = MANIFEST_DIR / f"{name}_manifest.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_image_id_to_lesion_id(metadata_csv: Path, image_ids: set[str]) -> dict[str, str]:
    """Map HAM10000 image_id -> lesion_id, restricted to ``image_ids``."""
    mapping: dict[str, str] = {}
    with open(metadata_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = row.get("image_id")
            if image_id in image_ids:
                mapping[image_id] = row.get("lesion_id", "")
    return mapping


def _run_ph2_vs_ham10000(ph2: dict[str, Any], ham10000: dict[str, Any]) -> dict[str, Any]:
    if ph2.get("status") == "ausente" or ham10000.get("status") == "ausente":
        return {"skipped": True, "reason": "um dos dois manifestos está ausente."}

    ph2_combined = ph2["image_manifest"] + ph2["mask_manifest"]
    ham_combined = ham10000["image_manifest"] + ham10000["mask_manifest"]
    result = overlap.compare_by_hash(ph2_combined, ham_combined)
    result["note"] = (
        "Comparação por conteúdo (sha256), combinando imagens+máscaras de "
        "cada lado num único manifesto por dataset (PH2 tem imagem+máscara "
        "por lesão; HAM10000 idem). Hash idêntico implica arquivo "
        "byte-idêntico, independente de nome/formato de id."
    )
    return result


def _run_isic2018_vs_ham10000(
    isic2018_task1: dict[str, Any], ham10000: dict[str, Any], metadata_csv: Path
) -> dict[str, Any]:
    if isic2018_task1.get("status") == "ausente" or ham10000.get("status") == "ausente":
        return {"skipped": True, "reason": "um dos dois manifestos está ausente."}

    isic_ids = isic2018_task1["pairing_image_vs_mask"]["matched"]
    ham_ids = ham10000["pairing_metadata_vs_images"]["matched"]
    id_overlap = overlap.compare_by_id(isic_ids, ham_ids)

    result: dict[str, Any] = {
        "id_overlap": id_overlap,
        "note": (
            "Comparação por identificador `ISIC_xxxxxxx` entre os ids "
            "image×mask confirmados do ISIC2018 Task1 (2.594) e os ids "
            "metadata×image confirmados do HAM10000 (10.015)."
        ),
    }

    overlapping_ids = id_overlap["matched"]
    if overlapping_ids:
        lesion_map = _load_image_id_to_lesion_id(metadata_csv, set(overlapping_ids))
        result["overlapping_image_ids"] = overlapping_ids
        result["image_id_to_lesion_id"] = lesion_map
        result["lesion_level_note"] = (
            "Mapeamento image_id -> lesion_id (apenas para os ids sobrepostos) "
            "via metadados do HAM10000, porque o protocolo trata sobreposição "
            "em nível de lesão, não só de imagem — a mesma lesão pode ter "
            "múltiplas imagens no HAM10000."
        )
    return result


def _run_content_vs_ham10000(
    content_manifest: dict[str, Any], ham10000: dict[str, Any], note: str
) -> dict[str, Any]:
    if ham10000.get("status") == "ausente":
        return {"skipped": True, "reason": "o manifesto do HAM10000 está ausente."}

    ham_combined = ham10000["image_manifest"] + ham10000["mask_manifest"]
    result = overlap.compare_by_hash(content_manifest, ham_combined)
    result["note"] = note
    return result


def main() -> None:
    manifests = {name: _load_manifest(name) for name in ("ph2", "ham10000", "isic2018_task1", "bcn20000")}
    isic_content = _load_manifest("isic2018_task1_content")
    bcn_content = _load_manifest("bcn20000_content")

    print("=== PH2 x HAM10000 (por conteúdo / sha256) ===")
    ph2_vs_ham = _run_ph2_vs_ham10000(manifests["ph2"], manifests["ham10000"])
    print(f"duplicate_count={ph2_vs_ham.get('duplicate_count', 'n/a')}")

    print("=== ISIC2018 Task1 x HAM10000 (por identificador) ===")
    isic_vs_ham = _run_isic2018_vs_ham10000(
        manifests["isic2018_task1"], manifests["ham10000"], HAM10000_METADATA_CSV
    )
    print(f"matched_ids={len(isic_vs_ham.get('id_overlap', {}).get('matched', []))}")

    print("=== ISIC2018 Task1 x HAM10000 (por conteúdo / sha256) ===")
    isic_content_vs_ham = _run_content_vs_ham10000(
        isic_content["manifest"],
        manifests["ham10000"],
        note=(
            "Comparação por conteúdo (sha256) entre as imagens do ISIC2018 "
            "Task1 hasheadas diretamente do zip (sem extração) e o "
            "manifesto imagem+máscara do HAM10000. Esta é uma checagem "
            "ADICIONAL à comparação por identificador ISIC_xxxxxxx já "
            "existente acima — id e conteúdo são checagens diferentes: ids "
            "iguais não implicam bytes iguais, e vice-versa."
        ),
    )
    print(f"duplicate_count={isic_content_vs_ham.get('duplicate_count', 'n/a')}")

    print("=== BCN20000 x HAM10000 (por conteúdo / sha256) ===")
    bcn_combined_content = bcn_content["train"]["manifest"] + bcn_content["test"]["manifest"]
    bcn_vs_ham = _run_content_vs_ham10000(
        bcn_combined_content,
        manifests["ham10000"],
        note=(
            "Comparação por conteúdo (sha256) entre as imagens do BCN20000 "
            "(splits treino+teste, materializados em D: e hasheados "
            "diretamente dos zips internos, sem extrair imagens "
            "individuais) e o manifesto imagem+máscara do HAM10000."
        ),
    )
    print(f"duplicate_count={bcn_vs_ham.get('duplicate_count', 'n/a')}")

    print("=== Pares não verificáveis nesta tarefa ===")
    for item in NOT_VERIFIABLE_PAIRS:
        print(f"- {item['pair']}: não verificável")

    report_text = _render_report(ph2_vs_ham, isic_vs_ham, isic_content_vs_ham, bcn_vs_ham)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"relatório gravado: {REPORT_PATH}")


def _render_content_hash_section(title: str, result: dict[str, Any], label_a: str, label_b: str) -> list[str]:
    lines = [f"## {title}", ""]
    if result.get("skipped"):
        lines.append(f"**Não executado**: {result['reason']}")
        lines.append("")
        return lines

    lines.append(result["note"])
    lines.append("")
    count = result["duplicate_count"]
    if count == 0:
        lines.append(
            f"**Resultado: nenhum arquivo byte-idêntico encontrado entre {label_a} e {label_b}.**"
        )
    else:
        lines.append(
            f"**Resultado: {count} hash(es) sha256 idêntico(s) encontrado(s) "
            "entre os dois datasets — candidato a sobreposição confirmada, "
            "requer decisão do pesquisador/planejamento sobre exclusão.**"
        )
        lines.append("")
        for sha256, sides in result["duplicates"].items():
            lines.append(f"- `{sha256}`:")
            lines.append(f"  - {label_a}: {sides['paths_a']}")
            lines.append(f"  - {label_b}: {sides['paths_b']}")
    lines.append("")
    return lines


def _render_report(
    ph2_vs_ham: dict[str, Any],
    isic_vs_ham: dict[str, Any],
    isic_content_vs_ham: dict[str, Any],
    bcn_vs_ham: dict[str, Any],
) -> str:
    from datetime import date

    lines = [
        f"# Detecção de Sobreposição/Duplicata entre Datasets — {date.today().isoformat()}",
        "",
        "Primeira verificação de sobreposição entre datasets exigida pelo "
        "protocolo experimental (\"Arquivos idênticos por SHA-256\", "
        "\"Identificadores das imagens... correspondências entre bases\"). "
        "**Isto é detecção e relato, não decisão de exclusão.** Encontrar uma "
        "sobreposição aqui não autoriza remover nada automaticamente; qualquer "
        "achado abaixo é um candidato que precisa de decisão do pesquisador.",
        "",
        "## PH2 x HAM10000 (comparação por conteúdo, sha256)",
        "",
    ]

    if ph2_vs_ham.get("skipped"):
        lines.append(f"**Não executado**: {ph2_vs_ham['reason']}")
        lines.append("")
    else:
        lines.append(ph2_vs_ham["note"])
        lines.append("")
        count = ph2_vs_ham["duplicate_count"]
        if count == 0:
            lines.append(
                "**Resultado: nenhum arquivo byte-idêntico encontrado entre PH2 "
                "(imagens+máscaras) e HAM10000 (imagens+máscaras).**"
            )
        else:
            lines.append(
                f"**Resultado: {count} hash(es) sha256 idêntico(s) encontrado(s) "
                "entre os dois datasets — candidato a sobreposição confirmada, "
                "requer decisão do pesquisador/planejamento sobre exclusão.**"
            )
            lines.append("")
            for sha256, sides in ph2_vs_ham["duplicates"].items():
                lines.append(f"- `{sha256}`:")
                lines.append(f"  - PH2: {sides['paths_a']}")
                lines.append(f"  - HAM10000: {sides['paths_b']}")
        lines.append("")

    lines.append("## ISIC2018 Task1 x HAM10000 (comparação por identificador `ISIC_xxxxxxx`)")
    lines.append("")
    if isic_vs_ham.get("skipped"):
        lines.append(f"**Não executado**: {isic_vs_ham['reason']}")
        lines.append("")
    else:
        lines.append(isic_vs_ham["note"])
        lines.append("")
        matched = isic_vs_ham["id_overlap"]["matched"]
        if not matched:
            lines.append(
                "**Resultado: nenhum id `ISIC_xxxxxxx` em comum entre o "
                "ISIC2018 Task1 (pareamento imagem×máscara) e o HAM10000 "
                "(pareamento metadado×imagem).**"
            )
        else:
            lines.append(
                f"**Resultado: {len(matched)} id(s) em comum encontrado(s) — "
                "candidato a sobreposição confirmada, requer decisão do "
                "pesquisador/planejamento sobre exclusão.**"
            )
            lines.append("")
            lines.append(f"- ids sobrepostos: {matched}")
            if "image_id_to_lesion_id" in isic_vs_ham:
                lines.append(f"- {isic_vs_ham['lesion_level_note']}")
                lines.append(f"- mapeamento image_id -> lesion_id: {isic_vs_ham['image_id_to_lesion_id']}")
        lines.append("")

    lines += _render_content_hash_section(
        "ISIC2018 Task1 x HAM10000 (comparação por conteúdo, sha256)",
        isic_content_vs_ham,
        label_a="ISIC2018 Task1",
        label_b="HAM10000",
    )
    lines += _render_content_hash_section(
        "BCN20000 x HAM10000 (comparação por conteúdo, sha256)",
        bcn_vs_ham,
        label_a="BCN20000",
        label_b="HAM10000",
    )
    lines.append(CONTENT_HASH_LIMITATION_NOTE)
    lines.append("")

    lines.append("## Pares não verificáveis nesta tarefa")
    lines.append("")
    for item in NOT_VERIFIABLE_PAIRS:
        lines.append(f"### {item['pair']}")
        lines.append("")
        lines.append(item["reason"])
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


if __name__ == "__main__":
    main()
