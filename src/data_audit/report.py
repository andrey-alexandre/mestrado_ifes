"""Render the consolidated Markdown inventory report from the 4 audit summaries."""
from __future__ import annotations

from datetime import date
from typing import Any


def _pairing_line(label: str, pairing: dict[str, list[str]]) -> str:
    return (
        f"- **{label}**: {len(pairing['matched'])} pareados, "
        f"{len(pairing['only_in_a'])} só no primeiro conjunto, "
        f"{len(pairing['only_in_b'])} só no segundo."
    )


def _render_ph2(summary: dict[str, Any]) -> str:
    if summary["status"] == "ausente":
        return "## PH2\n\n**Status: ausente** — caminho de dados ou de metadados não encontrado.\n"

    criteria = summary["diagnosis_criteria_side_by_side"]
    non_empty = criteria["histological_diagnosis_non_empty"]
    melanoma_text = criteria["histological_diagnosis_melanoma_text"]
    clinical = criteria["clinical_diagnosis_equals_2"]
    value_counts: dict[str, int] = criteria.get("histological_diagnosis_values", {})

    lines = [
        "## PH2",
        "",
        f"**Status: {summary['status']}**",
        "",
        f"- Diretórios de lesão encontrados: {summary['lesion_dirs_found']}",
        f"- Linhas de metadados (`PH2_dataset.txt`): {summary['metadata_rows']}",
        "",
        "### Pareamento",
        "",
        _pairing_line("Imagem × metadados", summary["pairing_image_vs_metadata"]),
        _pairing_line("Máscara × metadados", summary["pairing_mask_vs_metadata"]),
        _pairing_line("ROI × metadados", summary["pairing_roi_vs_metadata"]),
        _pairing_line("Imagem × máscara", summary["pairing_image_vs_mask"]),
        "",
        "### Critérios de diagnóstico (lado a lado, sem decidir qual é a correta)",
        "",
        f"- `Histological Diagnosis` não vazio: {non_empty['count']} lesões.",
        f"- `Histological Diagnosis` contendo \"melanoma\" (melanoma confirmado por "
        f"texto): {melanoma_text['count']} lesões.",
        f"- `Clinical Diagnosis == 2`: {clinical['count']} lesões.",
        "",
    ]

    if value_counts:
        non_melanoma_count = non_empty["count"] - melanoma_text["count"]
        lines.append(
            f"`Histological Diagnosis` não vazio ({non_empty['count']} registros) não "
            "é o mesmo que melanoma confirmado: o campo é preenchido para qualquer "
            "biópsia registrada, não só para melanoma. Distribuição dos valores "
            "distintos encontrados nos registros não vazios:"
        )
        lines.append("")
        for value, count in sorted(value_counts.items()):
            marca = "melanoma" if "melanoma" in value.lower() else "não melanoma"
            lines.append(f"- `{value}` ({marca}): {count}")
        lines.append("")
        lines.append(
            f"Logo, dos {non_empty['count']} registros histológicos não vazios, "
            f"**{melanoma_text['count']}** mencionam melanoma no texto e os demais "
            f"**{non_melanoma_count}** são biópsias confirmadas de lesões não "
            "melanoma. A comparação metodologicamente relevante para avaliar o "
            "critério clínico é **melanoma-por-texto × `Clinical Diagnosis == 2`** "
            f"({melanoma_text['count']} × {clinical['count']}), não "
            f"**não-vazio × `Clinical Diagnosis == 2`** ({non_empty['count']} × "
            f"{clinical['count']})."
        )
        lines.append("")

    return "\n".join(lines)


def _render_ham10000(summary: dict[str, Any]) -> str:
    if summary["status"] == "ausente":
        return "## HAM10000\n\n**Status: ausente** — caminho de imagens, máscaras ou metadados não encontrado.\n"

    lesion_conflicts = summary["lesion_id_to_dx_missing_or_conflicting"]
    lines = [
        "## HAM10000",
        "",
        f"**Status: {summary['status']}**",
        "",
        f"- Linhas de metadados: {summary['metadata_rows']}",
        f"- Arquivos de imagem (`.jpg`) encontrados: {summary['image_files_found']}",
        f"- Arquivos de máscara (`.png`) encontrados: {summary['mask_files_found']}",
        f"- Lesões distintas (`lesion_id`): {summary['distinct_lesions']}",
        "",
        "### Pareamento (3 direções)",
        "",
        _pairing_line("Metadados × imagens", summary["pairing_metadata_vs_images"]),
        _pairing_line("Metadados × máscaras", summary["pairing_metadata_vs_masks"]),
        _pairing_line("Imagens × máscaras", summary["pairing_images_vs_masks"]),
        "",
        "### `lesion_id` → `dx`",
        "",
        f"- Linhas sem `lesion_id`: {len(lesion_conflicts['missing_id'])}",
        f"- `lesion_id` com `dx` conflitante entre linhas: {len(lesion_conflicts['conflicting'])}",
        "",
        "### Distribuição por `dx`",
        "",
    ]
    for dx, count in sorted(summary["dx_counts"].items(), key=lambda kv: str(kv[0])):
        lines.append(f"- `{dx}`: {count}")
    lines.append("")
    lines.append(
        "### Distribuição de imagens por lesão (nº de imagens → nº de lesões com esse nº)"
    )
    lines.append("")
    for n_images, n_lesions in summary["images_per_lesion_distribution"].items():
        lines.append(f"- {n_images} imagem(ns): {n_lesions} lesões")
    lines.append("")
    hash_status = summary.get(
        "hash_status", "hashed" if summary["image_manifest"] else "skipped_no_previous"
    )
    if hash_status == "hashed":
        lines.append(
            "> Nota: hashing sha256 de ~20 mil arquivos (imagens + máscaras) foi "
            "executado nesta rodada; pode levar minutos — é esperado, não é falha."
        )
    elif hash_status == "reused_from_previous":
        lines.append(
            "> Nota: hashing sha256 pulado nesta rodada (`--skip-ham10000-hash`); "
            "`image_manifest`/`mask_manifest` foram reaproveitados de um manifesto "
            "anterior em disco (hashes reais preservados, não recalculados e não "
            "apagados)."
        )
    else:
        lines.append(
            "> Nota: hashing sha256 pulado nesta rodada (`--skip-ham10000-hash`) e "
            "nenhum manifesto anterior com hashes foi encontrado para reaproveitar; "
            "`image_manifest`/`mask_manifest` ficam vazios neste relatório (ainda não "
            "houve nenhuma rodada com hashing real)."
        )
    lines.append("")
    return "\n".join(lines)


def _render_isic2018_task1(summary: dict[str, Any]) -> str:
    if summary["status"] == "ausente":
        return (
            "## ISIC2018 Task1\n\n**Status: ausente** — zip de imagens e/ou zip de "
            "máscaras não encontrado.\n"
        )

    lines = [
        "## ISIC2018 Task1",
        "",
        f"**Status: {summary['status']}**",
        "",
        f"- Entradas totais no zip de imagens: {summary['images_zip_entry_count_total']}",
        f"- Entradas `.jpg` (imagens): {summary['images_zip_image_entries']}",
        "- Outras entradas no zip de imagens:",
    ]
    for e in summary["images_zip_other_entries"]:
        lines.append(f"  - `{e['name']}` ({e['size']} bytes)")
    lines.append(f"- Entradas totais no zip de máscaras: {summary['masks_zip_entry_count_total']}")
    lines.append(f"- Entradas `.png` (máscaras): {summary['masks_zip_mask_entries']}")
    lines.append("- Outras entradas no zip de máscaras:")
    for e in summary["masks_zip_other_entries"]:
        lines.append(f"  - `{e['name']}` ({e['size']} bytes)")
    lines.append("")
    lines.append("### Pareamento")
    lines.append("")
    lines.append(_pairing_line("Imagem × máscara", summary["pairing_image_vs_mask"]))
    lines.append("")
    lines.append(f"> {summary['note']}")
    lines.append("")
    return "\n".join(lines)


def _render_bcn20000(summary: dict[str, Any]) -> str:
    if summary["status"] == "ausente":
        return "## BCN20000\n\n**Status: ausente** — zip externo não encontrado.\n"

    train_conflicts = summary["train_lesion_id_to_diagnosis_missing_or_conflicting"]
    lines = [
        "## BCN20000",
        "",
        f"**Status: {summary['status']}**",
        "",
        "### Entradas do zip externo (não extraídas)",
        "",
    ]
    for e in summary["outer_zip_entries"]:
        lines.append(f"- `{e['name']}` ({e['size']} bytes)")
    lines += [
        "",
        f"- Linhas no CSV de treino: {summary['train_csv_rows']} "
        f"(lesões distintas: {summary['train_distinct_lesions']})",
        f"- Linhas no CSV de teste: {summary['test_csv_rows']} "
        f"(lesões distintas: {summary['test_distinct_lesions']})",
        "",
        "### `lesion_id` → `diagnosis` (treino)",
        "",
        f"- Linhas sem `lesion_id`: {len(train_conflicts['missing_id'])}",
        f"- `lesion_id` com `diagnosis` conflitante entre linhas: {len(train_conflicts['conflicting'])}",
        "",
        "### Distribuição por `diagnosis` (treino)",
        "",
    ]
    for dx, count in sorted(summary["train_diagnosis_counts"].items(), key=lambda kv: str(kv[0])):
        lines.append(f"- `{dx}`: {count}")
    lines += [
        "",
        f"- `diagnosis` não vazio no CSV de teste: {summary['test_diagnosis_non_empty_count']}/{summary['test_csv_rows']}",
        "",
        "### Observações estruturais",
        "",
        f"- {summary['notes']['test_diagnosis_empty']}",
        f"- {summary['notes']['lesion_id_format_divergence']}",
        f"- {summary['notes']['images_not_inventoried']}",
        "",
    ]
    return "\n".join(lines)


_RENDERERS = {
    "ph2": _render_ph2,
    "ham10000": _render_ham10000,
    "isic2018_task1": _render_isic2018_task1,
    "bcn20000": _render_bcn20000,
}


def render_report(summaries: dict[str, dict[str, Any]]) -> str:
    """Build the full consolidated Markdown report from the 4 dataset summaries.

    ``summaries`` keys must be a subset of {"ph2", "ham10000",
    "isic2018_task1", "bcn20000"}; missing keys are skipped.
    """
    header = [
        f"# Inventário e Validação Estrutural dos Datasets — {date.today().isoformat()}",
        "",
        "Inventário estrutural (sem amostragem, sem treino, sem chamadas a "
        "LLM/VLM, sem extração de zips grandes) dos 4 datasets do protocolo "
        "experimental: PH2, HAM10000, ISIC2018 Task1 e BCN20000.",
        "",
    ]
    sections = [
        _RENDERERS[name](summary).rstrip("\n")
        for name, summary in summaries.items()
        if name in _RENDERERS
    ]
    return "\n".join(header).rstrip("\n") + "\n\n" + "\n\n".join(sections) + "\n"
