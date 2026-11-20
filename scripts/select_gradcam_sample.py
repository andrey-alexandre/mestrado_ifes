"""Seleciona as 40 lesões (10 por célula TP/TN/FP/FN) para a galeria
Grad-CAM qualitativa, a partir das predições S0 já produzidas pela CNN
EfficientNetV2-S congelada sobre as 3.000 lesões (``s0_cnn_scores.csv``).

Não roda a CNN de novo, não ajusta o limiar, não faz inferência --
só lê o CSV já existente e sorteia (semente 42, ordenado por lesion_id
antes do sorteio -- ver ``src.cnn_baseline.gradcam_sampling``).

Usage:
    python scripts/select_gradcam_sample.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cnn_baseline.gradcam_sampling import DEFAULT_N_PER_CELL, DEFAULT_SEED, select_gradcam_sample  # noqa: E402

CNN_S0_SCORES = REPO_ROOT / "experiments" / "cnn_baseline" / "s0_ham10000_eval" / "s0_cnn_scores.csv"
OUTPUT_DIR = REPO_ROOT / "experiments" / "evaluation" / "gradcam"
OUTPUT_MANIFEST = OUTPUT_DIR / "gradcam_sample_manifest.csv"
OUTPUT_REPORT = OUTPUT_DIR / "gradcam_sample_report.md"


def main() -> None:
    if not CNN_S0_SCORES.exists():
        raise FileNotFoundError(f"{CNN_S0_SCORES} não encontrado -- a avaliação S0 da CNN precisa já existir.")

    with open(CNN_S0_SCORES, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} linhas carregadas de {CNN_S0_SCORES}.")

    result = select_gradcam_sample(rows, seed=DEFAULT_SEED, n_per_cell=DEFAULT_N_PER_CELL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for group, selected in result.selected_by_group.items():
        for row in selected:
            manifest_rows.append({
                "group": group, "lesion_id": row["lesion_id"], "image_id": row["image_id"],
                "dx": row.get("dx", ""), "true_binary_target": row["true_binary_target"],
                "predicted_binary_target": row["predicted_binary_target"],
                "predicted_score": row["predicted_score"], "seed": result.seed,
            })
    manifest_rows.sort(key=lambda r: (r["group"], r["lesion_id"]))

    with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "lesion_id", "image_id", "dx", "true_binary_target",
                                                "predicted_binary_target", "predicted_score", "seed"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Seleção da amostra Grad-CAM (40 lesões, 10 por célula de confusão)\n\n")
        f.write(f"- Semente: {result.seed}\n")
        f.write(f"- Fonte: `{CNN_S0_SCORES.relative_to(REPO_ROOT)}`\n")
        f.write(f"- Total selecionado: {sum(result.n_selected_by_group.values())}\n\n")
        f.write("| Célula | Elegíveis | Selecionados |\n|---|---|---|\n")
        for group in ("TP", "TN", "FP", "FN"):
            f.write(f"| {group} | {result.n_eligible_by_group[group]} | {result.n_selected_by_group[group]} |\n")
        if result.shortfall_groups:
            f.write("\n## Atalhos registrados (célula com menos de 10 elegíveis)\n\n")
            for group, note in result.shortfall_groups.items():
                f.write(f"- **{group}**: {note}\n")
        else:
            f.write("\nNenhum atalho -- todas as 4 células tinham ao menos 10 casos elegíveis.\n")

    print(f"\nManifesto: {OUTPUT_MANIFEST}")
    print(f"Relatório: {OUTPUT_REPORT}")
    for group in ("TP", "TN", "FP", "FN"):
        print(f"  {group}: {result.n_selected_by_group[group]}/{result.n_eligible_by_group[group]} elegíveis")
    if result.shortfall_groups:
        print("AVISO: ver atalhos registrados no relatório.")


if __name__ == "__main__":
    main()
