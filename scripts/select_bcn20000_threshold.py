"""Fix the binary decision threshold for the BCN20000 EfficientNetV2-S
baseline (stage 3) by the Youden index, using ONLY the internal BCN20000
validation scores (``val_scores.csv``, produced by
``scripts/train_bcn20000_efficientnet.py``). HAM10000 is never read or
referenced here -- it is reserved exclusively for the later S0 external
evaluation, which must reuse this threshold as a fixed input, never
re-tune it.

Usage:
    python scripts/select_bcn20000_threshold.py
    python scripts/select_bcn20000_threshold.py --val-scores path/to/val_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cnn_baseline.bcn20000_dataset import BINARY_TARGET_TO_LABEL  # noqa: E402
from src.cnn_baseline.threshold_selection import select_youden_threshold  # noqa: E402

DEFAULT_VAL_SCORES = str(REPO_ROOT / "experiments" / "cnn_baseline" / "training" / "val_scores.csv")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "cnn_baseline" / "training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-scores", default=DEFAULT_VAL_SCORES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    val_scores_path = Path(args.val_scores)
    with open(val_scores_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERRO: {val_scores_path} esta vazio.", file=sys.stderr)
        sys.exit(1)

    true_labels = [BINARY_TARGET_TO_LABEL[r["binary_target"]] for r in rows]
    predicted_scores = [float(r["predicted_score"]) for r in rows]

    selection = select_youden_threshold(true_labels, predicted_scores)
    selection["n_val_rows"] = len(rows)
    selection["n_positive"] = sum(true_labels)
    selection["n_negative"] = len(true_labels) - sum(true_labels)
    selection["val_scores_path"] = str(val_scores_path.resolve())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "threshold_selection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    md_path = output_dir / "threshold_selection.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "# Selecao de limiar (indice de Youden) -- BCN20000, validacao interna\n\n"
            f"- Linhas de validacao: {selection['n_val_rows']} "
            f"(melanoma={selection['n_positive']}, nao_melanoma={selection['n_negative']})\n"
            f"- Limiar escolhido: {selection['threshold']}\n"
            f"- Sensibilidade: {selection['sensitivity']}\n"
            f"- Especificidade: {selection['specificity']}\n"
            f"- Indice de Youden (J): {selection['youden_j']}\n"
            f"- Regra: {selection['rule']}\n"
            f"- Fonte: {selection['val_scores_path']}\n\n"
            "HAM10000 nao foi lido nem referenciado nesta execucao.\n"
        )

    print(f"limiar escolhido: {selection['threshold']}")
    print(f"sensibilidade: {selection['sensitivity']}  especificidade: {selection['specificity']}  J: {selection['youden_j']}")
    print(f"resultados gravados: {json_path}")
    print(f"relatorio gravado: {md_path}")


if __name__ == "__main__":
    main()
