"""Consolida o arquivo de revisão qualitativa preenchido
(``review_template.csv``) por modelo e critério: quantidade revisada,
ocorrências do problema, casos não avaliáveis, denominador efetivamente
avaliável, revisões ainda pendentes.

Não decide aprovação/exclusão de nenhum modelo ou caso -- só agrega as
contagens já registradas pelo revisor humano. Não faz nenhuma nova
inferência/chamada a LLM.

Usage:
    python scripts/consolidate_qualitative_reviews.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.qualitative_review_schema import (  # noqa: E402
    AUSENTE,
    CRITERION_BY_ID,
    NAO_AVALIAVEL,
    PENDENTE,
    PRESENTE,
)

OUTPUT_DIR = REPO_ROOT / "experiments" / "evaluation" / "qualitative_review"
REVIEW_TEMPLATE_CSV = OUTPUT_DIR / "review_template.csv"
MODEL_ID_KEY_JSON = OUTPUT_DIR / "model_id_key_CONFIDENCIAL.json"
CONSOLIDATION_CSV = OUTPUT_DIR / "qualitative_review_consolidation.csv"
CONSOLIDATION_MD = OUTPUT_DIR / "qualitative_review_consolidation.md"


def main() -> None:
    if not REVIEW_TEMPLATE_CSV.exists():
        raise FileNotFoundError(f"{REVIEW_TEMPLATE_CSV} não encontrado -- rode scripts/prepare_qualitative_review_materials.py primeiro.")

    with open(REVIEW_TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} linhas carregadas de {REVIEW_TEMPLATE_CSV}.")

    neutral_to_real = {}
    if MODEL_ID_KEY_JSON.exists():
        with open(MODEL_ID_KEY_JSON, encoding="utf-8") as f:
            neutral_to_real = json.load(f).get("neutral_to_real", {})

    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (row["model_label_neutral"], row["criterion_id"])
        c = counts.setdefault(key, {"total": 0, PRESENTE: 0, AUSENTE: 0, NAO_AVALIAVEL: 0, PENDENTE: 0})
        c["total"] += 1
        status = row["status"]
        if status not in (PRESENTE, AUSENTE, NAO_AVALIAVEL, PENDENTE):
            raise ValueError(f"status desconhecido em {REVIEW_TEMPLATE_CSV}: {status!r} (lesao {row['lesion_id']})")
        c[status if status else PENDENTE] += 1

    consolidated_rows = []
    for (model_label, criterion_id), c in sorted(counts.items()):
        denominador_avaliavel = c[PRESENTE] + c[AUSENTE]
        taxa = (c[PRESENTE] / denominador_avaliavel) if denominador_avaliavel else None
        consolidated_rows.append({
            "model_label_neutral": model_label,
            "model_label_real": neutral_to_real.get(model_label, ""),
            "criterion_id": criterion_id,
            "criterion_label": CRITERION_BY_ID[criterion_id].label if criterion_id in CRITERION_BY_ID else criterion_id,
            "n_lesoes_total": c["total"],
            "n_presente": c[PRESENTE],
            "n_ausente": c[AUSENTE],
            "n_nao_avaliavel": c[NAO_AVALIAVEL],
            "n_pendente": c[PENDENTE],
            "denominador_avaliavel": denominador_avaliavel,
            "taxa_ocorrencia_sobre_avaliavel": taxa,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONSOLIDATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(consolidated_rows[0].keys()) if consolidated_rows else [])
        writer.writeheader()
        writer.writerows(consolidated_rows)

    with open(CONSOLIDATION_MD, "w", encoding="utf-8") as f:
        f.write("# Consolidação da revisão qualitativa (por modelo e critério)\n\n")
        f.write("Análise dos resultados -- não constitui critério de aprovação de modelo nem autorização para excluir casos.\n\n")
        f.write("| Modelo | Critério | Revisado | Presente | Ausente | Não avaliável | Pendente | Denominador avaliável | Taxa (presente/avaliável) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in consolidated_rows:
            taxa_str = f"{r['taxa_ocorrencia_sobre_avaliavel']:.3f}" if r["taxa_ocorrencia_sobre_avaliavel"] is not None else "indefinida"
            f.write(
                f"| {r['model_label_neutral']} | {r['criterion_label']} | {r['n_lesoes_total']} | {r['n_presente']} | "
                f"{r['n_ausente']} | {r['n_nao_avaliavel']} | {r['n_pendente']} | {r['denominador_avaliavel']} | {taxa_str} |\n"
            )

    n_pendente_total = sum(r["n_pendente"] for r in consolidated_rows)
    print(f"\nConsolidação: {CONSOLIDATION_CSV}")
    print(f"Relatório: {CONSOLIDATION_MD}")
    print(f"Linhas ainda pendentes (status vazio) no total: {n_pendente_total}")
    if n_pendente_total > 0:
        print("AVISO: esta consolidação inclui revisões parcialmente preenchidas -- confira a coluna 'n_pendente' antes de usar os números como finais.")


if __name__ == "__main__":
    main()
