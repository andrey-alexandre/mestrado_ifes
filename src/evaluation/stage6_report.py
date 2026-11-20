"""Etapa 6 (continuação) -- montagem das linhas de tabela e escritores
CSV/Markdown genéricos para a análise quantitativa final (protocolo,
seção 6). Não calcula nada aqui -- só formata estruturas já calculadas
por `stage6_metrics`/`stage6_pairwise`/`stage6_bootstrap_ci`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.stage6_bootstrap_ci import CIResult, METRIC_NAMES
from src.evaluation.stage6_metrics import Stage6Indicators
from src.evaluation.stage6_pairwise import PairwiseIntersection
from src.evaluation.stage6_wilson import WilsonResult

METRIC_LABELS = {
    "accuracy": "Acurácia", "precision": "Precisão", "sensitivity": "Sensibilidade",
    "specificity": "Especificidade", "f1": "F1",
}


def _fmt_pct(x: float | None) -> str:
    return "indefinida" if x is None else f"{x * 100:.2f}%"


def format_ci_markdown(ci: CIResult | None) -> str:
    if ci is None or ci.point_estimate is None:
        return "indefinida"
    point = f"{ci.point_estimate * 100:.2f}%"
    if ci.ci_reported:
        return f"{point} [{ci.ci_low * 100:.2f}%, {ci.ci_high * 100:.2f}%]"
    return f"{point} [IC não reportado; {ci.n_valid_replicas}/{ci.n_total_replicas} réplicas válidas]"


def ci_to_dict(ci: CIResult | None) -> dict[str, Any]:
    if ci is None:
        return {"point_estimate": None, "ci_low": None, "ci_high": None,
                "n_valid_replicas": 0, "n_total_replicas": 0, "ci_reported": False}
    return {
        "point_estimate": ci.point_estimate, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
        "n_valid_replicas": ci.n_valid_replicas, "n_total_replicas": ci.n_total_replicas,
        "ci_reported": ci.ci_reported,
    }


def build_main_table_row(
    system_condition_label: str,
    universe_size: int,
    indicators: Stage6Indicators,
    ci_by_metric: dict[str, CIResult],
    is_provisional: bool,
) -> dict[str, Any]:
    bm = indicators.binary_metrics
    row: dict[str, Any] = {
        "sistema_condicao": system_condition_label,
        "provisorio": is_provisional,
        "binarios_validos": bm.n_binary,
        "total_manifesto": universe_size,
        "binarios_validos_pct": (bm.n_binary / universe_size) if universe_size else None,
        "acertos": bm.tp + bm.tn,
        "acertos_pct_sobre_total": indicators.accuracy_over_total,
        "nota": "métricas desta linha calculadas sobre os próprios casos binários válidos",
    }
    for name in METRIC_NAMES:
        ci = ci_by_metric.get(name)
        row[f"{name}_point"] = ci.point_estimate if ci else getattr(bm, name)
        row[f"{name}_ci_low"] = ci.ci_low if ci else None
        row[f"{name}_ci_high"] = ci.ci_high if ci else None
        row[f"{name}_ci_reported"] = ci.ci_reported if ci else False
        row[f"{name}_n_valid_replicas"] = ci.n_valid_replicas if ci else 0
    return row


def main_table_markdown_row(row: dict[str, Any], ci_by_metric: dict[str, CIResult]) -> dict[str, str]:
    return {
        "Sistema e condição": row["sistema_condicao"] + (" (PROVISÓRIO)" if row["provisorio"] else ""),
        "Binários válidos / total": f"{row['binarios_validos']} / {row['total_manifesto']} ({_fmt_pct(row['binarios_validos_pct'])})",
        "Acertos / total": f"{row['acertos']} / {row['total_manifesto']} ({_fmt_pct(row['acertos_pct_sobre_total'])})",
        "Acurácia [IC95%]": format_ci_markdown(ci_by_metric.get("accuracy")),
        "Precisão [IC95%]": format_ci_markdown(ci_by_metric.get("precision")),
        "Sensibilidade [IC95%]": format_ci_markdown(ci_by_metric.get("sensitivity")),
        "Especificidade [IC95%]": format_ci_markdown(ci_by_metric.get("specificity")),
        "F1 [IC95%]": format_ci_markdown(ci_by_metric.get("f1")),
    }


def format_wilson_markdown(w: WilsonResult | None) -> str:
    if w is None or w.point_estimate is None:
        return "indefinida"
    return f"{w.point_estimate * 100:.2f}% [{w.ci_low * 100:.2f}%, {w.ci_high * 100:.2f}%] (Wilson)"


def wilson_to_dict(w: WilsonResult | None) -> dict[str, Any]:
    if w is None:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "numerator": 0,
                "denominator": 0, "confidence": None, "method": None}
    return {
        "point_estimate": w.point_estimate, "ci_low": w.ci_low, "ci_high": w.ci_high,
        "numerator": w.numerator, "denominator": w.denominator, "confidence": w.confidence,
        "method": w.method,
    }


# Métricas individuais cujo IC passou a ser Wilson na revisão metodológica
# 2026-11-19 (decisão do pesquisador) -- F1 permanece bootstrap (não tem
# forma de proporção simples numerador/denominador: é a média harmônica de
# precisão e sensibilidade, não uma única contagem de sucessos sobre um
# único denominador).
WILSON_METRIC_NAMES = ("accuracy", "precision", "sensitivity", "specificity", "coverage_binary", "accuracy_over_total")


def build_main_table_row_wilson(
    system_condition_label: str,
    universe_size: int,
    indicators: Stage6Indicators,
    wilson_by_metric: dict[str, WilsonResult],
    f1_ci: CIResult,
    is_provisional: bool,
) -> dict[str, Any]:
    """Versão NOVA (revisão metodológica 2026-11-19) de
    ``build_main_table_row`` -- Wilson para acurácia/precisão/
    sensibilidade/especificidade/cobertura binária/acertos sobre o total;
    bootstrap só para F1. ``build_main_table_row`` (acima) permanece
    intocada, usada pela versão anterior do orquestrador
    (`scripts/run_final_quantitative_analysis.py`), para não alterar uma
    execução em andamento daquela versão."""
    bm = indicators.binary_metrics
    cov_w = wilson_by_metric.get("coverage_binary")
    acc_tot_w = wilson_by_metric.get("accuracy_over_total")
    row: dict[str, Any] = {
        "sistema_condicao": system_condition_label,
        "provisorio": is_provisional,
        "metodologia": "v2_wilson_proporcoes_bootstrap_f1_e_diferencas_2026_11_19",
        "binarios_validos": bm.n_binary,
        "total_manifesto": universe_size,
        "binarios_validos_pct": cov_w.point_estimate if cov_w else None,
        "binarios_validos_pct_ci_low": cov_w.ci_low if cov_w else None,
        "binarios_validos_pct_ci_high": cov_w.ci_high if cov_w else None,
        "binarios_validos_pct_method": cov_w.method if cov_w else None,
        "acertos": bm.tp + bm.tn,
        "acertos_pct_sobre_total": acc_tot_w.point_estimate if acc_tot_w else None,
        "acertos_pct_sobre_total_ci_low": acc_tot_w.ci_low if acc_tot_w else None,
        "acertos_pct_sobre_total_ci_high": acc_tot_w.ci_high if acc_tot_w else None,
        "acertos_pct_sobre_total_method": acc_tot_w.method if acc_tot_w else None,
        "nota": (
            "métricas desta linha calculadas sobre os próprios casos binários válidos; "
            "acurácia/precisão/sensibilidade/especificidade/cobertura binária/acertos sobre o total: "
            "IC95% de Wilson (intervalo binomial aproximado para a proporção observada, sem "
            "condicionamento por categoria diagnóstica); F1: IC95% por bootstrap estratificado"
        ),
    }
    for name in ("accuracy", "precision", "sensitivity", "specificity"):
        w = wilson_by_metric.get(name)
        row[f"{name}_point"] = w.point_estimate if w else getattr(bm, name)
        row[f"{name}_ci_low"] = w.ci_low if w else None
        row[f"{name}_ci_high"] = w.ci_high if w else None
        row[f"{name}_method"] = w.method if w else None
        row[f"{name}_numerator"] = w.numerator if w else None
        row[f"{name}_denominator"] = w.denominator if w else None
    row["f1_point"] = f1_ci.point_estimate if f1_ci else bm.f1
    row["f1_ci_low"] = f1_ci.ci_low if f1_ci else None
    row["f1_ci_high"] = f1_ci.ci_high if f1_ci else None
    row["f1_ci_reported"] = f1_ci.ci_reported if f1_ci else False
    row["f1_n_valid_replicas"] = f1_ci.n_valid_replicas if f1_ci else 0
    row["f1_method"] = "bootstrap"
    return row


def main_table_markdown_row_wilson(row: dict[str, Any], f1_ci: CIResult) -> dict[str, str]:
    def _wilson_cell(point, lo, hi):
        if point is None:
            return "indefinida"
        return f"{point * 100:.2f}% [{lo * 100:.2f}%, {hi * 100:.2f}%] (Wilson)"

    return {
        "Sistema e condição": row["sistema_condicao"] + (" (PROVISÓRIO)" if row["provisorio"] else ""),
        "Binários válidos / total": (
            f"{row['binarios_validos']} / {row['total_manifesto']} "
            f"({_wilson_cell(row['binarios_validos_pct'], row['binarios_validos_pct_ci_low'], row['binarios_validos_pct_ci_high'])})"
        ),
        "Acertos / total": (
            f"{row['acertos']} / {row['total_manifesto']} "
            f"({_wilson_cell(row['acertos_pct_sobre_total'], row['acertos_pct_sobre_total_ci_low'], row['acertos_pct_sobre_total_ci_high'])})"
        ),
        "Acurácia [IC95%]": _wilson_cell(row["accuracy_point"], row["accuracy_ci_low"], row["accuracy_ci_high"]),
        "Precisão [IC95%]": _wilson_cell(row["precision_point"], row["precision_ci_low"], row["precision_ci_high"]),
        "Sensibilidade [IC95%]": _wilson_cell(row["sensitivity_point"], row["sensitivity_ci_low"], row["sensitivity_ci_high"]),
        "Especificidade [IC95%]": _wilson_cell(row["specificity_point"], row["specificity_ci_low"], row["specificity_ci_high"]),
        "F1 [IC95% bootstrap]": format_ci_markdown(f1_ci),
    }


def build_operational_table_row(system_condition_label: str, universe_size: int, counts: dict[str, int]) -> dict[str, Any]:
    row: dict[str, Any] = {"sistema_condicao": system_condition_label, "total_manifesto": universe_size}
    row.update(counts)
    row["soma_checagem"] = sum(counts.values())
    row["soma_bate_total"] = row["soma_checagem"] == universe_size
    return row


def build_confusion_matrix_row(system_condition_label: str, indicators: Stage6Indicators) -> dict[str, Any]:
    bm = indicators.binary_metrics
    return {
        "sistema_condicao": system_condition_label,
        "tp": bm.tp, "tn": bm.tn, "fp": bm.fp, "fn": bm.fn, "n_binary": bm.n_binary,
    }


def build_pairwise_row(
    analysis_label: str, pairwise: PairwiseIntersection, diff_ci_by_metric: dict[str, CIResult],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "analise": analysis_label,
        "primeiro_sistema": pairwise.first_label,
        "segundo_sistema": pairwise.second_label,
        "n_intersecao": pairwise.n_intersection,
        "n_melanoma_intersecao": pairwise.n_melanoma,
        "n_nao_melanoma_intersecao": pairwise.n_nao_melanoma,
    }
    # Itera sobre as chaves de ``diff_ci_by_metric`` (não uma lista fixa de
    # nomes) para que esta função sirva tanto para o conjunto histórico de
    # 5 métricas quanto para o conjunto estendido de 7 (5 métricas +
    # cobertura binária + acertos sobre o total, revisão 2026-11-19) sem
    # duplicação -- comportamento idêntico ao anterior quando o chamador
    # passa só as 5 chaves originais.
    for name, ci in diff_ci_by_metric.items():
        row[f"diff_{name}_pp"] = (ci.point_estimate * 100) if (ci and ci.point_estimate is not None) else None
        row[f"diff_{name}_ci_low_pp"] = (ci.ci_low * 100) if (ci and ci.ci_low is not None) else None
        row[f"diff_{name}_ci_high_pp"] = (ci.ci_high * 100) if (ci and ci.ci_high is not None) else None
        row[f"diff_{name}_ci_reported"] = ci.ci_reported if ci else False
        row[f"diff_{name}_n_valid_replicas"] = ci.n_valid_replicas if ci else 0
        row[f"diff_{name}_method"] = "bootstrap"
    return row


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: str | Path, rows: Sequence[dict[str, str]], title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_(sem linhas)_")
    else:
        columns = list(rows[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
