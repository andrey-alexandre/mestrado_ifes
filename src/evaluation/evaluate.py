"""Carregamento de resultados, exibição de diagnósticos e cálculo de métricas."""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from src.evaluation.case_status import (
    AUSENTE,
    INDETERMINADO,
    NAO_RECONHECIDO,
    NEGATIVO,
    POSITIVO,
    classify_case_status,
)

logger = logging.getLogger(__name__)


def load_results(results_path: str) -> Dict:
    """Carrega o arquivo JSON de resultados do pipeline.

    Args:
        results_path: Caminho para o arquivo JSON gerado por run_pipeline.

    Returns:
        Dicionário de resultados indexado pelo caminho da imagem.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        result_dict = json.load(f)

    logger.info("Carregados %d resultados de %s", len(result_dict), results_path)
    return result_dict


def print_results(result_dict: Dict, verbose: bool = False) -> None:
    """Exibe os diagnósticos de cada imagem no logger.

    Args:
        result_dict: Dicionário de resultados (saída de load_results).
        verbose: Se True, exibe os detalhes completos de cada algoritmo.
    """
    sep = "=" * 100
    for path, event in result_dict.items():
        logger.info("\nImagem: %s", path)
        logger.info("  Real value (target): %s", event.get("real_value"))

        if verbose:
            logger.info("  Diagnóstico ABCD:\n\t%s", event.get("diagnosis_abcd"))
            logger.info(sep)
            logger.info("  Diagnóstico Menzies:\n\t%s", event.get("diagnosis_menzies"))
            logger.info(sep)
            logger.info("  Diagnóstico SPCL:\n\t%s", event.get("diagnosis_spcl"))
            logger.info(sep)

        validation = event.get("validation")
        if validation:
            diagnostico = (
                validation.get("diagnostico") if isinstance(validation, dict) else str(validation)
            )
            justificativa = (
                validation.get("justificativa") if isinstance(validation, dict) else ""
            )
            recomendacoes = (
                validation.get("recomendacoes") if isinstance(validation, dict) else ""
            )
            logger.info("  Diagnóstico Final: %s", diagnostico)
            logger.info("  Justificativa: %s", justificativa)
            logger.info("  Recomendações: %s", recomendacoes)

        logger.info("  Relatório RAG: %s", event.get("final_report"))
        logger.info(sep)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return (numerator / denominator) if denominator > 0 else None


def compute_metrics(result_dict: Dict) -> Optional[Dict]:
    """Calcula métricas comparando o diagnóstico final com o rótulo real.

    Correção de 2026-11-08 (etapa 6): o mapeamento antigo tratava
    silenciosamente "Lesão Indeterminada" como positivo. Isso foi removido.
    Cada caso é classificado por
    :func:`src.evaluation.case_status.classify_case_status` em um de cinco
    status explícitos (positivo/negativo/indeterminado/ausente/
    nao_reconhecido) e os quatro indicadores abaixo seguem a tabela
    aprovada pelo pesquisador (ver também
    ``src.evaluation.stage6_metrics``, que implementa a mesma regra para
    múltiplos sistemas/condições pareados por lesão):

      - As 5 métricas binárias (acurácia, precisão, sensibilidade,
        especificidade, F1) usam SÓ casos positivo/negativo —
        indeterminado nunca entra na matriz de confusão.
      - Cobertura binária = (positivo+negativo) / total com rótulo.
      - Taxa de saída válida = (positivo+negativo+indeterminado) / total
        com rótulo — indeterminado conta aqui, é uma saída válida.
      - Acurácia sobre o total = acertos / total com rótulo, onde
        indeterminado/ausente/nao_reconhecido contam como erro.

    Qualquer métrica sem denominador (ex. zero decisões binárias, zero
    positivos reais) volta como ``None`` — nunca 0.0.

    Args:
        result_dict: Dicionário de resultados (saída de load_results).

    Returns:
        Dicionário com os indicadores, ou None se não houver nenhum caso
        com rótulo real disponível.
    """
    n_total = 0
    n_positivo = n_negativo = n_indeterminado = n_ausente = n_nao_reconhecido = 0
    tp = tn = fp = fn = 0

    for path, event in result_dict.items():
        real_value = event.get("real_value", -1)
        if real_value == -1:
            continue  # rótulo ausente -- não entra em nenhum denominador

        n_total += 1
        status = classify_case_status(event)
        if status == POSITIVO:
            n_positivo += 1
        elif status == NEGATIVO:
            n_negativo += 1
        elif status == INDETERMINADO:
            n_indeterminado += 1
        elif status == AUSENTE:
            n_ausente += 1
        elif status == NAO_RECONHECIDO:
            n_nao_reconhecido += 1
            logger.warning("Diagnóstico não reconhecido em %s", path)

        if status in (POSITIVO, NEGATIVO):
            pred = 1 if status == POSITIVO else 0
            if real_value == 1 and pred == 1:
                tp += 1
            elif real_value == 0 and pred == 0:
                tn += 1
            elif real_value == 0 and pred == 1:
                fp += 1
            elif real_value == 1 and pred == 0:
                fn += 1

    if n_total == 0:
        logger.warning("Nenhum caso com rótulo real disponível para calcular métricas.")
        return None

    n_binary = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, n_binary)
    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)  # recall para melanoma
    specificity = _safe_div(tn, tn + fp)
    f1 = (
        _safe_div(2 * precision * sensitivity, precision + sensitivity)
        if precision is not None and sensitivity is not None
        else None
    )

    metrics = {
        "n_total": n_total,
        "n_binary": n_binary,
        "n_positivo": n_positivo,
        "n_negativo": n_negativo,
        "n_indeterminado": n_indeterminado,
        "n_ausente": n_ausente,
        "n_nao_reconhecido": n_nao_reconhecido,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "coverage_binary": _safe_div(n_positivo + n_negativo, n_total),
        "valid_output_rate": _safe_div(n_positivo + n_negativo + n_indeterminado, n_total),
        "accuracy_over_total": _safe_div(tp + tn, n_total),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }

    logger.info(
        "Métricas de avaliação (%d casos com rótulo, %d com decisão binária):",
        n_total, n_binary,
    )
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    return metrics
