"""Carregamento de resultados, exibição de diagnósticos e cálculo de métricas."""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Mapeamento do campo 'diagnostico' do SummaryAnswer para label binário
_DIAGNOSTICO_TO_LABEL = {
    "suspeita de melanoma": 1,
    "lesão indeterminada": 1,  # conservador: tratar como positivo
    "lesão benigna": 0,
}


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


def compute_metrics(result_dict: Dict) -> Optional[Dict]:
    """Calcula métricas de acurácia comparando o diagnóstico final com o rótulo real.

    Mapeia o campo 'validation.diagnostico' para uma predição binária:
    - "Suspeita de Melanoma" ou "Lesão Indeterminada" → 1
    - "Lesão Benigna" → 0

    Args:
        result_dict: Dicionário de resultados (saída de load_results).

    Returns:
        Dicionário com métricas (acurácia, VP, VN, FP, FN) ou None se
        não houver rótulos reais disponíveis.
    """
    y_true = []
    y_pred = []

    for path, event in result_dict.items():
        real_value = event.get("real_value", -1)
        if real_value == -1:
            continue  # rótulo ausente

        validation = event.get("validation")
        if not validation:
            continue

        diagnostico_str = (
            validation.get("diagnostico", "").lower()
            if isinstance(validation, dict)
            else str(validation).lower()
        )

        # Encontrar a categoria mais próxima pelo prefixo
        pred = -1
        for key, label in _DIAGNOSTICO_TO_LABEL.items():
            if key in diagnostico_str:
                pred = label
                break

        if pred == -1:
            logger.warning("Diagnóstico não reconhecido: '%s' em %s", diagnostico_str, path)
            continue

        y_true.append(real_value)
        y_pred.append(pred)

    if not y_true:
        logger.warning("Nenhum par (real, predito) disponível para calcular métricas.")
        return None

    n = len(y_true)
    tp = sum(1 for r, p in zip(y_true, y_pred) if r == 1 and p == 1)
    tn = sum(1 for r, p in zip(y_true, y_pred) if r == 0 and p == 0)
    fp = sum(1 for r, p in zip(y_true, y_pred) if r == 0 and p == 1)
    fn = sum(1 for r, p in zip(y_true, y_pred) if r == 1 and p == 0)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall para melanoma
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }

    logger.info("Métricas de avaliação (%d amostras):", n)
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    return metrics
