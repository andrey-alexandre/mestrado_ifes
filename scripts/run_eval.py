"""Entry point CLI para avaliação dos resultados do pipeline.

Uso:
    python scripts/run_eval.py --results results/llm_result_gpt-4o-mini_20241201.json
    python scripts/run_eval.py --results results/llm_result_gpt-4o-mini_20241201.json --verbose
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Garante que o diretório raiz do projeto está no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import load_results, print_results, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia os resultados do pipeline multiagente de diagnóstico."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Caminho para o arquivo de configuração YAML. (padrão: config/config.yaml)",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Caminho para o arquivo JSON de resultados gerado pelo pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe detalhes completos de cada algoritmo (ABCD, Menzies, SPCL).",
    )
    parser.add_argument(
        "--save-metrics",
        default=None,
        help="Caminho opcional para salvar as métricas calculadas em JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Carregando resultados de: %s", args.results)
    result_dict = load_results(args.results)

    # Exibir diagnósticos
    print_results(result_dict, verbose=args.verbose)

    # Calcular métricas
    metrics = compute_metrics(result_dict)

    # Salvar métricas em arquivo se solicitado
    if metrics and args.save_metrics:
        save_path = Path(args.save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        logger.info("Métricas salvas em: %s", save_path)


if __name__ == "__main__":
    main()
