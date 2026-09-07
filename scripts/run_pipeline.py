"""Entry point CLI para execução do pipeline completo de ponta a ponta.

Executa em sequência:
  1. Pré-processamento: extração de PDFs e construção do índice RAG
  2. Carregamento do dataset (ISIC ou PH2)
  3. Carregamento do modelo de segmentação (apenas no modo "model")
  4. Inferência: pipeline multiagente sobre todas as imagens
  5. Avaliação: métricas e exibição dos diagnósticos

Uso:
    # ISIC + U-Net (padrão)
    python scripts/run_pipeline.py --skip-preprocess

    # PH2 + segmentação pré-computada
    python scripts/run_pipeline.py --dataset ph2 --segmentation-mode precomputed --skip-preprocess

    # Tudo do zero com ISIC
    python scripts/run_pipeline.py --max-images 10
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Garante que o diretório raiz do projeto está no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import prepare_medical_texts, build_rag
from src.agents.agents import build_graph
from src.pipeline.pipeline import run_pipeline
from src.evaluation.evaluate import load_results, print_results, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline completo de diagnóstico de lesões cutâneas (pré-proc + inferência + avaliação)."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Caminho para o arquivo de configuração YAML. (padrão: config/config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=["isic", "ph2"],
        default=None,
        help=(
            "Dataset a usar: 'isic' (CSV + imagens .jpg) ou 'ph2' (estrutura de diretórios .bmp). "
            "Sobrescreve pipeline.dataset."
        ),
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=["model", "precomputed"],
        default=None,
        help=(
            "'model' — gera segmentação via U-Net em runtime (padrão para ISIC); "
            "'precomputed' — usa máscara pré-existente do dataset (padrão para PH2). "
            "Sobrescreve segmentation.mode."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Número máximo de imagens a processar. 0 = todas. Sobrescreve pipeline.max_images.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Nome do LLM. OpenAI: 'gpt-4o-mini'; Gemini: 'gemini-1.5-pro', 'gemini-2.0-flash'; "
            "Ollama local: 'llava:13b'. Sobrescreve llm.model_name."
        ),
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help=(
            "Pula a extração dos PDFs e assume que medical_texts.txt já existe. "
            "Use quando o RAG já foi construído anteriormente."
        ),
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "Amostra igual de classe 0 e classe 1. "
            "Com --max-images N, seleciona N//2 por classe. "
            "Sobrescreve pipeline.balanced."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Pula a etapa de avaliação ao final do pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe detalhes completos de cada diagnóstico na avaliação.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Carregar configuração base
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Aplicar overrides da linha de comando
    if args.dataset is not None:
        cfg["pipeline"]["dataset"] = args.dataset
    if args.segmentation_mode is not None:
        cfg["segmentation"]["mode"] = args.segmentation_mode
    if args.max_images is not None:
        cfg["pipeline"]["max_images"] = args.max_images
    if args.model_name is not None:
        cfg["llm"]["model_name"] = args.model_name
    if args.balanced:
        cfg["pipeline"]["balanced"] = True

    seg_mode = cfg["segmentation"]["mode"]
    dataset_type = cfg["pipeline"].get("dataset", "isic")

    logger.info("=== ETAPA 1: Pré-processamento ===")
    if args.skip_preprocess:
        logger.info("Pulando extração de PDFs (--skip-preprocess ativo).")
    else:
        prepare_medical_texts(
            zip_path=cfg["paths"]["rag_zip"],
            extracted_path=cfg["paths"]["extracted_pdfs"],
            output_txt_path=cfg["paths"]["medical_texts"],
        )

    logger.info("=== ETAPA 2: Construção do RAG ===")
    retriever = build_rag(
        medical_texts_path=cfg["paths"]["medical_texts"],
        embedding_model=cfg["rag"]["embedding_model"],
        chunk_size=cfg["rag"]["chunk_size"],
        chunk_overlap=cfg["rag"]["chunk_overlap"],
    )

    logger.info("=== ETAPA 3: Carregamento do dataset (%s) ===", dataset_type)
    if dataset_type == "ph2":
        from src.data.dataset import PH2Dataset
        dataset = PH2Dataset(
            dataset_dir=cfg["paths"]["ph2_dataset_dir"],
            random_seed=cfg["pipeline"]["random_seed"],
            metadata_txt=cfg["paths"].get("ph2_metadata_txt"),
        )
    else:
        from src.data.dataset import SkinLesionDataset
        dataset = SkinLesionDataset(
            metadata_csv=cfg["paths"]["metadata_csv"],
            image_dir=cfg["paths"]["image_dir"],
            random_seed=cfg["pipeline"]["random_seed"],
        )
    image_entries = dataset.get_image_paths(
        max_images=cfg["pipeline"]["max_images"],
        balanced=cfg["pipeline"].get("balanced", False),
    )

    logger.info("=== ETAPA 4: Modelo de segmentação (modo: %s) ===", seg_mode)
    seg_model = None
    if seg_mode == "model":
        from src.models.model import load_segmentation_model
        seg_model = load_segmentation_model(
            model_path=cfg["paths"]["model_path"],
            model_name=cfg["segmentation"]["model_name"],
            img_ch=cfg["segmentation"]["img_ch"],
            output_ch=cfg["segmentation"]["output_ch"],
            device=cfg["segmentation"]["device"],
        )
    else:
        logger.info("Modo 'precomputed': U-Net não carregado.")

    logger.info("=== ETAPA 5: Construção do grafo de agentes ===")
    compiled_graph = build_graph(
        retriever=retriever,
        model_name=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"],
        seg_threshold=cfg["segmentation"]["threshold"],
        segmentation_mode=seg_mode,
        seg_model=seg_model,
    )

    logger.info("=== ETAPA 6: Execução do pipeline de inferência ===")
    output_path = run_pipeline(
        compiled_graph=compiled_graph,
        image_entries=image_entries,
        output_dir=cfg["paths"]["output_dir"],
        model_name=cfg["llm"]["model_name"],
        filename_prefix=cfg["output"]["filename_prefix"],
        date_format=cfg["output"]["date_format"],
    )

    if not args.skip_eval:
        logger.info("=== ETAPA 7: Avaliação dos resultados ===")
        result_dict = load_results(output_path)
        print_results(result_dict, verbose=args.verbose)
        compute_metrics(result_dict)

    logger.info("Pipeline completo finalizado. Resultados em: %s", output_path)


if __name__ == "__main__":
    main()
