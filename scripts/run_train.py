"""Entry point CLI para execução do pipeline de inferência.

Uso:
    # ISIC + segmentação via U-Net (padrão)
    python scripts/run_train.py

    # PH2 + segmentação pré-computada
    python scripts/run_train.py --dataset ph2 --segmentation-mode precomputed

    # Overrides pontuais
    python scripts/run_train.py --max-images 20 --model-name gpt-4o-mini
    python scripts/run_train.py --config config/config.yaml --output-dir results/
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Garante que o diretório raiz do projeto está no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import build_rag
from src.agents.agents import build_graph
from src.pipeline.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline multiagente de diagnóstico de lesões cutâneas."
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
        "--model-path",
        default=None,
        help="Caminho para o checkpoint .pkl da U-Net. Sobrescreve paths.model_path.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Caminho para o CSV de metadados ISIC. Sobrescreve paths.metadata_csv.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Diretório das imagens ISIC. Sobrescreve paths.image_dir.",
    )
    parser.add_argument(
        "--ph2-dataset-dir",
        default=None,
        help="Raiz do dataset PH2. Sobrescreve paths.ph2_dataset_dir.",
    )
    parser.add_argument(
        "--ph2-metadata-txt",
        default=None,
        help="Arquivo PH2_dataset.txt com diagnósticos histológicos. Sobrescreve paths.ph2_metadata_txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de saída para os resultados JSON. Sobrescreve paths.output_dir.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed para reprodutibilidade do shuffle. Sobrescreve pipeline.random_seed.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=None,
        help=(
            "Amostra igual de classe 0 e classe 1. "
            "Com --max-images N, seleciona N//2 por classe. "
            "Sobrescreve pipeline.balanced."
        ),
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
    if args.model_path is not None:
        cfg["paths"]["model_path"] = args.model_path
    if args.metadata_csv is not None:
        cfg["paths"]["metadata_csv"] = args.metadata_csv
    if args.image_dir is not None:
        cfg["paths"]["image_dir"] = args.image_dir
    if args.ph2_dataset_dir is not None:
        cfg["paths"]["ph2_dataset_dir"] = args.ph2_dataset_dir
    if args.ph2_metadata_txt is not None:
        cfg["paths"]["ph2_metadata_txt"] = args.ph2_metadata_txt
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.random_seed is not None:
        cfg["pipeline"]["random_seed"] = args.random_seed
    if args.balanced:
        cfg["pipeline"]["balanced"] = True

    seg_mode = cfg["segmentation"]["mode"]
    dataset_type = cfg["pipeline"].get("dataset", "isic")

    logger.info("Configuração carregada de: %s", args.config)
    logger.info(
        "dataset=%s | segmentation_mode=%s | llm=%s | max_images=%d | seed=%d",
        dataset_type, seg_mode,
        cfg["llm"]["model_name"],
        cfg["pipeline"]["max_images"],
        cfg["pipeline"]["random_seed"],
    )

    # 1. Carregar dataset
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

    # 2. Construir RAG
    retriever = build_rag(
        medical_texts_path=cfg["paths"]["medical_texts"],
        embedding_model=cfg["rag"]["embedding_model"],
        chunk_size=cfg["rag"]["chunk_size"],
        chunk_overlap=cfg["rag"]["chunk_overlap"],
    )

    # 3. Carregar modelo de segmentação (apenas no modo "model")
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

    # 4. Construir grafo de agentes
    compiled_graph = build_graph(
        retriever=retriever,
        model_name=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"],
        seg_threshold=cfg["segmentation"]["threshold"],
        segmentation_mode=seg_mode,
        seg_model=seg_model,
    )

    # 5. Executar pipeline
    output_path = run_pipeline(
        compiled_graph=compiled_graph,
        image_entries=image_entries,
        output_dir=cfg["paths"]["output_dir"],
        model_name=cfg["llm"]["model_name"],
        filename_prefix=cfg["output"]["filename_prefix"],
        date_format=cfg["output"]["date_format"],
    )

    logger.info("Pipeline concluído. Resultados em: %s", output_path)


if __name__ == "__main__":
    main()
