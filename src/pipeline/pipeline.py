"""Loop principal de inferência: itera sobre imagens e coleta resultados dos agentes."""
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# (image_path, lesion_size_mm, target_label, seg_path_or_None)
ImageEntry = Tuple[str, float, int, Optional[str]]


def run_pipeline(
    compiled_graph,
    image_entries: List[ImageEntry],
    output_dir: str,
    model_name: str,
    filename_prefix: str = "llm_result",
    date_format: str = "%Y%m%d",
) -> str:
    """Executa o pipeline de inferência sobre uma lista de imagens.

    Para cada imagem, transmite o estado pelo grafo LangGraph, serializa os
    resultados estruturados e salva progressivamente em um arquivo JSON.

    Args:
        compiled_graph: Grafo LangGraph compilado (saída de build_graph).
        image_entries: Lista de (image_path, lesion_size_mm, target_label, seg_path_or_None).
                       seg_path é None no modo "model" e o caminho da máscara no modo "precomputed".
        output_dir: Diretório de destino para o arquivo JSON de resultados.
        model_name: Nome do LLM (usado no nome do arquivo de saída).
        filename_prefix: Prefixo do nome do arquivo JSON.
        date_format: Formato de data para o nome do arquivo.

    Returns:
        Caminho absoluto do arquivo JSON gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.date.today().strftime(date_format)
    output_filename = f"{filename_prefix}_{model_name}_{date_str}.json"
    output_path = str(Path(output_dir) / output_filename)

    result_dict = {}

    for j, (path, lesion_size, real_value, seg_path) in enumerate(image_entries):
        start_time = time.time()
        logger.info("%dª imagem iniciou: %s", j + 1, path)

        # Montar estado inicial; seg_path só é incluído no modo "precomputed"
        initial_state = {"image_path": path, "lesion_size": lesion_size}
        if seg_path is not None:
            initial_state["seg_path"] = seg_path

        # Transmitir o estado pelo grafo; stream_mode="values" retorna o estado
        # completo após cada nó
        for event in compiled_graph.stream(initial_state, stream_mode="values"):
            # Remover dados de imagem em Base64 para não inflar o JSON de saída
            for field in ["image_data", "seg_image_data"]:
                event.pop(field, None)
            event["real_value"] = int(real_value)
            result_dict[path] = event

        # Serializar objetos Pydantic para dict (necessário para json.dump)
        # Quando model_name == "llava:13b", os campos já são strings
        for field in ["diagnosis_abcd", "diagnosis_menzies", "diagnosis_spcl", "validation"]:
            val = result_dict[path].get(field)
            if val is not None and hasattr(val, "model_dump"):
                result_dict[path][field] = val.model_dump()

        # Salvar progressivamente após cada imagem para evitar perda em caso de erro
        with open(output_path, "w", encoding="utf-8") as arquivo:
            json.dump(result_dict, arquivo, indent=4, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        logger.info(
            "Tempo de análise da %dª imagem: %.4f segundos", j + 1, elapsed_time
        )

    logger.info("Pipeline concluído. Resultados salvos em: %s", output_path)
    return output_path
