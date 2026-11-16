"""Etapa 5 — regra reproduzível de seleção da amostra comum para a
análise qualitativa (ver docs/reconstruction/30_real_lesion_evaluation_protocol.md
§4). Decisão do pesquisador: a análise qualitativa é um RESULTADO da
avaliação (reportada ao lado das métricas/cobertura), não um filtro
aplicado depois de ver as saídas finais -- por isso esta regra é fixada
e registrada ANTES de qualquer execução real da matriz principal nas
3.000 lesões, com uma semente nova e documentada.

A amostra é escolhida sobre o MANIFESTO CONGELADO
(`experiments/sampling/sample_manifest.csv`, SHA-256 reverificado antes
de qualquer sorteio -- aborta se não bater, nunca escolhe outro arquivo
silenciosamente), estratificada por `dx` (garante presença de melanoma e
de cada subtipo não-melanoma com pelo menos 1 lesão quando disponível) e
por `binary_target`. A MESMA lista de `lesion_id` é usada para TODOS os
modelos e TODAS as condições (S0/S1/S2) -- é isto que a torna "comum
entre os modelos": o revisor olha o mesmo conjunto de lesões em cada
modelo, nunca uma amostra diferente por modelo.

Este script NÃO lê nenhuma saída de VLM/CNN -- só o manifesto e sua
composição por `dx`/`binary_target`. Roda ANTES da matriz principal.

STATUS: implementado, aguardando execução do pesquisador
(`feedback_no_autonomous_execution`) -- não executado pelo assistente
que o escreveu.

Uso:
    python scripts/select_qualitative_review_sample.py
    python scripts/select_qualitative_review_sample.py --sample-size 40
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "qualitative_review_sample")

# Hash do manifesto de 3.000 lesões congelado (seed 20261106), já usado em
# todos os scripts da Etapa 5 (S0/S1/S2, pilot12, CNN S0) -- mesma
# constante, copiada literalmente, nunca recalculada "de memória".
FROZEN_MANIFEST_SHA256 = "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"

# Semente NOVA e distinta de todas as já usadas no projeto (20261106 da
# amostra congelada, 20270203/20270204 do piloto de 12, 20270901 da
# imagem real dos controles de síntese) -- governa SÓ a seleção da
# amostra comum de revisão qualitativa.
QUALITATIVE_REVIEW_SAMPLE_SEED = 20271116

DEFAULT_SAMPLE_SIZE = 30
# Mínimo de lesões reservadas por grupo diagnóstico (dx), quando
# existirem candidatas -- garante que a amostra nunca fique concentrada
# só nos grupos majoritários (nv/bkl), mesmo com sample_size pequeno.
MIN_PER_DX_GROUP = 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    return parser


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: manifesto não encontrado: {manifest_path}")
    actual_hash = _sha256_of_file(manifest_path)
    if actual_hash != FROZEN_MANIFEST_SHA256:
        raise SystemExit(
            f"ERRO OPERACIONAL: SHA-256 de {manifest_path} não bate com o registro "
            f"congelado.\n  esperado: {FROZEN_MANIFEST_SHA256}\n  obtido:   {actual_hash}\n"
            "Nenhuma seleção foi feita -- a amostra comum de revisão qualitativa "
            "deve ser derivada só do manifesto congelado."
        )
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 3000:
        raise SystemExit(f"ERRO OPERACIONAL: {manifest_path} tem {len(rows)} linhas, esperado exatamente 3000.")
    return rows


def select_common_qualitative_sample(
    rows: list[dict[str, str]], sample_size: int, seed: int = QUALITATIVE_REVIEW_SAMPLE_SEED
) -> dict[str, Any]:
    """Regra determinística e reproduzível (mesma lista para qualquer
    execução com o mesmo `rows`/`sample_size`/`seed`):

    1. Agrupa por `dx`.
    2. Reserva `MIN_PER_DX_GROUP` lesão(ões) de CADA grupo `dx` presente
       no manifesto (sorteio determinístico via `random.Random(seed)`,
       dentro do próprio grupo, ordenado por `lesion_id` antes do
       sorteio para que a ordem de iteração do CSV nunca influencie o
       resultado).
    3. Preenche o restante da amostra (`sample_size` menos o já
       reservado) com um sorteio único, sem reposição, sobre TODAS as
       lesões ainda não reservadas, mesma semente, mesmo `random.Random`
       (não reinicializado entre os passos 2 e 3 -- um único gerador
       consumido em sequência, para que a amostra inteira dependa de um
       único estado determinístico, não de `len(grupos)` geradores
       independentes).
    4. A lista final é ordenada por `lesion_id` ascendente só para
       leitura humana do arquivo de saída -- a ORDEM de sorteio (passos
       2-3) é o que determina quais lesões entram, não a ordem de
       exibição.
    """
    by_dx: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dx.setdefault(row["dx"], []).append(row)

    rng = random.Random(seed)
    reserved_lesion_ids: list[str] = []
    reserved_detail: dict[str, list[str]] = {}
    for dx in sorted(by_dx):
        group_sorted = sorted(by_dx[dx], key=lambda r: r["lesion_id"])
        n_reserve = min(MIN_PER_DX_GROUP, len(group_sorted))
        chosen = rng.sample(group_sorted, n_reserve)
        chosen_ids = [r["lesion_id"] for r in chosen]
        reserved_lesion_ids.extend(chosen_ids)
        reserved_detail[dx] = chosen_ids

    if sample_size < len(reserved_lesion_ids):
        raise SystemExit(
            f"ERRO: --sample-size={sample_size} é menor que o número de grupos dx "
            f"presentes ({len(reserved_lesion_ids)}) -- aumente --sample-size ou "
            "reduza MIN_PER_DX_GROUP."
        )

    remaining_rows = [r for r in rows if r["lesion_id"] not in set(reserved_lesion_ids)]
    remaining_sorted = sorted(remaining_rows, key=lambda r: r["lesion_id"])
    n_extra = sample_size - len(reserved_lesion_ids)
    extra_chosen = rng.sample(remaining_sorted, n_extra) if n_extra > 0 else []
    extra_lesion_ids = [r["lesion_id"] for r in extra_chosen]

    final_lesion_ids = sorted(set(reserved_lesion_ids) | set(extra_lesion_ids))
    by_lesion_id = {r["lesion_id"]: r for r in rows}
    final_rows = [by_lesion_id[lid] for lid in final_lesion_ids]

    n_melanoma = sum(1 for r in final_rows if r["binary_target"] == "melanoma")
    n_nao_melanoma = sum(1 for r in final_rows if r["binary_target"] == "nao_melanoma")

    return {
        "seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(final_rows),
        "n_dx_groups_in_manifest": len(by_dx),
        "reserved_per_dx_group": reserved_detail,
        "extra_random_lesion_ids": sorted(extra_lesion_ids),
        "final_lesion_ids": final_lesion_ids,
        "final_rows": final_rows,
        "composition": {
            "melanoma": n_melanoma,
            "nao_melanoma": n_nao_melanoma,
            "by_dx": {dx: sum(1 for r in final_rows if r["dx"] == dx) for dx in sorted({r["dx"] for r in final_rows})},
        },
    }


# ---------------------------------------------------------------------------
# Critério de revisão qualitativa -- distingue observação/hipótese PRÓPRIA
# do revisor (permitida, desde que identificada como tal), de 3 tipos de
# erro: achado atribuído ao agente errado, parecer inventado, critério
# invertido (ex.: "avaliado e ausente" lido como "não avaliado", ou
# "parecer ausente" confundido com "lesão não tem a característica" --
# mesma distinção já aplicada em `src/multiagent/vlm_direct.py`, seção
# 12c, para os controles de síntese sintéticos).
# ---------------------------------------------------------------------------
QUALITATIVE_REVIEW_CRITERION_TEXT = (
    "OBSERVAÇÃO/HIPÓTESE PRÓPRIA DO REVISOR (permitida): qualquer leitura, "
    "suspeita ou comparação feita pelo PESQUISADOR/REVISOR sobre a saída do "
    "modelo deve ser registrada em 'reviewer_own_observation', claramente "
    "separada do que o modelo realmente escreveu -- nunca misturada ao "
    "campo 'model_output_summary'.\n"
    "ACHADO ATRIBUÍDO AO AGENTE ERRADO: a síntese/parecer atribui uma "
    "observação a um algoritmo (ABCD/Menzies/SPCL) que não a produziu, ou "
    "atribui ao modelo uma conclusão que na verdade é do revisor.\n"
    "PARECER INVENTADO: a saída menciona um critério, valor ou parecer que "
    "não existe na entrada real fornecida ao modelo para aquele caso.\n"
    "CRITÉRIO INVERTIDO: a saída troca 'avaliado e ausente' por 'não "
    "avaliado' (ou vice-versa), ou confunde 'parecer/papel ausente' com "
    "'a lesão não tem essa característica' -- mesma distinção da revisão "
    "de fidelidade dos controles sintéticos (doc 21), agora aplicada às "
    "lesões reais.\n"
    "Esta análise é reportada junto das métricas e da cobertura como "
    "RESULTADO da avaliação -- nunca usada para remover modelos ou casos "
    "depois de ver os resultados finais (decisão do pesquisador, ver "
    "docs/reconstruction/30_real_lesion_evaluation_protocol.md)."
)


def build_qualitative_review_template(final_rows: list[dict[str, str]], model_names: list[str]) -> dict[str, Any]:
    """Um item por (lesão, modelo) -- o revisor preenche depois que a
    matriz principal tiver rodado para os modelos listados. Os campos
    vazios (`model_output_summary`, vereditos) nunca são preenchidos
    automaticamente por este script."""
    items = []
    for row in final_rows:
        for model_name in model_names:
            items.append({
                "lesion_id": row["lesion_id"],
                "image_id": row["image_id"],
                "dx": row["dx"],
                "binary_target": row["binary_target"],
                "model_name": model_name,
                "model_output_summary": "",
                "reviewer_own_observation": "",
                "finding_misattributed_to_wrong_agent": None,
                "fabricated_parecer_detected": None,
                "inverted_criterion_detected": None,
                "reviewer_notes": "",
            })
    return {
        "rule_version": "qualitative_review_v1_2026_11_16",
        "criterion": QUALITATIVE_REVIEW_CRITERION_TEXT,
        "models": model_names,
        "n_lesions": len(final_rows),
        "n_items": len(items),
        "items": items,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(Path(args.manifest_path))
    selection = select_common_qualitative_sample(rows, args.sample_size)

    results_path = output_dir / "qualitative_review_sample_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2, default=str)

    manifest_csv_path = output_dir / "qualitative_review_sample_manifest.csv"
    with open(manifest_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lesion_id", "image_id", "dx", "binary_target"])
        writer.writeheader()
        for row in selection["final_rows"]:
            writer.writerow({k: row[k] for k in ("lesion_id", "image_id", "dx", "binary_target")})

    from src.multiagent.vlm_direct import MAIN_EVALUATION_MODELS  # noqa: E402

    template = build_qualitative_review_template(selection["final_rows"], list(MAIN_EVALUATION_MODELS))
    template_path = output_dir / "qualitative_review_template.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2, default=str)

    print(f"amostra comum selecionada: {selection['sample_size_actual']} lesões (semente {selection['seed']})")
    print(f"composição: {selection['composition']}")
    print(f"resultados: {results_path}")
    print(f"manifesto CSV: {manifest_csv_path}")
    print(f"template de revisão qualitativa ({len(MAIN_EVALUATION_MODELS)} modelos x {selection['sample_size_actual']} lesões = {template['n_items']} itens): {template_path}")


if __name__ == "__main__":
    main()
