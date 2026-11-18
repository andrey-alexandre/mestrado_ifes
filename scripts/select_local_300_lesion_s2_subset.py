"""Etapa 5 — redução de escopo da avaliação local (Qwen3-VL/MedGemma 1.5
4B) por viabilidade computacional, decisão operacional do pesquisador
enquanto aguarda retorno do orientador (ver
docs/reconstruction/30_real_lesion_evaluation_protocol.md §15 --
APROVAÇÃO DO ORIENTADOR AINDA PENDENTE).

Seleciona um subconjunto de 300 lesões DISTINTAS, estratificado
proporcionalmente pelas 7 categorias diagnósticas do HAM10000
(`src.data_audit.sampling.ELIGIBLE_DX_CATEGORIES`), com cotas pelo
método de Hamilton/maiores restos (`src.data_audit.sampling.
compute_quotas_largest_remainder` -- MESMA função já usada na amostra
congelada de 3.000, reusada aqui, nunca reimplementada), semente fixa
`LOCAL_300_SUBSET_SEED`.

Reusa a infraestrutura de amostragem existente (`src.data_audit.sampling`)
para o cálculo de cotas -- mas o SORTEIO em si é uma função NOVA
(`select_300_lesions_preserving_images`), porque `sampling.sample_lesions`
sorteia uma imagem por lesão a partir da população bruta do HAM10000 (um
segundo sorteio independente) -- aqui a imagem NUNCA é resorteada: cada
linha do subconjunto de 300 é EXATAMENTE a linha já existente no
manifesto congelado de 3.000 para aquela lesão (mesmo `image_id`, nunca
substituído).

A seleção usa SOMENTE `lesion_id`/`dx` do manifesto congelado -- NUNCA
consulta predições, status de execução (`call_log*.jsonl`), latência ou
qualidade de máscara (`s0_s1_s2_preparation_manifest.csv`) para decidir
inclusão/exclusão. Uma lesão com máscara predita vazia (`status_s2 ==
"indisponivel"`) pode ser sorteada normalmente -- a política de máscara
vazia já definida (`src.multiagent.presentation`) continua se aplicando
no plano de chamadas do executor, nunca aqui.

Este script NÃO sobrescreve silenciosamente um subconjunto já
registrado: se o manifesto de saída já existir, compara o SHA-256 do
conteúdo que seria gerado agora contra o existente -- idêntico = no-op
(reportado, nada regravado); diferente = aborta exigindo `--force`
explícito.

STATUS: implementado, NÃO executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) -- aguardando execução do
pesquisador.

Uso:
    python scripts/select_local_300_lesion_s2_subset.py
    python scripts/select_local_300_lesion_s2_subset.py --force  # só se o conteúdo mudaria de propósito
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from random import Random
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit.sampling import (  # noqa: E402
    ELIGIBLE_DX_CATEGORIES,
    compute_quotas_largest_remainder,
)
from src.multiagent import presentation as pres  # noqa: E402

DEFAULT_SOURCE_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset")

# Semente NOVA, distinta de todas as já usadas no projeto (20261106 da
# amostra congelada de 3.000, 20270203/20270204 do piloto de 12,
# 20270901 dos controles de síntese com imagem real, 20271116 da amostra
# qualitativa comum) -- governa SÓ esta seleção.
LOCAL_300_SUBSET_SEED = 20261118
TARGET_N = 300


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Sobrescreve um subconjunto já registrado SE o conteúdo gerado agora for diferente (nunca o padrão).")
    return parser


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def select_300_lesions_preserving_images(
    source_rows: list[dict[str, str]], seed: int = LOCAL_300_SUBSET_SEED, target_n: int = TARGET_N
) -> dict[str, Any]:
    """Seleciona ``target_n`` lesões distintas de ``source_rows`` (o
    manifesto congelado de 3.000 já carregado), estratificado por `dx`
    com cotas de Hamilton (`compute_quotas_largest_remainder`, reusada
    sem modificação). Para CADA lesão sorteada, a linha devolvida é
    EXATAMENTE a linha original de ``source_rows`` -- nunca um novo
    sorteio de imagem, nunca uma substituição. NÃO lê nada além de
    `lesion_id`/`dx` de ``source_rows`` -- nenhuma consulta a predição/
    status/latência/qualidade de máscara.
    """
    rows_by_lesion_id: dict[str, dict[str, str]] = {}
    for row in source_rows:
        lid = row["lesion_id"]
        if lid in rows_by_lesion_id:
            raise ValueError(f"lesion_id duplicado no manifesto de origem: {lid!r}")
        rows_by_lesion_id[lid] = row

    lesions_by_category: dict[str, list[str]] = {}
    for lid, row in rows_by_lesion_id.items():
        lesions_by_category.setdefault(row["dx"], []).append(lid)

    counts_by_category = {cat: len(lesions_by_category.get(cat, [])) for cat in ELIGIBLE_DX_CATEGORIES}
    quotas_result = compute_quotas_largest_remainder(counts_by_category, target_n)
    quotas = quotas_result["final_quotas"]

    rng = Random(seed)
    selected_lesion_ids: list[str] = []
    for category in sorted(quotas.keys()):
        quota = quotas[category]
        if quota == 0:
            continue
        pool = sorted(lesions_by_category.get(category, []))
        if quota > len(pool):
            raise ValueError(f"cota de {quota} para '{category}' excede as {len(pool)} lesões elegíveis dessa categoria no manifesto de origem.")
        chosen = rng.sample(pool, quota)
        selected_lesion_ids.extend(chosen)

    selected_lesion_ids_sorted = sorted(selected_lesion_ids)
    if len(set(selected_lesion_ids_sorted)) != target_n:
        raise AssertionError(f"esperado {target_n} lesões distintas, obtido {len(set(selected_lesion_ids_sorted))} -- erro interno de seleção.")

    selected_rows = [rows_by_lesion_id[lid] for lid in selected_lesion_ids_sorted]
    return {
        "seed": seed,
        "target_n": target_n,
        "quotas": quotas_result,
        "selected_rows": selected_rows,
    }


def _rows_to_csv_bytes(rows: list[dict[str, str]], fieldnames: list[str]) -> bytes:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue().encode("utf-8")


def _write_without_silently_overwriting(output_path: Path, content_bytes: bytes, force: bool) -> str:
    """Devolve ``"written"``, ``"unchanged_noop"`` ou levanta `SystemExit`
    se o arquivo já existir com conteúdo DIFERENTE e ``force`` for
    ``False`` -- nunca sobrescreve silenciosamente um subconjunto já
    registrado."""
    if output_path.exists():
        existing_hash = _sha256_of_file(output_path)
        new_hash = _sha256_of_bytes(content_bytes)
        if existing_hash == new_hash:
            return "unchanged_noop"
        if not force:
            raise SystemExit(
                f"ERRO: {output_path} já existe com conteúdo DIFERENTE do que seria gerado agora "
                f"(sha256 existente={existing_hash[:16]}..., novo={new_hash[:16]}...). "
                "Isto nunca é sobrescrito silenciosamente -- se a intenção é realmente substituir "
                "o subconjunto já registrado, reinvoque com --force explicitamente."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(content_bytes)
    return "written"


def main() -> None:
    args = build_arg_parser().parse_args()
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)

    print(f"Verificando manifesto de origem (SHA-256 contra o registro congelado): {source_manifest_path}")
    pres.verify_manifest_or_die(source_manifest_path)
    source_rows = pres.load_manifest_rows(source_manifest_path)
    if len(source_rows) != pres.EXPECTED_N_CASES:
        raise SystemExit(f"ERRO: {source_manifest_path} tem {len(source_rows)} linhas, esperado {pres.EXPECTED_N_CASES}.")
    source_sha256 = _sha256_of_file(source_manifest_path)
    print(f"Manifesto de origem OK: {len(source_rows)} linhas, SHA-256={source_sha256}")

    print(f"Selecionando {TARGET_N} lesões (semente {LOCAL_300_SUBSET_SEED}, cotas de Hamilton por dx)...")
    selection = select_300_lesions_preserving_images(source_rows, seed=LOCAL_300_SUBSET_SEED, target_n=TARGET_N)

    fieldnames = ["lesion_id", "image_id", "dx", "binary_target"]
    manifest_bytes = _rows_to_csv_bytes(selection["selected_rows"], fieldnames)
    manifest_path = output_dir / "local_300_s2_subset_manifest.csv"
    manifest_write_status = _write_without_silently_overwriting(manifest_path, manifest_bytes, args.force)
    print(f"Manifesto de 300 lesões: {manifest_write_status} -- {manifest_path}")

    subset_sha256 = _sha256_of_bytes(manifest_bytes)
    registry = {
        "target_n": TARGET_N,
        "seed": LOCAL_300_SUBSET_SEED,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": source_sha256,
        "source_manifest_sha256_registered": pres.REGISTERED_MANIFEST_SHA256,
        "subset_manifest_sha256": subset_sha256,
        "quotas": selection["quotas"],
        "n_selected_lesion_ids": len(selection["selected_rows"]),
        "composition_by_dx": {
            cat: sum(1 for r in selection["selected_rows"] if r["dx"] == cat) for cat in ELIGIBLE_DX_CATEGORIES
        },
        "note": (
            "Seleção reduzida de escopo por viabilidade computacional -- decisão operacional do "
            "pesquisador, aprovação do orientador AINDA PENDENTE (ver doc 30 §15). Lesões com "
            "mascara predita vazia (status_s2 indisponivel) NAO sao excluidas nem substituidas -- "
            "a politica vigente (src.multiagent.presentation) se aplica no plano de chamadas do "
            "executor, nunca nesta selecao. Selecao feita SO a partir de lesion_id/dx do manifesto "
            "congelado -- nenhuma consulta a predicao/status/latencia/qualidade de mascara."
        ),
    }
    registry_path = output_dir / "local_300_s2_subset_registry.json"
    registry_bytes = json.dumps(registry, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    registry_write_status = _write_without_silently_overwriting(registry_path, registry_bytes, args.force)
    print(f"Registro (cotas/semente/hashes): {registry_write_status} -- {registry_path}")

    print(f"\ncomposição por dx: {registry['composition_by_dx']}")
    print(f"soma = {sum(registry['composition_by_dx'].values())} (esperado {TARGET_N})")
    print(f"SHA-256 do subconjunto de 300: {subset_sha256}")


if __name__ == "__main__":
    main()
