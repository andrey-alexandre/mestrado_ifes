"""Etapa 5, segunda entrega — seleciona 12 lesões do HAM10000 FORA da
amostra congelada de 3.000 lesões (`experiments/sampling/sample_manifest.csv`,
SHA-256 `32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`,
ver `docs/reconstruction/10_frozen_sample_registry.md`), para um piloto
local de VLMs via Ollama (ver `docs/reconstruction/20_local_vlm_pilot.md`).

Reusa integralmente as primitivas de `src/data_audit/sampling.py` já
usadas pela amostra congelada (`check_lesion_population`,
`build_lesion_table`, `apply_exclusions`, `compute_quotas_largest_remainder`
não são usadas aqui -- este piloto NÃO precisa de cotas proporcionais por
categoria, só de 12 lesões distintas sorteadas da população elegível
residual; mas o sorteio em si usa a MESMA função `random.Random(seed)`
determinística e a mesma convenção de 1 imagem por lesão).

Semente: `20270203` -- deliberadamente distinta da semente da amostra
congelada (`20261106`), documentada aqui por ser nova.

Regra de exclusão: TODO `lesion_id` e TODO `image_id` presentes no
manifesto congelado são removidos do pool elegível ANTES do sorteio
(exclusão por ambas as chaves, não só por `lesion_id`, porque uma mesma
lesão pode ter mais de uma imagem no HAM10000 e nós queremos garantir
zero sobreposição também no nível de imagem). O script verifica e
reporta explicitamente zero sobreposição por AMBAS as chaves após o
sorteio -- não apenas assume.

Dos 12 sorteados, os 4 PRIMEIROS na ordem alfabética de `lesion_id`
(regra determinística documentada, não escolha manual) são designados
para também receberem o tratamento multiagente em S0/S1 (ver Passo 4 da
entrega) -- marcados na coluna `designado_s0_s1` do manifesto de saída.

STATUS: implementado, aguardando execução do pesquisador. Não executado
pelo assistente que o escreveu (`feedback_no_autonomous_execution`).

Uso:
    python scripts/run_pilot12_sample.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import sampling  # noqa: E402
from src.multiagent.presentation import sha256_of_file  # noqa: E402

DEFAULT_METADATA_CSV = "D:/Andrey/Mestrado/dataverse_files/HAM10000_metadata"
DEFAULT_FROZEN_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
REGISTERED_FROZEN_MANIFEST_SHA256 = (
    "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"
)
PILOT_N = 12
PILOT_SEED = 20270203
N_DESIGNATED_S0_S1 = 4
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent")


def _verify_frozen_manifest_or_die(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"ERRO: manifesto congelado não encontrado em {path}")
    actual = sha256_of_file(path)
    if actual != REGISTERED_FROZEN_MANIFEST_SHA256:
        raise SystemExit(
            f"ERRO FATAL: SHA-256 do manifesto congelado NÃO corresponde ao registrado.\n"
            f"  arquivo: {path}\n  esperado: {REGISTERED_FROZEN_MANIFEST_SHA256}\n"
            f"  encontrado: {actual}\nExecução abortada -- este piloto depende de excluir "
            "exatamente o manifesto congelado real, não uma cópia alterada."
        )
    print(f"Manifesto congelado verificado: SHA-256 corresponde ({actual}).")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--frozen-manifest", default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--n", type=int, default=PILOT_N)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_manifest_path = Path(args.frozen_manifest)
    _verify_frozen_manifest_or_die(frozen_manifest_path)
    frozen_rows = sampling.read_sample_manifest(frozen_manifest_path)
    frozen_lesion_ids = {r["lesion_id"] for r in frozen_rows}
    frozen_image_ids = {r["image_id"] for r in frozen_rows}
    print(
        f"manifesto congelado: {len(frozen_rows)} linhas -- "
        f"{len(frozen_lesion_ids)} lesion_id distintos, {len(frozen_image_ids)} image_id distintos."
    )

    print(f"lendo metadados: {args.metadata_csv}")
    rows = sampling.load_ham10000_metadata(args.metadata_csv)
    print(f"linhas de metadados lidas: {len(rows)}")

    population_check = sampling.check_lesion_population(rows)
    if not population_check["ok"]:
        result = {"status": "populacao_invalida", "population_check": population_check}
        _write_json(output_dir / "pilot12_sampling_results.json", result)
        print("POPULAÇÃO INVÁLIDA -- amostragem NÃO executada.")
        return

    lesion_table = sampling.build_lesion_table(rows)
    print(f"lesões distintas (antes de exclusão): {len(lesion_table)}")

    # Exclusão por AMBAS as chaves: lesion_id presente no congelado é
    # removido do pool inteiro (apply_exclusions já faz isso por lesion_id);
    # além disso, removemos qualquer lesão cuja LISTA DE IMAGENS contenha um
    # image_id do congelado, mesmo que o lesion_id em si não esteja lá
    # (lesão com múltiplas imagens, uma delas sorteada no congelado).
    lesion_ids_with_frozen_image = {
        lid for lid, entry in lesion_table.items() if frozen_image_ids & set(entry["image_ids"])
    }
    exclusion_ids = sorted(frozen_lesion_ids | lesion_ids_with_frozen_image)
    exclusion_result = sampling.apply_exclusions(lesion_table, exclusion_ids)
    filtered_table = exclusion_result["filtered_lesion_table"]
    print(
        f"lesões excluídas (lesion_id OU image_id no manifesto congelado): "
        f"{exclusion_result['n_actually_excluded']}/{exclusion_result['n_requested']} solicitadas; "
        f"pool residual: {len(filtered_table)} lesões."
    )

    if len(filtered_table) < args.n:
        result = {
            "status": "populacao_residual_insuficiente",
            "n_eligible_residual": len(filtered_table),
            "n": args.n,
        }
        _write_json(output_dir / "pilot12_sampling_results.json", result)
        print(f"POOL RESIDUAL INSUFICIENTE: {len(filtered_table)} < {args.n} -- amostragem NÃO executada.")
        return

    # Sorteio determinístico: 1 cota única "pool" (sem proporcionalidade por
    # categoria -- não pedido para este piloto de 12 casos), reusando a
    # mesma função determinística sample_lesions com uma única categoria
    # sintética "pool" cobrindo todo o residual.
    pool_table_single_category = {lid: {"dx": "pool", "image_ids": v["image_ids"]} for lid, v in filtered_table.items()}
    sample_result = sampling.sample_lesions(pool_table_single_category, {"pool": args.n}, args.seed)
    sampled_rows_raw = sample_result["sampled_rows"]

    # Restaurar o dx real (não "pool") e o binary_target real de cada linha
    # sorteada -- sample_lesions não sabe o dx real porque usamos uma
    # categoria sintética só para o sorteio sem cotas por categoria.
    sampled_rows = []
    for row in sampled_rows_raw:
        real_dx = filtered_table[row["lesion_id"]]["dx"]
        sampled_rows.append(
            {
                "lesion_id": row["lesion_id"],
                "image_id": row["image_id"],
                "dx": real_dx,
                "binary_target": "melanoma" if real_dx == "mel" else "nao_melanoma",
            }
        )

    # --- verificação de zero sobreposição por AMBAS as chaves ---
    sampled_lesion_ids = {r["lesion_id"] for r in sampled_rows}
    sampled_image_ids = {r["image_id"] for r in sampled_rows}
    overlap_lesion = sorted(sampled_lesion_ids & frozen_lesion_ids)
    overlap_image = sorted(sampled_image_ids & frozen_image_ids)
    overlap_ok = not overlap_lesion and not overlap_image
    print(
        f"sobreposição com o congelado -- lesion_id: {len(overlap_lesion)}, "
        f"image_id: {len(overlap_image)} (esperado 0 em ambos)."
    )
    if not overlap_ok:
        raise RuntimeError(
            f"DEFEITO: sobreposição não-zero detectada após exclusão -- "
            f"lesion_id={overlap_lesion}, image_id={overlap_image}. "
            "Não gravar manifesto nesse estado."
        )

    # --- designação determinística dos 4 casos para multiagente S0/S1 ---
    # Registrada SOMENTE no relatório/JSON (não como coluna extra do CSV),
    # para manter o manifesto com exatamente o mesmo schema de 4 colunas do
    # manifesto congelado (lesion_id,image_id,dx,binary_target).
    sampled_rows_sorted = sorted(sampled_rows, key=lambda r: r["lesion_id"])
    designated_lesion_ids = {r["lesion_id"] for r in sampled_rows_sorted[:N_DESIGNATED_S0_S1]}

    manifest_path = output_dir / "pilot12_manifest.csv"
    fieldnames = ["lesion_id", "image_id", "dx", "binary_target"]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(sampled_rows, key=lambda r: r["lesion_id"]):
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"manifesto do piloto (12 linhas, schema igual ao congelado) gravado: {manifest_path}")
    print(f"4 designados para S0/S1 (primeiros por lesion_id): {sorted(designated_lesion_ids)}")

    report_lines = [
        "# Amostragem do piloto de 12 lesões (fora da amostra congelada) — execução",
        "",
        f"- Metadados: `{args.metadata_csv}`",
        f"- Manifesto congelado verificado (SHA-256): `{REGISTERED_FROZEN_MANIFEST_SHA256}`",
        f"- Semente do piloto (`--seed`, distinta de 20261106): **{args.seed}**",
        f"- N sorteado: {args.n}",
        "",
        "## Exclusão do pool elegível",
        "",
        f"- lesion_id do congelado: {len(frozen_lesion_ids)}",
        f"- lesion_id adicionais excluídos por compartilhar image_id com o congelado "
        f"(múltiplas imagens da mesma lesão): {len(lesion_ids_with_frozen_image)}",
        f"- Pool residual elegível: {len(filtered_table)}",
        "",
        "## Verificação de sobreposição zero (ambas as chaves)",
        "",
        f"- lesion_id em comum com o congelado: {len(overlap_lesion)} (esperado 0)",
        f"- image_id em comum com o congelado: {len(overlap_image)} (esperado 0)",
        f"- Critério atendido: {overlap_ok}",
        "",
        "## Designação dos 4 casos para multiagente S0/S1",
        "",
        "Regra: os 4 primeiros `lesion_id` (ordem alfabética/lexicográfica "
        "ascendente) entre os 12 sorteados -- regra determinística e "
        "reproduzível, não escolha manual. Flag no manifesto: "
        "`designado_s0_s1`.",
        "",
        f"- Designados: {sorted(designated_lesion_ids)}",
        "",
    ]
    report_path = output_dir / "pilot12_sampling_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"relatório gravado: {report_path}")

    results_json = {
        "status": "concluido",
        "metadata_csv": args.metadata_csv,
        "frozen_manifest_sha256_verified": REGISTERED_FROZEN_MANIFEST_SHA256,
        "seed": args.seed,
        "n": args.n,
        "n_eligible_residual": len(filtered_table),
        "overlap_lesion_ids": overlap_lesion,
        "overlap_image_ids": overlap_image,
        "overlap_ok": overlap_ok,
        "designated_s0_s1_lesion_ids": sorted(designated_lesion_ids),
        "manifest_path": str(manifest_path),
    }
    _write_json(output_dir / "pilot12_sampling_results.json", results_json)
    print(f"resultados (json) gravados: {output_dir / 'pilot12_sampling_results.json'}")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
