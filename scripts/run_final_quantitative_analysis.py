"""Análise quantitativa final da dissertação -- protocolo aprovado (ver
roteiro entregue ao pesquisador). Lê os artefatos REAIS já produzidos pela
matriz de 3.000/300 lesões (CNN, Gemini, GPT, Qwen3-VL, MedGemma),
reutilizando os módulos já existentes do avaliador estatístico da Etapa 6
(`src.evaluation.stage6_metrics`/`stage6_bootstrap`/`stage6_pairing`) e os
novos módulos desta entrega (`stage6_sources`/`stage6_operational_status`/
`stage6_pairwise`/`stage6_bootstrap_ci`/`stage6_report`).

NÃO faz nenhuma inferência, NÃO altera nenhum artefato de origem, NÃO
sorteia nada. Só lê, classifica, agrega, faz bootstrap e escreve as
tabelas/arquivos de saída descritos no protocolo (seção 6).

Sistemas/condições incompletos (hoje: GPT-5.4-nano na matriz de 3.000,
execução parcial) são incluídos no relatório com os números REAIS que
existem até o momento da execução deste script, e EXPLICITAMENTE
marcados ``provisorio=True`` em toda tabela onde aparecem -- nunca
apresentados como uma avaliação concluída (protocolo, seção 1).

Usage:
    python scripts/run_final_quantitative_analysis.py
    python scripts/run_final_quantitative_analysis.py --n-replicas 200   # smoke test rapido, NAO oficial
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_bootstrap import generate_bootstrap_replicas  # noqa: E402
from src.evaluation.stage6_bootstrap_ci import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_REPLICAS,
    bootstrap_paired_difference_cis,
    bootstrap_single_system_metric_cis,
)
from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402
from src.evaluation.stage6_operational_status import (  # noqa: E402
    ALL_OPERATIONAL_STATUSES,
    classify_operational_status,
)
from src.evaluation.stage6_pairwise import compute_pairwise_comparison  # noqa: E402
from src.evaluation.stage6_report import (  # noqa: E402
    build_confusion_matrix_row,
    build_main_table_row,
    build_operational_table_row,
    build_pairwise_row,
    ci_to_dict,
    main_table_markdown_row,
    write_csv,
    write_json,
    write_markdown_table,
)
from src.evaluation.stage6_sources import (  # noqa: E402
    OfficialRecordSelection,
    S2Availability,
    build_lesion_events_for_role_condition,
    lesion_ids_by_category_from_manifest,
    load_call_log_rows,
    load_cnn_s0_records_as_synthetic_call_records,
    load_manifest_rows,
    load_s2_availability,
    merge_gemini_reprocessed_decisao_final,
    real_by_lesion_from_manifest,
    select_official_call_records,
)

# ---------------------------------------------------------------------------
# Caminhos dos artefatos reais já existentes (nunca regenerados aqui).
# ---------------------------------------------------------------------------
MANIFEST_3000 = REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv"
MANIFEST_300 = REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset" / "local_300_s2_subset_manifest.csv"
S0_S1_S2_MANIFEST = REPO_ROOT / "experiments" / "multiagent" / "s0_s1_s2_preparation_manifest.csv"
CNN_S0_SCORES = REPO_ROOT / "experiments" / "cnn_baseline" / "s0_ham10000_eval" / "s0_cnn_scores.csv"

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_CALL_LOG = REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / GEMINI_MODEL / "call_log_main.jsonl"
GEMINI_REPROCESSED = (
    REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / GEMINI_MODEL / "reprocessing"
    / "call_log_main.decisao_final_reprocessed.jsonl"
)

GPT_MODEL = "gpt-5.4-nano"
GPT_CALL_LOG = REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / GPT_MODEL / "call_log_main.jsonl"

MEDGEMMA_MODEL = "medgemma1.5:4b-it-q4_K_M"
MEDGEMMA_CALL_LOG = (
    REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset_matrix" / "medgemma1.5_4b-it-q4_K_M"
    / "call_log_main.jsonl"
)
QWEN_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
QWEN_CALL_LOG = (
    REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset_matrix" / "qwen3-vl_8b-instruct-q4_K_M"
    / "call_log_main.jsonl"
)

OUTPUT_DIR = REPO_ROOT / "experiments" / "evaluation" / "final_quantitative_analysis"

ROLE_SUMMARY = "summary"
ROLE_VLM_DIRETO = "vlm_direto"


# ---------------------------------------------------------------------------
# Uma "configuração" avaliada = um (rótulo, universo, manifesto, eventos
# por lesão). Construída uma vez por configuração e reusada por todas as
# análises que a referenciam.
# ---------------------------------------------------------------------------
@dataclass
class Configuration:
    label: str
    universe: str  # "3000" ou "300"
    manifest_rows: list[dict[str, str]]
    stage6_status_by_lesion: dict[str, str]
    operational_status_by_lesion: dict[str, str]
    operational_detail_by_lesion: dict[str, Optional[str]]
    is_provisional: bool
    provisional_reason: Optional[str]


def _classify_configuration(
    label: str,
    universe: str,
    manifest_rows: list[dict[str, str]],
    events_by_lesion: dict[str, Optional[dict[str, Any]]],
    condition: Optional[str],
    s2_availability: Optional[S2Availability],
    is_provisional: bool,
    provisional_reason: Optional[str],
) -> Configuration:
    stage6_status: dict[str, str] = {}
    op_status: dict[str, str] = {}
    op_detail: dict[str, Optional[str]] = {}
    for row in manifest_rows:
        lesion_id = row["lesion_id"]
        record = events_by_lesion.get(lesion_id)
        structural_reason = None
        if record is None and condition == "s2" and s2_availability is not None:
            status_s2 = s2_availability.status_by_lesion.get(lesion_id, "")
            if status_s2 != "disponivel":
                structural_reason = s2_availability.reason_by_lesion.get(lesion_id) or status_s2
        classification = classify_operational_status(
            record, condition=condition, structural_unavailability_reason=structural_reason,
        )
        stage6_status[lesion_id] = classification.stage6_status
        op_status[lesion_id] = classification.operational_status
        op_detail[lesion_id] = classification.detail
    return Configuration(
        label=label, universe=universe, manifest_rows=manifest_rows,
        stage6_status_by_lesion=stage6_status, operational_status_by_lesion=op_status,
        operational_detail_by_lesion=op_detail, is_provisional=is_provisional,
        provisional_reason=provisional_reason,
    )


def restrict_configuration_to_subset(config: Configuration, new_label: str, subset_manifest_rows: list[dict[str, str]]) -> Configuration:
    """Reusa a MESMA classificação já calculada por lesão (nunca reclassifica)
    só trocando o universo/manifesto para um subconjunto (ex.: as 300 lesões
    da avaliação complementar, dentro das 3.000 já classificadas para
    Gemini/GPT). Usado exclusivamente para a Análise 6 (complementar)."""
    subset_ids = {row["lesion_id"] for row in subset_manifest_rows}
    missing = subset_ids - set(config.stage6_status_by_lesion.keys())
    if missing:
        raise ValueError(f"{new_label}: {len(missing)} lesões do subconjunto não foram classificadas na config original")
    return Configuration(
        label=new_label, universe="300", manifest_rows=subset_manifest_rows,
        stage6_status_by_lesion={k: v for k, v in config.stage6_status_by_lesion.items() if k in subset_ids},
        operational_status_by_lesion={k: v for k, v in config.operational_status_by_lesion.items() if k in subset_ids},
        operational_detail_by_lesion={k: v for k, v in config.operational_detail_by_lesion.items() if k in subset_ids},
        is_provisional=config.is_provisional, provisional_reason=config.provisional_reason,
    )


def _operational_counts(config: Configuration) -> dict[str, int]:
    counts = {s: 0 for s in ALL_OPERATIONAL_STATUSES}
    for status in config.operational_status_by_lesion.values():
        counts[status] += 1
    return counts


# ---------------------------------------------------------------------------
# Carregamento das fontes reais.
# ---------------------------------------------------------------------------
def load_vlm_official_selection(model_name: str, call_log_path: Path, reprocessed_path: Optional[Path]) -> OfficialRecordSelection:
    rows = load_call_log_rows(call_log_path)
    if reprocessed_path is not None and reprocessed_path.exists():
        rows = merge_gemini_reprocessed_decisao_final(rows, reprocessed_path)
    return select_official_call_records(rows, model_name)


def build_configs_for_vlm(
    model_label: str, model_name: str, call_log_path: Path, reprocessed_path: Optional[Path],
    manifest_rows: list[dict[str, str]], universe: str, s2_availability: S2Availability,
    conditions: tuple[str, ...], include_vlm_direto: bool, n_planned_expected: Optional[int],
) -> tuple[dict[str, Configuration], OfficialRecordSelection]:
    if not call_log_path.exists():
        raise FileNotFoundError(f"Call log ausente para {model_label}: {call_log_path}")

    selection = load_vlm_official_selection(model_name, call_log_path, reprocessed_path)
    n_covered = len(selection.record_by_key)
    is_provisional = (n_planned_expected is not None) and (n_covered < n_planned_expected)
    provisional_reason = (
        f"{n_covered}/{n_planned_expected} chamadas planejadas cobertas por tentativa oficial -- execução em andamento"
        if is_provisional else None
    )

    configs: dict[str, Configuration] = {}
    if include_vlm_direto:
        events = build_lesion_events_for_role_condition(selection, ROLE_VLM_DIRETO, "s0", manifest_rows)
        label = f"{model_label} VLM-direto S0"
        configs[label] = _classify_configuration(
            label, universe, manifest_rows, events, condition="s0", s2_availability=None,
            is_provisional=is_provisional, provisional_reason=provisional_reason,
        )
    for condition in conditions:
        events = build_lesion_events_for_role_condition(selection, ROLE_SUMMARY, condition, manifest_rows)
        label = f"{model_label} multiagente {condition.upper()}"
        configs[label] = _classify_configuration(
            label, universe, manifest_rows, events, condition=condition, s2_availability=s2_availability,
            is_provisional=is_provisional, provisional_reason=provisional_reason,
        )
    return configs, selection


def build_cnn_config(label: str, universe: str, manifest_rows: list[dict[str, str]], csv_path: Path) -> Configuration:
    records_by_lesion = load_cnn_s0_records_as_synthetic_call_records(csv_path)
    events_by_lesion = {row["lesion_id"]: records_by_lesion.get(row["lesion_id"]) for row in manifest_rows}
    return _classify_configuration(
        label, universe, manifest_rows, events_by_lesion, condition="s0", s2_availability=None,
        is_provisional=False, provisional_reason=None,
    )


# ---------------------------------------------------------------------------
# Métricas + IC por configuração.
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    config: Configuration
    indicators: Any
    ci_by_metric: dict[str, Any]


def evaluate_configuration(config: Configuration, real_by_lesion: dict[str, str], replicas: list[list[str]]) -> ConfigResult:
    universe_ids = [row["lesion_id"] for row in config.manifest_rows]
    indicators = compute_stage6_indicators(universe_ids, config.stage6_status_by_lesion, real_by_lesion)
    ci_by_metric = bootstrap_single_system_metric_cis(
        replicas, config.stage6_status_by_lesion, real_by_lesion, indicators,
    )
    return ConfigResult(config=config, indicators=indicators, ci_by_metric=ci_by_metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicas", type=int, default=N_REPLICAS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    is_official_bootstrap_config = (args.n_replicas == N_REPLICAS and args.seed == BOOTSTRAP_SEED)

    manifest_3000 = load_manifest_rows(MANIFEST_3000)
    manifest_300 = load_manifest_rows(MANIFEST_300)
    s2_availability = load_s2_availability(S0_S1_S2_MANIFEST)
    real_3000 = real_by_lesion_from_manifest(manifest_3000)
    real_300 = real_by_lesion_from_manifest(manifest_300)
    by_category_3000 = lesion_ids_by_category_from_manifest(manifest_3000)
    by_category_300 = lesion_ids_by_category_from_manifest(manifest_300)

    print(f"[1/6] Gerando réplicas bootstrap (seed={args.seed}, n={args.n_replicas}) ...")
    replicas_3000 = generate_bootstrap_replicas(by_category_3000, args.n_replicas, args.seed)
    replicas_300 = generate_bootstrap_replicas(by_category_300, args.n_replicas, args.seed)

    print("[2/6] Carregando CNN S0 (3.000 e subconjunto 300) ...")
    cnn_3000 = build_cnn_config("CNN S0", "3000", manifest_3000, CNN_S0_SCORES)
    cnn_300 = build_cnn_config("CNN S0", "300", manifest_300, CNN_S0_SCORES)

    print("[3/6] Carregando Gemini (fonte oficial reprocessada de decisao_final) ...")
    gemini_configs_3000, gemini_selection = build_configs_for_vlm(
        "Gemini", GEMINI_MODEL, GEMINI_CALL_LOG, GEMINI_REPROCESSED,
        manifest_3000, "3000", s2_availability, conditions=("s0", "s1", "s2"),
        include_vlm_direto=True, n_planned_expected=None,
    )

    print("[4/6] Carregando GPT (pode estar PARCIAL -- marcado explicitamente se incompleto) ...")
    # n_planned para a matriz completa (vlm_direto S0 + ABCD/Menzies/SPCL/
    # summary em S0/S1/S2 por lesao disponivel) e identico para qualquer
    # modelo avaliado sob a mesma matriz de 3.000 -- reaproveita a contagem
    # ja fechada do Gemini (que concluiu 100% dela) em vez de recalcular.
    n_planned_full_scale = len(gemini_selection.record_by_key)
    gpt_configs_3000, gpt_selection = build_configs_for_vlm(
        "GPT", GPT_MODEL, GPT_CALL_LOG, None,
        manifest_3000, "3000", s2_availability, conditions=("s0", "s1", "s2"),
        include_vlm_direto=True, n_planned_expected=n_planned_full_scale,
    )

    print("[5/6] Carregando Qwen3-VL/MedGemma (complementar, 300 lesões, só S2) ...")
    medgemma_configs_300, _ = build_configs_for_vlm(
        "MedGemma", MEDGEMMA_MODEL, MEDGEMMA_CALL_LOG, None,
        manifest_300, "300", s2_availability, conditions=("s2",),
        include_vlm_direto=False, n_planned_expected=None,
    )
    qwen_configs_300, _ = build_configs_for_vlm(
        "Qwen3-VL", QWEN_MODEL, QWEN_CALL_LOG, None,
        manifest_300, "300", s2_availability, conditions=("s2",),
        include_vlm_direto=False, n_planned_expected=None,
    )

    gemini_s2_300 = restrict_configuration_to_subset(
        gemini_configs_3000["Gemini multiagente S2"], "Gemini multiagente S2 (300)", manifest_300,
    )
    gpt_s2_300 = restrict_configuration_to_subset(
        gpt_configs_3000["GPT multiagente S2"], "GPT multiagente S2 (300)", manifest_300,
    )

    all_configs: dict[str, Configuration] = {}
    all_configs.update(gemini_configs_3000)
    all_configs.update(gpt_configs_3000)
    all_configs["CNN S0 (3000)"] = cnn_3000
    all_configs["CNN S0 (300)"] = cnn_300
    all_configs[gemini_s2_300.label] = gemini_s2_300
    all_configs[gpt_s2_300.label] = gpt_s2_300
    all_configs.update(medgemma_configs_300)
    all_configs.update(qwen_configs_300)

    print("[6/6] Calculando indicadores + IC bootstrap por configuração ...")
    results: dict[str, ConfigResult] = {}
    for label, config in all_configs.items():
        replicas = replicas_3000 if config.universe == "3000" else replicas_300
        real_by_lesion = real_3000 if config.universe == "3000" else real_300
        results[label] = evaluate_configuration(config, real_by_lesion, replicas)

    # -----------------------------------------------------------------
    # Seis análises (protocolo, seção 5).
    # -----------------------------------------------------------------
    pairwise_specs: list[tuple[str, str, str]] = [
        ("Analise 1 (principal 3000): Gemini multiagente S2 - CNN S0", "Gemini multiagente S2", "CNN S0 (3000)"),
        ("Analise 1 (principal 3000): GPT multiagente S2 - CNN S0", "GPT multiagente S2", "CNN S0 (3000)"),
        ("Analise 2 (organizacao 3000): Gemini multiagente S0 - Gemini VLM-direto S0", "Gemini multiagente S0", "Gemini VLM-direto S0"),
        ("Analise 2 (organizacao 3000): GPT multiagente S0 - GPT VLM-direto S0", "GPT multiagente S0", "GPT VLM-direto S0"),
        ("Analise 3 (contorno referencia 3000): Gemini multiagente S1 - S0", "Gemini multiagente S1", "Gemini multiagente S0"),
        ("Analise 3 (contorno referencia 3000): GPT multiagente S1 - S0", "GPT multiagente S1", "GPT multiagente S0"),
        ("Analise 4 (contorno predito 3000): Gemini multiagente S2 - S0", "Gemini multiagente S2", "Gemini multiagente S0"),
        ("Analise 4 (contorno predito 3000): GPT multiagente S2 - S0", "GPT multiagente S2", "GPT multiagente S0"),
        ("Analise 5 (mascara predita vs referencia 3000): Gemini multiagente S2 - S1", "Gemini multiagente S2", "Gemini multiagente S1"),
        ("Analise 5 (mascara predita vs referencia 3000): GPT multiagente S2 - S1", "GPT multiagente S2", "GPT multiagente S1"),
    ]
    complementary_order = ["CNN S0 (300)", "Gemini multiagente S2 (300)", "GPT multiagente S2 (300)", "Qwen3-VL multiagente S2", "MedGemma multiagente S2"]
    for i in range(len(complementary_order)):
        for j in range(i + 1, len(complementary_order)):
            first, second = complementary_order[i], complementary_order[j]
            pairwise_specs.append((f"Analise 6 (complementar 300): {first} - {second}", first, second))

    pairwise_rows: list[dict[str, Any]] = []
    for analysis_label, first_label, second_label in pairwise_specs:
        if first_label not in all_configs or second_label not in all_configs:
            print(f"  [AVISO] Par ausente, pulando: {analysis_label}")
            continue
        config_a, config_b = all_configs[first_label], all_configs[second_label]
        universe_rows = config_a.manifest_rows  # mesmo universo nos dois lados, por construcao
        universe_ids = [row["lesion_id"] for row in universe_rows]
        real_by_lesion = real_3000 if config_a.universe == "3000" else real_300
        replicas = replicas_3000 if config_a.universe == "3000" else replicas_300

        pairwise = compute_pairwise_comparison(
            first_label, second_label, config_a.stage6_status_by_lesion, config_b.stage6_status_by_lesion,
            real_by_lesion, universe_ids,
        )
        intersection_set = set(pairwise.intersection_lesion_ids)
        diff_ci = bootstrap_paired_difference_cis(
            replicas, intersection_set, config_a.stage6_status_by_lesion, config_b.stage6_status_by_lesion,
            real_by_lesion, pairwise.first_metrics, pairwise.second_metrics,
        )
        row = build_pairwise_row(analysis_label, pairwise, diff_ci)
        row["primeiro_provisorio"] = config_a.is_provisional
        row["segundo_provisorio"] = config_b.is_provisional
        pairwise_rows.append(row)

    # -----------------------------------------------------------------
    # Saídas.
    # -----------------------------------------------------------------
    out = args.output_dir
    main_rows_json: list[dict[str, Any]] = []
    main_rows_md: list[dict[str, str]] = []
    operational_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for label, result in sorted(results.items()):
        universe_size = len(result.config.manifest_rows)
        main_row = build_main_table_row(label, universe_size, result.indicators, result.ci_by_metric, result.config.is_provisional)
        main_rows_json.append(main_row)
        main_rows_md.append(main_table_markdown_row(main_row, result.ci_by_metric))
        operational_rows.append(build_operational_table_row(label, universe_size, _operational_counts(result.config)))
        confusion_rows.append(build_confusion_matrix_row(label, result.indicators))

    write_csv(out / "tabela_principal.csv", main_rows_json)
    write_markdown_table(out / "tabela_principal.md", main_rows_md, "Tabela principal -- análise quantitativa final")
    write_csv(out / "tabela_operacional.csv", operational_rows)
    write_csv(out / "matrizes_confusao.csv", confusion_rows)
    write_csv(out / "tabela_pareada_suplementar.csv", pairwise_rows)

    metadata = {
        "bootstrap_n_replicas": args.n_replicas,
        "bootstrap_seed": args.seed,
        "bootstrap_is_official_protocol_config": is_official_bootstrap_config,
        "min_valid_replicas_required": 4750,
        "sources": {
            "manifest_3000": str(MANIFEST_3000), "manifest_300": str(MANIFEST_300),
            "s0_s1_s2_manifest": str(S0_S1_S2_MANIFEST), "cnn_s0_scores": str(CNN_S0_SCORES),
            "gemini_call_log": str(GEMINI_CALL_LOG), "gemini_reprocessed": str(GEMINI_REPROCESSED),
            "gpt_call_log": str(GPT_CALL_LOG), "medgemma_call_log": str(MEDGEMMA_CALL_LOG),
            "qwen_call_log": str(QWEN_CALL_LOG),
        },
        "configurations": {
            label: {
                "universe": r.config.universe, "is_provisional": r.config.is_provisional,
                "provisional_reason": r.config.provisional_reason,
                "n_lesions": len(r.config.manifest_rows),
            }
            for label, r in results.items()
        },
        "multi_attempt_keys_gemini": {str(k): v for k, v in gemini_selection.multi_attempt_keys.items()},
        "multi_attempt_keys_gpt": {str(k): v for k, v in gpt_selection.multi_attempt_keys.items()},
    }
    write_json(out / "metadados_reproducao.json", metadata)
    write_json(out / "resultados_completos_nao_arredondados.json", {
        "main_table": main_rows_json, "operational_table": operational_rows,
        "confusion_matrices": confusion_rows, "pairwise_table": pairwise_rows, "metadata": metadata,
    })

    print(f"\nConcluído. Saída em: {out}")
    for label, result in sorted(results.items()):
        flag = " [PROVISÓRIO]" if result.config.is_provisional else ""
        print(f"  {label}{flag}: n_binary={result.indicators.binary_metrics.n_binary}/{len(result.config.manifest_rows)}")


if __name__ == "__main__":
    main()
