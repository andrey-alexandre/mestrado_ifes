"""Verification of the stage-6 statistical evaluator with FABRICATED data
only -- no real HAM10000 metadata, no real pipeline result file, no CNN
checkpoint, no VLM call of any kind. Demonstrates, end to end:

  1. Pairing by lesion_id (not image path) across simulated systems/
     conditions, with missing / duplicate / orphan cases all surfaced
     explicitly, never silently dropped.
  2. The four denominator-distinct indicators (5 binary metrics /
     coverage-binária / taxa de saída válida / acertos-sobre-total), on
     simulated systems that each fail to diagnose a DIFFERENT subset of
     cases -- confirms no cross-contamination between systems.
  3. Stratified, paired bootstrap: quota preservation per replica, and the
     SAME replica (same resampled lesion_ids, with repetition) reused
     across every simulated system.

STATUS: DADOS SIMULADOS -- NAO E UMA AVALIACAO REAL. Nao le HAM10000, nao
usa checkpoint de CNN, nao chama nenhum VLM.

Usage:
    python scripts/verify_stage6_evaluator.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.case_status import classify_case_status  # noqa: E402
from src.evaluation.stage6_bootstrap import (  # noqa: E402
    generate_bootstrap_replicas,
    replica_category_counts,
)
from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402
from src.evaluation.stage6_pairing import (  # noqa: E402
    ManifestIndex,
    build_manifest_index,
    pair_results_by_lesion,
)

OUTPUT_DIR = REPO_ROOT / "experiments" / "evaluation"
N_BOOTSTRAP_REPLICAS_DEMO = 20  # so para demonstracao -- NAO e o B final da analise
BOOTSTRAP_SEED = 20261108


def _fabricate_manifest_rows() -> list[dict[str, str]]:
    """60 lesoes fabricadas, 3 categorias 'dx' (nv=30, mel=20, bkl=10) --
    cotas deliberadamente desiguais para exercitar a preservacao de cota
    por categoria. 'mel' -> binary_target 'melanoma'; 'nv'/'bkl' ->
    'nao_melanoma' (mesma convencao do protocolo real)."""
    rows = []
    counts = {"nv": 30, "mel": 20, "bkl": 10}
    i = 0
    for dx, n in counts.items():
        binary_target = "melanoma" if dx == "mel" else "nao_melanoma"
        for _ in range(n):
            i += 1
            rows.append({
                "lesion_id": f"HAM_{i:07d}",
                "image_id": f"ISIC_{i:07d}",
                "dx": dx,
                "binary_target": binary_target,
            })
    return rows


def _event(diagnostico: str | None) -> dict[str, Any]:
    if diagnostico is None:
        return {"validation": None}
    return {"validation": {"diagnostico": diagnostico, "justificativa": "simulado", "recomendacoes": "simulado"}}


def _fabricate_system_results(
    manifest_index: ManifestIndex,
    *,
    ausente_lesion_ids: set[str],
    indeterminado_lesion_ids: set[str],
    malformado_lesion_ids: set[str],
    duplicate_lesion_id: str,
    include_orphan: bool,
    path_prefix: str,
) -> dict[str, Any]:
    """Builds a path->event dict for one simulated system/condition.
    Correctness follows the manifest's binary_target exactly, EXCEPT for
    the lesion_ids passed in ``*_lesion_ids`` sets, which get the stated
    special status instead -- deliberately different per call site so
    different simulated systems miss/mishandle different cases."""
    results: dict[str, Any] = {}
    for lesion_id in manifest_index.lesion_ids:
        image_id = next(
            iid for iid, lid in manifest_index.image_id_to_lesion_id.items() if lid == lesion_id
        )
        path = f"{path_prefix}/{image_id}.jpg"

        if lesion_id in ausente_lesion_ids:
            continue  # omitido inteiramente -- "ausente" por falta de registro
        if lesion_id in malformado_lesion_ids:
            results[path] = _event("resposta livre fora do padrao esperado")
            continue
        if lesion_id in indeterminado_lesion_ids:
            results[path] = _event("Lesão Indeterminada")
            continue

        real = manifest_index.lesion_id_to_binary_target[lesion_id]
        diagnostico = "Suspeita de Melanoma" if real == "melanoma" else "Lesão Benigna"
        results[path] = _event(diagnostico)

    # Duplicata: um segundo caminho, prefixo diferente, mesmo image_id.
    dup_image_id = next(
        iid for iid, lid in manifest_index.image_id_to_lesion_id.items() if lid == duplicate_lesion_id
    )
    results[f"{path_prefix}/outra_pasta/{dup_image_id}.jpg"] = results[f"{path_prefix}/{dup_image_id}.jpg"]

    if include_orphan:
        results[f"{path_prefix}/ISIC_9999999.jpg"] = _event("Lesão Benigna")

    return results


def run_verification() -> dict[str, Any]:
    manifest_rows = _fabricate_manifest_rows()
    manifest_index = build_manifest_index(manifest_rows)
    all_lesion_ids = manifest_index.lesion_ids

    # Sistema A: perde 3 lesoes especificas (ausentes), 2 indeterminadas, 1 malformada.
    system_a_results = _fabricate_system_results(
        manifest_index,
        ausente_lesion_ids=set(all_lesion_ids[0:3]),
        indeterminado_lesion_ids=set(all_lesion_ids[3:5]),
        malformado_lesion_ids=set(all_lesion_ids[5:6]),
        duplicate_lesion_id=all_lesion_ids[10],
        include_orphan=True,
        path_prefix="/sim/sistema_a_S0",
    )
    # Sistema B: perde um subconjunto DIFERENTE (nao o mesmo da A) -- deliberado.
    system_b_results = _fabricate_system_results(
        manifest_index,
        ausente_lesion_ids=set(all_lesion_ids[20:25]),
        indeterminado_lesion_ids=set(all_lesion_ids[30:31]),
        malformado_lesion_ids=set(),
        duplicate_lesion_id=all_lesion_ids[40],
        include_orphan=False,
        path_prefix="/sim/sistema_b_S1",
    )

    systems = {"sistema_a_S0": system_a_results, "sistema_b_S1": system_b_results}

    report: dict[str, Any] = {"status": "DADOS SIMULADOS -- NAO E UMA AVALIACAO REAL", "systems": {}}

    status_by_lesion_by_system: dict[str, dict[str, str]] = {}
    for system_name, results_by_path in systems.items():
        pairing = pair_results_by_lesion(manifest_index, results_by_path)
        status_by_lesion = {
            lesion_id: classify_case_status(event)
            for lesion_id, event in pairing.lesion_id_to_event.items()
        }
        status_by_lesion_by_system[system_name] = status_by_lesion

        indicators = compute_stage6_indicators(
            expected_lesion_ids=all_lesion_ids,
            status_by_lesion=status_by_lesion,
            real_by_lesion=manifest_index.lesion_id_to_binary_target,
        )

        report["systems"][system_name] = {
            "pairing": {
                "n_missing": len(pairing.missing_lesion_ids),
                "missing_lesion_ids": pairing.missing_lesion_ids,
                "duplicate_lesion_ids": pairing.duplicate_lesion_ids,
                "orphan_paths": pairing.orphan_paths,
            },
            "indicators": asdict(indicators),
        }
        print(
            f"[{system_name}] ausentes={len(pairing.missing_lesion_ids)} "
            f"duplicados={len(pairing.duplicate_lesion_ids)} orfaos={len(pairing.orphan_paths)} "
            f"indeterminados={indicators.n_indeterminado} "
            f"acc_binaria={indicators.binary_metrics.accuracy} "
            f"acc_sobre_total={indicators.accuracy_over_total}"
        )

    # Bootstrap: mesma replica reaproveitada entre os dois sistemas simulados.
    replicas = generate_bootstrap_replicas(
        manifest_index.lesion_ids_by_category, n_replicas=N_BOOTSTRAP_REPLICAS_DEMO, seed=BOOTSTRAP_SEED
    )
    original_category_counts = {
        dx: len(ids) for dx, ids in manifest_index.lesion_ids_by_category.items()
    }
    lesion_id_to_category = {
        lesion_id: dx for dx, ids in manifest_index.lesion_ids_by_category.items() for lesion_id in ids
    }

    bootstrap_report: dict[str, Any] = {
        "n_replicas_demo": N_BOOTSTRAP_REPLICAS_DEMO,
        "nota": "numero de replicas so para demonstracao, NAO e o B final da analise definitiva.",
        "original_category_counts": original_category_counts,
        "quota_preserved_every_replica": True,
        "same_replica_reused_across_systems": True,
        "per_replica": [],
    }
    for i, replica in enumerate(replicas):
        counts = replica_category_counts(replica, lesion_id_to_category)
        quota_ok = all(counts.get(dx, 0) == n for dx, n in original_category_counts.items())
        if not quota_ok:
            bootstrap_report["quota_preserved_every_replica"] = False

        per_system_indicators = {}
        for system_name, status_by_lesion in status_by_lesion_by_system.items():
            ind = compute_stage6_indicators(
                expected_lesion_ids=replica,
                status_by_lesion=status_by_lesion,
                real_by_lesion=manifest_index.lesion_id_to_binary_target,
            )
            per_system_indicators[system_name] = {
                "accuracy_over_total": ind.accuracy_over_total,
                "binary_accuracy": ind.binary_metrics.accuracy,
            }

        if i < 3:  # só os 3 primeiros no relatório, para não inflar o JSON
            bootstrap_report["per_replica"].append({
                "replica_index": i,
                "category_counts": dict(counts),
                "quota_preserved": quota_ok,
                "replica_len": len(replica),
                "per_system": per_system_indicators,
            })

    # Confirmacao explicita: a mesma lista (com repeticoes) foi usada nos dois sistemas,
    # nao resorteada independentemente -- comparamos a replica usada, nao so o resultado.
    bootstrap_report["replica_identity_check"] = (
        "A mesma lista replicas[i] (com repeticoes) foi passada como expected_lesion_ids "
        "para TODOS os sistemas em cada replica -- nao houve resorteio independente por "
        "sistema. Verificado por construcao do loop acima, nao apenas por inspecao do resultado."
    )

    report["bootstrap"] = bootstrap_report
    print(
        f"[bootstrap] {N_BOOTSTRAP_REPLICAS_DEMO} replicas geradas -- "
        f"cota preservada em todas: {bootstrap_report['quota_preserved_every_replica']}"
    )

    return report


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Verificação do avaliador estatístico (Etapa 6) — dados simulados",
        "",
        f"**{report['status']}**",
        "",
    ]
    for system_name, data in report["systems"].items():
        p = data["pairing"]
        ind = data["indicators"]
        bm = ind["binary_metrics"]
        lines += [
            f"## Sistema simulado: `{system_name}`",
            "",
            "### Pareamento",
            f"- Lesões ausentes ({p['n_missing']}): `{p['missing_lesion_ids']}`",
            f"- Lesões duplicadas: `{p['duplicate_lesion_ids']}`",
            f"- Caminhos órfãos (sem linha no manifesto): `{p['orphan_paths']}`",
            "",
            "### Indicadores",
            f"- n_total={ind['n_total']} positivo={ind['n_positivo']} negativo={ind['n_negativo']} "
            f"indeterminado={ind['n_indeterminado']} ausente={ind['n_ausente']} "
            f"nao_reconhecido={ind['n_nao_reconhecido']}",
            f"- 5 métricas binárias (n_binary={bm['n_binary']}): accuracy={bm['accuracy']} "
            f"precision={bm['precision']} sensitivity={bm['sensitivity']} "
            f"specificity={bm['specificity']} f1={bm['f1']}",
            f"- Cobertura binária: {ind['coverage_binary']}",
            f"- Taxa de saída válida: {ind['valid_output_rate']}",
            f"- Acurácia sobre o total: {ind['accuracy_over_total']}",
            f"- Quebra de indeterminados por classe real: `{ind['indeterminado_breakdown']['by_real_class']}`",
            "",
        ]
    bs = report["bootstrap"]
    lines += [
        "## Bootstrap estratificado e pareado",
        "",
        f"- Réplicas de demonstração: {bs['n_replicas_demo']} ({bs['nota']})",
        f"- Cotas originais por categoria (`dx`): `{bs['original_category_counts']}`",
        f"- Cota preservada em TODAS as réplicas: {bs['quota_preserved_every_replica']}",
        f"- Mesma réplica reaproveitada entre sistemas: {bs['same_replica_reused_across_systems']}",
        f"- {bs['replica_identity_check']}",
        "",
        "### Amostra das 3 primeiras réplicas",
        "",
    ]
    for r in bs["per_replica"]:
        lines.append(
            f"- réplica {r['replica_index']}: tamanho={r['replica_len']} "
            f"cotas={r['category_counts']} cota_preservada={r['quota_preserved']} "
            f"por_sistema={r['per_system']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== VERIFICAÇÃO (DADOS SIMULADOS) — avaliador estatístico Etapa 6 ===")
    report = run_verification()

    results_path = OUTPUT_DIR / "stage6_verification_results.json"
    report_path = OUTPUT_DIR / "stage6_verification_report.md"
    log_path = OUTPUT_DIR / "stage6_verification.log"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_render_report_md(report))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    print(f"\nresultados (json) gravados: {results_path}")
    print(f"relatorio (md) gravado: {report_path}")
    print(f"log gravado: {log_path}")
    print("\nLEMBRETE: dados simulados -- nao e uma avaliacao real.")


if __name__ == "__main__":
    main()
