"""Checagem PRÉVIA, SEM INFERÊNCIA, antes de liberar o piloto das 12
lesões (ver
`docs/reconstruction/26_decision_contract_official_pilot_criterion.md` e
decisão do pesquisador de seguir para o piloto após o smoke test real
do VLM-direto ter passado).

Este script NÃO chama nenhum modelo, NÃO executa nenhuma chamada a
Ollama, e NÃO modifica nenhum manifesto -- lê só o que já está gravado
em disco e recomputa/confirma, nunca assume:

  1. O manifesto `pilot12_manifest.csv` tem exatamente 12 lesões
     DISTINTAS (12 `lesion_id` distintos, 12 `image_id` distintos, uma
     linha por lesão) -- reusa
     `src.data_audit.sampling.verify_lesion_and_image_uniqueness`
     (mesma função já usada nas verificações de amostragem deste
     projeto, não uma segunda cópia).
  2. ZERO sobreposição, por `lesion_id` E por `image_id`, com a amostra
     CONGELADA de 3.000 lesões (`experiments/sampling/sample_manifest.csv`)
     -- o SHA-256 desse manifesto é conferido primeiro contra o valor
     registrado (`docs/reconstruction/10_frozen_sample_registry.md`),
     igual ao padrão já usado em `scripts/run_pilot12_sample.py`/
     `scripts/run_synthesis_control_real_image_selection.py` (mesmo
     literal, não reinventado).
  3. As entradas S0/S1/S2 previstas -- recomputa
     `src.multiagent.vlm_direct.build_planned_calls` (mesma função que
     `scripts/run_local_vlm_pilot.py` usa de verdade) sobre o manifesto
     e os 4 `image_id` designados para S0/S1 (resolvidos pela MESMA
     lógica de `scripts/run_local_vlm_pilot.py._resolve_designated_image_ids`,
     reusada, não duplicada) -- confirma o total de 92 chamadas e a
     composição exata (12 vlm_direto/S0, 48 multiagente/S2, 32
     multiagente/{S0,S1} designados).
  4. Registra, para rastreabilidade do relatório final do piloto, a
     configuração fixa desta rodada: os 2 modelos candidatos
     (`qwen3-vl:8b-instruct-q4_K_M`, `medgemma1.5:4b-it-q4_K_M`) e a
     versão exata do parser do contrato de decisão
     (`src.evaluation.explicit_decision_contract.PARSER_VERSION`) que
     será usada como critério oficial (ver doc 26) -- nunca mantida
     silenciosamente se o parser for trocado depois.

Se QUALQUER checagem falhar, o relatório registra o problema
explicitamente (`ready_for_pilot12_execution: false` e a lista de
problemas) -- este script NUNCA decide por conta própria seguir para a
execução real; é só o pesquisador quem decide, depois de ler o
relatório.

STATUS: implementado, aguardando execução do pesquisador. Nada deste
script foi executado por este agente (`feedback_no_autonomous_execution`)
-- nenhum `python`/`pytest`/`git`/`ollama` foi chamado.

Uso:

    python scripts/run_pilot12_preflight_check.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_audit import sampling  # noqa: E402
from src.evaluation.explicit_decision_contract import (  # noqa: E402
    PARSER_VERSION as DECISION_CONTRACT_PARSER_VERSION,
)
from src.multiagent import vlm_direct as vd  # noqa: E402
from src.multiagent.presentation import sha256_of_file  # noqa: E402

import run_local_vlm_pilot as pilot_mod  # noqa: E402

# Mesmo literal já usado em scripts/run_pilot12_sample.py e
# scripts/run_synthesis_control_real_image_selection.py -- registrado em
# docs/reconstruction/10_frozen_sample_registry.md, não recomputado aqui
# a partir de nenhuma outra fonte.
REGISTERED_FROZEN_MANIFEST_SHA256 = "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"

PILOT12_N = 12

# Candidatos fixos desta rodada (ver docs/reconstruction/20_local_vlm_pilot.md
# e doc 26) -- registrados aqui para o relatório do piloto, não
# redescobertos de outro lugar.
MODEL_TAGS: tuple[str, ...] = ("qwen3-vl:8b-instruct-q4_K_M", "medgemma1.5:4b-it-q4_K_M")

DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_vlm_pilot_preflight_check")


def _read_pilot12_csv(path: Path) -> list[dict[str, str]]:
    """Leitura PURA (sem levantar se a contagem estiver errada -- o
    relatório deve conseguir registrar ISSO como problema, não travar
    antes de checar o resto)."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    problems: list[str] = []
    checks: dict[str, Any] = {}

    frozen_manifest_path = Path(pilot_mod.DEFAULT_MANIFEST_PATH).parents[1] / "sampling" / "sample_manifest.csv"
    pilot12_manifest_path = Path(pilot_mod.DEFAULT_MANIFEST_PATH)
    sampling_results_path = Path(pilot_mod.DEFAULT_SAMPLING_RESULTS_JSON)

    # --- 1. Hash do manifesto congelado (sem isso, nenhuma checagem de
    #     sobreposição abaixo pode ser confiada). ---
    frozen_rows: list[dict[str, Any]] = []
    if not frozen_manifest_path.exists():
        problems.append(f"manifesto congelado não encontrado: {frozen_manifest_path}")
        frozen_hash_ok = None
        frozen_hash_actual = None
    else:
        frozen_hash_actual = sha256_of_file(frozen_manifest_path)
        frozen_hash_ok = frozen_hash_actual == REGISTERED_FROZEN_MANIFEST_SHA256
        if not frozen_hash_ok:
            problems.append(
                f"SHA-256 do manifesto congelado não corresponde ao registrado -- "
                f"esperado {REGISTERED_FROZEN_MANIFEST_SHA256}, encontrado {frozen_hash_actual}."
            )
        else:
            frozen_rows = sampling.read_sample_manifest(frozen_manifest_path)
    checks["frozen_manifest"] = {
        "path": str(frozen_manifest_path),
        "sha256_expected": REGISTERED_FROZEN_MANIFEST_SHA256,
        "sha256_actual": frozen_hash_actual,
        "sha256_matches": frozen_hash_ok,
        "n_rows": len(frozen_rows),
    }

    # --- 2. Manifesto das 12 lesões: existe, 12 linhas, distinção. ---
    pilot12_rows: list[dict[str, str]] = []
    if not pilot12_manifest_path.exists():
        problems.append(f"manifesto do piloto de 12 lesões não encontrado: {pilot12_manifest_path}")
        uniqueness: dict[str, Any] = {"ok": False, "note": "manifesto não encontrado"}
    else:
        pilot12_rows = _read_pilot12_csv(pilot12_manifest_path)
        uniqueness = sampling.verify_lesion_and_image_uniqueness(pilot12_rows, expected_n=PILOT12_N)
        if not uniqueness["ok"]:
            problems.append(f"manifesto do piloto de 12 lesões não passou a checagem de unicidade: {uniqueness}")
    checks["pilot12_manifest_uniqueness"] = uniqueness

    # --- 3. Zero sobreposição (lesion_id E image_id) com a amostra
    #     congelada. ---
    overlap_check: dict[str, Any] = {"checked": False}
    if pilot12_rows and frozen_rows:
        pilot12_lesion_ids = {r["lesion_id"] for r in pilot12_rows}
        pilot12_image_ids = {r["image_id"] for r in pilot12_rows}
        frozen_lesion_ids = {r["lesion_id"] for r in frozen_rows}
        frozen_image_ids = {r["image_id"] for r in frozen_rows}
        overlap_lesion_ids = sorted(pilot12_lesion_ids & frozen_lesion_ids)
        overlap_image_ids = sorted(pilot12_image_ids & frozen_image_ids)
        overlap_check = {
            "checked": True,
            "overlap_lesion_ids": overlap_lesion_ids,
            "overlap_image_ids": overlap_image_ids,
            "ok": len(overlap_lesion_ids) == 0 and len(overlap_image_ids) == 0,
        }
        if not overlap_check["ok"]:
            problems.append(
                f"sobreposição encontrada entre o piloto de 12 e a amostra congelada: "
                f"lesion_ids={overlap_lesion_ids} image_ids={overlap_image_ids}"
            )
    else:
        overlap_check["note"] = "não verificado -- manifesto do piloto de 12 e/ou congelado indisponível/invalido (ver problemas acima)."
    checks["overlap_with_frozen_sample"] = overlap_check

    # --- 4. Entradas S0/S1/S2 previstas -- só se os manifestos acima
    #     estiverem OK (92 chamadas planejadas exige exatamente 12 linhas
    #     válidas e 4 image_id designados distintos). ---
    planned_calls_check: dict[str, Any] = {"checked": False}
    if uniqueness.get("ok") and overlap_check.get("ok"):
        try:
            designated_image_ids = pilot_mod._resolve_designated_image_ids(pilot12_rows, sampling_results_path)
            planned_calls = vd.build_planned_calls(pilot12_rows, designated_image_ids)
            n_vlm_direto = sum(1 for c in planned_calls if c["role"] == vd.ROLE_VLM_DIRETO)
            n_s2_multiagent = sum(1 for c in planned_calls if c["condition"] == "s2" and c["role"] != vd.ROLE_VLM_DIRETO)
            n_s0_s1_designated = sum(1 for c in planned_calls if c["condition"] in ("s0", "s1") and c["role"] != vd.ROLE_VLM_DIRETO)
            planned_calls_check = {
                "checked": True,
                "designated_image_ids_for_s0_s1": designated_image_ids,
                "n_total_planned": len(planned_calls),
                "n_vlm_direto_s0": n_vlm_direto,
                "n_s2_multiagent": n_s2_multiagent,
                "n_s0_s1_designated_multiagent": n_s0_s1_designated,
                "ok": len(planned_calls) == 92 and n_vlm_direto == 12 and n_s2_multiagent == 48 and n_s0_s1_designated == 32,
            }
            if not planned_calls_check["ok"]:
                problems.append(f"composição da matriz planejada não é a esperada: {planned_calls_check}")
        except Exception as exc:  # noqa: BLE001 -- registrado, nunca propagado silenciosamente
            planned_calls_check = {"checked": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            problems.append(f"falha ao recomputar a matriz planejada: {planned_calls_check['error']}")
    else:
        planned_calls_check["note"] = "não verificado -- depende das checagens 2 e 3 passarem primeiro."
    checks["planned_s0_s1_s2_calls"] = planned_calls_check

    # --- 5. Registro de configuração (modelos + versão do contrato). ---
    checks["configuration_registered_for_this_round"] = {
        "model_tags": list(MODEL_TAGS),
        "decision_contract_parser_version": DECISION_CONTRACT_PARSER_VERSION,
        "rule": "uma tentativa por chamada, sem retry, sem fallback para v1/v2, sem ajustes de vocabulário durante a execução",
        "scope_note": "esta liberação é só para as 12 lesões -- avanço para as 3.000 é decisão separada, após revisão do piloto",
    }

    ready = len(problems) == 0
    summary = {
        "ready_for_pilot12_execution": ready,
        "problems": problems,
        "checks": checks,
    }

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nready_for_pilot12_execution: {ready}")
    print(f"resultados (json) gravados: {results_path}")


if __name__ == "__main__":
    main()
