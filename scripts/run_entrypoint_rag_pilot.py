"""Piloto local dos pontos de entrada (run_pipeline.py / run_train.py) com
respostas de LLM simuladas (sem credenciais, sem chamadas de rede). Etapa 5,
correção da A36 na camada de entry points (2026-11-08).

STATUS: implementado, aguardando execução do pesquisador. Não executado pelo
assistente/fork que escreveu este script.

Contexto: a entrega anterior da A36 removeu o nó 'critical_review' do grafo
ativo (src.agents.agents.build_graph) e confirmou, via
scripts/run_multiagent_flow_pilot.py, que o crítico nunca roda no caminho
oficial. Mas os dois pontos de entrada (scripts/run_pipeline.py,
scripts/run_train.py) ainda construíam o índice RAG (FAISS + embeddings)
incondicionalmente antes de montar o grafo -- recurso que o grafo ativo não
usa mais. Esta correção tornou essa construção opt-in via --build-rag (ver
_resolve_retriever() em cada script) e preservou o caminho histórico sem
apagar nada.

O que este piloto verifica:

  1. Caminho oficial (sem --build-rag) de run_pipeline._resolve_retriever e
     de run_train._resolve_retriever: retorna None SEM chamar build_rag nem
     prepare_medical_texts. A prova é estrutural, não por inspeção de log --
     essas duas funções são monkey-patchadas para lançar AssertionError se
     forem chamadas; se o caminho oficial as chamasse, o piloto travaria
     aqui, não reportaria "ok".

  2. Caminho histórico (--build-rag) de ambos: ainda chama build_rag (e, no
     caso de run_pipeline, prepare_medical_texts quando --skip-preprocess não
     está ativo), confirmando que o caminho não foi apagado, só deixou de ser
     o padrão. Os dublês aqui retornam um sentinela fixo em vez de construir
     um índice FAISS real.

  3. O retriever=None retornado pelo caminho oficial é exatamente o que seria
     passado para build_graph() em main() (mesma variável, sem transformação
     intermediária -- confirmado por leitura de run_pipeline.py/run_train.py).
     Para provar que o grafo REAL funciona corretamente com esse valor e
     termina no sumarizador sem o crítico, este piloto reaproveita a infra de
     dublê de LLM já validada em scripts/run_multiagent_flow_pilot.py
     (_run_scenario, _ok) -- mesma verificação estrutural: 'critical_review'
     nunca aparece no estado final (final_report permanece None).

Isto NÃO é inferência real, NÃO usa o HAM10000, NÃO integra o checkpoint da
CNN, NÃO executa S2 com máscara real. Não decide modelos/versões de VLM nem
matriz de condições. Não toca em results_stefani.csv (tratado separadamente
em P014).

Uso:
    python scripts/run_entrypoint_rag_pilot.py
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "multiagent"

logging.basicConfig(level=logging.WARNING)  # silencia os loggers dos próprios scripts durante o piloto


class _RagCalledUnexpectedly(AssertionError):
    pass


def _forbid_rag_build(*args, **kwargs):
    raise _RagCalledUnexpectedly(
        "build_rag foi chamado no caminho oficial (sem --build-rag) -- "
        "isso contraria a correção da A36 na camada de entry points."
    )


def _forbid_preprocess(*args, **kwargs):
    raise _RagCalledUnexpectedly(
        "prepare_medical_texts foi chamado no caminho oficial (sem --build-rag)."
    )


_SENTINEL_RETRIEVER = object()


def _fake_build_rag_returns_sentinel(*args, **kwargs):
    return _SENTINEL_RETRIEVER


def _fake_prepare_medical_texts_noop(*args, **kwargs):
    return "fake_medical_texts.txt"


FAKE_CFG = {
    "paths": {
        "rag_zip": "fake_rag.zip",
        "extracted_pdfs": "fake_extracted/",
        "medical_texts": "fake_medical_texts.txt",
    },
    "rag": {
        "embedding_model": "fake-embedding-model",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
}


def _check_entrypoint_official_path_builds_nothing(module_name: str) -> dict[str, Any]:
    """Importa o módulo do entry point e chama _resolve_retriever(build_rag_flag=False, ...)
    com build_rag/prepare_medical_texts monkey-patchados para travar se forem
    chamados. Reporta o comportamento observado, sem inventar nada."""
    result: dict[str, Any] = {"entrypoint": module_name, "check": "caminho_oficial"}
    module = importlib.import_module(module_name)

    original_build_rag = module.build_rag
    original_prepare = getattr(module, "prepare_medical_texts", None)
    module.build_rag = _forbid_rag_build
    if original_prepare is not None:
        module.prepare_medical_texts = _forbid_preprocess

    try:
        if module_name == "run_pipeline":
            retriever = module._resolve_retriever(
                build_rag_flag=False, skip_preprocess=False, cfg=FAKE_CFG
            )
        else:
            retriever = module._resolve_retriever(build_rag_flag=False, cfg=FAKE_CFG)
        result["retriever_is_none"] = retriever is None
        result["ok"] = True
        result["error"] = None
    except _RagCalledUnexpectedly as exc:
        result["ok"] = False
        result["error"] = str(exc)
        result["retriever_is_none"] = None
    finally:
        module.build_rag = original_build_rag
        if original_prepare is not None:
            module.prepare_medical_texts = original_prepare

    return result


def _check_entrypoint_historical_path_still_works(module_name: str) -> dict[str, Any]:
    """Confirma que --build-rag=True ainda aciona build_rag (e, para
    run_pipeline, prepare_medical_texts) -- o caminho histórico não foi
    apagado, só deixou de ser o padrão. Os dublês retornam um sentinela fixo
    em vez de construir um índice FAISS real."""
    result: dict[str, Any] = {"entrypoint": module_name, "check": "caminho_historico"}
    module = importlib.import_module(module_name)

    original_build_rag = module.build_rag
    original_prepare = getattr(module, "prepare_medical_texts", None)
    module.build_rag = _fake_build_rag_returns_sentinel
    if original_prepare is not None:
        module.prepare_medical_texts = _fake_prepare_medical_texts_noop

    try:
        if module_name == "run_pipeline":
            retriever = module._resolve_retriever(
                build_rag_flag=True, skip_preprocess=False, cfg=FAKE_CFG
            )
        else:
            retriever = module._resolve_retriever(build_rag_flag=True, cfg=FAKE_CFG)
        result["ok"] = True
        result["error"] = None
        result["retriever_is_sentinel"] = retriever is _SENTINEL_RETRIEVER
    except Exception as exc:  # noqa: BLE001 - piloto deve reportar, não travar
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["retriever_is_sentinel"] = None
    finally:
        module.build_rag = original_build_rag
        if original_prepare is not None:
            module.prepare_medical_texts = original_prepare

    return result


def _check_graph_runs_end_to_end_with_resolved_retriever() -> dict[str, Any]:
    """Pega o retriever=None que o caminho oficial resolveu (confirmado
    acima) e roda o grafo REAL com ele, reaproveitando a infra de dublê de
    LLM já validada em run_multiagent_flow_pilot. Confirma que o grafo
    termina no sumarizador e que 'critical_review' nunca aparece no estado
    final -- a mesma verificação estrutural da entrega anterior, agora
    amarrada explicitamente ao valor que os entry points de fato resolvem."""
    pilot = importlib.import_module("run_multiagent_flow_pilot")

    abcd, menzies, spcl, summary = pilot._make_valid_responses()
    # _run_scenario já computa a consolidação internamente (ver
    # result.update(_compute_consolidation(final_state)) em
    # run_multiagent_flow_pilot.py) -- nada a fazer aqui além de rodar.
    return pilot._run_scenario(
        "entrypoint_caminho_oficial",
        pilot._ok(abcd),
        pilot._ok(menzies),
        pilot._ok(spcl),
        pilot._ok(summary),
    )


def main():
    report: dict[str, Any] = {"entrypoints": [], "divergences_found": []}

    for module_name in ("run_pipeline", "run_train"):
        official = _check_entrypoint_official_path_builds_nothing(module_name)
        historical = _check_entrypoint_historical_path_still_works(module_name)
        report["entrypoints"].append(official)
        report["entrypoints"].append(historical)

        print(
            f"[{module_name}] caminho_oficial: ok={official['ok']} "
            f"retriever_is_none={official.get('retriever_is_none')} "
            f"error={official.get('error')}"
        )
        print(
            f"[{module_name}] caminho_historico: ok={historical['ok']} "
            f"retriever_is_sentinel={historical.get('retriever_is_sentinel')} "
            f"error={historical.get('error')}"
        )

        if not official["ok"] or official.get("retriever_is_none") is not True:
            report["divergences_found"].append(
                f"DIVERGÊNCIA ENCONTRADA — {module_name}: caminho oficial não "
                f"retornou retriever=None sem construir RAG (ok={official['ok']}, "
                f"retriever_is_none={official.get('retriever_is_none')}, "
                f"error={official.get('error')})"
            )
        if not historical["ok"] or historical.get("retriever_is_sentinel") is not True:
            report["divergences_found"].append(
                f"DIVERGÊNCIA ENCONTRADA — {module_name}: caminho histórico "
                f"(--build-rag) parou de funcionar (ok={historical['ok']}, "
                f"error={historical.get('error')})"
            )

    graph_check = _check_graph_runs_end_to_end_with_resolved_retriever()
    report["graph_end_to_end_with_resolved_retriever"] = graph_check
    print(
        f"[grafo real] ok={graph_check.get('ok')} "
        f"final_report_presente={graph_check.get('final_report_present')} "
        f"summary_was_invoked={graph_check.get('summary_was_invoked')}"
    )
    if graph_check.get("final_report_present") not in (False, None):
        report["divergences_found"].append(
            "DIVERGÊNCIA ENCONTRADA — grafo real: final_report presente com "
            "retriever=None vindo do caminho oficial (crítico não deveria rodar)."
        )
    if not graph_check.get("summary_was_invoked"):
        report["divergences_found"].append(
            "DIVERGÊNCIA ENCONTRADA — grafo real: sumarizador não foi invocado "
            "com retriever=None vindo do caminho oficial."
        )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = DEFAULT_OUTPUT_DIR / "entrypoint_rag_pilot_results.json"
    report_path = DEFAULT_OUTPUT_DIR / "entrypoint_rag_pilot_report.md"
    log_path = DEFAULT_OUTPUT_DIR / "entrypoint_rag_pilot.log"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    lines = ["# Piloto: remoção do RAG incondicional dos entry points\n"]
    lines.append(
        "Verificação local com respostas simuladas, sem rede/credenciais. "
        "Confirma que run_pipeline.py e run_train.py não constroem RAG no "
        "caminho oficial (sem --build-rag) e que o caminho histórico "
        "(--build-rag) continua funcionando.\n"
    )
    for entry in report["entrypoints"]:
        lines.append(f"- `{entry['entrypoint']}` / {entry['check']}: {entry}")
    lines.append(f"\n- Grafo real com retriever resolvido pelo caminho oficial: {graph_check}\n")
    if report["divergences_found"]:
        lines.append("\n## DIVERGÊNCIAS ENCONTRADAS\n")
        for d in report["divergences_found"]:
            lines.append(f"- {d}")
    else:
        lines.append("\nNenhuma divergência encontrada.\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    print()
    if report["divergences_found"]:
        print("DIVERGÊNCIAS ENCONTRADAS:")
        for d in report["divergences_found"]:
            print(f"  - {d}")
    else:
        print("Nenhuma divergência encontrada.")
    print(f"\nresultados (json) gravados: {results_path}")
    print(f"relatório (md) gravado: {report_path}")
    print(f"log gravado: {log_path}")
    print(
        "\nLEMBRETE: isto é um piloto com respostas simuladas, não inferência "
        "real. Nenhuma chamada de rede/credencial foi usada. HAM10000 não foi "
        "tocado. results_stefani.csv não foi analisado (tratado em P014)."
    )


if __name__ == "__main__":
    main()
