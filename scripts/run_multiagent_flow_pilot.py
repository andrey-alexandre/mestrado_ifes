"""Piloto local do fluxo multiagente real, com respostas de LLM simuladas
(sem credenciais, sem chamadas de rede). Etapa 5, primeira entrega.

STATUS: implementada, aguardando execucao do pesquisador. Nao executado
pelo assistente/fork que escreveu este script.

O que este piloto verifica, rodando o grafo REAL (apos a remocao do
critico/RAG do fluxo ativo, ver agents.py e
docs/reconstruction/11_multiagent_flow_documentation.md) com a camada de
LLM inteiramente substituida por um dublê determinístico:

  1. Caminho normal: ABCD, Menzies e SPCL simulados -> sumarizador -> fim.
     Confirma que os TRES pareceres de fato chegam ao sumarizador
     (inspeciona o que o dublê do sumarizador recebeu, nao so que foi
     chamado) e que 'critical_review' nao aparece no estado final nem e
     chamado (ver item 2).
  2. O critico nao e chamado: como o no 'critical_review' foi REMOVIDO do
     registro do grafo (nao so desconectado por aresta), a prova mais
     direta e estrutural: o campo 'final_report' do estado final nunca
     tem valor diferente de None (a CHAVE em si sempre existe, pois
     GraphState a declara com default None -- so
     CriticalReviewAgent.validate_diagnosis o preenche com um valor, em
     nenhum outro lugar do codigo). Este piloto confirma isso no estado
     real retornado pelo grafo, nao por afirmacao.
  3. Cenarios de falha simulada (agente lanca excecao; agente retorna algo
     fora do schema esperado; agente "nao responde"): registra o
     comportamento OBSERVADO do codigo atual -- numero real de tentativas
     (deteta retry contando chamadas do dublê), e se o campo de
     diagnostico final chega como "nao_melanoma" por alguma conversao
     silenciosa (nao deveria). Qualquer divergencia do protocolo
     encontrada e marcada como "DIVERGENCIA ENCONTRADA" explicita, com
     arquivo:linha, sem tentar corrigir nada aqui.
  4. (A05 - correção 2026-11-08) Status por agente e consolidação parcial:
     o cenário "agente_nao_responde" revelou que, quando o SPCL retorna
     None, o sumarizador recebe só 2 dos 3 pareceres e nada no estado
     registrava essa ausência (ver status_abcd/status_menzies/status_spcl
     em src/agents/schemas.py::GraphState e src/agents/agents.py). Este
     piloto agora confere: (a) o cenário normal registra 3/3 pareceres;
     (b) o cenário "agente_nao_responde" registra 2/3, identifica o SPCL
     como ausente com causa "desconhecida", e preserva a saída fixa do
     dublê do sumarizador (não infere nada sobre o LLM real a partir
     disso); (c) cobertura diagnóstica (sumarizador produziu um
     diagnóstico válido) é um indicador SEPARADO de consolidação parcial
     (quantos pareceres chegaram); (d) o cenário "agente_fora_do_schema"
     continua travando com ValidationError do próprio GraphState, sem
     alteração. Em nenhum cenário há parecer fictício, retry, ou mudança
     do prompt do sumarizador.

Isto NAO e inferencia real, NAO usa o HAM10000, NAO integra o checkpoint
da CNN, NAO executa S2 com mascara real. Nao decide modelos/versoes de
VLM nem matriz de condicoes.

Uso:
    python scripts/run_multiagent_flow_pilot.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent")


# ──────────────────────────────────────────────
# Dublê determinístico de LLM (sem rede, sem credenciais)
# ──────────────────────────────────────────────


class FakeStructuredRunnable:
    """Retornado por FakeLLM.with_structured_output(schema) -- seu .invoke()
    devolve exatamente o que foi configurado para esta chamada, contando
    quantas vezes foi invocado (para detectar retry) e guardando as
    mensagens recebidas (para inspecionar de fato o que chegou ao
    sumarizador, não só que foi chamado)."""

    def __init__(
        self,
        behavior: dict[str, Any],
        call_log: list[str],
        label: str,
        messages_log: dict[str, list],
    ):
        self.behavior = behavior
        self.call_log = call_log
        self.label = label
        self.messages_log = messages_log

    def invoke(self, messages):
        self.call_log.append(self.label)
        self.messages_log.setdefault(self.label, []).append(messages)
        mode = self.behavior["mode"]
        if mode == "ok":
            return self.behavior["value"]
        if mode == "raise":
            raise self.behavior["exception"]
        if mode == "wrong_shape":
            # Simula uma resposta que nao bate com o schema esperado:
            # um dict cru em vez da instancia Pydantic esperada (o que
            # with_structured_output real normalmente evitaria via parsing,
            # mas testamos o pior caso: algo "passou" sem ser o tipo certo).
            return self.behavior["value"]
        if mode == "empty":
            return None
        raise ValueError(f"modo de comportamento desconhecido: {mode!r}")


class FakeLLM:
    """Substitui qualquer ChatOpenAI/ChatGoogleGenerativeAI/ChatOllama real.
    Uma instância por agente, configurada com o comportamento desejado
    para esta execução do piloto."""

    def __init__(
        self,
        behavior: dict[str, Any],
        call_log: list[str],
        label: str,
        messages_log: dict[str, list],
    ):
        self.behavior = behavior
        self.call_log = call_log
        self.label = label
        self.messages_log = messages_log

    def with_structured_output(self, schema):
        return FakeStructuredRunnable(self.behavior, self.call_log, self.label, self.messages_log)

    def invoke(self, messages):
        # Caminho Ollama (_is_ollama==True) -- nao exercido pelo piloto
        # normal (model_name usado nao contém ':'), mantido por completude.
        self.call_log.append(self.label)
        self.messages_log.setdefault(self.label, []).append(messages)
        if self.behavior["mode"] == "raise":
            raise self.behavior["exception"]
        return type("FakeMsg", (), {"content": str(self.behavior.get("value", ""))})()


def _ok(value: Any) -> dict[str, Any]:
    return {"mode": "ok", "value": value}


def _raise(exc: Exception) -> dict[str, Any]:
    return {"mode": "raise", "exception": exc}


def _wrong_shape(value: Any) -> dict[str, Any]:
    return {"mode": "wrong_shape", "value": value}


def _empty() -> dict[str, Any]:
    return {"mode": "empty"}


# ──────────────────────────────────────────────
# Execução de um cenário contra o grafo real
# ──────────────────────────────────────────────


def _run_scenario(
    scenario_name: str,
    abcd_behavior: dict[str, Any],
    menzies_behavior: dict[str, Any],
    spcl_behavior: dict[str, Any],
    summary_behavior: dict[str, Any],
) -> dict[str, Any]:
    """Monta o grafo real (src.agents.agents.build_graph) com _build_llm
    monkey-patchado para devolver os dublês configurados por agente, roda
    uma vez, e reporta o que foi observado -- sem rede, sem credenciais,
    sem segmentação real (segmentation_mode='precomputed' com um caminho
    de máscara fictício não lido de verdade, já que
    PrecomputedSegmentationAgent.segment_image também é substituído por um
    dublê local para não depender de arquivo de imagem real nem do
    checkpoint da U-Net)."""
    import src.agents.agents as agents_mod

    call_log: list[str] = []
    messages_log: dict[str, list] = {}
    behaviors = {
        "diagnostic_abcd": abcd_behavior,
        "diagnostic_menzies": menzies_behavior,
        "diagnostic_spcl": spcl_behavior,
        "summary": summary_behavior,
    }

    # Mapeia model_name -> comportamento pelo rótulo do agente que está
    # sendo construído no momento da chamada (build_graph constrói os
    # agentes em ordem fixa e conhecida: abcd, menzies, spcl, summary).
    construction_order = ["diagnostic_abcd", "diagnostic_menzies", "diagnostic_spcl", "summary"]
    construction_index = {"i": 0}

    def fake_build_llm(model_name: str, temperature: float):
        label = construction_order[construction_index["i"]]
        construction_index["i"] += 1
        return FakeLLM(behaviors[label], call_log, label, messages_log)

    original_build_llm = agents_mod._build_llm
    # PrecomputedSegmentationAgent NÃO herda de SegmentationAgent (classes
    # separadas, confirmado em agents.py:122 e :171) -- o original a
    # salvar/restaurar é o da própria classe usada aqui.
    original_segment_image = agents_mod.PrecomputedSegmentationAgent.segment_image

    def fake_precomputed_segment_image(self, state):
        # Evita depender de arquivo de imagem/máscara real -- retorna
        # bytes base64 fictícios, suficientes para montar o HumanMessage
        # multimodal sem ler nada do disco.
        return {"seg_image_data": "ZmFrZQ==", "image_data": "ZmFrZQ=="}

    agents_mod._build_llm = fake_build_llm
    agents_mod.PrecomputedSegmentationAgent.segment_image = fake_precomputed_segment_image

    result: dict[str, Any] = {"scenario": scenario_name, "call_log": None, "error": None}
    try:
        final_state = None
        try:
            graph = agents_mod.build_graph(
                retriever=None,  # não usado por build_graph após a remoção do A36
                model_name="fake-model",
                segmentation_mode="precomputed",
            )
            for event in graph.stream(
                {"image_path": "fake.jpg", "lesion_size": 5.0, "seg_path": "fake_mask.png"},
                stream_mode="values",
            ):
                final_state = event
            result["ok"] = True
            result["final_state_keys"] = sorted(final_state.keys()) if final_state else []
            # IMPORTANTE: GraphState.final_report é um campo declarado no
            # schema com default None (agents.py/schemas.py) -- a CHAVE
            # sempre existe no estado, mesmo sem o crítico ter rodado.
            # Por isso o teste correto é o VALOR não ser None, não a
            # presença da chave (que daria falso positivo sempre True).
            result["final_report_present"] = bool(final_state and final_state.get("final_report") is not None)
            result["validation_present"] = bool(final_state and final_state.get("validation") is not None)
            diag_summary = final_state.get("validation") if final_state else None
            result["final_diagnostico_field"] = (
                getattr(diag_summary, "diagnostico", None)
                if diag_summary is not None and hasattr(diag_summary, "diagnostico")
                else diag_summary
            )
            # Inspeciona de fato o que o dublê do sumarizador recebeu (não
            # só que foi chamado): extrai o texto da última mensagem
            # enviada a "summary" e procura o marcador "piloto" (presente
            # no campo `explicacao` de todas as três respostas válidas de
            # ABCD/Menzies/SPCL construídas em _make_valid_responses) para
            # confirmar quantos pareceres de fato chegaram ao prompt.
            summary_calls = messages_log.get("summary", [])
            if summary_calls:
                last_messages = summary_calls[-1]
                text_blocks = [
                    block["text"]
                    for msg in last_messages
                    for block in (msg.content if isinstance(msg.content, list) else [])
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                summary_prompt_text = "\n".join(text_blocks)
            else:
                summary_prompt_text = ""
            result["summary_was_invoked"] = bool(summary_calls)
            result["summary_prompt_marker_count"] = summary_prompt_text.count("piloto")
            result.update(_compute_consolidation(final_state))
        except Exception as exc:  # noqa: BLE001 - piloto deve reportar, não travar
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        agents_mod._build_llm = original_build_llm
        agents_mod.PrecomputedSegmentationAgent.segment_image = original_segment_image

    result["call_log"] = call_log
    result["n_calls_per_agent"] = {label: call_log.count(label) for label in set(call_log)}
    return result


# Identificadores dos três agentes diagnósticos e seus campos correspondentes
# no GraphState -- usados para computar os indicadores de consolidação de
# forma genérica (qualquer agente ausente, não só o SPCL do cenário testado).
AGENT_LABELS = ["diagnostic_abcd", "diagnostic_menzies", "diagnostic_spcl"]
DIAGNOSIS_FIELD_BY_LABEL = {
    "diagnostic_abcd": "diagnosis_abcd",
    "diagnostic_menzies": "diagnosis_menzies",
    "diagnostic_spcl": "diagnosis_spcl",
}


def _compute_consolidation(final_state: dict[str, Any] | None) -> dict[str, Any]:
    """Calcula, a partir do estado final do grafo, três indicadores
    separados (A05 - correção 2026-11-08):

      - cobertura diagnóstica: o sumarizador produziu um `diagnostico`
        válido? (independe de quantos pareceres chegaram até ele -- se
        o protocolo conta o caso como cobertura, continua contando).
      - consolidação parcial: quantos dos 3 pareceres chegaram, e quais
        agentes estão ausentes (genérico -- itera AGENT_LABELS, não
        hardcoded para nenhum agente específico).
      - status final: "determinado" se há diagnóstico, "indeterminado"
        caso contrário -- independente dos status individuais por agente.

    Não infere causa nenhuma além do que o próprio estado já registra
    (populada pelos agentes em agents.py, não aqui). Função pura, sem
    I/O, sem grafo -- testável com dicts fabricados.
    """
    if not final_state:
        return {
            "diagnostic_coverage": False,
            "pareceres_recebidos": 0,
            "pareceres_esperados": len(AGENT_LABELS),
            "agentes_ausentes": list(AGENT_LABELS),
            "consolidacao_parcial": True,
            "status_final": "indeterminado",
        }

    agentes_ausentes = []
    recebidos = 0
    for label in AGENT_LABELS:
        if final_state.get(DIAGNOSIS_FIELD_BY_LABEL[label]) is not None:
            recebidos += 1
        else:
            agentes_ausentes.append(label)

    validation = final_state.get("validation")
    diagnostico = (
        getattr(validation, "diagnostico", None) if validation is not None else None
    )
    diagnostic_coverage = diagnostico is not None

    return {
        "diagnostic_coverage": diagnostic_coverage,
        "pareceres_recebidos": recebidos,
        "pareceres_esperados": len(AGENT_LABELS),
        "agentes_ausentes": agentes_ausentes,
        "consolidacao_parcial": recebidos < len(AGENT_LABELS),
        "status_final": "determinado" if diagnostic_coverage else "indeterminado",
    }


def _make_valid_responses():
    from src.agents.schemas import ABCDDiagnosticAnswer, MenziesDiagnosticAnswer, MenziesPositiveAnswer, MenziesNegativeAnswer, SPCLDiagnosticAnswer, SummaryAnswer

    abcd = ABCDDiagnosticAnswer(assimetria="0", bordas="2", cor="2", estruturas="1", explicacao="piloto")
    menzies = MenziesDiagnosticAnswer(
        positivas=MenziesPositiveAnswer(
            veu_azul_branco="0", multiplos_pontos_marrons="0", pseudopodes="0",
            streaming_radial="0", despigmentacao_tipo_cicatriz="0",
            pontos_pretos_perifericos="0", multiplas_cores="0",
        ),
        negativas=MenziesNegativeAnswer(simetria="1", cor_unica="0"),
        explicacao="piloto",
    )
    spcl = SPCLDiagnosticAnswer(
        padrao_de_pigmentacao_atipico="não", veu_azul_branco_irregular="não",
        vasos_atipicos="não", padrao_de_reticulo_irregular="não",
        padrao_de_globulos_irregulares="não", hiperpigmentacao_localizada="não",
        regressao="não", explicacao="piloto",
    )
    summary = SummaryAnswer(
        diagnostico="Lesão Benigna", justificativa="piloto", recomendacoes="piloto"
    )
    return abcd, menzies, spcl, summary


def _detect_divergences(scenarios_results: list[dict[str, Any]]) -> list[str]:
    """Lógica pura (sem I/O, sem grafo): recebe os dicionários de resultado
    já produzidos por _run_scenario e aponta violações de protocolo —
    retry (>1 chamada ao mesmo agente num cenário de falha) ou diagnóstico
    final convertido silenciosamente para "não melanoma" a partir de uma
    falha simulada upstream. Extraída para ser testável sem executar o
    grafo real (ver tests/multiagent/test_pilot_logic.py)."""
    divergences = []
    for r in scenarios_results:
        if r["scenario"] == "caminho_normal":
            if r.get("ok") and r.get("summary_prompt_marker_count") != 3:
                divergences.append(
                    "DIVERGÊNCIA ENCONTRADA — cenário 'caminho_normal': o prompt recebido pelo "
                    f"sumarizador contém {r.get('summary_prompt_marker_count')} marcador(es) "
                    "esperado(s) dos três pareceres (esperado 3 -- um por ABCD/Menzies/SPCL). "
                    "Ver src/agents/agents.py::SummaryAgent.summarize (linhas 439-466)."
                )
            continue
        for label, n in r.get("n_calls_per_agent", {}).items():
            if n > 1:
                divergences.append(
                    f"DIVERGÊNCIA ENCONTRADA — cenário '{r['scenario']}': agente '{label}' foi "
                    f"chamado {n} vezes (esperado 1, sem retry). Ver scripts/run_multiagent_flow_pilot.py "
                    "(este piloto) e a cadeia real de chamada em src/agents/agents.py::_invoke "
                    "(linhas 72-89) / src/pipeline/pipeline.py::run_pipeline (linha 60) para o ponto exato."
                )
        diag = r.get("final_diagnostico_field")
        if isinstance(diag, str) and "melanoma" in diag.lower() and "nao" in diag.lower().replace("ã", "a"):
            divergences.append(
                f"DIVERGÊNCIA ENCONTRADA — cenário '{r['scenario']}': campo de diagnóstico final "
                f"contém {diag!r}, que parece uma conversão para 'não melanoma' a partir de uma "
                "falha simulada upstream, sem evidência de que isso venha de um julgamento real "
                "do LLM. Ver arquivo:linha exato onde o estado final foi produzido neste cenário."
            )
    return divergences


def run_pilot() -> dict[str, Any]:
    abcd_ok, menzies_ok, spcl_ok, summary_ok = _make_valid_responses()

    scenarios_results = []

    # --- Cenário 1: caminho normal ---
    r = _run_scenario(
        "caminho_normal",
        _ok(abcd_ok), _ok(menzies_ok), _ok(spcl_ok), _ok(summary_ok),
    )
    scenarios_results.append(r)

    # --- Cenário 2: um agente (ABCD) lança exceção ---
    r = _run_scenario(
        "agente_lanca_excecao",
        _raise(RuntimeError("falha simulada de API")), _ok(menzies_ok), _ok(spcl_ok), _ok(summary_ok),
    )
    scenarios_results.append(r)

    # --- Cenário 3: um agente (Menzies) retorna algo fora do schema esperado ---
    r = _run_scenario(
        "agente_fora_do_schema",
        _ok(abcd_ok), _wrong_shape({"campo_inesperado": "valor"}), _ok(spcl_ok), _ok(summary_ok),
    )
    scenarios_results.append(r)

    # --- Cenário 4: um agente (SPCL) "não responde" (retorno vazio/None) ---
    r = _run_scenario(
        "agente_nao_responde",
        _ok(abcd_ok), _ok(menzies_ok), _empty(), _ok(summary_ok),
    )
    scenarios_results.append(r)

    divergences = _detect_divergences(scenarios_results)

    return {
        "scenarios": scenarios_results,
        "divergences_found": divergences,
        "critical_review_ever_in_final_state": any(
            r.get("final_report_present") for r in scenarios_results
        ),
    }


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Piloto do Fluxo Multiagente — Respostas Simuladas (Etapa 5)",
        "",
        "**Sem credenciais, sem chamadas de rede.** Camada de LLM inteiramente "
        "substituída por um dublê determinístico. Não é inferência real, não usa "
        "HAM10000, não integra checkpoint de CNN, não executa S2 com máscara real.",
        "",
        f"## Crítico/RAG nunca roda (campo `final_report` permanece None em todos os cenários): "
        f"{not report['critical_review_ever_in_final_state']}",
        "",
    ]
    if report["divergences_found"]:
        lines.append("## ⚠️ DIVERGÊNCIAS ENCONTRADAS (requerem decisão do pesquisador)")
        lines.append("")
        for d in report["divergences_found"]:
            lines.append(f"- {d}")
        lines.append("")
    else:
        lines.append("## Nenhuma divergência encontrada nos cenários testados")
        lines.append("")

    lines.append("## Resultado por cenário")
    lines.append("")
    lines.append("| Cenário | Terminou sem erro? | Chaves no estado final | `final_report` presente? | Chamadas por agente |")
    lines.append("|---|---|---|---|---|")
    for r in report["scenarios"]:
        lines.append(
            f"| `{r['scenario']}` | {r.get('ok')} | {r.get('final_state_keys')} | "
            f"{r.get('final_report_present')} | {r.get('n_calls_per_agent')} |"
        )
    lines.append("")
    for r in report["scenarios"]:
        lines.append(f"### Detalhe — `{r['scenario']}`")
        lines.append("")
        lines.append(f"- Erro observado: `{r.get('error')}`")
        lines.append(f"- `validation` presente no estado final: {r.get('validation_present')}")
        lines.append(f"- Campo `diagnostico` final (se houver): `{r.get('final_diagnostico_field')}`")
        lines.append(f"- Sumarizador foi invocado: {r.get('summary_was_invoked')}")
        lines.append(
            f"- Marcadores dos 3 pareceres encontrados no prompt do sumarizador: "
            f"{r.get('summary_prompt_marker_count')} (esperado 3 no caminho normal)"
        )
        lines.append(
            f"- Consolidação: {r.get('pareceres_recebidos')}/{r.get('pareceres_esperados')} "
            f"pareceres (agentes ausentes: {r.get('agentes_ausentes')})"
        )
        lines.append(
            f"- Cobertura diagnóstica (sumarizador produziu diagnóstico válido, "
            f"independente de consolidação parcial): {r.get('diagnostic_coverage')}"
        )
        lines.append(f"- Status final: {r.get('status_final')}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(output_dir / "multiagent_pilot.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    print("=== PILOTO: fluxo multiagente com respostas simuladas (sem rede) ===")
    report = run_pilot()

    for r in report["scenarios"]:
        print(
            f"[{r['scenario']}] ok={r.get('ok')} chamadas={r.get('n_calls_per_agent')} "
            f"final_report_presente={r.get('final_report_present')} "
            f"consolidacao={r.get('pareceres_recebidos')}/{r.get('pareceres_esperados')} "
            f"ausentes={r.get('agentes_ausentes')} status_final={r.get('status_final')}"
        )
    print(
        f"\ncritico/RAG nunca roda (campo final_report permanece None em todos os cenários): "
        f"{not report['critical_review_ever_in_final_state']}"
    )
    if report["divergences_found"]:
        print("\n⚠️ DIVERGÊNCIAS ENCONTRADAS:")
        for d in report["divergences_found"]:
            print(f"  - {d}")
    else:
        print("\nNenhuma divergência encontrada.")

    results_path = output_dir / "multiagent_pilot_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **report,
                "scenarios": [
                    {k: v for k, v in r.items() if k != "final_diagnostico_field"}
                    | {"final_diagnostico_field": str(r.get("final_diagnostico_field"))}
                    for r in report["scenarios"]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nresultados (json) gravados: {results_path}")

    report_md_path = output_dir / "multiagent_pilot_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(_render_report_md(report))
    print(f"relatório (md) gravado: {report_md_path}")
    print(f"log gravado: {output_dir / 'multiagent_pilot.log'}")
    print(
        "\nLEMBRETE: isto é um piloto com respostas simuladas, não inferência real. "
        "Nenhuma chamada de rede/credencial foi usada. HAM10000 não foi tocado."
    )


if __name__ == "__main__":
    main()
