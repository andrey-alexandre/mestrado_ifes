"""Testes sintéticos (dados fabricados, não reais) para a lógica pura do
piloto do fluxo multiagente (scripts/run_multiagent_flow_pilot.py).

STATUS: implementados, aguardando execução do pesquisador (pytest). Não
executados pelo assistente/fork que os escreveu -- por diretiva explícita
de não executar nenhum código/teste nesta entrega.

Cobrem apenas funções puras e extraíveis (sem I/O, sem grafo real, sem
LLM real): os dublês de comportamento (_ok/_raise/_wrong_shape/_empty),
a detecção de divergências (_detect_divergences) e a renderização do
relatório Markdown (_render_report_md). A verificação end-to-end do grafo
real (com LLM mockado) já é feita pelo próprio piloto quando o
pesquisador o executa -- não é reimplementada aqui como teste isolado,
pois exigiria reconstruir o grafo real fora do escopo de um teste unitário
puro.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_multiagent_flow_pilot import (  # noqa: E402
    _compute_consolidation,
    _detect_divergences,
    _empty,
    _ok,
    _raise,
    _render_report_md,
    _wrong_shape,
)


# ──────────────────────────────────────────────
# Dublês de comportamento
# ──────────────────────────────────────────────


def test_ok_behavior_wraps_value():
    behavior = _ok("valor_simulado")
    assert behavior == {"mode": "ok", "value": "valor_simulado"}


def test_raise_behavior_wraps_exception():
    exc = RuntimeError("falha simulada")
    behavior = _raise(exc)
    assert behavior["mode"] == "raise"
    assert behavior["exception"] is exc


def test_wrong_shape_behavior_wraps_value():
    behavior = _wrong_shape({"campo": "inesperado"})
    assert behavior == {"mode": "wrong_shape", "value": {"campo": "inesperado"}}


def test_empty_behavior_has_no_value_key():
    behavior = _empty()
    assert behavior == {"mode": "empty"}


# ──────────────────────────────────────────────
# _detect_divergences — dados sintéticos fabricados, sem rodar o grafo
# ──────────────────────────────────────────────


def test_detect_divergences_no_issues_on_clean_scenarios():
    scenarios = [
        {"scenario": "caminho_normal", "ok": True, "summary_prompt_marker_count": 3, "n_calls_per_agent": {"diagnostic_abcd": 1}, "final_diagnostico_field": "Lesão Benigna"},
        {"scenario": "agente_lanca_excecao", "n_calls_per_agent": {"diagnostic_abcd": 1, "diagnostic_menzies": 1}, "final_diagnostico_field": None},
    ]
    assert _detect_divergences(scenarios) == []


def test_detect_divergences_flags_normal_path_missing_a_parecer():
    scenarios = [
        {"scenario": "caminho_normal", "ok": True, "summary_prompt_marker_count": 2, "n_calls_per_agent": {}, "final_diagnostico_field": "Lesão Benigna"},
    ]
    divergences = _detect_divergences(scenarios)
    assert len(divergences) == 1
    assert "DIVERGÊNCIA ENCONTRADA" in divergences[0]
    assert "caminho_normal" in divergences[0]


def test_detect_divergences_skips_marker_check_when_normal_path_errored():
    # Se o cenário normal falhou (ok=False), não faz sentido checar o
    # conteúdo do prompt do sumarizador -- ele pode nem ter sido chamado.
    scenarios = [
        {"scenario": "caminho_normal", "ok": False, "summary_prompt_marker_count": 0, "n_calls_per_agent": {}, "final_diagnostico_field": None},
    ]
    assert _detect_divergences(scenarios) == []


def test_detect_divergences_flags_retry_outside_normal_path():
    scenarios = [
        {"scenario": "agente_lanca_excecao", "n_calls_per_agent": {"diagnostic_abcd": 3}, "final_diagnostico_field": None},
    ]
    divergences = _detect_divergences(scenarios)
    assert len(divergences) == 1
    assert "DIVERGÊNCIA ENCONTRADA" in divergences[0]
    assert "diagnostic_abcd" in divergences[0]
    assert "3 vezes" in divergences[0]


def test_detect_divergences_ignores_retry_count_on_normal_path():
    # Por construção, o caminho normal nunca deveria ter >1 chamada, mas a
    # função deliberadamente não inspeciona esse cenário -- confirma que o
    # filtro por nome de cenário está de fato em vigor.
    scenarios = [
        {"scenario": "caminho_normal", "ok": True, "summary_prompt_marker_count": 3, "n_calls_per_agent": {"diagnostic_abcd": 5}, "final_diagnostico_field": "Lesão Benigna"},
    ]
    assert _detect_divergences(scenarios) == []


def test_detect_divergences_flags_silent_nao_melanoma_conversion():
    scenarios = [
        {"scenario": "agente_nao_responde", "n_calls_per_agent": {"diagnostic_spcl": 1}, "final_diagnostico_field": "não melanoma"},
    ]
    divergences = _detect_divergences(scenarios)
    assert len(divergences) == 1
    assert "DIVERGÊNCIA ENCONTRADA" in divergences[0]
    assert "não melanoma" in divergences[0]


def test_detect_divergences_does_not_flag_unrelated_diagnosis_text():
    scenarios = [
        {"scenario": "agente_fora_do_schema", "n_calls_per_agent": {"diagnostic_menzies": 1}, "final_diagnostico_field": "Lesão Suspeita de Melanoma"},
    ]
    # Contém "melanoma" mas não "nao"/"não" -- não deve ser sinalizado.
    assert _detect_divergences(scenarios) == []


def test_detect_divergences_handles_non_string_diagnostico_field():
    scenarios = [
        {"scenario": "agente_nao_responde", "n_calls_per_agent": {"diagnostic_spcl": 1}, "final_diagnostico_field": None},
    ]
    assert _detect_divergences(scenarios) == []


# ──────────────────────────────────────────────
# _compute_consolidation — dados sintéticos fabricados (A05 - correção)
# ──────────────────────────────────────────────


class _FakeValidation:
    def __init__(self, diagnostico):
        self.diagnostico = diagnostico


def test_compute_consolidation_all_three_pareceres_present():
    final_state = {
        "diagnosis_abcd": "x",
        "diagnosis_menzies": "y",
        "diagnosis_spcl": "z",
        "validation": _FakeValidation("Lesão Benigna"),
    }
    result = _compute_consolidation(final_state)
    assert result["pareceres_recebidos"] == 3
    assert result["pareceres_esperados"] == 3
    assert result["agentes_ausentes"] == []
    assert result["consolidacao_parcial"] is False
    assert result["diagnostic_coverage"] is True
    assert result["status_final"] == "determinado"


def test_compute_consolidation_identifies_missing_agent_generically():
    final_state = {
        "diagnosis_abcd": "x",
        "diagnosis_menzies": "y",
        "diagnosis_spcl": None,
        "validation": _FakeValidation("Lesão Benigna"),
    }
    result = _compute_consolidation(final_state)
    assert result["pareceres_recebidos"] == 2
    assert result["agentes_ausentes"] == ["diagnostic_spcl"]
    assert result["consolidacao_parcial"] is True
    # Cobertura diagnóstica é indicador separado: o sumarizador produziu
    # diagnóstico, então o caso conta na cobertura mesmo com consolidação
    # parcial.
    assert result["diagnostic_coverage"] is True
    assert result["status_final"] == "determinado"


def test_compute_consolidation_identifies_different_missing_agent():
    # Confirma que a lógica não está hardcoded para o SPCL -- qualquer
    # agente ausente é identificado pelo seu próprio rótulo.
    final_state = {
        "diagnosis_abcd": None,
        "diagnosis_menzies": "y",
        "diagnosis_spcl": "z",
        "validation": _FakeValidation("Lesão Benigna"),
    }
    result = _compute_consolidation(final_state)
    assert result["agentes_ausentes"] == ["diagnostic_abcd"]
    assert result["pareceres_recebidos"] == 2


def test_compute_consolidation_status_final_indeterminado_independent_of_agent_status():
    # Todos os 3 pareceres chegaram, mas o sumarizador não produziu um
    # diagnóstico válido -- status final é indeterminado independentemente
    # dos status individuais por agente.
    final_state = {
        "diagnosis_abcd": "x",
        "diagnosis_menzies": "y",
        "diagnosis_spcl": "z",
        "validation": _FakeValidation(None),
    }
    result = _compute_consolidation(final_state)
    assert result["pareceres_recebidos"] == 3
    assert result["consolidacao_parcial"] is False
    assert result["diagnostic_coverage"] is False
    assert result["status_final"] == "indeterminado"


def test_compute_consolidation_no_validation_object():
    final_state = {"diagnosis_abcd": "x", "diagnosis_menzies": "y", "diagnosis_spcl": "z"}
    result = _compute_consolidation(final_state)
    assert result["diagnostic_coverage"] is False
    assert result["status_final"] == "indeterminado"


def test_compute_consolidation_empty_final_state():
    result = _compute_consolidation(None)
    assert result["pareceres_recebidos"] == 0
    assert result["agentes_ausentes"] == [
        "diagnostic_abcd", "diagnostic_menzies", "diagnostic_spcl",
    ]
    assert result["consolidacao_parcial"] is True
    assert result["status_final"] == "indeterminado"


# ──────────────────────────────────────────────
# _render_report_md — fabricado, sem executar nada
# ──────────────────────────────────────────────


def _fake_report(divergences):
    return {
        "scenarios": [
            {
                "scenario": "caminho_normal",
                "ok": True,
                "final_state_keys": ["validation"],
                "final_report_present": False,
                "n_calls_per_agent": {"diagnostic_abcd": 1},
                "error": None,
                "validation_present": True,
                "final_diagnostico_field": "Lesão Benigna",
            }
        ],
        "divergences_found": divergences,
        "critical_review_ever_in_final_state": False,
    }


def test_render_report_md_without_divergences():
    md = _render_report_md(_fake_report([]))
    assert "Nenhuma divergência encontrada" in md
    assert "DIVERGÊNCIAS ENCONTRADAS" not in md
    assert "caminho_normal" in md


def test_render_report_md_with_divergences():
    md = _render_report_md(_fake_report(["DIVERGÊNCIA ENCONTRADA — exemplo sintético"]))
    assert "DIVERGÊNCIAS ENCONTRADAS" in md
    assert "DIVERGÊNCIA ENCONTRADA — exemplo sintético" in md


def test_render_report_md_reports_critical_review_absence_correctly():
    report = _fake_report([])
    report["critical_review_ever_in_final_state"] = False
    md = _render_report_md(report)
    assert "nunca roda (campo `final_report` permanece None em todos os cenários): True" in md
