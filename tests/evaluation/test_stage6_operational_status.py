from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_operational_status import (  # noqa: E402
    OP_AUSENCIA_EXECUCAO,
    OP_FALHA_TECNICA,
    OP_INDETERMINADO,
    OP_MASCARA_VAZIA,
    OP_NEGATIVO,
    OP_POSITIVO,
    OP_RECUSA,
    OP_RESPOSTA_INVALIDA,
    STAGE6_STATUS_MAP,
    classify_operational_status,
)
from src.evaluation.case_status import AUSENTE, INDETERMINADO, NAO_RECONHECIDO, NEGATIVO, POSITIVO  # noqa: E402


def _record(call_status="ok", decision_status="ok", decisao_final="melanoma", failure_cause=""):
    return {"call_status": call_status, "decision_status": decision_status, "decisao_final": decisao_final, "failure_cause": failure_cause}


def test_positivo_negativo_indeterminado():
    assert classify_operational_status(_record(decisao_final="melanoma")).operational_status == OP_POSITIVO
    assert classify_operational_status(_record(decisao_final="nao_melanoma")).operational_status == OP_NEGATIVO
    assert classify_operational_status(_record(decisao_final="indeterminado")).operational_status == OP_INDETERMINADO


def test_recusa_falha_tecnica_resposta_invalida():
    assert classify_operational_status(_record(call_status="recusa_explicita")).operational_status == OP_RECUSA
    assert classify_operational_status(_record(call_status="falha_tecnica")).operational_status == OP_FALHA_TECNICA
    assert classify_operational_status(_record(call_status="sem_resposta")).operational_status == OP_FALHA_TECNICA
    assert classify_operational_status(_record(decision_status="field_absent", decisao_final=None)).operational_status == OP_RESPOSTA_INVALIDA
    assert classify_operational_status(_record(decision_status="invalid_value", decisao_final=None)).operational_status == OP_RESPOSTA_INVALIDA


def test_ausencia_execucao_sem_registro():
    result = classify_operational_status(None)
    assert result.operational_status == OP_AUSENCIA_EXECUCAO
    assert result.stage6_status == AUSENTE


def test_nao_executado_e_ausencia_execucao_nao_falha():
    result = classify_operational_status(_record(call_status="nao_executado"))
    assert result.operational_status == OP_AUSENCIA_EXECUCAO


def test_mascara_vazia_apenas_quando_sem_registro_e_motivo_e_mascara_predita_vazia():
    result = classify_operational_status(None, condition="s2", structural_unavailability_reason="mascara_predita_vazia")
    assert result.operational_status == OP_MASCARA_VAZIA
    assert result.stage6_status == AUSENTE


def test_s2_indisponivel_por_outro_motivo_nao_e_confundido_com_mascara_vazia():
    result = classify_operational_status(None, condition="s2", structural_unavailability_reason="mascara_predita_arquivo_ausente_localmente")
    assert result.operational_status == OP_AUSENCIA_EXECUCAO
    assert "mascara_predita_arquivo_ausente_localmente" in result.detail


def test_registro_presente_tem_prioridade_sobre_indisponibilidade_estrutural():
    """Se existe um registro OFICIAL de chamada, a indisponibilidade
    estrutural de S2 não é consultada -- ela só importa quando record is None."""
    result = classify_operational_status(
        _record(decisao_final="melanoma"), condition="s2", structural_unavailability_reason="mascara_predita_vazia",
    )
    assert result.operational_status == OP_POSITIVO


def test_stage6_status_map_e_particao_mutuamente_exclusiva_e_completa():
    from src.evaluation.stage6_operational_status import ALL_OPERATIONAL_STATUSES
    mapped = {STAGE6_STATUS_MAP[s] for s in ALL_OPERATIONAL_STATUSES}
    assert mapped == {POSITIVO, NEGATIVO, INDETERMINADO, AUSENTE, NAO_RECONHECIDO}


def test_decisao_final_invalida_com_decision_status_ok_levanta_erro():
    with pytest.raises(ValueError):
        classify_operational_status(_record(decision_status="ok", decisao_final="algo_fora_do_vocabulario"))


def test_call_status_desconhecido_levanta_erro():
    with pytest.raises(ValueError):
        classify_operational_status(_record(call_status="status_inexistente"))
