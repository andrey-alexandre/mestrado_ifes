"""Etapa 6 (continuação) -- classificador OPERACIONAL de um evento de
decisão por lesão/sistema/condição, distinguindo indeterminação, recusa,
resposta inválida, falha técnica, ausência de execução e máscara vazia
(protocolo da análise quantitativa final, seção 2), SEM dupla contagem
(cada evento cai em exatamente um status operacional).

Este módulo é ADITIVO: não substitui `src.evaluation.case_status`
(`classify_case_status`) nem `src.evaluation.stage6_metrics`, que
continuam com seu vocabulário de 5 status (POSITIVO/NEGATIVO/
INDETERMINADO/AUSENTE/NAO_RECONHECIDO) intacto -- `STAGE6_STATUS_MAP`
abaixo é a ÚNICA ponte explícita e documentada entre os dois
vocabulários, usada para alimentar `stage6_metrics.compute_stage6_
indicators` sem perder a granularidade operacional (que é reportada
separadamente, na tabela operacional).

Fontes dos três sinais usados aqui, já existentes e reutilizados sem
modificação:
  - ``call_status`` -- vocabulário de
    ``src.multiagent.vlm_direct`` (STATUS_OK/STATUS_TECHNICAL_FAILURE/
    STATUS_NO_RESPONSE/STATUS_REFUSAL/STATUS_NOT_EXECUTED).
  - ``decision_status``/``decisao_final`` -- vocabulário de
    ``src.evaluation.explicit_decision_contract``.
  - ``reason_s2``/``status_s2`` -- colunas de
    `experiments/multiagent/s0_s1_s2_preparation_manifest.csv` (saída de
    `scripts/run_s0_s1_s2_preparation.py`), indicando por que uma
    lesão/condição nunca teve uma chamada planejada (ex.
    ``"mascara_predita_vazia"``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.evaluation.case_status import (
    AUSENTE,
    INDETERMINADO,
    NAO_RECONHECIDO,
    NEGATIVO,
    POSITIVO,
)
from src.evaluation.explicit_decision_contract import (
    DECISAO_INDETERMINADO,
    DECISAO_MELANOMA,
    DECISAO_NAO_MELANOMA,
    DECISION_STATUS_OK,
)
from src.multiagent.vlm_direct import (
    STATUS_NO_RESPONSE,
    STATUS_NOT_EXECUTED,
    STATUS_OK,
    STATUS_REFUSAL,
    STATUS_TECHNICAL_FAILURE,
)

# ---------------------------------------------------------------------------
# Vocabulário operacional -- fechado, mutuamente exclusivo (cada evento cai
# em exatamente um destes 8 status).
# ---------------------------------------------------------------------------
OP_POSITIVO = "positivo"
OP_NEGATIVO = "negativo"
OP_INDETERMINADO = "indeterminado"
OP_RECUSA = "recusa"
OP_RESPOSTA_INVALIDA = "resposta_invalida"
OP_FALHA_TECNICA = "falha_tecnica"
OP_AUSENCIA_EXECUCAO = "ausencia_execucao"
OP_MASCARA_VAZIA = "mascara_vazia"

ALL_OPERATIONAL_STATUSES = (
    OP_POSITIVO, OP_NEGATIVO, OP_INDETERMINADO, OP_RECUSA,
    OP_RESPOSTA_INVALIDA, OP_FALHA_TECNICA, OP_AUSENCIA_EXECUCAO,
    OP_MASCARA_VAZIA,
)

# Ponte ÚNICA e explícita para o vocabulário de 5 status de
# `case_status.ALL_STATUSES`, usada só para alimentar
# `stage6_metrics.compute_stage6_indicators` (acurácia/precisão/
# sensibilidade/especificidade/F1, cobertura, acertos-sobre-total). A
# granularidade operacional completa (recusa vs. falha técnica vs.
# ausência de execução vs. máscara vazia) é preservada na tabela
# operacional (`stage6_report`), nunca perdida aqui -- só agregada para o
# cálculo das 5 métricas, que não distingue "por que" não houve decisão
# binária.
STAGE6_STATUS_MAP: dict[str, str] = {
    OP_POSITIVO: POSITIVO,
    OP_NEGATIVO: NEGATIVO,
    OP_INDETERMINADO: INDETERMINADO,
    OP_RESPOSTA_INVALIDA: NAO_RECONHECIDO,
    OP_RECUSA: AUSENTE,
    OP_FALHA_TECNICA: AUSENTE,
    OP_AUSENCIA_EXECUCAO: AUSENTE,
    OP_MASCARA_VAZIA: AUSENTE,
}

# Reason code de `status_s2`/`reason_s2` (s0_s1_s2_preparation_manifest.csv)
# que mapeia especificamente para OP_MASCARA_VAZIA -- ver
# `src.multiagent.presentation` (vocabulário fechado de motivos de
# indisponibilidade de S2, Etapa 5). Qualquer OUTRO motivo de
# indisponibilidade (ex. arquivo não copiado localmente) é tratado como
# OP_AUSENCIA_EXECUCAO (não houve e não pode haver chamada), preservando o
# motivo bruto em ``detail`` -- nunca silenciosamente fundido com máscara
# vazia.
REASON_MASCARA_PREDITA_VAZIA = "mascara_predita_vazia"
STATUS_DISPONIVEL = "disponivel"


@dataclass
class OperationalClassification:
    operational_status: str
    stage6_status: str
    detail: Optional[str]


def classify_operational_status(
    record: Optional[dict],
    *,
    condition: Optional[str] = None,
    structural_unavailability_reason: Optional[str] = None,
) -> OperationalClassification:
    """Classifica um único evento (lesão, sistema, condição, papel) no
    vocabulário operacional de 8 status.

    Args:
        record: o registro da tentativa oficial já selecionada (ver
            ``src.evaluation.stage6_sources.select_official_call_records``)
            com as chaves ``call_status``/``decision_status``/
            ``decisao_final``, ou ``None`` quando nenhuma tentativa
            oficial existe para esta chave.
        condition: ``"s0"``/``"s1"``/``"s2"`` -- usado só para decidir se
            ``structural_unavailability_reason`` se aplica (a distinção
            máscara-vazia só existe para S2).
        structural_unavailability_reason: o valor de ``reason_s2`` (ou
            ``None`` se ``status_s2 == "disponivel"``) quando
            ``record is None`` E a ausência é estrutural (nunca houve
            payload a enviar), não uma lacuna de execução genérica.
    """
    if record is None:
        if condition == "s2" and structural_unavailability_reason:
            if structural_unavailability_reason == REASON_MASCARA_PREDITA_VAZIA:
                return OperationalClassification(
                    OP_MASCARA_VAZIA, STAGE6_STATUS_MAP[OP_MASCARA_VAZIA],
                    detail=structural_unavailability_reason,
                )
            return OperationalClassification(
                OP_AUSENCIA_EXECUCAO, STAGE6_STATUS_MAP[OP_AUSENCIA_EXECUCAO],
                detail=f"s2_indisponivel_motivo_nao_mascara_vazia:{structural_unavailability_reason}",
            )
        return OperationalClassification(
            OP_AUSENCIA_EXECUCAO, STAGE6_STATUS_MAP[OP_AUSENCIA_EXECUCAO], detail=None,
        )

    call_status = record.get("call_status")

    if call_status == STATUS_REFUSAL:
        return OperationalClassification(OP_RECUSA, STAGE6_STATUS_MAP[OP_RECUSA], record.get("failure_cause"))
    if call_status == STATUS_TECHNICAL_FAILURE:
        return OperationalClassification(OP_FALHA_TECNICA, STAGE6_STATUS_MAP[OP_FALHA_TECNICA], record.get("failure_cause"))
    if call_status == STATUS_NO_RESPONSE:
        return OperationalClassification(
            OP_FALHA_TECNICA, STAGE6_STATUS_MAP[OP_FALHA_TECNICA],
            detail=f"subtipo_sem_resposta:{record.get('failure_cause')}",
        )
    if call_status == STATUS_NOT_EXECUTED:
        return OperationalClassification(
            OP_AUSENCIA_EXECUCAO, STAGE6_STATUS_MAP[OP_AUSENCIA_EXECUCAO],
            detail=f"papel_pausado:{record.get('failure_cause')}",
        )
    if call_status != STATUS_OK:
        raise ValueError(f"call_status desconhecido/inesperado: {call_status!r}")

    decision_status = record.get("decision_status")
    if decision_status != DECISION_STATUS_OK:
        return OperationalClassification(
            OP_RESPOSTA_INVALIDA, STAGE6_STATUS_MAP[OP_RESPOSTA_INVALIDA], detail=decision_status,
        )

    decisao_final = record.get("decisao_final")
    if decisao_final == DECISAO_MELANOMA:
        return OperationalClassification(OP_POSITIVO, STAGE6_STATUS_MAP[OP_POSITIVO], detail=None)
    if decisao_final == DECISAO_NAO_MELANOMA:
        return OperationalClassification(OP_NEGATIVO, STAGE6_STATUS_MAP[OP_NEGATIVO], detail=None)
    if decisao_final == DECISAO_INDETERMINADO:
        return OperationalClassification(OP_INDETERMINADO, STAGE6_STATUS_MAP[OP_INDETERMINADO], detail=None)
    raise ValueError(f"decisao_final inesperado com decision_status OK: {decisao_final!r}")
