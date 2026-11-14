"""Protótipo EXPERIMENTAL (Etapa 5, continuação do doc 23): repete os 4
controles de síntese COM IMAGEM REAL (mesma imagem/registry de
`scripts/run_synthesis_control_real_image_selection.py`, já usada por
`scripts/run_synthesis_controls_real_image.py`), mas usando a instrução
de tarefa experimental do sumarizador
(`src.multiagent.experimental_decision_contract
.build_experimental_summary_task_instruction`) em vez do prompt oficial
(`SummaryAgent.summarize`, `src/agents/agents.py`), e classificando a
resposta pelo parser do contrato de decisão
(`src.evaluation.explicit_decision_contract.extract_decisao_final`) em
vez de `classify_case_status`/`classify_case_status_v2`.

Decisão do pesquisador (verbatim, ver
`docs/reconstruction/23_explicit_decision_contract_prototype.md`):
"repetir os quatro controles de síntese com imagem real, uma chamada
por cenário para Qwen3-VL e MedGemma, sem retry" -- esta rodada é uma
decisão SEPARADA e NÃO aprova automaticamente a troca do critério de
avanço do Estágio 1 (`scripts/run_local_vlm_pilot.py`, NÃO modificado).

Isolamento (garantido por desenho):
  - NÃO chama `vd.run_synthesis_control_real_image`/
    `vd.build_synthesis_control_state_real_image` (que usam o
    `SummaryAgent` oficial) -- caminho de chamada PARALELO e
    INDEPENDENTE, construído aqui.
  - NÃO modifica `src/agents/agents.py`, `src/multiagent/vlm_direct.py`,
    `src/evaluation/case_status.py`/`case_status_v2.py` nem
    `scripts/run_local_vlm_pilot.py`.
  - Reusa `vd.FIXED_SYNTHESIS_SCENARIOS` (os mesmos pareceres fixos dos
    4 cenários, intocados) e as primitivas de chamada/timeout/memória/
    classificação de causa já existentes em `src/multiagent/vlm_direct.py`
    (`call_with_timeout`, `classify_call_outcome`, `capture_memory_snapshot`,
    `TIMEOUT_S`) -- mesma convenção de "sem retry", não uma segunda
    implementação divergente.
  - Reusa `_image_message` de `src.agents.agents` -- já é o padrão
    estabelecido neste projeto (`src/multiagent/vlm_direct.py` importa o
    mesmo nome do mesmo módulo) para construir a mensagem multimodal; não
    é um prompt oficial, é só o encapsulamento genérico de
    imagem+texto numa `HumanMessage`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from PIL import Image

from langchain_ollama import ChatOllama

from src.agents.agents import _image_message
from src.evaluation.explicit_decision_contract import (
    DECISION_STATUS_OK,
    DecisionContractResult,
    extract_decisao_final,
)
from src.multiagent import vlm_direct as vd
from src.multiagent.experimental_decision_contract import (
    build_experimental_summary_task_instruction,
)
from src.models.model import convert_to_base64

CONDITION_EXPERIMENTAL_SYNTHESIS_REAL_IMAGE = "controle_sintese_imagem_real_contrato_experimental"


@dataclass
class ExperimentalDecisionCallRecord:
    """Registro de UMA chamada deste caminho experimental -- paralelo a
    `vd.CallRecord`, mas com os campos do contrato de decisão
    (`decision_status`/`decisao_final`) em vez de
    `classified_status`/`classified_status_v2` (esses dois parsers
    antigos não são usados aqui, por desenho)."""

    model_name: str
    scenario: str
    condition: str
    attempt_timestamp_utc: str
    elapsed_s: Optional[float]
    mem_before: Optional["vd.MemorySnapshot"]
    mem_after: Optional["vd.MemorySnapshot"]
    raw_response_text: Optional[str]
    call_status: str
    failure_cause: str
    template_extraction_status: Optional[str]
    decision_status: Optional[str]
    decisao_final: Optional[str]
    matched_raw_values: tuple[str, ...]

    def to_flat_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_name": self.model_name,
            "scenario": self.scenario,
            "condition": self.condition,
            "attempt_timestamp_utc": self.attempt_timestamp_utc,
            "elapsed_s": self.elapsed_s,
            "raw_response_text": self.raw_response_text,
            "call_status": self.call_status,
            "failure_cause": self.failure_cause,
            "template_extraction_status": self.template_extraction_status,
            "decision_status": self.decision_status,
            "decisao_final": self.decisao_final,
            "matched_raw_values": list(self.matched_raw_values),
        }
        for prefix, snap in (("mem_before", self.mem_before), ("mem_after", self.mem_after)):
            if snap is None:
                d[f"{prefix}_rss_bytes"] = None
                d[f"{prefix}_swap_used_bytes"] = None
            else:
                d[f"{prefix}_rss_bytes"] = snap.process_rss_bytes
                d[f"{prefix}_swap_used_bytes"] = snap.system_swap_used_bytes
        return d


def build_experimental_synthesis_control_prompt(scenario_name: str) -> str:
    """Monta a instrução de tarefa experimental do sumarizador para
    ``scenario_name``, a partir dos MESMOS pareceres fixos de
    `vd.FIXED_SYNTHESIS_SCENARIOS` (intocados) -- nunca um parecer
    inventado por este módulo."""
    scenario = vd.FIXED_SYNTHESIS_SCENARIOS[scenario_name]
    return build_experimental_summary_task_instruction(
        diagnosis_abcd=scenario["diagnosis_abcd"],
        diagnosis_menzies=scenario["diagnosis_menzies"],
        diagnosis_spcl=scenario.get("diagnosis_spcl"),
    )


def run_experimental_synthesis_control_real_image(
    model_name: str,
    scenario_name: str,
    image: Image.Image,
    timeout_s: float = vd.TIMEOUT_S,
) -> ExperimentalDecisionCallRecord:
    """UMA chamada (sem retry) ao sumarizador experimental, com a imagem
    REAL já selecionada/verificada externamente (ver
    `scripts/run_synthesis_control_real_image_selection.py`) e os
    pareceres fixos de ``scenario_name``. Classifica a resposta SÓ pelo
    contrato de decisão explícito -- nunca por `classify_case_status`/
    `classify_case_status_v2` (não importados aqui)."""
    text_prompt = build_experimental_synthesis_control_prompt(scenario_name)
    message = _image_message(convert_to_base64(image), text_prompt)
    llm = ChatOllama(model=model_name)

    mem_before = vd.capture_memory_snapshot() if vd.psutil is not None else None
    attempt_ts = datetime.now(timezone.utc).isoformat()
    outcome = vd.call_with_timeout(lambda: llm.invoke([message]).content, timeout_s=timeout_s)
    mem_after = vd.capture_memory_snapshot() if vd.psutil is not None else None

    status, cause, raw_text = vd.classify_call_outcome(outcome)

    decision_result: Optional[DecisionContractResult] = None
    if status == vd.STATUS_OK:
        decision_result = extract_decisao_final(raw_text)

    return ExperimentalDecisionCallRecord(
        model_name=model_name,
        scenario=scenario_name,
        condition=CONDITION_EXPERIMENTAL_SYNTHESIS_REAL_IMAGE,
        attempt_timestamp_utc=attempt_ts,
        elapsed_s=outcome.elapsed_s,
        mem_before=mem_before,
        mem_after=mem_after,
        raw_response_text=raw_text,
        call_status=status,
        failure_cause=cause,
        template_extraction_status=(decision_result.template_extraction_status if decision_result else None),
        decision_status=(decision_result.decision_status if decision_result else None),
        decisao_final=(decision_result.decisao_final if decision_result else None),
        matched_raw_values=(decision_result.matched_raw_values if decision_result else ()),
    )
