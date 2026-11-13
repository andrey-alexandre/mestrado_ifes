"""Etapa 5, segunda entrega — camada de chamada a VLMs locais via Ollama
(VLM-direto e multiagente-sobre-Ollama), controles de pré-execução, lógica
de abstenção sistemática, e classificação A34 — para o piloto de 12
lesões fora da amostra congelada (ver
`docs/reconstruction/20_local_vlm_pilot.md`).

Este módulo NÃO executa nada por si só (sem CLI, sem chamada real feita
pelo assistente que o escreveu — `feedback_no_autonomous_execution`). O
ponto de entrada executável é `scripts/run_local_vlm_pilot.py`.

Decisões já fechadas com o pesquisador, implementadas aqui sem
re-derivação:
  - Timeout por chamada: 120s — excedê-lo conta como falha técnica/
    sem-resposta (nunca um retry).
  - Pausa por abstenção sistemática: 3 falhas técnicas consecutivas, OU
    3 "sem resposta" consecutivas, OU 3 recusas explícitas consecutivas,
    dentro do MESMO papel (ABCD/Menzies/SPCL/sumarizador/VLM-direto) —
    três contadores distintos por papel, cada um reiniciado em qualquer
    sucesso (incluindo diagnóstico indeterminado válido — indeterminado
    NUNCA reinicia/dispara pausa como se fosse abstenção).
  - Nenhuma chamada é repetida (sem retry em lugar nenhum).
  - Casos que se tornam inalcançáveis por pausa são marcados
    `"nao_executado"` — status distinto de qualquer categoria de falha,
    nunca omitido.

Interpretações de design flagueadas para confirmação do pesquisador ANTES
de qualquer execução real (ver também o relatório):
  (1) o texto exato da instrução de tarefa do VLM-direto
      (`VLM_DIRECT_TASK_INSTRUCTION`);
  (2) o desenho dos controles de envio de imagem (`IMAGE_CONTROL_SPECS`,
      `IMAGE_CONTROL_PROMPT`);
  (3) os 4 inputs fixos dos controles de síntese (`FIXED_SYNTHESIS_SCENARIOS`);
  (4) a heurística de "achado clínico novo" nos controles de síntese
      (`_vocabulary_terms_not_in_inputs`);
  (5) a heurística de detecção de recusa explícita (`_REFUSAL_MARKERS`);
  (6) a definição de "crescimento sustentado de swap"
      (`detect_sustained_swap_growth`).
"""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw

try:
    from langchain_core.messages import HumanMessage
except ImportError:  # pragma: no cover
    from langchain.schema import HumanMessage

from langchain_ollama import ChatOllama

from src.agents.agents import (
    ABCDDiagnosticAgent,
    MenziesDiagnosticAgent,
    SPCLDiagnosticAgent,
    SummaryAgent,
    _image_message,
)
from src.agents.schemas import (
    ABCDDiagnosticAnswer,
    AgentStatus,
    GraphState,
    MenziesDiagnosticAnswer,
    SPCLDiagnosticAnswer,
    SummaryAnswer,
)
from src.evaluation.case_status import classify_case_status
from src.models.model import convert_to_base64

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # Dependência NOVA deste módulo (não estava em requirements.txt
    # antes desta entrega) -- ver requisitos abaixo e no relatório. O import não
    # falha o módulo inteiro: funções que dependem de psutil levantam
    # RuntimeError explícito só quando efetivamente chamadas.

try:
    import ollama as ollama_client
except ImportError:  # pragma: no cover
    ollama_client = None  # idem -- já era dependência transitiva de
    # langchain-ollama, mas nunca importada diretamente neste projeto até agora.

TIMEOUT_S = 120.0

ROLE_VLM_DIRETO = "vlm_direto"
ROLE_ABCD = "diagnostic_abcd"
ROLE_MENZIES = "diagnostic_menzies"
ROLE_SPCL = "diagnostic_spcl"
ROLE_SUMMARY = "summary"
ALL_ROLES = (ROLE_VLM_DIRETO, ROLE_ABCD, ROLE_MENZIES, ROLE_SPCL, ROLE_SUMMARY)
MULTIAGENT_ROLES_IN_ORDER = (ROLE_ABCD, ROLE_MENZIES, ROLE_SPCL, ROLE_SUMMARY)

STATUS_OK = "ok"
STATUS_TECHNICAL_FAILURE = "falha_tecnica"
STATUS_NO_RESPONSE = "sem_resposta"
STATUS_REFUSAL = "recusa_explicita"
STATUS_NOT_EXECUTED = "nao_executado"


# ---------------------------------------------------------------------------
# 1. Instrução de tarefa do VLM-direto (INTERPRETAÇÃO DE DESIGN #1)
# ---------------------------------------------------------------------------
# Pede explicitamente o MESMO vocabulário de 3 categorias que
# SummaryAnswer.diagnostico documenta ("Suspeita de Melanoma" / "Lesão
# Benigna" / "Lesão Indeterminada") -- sem isso, classify_case_status
# classificaria qualquer resposta de VLM-direto que não usasse essa frase
# exata como "nao_reconhecido" apenas por não usar a fraseologia esperada,
# o que seria um artefato do protocolo de avaliação, não um achado real de
# capacidade do modelo. NÃO É uma decisão óbvia (o VLM-direto, por
# definição, não passa pelos 3 algoritmos/sumarizador) -- precisa de
# confirmação explícita do pesquisador antes de qualquer execução real.
VLM_DIRECT_TASK_INSTRUCTION = """
<-- Identidade do Agente -->
Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.

<-- Instruções -->
Avalie a lesão cutânea na imagem fornecida e dê um parecer diagnóstico
único e direto.

Sua resposta DEVE conter, de forma explícita e literal, uma das três
expressões abaixo (exatamente como escritas, para permitir classificação
automática da sua resposta):
  - "Suspeita de Melanoma"
  - "Lesão Benigna"
  - "Lesão Indeterminada"

Use "Lesão Indeterminada" quando a evidência visual não permitir uma
conclusão razoável entre as outras duas opções -- isto é uma resposta
válida, não uma recusa.

Após a expressão escolhida, justifique brevemente sua escolha com base
nos achados visuais observados na imagem.
"""


# ---------------------------------------------------------------------------
# 2. Snapshot de memória/swap (psutil)
# ---------------------------------------------------------------------------
@dataclass
class MemorySnapshot:
    process_rss_bytes: int
    system_swap_used_bytes: int
    timestamp_utc: str


def capture_memory_snapshot() -> MemorySnapshot:
    if psutil is None:
        raise RuntimeError(
            "psutil não está instalado -- é uma dependência NOVA exigida por este "
            "módulo (não estava em requirements.txt antes desta entrega). "
            "Instale com `pip install psutil` antes de rodar o piloto real."
        )
    process = psutil.Process()
    return MemorySnapshot(
        process_rss_bytes=process.memory_info().rss,
        system_swap_used_bytes=psutil.swap_memory().used,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# 3. Informações do modelo (ollama.show), uma vez por modelo
# ---------------------------------------------------------------------------
def get_model_info(model_name: str) -> dict[str, Any]:
    """Chama ``ollama.show(model_name)`` UMA vez (chamador decide quando --
    o piloto chama isto uma vez por modelo no início, nunca por chamada).
    Devolve um dict serializável; nunca levanta por conta própria além do
    que ``ollama.show`` já levantaria (ex.: modelo não existe localmente)."""
    if ollama_client is None:
        raise RuntimeError(
            "o pacote 'ollama' não está instalado -- dependência já esperada "
            "(langchain-ollama) mas nunca importada diretamente neste projeto "
            "até esta entrega. Instale/confirme com `pip show ollama`."
        )
    info = ollama_client.show(model_name)
    # ollama.show pode devolver um objeto typed (ShowResponse) dependendo da
    # versão do cliente -- normaliza para dict puro e serializável.
    if hasattr(info, "model_dump"):
        info = info.model_dump()
    return dict(info)


# ---------------------------------------------------------------------------
# 4. Chamada com timeout de 120s, sem retry
# ---------------------------------------------------------------------------
@dataclass
class TimedCallOutcome:
    """Resultado de uma única tentativa de chamada, com timeout aplicado.

    ``timed_out=True`` significa que a chamada excedeu TIMEOUT_S -- o
    thread da chamada real pode continuar rodando em segundo plano (o
    ``ThreadPoolExecutor`` não tem como interromper uma chamada de rede já
    em andamento); isto é uma limitação conhecida e documentada, não um
    bug -- o piloto NUNCA espera o resultado atrasado nem o usa."""

    value: Any
    exception: Optional[BaseException]
    timed_out: bool
    elapsed_s: float


def call_with_timeout(fn: Callable[[], Any], timeout_s: float = TIMEOUT_S) -> TimedCallOutcome:
    """IMPORTANTE: NÃO usar ``with ThreadPoolExecutor() as executor:`` aqui
    -- a saída desse context manager chama ``executor.shutdown(wait=True)``,
    que BLOQUEIA até a thread da chamada lenta terminar, anulando na prática
    o timeout de 120s (o chamador ficaria esperando a chamada real completar
    mesmo após um ``TimeoutError``). Por isso o executor é criado e
    encerrado manualmente com ``shutdown(wait=False)`` -- a thread da
    chamada que excedeu o timeout continua rodando em segundo plano até
    terminar por conta própria (limitação conhecida e documentada na classe
    ``TimedCallOutcome``), mas esta função devolve imediatamente.
    """
    t0 = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        value = future.result(timeout=timeout_s)
        return TimedCallOutcome(value=value, exception=None, timed_out=False, elapsed_s=time.monotonic() - t0)
    except concurrent.futures.TimeoutError:
        return TimedCallOutcome(value=None, exception=None, timed_out=True, elapsed_s=time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001 -- causa registrada, nunca propagada aqui
        return TimedCallOutcome(value=None, exception=exc, timed_out=False, elapsed_s=time.monotonic() - t0)
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# 5. Classificação da causa de uma chamada (sucesso/falha técnica/sem
#    resposta/recusa) -- distinta da classificação A34 do CONTEÚDO.
# ---------------------------------------------------------------------------
# INTERPRETAÇÃO DE DESIGN #5: lista de marcadores textuais de recusa
# explícita. Heurística simples (substring, case-insensitive) -- não há
# especificação prévia no projeto para isto; precisa de confirmação do
# pesquisador antes de uso real, em especial para texto em português.
_REFUSAL_MARKERS = (
    "não posso ajudar",
    "não posso fornecer",
    "não posso fazer isso",
    "i cannot help",
    "i can't help",
    "i cannot provide",
    "i'm not able to",
    "as an ai",
    "não sou capaz de fornecer um diagnóstico",
    "não tenho capacidade de diagnosticar",
)


def classify_call_outcome(outcome: TimedCallOutcome) -> tuple[str, str, Optional[str]]:
    """Devolve ``(status, causa, raw_text)``.

    ``status`` é um de STATUS_OK/STATUS_TECHNICAL_FAILURE/STATUS_NO_RESPONSE/
    STATUS_REFUSAL. ``causa`` é uma string curta para log. ``raw_text`` é o
    texto bruto devolvido quando houver (preservado mesmo em falha parcial).
    """
    if outcome.timed_out:
        return STATUS_TECHNICAL_FAILURE, f"timeout_excedido_{TIMEOUT_S:.0f}s", None
    if outcome.exception is not None:
        return STATUS_TECHNICAL_FAILURE, f"excecao:{type(outcome.exception).__name__}:{outcome.exception}", None

    raw_text = outcome.value
    if raw_text is None:
        return STATUS_NO_RESPONSE, "resposta_none", None
    text = str(raw_text)
    if not text.strip():
        return STATUS_NO_RESPONSE, "resposta_vazia", text

    lowered = text.lower()
    for marker in _REFUSAL_MARKERS:
        if marker in lowered:
            return STATUS_REFUSAL, f"marcador_recusa:{marker}", text

    return STATUS_OK, "", text


# ---------------------------------------------------------------------------
# 6. Contadores de abstenção sistemática -- 3 por papel, independentes
# ---------------------------------------------------------------------------
@dataclass
class AbstentionCounters:
    """3 contadores consecutivos independentes para UM papel. Qualquer
    sucesso (STATUS_OK, incluindo diagnóstico indeterminado válido)
    reinicia TODOS os 3 a zero -- indeterminado nunca é tratado como
    abstenção."""

    technical_failure: int = 0
    no_response: int = 0
    refusal: int = 0
    paused: bool = False
    pause_reason: str = ""

    def register(self, status: str) -> None:
        if self.paused:
            return
        if status == STATUS_OK:
            self.technical_failure = 0
            self.no_response = 0
            self.refusal = 0
            return
        if status == STATUS_TECHNICAL_FAILURE:
            self.technical_failure += 1
            self.no_response = 0
            self.refusal = 0
        elif status == STATUS_NO_RESPONSE:
            self.no_response += 1
            self.technical_failure = 0
            self.refusal = 0
        elif status == STATUS_REFUSAL:
            self.refusal += 1
            self.technical_failure = 0
            self.no_response = 0
        else:
            raise ValueError(f"status desconhecido para contador de abstenção: {status!r}")

        if self.technical_failure >= 3:
            self.paused, self.pause_reason = True, "3_falhas_tecnicas_consecutivas"
        elif self.no_response >= 3:
            self.paused, self.pause_reason = True, "3_sem_resposta_consecutivas"
        elif self.refusal >= 3:
            self.paused, self.pause_reason = True, "3_recusas_explicitas_consecutivas"


@dataclass
class AbstentionTracker:
    """Um :class:`AbstentionCounters` por papel (ABCD/Menzies/SPCL/
    sumarizador/VLM-direto) -- pausas são SEMPRE por papel, nunca globais."""

    by_role: dict[str, AbstentionCounters] = field(
        default_factory=lambda: {role: AbstentionCounters() for role in ALL_ROLES}
    )

    def is_paused(self, role: str) -> bool:
        return self.by_role[role].paused

    def register(self, role: str, status: str) -> None:
        self.by_role[role].register(status)

    def trace(self) -> dict[str, dict[str, Any]]:
        return {
            role: {
                "technical_failure_consecutive": c.technical_failure,
                "no_response_consecutive": c.no_response,
                "refusal_consecutive": c.refusal,
                "paused": c.paused,
                "pause_reason": c.pause_reason,
            }
            for role, c in self.by_role.items()
        }


# ---------------------------------------------------------------------------
# 7. Registro de uma chamada (log por chamada)
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    model_name: str
    role: str
    condition: str
    lesion_id: str
    image_id: str
    attempt_timestamp_utc: str
    elapsed_s: Optional[float]
    mem_before: Optional[MemorySnapshot]
    mem_after: Optional[MemorySnapshot]
    raw_response_text: Optional[str]
    call_status: str  # STATUS_OK / _TECHNICAL_FAILURE / _NO_RESPONSE / _REFUSAL / _NOT_EXECUTED
    failure_cause: str
    classified_status: Optional[str]  # saída de classify_case_status, None se nao_executado

    def to_flat_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_name": self.model_name,
            "role": self.role,
            "condition": self.condition,
            "lesion_id": self.lesion_id,
            "image_id": self.image_id,
            "attempt_timestamp_utc": self.attempt_timestamp_utc,
            "elapsed_s": self.elapsed_s,
            "raw_response_text": self.raw_response_text,
            "call_status": self.call_status,
            "failure_cause": self.failure_cause,
            "classified_status": self.classified_status,
        }
        for prefix, snap in (("mem_before", self.mem_before), ("mem_after", self.mem_after)):
            if snap is None:
                d[f"{prefix}_rss_bytes"] = None
                d[f"{prefix}_swap_used_bytes"] = None
            else:
                d[f"{prefix}_rss_bytes"] = snap.process_rss_bytes
                d[f"{prefix}_swap_used_bytes"] = snap.system_swap_used_bytes
        return d


# ---------------------------------------------------------------------------
# 8. VLM-direto: uma chamada, S0, vocabulário de 3 categorias
# ---------------------------------------------------------------------------
def build_vlm_direct_message(image: Image.Image, task_instruction: str = VLM_DIRECT_TASK_INSTRUCTION) -> HumanMessage:
    seg_b64 = convert_to_base64(image)
    return _image_message(seg_b64, task_instruction)


def run_vlm_direct_call(
    model_name: str,
    image: Image.Image,
    lesion_id: str,
    image_id: str,
    timeout_s: float = TIMEOUT_S,
    task_instruction: str = VLM_DIRECT_TASK_INSTRUCTION,
) -> CallRecord:
    message = build_vlm_direct_message(image, task_instruction)
    llm = ChatOllama(model=model_name)

    mem_before = capture_memory_snapshot() if psutil is not None else None
    attempt_ts = datetime.now(timezone.utc).isoformat()
    outcome = call_with_timeout(lambda: llm.invoke([message]).content, timeout_s=timeout_s)
    mem_after = capture_memory_snapshot() if psutil is not None else None

    status, cause, raw_text = classify_call_outcome(outcome)
    classified = classify_case_status({"validation": raw_text}) if status == STATUS_OK else None

    return CallRecord(
        model_name=model_name,
        role=ROLE_VLM_DIRETO,
        condition="s0",
        lesion_id=lesion_id,
        image_id=image_id,
        attempt_timestamp_utc=attempt_ts,
        elapsed_s=outcome.elapsed_s,
        mem_before=mem_before,
        mem_after=mem_after,
        raw_response_text=raw_text,
        call_status=status,
        failure_cause=cause,
        classified_status=classified,
    )


# ---------------------------------------------------------------------------
# 9. Multiagente sobre Ollama, caso a caso, sem passar pelo build_graph()
#    compilado (ver justificativa abaixo).
# ---------------------------------------------------------------------------
# CONFIRMAÇÃO DE LEITURA DE CÓDIGO (não suposição): o grafo compilado por
# `src.agents.agents.build_graph()` tem uma única aresta de entrada
# START -> "segmentation", e o nó "segmentation" SEMPRE lê state.image_path
# (e, no modo precomputed, state.seg_path) para computar seg_image_data
# dentro do próprio nó -- não há como invocar o grafo compilado pulando
# esse nó e injetando um seg_image_data já pronto (LangGraph não expõe uma
# forma de "entrar" num nó do meio sem passar por START). Portanto este
# módulo NÃO modifica agents.py nem build_graph(): instancia diretamente
# ABCDDiagnosticAgent/MenziesDiagnosticAgent/SPCLDiagnosticAgent/
# SummaryAgent (as mesmas classes, os mesmos prompts, a mesma função
# _invoke/_image_message internas) e as chama na mesma ordem de
# dependência do grafo real (3 diagnósticos em paralelo logicamente, aqui
# sequenciais por simplicidade -- sem efeito no conteúdo, já que os 3 são
# independentes entre si -> sumarizador), usando um GraphState real
# (src.agents.schemas.GraphState) com seg_image_data já preenchido pela
# condição (S0/S1/S2) sob teste.
_ROLE_TO_AGENT_CLASS_AND_SCHEMA = {
    ROLE_ABCD: (ABCDDiagnosticAgent, ABCDDiagnosticAnswer),
    ROLE_MENZIES: (MenziesDiagnosticAgent, MenziesDiagnosticAnswer),
    ROLE_SPCL: (SPCLDiagnosticAgent, SPCLDiagnosticAnswer),
}


def run_multiagent_role_call(
    model_name: str,
    role: str,
    state: GraphState,
    lesion_id: str,
    image_id: str,
    condition: str,
    timeout_s: float = TIMEOUT_S,
) -> tuple[CallRecord, dict[str, Any]]:
    """Executa UM papel (ABCD/Menzies/SPCL/sumarizador) sobre ``state`` já
    populado com ``seg_image_data`` (e, para o sumarizador, com
    diagnosis_abcd/menzies/spcl já preenchidos pelas 3 chamadas
    anteriores do mesmo caso). Devolve ``(CallRecord, state_update_dict)``
    -- ``state_update_dict`` é exatamente o dict que o nó real do grafo
    devolveria (mesmo formato de `.analyze_lesion`/`.summarize`), para o
    chamador atualizar seu próprio ``state`` acumulado por caso.
    """
    mem_before = capture_memory_snapshot() if psutil is not None else None
    attempt_ts = datetime.now(timezone.utc).isoformat()

    if role == ROLE_SUMMARY:
        agent = SummaryAgent(model_name, SummaryAnswer)
        outcome = call_with_timeout(lambda: agent.summarize(state), timeout_s=timeout_s)
    else:
        agent_cls, schema = _ROLE_TO_AGENT_CLASS_AND_SCHEMA[role]
        agent = agent_cls(model_name, schema)
        outcome = call_with_timeout(lambda: agent.analyze_lesion(state), timeout_s=timeout_s)

    mem_after = capture_memory_snapshot() if psutil is not None else None

    if outcome.timed_out or outcome.exception is not None:
        status, cause, _ = classify_call_outcome(outcome)
        raw_text = None
        classified = None
        state_update: dict[str, Any] = {}
        if role != ROLE_SUMMARY:
            agcls_key = {ROLE_ABCD: "diagnosis_abcd", ROLE_MENZIES: "diagnosis_menzies", ROLE_SPCL: "diagnosis_spcl"}[role]
            status_key = {ROLE_ABCD: "status_abcd", ROLE_MENZIES: "status_menzies", ROLE_SPCL: "status_spcl"}[role]
            state_update = {agcls_key: None, status_key: AgentStatus(status="sem_resposta", causa=cause)}
    else:
        result_dict = outcome.value  # dict devolvido por analyze_lesion/summarize
        if role == ROLE_SUMMARY:
            raw_text = str(result_dict.get("validation")) if result_dict.get("validation") is not None else None
        else:
            diag_key = {ROLE_ABCD: "diagnosis_abcd", ROLE_MENZIES: "diagnosis_menzies", ROLE_SPCL: "diagnosis_spcl"}[role]
            raw_text = str(result_dict.get(diag_key)) if result_dict.get(diag_key) is not None else None
        fake_outcome = TimedCallOutcome(value=raw_text, exception=None, timed_out=False, elapsed_s=outcome.elapsed_s)
        status, cause, raw_text = classify_call_outcome(fake_outcome)
        classified = classify_case_status({"validation": raw_text}) if (status == STATUS_OK and role == ROLE_SUMMARY) else None
        state_update = result_dict

    record = CallRecord(
        model_name=model_name,
        role=role,
        condition=condition,
        lesion_id=lesion_id,
        image_id=image_id,
        attempt_timestamp_utc=attempt_ts,
        elapsed_s=outcome.elapsed_s,
        mem_before=mem_before,
        mem_after=mem_after,
        raw_response_text=raw_text,
        call_status=status,
        failure_cause=cause,
        classified_status=classified,
    )
    return record, state_update


def not_executed_record(model_name: str, role: str, condition: str, lesion_id: str, image_id: str, pause_reason: str) -> CallRecord:
    return CallRecord(
        model_name=model_name, role=role, condition=condition, lesion_id=lesion_id, image_id=image_id,
        attempt_timestamp_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=None,
        mem_before=None, mem_after=None, raw_response_text=None,
        call_status=STATUS_NOT_EXECUTED, failure_cause=f"papel_pausado:{pause_reason}", classified_status=None,
    )


def build_graph_state(image: Image.Image, image_path_label: str) -> GraphState:
    """``image`` já deve estar na condição (S0/S1/S2) correta -- esta função
    só monta o ``GraphState`` real com ``seg_image_data`` preenchido;
    ``image_path``/``lesion_size`` são campos exigidos pelo schema mas não
    usados pela lógica de analyze_lesion/summarize chamada aqui."""
    return GraphState(
        image_path=image_path_label,
        lesion_size=0.0,
        seg_image_data=convert_to_base64(image),
    )


# ---------------------------------------------------------------------------
# 10. Planejamento da matriz principal — exatamente 92 chamadas
# ---------------------------------------------------------------------------
def build_planned_calls(pilot12_rows: list[dict[str, str]], designated_image_ids: list[str]) -> list[dict[str, str]]:
    """``pilot12_rows``: as 12 linhas do manifesto do piloto (lesion_id,
    image_id,...). ``designated_image_ids``: os 4 image_id designados para
    multiagente S0/S1 (ver scripts/run_pilot12_sample.py).

    Devolve a lista ordenada de chamadas planejadas: VLM-direto x S0 (12),
    multiagente x S2 x 4 papéis (48), multiagente x {S0,S1} x 4 designados
    x 4 papéis (32). Total EXATO = 92, verificado por assert abaixo (não
    só no docstring)."""
    if len(pilot12_rows) != 12:
        raise ValueError(f"esperado exatamente 12 linhas, recebido {len(pilot12_rows)}")
    if len(set(designated_image_ids)) != 4:
        raise ValueError(f"esperado exatamente 4 image_id designados distintos, recebido {designated_image_ids}")

    calls: list[dict[str, str]] = []
    for row in pilot12_rows:
        calls.append({"role": ROLE_VLM_DIRETO, "condition": "s0", "lesion_id": row["lesion_id"], "image_id": row["image_id"]})

    for row in pilot12_rows:
        for role in MULTIAGENT_ROLES_IN_ORDER:
            calls.append({"role": role, "condition": "s2", "lesion_id": row["lesion_id"], "image_id": row["image_id"]})

    designated_rows = [r for r in pilot12_rows if r["image_id"] in designated_image_ids]
    if len(designated_rows) != 4:
        raise ValueError(f"designated_image_ids não corresponde a 4 linhas do manifesto: {designated_image_ids}")
    for row in designated_rows:
        for condition in ("s0", "s1"):
            for role in MULTIAGENT_ROLES_IN_ORDER:
                calls.append({"role": role, "condition": condition, "lesion_id": row["lesion_id"], "image_id": row["image_id"]})

    assert len(calls) == 92, f"matriz planejada tem {len(calls)} chamadas, esperado exatamente 92"
    n_vlm_direto = sum(1 for c in calls if c["role"] == ROLE_VLM_DIRETO)
    n_s2_multiagent = sum(1 for c in calls if c["condition"] == "s2" and c["role"] != ROLE_VLM_DIRETO)
    n_s0_s1_designated = sum(1 for c in calls if c["condition"] in ("s0", "s1") and c["role"] != ROLE_VLM_DIRETO)
    assert n_vlm_direto == 12 and n_s2_multiagent == 48 and n_s0_s1_designated == 32
    return calls


# ---------------------------------------------------------------------------
# 11. Controles de envio de imagem (INTERPRETAÇÃO DE DESIGN #2)
# ---------------------------------------------------------------------------
CONTROL_IMAGE_SIZE = 400
CONTROL_SHAPE_SIZE = 120
CONTROL_MARGIN = 20

QUADRANT_ALIASES: dict[str, tuple[str, ...]] = {
    "superior-esquerdo": ("superior esquerdo", "superior-esquerdo", "canto superior esquerdo", "top left", "top-left"),
    "superior-direito": ("superior direito", "superior-direito", "canto superior direito", "top right", "top-right"),
    "inferior-esquerdo": ("inferior esquerdo", "inferior-esquerdo", "canto inferior esquerdo", "bottom left", "bottom-left"),
    "inferior-direito": ("inferior direito", "inferior-direito", "canto inferior direito", "bottom right", "bottom-right"),
}

IMAGE_CONTROL_SPECS: list[dict[str, Any]] = [
    {"name": "controle_1_vermelho_sup_esq", "quadrant": "superior-esquerdo", "color_name": "vermelho", "color_rgb": (220, 30, 30)},
    {"name": "controle_2_azul_inf_dir", "quadrant": "inferior-direito", "color_name": "azul", "color_rgb": (30, 60, 220)},
    {"name": "controle_3_verde_sup_dir", "quadrant": "superior-direito", "color_name": "verde", "color_rgb": (30, 180, 60)},
]

IMAGE_CONTROL_PROMPT = """
Esta é uma imagem de teste sintética (não é uma lesão de pele). Ela é
dividida em 4 quadrantes. Exatamente UM quadrante contém um quadrado
colorido sobre um fundo escuro; os outros 3 quadrantes estão vazios
(apenas o fundo escuro).

Responda apenas: (1) em qual quadrante está o quadrado colorido
(superior-esquerdo, superior-direito, inferior-esquerdo ou
inferior-direito) e (2) de qual cor ele é.
"""


def _quadrant_box(quadrant: str, size: int = CONTROL_IMAGE_SIZE, shape: int = CONTROL_SHAPE_SIZE, margin: int = CONTROL_MARGIN) -> tuple[int, int, int, int]:
    positions = {
        "superior-esquerdo": (margin, margin),
        "superior-direito": (size - shape - margin, margin),
        "inferior-esquerdo": (margin, size - shape - margin),
        "inferior-direito": (size - shape - margin, size - shape - margin),
    }
    x, y = positions[quadrant]
    return (x, y, x + shape, y + shape)


def build_image_sending_control_image(spec: dict[str, Any]) -> Image.Image:
    img = Image.new("RGB", (CONTROL_IMAGE_SIZE, CONTROL_IMAGE_SIZE), (15, 15, 15))
    draw = ImageDraw.Draw(img)
    draw.rectangle(_quadrant_box(spec["quadrant"]), fill=spec["color_rgb"])
    return img


def evaluate_image_control_response(raw_text: Optional[str], spec: dict[str, Any]) -> dict[str, Any]:
    text = (raw_text or "").lower()
    color_found = spec["color_name"] in text
    quadrant_aliases = QUADRANT_ALIASES[spec["quadrant"]]
    quadrant_found = any(alias in text for alias in quadrant_aliases)
    return {
        "name": spec["name"],
        "color_found": color_found,
        "quadrant_found": quadrant_found,
        "passed": color_found and quadrant_found,
    }


def evaluate_image_sending_controls(results: list[dict[str, Any]]) -> dict[str, Any]:
    """``results``: lista de ``{"spec": spec, "raw_text": raw_text}`` para
    cada controle executado. Confirma (a) cada resposta individualmente
    identifica corretamente SUA PRÓPRIA imagem (cor+quadrante), e (b) as
    respostas não são todas idênticas entre si (sinal de que o modelo
    poderia estar ignorando a imagem e respondendo de forma genérica)."""
    per_control = [evaluate_image_control_response(r["raw_text"], r["spec"]) for r in results]
    all_individually_passed = all(p["passed"] for p in per_control)
    distinct_raw_texts = {str(r["raw_text"]).strip().lower() for r in results}
    varies_with_content = len(distinct_raw_texts) > 1 or len(results) <= 1
    overall_pass = all_individually_passed and varies_with_content
    return {
        "per_control": per_control,
        "all_individually_passed": all_individually_passed,
        "responses_vary_with_image_content": varies_with_content,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# 12. Controles de síntese — 4 inputs fixos (INTERPRETAÇÃO DE DESIGN #3)
# ---------------------------------------------------------------------------
def _fixed_control_image() -> Image.Image:
    return Image.new("RGB", (64, 64), (128, 128, 128))


FIXED_SYNTHESIS_SCENARIOS: dict[str, dict[str, Any]] = {
    "unanime_positivo": {
        "diagnosis_abcd": ABCDDiagnosticAnswer(
            assimetria="2: assimetria em ambos os eixos, lesão claramente irregular.",
            bordas="7: bordas nítidas em quase todos os quadrantes.",
            cor="5: cinco cores presentes (branco/bege, vermelho/rosa, marrom claro, marrom escuro, preto).",
            estruturas="4: rede pigmentada, linhas ramificadas, pontos e glóbulos observados.",
            explicacao="Soma ponderada resulta em escore acima de 5.45 -- lesão maligna pelo critério ABCD.",
        ),
        "diagnosis_menzies": MenziesDiagnosticAnswer(
            positivas={"veu_azul_branco": "1", "multiplos_pontos_marrons": "1", "pseudopodes": "1", "streaming_radial": "0",
                       "despigmentacao_tipo_cicatriz": "0", "pontos_pretos_perifericos": "1", "multiplas_cores": "1"},
            negativas={"simetria": "0", "cor_unica": "0"},
            explicacao="Múltiplas características positivas presentes e nenhuma negativa -- melanoma pelo Método Menzies.",
        ),
        "diagnosis_spcl": SPCLDiagnosticAnswer(
            padrao_de_pigmentacao_atipico="Sim: padrão atípico claramente identificável.",
            veu_azul_branco_irregular="Sim: véu irregular presente.",
            vasos_atipicos="Não considerado neste critério.",
            padrao_de_reticulo_irregular="Sim.",
            padrao_de_globulos_irregulares="Sim.",
            hiperpigmentacao_localizada="Sim.",
            regressao="Não.",
            explicacao="Pontuação total 7, acima do limiar de 3 -- suspeita pelo SPCL.",
        ),
    },
    "unanime_negativo": {
        "diagnosis_abcd": ABCDDiagnosticAnswer(
            assimetria="0: nenhuma assimetria relevante.",
            bordas="1: bordas indistintas na maioria dos quadrantes.",
            cor="1: apenas marrom claro presente.",
            estruturas="1: apenas rede pigmentada regular observada.",
            explicacao="Soma ponderada bem abaixo de 4.75 -- lesão benigna pelo critério ABCD.",
        ),
        "diagnosis_menzies": MenziesDiagnosticAnswer(
            positivas={"veu_azul_branco": "0", "multiplos_pontos_marrons": "0", "pseudopodes": "0", "streaming_radial": "0",
                       "despigmentacao_tipo_cicatriz": "0", "pontos_pretos_perifericos": "0", "multiplas_cores": "0"},
            negativas={"simetria": "1", "cor_unica": "1"},
            explicacao="Nenhuma característica positiva e ambas as negativas presentes -- lesão benigna pelo Método Menzies.",
        ),
        "diagnosis_spcl": SPCLDiagnosticAnswer(
            padrao_de_pigmentacao_atipico="Não.",
            veu_azul_branco_irregular="Não.",
            vasos_atipicos="Não.",
            padrao_de_reticulo_irregular="Não.",
            padrao_de_globulos_irregulares="Não.",
            hiperpigmentacao_localizada="Não.",
            regressao="Não.",
            explicacao="Pontuação total 0, abaixo do limiar de 3 -- sem suspeita pelo SPCL.",
        ),
    },
    "conflitante": {
        "diagnosis_abcd": ABCDDiagnosticAnswer(
            assimetria="0: nenhuma assimetria relevante.",
            bordas="2: bordas indistintas na maior parte.",
            cor="1: apenas marrom claro presente.",
            estruturas="1: apenas rede pigmentada regular observada.",
            explicacao="Soma ponderada abaixo de 4.75 -- lesão benigna pelo critério ABCD.",
        ),
        "diagnosis_menzies": MenziesDiagnosticAnswer(
            positivas={"veu_azul_branco": "1", "multiplos_pontos_marrons": "1", "pseudopodes": "1", "streaming_radial": "1",
                       "despigmentacao_tipo_cicatriz": "0", "pontos_pretos_perifericos": "1", "multiplas_cores": "1"},
            negativas={"simetria": "0", "cor_unica": "0"},
            explicacao="Múltiplas características positivas presentes -- melanoma pelo Método Menzies.",
        ),
        "diagnosis_spcl": SPCLDiagnosticAnswer(
            padrao_de_pigmentacao_atipico="Não.",
            veu_azul_branco_irregular="Não.",
            vasos_atipicos="Não.",
            padrao_de_reticulo_irregular="Sim.",
            padrao_de_globulos_irregulares="Não.",
            hiperpigmentacao_localizada="Não.",
            regressao="Não.",
            explicacao="Pontuação total 1, abaixo do limiar de 3 -- sem suspeita pelo SPCL (borderline).",
        ),
    },
    "um_parecer_ausente": {
        "diagnosis_abcd": ABCDDiagnosticAnswer(
            assimetria="0: nenhuma assimetria relevante.",
            bordas="1: bordas indistintas.",
            cor="1: apenas marrom claro presente.",
            estruturas="1: apenas rede pigmentada regular.",
            explicacao="Soma ponderada abaixo de 4.75 -- lesão benigna pelo critério ABCD.",
        ),
        "diagnosis_menzies": MenziesDiagnosticAnswer(
            positivas={"veu_azul_branco": "0", "multiplos_pontos_marrons": "0", "pseudopodes": "0", "streaming_radial": "0",
                       "despigmentacao_tipo_cicatriz": "0", "pontos_pretos_perifericos": "0", "multiplas_cores": "0"},
            negativas={"simetria": "1", "cor_unica": "1"},
            explicacao="Nenhuma característica positiva, ambas negativas presentes -- lesão benigna pelo Método Menzies.",
        ),
        # SPCL ausente -- reusa o mecanismo existente de status_spcl/A05
        # (ver src/agents/schemas.py::GraphState, src/agents/agents.py).
        "diagnosis_spcl": None,
        "status_spcl": AgentStatus(status="sem_resposta", causa="controle_sintetico_parecer_ausente"),
    },
}


# INTERPRETAÇÃO DE DESIGN #4: vocabulário fechado de termos clínicos usados
# nos 3 prompts reais (agents.py) -- heurística de "achado novo": qualquer
# termo desta lista presente na JUSTIFICATIVA do sumarizador que não
# aparece (nem como substring) em NENHUMA das 3 (ou 2) explicações de
# entrada é reportado como candidato a achado introduzido pelo sumarizador.
# Esta é uma aproximação lexical, não semântica -- não prova ausência real
# de alucinação clínica, só torna visível um sinal objetivo e reproduzível
# para o pesquisador revisar manualmente. Precisa de confirmação antes de
# ser tratada como critério de aprovação automática.
_CLINICAL_VOCABULARY = (
    "véu azul-branco", "veu azul-branco", "pseudópodes", "pseudopodes", "streaming radial",
    "despigmentação tipo cicatriz", "pontos pretos periféricos", "rede pigmentada",
    "linhas ramificadas", "glóbulos", "globulos", "retículo irregular", "reticulo irregular",
    "vasos atípicos", "vasos atipicos", "hiperpigmentação localizada", "hiperpigmentacao localizada",
    "regressão", "regressao", "padrão de pigmentação atípico", "padrao de pigmentacao atipico",
)


def _vocabulary_terms_not_in_inputs(justificativa: str, scenario: dict[str, Any]) -> list[str]:
    input_texts = []
    for key in ("diagnosis_abcd", "diagnosis_menzies", "diagnosis_spcl"):
        value = scenario.get(key)
        if value is not None:
            input_texts.append(str(getattr(value, "explicacao", value)).lower())
    combined_inputs = " ".join(input_texts)
    justificativa_lower = (justificativa or "").lower()
    return [term for term in _CLINICAL_VOCABULARY if term in justificativa_lower and term not in combined_inputs]


def build_synthesis_control_state(scenario_name: str) -> GraphState:
    scenario = FIXED_SYNTHESIS_SCENARIOS[scenario_name]
    return GraphState(
        image_path="controle_sintetico",
        lesion_size=0.0,
        seg_image_data=convert_to_base64(_fixed_control_image()),
        diagnosis_abcd=scenario["diagnosis_abcd"],
        diagnosis_menzies=scenario["diagnosis_menzies"],
        diagnosis_spcl=scenario.get("diagnosis_spcl"),
        status_spcl=scenario.get("status_spcl"),
    )


def run_synthesis_control(model_name: str, scenario_name: str, timeout_s: float = TIMEOUT_S) -> dict[str, Any]:
    state = build_synthesis_control_state(scenario_name)
    record, state_update = run_multiagent_role_call(
        model_name=model_name, role=ROLE_SUMMARY, state=state,
        lesion_id="controle_sintetico", image_id=f"controle_{scenario_name}",
        condition="controle_sintese", timeout_s=timeout_s,
    )
    validation = state_update.get("validation")
    classified = classify_case_status({"validation": validation})
    parseable = classified not in ("ausente", "nao_reconhecido")

    justificativa = ""
    if validation is not None:
        justificativa = getattr(validation, "justificativa", None) or str(validation)
    novel_terms = _vocabulary_terms_not_in_inputs(justificativa, FIXED_SYNTHESIS_SCENARIOS[scenario_name])

    return {
        "scenario": scenario_name,
        "call_record": record,
        "classified_status": classified,
        "parseable": parseable,
        "novel_clinical_terms_candidates": novel_terms,
        "no_novel_terms_detected": len(novel_terms) == 0,
        "passed": parseable and len(novel_terms) == 0,
    }


# ---------------------------------------------------------------------------
# 13. Classificação A34 — idêntica para VLM-direto e multiagente-sumarizador
# ---------------------------------------------------------------------------
def classify_binary_coverage(classified_status: Optional[str]) -> Optional[bool]:
    """``True`` se positivo, ``False`` se negativo, ``None`` se
    indeterminado/ausente/não_reconhecido/não_executado -- cobertura
    binária exclui tudo que não seja positivo/negativo (mesma convenção
    de src.evaluation.stage6_metrics)."""
    if classified_status == "positivo":
        return True
    if classified_status == "negativo":
        return False
    return None


# ---------------------------------------------------------------------------
# 14. "Crescimento sustentado de swap" (INTERPRETAÇÃO DE DESIGN #6)
# ---------------------------------------------------------------------------
def detect_sustained_swap_growth(swap_samples_bytes: list[int], floor_bytes: int = 0) -> dict[str, Any]:
    """Definição ADOTADA (precisa de aval do pesquisador antes de valer
    como critério de aprovação/reprovação do piloto): "crescimento
    sustentado" = sequência MONOTÔNICA NÃO-DECRESCENTE de amostras de swap
    ao longo de TODAS as chamadas realizadas, estritamente acima de
    ``floor_bytes`` em algum ponto (um piso abaixo do qual flutuações são
    ignoradas, documentado explicitamente, não decidido silenciosamente -- o
    valor default 0 aqui NÃO é uma recomendação, é só um valor neutro para
    permitir chamar a função sem argumento quando o pesquisador ainda não
    tiver fixado um piso real).
    """
    above_floor = [s for s in swap_samples_bytes if s > floor_bytes]
    is_monotonic_non_decreasing = all(
        swap_samples_bytes[i] <= swap_samples_bytes[i + 1] for i in range(len(swap_samples_bytes) - 1)
    )
    return {
        "floor_bytes": floor_bytes,
        "n_samples": len(swap_samples_bytes),
        "n_samples_above_floor": len(above_floor),
        "is_monotonic_non_decreasing_across_all_calls": is_monotonic_non_decreasing,
        "sustained_growth_detected": is_monotonic_non_decreasing and len(above_floor) > 0,
        "min_bytes": min(swap_samples_bytes) if swap_samples_bytes else None,
        "max_bytes": max(swap_samples_bytes) if swap_samples_bytes else None,
        "definition_needs_researcher_signoff": True,
    }
