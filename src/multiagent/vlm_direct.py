"""Etapa 5, segunda entrega (REWORK) — camada de chamada a VLMs locais via
Ollama (VLM-direto e multiagente-sobre-Ollama), controles de pré-execução,
lógica de abstenção sistemática, e classificação A34 — para o piloto de 12
lesões fora da amostra congelada (ver
`docs/reconstruction/20_local_vlm_pilot.md`).

Este é o REWORK da segunda entrega original (commit `b097037` em
`feature/inspection`), incorporando as 10 decisões fechadas pelo
pesquisador sobre o primeiro rascunho (ver seção 5 do doc, "decisões
fechadas pelo pesquisador"). Mudanças de arquitetura relevantes nesta
versão (não apenas cosméticas):
  - controles de envio de imagem agora são POR PAPEL VISUAL (4 papéis x 3
    controles = 12 chamadas, não 3 compartilhadas);
  - um novo controle "sem imagem" (texto puro, sem entrada `image_url`);
  - total de chamadas planejadas por modelo passa a ser 108 (92 da matriz
    + 16 de controle), não 92;
  - a heurística de "achado clínico novo" deixou de ser pass/fail
    automático — agora é só sinal de triagem, com um gate de revisão
    manual explícito antes da matriz principal (ver
    `scripts/run_local_vlm_pilot.py`);
  - a regra de "crescimento sustentado de swap" deixou de ser monotonia
    sobre TODAS as chamadas — agora é definida sobre janelas de 5 amostras
    da MATRIZ PRINCIPAL, com baseline explícito pós-controles;
  - a classificação de recusa ficou mais estrita (recusa só se não houver
    também um parecer categórico válido na mesma resposta).

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
  - Texto exato da instrução de tarefa do VLM-direto, desenho dos
    controles de envio de imagem por papel, os 4 cenários de síntese
    (com o ajuste "não avaliado" no cenário conflitante), a heurística de
    recusa (revisada), a definição de crescimento sustentado de swap
    (nova), e a designação estratificada dos "4 de 12" — todas fechadas
    pelo pesquisador, ver seção 5 do doc.

Interpretações de design NOVAS introduzidas neste rework, ainda sem
confirmação explícita do pesquisador (distintas das 10 já fechadas — ver
também o relatório desta entrega):
  (11) a fraseologia aceita como reconhecimento de "não há imagem" no
       controle sem imagem (`_NO_IMAGE_ACK_MARKERS`);
  (12) os dois vocabulários de controle de imagem reusados para os 4
       papéis visuais (mesmas duas combinações cor+quadrante, só
       renomeadas por papel — não um conjunto de 8 combinações distintas);
  (13) o texto exato da sub-semente do 4º sorteio de `run_pilot12_sample.py`
       (`FOURTH_SLOT_SEED`) e do mecanismo de redistribuição de vaga
       quando um grupo diagnóstico não tem nenhuma lesão com máscara de
       referência válida.

ADENDO (2026-11-13, pós-execução real do Estágio 1 para
qwen3-vl:8b-instruct-q4_K_M): `CallRecord` ganhou um campo ADITIVO
`classified_status_v2` -- saída do NOVO classificador versionado
`src.evaluation.case_status_v2.classify_case_status_v2`, calculada em
paralelo a `classified_status` (v1, `classify_case_status`, intocado)
nos mesmos pontos (`run_vlm_direct_call`, `run_multiagent_role_call`,
`run_synthesis_control`). Motivação, desenho e limitações do v2: ver
`src/evaluation/case_status_v2.py` e
`docs/reconstruction/20_local_vlm_pilot.md`. Nenhum gate existente
(pausa/abstenção, "passed" dos controles de síntese, binary_coverage do
relatório) passou a depender de `classified_status_v2` -- é só mais um
campo reportado ao lado do v1.
"""
from __future__ import annotations

import concurrent.futures
import statistics
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
from src.evaluation.case_status import _KEYWORD_TO_STATUS, classify_case_status
from src.evaluation.case_status_v2 import PARSER_VERSION as CASE_STATUS_V2_PARSER_VERSION
from src.evaluation.case_status_v2 import classify_case_status_v2
from src.models.model import convert_to_base64
from src.multiagent.presentation import images_are_effectively_equal

# Vocabulário das 3 categorias de parecer válido, reusado tal como
# `classify_case_status` o usa -- ITEM 5 da entrega exige reusar esta
# lista (não hardcodar uma segunda cópia) para decidir se uma resposta com
# um marcador de recusa também contém um parecer categórico válido.
_VALID_PARECER_KEYWORDS: tuple[str, ...] = tuple(kw for kw, _ in _KEYWORD_TO_STATUS)

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
# 1. Instrução de tarefa do VLM-direto -- DECISÃO FECHADA #1
# ---------------------------------------------------------------------------
# Aprovado pelo pesquisador: mesmas 3 categorias/mesmo significado do
# sumarizador. Texto livre (NÃO JSON), sem pedir aplicação de
# ABCD/Menzies/SPCL -- o VLM-direto, por definição, não passa pelos 3
# algoritmos. O texto bruto é preservado e reclassificado post-hoc pela
# MESMA `classify_case_status` usada para o sumarizador (reuso unmodified,
# já era assim e continua assim).
#
# NOTA DE REGISTRO (não é mudança de comportamento de código -- registro
# exigido pelo pesquisador): o status "negativo" devolvido por
# `classify_case_status` vem literalmente da palavra-chave "lesão
# benigna" no texto. Nesta comparação A34, "negativo" está sendo usado
# como PROXY do `binary_target` "não_melanoma" -- convenção já existente
# no projeto (não introduzida por esta instrução de VLM-direto). Sempre
# que o relatório discutir cobertura binária, "negativo" deve ser lido
# como esse proxy, não como um diagnóstico literal de benignidade.
VLM_DIRECT_TASK_INSTRUCTION = """
<-- Identidade do Agente -->
Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.

<-- Instruções -->
Analise a imagem dermatoscópica fornecida para a tarefa melanoma versus
não melanoma. Dê um parecer breve em português e encerre com uma das três
categorias abaixo (exatamente como escritas, para permitir classificação
automática da sua resposta):
  - "Suspeita de Melanoma"
  - "Lesão Benigna"
  - "Lesão Indeterminada"

Se a imagem não sustentar uma decisão binária, use a categoria
indeterminada -- isto é uma resposta válida, não uma recusa.
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
# DECISÃO FECHADA #5 (revisão da lista original -- INTERPRETAÇÃO DE DESIGN
# ainda aberta, só a LÓGICA de quando um marcador vale como recusa foi
# fechada pelo pesquisador): lista de marcadores textuais de recusa
# explícita. Heurística simples (substring, case-insensitive) -- não há
# especificação prévia no projeto para isto; precisa de confirmação do
# pesquisador antes de uso real, em especial para texto em português.
#
# Revisão: "não consigo distinguir pela imagem" foi deliberadamente NÃO
# incluída aqui (e não estava na lista original) -- é linguagem legítima
# de um parecer indeterminado (classify_case_status classificaria como
# "indeterminado" via "lesão indeterminada" se o modelo também emitir a
# categoria exigida), não uma recusa. Revisado e confirmado como ausente
# nesta entrega, não apenas assumido.
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


def _contains_valid_parecer(text_lower: str) -> bool:
    """``True`` se ``text_lower`` contém um dos 3 rótulos categóricos que
    `classify_case_status` reconhece -- reusa o MESMO vocabulário (import
    de `src.evaluation.case_status._KEYWORD_TO_STATUS`), não uma segunda
    cópia hardcoded."""
    return any(keyword in text_lower for keyword in _VALID_PARECER_KEYWORDS)


def classify_call_outcome(outcome: TimedCallOutcome) -> tuple[str, str, Optional[str]]:
    """Devolve ``(status, causa, raw_text)``.

    ``status`` é um de STATUS_OK/STATUS_TECHNICAL_FAILURE/STATUS_NO_RESPONSE/
    STATUS_REFUSAL. ``causa`` é uma string curta para log. ``raw_text`` é o
    texto bruto devolvido quando houver (preservado mesmo em falha parcial).

    DECISÃO FECHADA #5: um marcador de recusa só classifica a resposta
    como STATUS_REFUSAL se o texto NÃO contiver também um parecer válido
    (um dos 3 rótulos categóricos). Se um marcador de recusa aparecer MAS
    a resposta também emitir um parecer categórico, a resposta é
    STATUS_OK (o conteúdo é classificado normalmente por
    `classify_case_status` depois, fora desta função).
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
            if _contains_valid_parecer(lowered):
                # Marcador de recusa presente, mas a resposta também
                # contém um parecer categórico válido -- não classificar
                # como recusa (decisão fechada #5). Segue para OK abaixo.
                break
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
    # NOVO (aditivo, nunca substitui classified_status acima): saída do
    # classificador v2 "conclusão final"
    # (src.evaluation.case_status_v2.classify_case_status_v2), calculada
    # EM PARALELO ao v1 para o mesmo texto bruto, nos mesmos pontos de
    # chamada -- nunca usada para decidir pausa/abstenção (isso continua
    # lendo só `call_status`, ver AbstentionTracker.register em
    # scripts/run_local_vlm_pilot.py) nem para substituir
    # `classified_status` em nenhum relatório histórico.
    classified_status_v2: Optional[str] = None

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
            "classified_status_v2": self.classified_status_v2,
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
    # Aditivo: v2 calculado em paralelo, mesmo gate (status == STATUS_OK),
    # mesmo texto bruto -- nunca substitui `classified` (v1) acima.
    classified_v2 = classify_case_status_v2({"validation": raw_text}) if status == STATUS_OK else None

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
        classified_status_v2=classified_v2,
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
        classified_v2 = None
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
        # Aditivo: v2 calculado em paralelo ao v1, MESMO gate (só papel
        # sumarizador, mesma condição de STATUS_OK) -- aplicado
        # IDENTICAMENTE ao v1, nunca um caso especial por papel.
        classified_v2 = classify_case_status_v2({"validation": raw_text}) if (status == STATUS_OK and role == ROLE_SUMMARY) else None
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
        classified_status_v2=classified_v2,
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
#
# NOVO PARÂMETRO FECHADO (ver doc, "108 chamadas/teto de 4h"): o TOTAL de
# chamadas planejadas por execução de modelo (controles + matriz) passa a
# ser 108 = 92 (matriz, assert abaixo INALTERADO) + 16 de controle
# (12 de envio de imagem: 4 papéis x 3 + 4 de síntese). As 92 da matriz em
# si NÃO mudam -- só a contagem "total da execução" reportada muda de 92
# para 108. 4h é um TETO OPERACIONAL para a execução completa de um
# modelo (controles + matriz), não uma meta de desempenho -- passar do
# teto não estabelece por si só viabilidade para os 3.000 reais (ver
# projeção de duração do item 10 da entrega / seção 10 do doc).
# ---------------------------------------------------------------------------
N_IMAGE_SENDING_CONTROL_CALLS = 4 * 3  # 4 papéis visuais (VISUAL_ROLES, definido mais abaixo) x 3 controles = 12
N_SYNTHESIS_CONTROL_CALLS = 4
N_CONTROL_CALLS_TOTAL = N_IMAGE_SENDING_CONTROL_CALLS + N_SYNTHESIS_CONTROL_CALLS  # 16
N_MAIN_MATRIX_CALLS = 92
N_TOTAL_PLANNED_CALLS_PER_MODEL = N_MAIN_MATRIX_CALLS + N_CONTROL_CALLS_TOTAL  # 108
assert N_TOTAL_PLANNED_CALLS_PER_MODEL == 108


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
# 11. Controles de envio de imagem -- DECISÃO FECHADA #2: agora POR PAPEL
#     VISUAL (vlm_direto, diagnostic_abcd, diagnostic_menzies,
#     diagnostic_spcl). Para CADA papel: 2 controles de imagem (cor E
#     quadrante diferentes entre si) + 1 controle "sem imagem" (texto
#     puro) = 3 controles/papel x 4 papéis = 12 chamadas de controle de
#     imagem (não 3 compartilhadas como na primeira entrega).
# ---------------------------------------------------------------------------
CONTROL_IMAGE_SIZE = 400
CONTROL_SHAPE_SIZE = 120
CONTROL_MARGIN = 20

VISUAL_ROLES = (ROLE_VLM_DIRETO, ROLE_ABCD, ROLE_MENZIES, ROLE_SPCL)

QUADRANT_ALIASES: dict[str, tuple[str, ...]] = {
    "superior-esquerdo": ("superior esquerdo", "superior-esquerdo", "canto superior esquerdo", "top left", "top-left"),
    "superior-direito": ("superior direito", "superior-direito", "canto superior direito", "top right", "top-right"),
    "inferior-esquerdo": ("inferior esquerdo", "inferior-esquerdo", "canto inferior esquerdo", "bottom left", "bottom-left"),
    "inferior-direito": ("inferior direito", "inferior-direito", "canto inferior direito", "bottom right", "bottom-right"),
}

# INTERPRETAÇÃO DE DESIGN #12 (nova): as 2 combinações cor+quadrante abaixo
# diferem em AMBOS os eixos uma da outra (exigido pela decisão #2) e são
# REUSADAS idênticas para os 4 papéis visuais -- só o nome/rótulo do
# controle é prefixado pelo papel (não são 8 combinações distintas). Isto
# é uma escolha de conveniência deste rework, não uma regra ditada
# previamente; precisa de confirmação do pesquisador como as demais
# interpretações novas.
_BASE_IMAGE_CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {"quadrant": "superior-esquerdo", "color_name": "vermelho", "color_rgb": (220, 30, 30)},
    {"quadrant": "inferior-direito", "color_name": "azul", "color_rgb": (30, 60, 220)},
)


def build_image_control_specs_for_role(role: str) -> list[dict[str, Any]]:
    """Devolve os 2 specs de imagem (cor+quadrante) para ``role``, com nome
    prefixado pelo papel para facilitar leitura do log -- mesmas 2
    combinações de `_BASE_IMAGE_CONTROL_SPECS` para todos os papéis."""
    specs = []
    for i, base in enumerate(_BASE_IMAGE_CONTROL_SPECS, start=1):
        spec = dict(base)
        spec["name"] = f"controle_imagem_{role}_{i}_{base['color_name']}_{base['quadrant']}"
        spec["role"] = role
        specs.append(spec)
    return specs


# Todas as specs de imagem dos 4 papéis, para conveniência de quem quiser
# a lista completa (12 entradas: 4 papéis x 2 specs de imagem; o 3º
# controle de cada papel é o "sem imagem", que não tem spec de imagem).
IMAGE_CONTROL_SPECS_BY_ROLE: dict[str, list[dict[str, Any]]] = {
    role: build_image_control_specs_for_role(role) for role in VISUAL_ROLES
}

IMAGE_CONTROL_PROMPT = """
Esta é uma imagem de teste sintética (não é uma lesão de pele). Ela é
dividida em 4 quadrantes. Exatamente UM quadrante contém um quadrado
colorido sobre um fundo escuro; os outros 3 quadrantes estão vazios
(apenas o fundo escuro).

Responda apenas: (1) em qual quadrante está o quadrado colorido
(superior-esquerdo, superior-direito, inferior-esquerdo ou
inferior-direito) e (2) de qual cor ele é.
"""

# DECISÃO FECHADA #2: controle "sem imagem" -- mensagem SÓ texto, sem
# nenhuma entrada `image_url` no payload (ver `_text_only_control_message`
# abaixo). O modelo deve indicar explicitamente que não recebeu/não
# consegue ver uma imagem.
NO_IMAGE_CONTROL_PROMPT = """
Esta é uma mensagem de teste. Não há nenhuma imagem anexada a esta
mensagem. Responda apenas: em qual quadrante estaria um quadrado colorido
e de qual cor ele seria, com base na imagem que acompanha esta mensagem.
"""

# INTERPRETAÇÃO DE DESIGN #11 (nova): fraseologia aceita como o modelo
# reconhecendo explicitamente a ausência de imagem. Heurística de
# substring (case-insensitive) -- pode não cobrir todas as formas reais de
# um VLM local dizer "não há imagem"; precisa de confirmação do
# pesquisador, mesmo padrão das demais heurísticas de marcador textual
# deste módulo.
_NO_IMAGE_ACK_MARKERS: tuple[str, ...] = (
    "não há imagem",
    "nenhuma imagem",
    "não recebi imagem",
    "não recebi nenhuma imagem",
    "não foi fornecida nenhuma imagem",
    "não consigo ver",
    "não consigo visualizar",
    "sem imagem",
    "no image",
    "there is no image",
    "i don't see an image",
    "i do not see an image",
    "i cannot see an image",
    "no image was provided",
    "no image provided",
)


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


def _text_only_control_message(text: str) -> HumanMessage:
    """Mensagem de controle "sem imagem" -- NENHUMA entrada ``image_url``
    no payload (diferente de `_image_message`, que sempre embute uma
    imagem). Usada para o 3º controle de cada papel visual."""
    return HumanMessage(content=[{"type": "text", "text": text}])


def evaluate_image_control_response(raw_text: Optional[str], spec: dict[str, Any]) -> dict[str, Any]:
    text = (raw_text or "").lower()
    color_found = spec["color_name"] in text
    quadrant_aliases = QUADRANT_ALIASES[spec["quadrant"]]
    quadrant_found = any(alias in text for alias in quadrant_aliases)
    return {
        "name": spec["name"],
        "role": spec.get("role"),
        "kind": "imagem",
        "color_found": color_found,
        "quadrant_found": quadrant_found,
        "passed": color_found and quadrant_found,
    }


def evaluate_no_image_control_response(raw_text: Optional[str], role: str, name: str) -> dict[str, Any]:
    """O controle "sem imagem" passa se o texto contiver algum marcador de
    `_NO_IMAGE_ACK_MARKERS` (INTERPRETAÇÃO DE DESIGN #11)."""
    text = (raw_text or "").lower()
    acknowledged = any(marker in text for marker in _NO_IMAGE_ACK_MARKERS)
    return {
        "name": name,
        "role": role,
        "kind": "sem_imagem",
        "acknowledged_no_image": acknowledged,
        "passed": acknowledged,
    }


def verify_image_control_payload_sent_correct_image(message: HumanMessage, expected_image: Image.Image) -> bool:
    """Decodifica a imagem base64 realmente embutida em ``message`` e
    confirma que corresponde a ``expected_image`` -- reusa
    `src.multiagent.presentation.images_are_effectively_equal` (mesmo
    padrão de comparação já usado pela "dublê de payload" da primeira
    entrega da Etapa 5), não reimplementa comparação de imagem.
    """
    image_entries = [item for item in message.content if item.get("type") == "image_url"]
    if len(image_entries) != 1:
        return False
    import base64
    from io import BytesIO

    data_url = image_entries[0]["image_url"]["url"]
    b64_data = data_url.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64_data)
    sent_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    return images_are_effectively_equal(sent_image, expected_image)


def verify_no_image_control_payload_has_no_image(message: HumanMessage) -> bool:
    """``True`` se ``message`` não tiver NENHUMA entrada ``image_url`` --
    confirma por construção que o controle "sem imagem" realmente não
    envia imagem nenhuma."""
    return not any(item.get("type") == "image_url" for item in message.content)


def evaluate_image_sending_controls_for_role(results: list[dict[str, Any]]) -> dict[str, Any]:
    """``results``: lista de dicts, um por controle executado para UM
    papel visual, cada um com ``{"kind": "imagem"|"sem_imagem", "spec":
    spec_ou_None, "raw_text": raw_text, "payload_correct": bool}``.
    Espera exatamente 3 resultados (2 de imagem + 1 sem imagem) por papel,
    mas não impõe isso aqui (quem chama em `run_local_vlm_pilot.py`
    garante a composição)."""
    per_control = []
    for r in results:
        if r["kind"] == "imagem":
            evaluation = evaluate_image_control_response(r["raw_text"], r["spec"])
        else:
            evaluation = evaluate_no_image_control_response(r["raw_text"], r["role"], r["name"])
        evaluation["payload_correct"] = r.get("payload_correct")
        per_control.append(evaluation)

    all_individually_passed = all(p["passed"] for p in per_control)
    all_payloads_correct = all(p["payload_correct"] for p in per_control)
    image_raw_texts = {str(r["raw_text"]).strip().lower() for r in results if r["kind"] == "imagem"}
    varies_with_content = len(image_raw_texts) > 1 or sum(1 for r in results if r["kind"] == "imagem") <= 1
    overall_pass = all_individually_passed and all_payloads_correct and varies_with_content
    return {
        "per_control": per_control,
        "all_individually_passed": all_individually_passed,
        "all_payloads_correct": all_payloads_correct,
        "responses_vary_with_image_content": varies_with_content,
        "overall_pass": overall_pass,
    }


def evaluate_image_sending_controls(results_by_role: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """``results_by_role``: ``{papel: [resultados do papel, ver
    evaluate_image_sending_controls_for_role]}`` para os 4 papéis visuais.
    Passa geral SE E SOMENTE SE os 4 papéis passarem individualmente --
    um papel cujo caminho multimodal falhe não deve ser mascarado pelo
    sucesso dos outros 3."""
    per_role = {role: evaluate_image_sending_controls_for_role(results) for role, results in results_by_role.items()}
    overall_pass = all(e["overall_pass"] for e in per_role.values())
    return {
        "per_role": per_role,
        "n_roles_evaluated": len(per_role),
        "n_roles_passed": sum(1 for e in per_role.values() if e["overall_pass"]),
        "overall_pass": overall_pass,
    }


def _build_role_llm(model_name: str, role: str):
    """Constrói o LLM de ``role`` pelo MESMO caminho de construção que o
    papel usa na matriz principal -- para `vlm_direto`, ``ChatOllama(model=...)``
    direto (idêntico a `run_vlm_direct_call`); para os 3 papéis
    diagnósticos, instancia a classe de agente real (o mesmo `_build_llm`
    interno de `src.agents.agents` roda dentro do `__init__`) e devolve o
    atributo `.llm` já construído -- NUNCA chama `.analyze_lesion` (que
    monta sempre o prompt clínico real, intocado nesta entrega)."""
    if role == ROLE_VLM_DIRETO:
        return ChatOllama(model=model_name)
    agent_cls, schema = _ROLE_TO_AGENT_CLASS_AND_SCHEMA[role]
    agent = agent_cls(model_name, schema)
    return agent.llm


def run_image_sending_control_call(
    model_name: str,
    role: str,
    kind: str,
    spec: Optional[dict[str, Any]],
    timeout_s: float = TIMEOUT_S,
) -> dict[str, Any]:
    """Executa UM controle de envio de imagem para ``role`` (``kind`` é
    ``"imagem"`` ou ``"sem_imagem"``), pelo caminho de construção de LLM
    PRÓPRIO do papel (ver `_build_role_llm`) -- com o prompt/imagem de
    CONTROLE, nunca o prompt clínico real (nota do pesquisador: "Esses
    controles verificam o caminho multimodal; os prompts diagnósticos
    ficam intactos no piloto dermatoscópico").

    Devolve um dict com ``call_record`` (CallRecord), ``kind``, ``spec``,
    ``role``, ``name``, ``raw_text`` e ``payload_correct`` (bool --
    confirma que o payload realmente enviado continha a imagem esperada,
    ou nenhuma entrada de imagem no caso "sem_imagem"; nunca revela a
    resposta esperada via nome de arquivo/metadado enviado ao modelo).
    """
    llm = _build_role_llm(model_name, role)

    if kind == "imagem":
        control_image = build_image_sending_control_image(spec)
        message = _image_message(convert_to_base64(control_image), IMAGE_CONTROL_PROMPT)
        payload_correct = verify_image_control_payload_sent_correct_image(message, control_image)
        name = spec["name"]
    else:
        message = _text_only_control_message(NO_IMAGE_CONTROL_PROMPT)
        payload_correct = verify_no_image_control_payload_has_no_image(message)
        name = f"controle_imagem_{role}_3_sem_imagem"

    attempt_ts = datetime.now(timezone.utc).isoformat()
    outcome = call_with_timeout(lambda: llm.invoke([message]).content, timeout_s=timeout_s)
    status, cause, raw_text = classify_call_outcome(outcome)

    record = CallRecord(
        model_name=model_name, role=f"controle_imagem_{role}", condition=name,
        lesion_id="controle_sintetico", image_id=name,
        attempt_timestamp_utc=attempt_ts, elapsed_s=outcome.elapsed_s,
        mem_before=None, mem_after=None, raw_response_text=raw_text,
        call_status=status, failure_cause=cause, classified_status=None,
    )
    return {
        "call_record": record, "kind": kind, "spec": spec, "role": role,
        "name": name, "raw_text": raw_text, "payload_correct": payload_correct,
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
        # DECISÃO FECHADA #3: `vasos_atipicos` deste cenário foi alterado
        # para uma fraseologia inequívoca de "não avaliado" (NÃO "Sim"/
        # "Não" -- um critério avaliado-e-ausente seria "Não.", o que é
        # uma afirmação distinta de "não avaliado"). Isto é o que permite
        # testar se o sumarizador preserva a distinção entre "não
        # avaliado" e "avaliado e ausente" (ver
        # `_check_not_evaluated_not_misrepresented_as_absent` abaixo).
        "diagnosis_spcl": SPCLDiagnosticAnswer(
            padrao_de_pigmentacao_atipico="Não.",
            veu_azul_branco_irregular="Não.",
            vasos_atipicos="Não avaliado: este critério não foi avaliado nesta análise.",
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


# INTERPRETAÇÃO DE DESIGN #4 (mantida como interpretação de VOCABULÁRIO --
# só o PAPEL desta heurística no fluxo de aprovação foi fechado pelo
# pesquisador, ver DECISÃO FECHADA #4 abaixo): vocabulário fechado de
# termos clínicos usados nos 3 prompts reais (agents.py) -- heurística de
# "achado novo": qualquer termo desta lista presente na JUSTIFICATIVA do
# sumarizador que não aparece (nem como substring) em NENHUMA das 3 (ou 2)
# explicações de entrada é reportado como candidato a achado introduzido
# pelo sumarizador. Esta é uma aproximação lexical, não semântica -- não
# prova ausência real de alucinação clínica, só torna visível um sinal
# objetivo e reproduzível para o pesquisador revisar manualmente.
_CLINICAL_VOCABULARY = (
    "véu azul-branco", "veu azul-branco", "pseudópodes", "pseudopodes", "streaming radial",
    "despigmentação tipo cicatriz", "pontos pretos periféricos", "rede pigmentada",
    "linhas ramificadas", "glóbulos", "globulos", "retículo irregular", "reticulo irregular",
    "vasos atípicos", "vasos atipicos", "hiperpigmentação localizada", "hiperpigmentacao localizada",
    "regressão", "regressao", "padrão de pigmentação atípico", "padrao de pigmentacao atipico",
)

# DECISÃO FECHADA #4: estas duas listas são heurísticas lexicais
# APROXIMADAS (não provam ausência real do problema) para dois cuidados
# distintos ao redor da distinção "não avaliado" vs "avaliado e ausente":
#   - frases que afirmam EXPLICITAMENTE que um critério foi avaliado (uso
#     no cenário "conflitante", onde vasos_atipicos é "não avaliado" --
#     se a justificativa do sumarizador usar uma destas frases referindo-
#     se a esse critério, é candidato a ter confundido "não avaliado" com
#     "avaliado e ausente");
#   - frases que convertem ausência/não-avaliação em "ausente na lesão"
#     (uso no cenário "um_parecer_ausente", onde o parecer SPCL inteiro
#     está ausente -- a justificativa não deve alegar que a LESÃO carece
#     da característica, e sim que o PARECER está ausente).
_EVALUATED_AND_ABSENT_PHRASES: tuple[str, ...] = (
    "foi avaliado e está ausente", "avaliado e ausente", "critério avaliado",
    "foi avaliado e não", "avaliamos este critério", "avaliado, não",
)
_ABSENCE_AS_LESION_LACKS_FEATURE_PHRASES: tuple[str, ...] = (
    "ausente na lesão", "a lesão não apresenta", "a lesão não possui",
    "lesão não apresenta esse critério", "característica ausente na lesão",
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


def _check_not_evaluated_not_misrepresented_as_absent(justificativa: str) -> dict[str, Any]:
    """Heurística APROXIMADA (lexical, não semântica -- flagueada
    explicitamente, mesmo espírito da heurística de achado novo) para o
    cenário "conflitante": a justificativa do sumarizador não deveria
    alegar que o critério `vasos_atipicos` foi "avaliado e ausente" --
    ele foi marcado como "não avaliado" no input fixo. Não detecta
    menções implícitas/paráfrases que não usem uma das frases vigiadas."""
    text = (justificativa or "").lower()
    hits = [p for p in _EVALUATED_AND_ABSENT_PHRASES if p in text]
    return {
        "heuristic_is_approximate": True,
        "phrases_found": hits,
        "likely_misrepresented_as_evaluated_absent": len(hits) > 0,
    }


def _check_absence_not_misrepresented_as_lesion_lacks_feature(justificativa: str) -> dict[str, Any]:
    """Heurística APROXIMADA equivalente para o cenário "um_parecer_ausente":
    a justificativa não deveria confundir "parecer SPCL ausente" com "a
    lesão não tem essa característica"."""
    text = (justificativa or "").lower()
    hits = [p for p in _ABSENCE_AS_LESION_LACKS_FEATURE_PHRASES if p in text]
    return {
        "heuristic_is_approximate": True,
        "phrases_found": hits,
        "likely_misrepresented_absence_as_lesion_lacks_feature": len(hits) > 0,
    }


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
    """DECISÃO FECHADA #4: a heurística de vocabulário (`novel_clinical_terms_candidates`)
    NUNCA mais determina ``passed`` sozinha -- é só um sinal de TRIAGEM
    reportado para revisão manual do pesquisador (ver
    `build_synthesis_manual_review_template`/o fluxo de 2 estágios em
    `scripts/run_local_vlm_pilot.py`). ``passed`` aqui reflete só o
    critério automatizável (parsing bem-sucedido pela A34); os heurísticos
    lexicais (achado novo + as 2 checagens de "não avaliado") são
    reportados sempre, nunca usados para aprovar/reprovar sozinhos.

    GATE OFICIAL (decisão do pesquisador, 2026-11-14, verbatim em
    `docs/reconstruction/20_local_vlm_pilot.md` §9c): "parseable"/"passed"
    abaixo são definidos SÓ pelo classificador v2 ("conclusão final",
    `src.evaluation.case_status_v2.classify_case_status_v2`) -- v2 passou
    4/4 nos controles de síntese reais deste piloto (incluindo a correção
    esperada de `um_parecer_ausente`, antes `nao_reconhecido`/reprovado
    pelo v1, e a conclusão afirmativa de suspeita de melanoma de
    `unanime_positivo` que v2 perdia antes da extensão de 2026-11-14). v1
    (`classify_case_status`) continua computado e reportado ao lado --
    SÓ para rastreabilidade (campo `classified_status_v1_historical`
    abaixo) -- NUNCA como fallback/desempate: se v2 disser
    `"nao_reconhecido"`, isto é final, mesmo que v1 diria outra coisa
    (não há nenhum `v2_result or v1_result` nesta função).
    """
    state = build_synthesis_control_state(scenario_name)
    record, state_update = run_multiagent_role_call(
        model_name=model_name, role=ROLE_SUMMARY, state=state,
        lesion_id="controle_sintetico", image_id=f"controle_{scenario_name}",
        condition="controle_sintese", timeout_s=timeout_s,
    )
    validation = state_update.get("validation")
    # Rastreabilidade APENAS -- nunca lido para decidir "parseable"/"passed"
    # abaixo, nunca usado como fallback/desempate do resultado v2.
    classified_v1_historical = classify_case_status({"validation": validation})
    # GATE OFICIAL (ver docstring acima) -- "parseable"/"passed" dependem
    # só deste resultado.
    classified_v2 = classify_case_status_v2({"validation": validation})
    parseable = classified_v2 not in ("ausente", "nao_reconhecido")

    justificativa = ""
    if validation is not None:
        justificativa = getattr(validation, "justificativa", None) or str(validation)
    novel_terms = _vocabulary_terms_not_in_inputs(justificativa, FIXED_SYNTHESIS_SCENARIOS[scenario_name])

    result: dict[str, Any] = {
        "scenario": scenario_name,
        "call_record": record,
        # Gate oficial (ver docstring) -- fonte de "parseable"/"passed".
        "classified_status_v2": classified_v2,
        "official_classification_parser_version": CASE_STATUS_V2_PARSER_VERSION,
        # Rastreabilidade só -- nunca usado para decidir passed/parseable,
        # nunca fallback/desempate do v2.
        "classified_status_v1_historical": classified_v1_historical,
        "parseable": parseable,
        "justificativa": justificativa,
        "novel_clinical_terms_candidates": novel_terms,
        "no_novel_terms_detected": len(novel_terms) == 0,
        # "passed" automatizado = só parsing A34 pelo v2 -- achado novo é
        # triagem, nunca gate automático (decisão fechada #4).
        "passed": parseable,
        "requires_manual_review": True,
    }
    if scenario_name == "conflitante":
        result["not_evaluated_vs_evaluated_absent_check"] = _check_not_evaluated_not_misrepresented_as_absent(justificativa)
    if scenario_name == "um_parecer_ausente":
        result["absence_vs_lesion_lacks_feature_check"] = _check_absence_not_misrepresented_as_lesion_lacks_feature(justificativa)
    return result


# ---------------------------------------------------------------------------
# 12b. Template de revisão manual -- DECISÃO FECHADA #4: o driver escreve
#      este template e PARA após os 4 controles de síntese; a matriz
#      principal só é tentada numa segunda invocação, depois do
#      pesquisador preencher o veredito manual de cada cenário.
# ---------------------------------------------------------------------------
MANUAL_REVIEW_VERDICT_NO_NOVEL_FINDING = "sem_achado_novo"
MANUAL_REVIEW_VERDICT_NOVEL_FINDING_FOUND = "achado_novo_encontrado"
VALID_MANUAL_REVIEW_VERDICTS = (MANUAL_REVIEW_VERDICT_NO_NOVEL_FINDING, MANUAL_REVIEW_VERDICT_NOVEL_FINDING_FOUND)


def build_synthesis_manual_review_template(synthesis_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Um item por cenário de síntese, com os sinais de triagem já
    calculados e um placeholder para o pesquisador preencher à mão
    (``manual_review_verdict``: ``None`` até ser preenchido;
    ``reviewer_notes``: string livre). Nunca preenchido automaticamente."""
    items = []
    for r in synthesis_results:
        items.append({
            "scenario": r["scenario"],
            "justificativa": r.get("justificativa", ""),
            "novel_clinical_terms_candidates": r.get("novel_clinical_terms_candidates", []),
            "not_evaluated_vs_evaluated_absent_check": r.get("not_evaluated_vs_evaluated_absent_check"),
            "absence_vs_lesion_lacks_feature_check": r.get("absence_vs_lesion_lacks_feature_check"),
            "manual_review_verdict": None,
            "reviewer_notes": "",
        })
    return {
        "instructions": (
            "Revise manualmente cada cenário abaixo. Preencha "
            "'manual_review_verdict' com exatamente "
            f"'{MANUAL_REVIEW_VERDICT_NO_NOVEL_FINDING}' ou "
            f"'{MANUAL_REVIEW_VERDICT_NOVEL_FINDING_FOUND}' (strings exatas) "
            "e 'reviewer_notes' com observações livres. Depois rode "
            "run_local_vlm_pilot.py --synthesis-manual-review-file <este arquivo> "
            "para prosseguir à matriz principal."
        ),
        "valid_verdicts": list(VALID_MANUAL_REVIEW_VERDICTS),
        "scenarios": items,
    }


def evaluate_manual_review_file(manual_review_data: dict[str, Any]) -> dict[str, Any]:
    """Confere se TODOS os cenários têm
    ``manual_review_verdict == MANUAL_REVIEW_VERDICT_NO_NOVEL_FINDING`` --
    qualquer veredito ausente/``None``/diferente disso bloqueia a
    progressão (nunca assume/adivinha um veredito)."""
    scenarios = manual_review_data.get("scenarios", [])
    missing = [s["scenario"] for s in scenarios if s.get("manual_review_verdict") is None]
    novel_found = [s["scenario"] for s in scenarios if s.get("manual_review_verdict") == MANUAL_REVIEW_VERDICT_NOVEL_FINDING_FOUND]
    invalid = [
        s["scenario"] for s in scenarios
        if s.get("manual_review_verdict") is not None and s.get("manual_review_verdict") not in VALID_MANUAL_REVIEW_VERDICTS
    ]
    all_clear = len(scenarios) > 0 and not missing and not novel_found and not invalid
    return {
        "n_scenarios": len(scenarios),
        "missing_verdict_scenarios": missing,
        "novel_finding_scenarios": novel_found,
        "invalid_verdict_scenarios": invalid,
        "all_scenarios_cleared": all_clear,
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
# 14. "Crescimento sustentado de swap" -- DECISÃO FECHADA #6 (substitui
#     POR COMPLETO a regra monotônica-sobre-todas-as-chamadas da primeira
#     entrega).
# ---------------------------------------------------------------------------
# Constante OPERACIONAL deste piloto -- NÃO é um limite universal de
# máquina, é um limiar de triagem escolhido pelo pesquisador para este
# experimento especificamente (512 MiB acima do baseline pós-controles).
SWAP_GROWTH_THRESHOLD_BYTES = 512 * 1024 * 1024  # 512 MiB
SWAP_GROWTH_WINDOW_SIZE = 5


def detect_sustained_swap_growth(
    main_matrix_swap_samples_bytes: list[int],
    baseline_swap_bytes: int,
    window_size: int = SWAP_GROWTH_WINDOW_SIZE,
    growth_threshold_bytes: int = SWAP_GROWTH_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Definição FECHADA pelo pesquisador (decisão #6) -- implementada
    aqui como regra operacional do piloto, não mais pendente de aval:

      1. ``baseline_swap_bytes`` é um snapshot explícito capturado pelo
         chamador DEPOIS do modelo carregado e dos controles completos,
         ANTES da primeira chamada da matriz principal -- é um parâmetro
         OBRIGATÓRIO aqui, nunca derivado da própria lista de amostras.
      2. ``main_matrix_swap_samples_bytes`` contém SÓ as amostras
         coletadas DURANTE a matriz principal (nunca as dos controles).
      3. As amostras são particionadas em janelas NÃO-SOBREPOSTAS de
         ``window_size`` amostras consecutivas (default 5); uma janela
         incompleta ao final é DESCARTADA (não arredondada para baixo
         silenciosamente -- ``n_samples_discarded_incomplete_window``
         registra quantas amostras ficaram de fora).
      4. Mediana por janela.
      5. ``sustained_growth_detected = True`` SOMENTE SE existirem 2
         janelas CONSECUTIVAS cujas medianas sejam AMBAS
         >= ``baseline_swap_bytes + growth_threshold_bytes`` E a mediana
         da segunda janela seja ESTRITAMENTE MAIOR que a da primeira.
      6. Um padrão "pico então estável" (uma janela cruza o limiar, a
         próxima também está no/acima do limiar mas NÃO supera a mediana
         da anterior) é reportado SEPARADAMENTE em
         ``spike_then_stable_detected`` -- nunca contado como crescimento
         sustentado.

    Todas as janelas/medianas brutas são reportadas para inspeção, mesmo
    quando nenhum dos dois flags dispara.
    """
    n_complete_windows = len(main_matrix_swap_samples_bytes) // window_size if window_size > 0 else 0
    windows = [
        main_matrix_swap_samples_bytes[i * window_size:(i + 1) * window_size]
        for i in range(n_complete_windows)
    ]
    window_medians = [statistics.median(w) for w in windows]
    threshold = baseline_swap_bytes + growth_threshold_bytes

    sustained_growth_window_pairs: list[tuple[int, int]] = []
    spike_then_stable_window_pairs: list[tuple[int, int]] = []
    for i in range(len(window_medians) - 1):
        m0, m1 = window_medians[i], window_medians[i + 1]
        both_at_or_above_threshold = m0 >= threshold and m1 >= threshold
        if not both_at_or_above_threshold:
            continue
        if m1 > m0:
            sustained_growth_window_pairs.append((i, i + 1))
        else:
            spike_then_stable_window_pairs.append((i, i + 1))

    return {
        "baseline_swap_bytes": baseline_swap_bytes,
        "growth_threshold_bytes": growth_threshold_bytes,
        "window_size": window_size,
        "n_samples": len(main_matrix_swap_samples_bytes),
        "n_complete_windows": n_complete_windows,
        "n_samples_discarded_incomplete_window": len(main_matrix_swap_samples_bytes) - n_complete_windows * window_size,
        "window_medians_bytes": window_medians,
        "threshold_bytes_absolute": threshold,
        "sustained_growth_window_pairs": sustained_growth_window_pairs,
        "sustained_growth_detected": len(sustained_growth_window_pairs) > 0,
        "spike_then_stable_window_pairs": spike_then_stable_window_pairs,
        "spike_then_stable_detected": len(spike_then_stable_window_pairs) > 0,
        "definition_needs_researcher_signoff": False,  # decisão #6 já fechada -- reportado para inspeção, não mais pendente.
    }


# ---------------------------------------------------------------------------
# 15. Relatório informacional para ABCD/Menzies/SPCL -- DECISÃO FECHADA #9
#     (extensão): "zero parecer conclusivo" continua restrito a
#     vlm_direto/summary (único lugar onde classify_case_status se
#     aplica); para os 3 papéis diagnósticos SEM campo categórico,
#     reportamos (NUNCA um gate pass/fail) validade de resposta (já
#     coberta por call_status), recusas explícitas, e uma contagem
#     lexical informativa de "não avaliado"/"não considerado".
# ---------------------------------------------------------------------------
_NOT_EVALUATED_PHRASES: tuple[str, ...] = (
    "não avaliado", "nao avaliado", "não considerado", "nao considerado",
    "não avaliada", "nao avaliada", "não considerada", "nao considerada",
)


def count_not_evaluated_phrases(raw_text: Optional[str]) -> int:
    """Contagem lexical INFORMATIVA (não um veredito) de quantas vezes uma
    frase de `_NOT_EVALUATED_PHRASES` aparece no texto bruto de uma
    resposta ABCD/Menzies/SPCL -- mesmo estilo da heurística de achado
    novo (substring, case-insensitive)."""
    text = (raw_text or "").lower()
    return sum(text.count(phrase) for phrase in _NOT_EVALUATED_PHRASES)


def summarize_diagnostic_role_reporting(records: list["CallRecord"], role: str) -> dict[str, Any]:
    """``records``: todos os CallRecord da matriz principal para ``role``
    (um de ROLE_ABCD/ROLE_MENZIES/ROLE_SPCL). Devolve contagens
    informativas -- NUNCA aplica o flag "zero parecer conclusivo" (que só
    faz sentido para vlm_direto/summary, os únicos papéis com campo de
    diagnóstico categorizável por `classify_case_status`)."""
    role_records = [r for r in records if r.role == role]
    n_ok = sum(1 for r in role_records if r.call_status == STATUS_OK)
    n_refusal = sum(1 for r in role_records if r.call_status == STATUS_REFUSAL)
    n_not_evaluated_phrases_total = sum(count_not_evaluated_phrases(r.raw_response_text) for r in role_records)
    return {
        "role": role,
        "n_calls": len(role_records),
        "n_ok": n_ok,
        "n_explicit_refusals": n_refusal,
        "n_not_evaluated_phrases_total": n_not_evaluated_phrases_total,
        "note": (
            "Papel sem campo de diagnóstico categórico -- o flag 'zero "
            "parecer conclusivo' (roles_with_zero_conclusive_opinion) NÃO "
            "se aplica aqui por construção; ver decisão fechada #9."
        ),
    }


# ---------------------------------------------------------------------------
# 16. Projeção de duração em escala real -- DECISÃO FECHADA #10 (extensão):
#     soma do elapsed_s das 4 chamadas sequenciais (ABCD->Menzies->SPCL->
#     sumarizador) por lesão+condição, usada para projetar (não prometer)
#     uma duração aproximada de uma hipotética execução em escala real.
# ---------------------------------------------------------------------------
def compute_per_lesion_condition_elapsed_totals(records: list["CallRecord"]) -> dict[tuple[str, str], float]:
    """Soma `elapsed_s` das 4 chamadas sequenciais (ABCD, Menzies, SPCL,
    sumarizador) por par (lesion_id, condition) -- ignora chamadas com
    `elapsed_s is None` (ex.: `nao_executado`) somando 0.0 para essas."""
    totals: dict[tuple[str, str], float] = {}
    for r in records:
        if r.role not in MULTIAGENT_ROLES_IN_ORDER:
            continue
        key = (r.lesion_id, r.condition)
        totals[key] = totals.get(key, 0.0) + (r.elapsed_s or 0.0)
    return totals


def project_full_scale_duration(per_lesion_condition_totals: list[float], hypothetical_n: int) -> dict[str, Any]:
    """Projeta (NUNCA promete) a duração de uma hipotética execução de
    ``hypothetical_n`` casos a partir da média dos totais por lesão+
    condição MEDIDOS NESTE PILOTO (fórmula: média * hypothetical_n).

    Caveat explícito (sempre incluído no resultado, não só em docstring):
    esta projeção extrapola a partir dos tempos por chamada DO PILOTO
    (modelo já carregado, 12 lesões, máquina do pesquisador) -- não é uma
    promessa sobre o tempo real de uma execução de 3.000 lesões, que pode
    diferir por hardware, estado de carregamento do modelo entre
    execuções, contenção de recursos, e variação do próprio modelo local
    (sem seed garantido, ver decisão #8)."""
    if not per_lesion_condition_totals:
        return {
            "n_samples": 0,
            "mean_per_lesion_condition_elapsed_s": None,
            "hypothetical_n": hypothetical_n,
            "projected_total_s": None,
            "caveat": (
                "Sem amostras -- nenhuma projeção calculada. Projeção, quando "
                "calculável, extrapola só dos tempos por chamada DESTE "
                "PILOTO (modelo já carregado, 12 lesões); não é uma "
                "promessa de tempo real para 3.000 lesões (hardware/estado "
                "de carregamento/variação do modelo local podem diferir)."
            ),
        }
    mean_s = statistics.mean(per_lesion_condition_totals)
    projected_total_s = mean_s * hypothetical_n
    return {
        "n_samples": len(per_lesion_condition_totals),
        "mean_per_lesion_condition_elapsed_s": mean_s,
        "hypothetical_n": hypothetical_n,
        "projected_total_s": projected_total_s,
        "projected_total_hours": projected_total_s / 3600.0,
        "caveat": (
            "Projeção extrapola só dos tempos por chamada DESTE PILOTO "
            "(modelo já carregado, 12 lesões, máquina do pesquisador); NÃO "
            "é uma promessa de tempo real para uma execução de 3.000 "
            "lesões -- hardware, estado de carregamento do modelo entre "
            "execuções, contenção de recursos e variação do próprio modelo "
            "local (sem seed garantido) podem diferir substancialmente."
        ),
    }
