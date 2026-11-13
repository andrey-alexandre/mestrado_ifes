"""Testes com dados/LLM 100% sintéticos (mock no limite do Ollama -- nunca
requer um servidor Ollama real) para `src/multiagent/vlm_direct.py` (Etapa
5, segunda entrega: piloto local de VLM).

STATUS: implementados, aguardando execução do pesquisador
(`python -m pytest tests/multiagent/test_vlm_direct.py -v`). Não
executados pelo assistente/fork que os escreveu
(`feedback_no_autonomous_execution`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.agents.agents as agents_module  # noqa: E402
from src.agents.schemas import ABCDDiagnosticAnswer  # noqa: E402
from src.multiagent import vlm_direct as vd  # noqa: E402


# ---------------------------------------------------------------------------
# Dublês de LLM -- nenhuma rede, nenhum Ollama real.
# ---------------------------------------------------------------------------
class _FakeOllamaResponse:
    def __init__(self, content):
        self.content = content


class _FakeChatOllama:
    """Substitui langchain_ollama.ChatOllama -- usado via monkeypatch de
    `vd.ChatOllama`."""

    _next_content: str | Exception | None = "resposta padrão de teste"
    _sleep_s: float = 0.0

    def __init__(self, model=None, **kwargs):
        self.model = model

    def invoke(self, messages):
        if isinstance(self._next_content, Exception):
            raise self._next_content
        if self._sleep_s:
            import time

            time.sleep(self._sleep_s)
        return _FakeOllamaResponse(self._next_content)


class _FakeLLMForAgents:
    """Substitui o retorno de `src.agents.agents._build_llm` -- usado via
    monkeypatch de `agents_module._build_llm`. Como `model_name` contém
    ':' nos testes (convenção Ollama), `_invoke` chama `.invoke(messages)
    .content` diretamente (caminho Ollama em `_invoke`)."""

    next_content: str | None = "texto simulado"

    def invoke(self, messages):
        return _FakeOllamaResponse(type(self).next_content)


@pytest.fixture(autouse=True)
def _patch_llms(monkeypatch):
    monkeypatch.setattr(vd, "ChatOllama", _FakeChatOllama)
    monkeypatch.setattr(agents_module, "_build_llm", lambda model_name, temperature: _FakeLLMForAgents())
    _FakeChatOllama._next_content = "resposta padrão de teste"
    _FakeChatOllama._sleep_s = 0.0
    _FakeLLMForAgents.next_content = "texto simulado"
    yield


MODEL_TAG = "fake-model:test"


# ---------------------------------------------------------------------------
# build_planned_calls -- exatamente 92
# ---------------------------------------------------------------------------
def _fake_12_rows():
    return [{"lesion_id": f"HAM_{i:03d}", "image_id": f"ISIC_{i:03d}"} for i in range(12)]


def test_build_planned_calls_totals_exactly_92():
    rows = _fake_12_rows()
    designated = [r["image_id"] for r in sorted(rows, key=lambda r: r["lesion_id"])[:4]]
    calls = vd.build_planned_calls(rows, designated)
    assert len(calls) == 92
    assert sum(1 for c in calls if c["role"] == vd.ROLE_VLM_DIRETO) == 12
    assert sum(1 for c in calls if c["condition"] == "s2" and c["role"] != vd.ROLE_VLM_DIRETO) == 48
    assert sum(1 for c in calls if c["condition"] in ("s0", "s1") and c["role"] != vd.ROLE_VLM_DIRETO) == 32


def test_build_planned_calls_rejects_wrong_row_count():
    with pytest.raises(ValueError):
        vd.build_planned_calls(_fake_12_rows()[:11], ["ISIC_000"])


def test_build_planned_calls_rejects_wrong_designated_count():
    with pytest.raises(ValueError):
        vd.build_planned_calls(_fake_12_rows(), ["ISIC_000", "ISIC_001"])


# ---------------------------------------------------------------------------
# classify_call_outcome
# ---------------------------------------------------------------------------
def test_classify_call_outcome_timeout_is_technical_failure():
    outcome = vd.TimedCallOutcome(value=None, exception=None, timed_out=True, elapsed_s=120.0)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_TECHNICAL_FAILURE
    assert "timeout" in cause
    assert raw is None


def test_classify_call_outcome_exception_is_technical_failure():
    outcome = vd.TimedCallOutcome(value=None, exception=RuntimeError("falha de rede"), timed_out=False, elapsed_s=0.1)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_TECHNICAL_FAILURE
    assert "RuntimeError" in cause


def test_classify_call_outcome_empty_text_is_no_response():
    outcome = vd.TimedCallOutcome(value="   ", exception=None, timed_out=False, elapsed_s=0.1)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_NO_RESPONSE


def test_classify_call_outcome_refusal_marker_detected():
    outcome = vd.TimedCallOutcome(value="I cannot help with medical diagnosis.", exception=None, timed_out=False, elapsed_s=0.1)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_REFUSAL


def test_classify_call_outcome_normal_text_is_ok():
    outcome = vd.TimedCallOutcome(value="Lesão Benigna. Sem achados relevantes.", exception=None, timed_out=False, elapsed_s=0.1)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_OK
    assert raw == "Lesão Benigna. Sem achados relevantes."


# ---------------------------------------------------------------------------
# AbstentionCounters / AbstentionTracker
# ---------------------------------------------------------------------------
def test_abstention_counters_pause_after_3_consecutive_technical_failures():
    c = vd.AbstentionCounters()
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    assert c.paused is False
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    assert c.paused is True
    assert c.pause_reason == "3_falhas_tecnicas_consecutivas"


def test_abstention_counters_success_resets_all_counters():
    c = vd.AbstentionCounters()
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    c.register(vd.STATUS_OK)
    assert c.technical_failure == 0
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    assert c.paused is False  # só 2 consecutivas depois do reset


def test_abstention_counters_indeterminate_is_success_never_pauses():
    # "indeterminado" é classified_status, não call_status -- call_status
    # para um diagnóstico indeterminado válido é STATUS_OK (chegou uma
    # resposta parseável). Confirma que isso nunca é tratado como falha.
    c = vd.AbstentionCounters()
    for _ in range(5):
        c.register(vd.STATUS_OK)
    assert c.paused is False
    assert c.technical_failure == 0 and c.no_response == 0 and c.refusal == 0


def test_abstention_counters_mixed_categories_do_not_accumulate_across_each_other():
    c = vd.AbstentionCounters()
    c.register(vd.STATUS_TECHNICAL_FAILURE)
    c.register(vd.STATUS_NO_RESPONSE)
    c.register(vd.STATUS_REFUSAL)
    assert c.paused is False
    assert c.technical_failure == 0 and c.no_response == 0 and c.refusal == 1


def test_abstention_tracker_pauses_are_per_role_not_global():
    tracker = vd.AbstentionTracker()
    for _ in range(3):
        tracker.register(vd.ROLE_ABCD, vd.STATUS_TECHNICAL_FAILURE)
    assert tracker.is_paused(vd.ROLE_ABCD) is True
    assert tracker.is_paused(vd.ROLE_MENZIES) is False


# ---------------------------------------------------------------------------
# Controles de envio de imagem
# ---------------------------------------------------------------------------
def test_evaluate_image_control_response_correct_answer_passes():
    spec = vd.IMAGE_CONTROL_SPECS[0]  # vermelho, superior-esquerdo
    result = vd.evaluate_image_control_response("O quadrado está no canto superior esquerdo e é vermelho.", spec)
    assert result["passed"] is True


def test_evaluate_image_control_response_wrong_quadrant_fails():
    spec = vd.IMAGE_CONTROL_SPECS[0]
    result = vd.evaluate_image_control_response("O quadrado está no canto inferior direito e é vermelho.", spec)
    assert result["passed"] is False
    assert result["color_found"] is True
    assert result["quadrant_found"] is False


def test_evaluate_image_sending_controls_flags_identical_responses_as_not_varying():
    results = [
        {"spec": vd.IMAGE_CONTROL_SPECS[0], "raw_text": "sempre a mesma resposta genérica"},
        {"spec": vd.IMAGE_CONTROL_SPECS[1], "raw_text": "sempre a mesma resposta genérica"},
    ]
    evaluation = vd.evaluate_image_sending_controls(results)
    assert evaluation["responses_vary_with_image_content"] is False
    assert evaluation["overall_pass"] is False


def test_evaluate_image_sending_controls_passes_when_all_correct_and_varying():
    results = [
        {"spec": spec, "raw_text": f"quadrado {spec['color_name']} no canto {spec['quadrant'].replace('-', ' ')}"}
        for spec in vd.IMAGE_CONTROL_SPECS
    ]
    evaluation = vd.evaluate_image_sending_controls(results)
    assert evaluation["all_individually_passed"] is True
    assert evaluation["overall_pass"] is True


# ---------------------------------------------------------------------------
# run_vlm_direct_call (mock no limite do ChatOllama)
# ---------------------------------------------------------------------------
def test_run_vlm_direct_call_success_classified_positivo():
    from PIL import Image

    _FakeChatOllama._next_content = "Suspeita de Melanoma. Bordas irregulares observadas."
    image = Image.new("RGB", (10, 10))
    record = vd.run_vlm_direct_call(MODEL_TAG, image, "HAM_001", "ISIC_001")
    assert record.call_status == vd.STATUS_OK
    assert record.classified_status == "positivo"
    assert record.role == vd.ROLE_VLM_DIRETO
    assert record.condition == "s0"


def test_run_vlm_direct_call_timeout_is_technical_failure_not_retried():
    from PIL import Image

    _FakeChatOllama._sleep_s = 0.3
    image = Image.new("RGB", (10, 10))
    record = vd.run_vlm_direct_call(MODEL_TAG, image, "HAM_002", "ISIC_002", timeout_s=0.05)
    assert record.call_status == vd.STATUS_TECHNICAL_FAILURE
    assert "timeout" in record.failure_cause
    assert record.classified_status is None


def test_run_vlm_direct_call_exception_is_technical_failure():
    from PIL import Image

    _FakeChatOllama._next_content = RuntimeError("conexão recusada")
    image = Image.new("RGB", (10, 10))
    record = vd.run_vlm_direct_call(MODEL_TAG, image, "HAM_003", "ISIC_003")
    assert record.call_status == vd.STATUS_TECHNICAL_FAILURE


# ---------------------------------------------------------------------------
# run_multiagent_role_call (mock no limite de _build_llm)
# ---------------------------------------------------------------------------
def test_run_multiagent_role_call_abcd_success_updates_state():
    from PIL import Image

    _FakeLLMForAgents.next_content = "Assimetria 0, bordas regulares -- lesão benigna."
    image = Image.new("RGB", (10, 10))
    state = vd.build_graph_state(image, "img_s2")
    record, state_update = vd.run_multiagent_role_call(MODEL_TAG, vd.ROLE_ABCD, state, "HAM_001", "ISIC_001", "s2")
    assert record.call_status == vd.STATUS_OK
    assert state_update["diagnosis_abcd"] is not None
    assert state_update["status_abcd"].status == "ok"


def test_run_multiagent_role_call_summary_classifies_output():
    from PIL import Image

    _FakeLLMForAgents.next_content = "Lesão Benigna. Consolidação sem achados suspeitos."
    image = Image.new("RGB", (10, 10))
    state = vd.GraphState(
        image_path="x", lesion_size=0.0,
        seg_image_data=vd.convert_to_base64(image),
        diagnosis_abcd=ABCDDiagnosticAnswer(assimetria="0", bordas="1", cor="1", estruturas="1", explicacao="benigno"),
    )
    record, state_update = vd.run_multiagent_role_call(MODEL_TAG, vd.ROLE_SUMMARY, state, "HAM_001", "ISIC_001", "s2")
    assert record.call_status == vd.STATUS_OK
    assert record.classified_status == "negativo"
    assert state_update["validation"] == "Lesão Benigna. Consolidação sem achados suspeitos."


def test_run_multiagent_role_call_timeout_marks_agent_sem_resposta():
    from PIL import Image
    import time as _time

    class _SlowFakeLLM(_FakeLLMForAgents):
        def invoke(self, messages):
            _time.sleep(0.3)
            return _FakeOllamaResponse("nunca deveria chegar aqui")

    import src.agents.agents as am

    orig = am._build_llm
    am._build_llm = lambda model_name, temperature: _SlowFakeLLM()
    try:
        image = Image.new("RGB", (10, 10))
        state = vd.build_graph_state(image, "img_s2")
        record, state_update = vd.run_multiagent_role_call(
            MODEL_TAG, vd.ROLE_MENZIES, state, "HAM_001", "ISIC_001", "s2", timeout_s=0.05
        )
        assert record.call_status == vd.STATUS_TECHNICAL_FAILURE
        assert state_update["diagnosis_menzies"] is None
        assert state_update["status_menzies"].status == "sem_resposta"
    finally:
        am._build_llm = orig


def test_not_executed_record_has_distinct_status():
    record = vd.not_executed_record(MODEL_TAG, vd.ROLE_SPCL, "s2", "HAM_001", "ISIC_001", "3_falhas_tecnicas_consecutivas")
    assert record.call_status == vd.STATUS_NOT_EXECUTED
    assert record.classified_status is None


# ---------------------------------------------------------------------------
# Controles de síntese -- inputs fixos, detecção de achado novo
# ---------------------------------------------------------------------------
def test_run_synthesis_control_unanime_negativo_passes_with_consistent_output():
    _FakeLLMForAgents.next_content = (
        "Lesão Benigna. Os três pareceres concordam em ausência de achados suspeitos relevantes."
    )
    result = vd.run_synthesis_control(MODEL_TAG, "unanime_negativo")
    assert result["parseable"] is True
    assert result["classified_status"] == "negativo"
    assert result["passed"] is True


def test_run_synthesis_control_flags_novel_clinical_term_not_in_inputs():
    # O cenário "um_parecer_ausente" não menciona "pseudópodes" em nenhum
    # dos pareceres de entrada -- se o sumarizador introduzir esse termo,
    # deve ser sinalizado como candidato a achado novo.
    _FakeLLMForAgents.next_content = (
        "Lesão Indeterminada. Observam-se possíveis pseudópodes não mencionados nos pareceres originais."
    )
    result = vd.run_synthesis_control(MODEL_TAG, "um_parecer_ausente")
    assert "pseudópodes" in result["novel_clinical_terms_candidates"] or "pseudopodes" in result["novel_clinical_terms_candidates"]
    assert result["no_novel_terms_detected"] is False
    assert result["passed"] is False


def test_run_synthesis_control_unparseable_output_fails():
    _FakeLLMForAgents.next_content = "resposta sem nenhuma categoria reconhecível"
    result = vd.run_synthesis_control(MODEL_TAG, "conflitante")
    assert result["parseable"] is False
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# classify_binary_coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status,expected",
    [("positivo", True), ("negativo", False), ("indeterminado", None), ("ausente", None), ("nao_reconhecido", None), (None, None)],
)
def test_classify_binary_coverage(status, expected):
    assert vd.classify_binary_coverage(status) is expected


# ---------------------------------------------------------------------------
# detect_sustained_swap_growth
# ---------------------------------------------------------------------------
def test_detect_sustained_swap_growth_true_for_monotonic_sequence_above_floor():
    samples = [100, 150, 150, 200, 250]
    result = vd.detect_sustained_swap_growth(samples, floor_bytes=50)
    assert result["is_monotonic_non_decreasing_across_all_calls"] is True
    assert result["sustained_growth_detected"] is True


def test_detect_sustained_swap_growth_false_for_fluctuating_sequence():
    samples = [100, 200, 150, 300, 100]
    result = vd.detect_sustained_swap_growth(samples, floor_bytes=50)
    assert result["is_monotonic_non_decreasing_across_all_calls"] is False
    assert result["sustained_growth_detected"] is False


def test_detect_sustained_swap_growth_definition_flagged_for_signoff():
    result = vd.detect_sustained_swap_growth([10, 20])
    assert result["definition_needs_researcher_signoff"] is True
