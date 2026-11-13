"""Testes com dados/LLM 100% sintéticos (mock no limite do Ollama -- nunca
requer um servidor Ollama real) para `src/multiagent/vlm_direct.py` (Etapa
5, segunda entrega: piloto local de VLM -- REWORK incorporando as 10
decisões fechadas pelo pesquisador sobre o primeiro rascunho).

Reescrito/estendido nesta entrega: os testes antigos de controle
monotônico de swap, de 3 controles de imagem compartilhados, e de
designação "4 primeiros por ordem alfabética" testavam comportamento que
foi DELIBERADAMENTE substituído -- não fazia sentido preservá-los; foram
removidos e substituídos pelos equivalentes do novo desenho.

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
# build_planned_calls -- exatamente 92 (inalterado nesta entrega)
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


def test_total_planned_calls_per_model_is_108():
    # Novo parâmetro fechado: 92 (matriz, inalterado) + 16 (controles) = 108.
    assert vd.N_MAIN_MATRIX_CALLS == 92
    assert vd.N_IMAGE_SENDING_CONTROL_CALLS == 12
    assert vd.N_SYNTHESIS_CONTROL_CALLS == 4
    assert vd.N_TOTAL_PLANNED_CALLS_PER_MODEL == 108


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


def test_classify_call_outcome_refusal_marker_with_valid_parecer_is_ok_not_refusal():
    # DECISÃO FECHADA #5: marcador de recusa presente, MAS a resposta
    # também contém um parecer categórico válido -- deve ser STATUS_OK,
    # não STATUS_REFUSAL.
    text = "I cannot help with a full diagnosis, but based on the image: Lesão Indeterminada."
    outcome = vd.TimedCallOutcome(value=text, exception=None, timed_out=False, elapsed_s=0.1)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_OK
    assert raw == text


def test_classify_call_outcome_refusal_marker_alone_without_parecer_is_refusal():
    text = "I cannot help with that request at all."
    outcome = vd.TimedCallOutcome(value=text, exception=None, timed_out=False, elapsed_s=0.1)
    status, cause, raw = vd.classify_call_outcome(outcome)
    assert status == vd.STATUS_REFUSAL


def test_indeterminate_phrase_not_in_refusal_markers_list():
    # Revisão da decisão #5: "não consigo distinguir pela imagem" não deve
    # estar na lista de marcadores de recusa -- é linguagem legítima de
    # parecer indeterminado, não recusa.
    assert "não consigo distinguir pela imagem" not in vd._REFUSAL_MARKERS


# ---------------------------------------------------------------------------
# AbstentionCounters / AbstentionTracker (inalterado nesta entrega)
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
# Controles de envio de imagem -- POR PAPEL VISUAL (decisão fechada #2)
# ---------------------------------------------------------------------------
def test_image_control_specs_for_role_differ_in_both_color_and_quadrant():
    specs = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)
    assert len(specs) == 2
    assert specs[0]["color_name"] != specs[1]["color_name"]
    assert specs[0]["quadrant"] != specs[1]["quadrant"]


def test_image_control_specs_by_role_covers_all_4_visual_roles():
    assert set(vd.IMAGE_CONTROL_SPECS_BY_ROLE.keys()) == set(vd.VISUAL_ROLES)
    assert len(vd.VISUAL_ROLES) == 4
    for role, specs in vd.IMAGE_CONTROL_SPECS_BY_ROLE.items():
        assert len(specs) == 2
        assert all(spec["role"] == role for spec in specs)


def test_evaluate_image_control_response_correct_answer_passes():
    spec = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)[0]
    result = vd.evaluate_image_control_response(
        f"O quadrado está no canto {spec['quadrant'].replace('-', ' ')} e é {spec['color_name']}.", spec
    )
    assert result["passed"] is True


def test_evaluate_image_control_response_wrong_quadrant_fails():
    spec = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)[0]
    result = vd.evaluate_image_control_response("O quadrado está em outro lugar qualquer.", spec)
    assert result["passed"] is False
    assert result["quadrant_found"] is False


def test_no_image_control_message_has_no_image_entry():
    message = vd._text_only_control_message(vd.NO_IMAGE_CONTROL_PROMPT)
    assert all(item.get("type") != "image_url" for item in message.content)
    assert vd.verify_no_image_control_payload_has_no_image(message) is True


def test_evaluate_no_image_control_response_detects_acknowledgment():
    result = vd.evaluate_no_image_control_response("Não há imagem anexada a esta mensagem.", vd.ROLE_SPCL, "ctrl")
    assert result["acknowledged_no_image"] is True
    assert result["passed"] is True


def test_evaluate_no_image_control_response_rejects_hallucinated_answer():
    result = vd.evaluate_no_image_control_response("O quadrado é vermelho e está no canto superior esquerdo.", vd.ROLE_SPCL, "ctrl")
    assert result["acknowledged_no_image"] is False
    assert result["passed"] is False


def test_verify_image_control_payload_sent_correct_image_true_for_matching_image():
    spec = vd.build_image_control_specs_for_role(vd.ROLE_VLM_DIRETO)[0]
    control_image = vd.build_image_sending_control_image(spec)
    message = vd._image_message(vd.convert_to_base64(control_image), vd.IMAGE_CONTROL_PROMPT)
    assert vd.verify_image_control_payload_sent_correct_image(message, control_image) is True


def test_verify_image_control_payload_sent_correct_image_false_for_mismatched_image():
    spec_a = vd.build_image_control_specs_for_role(vd.ROLE_VLM_DIRETO)[0]
    spec_b = vd.build_image_control_specs_for_role(vd.ROLE_VLM_DIRETO)[1]
    image_a = vd.build_image_sending_control_image(spec_a)
    image_b = vd.build_image_sending_control_image(spec_b)
    message = vd._image_message(vd.convert_to_base64(image_a), vd.IMAGE_CONTROL_PROMPT)
    assert vd.verify_image_control_payload_sent_correct_image(message, image_b) is False


def test_evaluate_image_sending_controls_for_role_flags_identical_image_responses_as_not_varying():
    specs = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)
    results = [
        {"kind": "imagem", "spec": specs[0], "raw_text": "sempre a mesma resposta genérica", "payload_correct": True},
        {"kind": "imagem", "spec": specs[1], "raw_text": "sempre a mesma resposta genérica", "payload_correct": True},
        {"kind": "sem_imagem", "role": vd.ROLE_ABCD, "name": "x", "raw_text": "não há imagem", "payload_correct": True},
    ]
    evaluation = vd.evaluate_image_sending_controls_for_role(results)
    assert evaluation["responses_vary_with_image_content"] is False
    assert evaluation["overall_pass"] is False


def test_evaluate_image_sending_controls_for_role_passes_when_all_correct_and_varying():
    specs = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)
    results = [
        {"kind": "imagem", "spec": spec, "raw_text": f"quadrado {spec['color_name']} no canto {spec['quadrant'].replace('-', ' ')}", "payload_correct": True}
        for spec in specs
    ]
    results.append({"kind": "sem_imagem", "role": vd.ROLE_ABCD, "name": "x", "raw_text": "não há imagem anexada", "payload_correct": True})
    evaluation = vd.evaluate_image_sending_controls_for_role(results)
    assert evaluation["all_individually_passed"] is True
    assert evaluation["all_payloads_correct"] is True
    assert evaluation["overall_pass"] is True


def test_evaluate_image_sending_controls_requires_all_4_roles_to_pass():
    specs = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)
    passing_results = [
        {"kind": "imagem", "spec": spec, "raw_text": f"quadrado {spec['color_name']} no canto {spec['quadrant'].replace('-', ' ')}", "payload_correct": True}
        for spec in specs
    ]
    passing_results.append({"kind": "sem_imagem", "role": vd.ROLE_ABCD, "name": "x", "raw_text": "não há imagem", "payload_correct": True})
    failing_results = [
        {"kind": "imagem", "spec": spec, "raw_text": "resposta errada", "payload_correct": True}
        for spec in vd.build_image_control_specs_for_role(vd.ROLE_SPCL)
    ]
    failing_results.append({"kind": "sem_imagem", "role": vd.ROLE_SPCL, "name": "y", "raw_text": "o quadrado é azul", "payload_correct": True})
    evaluation = vd.evaluate_image_sending_controls({vd.ROLE_ABCD: passing_results, vd.ROLE_SPCL: failing_results})
    assert evaluation["per_role"][vd.ROLE_ABCD]["overall_pass"] is True
    assert evaluation["per_role"][vd.ROLE_SPCL]["overall_pass"] is False
    assert evaluation["overall_pass"] is False


# ---------------------------------------------------------------------------
# run_image_sending_control_call -- caminho de construção de LLM PRÓPRIO
# de cada papel (decisão fechada #2).
# ---------------------------------------------------------------------------
def test_run_image_sending_control_call_vlm_direto_uses_chat_ollama_path():
    _FakeChatOllama._next_content = "resposta de controle"
    spec = vd.build_image_control_specs_for_role(vd.ROLE_VLM_DIRETO)[0]
    result = vd.run_image_sending_control_call(MODEL_TAG, vd.ROLE_VLM_DIRETO, "imagem", spec)
    assert result["call_record"].call_status == vd.STATUS_OK
    assert result["payload_correct"] is True
    assert result["call_record"].role == f"controle_imagem_{vd.ROLE_VLM_DIRETO}"


def test_run_image_sending_control_call_abcd_uses_agent_llm_path():
    _FakeLLMForAgents.next_content = "resposta de controle do agente ABCD"
    spec = vd.build_image_control_specs_for_role(vd.ROLE_ABCD)[0]
    result = vd.run_image_sending_control_call(MODEL_TAG, vd.ROLE_ABCD, "imagem", spec)
    assert result["call_record"].call_status == vd.STATUS_OK
    assert result["call_record"].raw_response_text == "resposta de controle do agente ABCD"
    assert result["payload_correct"] is True


def test_run_image_sending_control_call_no_image_kind_sends_no_image():
    _FakeLLMForAgents.next_content = "Não há imagem anexada a esta mensagem."
    result = vd.run_image_sending_control_call(MODEL_TAG, vd.ROLE_MENZIES, "sem_imagem", None)
    assert result["kind"] == "sem_imagem"
    assert result["payload_correct"] is True
    assert result["call_record"].call_status == vd.STATUS_OK


# ---------------------------------------------------------------------------
# run_vlm_direct_call (mock no limite do ChatOllama) -- inalterado
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
# run_multiagent_role_call (mock no limite de _build_llm) -- inalterado
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
# Controles de síntese -- "achado novo" é TRIAGEM, não pass/fail (decisão
# fechada #4); cenário "conflitante" tem vasos_atipicos "não avaliado"
# (decisão fechada #3).
# ---------------------------------------------------------------------------
def test_conflitante_scenario_vasos_atipicos_is_unambiguously_not_evaluated():
    spcl = vd.FIXED_SYNTHESIS_SCENARIOS["conflitante"]["diagnosis_spcl"]
    text = spcl.vasos_atipicos.lower()
    assert "não avaliado" in text or "nao avaliado" in text
    assert text.strip() not in ("sim.", "não.", "nao.")


def test_run_synthesis_control_passed_reflects_only_parseability_not_novel_terms():
    # DECISÃO FECHADA #4: mesmo com um termo clínico "novo" na
    # justificativa, "passed" deve ser True se a saída for parseável --
    # achado novo nunca mais é gate automático.
    _FakeLLMForAgents.next_content = (
        "Lesão Indeterminada. Observam-se possíveis pseudópodes não mencionados nos pareceres originais."
    )
    result = vd.run_synthesis_control(MODEL_TAG, "um_parecer_ausente")
    assert result["parseable"] is True
    assert result["passed"] is True  # parseável -- achado novo é só triagem
    assert "requires_manual_review" in result and result["requires_manual_review"] is True
    assert len(result["novel_clinical_terms_candidates"]) > 0  # sinal de triagem presente, mas não reprova


def test_run_synthesis_control_unparseable_output_fails():
    _FakeLLMForAgents.next_content = "resposta sem nenhuma categoria reconhecível"
    result = vd.run_synthesis_control(MODEL_TAG, "conflitante")
    assert result["parseable"] is False
    assert result["passed"] is False


def test_run_synthesis_control_conflitante_includes_not_evaluated_check():
    _FakeLLMForAgents.next_content = "Lesão Indeterminada. Vasos atípicos não avaliados nesta análise."
    result = vd.run_synthesis_control(MODEL_TAG, "conflitante")
    assert "not_evaluated_vs_evaluated_absent_check" in result
    assert result["not_evaluated_vs_evaluated_absent_check"]["heuristic_is_approximate"] is True


def test_check_not_evaluated_not_misrepresented_as_absent_flags_suspicious_phrase():
    check = vd._check_not_evaluated_not_misrepresented_as_absent("O critério foi avaliado e está ausente na imagem.")
    assert check["likely_misrepresented_as_evaluated_absent"] is True


def test_check_absence_not_misrepresented_as_lesion_lacks_feature_flags_suspicious_phrase():
    check = vd._check_absence_not_misrepresented_as_lesion_lacks_feature("A lesão não apresenta vasos atípicos.")
    assert check["likely_misrepresented_absence_as_lesion_lacks_feature"] is True


def test_build_synthesis_manual_review_template_has_one_entry_per_scenario_with_null_verdict():
    _FakeLLMForAgents.next_content = "Lesão Benigna. Sem achados relevantes."
    synthesis_results = [
        {k: v for k, v in vd.run_synthesis_control(MODEL_TAG, name).items() if k != "call_record"}
        for name in vd.FIXED_SYNTHESIS_SCENARIOS
    ]
    template = vd.build_synthesis_manual_review_template(synthesis_results)
    assert len(template["scenarios"]) == len(vd.FIXED_SYNTHESIS_SCENARIOS)
    for item in template["scenarios"]:
        assert item["manual_review_verdict"] is None
        assert item["reviewer_notes"] == ""


def test_evaluate_manual_review_file_blocks_on_missing_verdict():
    data = {"scenarios": [{"scenario": "unanime_positivo", "manual_review_verdict": None}]}
    evaluation = vd.evaluate_manual_review_file(data)
    assert evaluation["all_scenarios_cleared"] is False
    assert "unanime_positivo" in evaluation["missing_verdict_scenarios"]


def test_evaluate_manual_review_file_blocks_on_novel_finding_found():
    data = {"scenarios": [{"scenario": "conflitante", "manual_review_verdict": vd.MANUAL_REVIEW_VERDICT_NOVEL_FINDING_FOUND}]}
    evaluation = vd.evaluate_manual_review_file(data)
    assert evaluation["all_scenarios_cleared"] is False
    assert "conflitante" in evaluation["novel_finding_scenarios"]


def test_evaluate_manual_review_file_passes_when_all_cleared():
    data = {
        "scenarios": [
            {"scenario": name, "manual_review_verdict": vd.MANUAL_REVIEW_VERDICT_NO_NOVEL_FINDING}
            for name in vd.FIXED_SYNTHESIS_SCENARIOS
        ]
    }
    evaluation = vd.evaluate_manual_review_file(data)
    assert evaluation["all_scenarios_cleared"] is True


def test_evaluate_manual_review_file_blocks_on_invalid_verdict_string():
    data = {"scenarios": [{"scenario": "unanime_negativo", "manual_review_verdict": "talvez"}]}
    evaluation = vd.evaluate_manual_review_file(data)
    assert evaluation["all_scenarios_cleared"] is False
    assert "unanime_negativo" in evaluation["invalid_verdict_scenarios"]


# ---------------------------------------------------------------------------
# classify_binary_coverage (inalterado)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status,expected",
    [("positivo", True), ("negativo", False), ("indeterminado", None), ("ausente", None), ("nao_reconhecido", None), (None, None)],
)
def test_classify_binary_coverage(status, expected):
    assert vd.classify_binary_coverage(status) is expected


# ---------------------------------------------------------------------------
# detect_sustained_swap_growth -- REESCRITO (decisão fechada #6):
# janelas de 5 sobre a matriz principal, baseline explícito, 2 janelas
# consecutivas acima do limiar com crescimento estrito.
# ---------------------------------------------------------------------------
def test_detect_sustained_swap_growth_true_for_two_consecutive_growing_windows_above_threshold():
    baseline = 0
    threshold = vd.SWAP_GROWTH_THRESHOLD_BYTES
    # janela 0: mediana bem abaixo do limiar
    window0 = [0, 0, 0, 0, 0]
    # janela 1: mediana acima do limiar
    window1 = [threshold + 10, threshold + 10, threshold + 10, threshold + 10, threshold + 10]
    # janela 2: mediana acima do limiar e MAIOR que a janela 1
    window2 = [threshold + 1000, threshold + 1000, threshold + 1000, threshold + 1000, threshold + 1000]
    samples = window0 + window1 + window2
    result = vd.detect_sustained_swap_growth(samples, baseline_swap_bytes=baseline)
    assert result["sustained_growth_detected"] is True
    assert (1, 2) in result["sustained_growth_window_pairs"]
    assert result["spike_then_stable_detected"] is False


def test_detect_sustained_swap_growth_spike_then_stable_is_flagged_separately():
    baseline = 0
    threshold = vd.SWAP_GROWTH_THRESHOLD_BYTES
    window0 = [threshold + 2000] * 5  # já cruza o limiar (o "pico")
    window1 = [threshold + 1000] * 5  # acima do limiar mas NÃO supera a mediana anterior
    samples = window0 + window1
    result = vd.detect_sustained_swap_growth(samples, baseline_swap_bytes=baseline)
    assert result["sustained_growth_detected"] is False
    assert result["spike_then_stable_detected"] is True
    assert (0, 1) in result["spike_then_stable_window_pairs"]


def test_detect_sustained_swap_growth_false_when_below_threshold():
    baseline = 0
    samples = [100, 150, 150, 200, 250, 100, 120, 130, 140, 150]
    result = vd.detect_sustained_swap_growth(samples, baseline_swap_bytes=baseline)
    assert result["sustained_growth_detected"] is False
    assert result["spike_then_stable_detected"] is False


def test_detect_sustained_swap_growth_baseline_is_required_and_used():
    threshold = vd.SWAP_GROWTH_THRESHOLD_BYTES
    samples = [threshold] * 10
    # baseline alto o suficiente para que "threshold" absoluto não seja
    # mais alcançado em relação a baseline + growth_threshold_bytes.
    result_high_baseline = vd.detect_sustained_swap_growth(samples, baseline_swap_bytes=threshold * 10)
    assert result_high_baseline["sustained_growth_detected"] is False
    result_zero_baseline = vd.detect_sustained_swap_growth(samples, baseline_swap_bytes=0)
    # mediana constante (não estritamente crescente) -- não deve ser sustained_growth
    assert result_zero_baseline["sustained_growth_detected"] is False


def test_detect_sustained_swap_growth_discards_incomplete_trailing_window():
    result = vd.detect_sustained_swap_growth([1, 2, 3], baseline_swap_bytes=0)
    assert result["n_complete_windows"] == 0
    assert result["n_samples_discarded_incomplete_window"] == 3


def test_detect_sustained_swap_growth_definition_no_longer_needs_signoff():
    # Decisão fechada #6: a regra numérica já foi aprovada -- não é mais
    # "pendente", mesmo que ainda reporte as janelas brutas para inspeção.
    result = vd.detect_sustained_swap_growth([1] * 10, baseline_swap_bytes=0)
    assert result["definition_needs_researcher_signoff"] is False


# ---------------------------------------------------------------------------
# Relatório informativo para ABCD/Menzies/SPCL + projeção de duração
# (extensões dos itens 9 e 10 da entrega).
# ---------------------------------------------------------------------------
def test_count_not_evaluated_phrases_counts_occurrences():
    text = "Este critério não avaliado. Outro critério não considerado também."
    assert vd.count_not_evaluated_phrases(text) >= 2


def test_summarize_diagnostic_role_reporting_never_includes_zero_conclusive_flag():
    record_ok = vd.CallRecord(
        model_name=MODEL_TAG, role=vd.ROLE_ABCD, condition="s2", lesion_id="L1", image_id="I1",
        attempt_timestamp_utc="t", elapsed_s=1.0, mem_before=None, mem_after=None,
        raw_response_text="Critério não avaliado.", call_status=vd.STATUS_OK, failure_cause="", classified_status=None,
    )
    report = vd.summarize_diagnostic_role_reporting([record_ok], vd.ROLE_ABCD)
    assert report["n_ok"] == 1
    assert report["n_not_evaluated_phrases_total"] >= 1
    assert "zero" in report["note"].lower()


def test_compute_per_lesion_condition_elapsed_totals_sums_the_4_sequential_roles():
    records = [
        vd.CallRecord(model_name=MODEL_TAG, role=role, condition="s2", lesion_id="L1", image_id="I1",
                       attempt_timestamp_utc="t", elapsed_s=2.0, mem_before=None, mem_after=None,
                       raw_response_text="x", call_status=vd.STATUS_OK, failure_cause="", classified_status=None)
        for role in vd.MULTIAGENT_ROLES_IN_ORDER
    ]
    totals = vd.compute_per_lesion_condition_elapsed_totals(records)
    assert totals[("L1", "s2")] == pytest.approx(8.0)


def test_project_full_scale_duration_uses_mean_times_hypothetical_n_and_includes_caveat():
    result = vd.project_full_scale_duration([10.0, 20.0, 30.0], hypothetical_n=100)
    assert result["mean_per_lesion_condition_elapsed_s"] == pytest.approx(20.0)
    assert result["projected_total_s"] == pytest.approx(2000.0)
    assert "caveat" in result and "NÃO é uma promessa" in result["caveat"]


def test_project_full_scale_duration_handles_empty_input():
    result = vd.project_full_scale_duration([], hypothetical_n=100)
    assert result["projected_total_s"] is None
    assert "caveat" in result
