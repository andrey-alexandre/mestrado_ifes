"""Testes de `src.evaluation.content_recovery.recover_raw_response_text`
-- recuperação segura (via `ast.literal_eval`, nunca `eval()`) de
`raw_response_text` corrompido pelo bug de representação de conteúdo do
Gemini (2026-11-18). Cobre exatamente os 4 cenários pedidos pelo
pesquisador: strings (já limpas), blocos textuais (lista real
contaminada), metadados (bloco com campo "extras"/assinatura), e campo
realmente ausente (texto limpo sem `decisao_final` nenhum).

STATUS: implementado, aguardando execução do pesquisador
(`python -m pytest tests/evaluation/test_content_recovery.py -v`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.content_recovery import (  # noqa: E402
    IRRECOVERABLE_PARSE_ERROR,
    IRRECOVERABLE_UNEXPECTED_STRUCTURE,
    RECOVERED_FROM_LIST_REPR,
    RECOVERY_NOT_APPLICABLE,
    RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT,
    recover_raw_response_text,
)
from src.evaluation.explicit_decision_contract import (  # noqa: E402
    DECISAO_INDETERMINADO,
    DECISAO_MELANOMA,
    DECISAO_NAO_MELANOMA,
    DECISION_STATUS_FIELD_ABSENT,
    DECISION_STATUS_OK,
    extract_decisao_final,
)


# ---------------------------------------------------------------------------
# Cenário 1: string já limpa (nunca contaminada) -- "strings"
# ---------------------------------------------------------------------------
def test_plain_clean_string_with_decision_not_needed():
    raw = "Parecer normal em português.\n\ndecisao_final: melanoma"
    result = recover_raw_response_text(raw)
    assert result.recovery_status == RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT
    assert result.recovered_text == raw
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decision_status == DECISION_STATUS_OK
    assert decision.decisao_final == DECISAO_MELANOMA


def test_plain_string_with_accented_portuguese_text_not_needed():
    raw = "Lesão com assimetria e bordas irregulares.\n\ndecisao_final: nao_melanoma"
    result = recover_raw_response_text(raw)
    assert result.recovery_status == RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decisao_final == DECISAO_NAO_MELANOMA


# ---------------------------------------------------------------------------
# Cenário 2: blocos textuais -- lista real contaminada (o bug real do Gemini)
# ---------------------------------------------------------------------------
def test_contaminated_list_repr_recovers_decision_indeterminado():
    # Reproduz EXATAMENTE o padrão real do log do Gemini: str() de uma
    # lista com um bloco de texto contendo quebras de linha reais, que
    # ao ser stringificada escapa as quebras em \n literal.
    original_text_with_real_newlines = (
        "A imagem apresenta achados atípicos.\n\ndecisao_final: indeterminado"
    )
    contaminated = str([{"type": "text", "text": original_text_with_real_newlines}])
    assert "\\n" in contaminated  # confirma que a contaminação ocorreu de verdade

    result = recover_raw_response_text(contaminated)
    assert result.recovery_status == RECOVERED_FROM_LIST_REPR
    assert result.recovered_text == original_text_with_real_newlines
    assert "\\n" not in result.recovered_text

    decision = extract_decisao_final(result.recovered_text)
    assert decision.decision_status == DECISION_STATUS_OK
    assert decision.decisao_final == DECISAO_INDETERMINADO


def test_contaminated_list_repr_recovers_decision_melanoma():
    original = "Suspeita de malignidade.\n\ndecisao_final: melanoma"
    contaminated = str([{"type": "text", "text": original}])
    result = recover_raw_response_text(contaminated)
    assert result.recovery_status == RECOVERED_FROM_LIST_REPR
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decisao_final == DECISAO_MELANOMA


# ---------------------------------------------------------------------------
# Cenário 3: metadados -- bloco com campo "extras"/assinatura (exatamente
# o formato real observado: {'type': 'text', 'text': ..., 'extras': {...}})
# ---------------------------------------------------------------------------
def test_contaminated_list_with_metadata_extras_block_recovers_decision():
    original = "Parecer completo com achados.\n\ndecisao_final: nao_melanoma"
    contaminated = str(
        [
            {
                "type": "text",
                "text": original,
                "extras": {"signature": "EpwBCpkBAWkUfRNtKadPir9Ft1fJAQrPwwTqYABACXbtBMFYSVPbEkItwXyhpdV8M22Q6"},
            }
        ]
    )
    result = recover_raw_response_text(contaminated)
    assert result.recovery_status == RECOVERED_FROM_LIST_REPR
    assert result.recovered_text == original
    assert "signature" not in result.recovered_text
    assert "extras" not in result.recovered_text
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decisao_final == DECISAO_NAO_MELANOMA


def test_real_gemini_log_sample_recovers_correctly():
    """Amostra real copiada do log
    (`experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/
    call_log_main.jsonl`, HAM_0000005/ISIC_0024579) -- não um caso
    sintético, confirma a recuperação sobre dado real já salvo."""
    contaminated = (
        "[{'type': 'text', 'text': 'A imagem dermatoscópica apresenta características "
        "que requerem cautela. Observa-se uma lesão com áreas de pigmentação irregular, "
        "incluindo estruturas como glóbulos e áreas de rede pigmentar atípica ou "
        "desestruturada.\\n\\ndecisao_final: indeterminado', 'extras': {'signature': "
        "'EpwBCpkBAWkUfRNtKadPir9Ft1fJAQrPwwTqYABACXbtBMFYSVPbEkItwXyhpdV8M22Q6VpuRtJoS9z5t6rnNILty/"
        "EJ9PMGL8OJx2ese/QuXgyU2IBijEiN7BlW+wf5wZ4Nca8ULM4RX78diXQMPF9ali1UHsYvpNQC3XM5op8E6nh2MratIU3"
        "GFMOeSoPcpUpHHPGdPryRowGCapwF'}}]"
    )
    result = recover_raw_response_text(contaminated)
    assert result.recovery_status == RECOVERED_FROM_LIST_REPR
    assert result.recovered_text.endswith("decisao_final: indeterminado")
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decision_status == DECISION_STATUS_OK
    assert decision.decisao_final == DECISAO_INDETERMINADO


# ---------------------------------------------------------------------------
# Cenário 4: campo realmente ausente -- texto limpo (ou recuperado) sem
# NENHUMA linha decisao_final (o modelo simplesmente não emitiu o campo)
# ---------------------------------------------------------------------------
def test_plain_text_genuinely_missing_decision_field():
    raw = "Parecer longo sem a linha de decisão no formato esperado."
    result = recover_raw_response_text(raw)
    assert result.recovery_status == RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decision_status == DECISION_STATUS_FIELD_ABSENT
    assert decision.decisao_final is None


def test_contaminated_list_recovered_but_decision_field_still_genuinely_missing():
    original = "Texto recuperado, mas o modelo nunca escreveu o campo de decisão."
    contaminated = str([{"type": "text", "text": original}])
    result = recover_raw_response_text(contaminated)
    assert result.recovery_status == RECOVERED_FROM_LIST_REPR
    decision = extract_decisao_final(result.recovered_text)
    assert decision.decision_status == DECISION_STATUS_FIELD_ABSENT


# ---------------------------------------------------------------------------
# None / irrecuperável -- nunca eval(), nunca inventa um valor parcial
# ---------------------------------------------------------------------------
def test_none_is_not_applicable():
    result = recover_raw_response_text(None)
    assert result.recovery_status == RECOVERY_NOT_APPLICABLE
    assert result.recovered_text is None


def test_truncated_list_repr_is_irrecoverable_parse_error_not_silently_skipped():
    truncated = "[{'type': 'text', 'text': 'início do texto que foi cortado no meio"
    result = recover_raw_response_text(truncated)
    assert result.recovery_status == IRRECOVERABLE_PARSE_ERROR
    assert result.recovered_text is None
    assert result.parse_error_detail is not None


def test_list_with_no_text_blocks_recovers_as_empty_string_not_an_error():
    weird = str([{"type": "thinking_without_text_block", "payload": 123}])
    result = recover_raw_response_text(weird)
    # bloco não tem type=="text" -- extract_text_from_llm_content o ignora
    # e devolve string vazia (não é TypeError, é "nenhum bloco de texto
    # encontrado") -- comportamento correto: recuperação "bem-sucedida"
    # de um texto vazio, não um erro de estrutura.
    assert result.recovery_status == RECOVERED_FROM_LIST_REPR
    assert result.recovered_text == ""


def test_list_with_genuinely_unexpected_block_type_raises_structure_error():
    weird = str([42, {"type": "text", "text": "ok"}])
    result = recover_raw_response_text(weird)
    assert result.recovery_status == IRRECOVERABLE_UNEXPECTED_STRUCTURE
    assert result.recovered_text is None


def test_never_uses_eval_malicious_input_never_executes_code():
    """Confirma que uma string malformada no estilo de código executável
    NUNCA é executada -- `ast.literal_eval` levanta erro de sintaxe/valor
    para qualquer coisa que não seja um literal puro, nunca avalia uma
    chamada de função."""
    malicious = "__import__('os').system('echo should_never_run')"
    result = recover_raw_response_text(malicious)
    # Não é uma lista/dict literal -- tratado como texto comum (prosa),
    # nunca executado.
    assert result.recovery_status == RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT
    assert result.recovered_text == malicious
