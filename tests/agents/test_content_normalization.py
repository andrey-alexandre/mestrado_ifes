"""Testes de `src.agents.agents.extract_text_from_llm_content` -- correção
do bug real encontrado na execução do Gemini (2026-11-18, ver
docs/reconstruction/30_real_lesion_evaluation_protocol.md §11): alguns
provedores devolvem `.content` como lista de blocos, não string, e o
código anterior fazia `str(lista)` em algum ponto posterior, escapando
quebras de linha reais em `\\n` literal e quebrando a extração de
`decisao_final`.

STATUS: implementado, aguardando execução do pesquisador
(`python -m pytest tests/agents/test_content_normalization.py -v`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.agents import extract_text_from_llm_content  # noqa: E402


def test_none_returns_none():
    assert extract_text_from_llm_content(None) is None


def test_plain_string_returned_identical_no_transformation():
    text = "linha 1\nlinha 2\n\ndecisao_final: melanoma"
    assert extract_text_from_llm_content(text) is text


def test_plain_string_with_real_newlines_never_modified():
    text = "texto com\nquebras\nreais e\nespaços   extras"
    result = extract_text_from_llm_content(text)
    assert result == text
    assert "\\n" not in result


def test_list_of_single_text_block_extracts_text_preserving_real_newlines():
    content = [{"type": "text", "text": "parecer real.\n\ndecisao_final: indeterminado"}]
    result = extract_text_from_llm_content(content)
    assert result == "parecer real.\n\ndecisao_final: indeterminado"
    assert "\n" in result
    assert "\\n" not in result


def test_list_with_metadata_extras_block_ignores_non_text_fields():
    content = [
        {
            "type": "text",
            "text": "resposta real\n\ndecisao_final: nao_melanoma",
            "extras": {"signature": "EpwBCpkBAWkUfRNtKadPir9Ft1fJAQrPwwTqYABACXbtBMFYSVPbEkItwXyhpdV8M22Q6"},
        }
    ]
    result = extract_text_from_llm_content(content)
    assert result == "resposta real\n\ndecisao_final: nao_melanoma"
    assert "signature" not in result
    assert "extras" not in result


def test_list_with_multiple_text_blocks_concatenated_in_order_no_separator():
    content = [{"type": "text", "text": "parte um. "}, {"type": "text", "text": "parte dois."}]
    assert extract_text_from_llm_content(content) == "parte um. parte dois."


def test_list_with_non_text_block_type_skipped_entirely():
    content = [
        {"type": "thinking", "thinking": "raciocínio interno não deve aparecer"},
        {"type": "text", "text": "resposta final visível"},
    ]
    result = extract_text_from_llm_content(content)
    assert result == "resposta final visível"
    assert "raciocínio" not in result


def test_list_of_plain_strings_concatenated():
    content = ["parte a", "parte b"]
    assert extract_text_from_llm_content(content) == "parte aparte b"


def test_list_with_dict_missing_text_key_but_type_text_treated_as_empty():
    content = [{"type": "text"}, {"type": "text", "text": "única parte real"}]
    assert extract_text_from_llm_content(content) == "única parte real"


def test_unexpected_type_raises_type_error():
    with pytest.raises(TypeError):
        extract_text_from_llm_content(12345)


def test_list_with_unexpected_block_type_raises_type_error():
    with pytest.raises(TypeError):
        extract_text_from_llm_content([object()])


def test_real_contaminated_pattern_from_gemini_log_recovers_correctly():
    """Reproduz EXATAMENTE a estrutura real observada no log do Gemini
    (`experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/
    call_log_main.jsonl`, HAM_0000005/ISIC_0024579) -- confirma que, se
    esta normalização tivesse sido aplicada na hora, a extração de
    `decisao_final` teria funcionado desde o início."""
    content = [
        {
            "type": "text",
            "text": (
                "A imagem dermatoscópica apresenta características que requerem cautela. "
                "Observa-se uma lesão com áreas de pigmentação irregular.\n\n"
                "decisao_final: indeterminado"
            ),
            "extras": {"signature": "EpwBCpkBAWkUfRNtKadPir9Ft1fJAQrPwwTqYABACXbtBMFYSVPbEkItwXyhpdV8M22Q6"},
        }
    ]
    result = extract_text_from_llm_content(content)
    assert result.endswith("decisao_final: indeterminado")
    assert "\\n" not in result

    from src.evaluation.explicit_decision_contract import DECISAO_INDETERMINADO, DECISION_STATUS_OK, extract_decisao_final

    decision = extract_decisao_final(result)
    assert decision.decision_status == DECISION_STATUS_OK
    assert decision.decisao_final == DECISAO_INDETERMINADO
