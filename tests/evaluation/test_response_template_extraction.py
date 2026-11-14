"""Testes de `src.evaluation.response_template_extraction`.

Reusa, como fixtures, os textos BRUTOS REAIS (literais, copiados sem
paráfrase) dos 8 cases salvos em
`experiments/multiagent/synthesis_controls_real_image/results.json` --
ver `docs/reconstruction/22_final_answer_extraction_preprocessing.md`
para o contexto completo.

Convenção deste diretório reusada: `tests/evaluation/test_case_status_v2.py`
testa o classificador v2 (não tocado aqui); este arquivo testa SÓ a etapa
de extração, estruturalmente anterior e separada.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.response_template_extraction import (  # noqa: E402
    STATUS_EMPTY_AFTER_EXTRACTION,
    STATUS_EXTRACTED_OK,
    STATUS_INCOMPLETE_BLOCK,
    STATUS_NO_TEMPLATE_DETECTED,
    extract_final_answer,
)

# ---------------------------------------------------------------------------
# Fixture real 1: medgemma1.5:4b-it-q4_K_M, cenário `unanime_positivo`,
# model_tags[1] em results.json -- EXCERTO copiado LITERALMENTE (sem
# alterar nenhum caractere) do campo `raw_response_text` do 5o item de
# `calls` (índice 4, 0-based) em
# experiments/multiagent/synthesis_controls_real_image/results.json.
# É um EXCERTO (início do bloco de pensamento + a transição real pelo
# delimitador de fechamento + a resposta final completa), não o texto
# completo de ~1500 palavras -- cada trecho abaixo é uma substring exata
# do original, só que o miolo do bloco de pensamento (que não afeta a
# lógica de extração, só o tamanho do fixture) foi omitido.
# ---------------------------------------------------------------------------
REAL_RAW_TEXT_MEDGEMMA_UNANIME_POSITIVO = (
    "<unused94>thought\n"
    "My thinking process to arrive at the consolidated diagnosis goes like this:\n\n"
    "1.  **Understand the Goal:** The primary objective is to synthesize three different dermatoscopic algorithms (ABCD, Menzies, SPCL) into a single, consolidated diagnosis, justifying the final decision based on the provided image and the input from each algorithm. The key is to integrate the findings, identify conflicts, and ultimately produce a coherent, evidence-based conclusion, while avoiding definitive diagnosis (as per the prompt's instruction).\n\n"
    "2.  **Analyze Each Algorithm's Input:** I need to break down the information provided by each algorithm for the specific lesion in the image.\n\n"
    "    *   **ABCD Analysis:**\n"
    "        *   Asimetria: High (2) - Clearly irregular.\n"
    "        *   Bordas: Mostly sharp (7) - Nítidas em quase todos os quadrantes.\n"
    "        *   Cor: High (5) - Five colors present (white/beige, red/pink, light brown, dark brown, black).\n"
    "        *   Estruturas: High (4) - Pigment network, branched lines, dots/globules observed.\n"
    "        *   Scoring: > 5.45 - Suggests malignancy.\n"
    "        *   Justification: The combination of features (asymmetry, multiple colors, atypical structures) points towards malignancy.\n\n"
    "using strong but non-definitive language as per instructions).<unused95>Análise Integrada dos Pareceres:\n\n"
    "Todos os três algoritmos (ABCD, Menzies, SPCL) identificam características de alta preocupação na lesão.\n\n"
    "1.  **ABCD:** Pontuação alta (>5.45) devido à assimetria, múltiplas cores (5) e estruturas atípicas (rede, linhas ramificadas, pontos/glóbulos).\n"
    "2.  **Menzies:** Múltiplos critérios positivos (veu azul-branco, múltiplos pontos marrons, pseudopodes, pontos pretóricos, múltiplas cores) e ausência de critérios negativos, indicando alta probabilidade de melanoma.\n"
    "3.  **SPCL:** Pontuação alta (7 pontos) com múltiplos critérios específicos (padrão de pigmentação atípico, véu irregular, padrão de retículo irregular, glóbulos irregulares, hiperpigmentação localizada), indicando alta suspeita.\n\n"
    "**Conclusão:** A lesão apresenta uma combinação de características altamente sugestivas de melanoma, confirmadas por múltiplos algoritmos com diferentes abordagens. A presença de múltiplos critérios positivos no Método Menzies e a pontuação alta no SPCL, juntamente com os achados do ABCD, indicam um **diagnóstico de alta suspeita por melanoma**."
)


def test_real_medgemma_response_extracts_final_answer_after_closing_delimiter():
    result = extract_final_answer(REAL_RAW_TEXT_MEDGEMMA_UNANIME_POSITIVO)
    assert result.status == STATUS_EXTRACTED_OK
    assert result.template_removed is True
    assert result.delimiter_pair_used == ("<unused94>thought", "<unused95>")
    # A resposta final extraída deve começar logo após "<unused95>", sem
    # nenhum fragmento do bloco de pensamento interno.
    assert result.final_answer_text is not None
    assert result.final_answer_text.startswith("Análise Integrada dos Pareceres:")
    assert "<unused94>" not in result.final_answer_text
    assert "<unused95>" not in result.final_answer_text
    assert "My thinking process" not in result.final_answer_text
    assert result.final_answer_text.endswith(
        "**diagnóstico de alta suspeita por melanoma**."
    )


# ---------------------------------------------------------------------------
# Fixture real 2: qwen3-vl:8b-instruct-q4_K_M, cenário `um_parecer_ausente`
# -- EXCERTO copiado LITERALMENTE (sem alterar nenhum caractere, só
# omitindo trechos do meio -- "Justificativa" de cada algoritmo,
# "Conflito ou Divergência?", "Características Relevantes Observadas" --
# que não afetam a lógica de extração) do campo `raw_response_text` do
# 4o item de `calls` (índice 3, 0-based) em
# experiments/multiagent/synthesis_controls_real_image/results.json.
# Nenhum marcador `<unused94>`/`<unused95>` deve existir neste texto --
# é exatamente o fato confirmado por leitura manual de todas as 4
# respostas brutas reais do qwen3-vl neste piloto.
# ---------------------------------------------------------------------------
REAL_RAW_TEXT_QWEN_UM_PARECER_AUSENTE = (
    "**Diagnóstico Consolidado e Fundamentado**\n\n---\n\n"
    "**Conclusão Geral**:  \n"
    "**Lesão benigna**, conforme os critérios dos três algoritmos avaliados. "
    "Não há evidência suficiente para suspeitar de malignidade com base nas características observadas.\n\n---\n\n"
    "**Análise Integrada dos Três Pareceres:**\n\n"
    "1. **Algoritmo ABCD**  \n"
    "   - Resultado: **Benigno** (soma ponderada < 4.75)  \n\n"
    "2. **Algoritmo Menzies**  \n"
    "   - Resultado: **Benigno**  \n\n"
    "3. **Algoritmo SPCL (Seven Point Checklist)**  \n"
    "   - Resultado: **Nenhum parecer fornecido**  \n\n---\n\n"
    "**Conclusão Final:**  \n"
    "Os algoritmos ABCD e Menzies são **coerentes entre si** e indicam uma lesão benigna. "
    "A imagem visualmente confirma a uniformidade de cor, ausência de padrões atípicos e estruturas "
    "que poderiam sugerir malignidade. A lesão atende aos critérios de benignidade de ambos os sistemas.\n\n"
    "**Diagnóstico consolidado: Lesão benigna.**\n\n"
    "*(Sem diagnóstico clínico, apenas análise baseada nos algoritmos fornecidos.)*"
)


def test_real_qwen_response_without_markers_passes_through_unchanged():
    result = extract_final_answer(REAL_RAW_TEXT_QWEN_UM_PARECER_AUSENTE)
    assert result.status == STATUS_NO_TEMPLATE_DETECTED
    assert result.template_removed is False
    assert result.delimiter_pair_used is None
    assert result.final_answer_text == REAL_RAW_TEXT_QWEN_UM_PARECER_AUSENTE


# ---------------------------------------------------------------------------
# Casos sintéticos (não copiados de results.json -- construídos para
# exercitar os dois ramos de "problema registrado" pedidos explicitamente
# pelo pesquisador: bloco incompleto e texto vazio após extração).
# ---------------------------------------------------------------------------
def test_incomplete_block_without_closing_delimiter_is_flagged_never_classified():
    raw = "<unused94>thought\nAlgum raciocínio interno que nunca termina com o marcador de fechamento esperado."
    result = extract_final_answer(raw)
    assert result.status == STATUS_INCOMPLETE_BLOCK
    assert result.final_answer_text is None
    assert result.template_removed is False
    assert result.delimiter_pair_used == ("<unused94>thought", "<unused95>")


def test_empty_after_extraction_is_flagged_never_classified():
    raw = "<unused94>thought\nRaciocínio interno qualquer.<unused95>   \n   \t  "
    result = extract_final_answer(raw)
    assert result.status == STATUS_EMPTY_AFTER_EXTRACTION
    assert result.final_answer_text is None
    assert result.delimiter_pair_used == ("<unused94>thought", "<unused95>")


def test_empty_or_none_raw_text_is_treated_as_no_template_detected():
    assert extract_final_answer("").status == STATUS_NO_TEMPLATE_DETECTED
    assert extract_final_answer(None).status == STATUS_NO_TEMPLATE_DETECTED
    assert extract_final_answer(None).final_answer_text is None
