"""Protótipo EXPERIMENTAL (Etapa 5, achado do doc 22): parser de um
contrato de decisão final EXPLÍCITO, comum ao VLM-direto e ao
sumarizador.

Motivação REAL (não hipotética): a revisão de fidelidade confirmada com
execução real em
`docs/reconstruction/22_final_answer_extraction_preprocessing.md §3`
mostrou que o classificador v2
(`src.evaluation.case_status_v2.classify_case_status_v2`) pode capturar
a palavra "benigno" dentro de uma frase que ENUMERA pareceres
conflitantes ("resultados que variam de benigno (ABCD) a melanoma
(Menzies) e borderline (SPCL)") como se fosse um veredito negativo --
mesmo quando o sumarizador NUNCA tomou essa decisão. Isso é uma falha
de CORRESPONDÊNCIA de um parser que infere o rótulo a partir de
palavras-chave no corpo do texto.

Este módulo não tenta corrigir essa inferência -- substitui a
inferência por um CAMPO DECLARADO EXPLICITAMENTE pelo próprio modelo,
que o parser aqui só LÊ (nunca deduz a partir da justificativa).

Contrato (sintaxe única, documentada aqui -- fonte de verdade para as
duas instruções de tarefa experimentais em
`src.multiagent.experimental_decision_contract`): uma linha dedicada,
em qualquer ponto do texto (tipicamente ao final), exatamente no
formato

    decisao_final: <valor>

onde ``<valor>`` deve ser um de ``"melanoma"``, ``"nao_melanoma"`` ou
``"indeterminado"`` (comparação case-insensitive só quanto à CAIXA do
valor -- "Melanoma"/"MELANOMA" são aceitos, "melanoma." com ponto
final ou "suspeita de melanoma" como valor NÃO são -- nenhuma
correspondência fuzzy/aproximada é feita aqui, ao contrário do v2). A
chave ``decisao_final`` é comparada case-insensitive e sem acentos
(já é a forma como foi escrita nas instruções experimentais). Markdown
em torno da linha (negrito, marcadores de lista) NÃO é reconhecido por
desenho -- a linha deve estar exatamente como instruído.

EXCEÇÃO EXPLÍCITA, única e documentada (adicionada após a 1ª execução
real com imagem real, ver
`docs/reconstruction/25_accent_variant_and_reprocessing.md`):
``"não_melanoma"`` (grafia correta em português, com "ã") é aceito como
VARIANTE LITERAL de ``"nao_melanoma"`` -- só essa forma específica, só
para esse valor. Observado em 2 das 8 respostas reais do medgemma1.5
(`decisao_final: não_melanoma`), que a comparação estritamente ASCII
original rejeitava como `invalid_value` mesmo sendo uma decisão
inequívoca e corretamente grafada. Isto NÃO reabre a porta para
correspondência fuzzy: é uma entrada FECHADA e SÓ-OBSERVADA na tabela
`_VALUE_VARIANTS` abaixo (mesmo espírito do vocabulário fechado de
`_KNOWN_TEMPLATE_DELIMITER_PAIRS` em `response_template_extraction.py`)
-- qualquer outra grafia (com ou sem acento, de qualquer um dos 3
valores) que não esteja explicitamente listada ali continua caindo em
`invalid_value`, nunca aceita por aproximação. A exceção se aplica
IDENTICAMENTE ao VLM-direto e ao sumarizador, porque os dois usam esta
mesma função.

Este módulo é PURO e ISOLADO:
  - roda ESTRUTURALMENTE APÓS `src.evaluation.response_template_extraction
    .extract_final_answer` (reusado aqui, não reimplementado) -- o campo
    é procurado SÓ na resposta final já livre do bloco de pensamento do
    template de chat, nunca no texto bruto completo. Isto é o que
    garante que um rótulo mencionado dentro do bloco de pensamento (ex.:
    um rascunho "decisao_final: melanoma" dentro de
    `<unused94>thought ... <unused95>`) NUNCA seja lido -- só a linha
    que aparece DEPOIS do delimitador de fechamento conta.
  - NÃO importa nada de `case_status.py`/`case_status_v2.py` -- é uma
    via de classificação SEPARADA e PARALELA, não uma substituição
    (ambos os parsers antigos continuam intocados e em uso onde já
    estavam).
  - NÃO é chamado por nenhum caminho de produção
    (`src/multiagent/vlm_direct.py`, `scripts/run_local_vlm_pilot.py`)
    -- protótipo para verificação manual nesta entrega, sem inferência
    real (ver `scripts/verify_explicit_decision_contract.py`).

Falhas de FORMATO (campo ausente, repetido -- mesmo com valores iguais
--, ou com valor fora do vocabulário fechado) são sempre reportadas
como ``decision_status`` distinto de ``DECISION_STATUS_OK`` e NUNCA
resolvidas por dedução a partir do corpo do texto -- nenhum fallback
para o parser v2 nem para qualquer heurística lexical existe aqui, por
instrução explícita do pesquisador.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.evaluation.response_template_extraction import (
    PROBLEM_STATUSES as _TEMPLATE_PROBLEM_STATUSES,
    extract_final_answer,
)

# Versão do parser -- trocada (nunca mantida silenciosamente) sempre que
# o vocabulário fechado/a sintaxe/as variantes aceitas mudarem (mesma
# convenção de `case_status_v2.PARSER_VERSION`). Usada por
# `scripts/run_local_vlm_pilot.py` como `official_classification_parser_version`
# a partir da troca de gate do doc 26.
PARSER_VERSION = "explicit_decision_contract_v1_accented_nao_melanoma_variant_2026_11_14"

# ---------------------------------------------------------------------------
# Vocabulário fechado de decisão -- os 3 valores CANÔNICOS devolvidos em
# `decisao_final` quando `decision_status == DECISION_STATUS_OK` (nunca a
# string literal da variante aceita -- ver `_VALUE_VARIANTS` abaixo, que
# sempre normaliza de volta para um destes 3).
# ---------------------------------------------------------------------------
DECISAO_MELANOMA = "melanoma"
DECISAO_NAO_MELANOMA = "nao_melanoma"
DECISAO_INDETERMINADO = "indeterminado"

ALL_DECISIONS: tuple[str, ...] = (
    DECISAO_MELANOMA,
    DECISAO_NAO_MELANOMA,
    DECISAO_INDETERMINADO,
)

# ---------------------------------------------------------------------------
# Variantes literais ACEITAS por valor canônico -- vocabulário FECHADO e
# SÓ-OBSERVADO (nenhuma entrada hipotética). Cada chave é um valor de
# `ALL_DECISIONS`; cada tupla contém o próprio valor canônico (sempre
# aceito) e, quando observada em dados reais, uma grafia alternativa
# EXATA. Comparação feita após `.strip().lower()` do valor capturado pela
# regex -- nunca uma segunda rodada de regex/fuzzy aqui.
#
# Entrada real observada (2026-11-14, 1ª execução real com imagem real,
# ver docs/reconstruction/25_accent_variant_and_reprocessing.md): o
# medgemma1.5:4b-it-q4_K_M escreveu `decisao_final: não_melanoma` (com
# "ã", grafia correta em português) em 2 das 4 respostas, em vez do
# token ASCII `nao_melanoma` pedido literalmente na instrução
# experimental. Adicionada como variante aceita SÓ para
# `DECISAO_NAO_MELANOMA` -- `"melanoma"`/`"indeterminado"` não têm
# variante conhecida (nenhuma observada ainda; se uma for observada no
# futuro, deve ser adicionada aqui pelo mesmo processo, nunca por uma
# regra geral de "remover acentos").
_VALUE_VARIANTS: dict[str, tuple[str, ...]] = {
    DECISAO_MELANOMA: (DECISAO_MELANOMA,),
    DECISAO_NAO_MELANOMA: (DECISAO_NAO_MELANOMA, "não_melanoma"),
    DECISAO_INDETERMINADO: (DECISAO_INDETERMINADO,),
}

# Mapa inverso (variante literal -> valor canônico), construído uma vez a
# partir de `_VALUE_VARIANTS` -- nunca duplicado manualmente em outro
# lugar do módulo.
_VARIANT_TO_CANONICAL: dict[str, str] = {
    variant: canonical
    for canonical, variants in _VALUE_VARIANTS.items()
    for variant in variants
}

DECISION_FIELD_KEY = "decisao_final"

# ---------------------------------------------------------------------------
# Status de resultado do parser -- vocabulário fechado, distinto do
# `status`/vocabulário de `response_template_extraction` (que descreve o
# PRÉ-PROCESSAMENTO de template, uma etapa anterior e separada).
# ---------------------------------------------------------------------------
DECISION_STATUS_OK = "ok"
# A extração do template (etapa anterior) falhou -- nunca chegamos a
# procurar o campo porque não há um texto final confiável para procurar
# nele (ver `response_template_extraction.PROBLEM_STATUSES`). Distinto de
# qualquer falha de FORMATO do campo em si.
DECISION_STATUS_EXTRACTION_FAILED = "extraction_failed"
# O texto final (já livre do template) não contém NENHUMA linha
# correspondente ao formato exigido.
DECISION_STATUS_FIELD_ABSENT = "field_absent"
# O texto final contém MAIS DE UMA linha correspondente ao formato
# exigido -- inclusive quando os valores são idênticos entre si (decisão
# explícita desta tarefa: a REPETIÇÃO em si já é uma falha de formato,
# não é "tratada como válida porque os valores concordam").
DECISION_STATUS_FIELD_REPEATED = "field_repeated"
# Exatamente uma linha encontrada, mas o valor declarado não é um dos 3
# tokens do vocabulário fechado `ALL_DECISIONS`.
DECISION_STATUS_INVALID_VALUE = "invalid_value"

ALL_DECISION_STATUSES: tuple[str, ...] = (
    DECISION_STATUS_OK,
    DECISION_STATUS_EXTRACTION_FAILED,
    DECISION_STATUS_FIELD_ABSENT,
    DECISION_STATUS_FIELD_REPEATED,
    DECISION_STATUS_INVALID_VALUE,
)

# Falhas de FORMATO do campo em si -- distintas, por desenho, de
# DECISION_STATUS_EXTRACTION_FAILED (que é uma falha de uma etapa
# ANTERIOR e separada, o pré-processamento de template -- nunca chegamos
# a procurar o campo nesse caso). Esta tupla NUNCA inclui
# DECISION_STATUS_EXTRACTION_FAILED -- misturar as duas categorias
# apagaria a distinção entre "falha de extração do template", "falha de
# formato do campo declarado" e "decisão indeterminada válida" que esta
# tarefa pede explicitamente para preservar.
FIELD_FORMAT_FAILURE_STATUSES: tuple[str, ...] = (
    DECISION_STATUS_FIELD_ABSENT,
    DECISION_STATUS_FIELD_REPEATED,
    DECISION_STATUS_INVALID_VALUE,
)

# Regex da linha dedicada: chave `decisao_final` (case-insensitive),
# dois-pontos, valor = resto da linha (sem espaço de borda). `re.MULTILINE`
# para `^`/`$` por linha; `re.IGNORECASE` só afeta a chave literal aqui --
# a comparação do VALOR contra `ALL_DECISIONS` é feita separadamente
# (abaixo, após `.lower()` explícito), nunca dentro da regex.
_DECISION_LINE_PATTERN = re.compile(
    r"^[ \t]*decisao_final[ \t]*:[ \t]*(.+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class DecisionContractResult:
    """Resultado da leitura do contrato de decisão final.

    Campos:
      raw_response_text -- o texto bruto ORIGINAL, preservado sem
        qualquer alteração, para auditoria (nunca sobrescrito/perdido,
        mesmo quando `decision_status != DECISION_STATUS_OK`).
      template_extraction_status -- o `status` devolvido por
        `extract_final_answer` para `raw_response_text` (etapa anterior,
        reusada sem modificação). Registrado aqui para que quem ler o
        resultado saiba SEMPRE por que `final_answer_text` é o que é,
        sem precisar rechamar `extract_final_answer`.
      final_answer_text -- o texto efetivamente escaneado à procura do
        campo (já livre do bloco de template, quando havia um). ``None``
        quando `template_extraction_status` é um dos
        `response_template_extraction.PROBLEM_STATUSES` (mesma regra:
        nunca escanear texto parcial/indefinido).
      decision_status -- um de `ALL_DECISION_STATUSES`.
      decisao_final -- a decisão lida (um de `ALL_DECISIONS`), SOMENTE
        quando `decision_status == DECISION_STATUS_OK`. ``None`` em
        qualquer outro caso -- nunca um valor "melhor esforço".
      matched_raw_values -- tupla com o valor BRUTO (antes de
        `.strip().lower()`) de cada linha que bateu no padrão da chave,
        na ordem em que aparecem no texto. Tamanho 0 (ausente), 1 (caso
        normal) ou >1 (repetido) -- preservado mesmo em caso de falha,
        para diagnóstico/auditoria humana do porquê da falha.
    """

    raw_response_text: Optional[str]
    template_extraction_status: str
    final_answer_text: Optional[str]
    decision_status: str
    decisao_final: Optional[str]
    matched_raw_values: tuple[str, ...]


def extract_decisao_final(raw_text: Optional[str]) -> DecisionContractResult:
    """Lê o campo `decisao_final` de ``raw_text``.

    Ordem de operações (nunca invertida):
      1. `extract_final_answer(raw_text)` -- remove o bloco de template
         conhecido, se houver (reuso sem modificação). Se essa etapa
         reportar um problema (`PROBLEM_STATUSES`), devolve
         `DECISION_STATUS_EXTRACTION_FAILED` imediatamente -- nunca
         tenta procurar o campo no texto bruto/parcial.
      2. Procura TODAS as linhas que correspondem ao formato
         `decisao_final: <valor>` no texto final (não no texto bruto
         completo -- ver docstring do módulo sobre por que isto importa
         para o caso "rótulo dentro do bloco de pensamento").
      3. 0 ocorrências -> `DECISION_STATUS_FIELD_ABSENT`.
         >1 ocorrências -> `DECISION_STATUS_FIELD_REPEATED` (mesmo se os
         valores coincidirem).
         Exatamente 1 ocorrência, valor (após `.strip().lower()`) fora
         de `_VARIANT_TO_CANONICAL` (os 3 valores canônicos + a única
         variante acentuada aceita, ver docstring do módulo) ->
         `DECISION_STATUS_INVALID_VALUE`.
         Exatamente 1 ocorrência válida -> `DECISION_STATUS_OK`,
         `decisao_final` = o valor CANÔNICO correspondente (nunca a
         variante literal capturada -- ex.: "não_melanoma" normaliza
         para `"nao_melanoma"` em `decisao_final`, mas o texto
         originalmente capturado continua disponível em
         `matched_raw_values` para auditoria).

    Nunca deduz a decisão a partir da justificativa/corpo do texto --
    só lê a linha declarada. Nunca usa nenhum outro parser (v1/v2) como
    fallback em caso de falha.
    """
    extraction = extract_final_answer(raw_text)

    if extraction.status in _TEMPLATE_PROBLEM_STATUSES:
        return DecisionContractResult(
            raw_response_text=raw_text,
            template_extraction_status=extraction.status,
            final_answer_text=None,
            decision_status=DECISION_STATUS_EXTRACTION_FAILED,
            decisao_final=None,
            matched_raw_values=(),
        )

    text_to_scan = extraction.final_answer_text or ""
    matched_raw_values = tuple(
        match.strip() for match in _DECISION_LINE_PATTERN.findall(text_to_scan)
    )

    if len(matched_raw_values) == 0:
        decision_status = DECISION_STATUS_FIELD_ABSENT
        decisao_final: Optional[str] = None
    elif len(matched_raw_values) > 1:
        decision_status = DECISION_STATUS_FIELD_REPEATED
        decisao_final = None
    else:
        normalized = matched_raw_values[0].lower()
        canonical = _VARIANT_TO_CANONICAL.get(normalized)
        if canonical is None:
            decision_status = DECISION_STATUS_INVALID_VALUE
            decisao_final = None
        else:
            decision_status = DECISION_STATUS_OK
            decisao_final = canonical

    return DecisionContractResult(
        raw_response_text=raw_text,
        template_extraction_status=extraction.status,
        final_answer_text=extraction.final_answer_text,
        decision_status=decision_status,
        decisao_final=decisao_final,
        matched_raw_values=matched_raw_values,
    )
