"""Extração da RESPOSTA FINAL a partir do texto bruto salvo de uma chamada
de VLM -- etapa estruturalmente ANTERIOR a qualquer classificação
(`src.evaluation.case_status_v2.classify_case_status_v2_detailed` ou
`src.evaluation.case_status.classify_case_status`).

Motivação REAL (não hipotética): revisão de fidelidade dos 4 controles de
síntese reais do modelo `medgemma1.5:4b-it-q4_K_M`
(`experiments/multiagent/synthesis_controls_real_image/results.json`,
fechamento em
`docs/reconstruction/21_synthesis_control_fidelity_review_closure.md §5`)
encontrou que as 4 respostas brutas desse modelo chegam ao pipeline com um
bloco de cadeia-de-pensamento interna do template de chat do
Ollama/medgemma, NUNCA filtrado, concatenado ANTES da resposta final de
fato entregue. Os 4 textos brutos reais (citação literal, não paráfrase,
de `results.json`) têm o formato:

    "<unused94>thought\n<...bloco de raciocínio interno, às vezes muito
    mais longo que a resposta final...><unused95><resposta final de
    fato>"

Exemplo literal observado (início do caso `unanime_positivo`):
    "<unused94>thought\nMy thinking process to arrive at the consolidated
    diagnosis goes like this:\n\n1.  **Understand the Goal:** ..."
e o ponto de transição real (mesmo caso, citação literal):
    "...using strong but non-definitive language as per
    instructions).<unused95>Análise Integrada dos Pareceres:\n\n..."

A suspeita registrada no fechamento §5 (nunca confirmada antes desta
tarefa) é que a heurística de "conclusão final" de
`case_status_v2._extract_final_conclusion_region` -- que opera por METADE
FINAL DO DOCUMENTO por índice de parágrafo -- pode estar varrendo texto do
BLOCO DE RACIOCÍNIO (que é desproporcionalmente longo) em vez da resposta
final de fato entregue, no caso `medgemma1.5/conflitante`. Esta tarefa
extrai a resposta final ANTES do parser rodar, para que essa suspeita
possa ser testada com o texto correto chegando ao classificador -- SEM
modificar `case_status_v2.py` nem os prompts reais (ver
`scripts/run_reprocess_synthesis_controls_final_answer_extraction.py` e
`docs/reconstruction/22_final_answer_extraction_preprocessing.md`).

Decisão de design explícita do pesquisador (verbatim, ver pedido desta
tarefa): "Remova apenas blocos de template com delimitadores
reconhecidos; se o bloco estiver incompleto ou não sobrar resposta final,
registre o problema sem tentar classificar o texto bruto." -- ou seja,
este módulo NUNCA tenta "adivinhar" uma resposta final quando a extração
falha; ele registra o problema explicitamente (`status`) e devolve
`final_answer_text=None` nesses casos, para que o chamador NUNCA alimente
texto parcial/indefinido ao classificador.

Este módulo é PURO e não importa nada de `case_status_v2.py` (nem de
`case_status.py`): é uma etapa de pré-processamento estruturalmente
separada da classificação, que roda ANTES dela. Não altera o
comportamento de nenhum parser existente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Vocabulário FECHADO e SÓ-OBSERVADO de pares de delimitador de template de
# chat conhecidos -- mesmo espírito/mesma transparência de outros
# vocabulários fechados já existentes no projeto (ex.:
# `case_status_v2._AFFIRMATIVE_QUALIFIER_WORDS`,
# `case_status_v2._HEDGE_MARKERS`): uma lista pequena, inspecionável,
# justificada por evidência REAL observada, não uma tentativa de cobrir
# "todo template de chat possível".
#
# Esta entrada única foi confirmada OBSERVADA NAS 4 RESPOSTAS BRUTAS REAIS
# de `medgemma1.5:4b-it-q4_K_M` em
# `experiments/multiagent/synthesis_controls_real_image/results.json`
# (cenários unanime_positivo, unanime_negativo, conflitante,
# um_parecer_ausente) -- citações literais acima no docstring do módulo.
#
# AVISO EXPLÍCITO (fazer parte de qualquer uso deste módulo): esta lista é
# FECHADA E SÓ-OBSERVADA. Outros modelos, outras quantizações, ou mesmo
# outra versão do MESMO modelo/template, podem usar marcadores
# DIFERENTES, não cobertos aqui (ex.: `<|thinking|>`/`<|end_thinking|>`,
# `<think>`/`</think>`, etc. -- nenhum desses foi observado neste piloto,
# citados aqui só como exemplos do tipo de marcador que ESTE módulo NÃO
# reconhece). Um texto bruto de um modelo diferente com vazamento de
# template usando outro par de marcadores cairá em
# `status="no_template_detected"` (tratado como "sem template", não como
# erro) -- isso é uma LIMITAÇÃO CONHECIDA, não um bug: ampliar esta lista
# exige observar o marcador real em dados reais primeiro, no mesmo
# espírito do resto deste projeto ("nunca hipotético").
_KNOWN_TEMPLATE_DELIMITER_PAIRS: tuple[tuple[str, str], ...] = (
    ("<unused94>thought", "<unused95>"),
)

# Status possíveis -- vocabulário fechado, usado tanto pelo dataclass
# devolvido quanto pelo script de reprocessamento para decidir se deve ou
# não tentar classificar o texto resultante.
STATUS_EXTRACTED_OK = "extracted_ok"
STATUS_NO_TEMPLATE_DETECTED = "no_template_detected"
STATUS_INCOMPLETE_BLOCK = "incomplete_block_no_closing_delimiter"
STATUS_EMPTY_AFTER_EXTRACTION = "empty_after_extraction"

ALL_EXTRACTION_STATUSES = (
    STATUS_EXTRACTED_OK,
    STATUS_NO_TEMPLATE_DETECTED,
    STATUS_INCOMPLETE_BLOCK,
    STATUS_EMPTY_AFTER_EXTRACTION,
)

# Status que representam um PROBLEMA registrado -- o pesquisador pediu
# explicitamente que, nesses casos, NENHUMA tentativa de classificação
# seja feita sobre o texto bruto/parcial. O chamador (script de
# reprocessamento, e futuramente `run_vlm_direct_call`/
# `run_multiagent_role_call`, quando/se isso for aprovado) deve checar
# `result.status in PROBLEM_STATUSES` antes de classificar.
PROBLEM_STATUSES = (STATUS_INCOMPLETE_BLOCK, STATUS_EMPTY_AFTER_EXTRACTION)


@dataclass
class ExtractionResult:
    """Resultado da extração da resposta final.

    Campos:
      final_answer_text -- o texto a ser classificado. Para
        `status in PROBLEM_STATUSES`, é SEMPRE `None` (nunca uma string
        parcial/vazia "quase vazia" -- decisão explícita do pesquisador de
        nunca tentar classificar nesse caso). Para
        `status == no_template_detected`, é o texto bruto ORIGINAL,
        inalterado (nenhuma remoção ocorreu). Para
        `status == extracted_ok`, é o texto após o delimitador de
        fechamento, com espaço em branco de borda removido (`.strip()`).
      template_removed -- `True` somente quando uma remoção de fato
        ocorreu e produziu texto (status `extracted_ok`). `False` para
        `no_template_detected` (nada para remover) e também para
        `incomplete_block_no_closing_delimiter` (delimitador de abertura
        encontrado, mas a remoção NUNCA foi concluída -- não houve, de
        fato, uma remoção bem-sucedida, por isso `False`, não `True`,
        apesar do delimitador de abertura ter sido detectado). Esta é uma
        decisão de design DESTA tarefa, não dita verbatim pelo pedido
        original -- documentada aqui para transparência.
      status -- um de `ALL_EXTRACTION_STATUSES`.
      delimiter_pair_used -- a tupla `(abertura, fechamento)` de
        `_KNOWN_TEMPLATE_DELIMITER_PAIRS` que foi usada/detectada, ou
        `None` se nenhum delimitador de abertura foi encontrado
        (`no_template_detected`).
    """

    final_answer_text: Optional[str]
    template_removed: bool
    status: str
    delimiter_pair_used: Optional[tuple[str, str]]


def extract_final_answer(raw_text: Optional[str]) -> ExtractionResult:
    """Extrai a resposta final de ``raw_text``, removendo apenas blocos de
    template delimitados por um par CONHECIDO
    (``_KNOWN_TEMPLATE_DELIMITER_PAIRS``). Nunca tenta heurística alguma
    sobre o CONTEÚDO do texto (isso é trabalho do classificador,
    estruturalmente posterior e não tocado por este módulo).

    Regras (na ordem em que são aplicadas):

    1. Texto vazio/``None`` -- devolvido como está, sem template, para que
       o chamador trate igual a qualquer outro "sem template detectado"
       (não é um caso especial deste módulo; o classificador já tem seu
       próprio tratamento de texto vazio).
    2. Nenhum delimitador de ABERTURA conhecido aparece em `raw_text` em
       lugar nenhum -> ``no_template_detected``, ``final_answer_text`` =
       `raw_text` inalterado, ``template_removed=False``. Caso esperado
       para TODAS as 4 respostas brutas reais de
       `qwen3-vl:8b-instruct-q4_K_M` deste piloto (confirmado por leitura
       manual -- nenhuma contém `<unused94>`/`<unused95>`).
    3. Delimitador de ABERTURA encontrado, mas NENHUM delimitador de
       FECHAMENTO correspondente aparece em lugar nenhum DEPOIS dele ->
       ``incomplete_block_no_closing_delimiter``,
       ``final_answer_text=None`` -- problema registrado, nunca
       classificado.
    4. Delimitador de FECHAMENTO encontrado, mas o texto remanescente
       (depois dele, após `.strip()`) é vazio -> ``empty_after_extraction``,
       ``final_answer_text=None`` -- problema registrado, nunca
       classificado.
    5. Caso contrário -> ``extracted_ok``, ``final_answer_text`` = texto
       depois do delimitador de fechamento, com `.strip()`.

    Quando mais de um par de delimitador conhecido tem uma abertura
    presente no texto, o par cuja abertura ocorre MAIS À ESQUERDA (menor
    índice) é o escolhido -- não ocorre na prática neste piloto (só um
    par conhecido), documentado aqui por completude caso o vocabulário
    seja ampliado no futuro com evidência real de outro modelo.
    """
    if not raw_text:
        return ExtractionResult(
            final_answer_text=raw_text,
            template_removed=False,
            status=STATUS_NO_TEMPLATE_DETECTED,
            delimiter_pair_used=None,
        )

    best: Optional[tuple[int, str, str]] = None
    for opening, closing in _KNOWN_TEMPLATE_DELIMITER_PAIRS:
        idx = raw_text.find(opening)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, opening, closing)

    if best is None:
        return ExtractionResult(
            final_answer_text=raw_text,
            template_removed=False,
            status=STATUS_NO_TEMPLATE_DETECTED,
            delimiter_pair_used=None,
        )

    open_idx, opening, closing = best
    close_idx = raw_text.find(closing, open_idx + len(opening))
    if close_idx == -1:
        return ExtractionResult(
            final_answer_text=None,
            template_removed=False,
            status=STATUS_INCOMPLETE_BLOCK,
            delimiter_pair_used=(opening, closing),
        )

    remainder = raw_text[close_idx + len(closing):].strip()
    if not remainder:
        return ExtractionResult(
            final_answer_text=None,
            template_removed=True,
            status=STATUS_EMPTY_AFTER_EXTRACTION,
            delimiter_pair_used=(opening, closing),
        )

    return ExtractionResult(
        final_answer_text=remainder,
        template_removed=True,
        status=STATUS_EXTRACTED_OK,
        delimiter_pair_used=(opening, closing),
    )
