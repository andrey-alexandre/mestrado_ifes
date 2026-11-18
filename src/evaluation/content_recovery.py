"""Recuperação segura de `raw_response_text` corrompido pelo bug de
representação de conteúdo do Gemini (2026-11-18, ver
docs/reconstruction/30_real_lesion_evaluation_protocol.md §11):
`.content` devolvido como lista de blocos, stringificado via `str(lista)`
em algum ponto anterior à correção (`src.agents.agents.
extract_text_from_llm_content`), escapando quebras de linha reais em
`\\n` literal e quebrando a extração de `decisao_final` (regex ancorado
em início de linha).

Este módulo NÃO refaz nenhuma chamada a modelo -- opera inteiramente
sobre texto já salvo (`raw_response_text` de um `call_log*.jsonl`
existente). Usado por `scripts/reprocess_decisao_final.py`.

Segurança: usa `ast.literal_eval`, NUNCA `eval()` -- `literal_eval` só
aceita literais Python (strings, números, listas, dicts, tuplas, sets,
True/False/None), nunca executa chamadas de função ou código arbitrário,
mesmo que o texto de entrada seja hostil/malformado.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional

from src.agents.agents import extract_text_from_llm_content

CONTENT_RECOVERY_VERSION = "content_recovery_v1_2026_11_18"

RECOVERY_NOT_APPLICABLE = "not_applicable"
RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT = "not_needed_already_plain_text"
RECOVERED_FROM_LIST_REPR = "recovered_from_list_repr"
IRRECOVERABLE_PARSE_ERROR = "irrecoverable_parse_error"
IRRECOVERABLE_UNEXPECTED_STRUCTURE = "irrecoverable_unexpected_structure"

# Heurística SÓ para distinguir, entre os casos em que `ast.literal_eval`
# falha, "nunca foi contaminado" (prosa normal) de "foi contaminado mas
# ficou tão corrompido que nem o literal Python pôde ser reconstruído" --
# NUNCA usada para decidir se o texto É recuperável (isso é sempre
# `ast.literal_eval` + `extract_text_from_llm_content`, nunca uma
# heurística de substring). Puramente informativa no relatório.
_LIST_REPR_PREFIXES = ("[{", "[ {", "[")


@dataclass
class RecoveryResult:
    recovery_status: str
    recovered_text: Optional[str]
    parse_error_detail: Optional[str] = None


def recover_raw_response_text(raw_response_text: Optional[str]) -> RecoveryResult:
    """Tenta recuperar o texto real a partir de ``raw_response_text`` já
    salvo (possivelmente corrompido pelo bug de `str(lista)`).

    Algoritmo (NUNCA usa `eval()`, NUNCA stringifica heuristicamente):
      1. ``None`` -> `RECOVERY_NOT_APPLICABLE` (nada para recuperar).
      2. ``ast.literal_eval(raw_response_text)`` -- se levantar
         ``SyntaxError``/``ValueError`` (texto não é um literal Python
         válido): o texto nunca foi corrompido por este bug específico
         (prosa normal não é sintaxe de literal Python) ->
         `RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT` (``recovered_text`` =
         o próprio ``raw_response_text``, sem nenhuma alteração) -- SALVO
         se o texto come;ar com um prefixo de lista/dict (`_LIST_REPR_PREFIXES`),
         caso em que é reportado como `IRRECOVERABLE_PARSE_ERROR` em vez
         disso (sinal informativo de que PARECIA ser uma representação
         de lista mas não pôde ser reconstruída -- usado só para o
         relatório, nunca para decidir recuperabilidade -- a decisão real
         é sempre `ast.literal_eval` + `extract_text_from_llm_content`).
      3. Se `literal_eval` tiver sucesso mas o resultado não for uma
         `list` -- nunca visto no bug real, mas tratado com segurança:
         mesma regra do passo 2 (não precisa de recuperação, o valor
         original não era uma representação de lista de blocos).
      4. Se o resultado for uma `list`: aplica
         `extract_text_from_llm_content` (MESMA função usada na correção
         prospectiva, nunca uma segunda implementação) -- sucesso ->
         `RECOVERED_FROM_LIST_REPR`; `TypeError` (estrutura de bloco
         inesperada) -> `IRRECOVERABLE_UNEXPECTED_STRUCTURE`
         (``recovered_text=None`` -- nunca um valor parcial/adivinhado).
    """
    if raw_response_text is None:
        return RecoveryResult(RECOVERY_NOT_APPLICABLE, None)

    try:
        parsed = ast.literal_eval(raw_response_text)
    except (SyntaxError, ValueError) as exc:
        if raw_response_text.strip().startswith(_LIST_REPR_PREFIXES):
            return RecoveryResult(IRRECOVERABLE_PARSE_ERROR, None, parse_error_detail=f"{type(exc).__name__}: {exc}")
        return RecoveryResult(RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT, raw_response_text)

    if not isinstance(parsed, list):
        return RecoveryResult(RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT, raw_response_text)

    try:
        recovered_text = extract_text_from_llm_content(parsed)
    except TypeError as exc:
        return RecoveryResult(IRRECOVERABLE_UNEXPECTED_STRUCTURE, None, parse_error_detail=str(exc))

    return RecoveryResult(RECOVERED_FROM_LIST_REPR, recovered_text)
