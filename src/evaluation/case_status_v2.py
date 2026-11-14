"""NOVO classificador de status de caso -- v2, "conclusão final".

Motivação REAL (não hipotética): piloto local de VLM
`qwen3-vl:8b-instruct-q4_K_M`, Estágio 1, controle de síntese
`um_parecer_ausente` (texto bruto salvo em
`experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/
call_log.jsonl` e `report.md`). O classificador v1
(`src.evaluation.case_status.classify_case_status`) devolveu
`"nao_reconhecido"` para esse caso porque o modelo nunca escreveu a
frase literal "lesão benigna": no meio do texto usou "Lesão Cutânea
Benigna" (a palavra "Cutânea" quebra o substring) e no veredito final
usou apenas "BENIGNO", isolado, sem a palavra "lesão". Esse é o gap já
registrado em `docs/reconstruction/15_multiagent_experimental_matrix.md
§6.1` (modelos locais via Ollama, sem `with_structured_output`,
parafraseiam e não batem a frase literal exigida pela A34).

Decisão do pesquisador (2026-11-13, verbatim em
`docs/reconstruction/20_local_vlm_pilot.md`): NÃO modificar
`classify_case_status` (reusado em avaliações históricas/stage6 --
"mantenha as avaliações históricas intactas"). Construir, em vez disso,
um classificador NOVO e SEPARADAMENTE VERSIONADO, que:
  1. extrai a CONCLUSÃO FINAL do texto -- não varre o corpo inteiro
     procurando palavras-chave (a menção no meio do texto
     "Lesão Cutânea Benigna" NÃO deve, por si só, disparar a
     classificação);
  2. reusa o MESMO vocabulário de status de `case_status`
     (POSITIVO/NEGATIVO/INDETERMINADO/AUSENTE/NAO_RECONHECIDO);
  3. aceita, nessa região de conclusão final (só lá, nunca no corpo
     inteiro), um mapeamento explícito e versionado de variantes extras
     -- no mínimo "benigno"/"benigna" e "lesão cutânea benigna" como
     negativo inequívoco;
  4. trata negação de forma aproximada (ex.: "não é benigna, é suspeita
     de melanoma" não deve cair em negativo só porque "benigna"
     aparece);
  5. segue o ÚLTIMO veredito categórico estabelecido quando o corpo tem
     conclusões conflitantes entre parágrafos (ex.: cenário real
     "conflitante" deste piloto: ABCD=benigno, Menzies=melanoma,
     SPCL=borderline, com veredito final "Suspeita de Melanoma segundo
     critério Menzies");
  6. preserva indeterminado/ausente/não_reconhecido como estados
     distintos -- nunca dobrados em positivo/negativo.

Este módulo é aditivo: não é chamado por `classify_case_status`, por
`src.evaluation.stage6_metrics`, nem por `vlm_direct.classify_call_outcome`
(que resolve recusa/falha técnica, algo não relacionado e não tocado
aqui). É usado ADITIVAMENTE em `src/multiagent/vlm_direct.py`
(`run_vlm_direct_call`/`run_multiagent_role_call`, novo campo
`classified_status_v2` em `CallRecord`, calculado ao lado de
`classified_status` sem substituí-lo) e no script de reclassificação
offline `scripts/run_reclassify_pilot_with_v2_parser.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from src.evaluation.case_status import (
    AUSENTE,
    INDETERMINADO,
    NAO_RECONHECIDO,
    NEGATIVO,
    POSITIVO,
)

PARSER_VERSION = "v2_conclusion_line_positivo_suspeita_melanoma_2026_11_14"

# Entrega 2026-11-14 (ver docs/reconstruction/20_local_vlm_pilot.md §9b):
# decisão verbatim do pesquisador -- "Escolha a opção 1, mas não aceite
# qualquer ocorrência de 'suspeita ... melanoma'. Aceite a variante
# quando a conclusão final expressar suspeita afirmativa e inequívoca de
# melanoma, como no caso relatado: 'suspeita clara e consistente de
# melanoma maligno'." -- extensão do rótulo POSITIVO (ver seção 1b/2b
# abaixo) para cobrir o cenário real `unanime_positivo` deste piloto
# (v1 acertou por coincidência de substring; v2 original errou porque a
# região de conclusão final real paraphraseia "suspeita de melanoma"
# com palavras intensificadoras no meio). O PARSER_VERSION foi trocado
# (nunca mantido silenciosamente) para que resultados antigos e novos
# sejam rastreáveis a qual versão do parser os produziu.

# ---------------------------------------------------------------------------
# 1. Mapeamento explícito e versionado de variantes de rótulo aceitas --
#    estrutura inspecionável (dict de tuplas), não literais espalhados.
#    Reusa o MESMO vocabulário de status de case_status (import acima),
#    nunca redefine novos nomes de status.
#
#    IMPORTANTE: estas variantes só são procuradas na REGIÃO DE CONCLUSÃO
#    FINAL extraída por `_extract_final_conclusion_region` -- nunca no
#    corpo inteiro da resposta (é exatamente essa restrição que evita o
#    falso-positivo de "Lesão Cutânea Benigna" no meio do texto do
#    cenário real `um_parecer_ausente` disparar a classificação por si
#    só, independente de ser ou não a conclusão real).
#
#    A entrada de INDETERMINADO é EXATAMENTE a mesma frase literal de
#    `case_status._KEYWORD_TO_STATUS` -- não foi tocada por esta entrega.
#    A entrada de POSITIVO listada aqui ("suspeita de melanoma") também
#    é a mesma frase literal original; a EXTENSÃO pedida pelo pesquisador
#    em 2026-11-14 ("suspeita <intensificador> de melanoma", ex. "suspeita
#    clara e consistente de melanoma") é implementada SEPARADAMENTE via
#    `_AFFIRMATIVE_SUSPEITA_MELANOMA_PATTERN` (ver seção 1b, abaixo da
#    negação) -- não foi adicionada a este dict porque é um padrão regex
#    com vocabulário fechado de intensificadores, não uma lista plana de
#    frases literais; ambas as fontes de match POSITIVO passam pelas
#    mesmas checagens de negação/"sem"-prefixo/incerteza antes de contar
#    como válidas (ver `_find_all_label_matches`). A palavra solta
#    "melanoma", sem "suspeita", continua deliberadamente NÃO aceita
#    (aparece com frequência em discussões analíticas que NÃO são o
#    veredito, como no cenário real "conflitante" -- "Suspeita de
#    Melanoma segundo critério Menzies" já é capturada pela frase
#    literal/padrão existente, sem precisar da palavra solta).
# ---------------------------------------------------------------------------
LABEL_VARIANTS_V2: dict[str, tuple[str, ...]] = {
    POSITIVO: (
        "suspeita de melanoma",  # paridade com v1/case_status._KEYWORD_TO_STATUS
    ),
    NEGATIVO: (
        "lesão benigna", "lesao benigna",            # paridade com v1
        "lesão cutânea benigna", "lesao cutanea benigna",  # NOVO -- pedido explícito do pesquisador
        "benigno", "benigna",                         # NOVO -- pedido explícito do pesquisador
    ),
    INDETERMINADO: (
        "lesão indeterminada", "lesao indeterminada",  # paridade com v1
    ),
}

# Variantes que são PALAVRA ISOLADA (precisam de fronteira de palavra para
# não casar "benignos"/"benignas" no plural -- decisão de design: no
# cenário real "conflitante", a frase "resultados benignos dos algoritmos
# ABCD e SPCL" (plural, adjetivo descrevendo OUTROS algoritmos, não um
# veredito) não deve disparar NEGATIVO; só a forma singular inequívoca
# "benigno"/"benigna" conta como rótulo, conforme pedido do pesquisador
# ("aceite... apenas quando forem conclusões inequívocas").
_WHOLE_WORD_VARIANTS = frozenset({"benigno", "benigna"})

# Ordem de prioridade usada só como tiebreak estável quando há múltiplos
# candidatos à MESMA posição (não deveria ocorrer na prática) -- não afeta
# a escolha por posição (ver `_find_last_valid_label_match`).
_STATUS_PRIORITY_ORDER = (POSITIVO, NEGATIVO, INDETERMINADO)


# ---------------------------------------------------------------------------
# 2. Negação -- heurística APROXIMADA (não é NLP real), flagueada
#    explicitamente como tal, no mesmo espírito das heurísticas lexicais
#    já existentes em `src/multiagent/vlm_direct.py`
#    (`_vocabulary_terms_not_in_inputs`,
#    `_check_not_evaluated_not_misrepresented_as_absent`, etc.): olha só
#    para uma palavra de negação ("não"/"nao") dentro de uma janela curta
#    de tokens IMEDIATAMENTE antes do rótulo casado. Não entende escopo
#    de negação real, não lida com negação de dupla camada, não entende
#    "nunca", "jamais", etc. -- só os dois tokens literais pedidos pelo
#    pesquisador ("não"/"nao").
# ---------------------------------------------------------------------------
_NEGATION_WORDS = frozenset({"não", "nao"})
_NEGATION_WINDOW_WORDS = 4
# Pontuação tratada como limite de cláusula/frase -- a janela de negação
# nunca atravessa um destes caracteres. Isto evita o falso-positivo óbvio
# de "não é benigna, é suspeita de melanoma" negar também "suspeita de
# melanoma" só porque "não" aparece 4 tokens atrás (que seria o caso sem
# este limite de cláusula) -- a negação de "benigna" não deveria "vazar"
# para a cláusula seguinte, separada por vírgula.
_CLAUSE_BOUNDARY_CHARS = (".", "!", "?", ";", ",", ":", "\n")


# ---------------------------------------------------------------------------
# 1b. Extensão POSITIVO 2026-11-14 -- "suspeita ... melanoma" com palavras
#     intensificadoras/descritivas entre "suspeita" e "de melanoma", desde
#     que a conclusão final expresse suspeita AFIRMATIVA E INEQUÍVOCA
#     (decisão verbatim do pesquisador, ver `PARSER_VERSION` acima).
#
#     Caso real de regressão (`unanime_positivo`, Estágio 1,
#     qwen3-vl:8b-instruct-q4_K_M): a região de conclusão final real é
#     "Há suspeita clara e consistente de melanoma maligno com base em
#     três algoritmos..." -- a frase literal "suspeita de melanoma" NÃO
#     aparece como substring (as palavras "clara e consistente" quebram
#     o match), mas o veredito é inequivocamente afirmativo. v1 acertou
#     "positivo" só porque o TÍTULO do documento ("Diagnóstico
#     Consolidado: Suspeita de Melanoma Maligno") contém a frase literal
#     -- v1 varre o corpo inteiro, v2 (corretamente, por design) só olha
#     a região de conclusão final, que não repete a frase literal.
#
#     Regra implementada (NÃO "contém as duas palavras em qualquer
#     lugar da cláusula" -- isso seria arriscado demais, pegaria listas
#     de diagnóstico diferencial): exige literalmente a palavra
#     "suspeita", seguida de 0 a 6 palavras de um vocabulário FECHADO de
#     intensificadores/conectivos (`_AFFIRMATIVE_INTENSIFIER_WORDS`),
#     seguida de "de melanoma". A ausência de pontuação entre as palavras
#     do vocabulário fechado (regex exige espaço, nunca vírgula/ponto)
#     já restringe o match à MESMA cláusula, no mesmo espírito de
#     `_CLAUSE_BOUNDARY_CHARS` usado pelo resto do módulo -- qualquer
#     vírgula/ponto entre "suspeita" e "de melanoma" quebra o match
#     (ex.: "suspeita, segundo o laudo, de melanoma" NÃO casa).
# ---------------------------------------------------------------------------
_AFFIRMATIVE_INTENSIFIER_WORDS = (
    "clara", "claro", "consistente", "forte", "elevada", "alta",
    "significativa", "importante", "evidente", "real", "concreta",
    "moderada", "considerável", "considerada", "e",
)
_AFFIRMATIVE_SUSPEITA_MELANOMA_PATTERN = re.compile(
    r"suspeita(?:\s+(?:" + "|".join(_AFFIRMATIVE_INTENSIFIER_WORDS) + r")){0,6}\s+de\s+melanoma"
)

# "sem suspeita de melanoma" -- REJEITAR. Decisão explícita do
# pesquisador: "sem" imediatamente antes de "suspeita" bloqueia o match.
# Isto é uma checagem DISTINTA de `_is_negated`: `_is_negated` procura
# "não"/"nao" nos tokens imediatamente antes do RÓTULO casado (que, para
# o padrão "suspeita de melanoma", seria o próprio "suspeita" -- a
# mesma posição). O padrão real pedido usa a palavra "sem", não
# "não"/"nao" -- um padrão lexical diferente, por isso não é capturado
# por `_is_negated` e precisa de uma checagem própria, documentada aqui
# em vez de generalizar `_NEGATION_WORDS` (generalizar arriscaria mudar
# o comportamento de negação para TODOS os rótulos -- negativo,
# indeterminado -- o que não foi pedido).
_SEM_PREFIX_WORD = "sem"
_SEM_PREFIX_WINDOW_WORDS = 2

# "pode haver suspeita de melanoma, mas não é conclusivo" -- julgamento
# do pesquisador (documentado explicitamente como tal, não uma regra
# linguística geral): quando a MESMA SENTENÇA (limite mais largo que
# cláusula -- só pontuação final "." "!" "?"/quebra de linha, para
# incluir a cláusula seguinte separada por vírgula) contém um marcador
# textual de "isto não é conclusivo", a suspeita deixa de ser tratada
# como afirmação inequívoca, mesmo que a frase "suspeita de melanoma"
# apareça literalmente. Aplicado a QUALQUER match POSITIVO baseado em
# "suspeita" (tanto a variante literal antiga quanto a extensão nova),
# para que "pode haver suspeita de melanoma, mas não é conclusivo" não
# vire positivo só porque a frase literal antiga ainda bate.
_HEDGE_SENTENCE_BOUNDARY_CHARS = (".", "!", "?", "\n")
_HEDGE_MARKERS = (
    "mas não é conclusivo", "mas nao e conclusivo",
    "mas não é conclusiva", "mas nao e conclusiva",
    "não é conclusivo", "nao e conclusivo",
    "não é conclusiva", "nao e conclusiva",
    "inconclusivo", "inconclusiva",
)


def _is_sem_prefixed(text_lower: str, match_start: int) -> bool:
    """Ver comentário acima (`_SEM_PREFIX_WORD`). Restringe a busca por
    "sem" à cláusula atual (mesmo limite de `_CLAUSE_BOUNDARY_CHARS` já
    usado por `_is_negated`) e, dentro dela, às últimas
    `_SEM_PREFIX_WINDOW_WORDS` palavras imediatamente antes do match."""
    preceding = text_lower[:match_start]
    boundary_idx = max((preceding.rfind(ch) for ch in _CLAUSE_BOUNDARY_CHARS), default=-1)
    clause = preceding[boundary_idx + 1:]
    words = re.findall(r"[\wÀ-ÿ]+", clause)
    window = words[-_SEM_PREFIX_WINDOW_WORDS:]
    return _SEM_PREFIX_WORD in window


def _is_hedged_uncertain(text_lower: str, match_start: int, match_end: int) -> bool:
    """Ver comentário acima (`_HEDGE_MARKERS`). Examina a SENTENÇA que
    contém o match (não só a cláusula) porque o marcador de incerteza
    ("mas não é conclusivo") tipicamente aparece numa cláusula seguinte,
    separada por vírgula, não imediatamente antes do rótulo."""
    start_candidates = [text_lower.rfind(ch, 0, match_start) for ch in _HEDGE_SENTENCE_BOUNDARY_CHARS]
    sentence_start = max(start_candidates, default=-1)
    end_candidates = [c for c in (text_lower.find(ch, match_end) for ch in _HEDGE_SENTENCE_BOUNDARY_CHARS) if c != -1]
    sentence_end = min(end_candidates) if end_candidates else len(text_lower)
    sentence = text_lower[sentence_start + 1:sentence_end]
    return any(marker in sentence for marker in _HEDGE_MARKERS)


def _is_negated(text_lower: str, match_start: int) -> bool:
    """HEURÍSTICA APROXIMADA (decisão fechada do pesquisador: "implemente
    um simples... e flagueie como heurística aproximada, não solução de
    NLP completa"). Restringe a busca por "não"/"nao" à cláusula atual
    (texto depois da ÚLTIMA pontuação de limite de cláusula antes do
    match -- ver `_CLAUSE_BOUNDARY_CHARS`) e, dentro dela, olha só as
    últimas `_NEGATION_WINDOW_WORDS` palavras imediatamente antes do
    match. Não entende negação de escopo real (ex.: "não é nem benigna
    nem suspeita", negação distribuída sobre duas cláusulas, viria a
    classificar só a primeira como negada) -- documentado como limitação
    conhecida, não uma solução de NLP completa."""
    preceding = text_lower[:match_start]
    boundary_idx = max((preceding.rfind(ch) for ch in _CLAUSE_BOUNDARY_CHARS), default=-1)
    clause = preceding[boundary_idx + 1:]
    words = re.findall(r"[\wÀ-ÿ]+", clause)
    window = words[-_NEGATION_WINDOW_WORDS:]
    return any(w in _NEGATION_WORDS for w in window)


@dataclass
class LabelMatch:
    status: str
    variant: str
    start: int
    negated: bool


def _find_all_label_matches(text_lower: str) -> list[LabelMatch]:
    matches: list[LabelMatch] = []
    for status, variants in LABEL_VARIANTS_V2.items():
        for variant in variants:
            if variant in _WHOLE_WORD_VARIANTS:
                pattern = r"\b" + re.escape(variant) + r"\b"
                for m in re.finditer(pattern, text_lower):
                    matches.append(LabelMatch(status=status, variant=variant, start=m.start(),
                                               negated=_is_negated(text_lower, m.start())))
            else:
                start = 0
                while True:
                    idx = text_lower.find(variant, start)
                    if idx == -1:
                        break
                    end = idx + len(variant)
                    negated = _is_negated(text_lower, idx)
                    if status == POSITIVO:
                        # Extensão 2026-11-14: a variante literal antiga
                        # "suspeita de melanoma" também passa pelas novas
                        # checagens de "sem"-prefixo/incerteza -- ver
                        # comentário em `_is_hedged_uncertain` (sem isto,
                        # "sem suspeita de melanoma" e "pode haver
                        # suspeita de melanoma, mas não é conclusivo"
                        # continuariam batendo por esta variante literal,
                        # mesmo com a extensão nova corretamente
                        # rejeitando o padrão regex equivalente).
                        negated = negated or _is_sem_prefixed(text_lower, idx) or _is_hedged_uncertain(text_lower, idx, end)
                    matches.append(LabelMatch(status=status, variant=variant, start=idx, negated=negated))
                    start = end

    # Extensão POSITIVO 2026-11-14 -- "suspeita <intensificadores> de
    # melanoma" (ver `_AFFIRMATIVE_SUSPEITA_MELANOMA_PATTERN`). Caso real
    # de regressão: `unanime_positivo` (ver docstring da seção 1b acima).
    for m in _AFFIRMATIVE_SUSPEITA_MELANOMA_PATTERN.finditer(text_lower):
        negated = (
            _is_negated(text_lower, m.start())
            or _is_sem_prefixed(text_lower, m.start())
            or _is_hedged_uncertain(text_lower, m.start(), m.end())
        )
        matches.append(LabelMatch(status=POSITIVO, variant=m.group(0), start=m.start(), negated=negated))
    return matches


def _find_last_valid_label_match(text_lower: str) -> Optional[LabelMatch]:
    """Entre todos os casamentos de rótulo NÃO negados no texto dado,
    devolve o que está mais perto do FIM do texto (maior `start`) --
    aproxima "o veredito efetivamente declarado" quando a mesma região
    de conclusão final menciona mais de uma categoria (ex.: "não é
    benigna, é suspeita de melanoma": "benigna" é descartado pela
    negação, "suspeita de melanoma" (mais ao final) é o match válido
    remanescente e é devolvido)."""
    candidates = [m for m in _find_all_label_matches(text_lower) if not m.negated]
    if not candidates:
        return None
    candidates.sort(key=lambda m: m.start)
    return candidates[-1]


# ---------------------------------------------------------------------------
# 3. Extração da "conclusão final" -- HEURÍSTICA APROXIMADA E DOCUMENTADA
#    (decisão fechada do pesquisador: "design a documented, reasonable
#    heuristic... flag this heuristic explicitly as approximate/needing
#    confirmation, same pattern as the other lexical heuristics already
#    in this codebase").
#
#    Passos:
#      1. Separa o texto em parágrafos (por linha em branco; se não houver
#         nenhuma linha em branco, cai para separar por linha simples).
#      2. Procura, em QUALQUER parágrafo, um dos marcadores de conclusão
#         abaixo (case-insensitive). Mantém só os marcadores que caem na
#         METADE FINAL do documento (por índice de parágrafo) -- isso
#         evita ancorar num cabeçalho/título que usa a mesma palavra no
#         início do texto (ex.: cenário real "conflitante" começa com
#         "Diagnóstico Consolidado: ..." como TÍTULO, não como veredito).
#      3. Se houver ao menos um marcador qualificado, a REGIÃO DE
#         CONCLUSÃO FINAL é definida como todos os parágrafos do PRIMEIRO
#         marcador qualificado (o de menor índice entre os da metade
#         final) até o fim do documento -- inclui, deliberadamente,
#         parágrafos subsequentes como "Recomendação Prática"/"Resumo
#         Final" que também fazem parte do encerramento do texto.
#      4. Se NENHUM marcador for encontrado em lugar nenhum do documento,
#         cai para os marcadores de linha que começam com "✅"; se também
#         não houver, cai para o ÚLTIMO parágrafo não vazio.
#      5. Dentro da região de conclusão final, os parágrafos são
#         escaneados EM ORDEM REVERSA (do fim para o início) e o primeiro
#         rótulo válido (não negado) encontrado é o veredito -- "o match
#         mais perto do fim da região vence", o que captura o veredito
#         real mesmo quando parágrafos finais de encerramento (ex.: nota
#         de rodapé, aviso legal) não repetem o rótulo.
#
#    LIMITAÇÃO CONHECIDA (igual ao espírito das heurísticas lexicais já
#    existentes neste módulo/`vlm_direct.py`): é uma heurística posicional
#    e lexical, não uma compreensão real do texto. Pode errar em
#    estruturas de resposta muito diferentes das observadas neste piloto
#    (ex.: um modelo que escreva o veredito real ANTES de uma longa
#    análise posterior, sem repeti-lo ao final). Ainda sem confirmação
#    explícita do pesquisador sobre cobrir todos os estilos de resposta
#    possíveis -- reportar isso é parte do pedido ("flag this heuristic
#    explicitly as approximate/needing confirmation").
# ---------------------------------------------------------------------------
_CONCLUSION_MARKERS: tuple[str, ...] = (
    "conclusão final", "conclusao final",
    "diagnóstico consolidado", "diagnostico consolidado",
    "avaliação final", "avaliacao final",
    "conclusão integrada", "conclusao integrada",
    "veredito final",
    "resumo final",
)


def _split_paragraphs(text: str) -> list[str]:
    if "\n\n" in text:
        paragraphs = [p.strip() for p in text.split("\n\n")]
    else:
        paragraphs = [p.strip() for p in text.split("\n")]
    return [p for p in paragraphs if p]


def _paragraph_has_marker(paragraph_lower: str) -> bool:
    return any(marker in paragraph_lower for marker in _CONCLUSION_MARKERS)


def _paragraph_starts_with_check_emoji(paragraph: str) -> bool:
    stripped = paragraph.strip().lstrip("*_> ").strip()
    return stripped.startswith("✅")


@dataclass
class FinalConclusionExtraction:
    region_text: str
    region_start_paragraph_index: Optional[int]
    region_end_paragraph_index: Optional[int]
    extraction_fallback_used: str  # "marker_in_final_half" / "check_emoji_line" / "last_paragraph" / "empty_text"


def _extract_final_conclusion_region(text: str) -> FinalConclusionExtraction:
    paragraphs = _split_paragraphs(text)
    n = len(paragraphs)
    if n == 0:
        return FinalConclusionExtraction(region_text="", region_start_paragraph_index=None,
                                          region_end_paragraph_index=None, extraction_fallback_used="empty_text")

    lower_paragraphs = [p.lower() for p in paragraphs]
    final_half_start_index = n - (n + 1) // 2  # primeiro índice da metade final (arredondando p/ cima)

    marker_indices_in_final_half = [
        i for i in range(final_half_start_index, n) if _paragraph_has_marker(lower_paragraphs[i])
    ]
    if marker_indices_in_final_half:
        start_idx = min(marker_indices_in_final_half)
        region = paragraphs[start_idx:]
        return FinalConclusionExtraction(
            region_text="\n\n".join(region), region_start_paragraph_index=start_idx,
            region_end_paragraph_index=n - 1, extraction_fallback_used="marker_in_final_half",
        )

    check_emoji_indices = [i for i in range(n) if _paragraph_starts_with_check_emoji(paragraphs[i])]
    if check_emoji_indices:
        start_idx = check_emoji_indices[-1]
        region = paragraphs[start_idx:]
        return FinalConclusionExtraction(
            region_text="\n\n".join(region), region_start_paragraph_index=start_idx,
            region_end_paragraph_index=n - 1, extraction_fallback_used="check_emoji_line",
        )

    return FinalConclusionExtraction(
        region_text=paragraphs[-1], region_start_paragraph_index=n - 1,
        region_end_paragraph_index=n - 1, extraction_fallback_used="last_paragraph",
    )


# ---------------------------------------------------------------------------
# 4. API pública
# ---------------------------------------------------------------------------
@dataclass
class CaseStatusV2Result:
    status: str
    parser_version: str
    extraction: Optional[FinalConclusionExtraction]
    matched_variant: Optional[str]
    matched_category: Optional[str]


def classify_case_status_v2_detailed(event: dict[str, Any] | None) -> CaseStatusV2Result:
    """Versão detalhada -- devolve também a região de conclusão final
    extraída e o rótulo casado, para transparência/auditoria no relatório
    de reclassificação (`scripts/run_reclassify_pilot_with_v2_parser.py`).
    """
    if not event:
        return CaseStatusV2Result(status=AUSENTE, parser_version=PARSER_VERSION, extraction=None,
                                   matched_variant=None, matched_category=None)

    validation = event.get("validation")
    if validation is None:
        return CaseStatusV2Result(status=AUSENTE, parser_version=PARSER_VERSION, extraction=None,
                                   matched_variant=None, matched_category=None)

    if isinstance(validation, dict):
        diagnostico = validation.get("diagnostico")
        if not diagnostico:
            return CaseStatusV2Result(status=NAO_RECONHECIDO, parser_version=PARSER_VERSION, extraction=None,
                                       matched_variant=None, matched_category=None)
        text = str(diagnostico)
    else:
        text = str(validation)

    if not text.strip():
        return CaseStatusV2Result(status=NAO_RECONHECIDO, parser_version=PARSER_VERSION, extraction=None,
                                   matched_variant=None, matched_category=None)

    extraction = _extract_final_conclusion_region(text)
    match = _find_last_valid_label_match(extraction.region_text.lower())
    if match is None:
        return CaseStatusV2Result(status=NAO_RECONHECIDO, parser_version=PARSER_VERSION, extraction=extraction,
                                   matched_variant=None, matched_category=None)

    return CaseStatusV2Result(status=match.status, parser_version=PARSER_VERSION, extraction=extraction,
                               matched_variant=match.variant, matched_category=match.status)


def classify_case_status_v2(event: dict[str, Any] | None) -> str:
    """API curta, mesma assinatura/contrato de retorno de
    `classify_case_status` (um dos cinco status de
    `src.evaluation.case_status.ALL_STATUSES`) -- para reuso idêntico em
    VLM-direto e sumarizador, sem casos especiais por papel."""
    return classify_case_status_v2_detailed(event).status
