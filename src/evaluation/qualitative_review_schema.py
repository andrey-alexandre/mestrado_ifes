"""Esquema da revisão qualitativa das sínteses multiagente (Gemini/GPT/
Qwen3-VL/MedGemma, condição S2, as 300 lesões já congeladas para os
modelos locais -- ver ``experiments/multiagent/local_300_s2_subset``).

Esta é uma ANÁLISE dos resultados, não um critério de aprovação de
modelo nem autorização para excluir casos (decisão do pesquisador,
preservada aqui só como documentação -- nenhum código deste módulo
decide aprovação/exclusão).

Vocabulário fechado dos 10 problemas revisáveis + 1 marcador separado
(observação/hipótese própria claramente identificada -- NUNCA tratado
automaticamente como problema). Status possíveis para cada
problema/marcador: ``PRESENTE``/``AUSENTE``/``NAO_AVALIAVEL``, ou vazio
(``""``) quando a revisão humana ainda está pendente -- vazio NUNCA é
interpretado como ``AUSENTE`` em nenhum lugar deste módulo ou dos
scripts que o consomem.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

PRESENTE = "presente"
AUSENTE = "ausente"
NAO_AVALIAVEL = "nao_avaliavel"
PENDENTE = ""  # vazio -- revisão humana ainda não preenchida

ALL_STATUS_VALUES = (PRESENTE, AUSENTE, NAO_AVALIAVEL, PENDENTE)


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    label: str
    description: str
    is_problem: bool  # False só para o marcador de observação/hipótese própria


CRITERIA: tuple[Criterion, ...] = (
    Criterion("atribuicao_incorreta", "Atribuição incorreta",
              "Atribui a um agente achado ou conclusão que ele não declarou.", True),
    Criterion("parecer_inventado", "Parecer inventado",
              "Apresenta como existente um parecer que não foi recebido.", True),
    Criterion("distorcao_inversao_criterio", "Distorção ou inversão de critério",
              "Altera um valor, presença/ausência, pontuação ou conteúdo declarado.", True),
    Criterion("confusao_avaliacao", "Confusão de avaliação",
              "Trata \"avaliado e ausente\" como \"não avaliado\", ou o inverso.", True),
    Criterion("confusao_ausencia", "Confusão de ausência",
              "Transforma ausência de parecer em ausência de característica da lesão.", True),
    Criterion("observacao_propria_sem_origem_clara", "Observação própria sem origem clara",
              "Não permite distinguir análise própria da imagem de informação atribuída aos agentes.", True),
    Criterion("hipotese_como_confirmacao", "Hipótese apresentada como confirmação",
              "Converte uma possibilidade explicitamente considerada em achado afirmado, sem fundamentação declarada.", True),
    Criterion("contradicao_interna", "Contradição interna",
              "Conclusão e justificativa são incompatíveis, sem explicação para a divergência.", True),
    Criterion("falso_consenso_divergencia_omitida", "Falso consenso ou divergência omitida",
              "Apresenta concordância inexistente ou apaga divergência relevante.", True),
    Criterion("decisao_sem_fundamentacao_rastreavel", "Decisão sem fundamentação rastreável",
              "Não permite identificar o suporte declarado para o veredito.", True),
    # Marcador separado -- NUNCA um problema por si só (protocolo, seção
    # 2: "Registre separadamente a ocorrência de observação/hipótese
    # própria claramente identificada, sem tratá-la automaticamente como
    # problema.").
    Criterion("observacao_hipotese_propria_identificada", "Observação/hipótese própria claramente identificada",
              "O sumarizador identifica explicitamente uma observação ou hipótese própria (não atribuída aos agentes) -- registrado, não é um problema por si.", False),
)

PROBLEM_CRITERIA = tuple(c for c in CRITERIA if c.is_problem)
CRITERION_BY_ID = {c.criterion_id: c for c in CRITERIA}

REVIEW_CSV_FIELDNAMES = (
    "lesion_id", "model_label_neutral", "criterion_id", "criterion_label",
    "status", "evidence_excerpt", "source_reference", "reviewer_comment",
    "reviewer_id", "review_date",
)

MODEL_REAL_LABELS = ("Gemini", "GPT", "Qwen3-VL", "MedGemma")
NEUTRAL_MODEL_ID_MAPPING_SEED = 20261120  # distinta de todas as outras semeaduras do projeto


def build_neutral_model_id_mapping(seed: int = NEUTRAL_MODEL_ID_MAPPING_SEED) -> dict[str, str]:
    """Permutação ÚNICA e GLOBAL (não por lesão -- decisão de simplicidade
    documentada) de rótulo real -> identificador neutro ("Modelo A/B/C/D"),
    determinística pela semente. A chave de correspondência (este dicionário)
    deve ser salva separadamente do material de revisão, nunca junto com
    ele -- ver ``scripts/prepare_qualitative_review_materials.py``.

    NÃO é cegamento garantido: um revisor que reconheça o estilo de
    escrita de um modelo específico pode inferir a identidade real; isto
    é documentado explicitamente, nunca apresentado como blindagem
    rigorosa."""
    neutral_ids = ["Modelo A", "Modelo B", "Modelo C", "Modelo D"]
    rng = random.Random(seed)
    shuffled_reals = list(MODEL_REAL_LABELS)
    rng.shuffle(shuffled_reals)
    return dict(zip(shuffled_reals, neutral_ids))


def build_pending_review_rows(lesion_id: str, model_label_neutral: str, text_available: bool) -> list[dict[str, str]]:
    """Linhas-modelo (uma por critério/marcador) para ``lesion_id``/
    ``model_label_neutral``, no formato de ``REVIEW_CSV_FIELDNAMES``.

    ``text_available``: quando ``False`` (síntese literalmente ausente --
    falha/recusa/ausência de execução/máscara vazia SEM texto bruto
    analisável), cada linha recebe ``status=NAO_AVALIAVEL`` com
    ``reviewer_comment="auto: sintese indisponivel para analise"`` -- um
    default ESTRUTURAL (não há texto para julgar nenhum critério), nunca
    uma presunção sobre qual problema existiria se houvesse texto. Quando
    ``True`` (inclusive quando a chamada falhou/foi recusada mas ainda
    produziu texto analisável -- protocolo, seção 1), toda linha fica
    ``status=PENDENTE`` (vazio), aguardando julgamento humano -- nenhum
    valor é pré-escolhido."""
    rows = []
    for criterion in CRITERIA:
        rows.append({
            "lesion_id": lesion_id, "model_label_neutral": model_label_neutral,
            "criterion_id": criterion.criterion_id, "criterion_label": criterion.label,
            "status": NAO_AVALIAVEL if not text_available else PENDENTE,
            "evidence_excerpt": "", "source_reference": "",
            "reviewer_comment": "auto: sintese indisponivel para analise" if not text_available else "",
            "reviewer_id": "", "review_date": "",
        })
    return rows
