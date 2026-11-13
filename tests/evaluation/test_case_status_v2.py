"""Testes do classificador v2 "conclusão final"
(`src.evaluation.case_status_v2`). Reusa, como fixtures, os textos
BRUTOS REAIS (literais, não paráfrases) coletados no piloto local de VLM
`qwen3-vl:8b-instruct-q4_K_M` (Estágio 1), salvos em
`experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/
call_log.jsonl` -- ver `docs/reconstruction/20_local_vlm_pilot.md` para o
contexto completo da decisão do pesquisador que motivou este módulo.

Convenções deste diretório reusadas: `tests/evaluation/test_case_status.py`
testa só o v1 (não alterado, não duplicado aqui).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.case_status import (  # noqa: E402
    AUSENTE,
    INDETERMINADO,
    NAO_RECONHECIDO,
    NEGATIVO,
    POSITIVO,
    classify_case_status,
)
from src.evaluation.case_status_v2 import (  # noqa: E402
    PARSER_VERSION,
    classify_case_status_v2,
    classify_case_status_v2_detailed,
)

# ---------------------------------------------------------------------------
# Fixtures: textos brutos REAIS do piloto (literais, copiados de
# call_log.jsonl -- scenario "um_parecer_ausente" e "conflitante", papel
# summary, model qwen3-vl:8b-instruct-q4_K_M).
# ---------------------------------------------------------------------------
REAL_RAW_TEXT_UM_PARECER_AUSENTE = """A análise dos três pareceres técnicos fornecidos — ABCD, Menzies e SPCL — indica uma **concordância unânime de benignidade** da lesão cutânea, embora o SPCL não tenha sido avaliado (não foi fornecido nenhum resultado). Abaixo, apresento a integração dos pareceres e a fundamentação do diagnóstico consolidado, conforme solicitado.

---

### 🔍 **Análise Integrada dos Pareceres:**

#### 1️⃣ **Algoritmo ABCD**
- **Resultados:**
  - Assimetria: 0 (benigna)
  - Bordas: 1 (indistintas — *leve preocupação, mas não alarmante*)
  - Cor: 1 (apenas marrom claro — benigno)
  - Estruturas: 1 (rede pigmentada regular — benigna)
- **Conclusão:** Soma ponderada < 4.75 → **Benigna.**
- **Observação:** O parâmetro “bordas indistintas” é um fator de risco moderado, mas não atinge o nível de suspeita neoplásica no critério ABCD, especialmente quando combinado com ausência de outras irregularidades.

#### 2️⃣ **Método Menzies**
- **Resultados:**
  - **Nenhuma característica positiva** (veu azul, pseudopodes, streaming radial, etc.) detectada.
  - **Características negativas presentes:** simetria e cor única.
- **Conclusão:** "Nenhuma característica positiva, ambas negativas presentes" → **Benigna.**
- **Observação:** O Menzies é um método robusto para identificar lesões malignas. A ausência de sinais típicos de melanoma (como pseudopodes ou streaming radial) e a presença de simetria e cor homogênea reforçam a benignidade da lesão.

#### 3️⃣ **SPCL (Seven Point Checklist)**
- **Informação fornecida:** *None*
- **Observação:** Como não há resultado do SPCL, não é possível integrar esse critério. No entanto, a ausência de qualquer resultado não implica risco; simplesmente indica que o método não foi aplicado ou não gerou dados.

---

### 🧩 **Concordância entre os Métodos**

| Critério              | ABCD        | Menzies     | SPCL         | Conclusão          |
|----------------------|-------------|-------------|--------------|--------------------|
| Benignidade          | Benigna     | Benigna     | Não aplicado | **Concordância total** |
| Padrões de risco     | Bordas indistintas (leve) | Sem sinais positivos | – | **Não indicam risco significativo** |
| Conclusão geral      | Benigna     | Benigna     | –            | **Benignidade confirmada** |

---

### 📊 **Fundamentação do Diagnóstico Consolidado:**

> **Diagnóstico Consolidado: Lesão Cutânea Benigna — Pode ser caracterizada como um nevo (nevus) com características típicas.**

- **Justificativa:**
  Os dois algoritmos com base em evidências clínicas (ABCD e Menzies) concordam que a lesão é benigna, com padrões que não atendem aos critérios de alerta para malignidade.
  A ausência de características atípicas em ambos (como multicromia, bordas irregulares, estruturas atípicas ou padrões de crescimento) reforça a benignidade.
  Embora o parâmetro “bordas indistintas” no ABCD possa levantar uma dúvida mínima, ele é considerado em escala de risco moderado e, nesse contexto, não é suficiente para alterar a conclusão de benignidade, especialmente com o apoio do Menzies, que é mais específico para melanomas.

- **Avaliação da Imagem (se houvesse):**
  Como não há imagem real (ou ela é um fundo cinza), não é possível verificar visualmente os detalhes. No entanto, os pareceres já são robustos o suficiente para sustentar o diagnóstico sem necessidade de reavaliação visual — especialmente considerando a concordância entre os dois métodos.

---

### ✅ **Conclusão Final:**

> **A lesão cutânea é considerada BENIGNA com base na análise dos algoritmos ABCD e Menzies. A ausência de resultado do SPCL não altera essa conclusão. Os padrões observados não sugerem risco de malignidade, e a lesão é compatível com um nevo com características típicas.**

---

### 📌 **Nota Importante (Reiterada):**

> **Esse sistema não fornece diagnósticos clínicos nem opiniões médicas.**
> A análise apresentada é meramente uma síntese técnica dos dados fornecidos pelos algoritmos.
> Para fins clínicos, a avaliação deve sempre ser realizada por um profissional de saúde qualificado, com o exame físico completo e, se necessário, biópsia ou acompanhamento por imagem de alta resolução.

---

✅ **Diagnóstico Consolidado: BENIGNO (Nevus com características típicas)**
🔍 *Avaliado por ABCD e Menzies; SPCL não avaliado.*"""

REAL_RAW_TEXT_CONFLITANTE = """**Diagnóstico Consolidado: Lesão de Baixo Risco com Suspeita Discordante de Alta Gravidade (Avaliação Integrada)**

---

### 🔍 Análise Integrada dos Três Pares de Pareceres

#### **1️⃣ Algoritmo ABCD – Resultado Benigno (Soma ponderada < 4.75)**
- **Assimetria:** Ausente — confere baixo risco.
- **Bordas:** Indistintas em parte — critério de alerta moderado, mas não suficiente para classificação de alto risco.
- **Cor:** Limitada a marrom claro — sugere ausência de pigmentação heterogênea.
- **Estruturas:** Rede pigmentada regular — padrão benigno.
- **Conclusão ABCD:** Benigno.

➡️ Este algoritmo indica segurança clínica baseada em critérios tradicionais de avaliação de neoplasias cutâneas.

---

#### **2️⃣ Algoritmo Menzies – Resultado de Alto Risco (Melanoma)**
- **Características positivas identificadas:**
  - Veu azul/branco (1)
  - Pontos marrons múltiplos (1)
  - Pseudopodes (1)
  - Streaming radial (1)
  - Pontos pretos periféricos (1)
  - Múltiplas cores (1)
- **Critérios negativos:**
  - Simetria ausente (0)
  - Cor única (0) — não é um fator negativo neste contexto, pois o critério é ativo em múltiplas cores.

➡️ O Menzies identifica **7 características positivas**, uma combinação que, segundo a literatura, tem alto valor preditivo para melanoma (especialmente em lesões com bordas irregulares e padrões de pigmentação complexos).

---

#### **3️⃣ Algoritmo SPCL – Resultado de Baixo Risco (Borderline)**
- **Critérios positivos:**
  - Rede pigmentada irregular (Sim) → pontuação 1.
- **Critérios negativos:**
  - Não há regressão, hiperpigmentação localizada, padrão de globulos irregulares.
- **OBS:** Vaso atípico não avaliado.

➡️ O SPCL atribui apenas **1 ponto**, abaixo do limiar de 3, considerando a lesão como *borderline*, ou seja, com risco baixo mas não descartado.

---

### ⚖️ Conflito entre os Algoritmos

| Algoritmo   | Classificação | Risco        | Fundamentação |
|-------------|---------------|--------------|---------------|
| ABCD        | Benigno       | Baixo        | Estruturas e cor regulares, sem critérios de alerta. |
| Menzies     | Melanoma      | Alto         | 7 características positivas de alto risco. |
| SPCL        | Borderline    | Baixo/Médio | 1 ponto de irregularidade, abaixo do limiar. |

→ **Conflito significativo entre Menzies e ABCD/SPCL.**

---

### 🧩 Análise da Imagem Segmentada (Inferência Visual Baseada nos Pareceres)

**Embora a imagem segmentada não contenha informações visuais (pois o campo está vazio), é crucial reforçar que:**

> **A imagem segmentada é o único elemento que poderia resolver o conflito.**

- Em uma situação real, a imagem destacaria:
  - **Bordas indistintas** (concordância com ABCD e SPCL — ambos consideram bordas irregulares como alerta, mas o Menzies não vê isso como suficiente para rejeitar a lesão).
  - **Pseudopodes e streaming radial** (característicos do Menzies).
  - **Multicolorismo ou veu azul/branco** (reforçando Menzies).

👉 *Sem a imagem real, a análise baseia-se exclusivamente nos dados dos algoritmos — e o Menzies é o único que identifica múltiplas características que, em conjunto, aumentam significativamente o risco.*

---

### 📊 Conclusão Integrada

✅ **A análise dos três algoritmos revela um conflito diagnóstico crítico.**

- **Menzies aponta alto risco com base em múltiplos critérios de alta especificidade.**
- **ABCD e SPCL, apesar de terem sido projetados para identificar lesões benignas, não são robustos quando frente a sinais de alto risco como pseudopodes e veu azul/branco.**

➡️ **Avaliação Final: O diagnóstico mais apropriado com base nos algoritmos fornecidos é:**
> **Suspeita de Melanoma segundo critério Menzies.**

⚠️ **Aviso: NÃO se trata de diagnóstico clínico. Esta análise é apenas uma inferência baseada em critérios técnicos. Uma avaliação clínica por dermatologista com exame macroscópico e, se necessário, biópsia, é indispensável.**

---

### ✅ Recomendação Prática

1. **Imagem visual real da lesão** deve ser submetida ao sistema para que o algoritmo de análise visual possa validar ou refutar os pareceres.
2. **Biopsia comum** (se indicada pela clínica) deve ser considerada mesmo que o Menzies sugira alto risco.
3. **Reavaliação com algoritmo combinado ou de maior precisão** (ex: Deep Learning com CNNs) pode ser útil para reconciliar os dados.

---

### 🎯 Resumo Final

> **Conclusão Integrada: A lesão apresenta múltiplos sinais de alto risco identificados pelo Método Menzies, em conflito com resultados benignos dos algoritmos ABCD e SPCL. Considerando a gravidade dos sinais identificados, recomenda-se **avaliação clínica de alto nível e possível biópsia**.**

**⚠️ Atente: Esta é apenas uma análise de dados analíticos — não é substituto de um diagnóstico clínico real.**"""


# ---------------------------------------------------------------------------
# 1. Caso real "um_parecer_ausente" -- o gap que motivou este módulo.
# ---------------------------------------------------------------------------
def test_real_um_parecer_ausente_v1_unaffected():
    """v1 (classify_case_status, intocado) deve continuar devolvendo
    exatamente o que foi registrado historicamente no piloto real --
    nao_reconhecido -- confirmando que este módulo NOVO não alterou o
    comportamento do v1."""
    assert classify_case_status({"validation": REAL_RAW_TEXT_UM_PARECER_AUSENTE}) == NAO_RECONHECIDO


def test_real_um_parecer_ausente_v2_classifies_negativo():
    """v2 deve reconhecer o veredito final real ("✅ Diagnóstico
    Consolidado: BENIGNO (Nevus com características típicas)") como
    negativo -- a variante "benigno" só é aceita porque aparece na REGIÃO
    DE CONCLUSÃO FINAL extraída, não porque a palavra aparece em algum
    lugar do corpo."""
    result = classify_case_status_v2_detailed({"validation": REAL_RAW_TEXT_UM_PARECER_AUSENTE})
    assert result.status == NEGATIVO
    assert result.parser_version == PARSER_VERSION
    assert result.matched_variant in ("benigno", "benigna")
    # a heurística NÃO decidiu só por causa da menção no meio do texto
    # ("Lesão Cutânea Benigna", mais perto do início) -- confirma que o
    # match usado foi o da região de conclusão final (perto do fim).
    assert "benigno" in result.extraction.region_text.lower()


# ---------------------------------------------------------------------------
# 2. Caso real "conflitante" -- conclusões conflitantes no corpo, veredito
#    final claro.
# ---------------------------------------------------------------------------
def test_real_conflitante_v1_already_positivo():
    """Documenta o comportamento v1 real já observado no piloto (positivo,
    por sorte de substring -- "suspeita de melanoma" aparece no veredito
    final). Não é alterado por este módulo."""
    assert classify_case_status({"validation": REAL_RAW_TEXT_CONFLITANTE}) == POSITIVO


def test_real_conflitante_v2_follows_final_verdict_not_early_mentions():
    """v2 deve seguir o ÚLTIMO veredito estabelecido ("Avaliação Final:
    ... Suspeita de Melanoma segundo critério Menzies"), não as menções
    conflitantes anteriores no corpo (ABCD=benigno, SPCL=borderline)."""
    result = classify_case_status_v2_detailed({"validation": REAL_RAW_TEXT_CONFLITANTE})
    assert result.status == POSITIVO
    assert result.matched_variant == "suspeita de melanoma"


# ---------------------------------------------------------------------------
# 3. Negação -- heurística aproximada, sintética.
# ---------------------------------------------------------------------------
def test_negation_synthetic_not_misclassified_as_negativo():
    text = "Análise dos critérios: não é benigna, é suspeita de melanoma."
    assert classify_case_status_v2({"validation": text}) == POSITIVO


def test_negation_synthetic_double_mention_negation_applies_to_nearest_label_only():
    """A heurística de negação olha só os 4 tokens IMEDIATAMENTE antes do
    rótulo casado -- documentado como aproximado, não escopo de frase
    real. Este teste fixa esse comportamento conhecido."""
    text = "Conclusão final: não é suspeita de melanoma, é lesão benigna."
    # "suspeita de melanoma" é negado (precedido por "não é"); "lesão
    # benigna" não é negado -> resultado esperado é negativo.
    assert classify_case_status_v2({"validation": text}) == NEGATIVO


# ---------------------------------------------------------------------------
# 4. Múltiplas categorias mencionadas no corpo, um veredito final claro
#    (sintético, variação do padrão real "conflitante").
# ---------------------------------------------------------------------------
def test_multiple_body_mentions_one_clear_final_verdict():
    text = (
        "Parecer ABCD: lesão benigna.\n\n"
        "Parecer Menzies: suspeita de melanoma.\n\n"
        "Parecer SPCL: lesão indeterminada.\n\n"
        "---\n\n"
        "### Conclusão Final\n\n"
        "Diagnóstico Consolidado: Lesão Benigna, conforme critério predominante."
    )
    assert classify_case_status_v2({"validation": text}) == NEGATIVO


# ---------------------------------------------------------------------------
# 5. indeterminado / ausente / recusa-adjacente nunca caem em
#    positivo/negativo.
# ---------------------------------------------------------------------------
def test_indeterminado_stays_indeterminado():
    text = "### Conclusão Final\n\nDiagnóstico Consolidado: Lesão Indeterminada -- não é possível decidir com segurança."
    assert classify_case_status_v2({"validation": text}) == INDETERMINADO


def test_ausente_when_validation_is_none():
    assert classify_case_status_v2({"validation": None}) == AUSENTE
    assert classify_case_status_v2(None) == AUSENTE
    assert classify_case_status_v2({}) == AUSENTE


def test_refusal_adjacent_text_without_any_category_label_is_nao_reconhecido():
    text = "Não consigo fornecer uma conclusão com segurança a partir desta imagem."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status == NAO_RECONHECIDO
    assert result.status not in (POSITIVO, NEGATIVO)


def test_empty_string_validation_is_nao_reconhecido():
    assert classify_case_status_v2({"validation": "   "}) == NAO_RECONHECIDO


# ---------------------------------------------------------------------------
# 6. Variantes negativas extras -- mapeamento explícito e versionado.
# ---------------------------------------------------------------------------
def test_lesao_cutanea_benigna_variant_in_final_conclusion_is_negativo():
    text = "### Conclusão Final\n\nDiagnóstico Consolidado: Lesão Cutânea Benigna, nevo típico."
    assert classify_case_status_v2({"validation": text}) == NEGATIVO


def test_plural_benignos_does_not_match_bare_label_variant():
    """Decisão de design: só as formas singulares exatas "benigno"/
    "benigna" contam como rótulo inequívoco -- "benignos"/"benignas" no
    plural costuma ser um adjetivo descrevendo outra coisa (ex.:
    "resultados benignos dos algoritmos ABCD e SPCL" no cenário real
    "conflitante"), não um veredito em si."""
    text = "### Conclusão Final\n\nOs resultados benignos dos outros critérios não mudam a avaliação geral."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status == NAO_RECONHECIDO
