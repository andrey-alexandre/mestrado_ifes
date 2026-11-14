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

REAL_RAW_TEXT_UNANIME_POSITIVO = """**Diagnóstico Consolidado: Suspeita de Melanoma Maligno**

---

### 📊 **Integração dos Três Pareceres Técnicos**

Os três algoritmos — ABCD, Menzies e SPCL — convergiram de forma consistente para um diagnóstico de **suspeita de melanoma maligno**, com múltiplas evidências de alto risco que se repetem em todos os critérios analisados.

---

### 🔍 **Análise Comparativa**

#### 1️⃣ **ABCD – Escore > 5.45 → Lesão Maligna**
- **Assimetria 2** (irregular em ambos os eixos) → indica ausência de simetria, padrão de alto risco.
- **Bordas 7** (nítidas em quase todos os quadrantes) — embora nítidas, a regularidade não é um fator de baixo risco, e a combinação com outras características eleva o risco.
- **Cor 5** → presença de **cinco cores distintas** (branco/bege, vermelho/rosa, marrom claro, marrom escuro, preto) → **padrão de multi-coloração extremamente significativo**.
- **Estruturas 4** → rede pigmentada, linhas ramificadas, pontos e glóbulos → sinais clássicos de atipia morfológica.
- **Conclusão ABCD**: Lesão maligna pelo critério ABCD.

#### 2️⃣ **Menzies – Múltiplas Características Positivas**
- **Véu azul-branco (1)** → presença de véu iridescência, típico de melanoma.
- **Múltiplos pontos marrons (1)** → indica atipia.
- **Pseudopodes (1)** → estruturas pontiagudas que se projetam, sinal de malignidade.
- **Pontos pretos periféricos (1)** → bordas com pontos escuros, indicativo de regressão ou malignidade.
- **Múltiplas cores (1)** → novamente, confirma a coloração atípica.
- **Ausência de negativas** → não há simetria nem cor única — **fator decisivo para diagnóstico**.
- **Conclusão Menzies**: Melanoma pelo Método Menzies.

#### 3️⃣ **SPCL – Pontuação 7 → Suspeita Alta**
- **Padrão de pigmentação atípico (Sim)** → confirma o padrão irregular.
- **Véu azul-branco irregular (Sim)** → confirma a presença de véu com padrão anatômico atípico.
- **Padrão de retículo irregular (Sim)** → rede pigmentada mal definida.
- **Padrão de glóbulos irregulares (Sim)** → pontos pigmentados com forma atípica.
- **Hiperpigmentação localizada (Sim)** → presença de áreas hiperpigmentadas, indicando crescimento ou inflamação.
- **Conclusão SPCL**: Suspeita elevada — **pontuação 7, acima do limiar de 3**.

---

### 🔬 **Concordância entre os Algoritmos**

- Todos os três algoritmos identificaram a presença de **multi-coloração**, **bordas irregulares**, **estruturas atípicas** e **padrões pigmentados anormais**.
- O **Menzies** é o mais específico em detectar critérios morfológicos complexos, como véu azul-branco, pseudopodes e pontos pretos periféricos.
- O **SPCL** reforça a presença de padrões morfológicos atípicos em múltiplos parâmetros.
- O **ABCD**, embora mais genérico, teve a pontuação mais alta (>5.45), validando o diagnóstico com base em critérios clássicos.

---

### ⚠️ **Evidências de Alto Risco Reincidentes**

- **Multi-coloração** (ABCD, Menzies, SPCL) → **indício decisivo de malignidade.**
- **Bordas irregulares / assimetria** (ABCD, Menzies) → **padrão de malignidade clássico.**
- **Estruturas atípicas (rede, glóbulos, pseudopodes)** (ABCD, Menzies, SPCL) → **sinais de crescimento dissemínado.**

---

### ✅ **Conclusão Final**

> **Há suspeita clara e consistente de melanoma maligno com base em três algoritmos independentes e complementares — ABCD, Menzies e SPCL — todos concordando em identificar múltiplas características de alto risco.**

---

### 📌 **Recomendação Clínica**

Não é possível fornecer diagnóstico clínico. Porém, **é imperativo que a lesão seja avaliada por um dermatologista especialista em melanoma, com exame clínico e biópsia se necessário**. A lesão apresenta padrões morfológicos consistentes com malignidade, e deve ser considerada com **alta prioridade de avaliação diagnóstica e tratamento**.

---

**Atenção: este parecer é baseado em análise de imagem segmentada e critérios diagnósticos. Não substitui avaliação clínica, exame de biópsia ou consulta com especialista.**"""

# Texto bruto REAL da SEGUNDA execução real do Estágio 1 (gate v2 oficial,
# ver docs/reconstruction/20_local_vlm_pilot.md §9d), mesmo cenário de
# controle `unanime_positivo`, modelo qwen3-vl:8b-instruct-q4_K_M --
# inferência local via Ollama não é determinística, então o modelo deu
# uma resposta DIFERENTE da primeira execução (REAL_RAW_TEXT_UNANIME_POSITIVO
# acima). Copiado verbatim de
# experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/report.md
# (campo "justificativa" do cenário "unanime_positivo").
REAL_RAW_TEXT_UNANIME_POSITIVO_SEGUNDA_EXECUCAO = """**Diagnóstico Consolidado – Lesão Cutânea Suspeita de Melanoma**

---

**1. Conclusão Integrada:**

Baseado na análise dos três algoritmos (ABCD, Menzies e SPCL), **a lesão apresenta múltiplas características clínicas de alto risco, consistentes entre os critérios, sugerindo uma lesão maligna com alta probabilidade de ser melanoma.**

Os três pareceres convergem em sua conclusão crítica:
✅ **ABCD:** Escore > 5,45 — indica lesão maligna.
✅ **Menzies:** Múltiplas características positivas (veu azul-branco, pontos marrons, pseudopodes, pontos pretos, múltiplas cores) sem negativas — confirma diagnóstico de melanoma.
✅ **SPCL:** Pontuação 7 (acima de 3) — indica suspeita significativa de melanoma.

---

**2. Análise de Concorrência e Conflito:**

Não há conflito significativo entre os algoritmos.
Todos os métodos identificaram:
- **Assimetria e bordas irregulares** (ABCD; Menzies com pseudopodes e bordas não simétricas; SPCL com padrão de retículo irregular),
- **Múltiplas cores** (ABCD: 5 cores; Menzies: múltiplas cores; SPCL: hiperpigmentação localizada),
- **Estruturas atípicas** (ABCD: rede pigmentada, glóbulos; Menzies: pseudopodes; SPCL: padrão de globulos e retículo irregulares).

A **concordância entre os algoritmos é quase perfeita** em relação aos padrões mais importantes do melanoma.

---

**3. Padrões de Alto Risco Identificados (Reforçados por Todos os Algoritmos):**

- **Assimetria extrema** (ABCD: 2/2; Menzies: simetria ausente; SPCL: padrão atípico).
- **Bordas irregulares** (ABCD: 7/7 bordas nítidas, mas com contorno irregular; Menzies: pseudopodes; SPCL: véu irregular).
- **Múltiplas cores** (ABCD: 5 cores; Menzies: múltiplas cores; SPCL: hiperpigmentação + cores implícitas).
- **Estruturas atípicas** (ABCD: rede, glóbulos; Menzies: pseudopodes; SPCL: retículo irregular + globulos).
- **Presença de pontos pretos periféricos** (Menzies; SPCL sugere regressão ausente, mas não contradiz a presença de pontos).
- **Pseudopodes e streaming radial ausente?** — Menzies não registrou streaming radial, mas registrou pseudopodes, o que é marcador importante.

---

**4. Considerações sobre a Imagem (Se disponível):**

*Nota: A imagem segmentada foi mencionada como fornecida, mas não está visível em meu ambiente atual. Contudo, considero que os três algoritmos foram consistentemente aplicados com base na lesão visualmente segmentada. Se a imagem revelasse características que contradizem os padrões identificados (ex: lesão completamente simétrica ou sem bordas irregulares), poderia haver reconsideração — mas, **não há indícios de tal discordância nos relatórios fornecidos**.*

---

**5. Recomendação Clínica (Não diagnóstica, apenas orientação):**

📌 **A lesão apresenta comportamento atípico consistentemente descrito por três critérios independentes e robustos.**

🔍 **Avaliação por um dermatologista com experiência em melanoma é urgente.**

🩺 **Recomenda-se biópsia de todo o lesão com margem adequada, preferencialmente por excisão completa, seguida de avaliação histopatológica com análise de bordas e microscopia.**

📌 **O uso de algoritmos como Menzies e SPCL, além do ABCD, aumenta a confiança diagnóstica, mesmo que o método ABCD seja mais amplamente utilizado.**

---

**Resumo Final:**

> 📌 **Lesão cutânea com comportamento atípico característico de melanoma.**
> 📌 **Todos os três algoritmos confirmam suspeita diagnóstica de melanoma.**
> 📌 **Avaliação clínica imediata por dermatologista é essencial.**
> 📌 **Biópsia com excisão completa é a próxima etapa terapêutica recomendada.**

---

*Este relatório é uma síntese da análise de critérios clínicos, não um diagnóstico. A confirmação histopatológica é indispensável.*"""

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
# 2b. Caso real "unanime_positivo" -- REGRESSÃO (Estágio 1,
#     qwen3-vl:8b-instruct-q4_K_M). A região de conclusão final real é
#     "Há suspeita clara e consistente de melanoma maligno com base em
#     três algoritmos..." -- a frase literal "suspeita de melanoma" NÃO
#     aparece ali (só no TÍTULO do documento, fora da região de
#     conclusão final). Antes da extensão de 2026-11-14, v2 devolvia
#     "nao_reconhecido" para este caso real -- ver
#     docs/reconstruction/20_local_vlm_pilot.md §9b.
# ---------------------------------------------------------------------------
def test_real_unanime_positivo_v1_unaffected():
    """v1 (intocado) continua acertando "positivo" por coincidência de
    substring no TÍTULO do documento ("Diagnóstico Consolidado: Suspeita
    de Melanoma Maligno") -- não alterado por este módulo."""
    assert classify_case_status({"validation": REAL_RAW_TEXT_UNANIME_POSITIVO}) == POSITIVO


def test_real_unanime_positivo_v2_now_classifies_positivo():
    """Teste de regressão do gap real encontrado na execução do piloto:
    a extensão "suspeita <intensificador> de melanoma" deve reconhecer
    "suspeita clara e consistente de melanoma" como afirmação inequívoca,
    usando a REGIÃO DE CONCLUSÃO FINAL (não o título, que fica fora da
    metade final do documento)."""
    result = classify_case_status_v2_detailed({"validation": REAL_RAW_TEXT_UNANIME_POSITIVO})
    assert result.status == POSITIVO
    assert result.parser_version == PARSER_VERSION
    assert "suspeita" in result.matched_variant
    assert "melanoma" in result.matched_variant
    # confirma que o match usado foi da extensão nova (palavras
    # intensificadoras entre "suspeita" e "de melanoma"), não da frase
    # literal antiga (que não aparece na região de conclusão final real).
    assert result.matched_variant != "suspeita de melanoma"
    assert "clara" in result.matched_variant and "consistente" in result.matched_variant


# ---------------------------------------------------------------------------
# 2b2. Caso real "unanime_positivo", SEGUNDA execução real (gate v2
#      oficial, ver docs/reconstruction/20_local_vlm_pilot.md §9d).
#      Inferência local via Ollama não é determinística -- o modelo deu
#      um texto bruto DIFERENTE desta vez, com "suspeita diagnóstica de
#      melanoma" no "Resumo Final". Este é um teste de regressão
#      DISTINTO de `test_real_unanime_positivo_v2_now_classifies_positivo`
#      (que usa REAL_RAW_TEXT_UNANIME_POSITIVO, a primeira execução,
#      "suspeita clara e consistente de melanoma") -- ambos são reais e
#      válidos, cobrindo duas paráfrases reais distintas que o modelo já
#      produziu para o mesmo cenário de controle.
# ---------------------------------------------------------------------------
def test_real_unanime_positivo_segunda_execucao_v1_still_positivo():
    """v1 (intocado) continua acertando "positivo" por coincidência de
    substring no TÍTULO desta segunda resposta real ("Diagnóstico
    Consolidado – Lesão Cutânea Suspeita de Melanoma")."""
    assert classify_case_status({"validation": REAL_RAW_TEXT_UNANIME_POSITIVO_SEGUNDA_EXECUCAO}) == POSITIVO


def test_real_unanime_positivo_segunda_execucao_v2_now_classifies_positivo():
    """Teste de regressão do gap real encontrado na SEGUNDA execução real
    do piloto (texto bruto diferente da primeira execução, mesmo
    cenário): a extensão do vocabulário fechado com "diagnóstica"/
    "clínica" (qualificadores, não intensificadores -- decisão verbatim
    do pesquisador) deve reconhecer "suspeita diagnóstica de melanoma"
    como afirmação inequívoca, usando a REGIÃO DE CONCLUSÃO FINAL (o
    "Resumo Final", não o título, que fica fora da metade final do
    documento)."""
    result = classify_case_status_v2_detailed({"validation": REAL_RAW_TEXT_UNANIME_POSITIVO_SEGUNDA_EXECUCAO})
    assert result.status == POSITIVO
    assert result.parser_version == PARSER_VERSION
    assert "suspeita" in result.matched_variant
    assert "melanoma" in result.matched_variant
    assert "diagnóstica" in result.matched_variant
    # confirma que o match usado foi dentro do "Resumo Final" (região de
    # conclusão final), não do título fora da metade final do documento.
    assert "resumo final" in result.extraction.region_text.lower()


# ---------------------------------------------------------------------------
# 2c. Extensão POSITIVO 2026-11-14 -- casos de rejeição explícitos pedidos
#     pelo pesquisador (sintéticos).
# ---------------------------------------------------------------------------
def test_sem_suspeita_de_melanoma_is_not_positivo():
    """"sem" imediatamente antes de "suspeita" bloqueia o match -- tanto
    o padrão novo quanto a frase literal antiga."""
    text = "### Conclusão Final\n\nSem suspeita de melanoma; achados compatíveis com lesão comum."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status != POSITIVO


def test_sem_suspeita_diagnostica_de_melanoma_is_not_positivo():
    """Pedido explícito do pesquisador: "sem suspeita diagnóstica de
    melanoma" NÃO deve ser positivo -- "sem" está imediatamente antes de
    "suspeita", independentemente da palavra qualificadora
    ("diagnóstica") que aparece DEPOIS, entre "suspeita" e "de melanoma".
    `_is_sem_prefixed` olha só as palavras ANTES do início do match
    ("suspeita"), então o qualificador no meio não deveria precisar de
    nenhuma mudança de código para continuar bloqueado -- este teste
    confirma isso na prática, com o vocabulário novo já em vigor."""
    text = "### Conclusão Final\n\nSem suspeita diagnóstica de melanoma; achados compatíveis com lesão comum."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status != POSITIVO


def test_suspeita_clinica_de_melanoma_is_positivo():
    """Variante explícita pedida pelo pesquisador: "suspeita clínica de
    melanoma" deve ser aceita como positivo -- "clínica" é um
    qualificador, não um intensificador, mas a suspeita é afirmativa e
    inequívoca na conclusão final (mesma regra já aprovada)."""
    text = "### Conclusão Final\n\nHá suspeita clínica de melanoma com base nos três algoritmos."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status == POSITIVO
    assert "clínica" in result.matched_variant


def test_melanoma_nao_pode_ser_descartado_is_not_positivo():
    """Frase hedge/incerta que nunca contém a palavra "suspeita" -- o
    padrão novo exige literalmente "suspeita", então esta frase não deve
    disparar positivo por nenhum caminho (nem o literal antigo, nem a
    extensão nova); confirma que NÃO foi adicionado um match genérico de
    "melanoma" solto."""
    text = "### Conclusão Final\n\nMelanoma não pode ser descartado com os dados disponíveis."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status != POSITIVO
    assert result.status == NAO_RECONHECIDO


def test_uncertain_suspeita_mas_nao_conclusivo_is_not_positivo():
    """JULGAMENTO DO PESQUISADOR (documentado, não regra geral): "pode
    haver suspeita de melanoma, mas não é conclusivo" tem a frase literal
    "suspeita de melanoma", mas a cláusula seguinte ("mas não é
    conclusivo") undercuts a afirmação -- decisão de projeto: NÃO tratar
    como positivo (poderia ter sido o contrário; ver `_is_hedged_uncertain`
    para a justificativa completa)."""
    text = "### Conclusão Final\n\nPode haver suspeita de melanoma, mas não é conclusivo com os dados atuais."
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status != POSITIVO


def test_differential_diagnosis_listing_known_residual_risk():
    """RISCO RESIDUAL CONHECIDO, documentado explicitamente (não
    totalmente evitável pela heurística lexical atual): quando uma
    LISTAGEM de diagnóstico diferencial menciona "suspeita de melanoma"
    dentro da mesma cláusula de uma lista de possibilidades (em vez de um
    veredito isolado), o parser ainda classifica como positivo, porque a
    frase literal bate dentro da região de conclusão final. Este teste
    FIXA o comportamento atual como risco conhecido -- não afirma que
    seja o comportamento desejável, e uma heurística mais sofisticada
    (ex.: detectar "diagnóstico diferencial inclui ..." como contexto de
    listagem) não foi implementada nesta entrega por não ter sido pedida
    e por ser arriscada de generalizar sem mais dados reais."""
    text = (
        "### Conclusão Final\n\n"
        "Diagnóstico diferencial inclui suspeita de melanoma, nevo atípico "
        "e ceratose seborreica, a confirmar com biópsia."
    )
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status == POSITIVO  # risco residual conhecido, não uma correção pedida


def test_differential_diagnosis_listing_with_qualifier_outside_final_region_is_not_positivo():
    """Pedido explícito do pesquisador (2026-11-14, segunda rodada):
    "uma menção apenas no diagnóstico diferencial [não deve ser]
    classificada como positiva" -- com a NOVA palavra qualificadora
    ("diagnóstica") que motivou esta entrega.

    Raciocínio sobre a heurística de REGIÃO (não um guarda novo, ver
    docstring de `_extract_final_conclusion_region`): no corpo real de
    uma resposta, uma listagem de diagnóstico diferencial tipicamente
    aparece numa seção de "Análise"/"Discussão", ANTES do marcador de
    conclusão final (ex. "Conclusão Final"/"Diagnóstico Consolidado"), e
    portanto fica FORA da região extraída -- a heurística de extração já
    protege este caso sem precisar de um guarda textual dedicado
    ("diagnóstico diferencial inclui ..."). Este teste confirma isso na
    prática: a listagem aparece num parágrafo anterior ao marcador de
    conclusão, e o parecer final não repete a suspeita.

    RISCO RESIDUAL AINDA NÃO CORRIGIDO (ver
    `test_differential_diagnosis_listing_known_residual_risk`, acima): SE
    um modelo escrever a listagem de diferencial DENTRO do próprio
    parágrafo de conclusão final (como naquele teste), o parser ainda
    classifica positivo -- a proteção aqui vem inteiramente da heurística
    de extração de região, não de uma análise de contexto de listagem.
    Isto NÃO está corrigido nesta entrega (não foi pedido um guarda
    textual; generalizar sem mais dados reais seria arriscado) e
    permanece documentado como risco conhecido, não coberto para todo
    estilo de resposta possível."""
    text = (
        "### Análise\n\n"
        "O diagnóstico diferencial inclui suspeita diagnóstica de melanoma, "
        "nevo atípico e ceratose seborreica, a confirmar com biópsia.\n\n"
        "### Conclusão Final\n\n"
        "A lesão apresenta padrão benigno, compatível com nevo comum."
    )
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status != POSITIVO
    assert result.status == NEGATIVO
    # confirma que a listagem de diferencial ficou FORA da região de
    # conclusão final extraída -- é essa exclusão, não uma análise de
    # contexto, que evita o falso-positivo aqui.
    assert "diagnóstico diferencial" not in result.extraction.region_text.lower()


# ---------------------------------------------------------------------------
# 2d. Conclusão conflitante sintética, análoga ao cenário real
#     "conflitante", mas usando a NOVA variante intensificada no veredito
#     final (confirma que a resolução "último veredito vence" continua
#     funcionando com a extensão nova).
# ---------------------------------------------------------------------------
def test_conflicting_body_with_new_intensified_positivo_final_verdict():
    text = (
        "Parecer ABCD: lesão benigna.\n\n"
        "Parecer Menzies: suspeita de melanoma.\n\n"
        "Parecer SPCL: lesão indeterminada.\n\n"
        "---\n\n"
        "### Conclusão Final\n\n"
        "Há suspeita forte e consistente de melanoma, segundo critério Menzies."
    )
    result = classify_case_status_v2_detailed({"validation": text})
    assert result.status == POSITIVO
    assert result.matched_variant != "suspeita de melanoma"


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
