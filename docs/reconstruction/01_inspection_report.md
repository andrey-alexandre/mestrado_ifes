# Relatório de Inspeção do Repositório — 2026-11-04

## Local do relatório e justificativa

Não existe, em nenhuma branch verificada (local ou `origin/*`), uma pasta de
documentação de "continuidade/reconstrução" com convenção de nomes já
estabelecida. O que existe de mais próximo é `reports/diagnostic_audit_2026-10-19/`
(presente na árvore de trabalho, **não versionado** nesta branch), mas é um
pacote autocontido de auditoria científica pontual (`01_executive_summary.md`
… `08_case_level_results.csv`, com scripts e tabelas próprios), não uma série
de documentos de reconstrução. Por isso este relatório foi criado em
`docs/reconstruction/01_inspection_report.md`, conforme a opção padrão definida
na tarefa.

Esta inspeção é **somente leitura**: nenhum arquivo existente foi alterado,
movido, renomeado ou excluído. Foi criado apenas o diretório
`docs/reconstruction/` (vazio antes desta tarefa) e este arquivo.

---

## ⚠️ Achado de segurança (ação recomendada, não executada)

Durante a varredura por segredos, foi encontrado o que parece ser uma
**credencial de API em texto puro** (padrão de chave da OpenAI) em:

- `LangChain.ipynb` (arquivo não rastreado pelo git na árvore de trabalho atual).
- **Histórico do git**, no arquivo `CAD.ipynb`, nos commits `1c80447` e
  `89f1fc2` — ambos alcançáveis a partir de refs em `origin/*`. Isso indica que
  a credencial provavelmente já foi enviada ao remoto (GitHub), mesmo que o
  arquivo tenha sido removido depois (commit `b1c1c41 Delete CAD.ipynb`
  remove o arquivo da árvore, **não do histórico**).

O conteúdo da credencial não é reproduzido neste relatório nem em nenhuma
outra saída. Recomenda-se:
1. Revogar essa chave OpenAI imediatamente, independentemente da visibilidade
   do repositório.
2. Emitir uma nova chave e carregá-la via variável de ambiente/segredo, nunca
   hardcoded em notebook.
3. Se o repositório for ou já tiver sido público, tratar a chave como
   definitivamente exposta; reescrever o histórico (`git filter-repo`/BFG) é
   boa prática de higiene, mas não desfaz uma exposição pública já ocorrida.

Nenhuma dessas ações foi executada — são decisões do pesquisador.

---

## 1. Estado do git

- **Branch atual:** `feature/inspection` (igual, commit a commit, a `origin/feature/inspection`; sem divergência).
- **HEAD:** `a557ba4538b0ffccb8e49db15aecd6b3a71e0d16` — mesmo commit de `main` (merge da PR #3 `feature/quali`).
- **Working tree:** sem alterações em arquivos rastreados. 14 entradas não rastreadas na raiz/subpastas (listadas abaixo, seção 2).
- **Últimos commits na branch atual** (`git log -10 --oneline`):
  ```
  a557ba4 Merge pull request #3 from andrey-alexandre/feature/quali
  3fe1e2d Add files via upload
  b1c1c41 Delete CAD.ipynb
  46cb424 Delete src directory
  996526a Merge pull request #1 from andrey-alexandre/feature/migracao_langgraph
  d5abfa0 Criação de pasta de arquivos do código original
  a23ab9f Add files via upload
  c35458b Ajustes para tornar o código funcional
  99bc4f6 Add files via upload
  80d6b09 Add files via upload
  ```
- **`git ls-files` nesta branch retorna exatamente 22 arquivos** — o repositório
  nesta branch é bem menor do que a árvore de trabalho sugere; a maior parte
  do que aparece em `find`/`ls` é não rastreado ou pertence a outras branches
  (ver seção 2).
- **Outras branches locais e remotas relevantes**, todas divergindo do mesmo
  ponto-base `a557ba4` (confirmado via `git merge-base`):
  | Branch | Commits além de `a557ba4` | Observação |
  |---|---|---|
  | `feature/segmentation` | **0** | Ponteiro idêntico a `main`/`a557ba4`. O nome sugere trabalho de segmentação, mas **não há nenhum commit de segmentação nela** — é uma branch "vazia", desatualizada em relação à pista inicial. |
  | `exp/dataset_validation` | 5 (origin) / 6 (local, com `459bf6d` ainda não enviado ao remoto) | Contém **tanto** o código de segmentação (`src/segmentation/`) **quanto** o de validação de dataset (`src/dataset_validation/`) e os testes correspondentes. |
  | `feature/feat_extraction` | 1 (`cc7d788 Criação da Feature Extraction`) | Contém `src/features/`, ADRs extensos e testes de borda ABCD. |
  | `feature/classification_baseline` | 1 (`56eda43 feat: Add ADR...`) | Adiciona **apenas** um ADR (`ADR-001 - Selection of EfficientNetV2-S...`); o código (`scripts/train_resnet50_ph2.py`, `src/models/model.py`) é **byte-idêntico** ao desta branch (confirmado via `git diff`) — ou seja, a decisão de arquitetura está registrada, a implementação ainda não. |
- Branches remotas adicionais não inspecionadas em profundidade (fora do
  escopo desta tarefa, apenas listadas por completude): `origin/develop`,
  `origin/examples`, `origin/feature/crewai`, `origin/feature/ml_task`,
  `origin/jules_wip_13570567048251198718`, `origin/new_jules`,
  `origin/refactor-notebook-to-project-9155276862737670784`.

---

## 2. Mapa da estrutura do repositório

### 2.1 Rastreado pelo git nesta branch (22 arquivos)

```
.gitignore
PH2_dataset.txt
README.md
config/config.yaml
requirements.txt
scripts/run_eval.py
scripts/run_pipeline.py
scripts/run_train.py
scripts/train_resnet50_ph2.py
src/__init__.py
src/agents/__init__.py
src/agents/agents.py
src/agents/schemas.py
src/data/__init__.py
src/data/dataset.py
src/data/preprocess.py
src/evaluation/__init__.py
src/evaluation/evaluate.py
src/models/__init__.py
src/models/model.py
src/pipeline/__init__.py
src/pipeline/pipeline.py
```

### 2.2 Presente na árvore de trabalho, mas **não rastreado** nesta branch

Confirmado via `git status` / `git ls-files`:

- `BUNDLE_README.txt`, `LangChain.ipynb`, `allclasses_bundle.zip`, `config.zip`
- `PH2/` — dataset completo (dezenas de milhares de arquivos `.bmp`, imagens + máscaras + ROIs)
- `analysis_gemini_{detail,summary}.csv`, `analysis_gpt_{detail,summary}.csv`
- `llm_result_gemini-3.1-flash-lite_2026060{6,9}.{csv,json}`, `llm_result_gpt-5.4-nano_2026060{6,9}.{csv,json}`
- `results.csv`, `results_stefani.csv`
- `docs/dataset_validation/report_allclasses.md` (único arquivo em `docs/` nesta árvore de trabalho)
- `reports/diagnostic_audit_2026-10-19/` — pacote completo de auditoria (relatórios `01`–`06`, `scripts/`, `tables/`, `figures/`, `audit_manifest.json`)
- `results/abcd_border_*` — seis subpastas com experimentos de validação geométrica do critério de Borda do ABCD (JSON/MD/PNG)
- `experiments/dataset_validation_allclasses/{audit,quality,reports}/` — **pastas vazias**, sem arquivos dentro

A proveniência desses arquivos **não é reconstruível pelo git** (não há commit
que os introduza nesta branch). Pelo conteúdo, grande parte corresponde a
artefatos de experimentos que também existem *versionados* em
`exp/dataset_validation` (ex.: `docs/dataset_validation/report_allclasses.md`),
sugerindo cópia manual de resultados gerados em outra máquina/ambiente (ver
`BUNDLE_README.txt`, que instrui explicitamente a "descompactar por cima" e
"trazer de volta" artefatos de outra máquina) — isso é consistente com o
observado, mas não foi confirmado rastreando commits.

### 2.3 Diretórios que existem, mas estão **vazios de código-fonte** nesta branch

Achado importante: `src/dataset_validation/`, `src/segmentation/`,
`src/features/`, `tests/`, `tests/dataset_validation/`, `tests/segmentation/`
existem como diretórios no disco, mas contêm **apenas bytecode `__pycache__`**
remanescente (`.pyc`), sem nenhum arquivo `.py` — e `git ls-files` confirma
que nada ali é rastreado nesta branch. Ou seja, a "pista" de que haveria
código de segmentação/validação/features nesta branch **não se confirma**: o
`__pycache__` é resíduo de execução anterior (provavelmente com outra branch
ou ambiente), não evidência de código presente aqui. O código real desses
três módulos existe apenas em `exp/dataset_validation` (segmentação +
validação de dataset) e `feature/feat_extraction` (features), ambas branches
não mescladas nesta.

### 2.4 Ambiente local (ignorado pelo git, não inspecionado em detalhe)

`.venv/`, `.pytest_cache/`, `.pytest_tmp/` (este último com permissão negada
para listagem no Windows — não investigado além disso), `.claude/settings.local.json`.

---

## 3. Pontos de entrada, dependências, testes, dados/checkpoints

### 3.1 Pontos de entrada (CLIs em `scripts/`, todos nesta branch)

| Script | Função |
|---|---|
| `scripts/run_pipeline.py` | Pipeline completo: pré-processamento RAG → dataset → (opcional) modelo de segmentação → grafo de agentes → inferência → avaliação. |
| `scripts/run_train.py` | Igual ao anterior, mas sem a etapa de avaliação final (apenas inferência). |
| `scripts/run_eval.py` | Avalia um JSON de resultados já gerado (`load_results` + `compute_metrics`). |
| `scripts/train_resnet50_ph2.py` | Treina um baseline ResNet-50 (não EfficientNetV2-S — ver seção 7) no PH2, split 70/30 estratificado. |

Todos usam `sys.path.insert(0, parents[1])` para localizar `src/` — não há
`pyproject.toml`/`setup.py`; o projeto não é instalável como pacote, precisa
ser executado a partir da raiz do repositório.

### 3.2 Dependências

`requirements.txt` (22 linhas): LangChain/LangGraph + providers
(`langchain-openai`, `langchain-google-genai`, `langchain-ollama`),
`pymupdf`/`faiss-cpu` (RAG), `sentence-transformers` (embeddings),
`torch`/`torchvision`/`Pillow` (segmentação), `pandas`/`numpy`/`scikit-learn`,
`pyyaml`.

**Dependência externa não declarada:** `src/models/model.py:14` importa
`from Image_Segmentation.code.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net`.
Esse pacote **não está em `requirements.txt`**, não foi encontrado vendorizado
em nenhum lugar do repositório, nem instalado em `.venv/` (confirmado por
busca). O docstring de `src/segmentation/strategies/unet.py` na branch
`exp/dataset_validation` confirma textualmente a dependência: "o pacote
externo 'Image_Segmentation.code.network' (LeeJunHyun/Image_Segmentation)
precisa estar no PYTHONPATH". Hoje, em qualquer branch, chamar
`segmentation.mode = "model"` falha com `ModuleNotFoundError` se esse pacote
não for instalado manualmente por fora do fluxo documentado.

### 3.3 Testes

**Nenhum teste está rastreado nesta branch.** `git ls-files tests` retorna
vazio. Testes existem apenas em:
- `feature/feat_extraction`: `tests/test_abcd_border_*.py` (6 arquivos) + `tests/test_feature_extraction.py`.
- `exp/dataset_validation`: `tests/dataset_validation/*.py` (5 arquivos) + `tests/segmentation/*.py` (5 arquivos).

Não há `pytest.ini`/`pyproject.toml` com configuração de testes em nenhuma
branch inspecionada.

### 3.4 Dados e checkpoints locais (caminhos, não conteúdo)

| Item | Status local |
|---|---|
| `PH2/` (imagens + máscaras + ROIs) | **Presente** na raiz do repo, não rastreado, estrutura `{NAME}/{NAME}_Dermoscopic_Image/{NAME}.bmp` + `{NAME}_lesion/{NAME}_lesion.bmp`. |
| `PH2_dataset.txt` (diagnóstico histológico pipe-delimited) | **Presente e rastreado.** |
| Checkpoint U-Net (`config/config.yaml:15` espera `U_Net-100-0.0005-70-0.6667.pkl`) | **Ausente** — não encontrado em nenhum lugar do repositório local (busca por `*.pkl`/`*.pt`/`*.pth` fora de `PH2/` e `.venv/` não retornou nada). |
| Checkpoints ResNet/EfficientNet | **Ausentes** — nenhum arquivo de pesos encontrado. |
| Dataset ISIC (`config/config.yaml:7-9`: `"Image/sample-image/image"`, `"Image/sample-metadata.csv"`) | **Ausente** localmente; caminho relativo não existe na árvore atual. |
| `config/config.yaml:11,13`: `ph2_dataset_dir`/`ph2_metadata_txt` apontam para `/content/Image/...` | Caminho de **Google Colab**, não válido nesta máquina — acoplamento de ambiente (ver seção 6). |
| Metadados HAM10000 (`C:/Users/andre/Downloads/Mestrado_V3/HAM10000_metadata`, referenciado em `reports/diagnostic_audit_2026-10-19/README.md`) | **Caminho existe** nesta máquina (confirmado); conteúdo/integridade não verificados. Imagens HAM10000 (bytes) não localizadas — a própria auditoria já registra essa ausência. |

---

## 4. Implementações reutilizáveis identificadas

- **`src/data/dataset.py`** — `SkinLesionDataset` (CSV ISIC-like) e `PH2Dataset`
  (estrutura de diretório PH2) expõem a mesma interface
  `get_image_paths() -> List[(path, lesion_size_mm, target, seg_path|None)]`,
  incluindo amostragem balanceada por classe com seed fixa. Útil como ponto
  de partida para loaders de outros datasets (BCN20000, ISIC2018/2019),
  desde que se mantenha o mesmo contrato de tupla.
- **`src/data/preprocess.py`** — extração de PDFs + construção de índice FAISS
  (`build_rag`) é genérica e não depende de domínio dermatológico específico.
- **`src/models/model.py`** — `MODEL_CLASSES` (dict factory para
  U_Net/R2U_Net/AttU_Net/R2AttU_Net), `load_segmentation_model`,
  `convert_to_base64`, `segment_image`, `apply_precomputed_mask`. O padrão de
  factory já usado aqui é o mesmo espírito do que o protocolo pede para a
  infraestrutura de segmentação intercambiável — e já foi de fato estendido
  nesse sentido em `exp/dataset_validation:src/segmentation/strategies/`
  (classe-base `SegmentationStrategy` + implementação `unet.py` que reusa
  exatamente este `MODEL_CLASSES`).
- **`src/pipeline/pipeline.py::run_pipeline`** — runner agnóstico a dataset e
  modelo, faz streaming do grafo LangGraph e salva progressivamente em JSON
  (linha a linha, após cada imagem) — reduz perda de trabalho em execuções
  longas, mas não implementa retomada automática entre execuções (ver seção 6).
- **`src/evaluation/evaluate.py`** — `compute_metrics` calcula
  acurácia/sensibilidade/especificidade a partir de um mapeamento fixo de
  string (`_DIAGNOSTICO_TO_LABEL`, linhas 10–14); simples, mas rígido (depende
  de substring match em `validation.diagnostico`, sem tolerância a variações
  de fraseado do LLM).
- **Padrão de configuração** — todos os 4 scripts leem o mesmo
  `config/config.yaml` com overrides pontuais via CLI; estrutura de seções
  (`paths`, `segmentation`, `llm`, `rag`, `pipeline`, `output`, `training`)
  é consistente e já comporta extensão.

---

## 5. Sistema multiagente

### 5.1 Agentes diagnósticos

Todos em `src/agents/agents.py`, cada um instanciando seu próprio LLM via
`_build_llm` (`agents.py:42-69`, seleciona `ChatOllama`/`ChatGoogleGenerativeAI`/`ChatOpenAI`
por prefixo do nome do modelo):

| Agente | Classe / linhas | Prompt / regra de pontuação (texto cravado no código) |
|---|---|---|
| ABCD | `ABCDDiagnosticAgent.analyze_lesion`, `agents.py:216-268` | Pesos 1.3/0.1/0.5/0.5 para A/B/C/D; soma < 4.75 benigna, 4.75–5.45 suspeita, > 5.45 maligna (`agents.py:263`). |
| Menzies | `MenziesDiagnosticAgent.analyze_lesion`, `agents.py:289-338` | ≥1 positiva e 0 negativas → melanoma; 1 negativa → suspeita; 2 negativas e 0 positivas → benigna (`agents.py:333`). |
| SPCL | `SPCLDiagnosticAgent.analyze_lesion`, `agents.py:359-404` | Soma ≥3 → suspeita (`agents.py:398`). |

Schemas Pydantic correspondentes em `src/agents/schemas.py:7-58`. Observação
já documentada de forma independente em
`reports/diagnostic_audit_2026-10-19/04_multiagent_root_cause.md:11`: o
prompt do Menzies lista nove características positivas, mas
`MenziesPositiveAnswer` (`schemas.py:17-26`) só tem sete campos — duas não
são capturáveis estruturalmente. Esse relatório de auditoria (não produzido
por mim, já presente na árvore de trabalho) também documenta divergências
semelhantes no SPCL e no cálculo ABCD; não foi reverificado byte a byte aqui,
apenas citado como achado prévio coerente com a leitura direta do código.

### 5.2 Segmentação

Dois agentes alternativos, selecionados em `build_graph(segmentation_mode=...)`
(`agents.py:540-552`):
- `SegmentationAgent` (`agents.py:122-164`) — roda a U-Net em tempo de execução via `src.models.model.segment_image`.
- `PrecomputedSegmentationAgent` (`agents.py:171-195`) — aplica uma máscara pré-computada (modo PH2) via `apply_precomputed_mask`.

### 5.3 Sumarizador — regra real (lida diretamente do código)

`SummaryAgent.summarize`, `agents.py:425-468`:
1. Serializa os três diagnósticos para JSON/texto (`_serialize_diagnosis`, `agents.py:111-115`).
2. Monta **um único prompt multimodal** contendo os três pareceres serializados
   mais a imagem segmentada (`agents.py:443-464`), pedindo para "integrar os
   três pareceres... construir um diagnóstico consolidado e fundamentado",
   considerando grau de concordância, padrões de alto risco recorrentes,
   confiabilidade dos critérios e, em caso de divergência, usar a própria
   imagem para decidir qual parecer é mais condizente (`agents.py:458-463`).
3. Chama o LLM **uma quarta vez** (além das três chamadas ABCD/Menzies/SPCL)
   com `with_structured_output(SummaryAnswer)`, retornando `diagnostico`
   (string livre: "Suspeita de Melanoma"/"Lesão Benigna"/"Lesão Indeterminada"
   são os valores sugeridos no schema, mas o campo aceita qualquer string —
   `schemas.py:64-66`), `justificativa` e `recomendacoes`.

**Não há, em nenhum ponto do código, soma ponderada, votação ou limiar
determinístico combinando os três escores.** A "regra" é inteiramente o
julgamento livre do LLM, condicionado por instruções qualitativas. Esta
leitura confirma, de forma independente, a mesma conclusão já registrada em
`reports/diagnostic_audit_2026-10-19/04_multiagent_root_cause.md:87`: "Não há
coeficientes de ponderação, limiar de score total ou condição determinística
de indeterminação."

`src/evaluation/evaluate.py:10-14` mapeia o texto de `diagnostico` para
binário por substring (case-insensitive): "suspeita de melanoma" → 1, "lesão
indeterminada" → 1 (tratamento conservador explícito no comentário), "lesão
benigna" → 0. Essa é uma regra de **pós-processamento para métricas**, não a
regra do sumarizador em si.

### 5.4 Agente crítico / RAG

`CriticalReviewAgent`, `agents.py:475-507`, **está presente no código e
conectado no grafo**: `agents.py:572-575` o registra como nó
`"critical_review"`, e `agents.py:586-587` liga `summary → critical_review → END`.
Isso contradiz uma leitura literal de "sem agente crítico/RAG ativo" como
"não existe" — ele existe e roda a cada execução do pipeline nesta branch.

Entretanto, por leitura de `validate_diagnosis` (`agents.py:487-507`): ele
apenas recupera documentos do `retriever` e concatena os primeiros 200
caracteres do primeiro resultado em uma string fixa
(`"Confirmação baseada em literatura médica: {...}"`), **sem nunca ler ou
alterar `state.validation`**. Ou seja: o nó existe e executa, mas é incapaz,
pelo fluxo de código atual, de mudar o diagnóstico — é diagnosticamente
inerte. Esse comportamento também já havia sido documentado de forma
independente em `reports/diagnostic_audit_2026-10-19/04_multiagent_root_cause.md:89`,
que acrescenta que, nos pilotos já executados, o retriever era um
`NullRetriever` com RAG desativado. Classificação mais precisa:
**parcial/presente-mas-inerte**, não simplesmente "ausente" nem "ativo" no
sentido de influenciar o resultado.

### 5.5 Feature Engineering (FE)

Não existe nesta branch (`src/features/` sem `.py` rastreado). Código
completo (11 módulos + ADRs + testes) existe apenas em
`feature/feat_extraction`. Não encontrei, em nenhuma branch, evidência de que
esse código de FE esteja de fato ligado ao grafo de agentes (`build_graph`
não referencia `src.features` em nenhuma versão inspecionada). O próprio
`reports/diagnostic_audit_2026-10-19/04_multiagent_root_cause.md:125` já
registra isso: "No grafo executado dos pilotos não há nó FE, pacote FE no
estado nem campos computacionais nas respostas."

### 5.6 Retry

Busca por `retry|tenacity|backoff|max_retries` em `src/` e `scripts/`
(arquivos-fonte, não bytecode) **não retornou nenhuma ocorrência**. Não há
wrapper de retry em volta das chamadas `.invoke()`/`with_structured_output()`
dos LLMs, nem no loop de `run_pipeline` (`pipeline.py:49-82`, que também não
tem `try/except` — uma exceção em qualquer imagem interrompe o loop inteiro,
sem registro separado de "falha técnica"). Isso é consistente com a exigência
do protocolo de "sem retry", mas também significa que **não há hoje nenhum
mecanismo de registro de falha técnica distinto de indeterminação
diagnóstica** nesta branch — ponto pendente, não resolvido aqui.

### 5.7 Nota lateral de implementação (não é segredo, é um provável bug)

`_build_llm` (`agents.py:60-68`) passa os literais `"GOOGLE_API_KEY"` e
`"OPENAI_API_KEY"` como **valor** dos parâmetros `google_api_key=`/`openai_api_key=`,
em vez de ler a variável de ambiente correspondente (ex.: `os.environ["OPENAI_API_KEY"]`).
Não verifiquei em runtime se os wrappers do LangChain ignoram esse valor e
caem para a variável de ambiente por conta própria; registrado como ponto a
confirmar antes de qualquer execução real.

---

## 6. Acoplamentos relevantes para a reconstrução

1. **Dependência externa não empacotada:** `Image_Segmentation.code.network`
   (seção 3.2) — bloqueia qualquer execução de segmentação em modo `"model"`
   até ser instalada manualmente; não está documentada em `requirements.txt`
   em nenhuma branch verificada.
2. **Caminhos hardcoded de ambiente (Colab):** `config/config.yaml:11,13`
   (`/content/Image/...`) não existem nesta máquina; qualquer execução exige
   edição manual do YAML ou overrides de CLI.
3. **Checkpoint de segmentação ausente:** `config/config.yaml:15` referencia
   um arquivo `.pkl` que não está no repositório nem localmente.
4. **Trabalho fragmentado em branches não mescladas**, todas divergindo do
   mesmo commit-base (`a557ba4`), sem rebase entre si:
   - `exp/dataset_validation` (segmentação + validação de dataset + testes)
   - `feature/feat_extraction` (feature engineering + ADRs + testes)
   - `feature/classification_baseline` (apenas ADR de arquitetura CNN)
   - `feature/segmentation` (vazia — nome não reflete conteúdo atual)
   Reconciliar esses trabalhos é uma tarefa de integração/merge real, não
   apenas "ler a outra branch" — por exemplo, a infraestrutura de
   segmentação em `exp/dataset_validation` não tem visibilidade do código de
   FE em `feature/feat_extraction`, e vice-versa.
5. **Artefatos não rastreados na árvore de trabalho** (docs/, reports/,
   results/, csv/json soltos, dois `.zip`) cuja proveniência não é
   reconstruível via git — risco de confundir "o que está no código" com "o
   que está fisicamente no disco desta máquina".
6. **Ausência de empacotamento** (`pyproject.toml`/`setup.py`): todo script
   depende de `sys.path.insert` e de ser executado a partir da raiz;
   reconstrução em outra máquina precisa replicar essa convenção exatamente.
7. **Credencial exposta** (seção "Achado de segurança") é também um
   acoplamento de governança: o repositório não pode ser compartilhado
   publicamente sem tratar isso primeiro.

---

## 7. Protocolo aprovado × estado atual do repositório

Convenção de status: **implementado** (existe e funciona como descrito no
protocolo) / **parcial** (existe algo relacionado, mas incompleto, divergente,
ou em branch não mesclada) / **ausente** (nada encontrado) / **não
verificado** (não foi possível confirmar por leitura direta no tempo desta
inspeção).

| Item do protocolo | Status | Evidência / observação |
|---|---|---|
| Dataset segmentação dev=ISIC2018 Task1 / eval=HAM10000 | **ausente/não verificado** | Nenhum loader de ISIC2018 Task1 encontrado. `src/segmentation/folds.py` (branch `exp/dataset_validation`) opera sobre **PH2**, não ISIC2018. |
| Dataset CNN dev=BCN20000 / eval=HAM10000 | **ausente** | `train_resnet50_ph2.py` treina e avalia só em PH2 (split interno 70/30), não em BCN20000/HAM10000. |
| Multiagente avaliado no mesmo diagnóstico do HAM10000 | **parcial** | `src/dataset_validation/allclasses.py` (branch `exp/dataset_validation`) já executa o multiagente sobre HAM10000 (7 classes, 20/classe = 140 imagens), mas é descrito no próprio relatório como "exploratório" — muito abaixo da amostra de 3.000 do protocolo. |
| PH2 complementar/opcional | **implementado** | `PH2Dataset` (`src/data/dataset.py:93-211`) funcional e testado com `PH2_dataset.txt`. |
| ISIC2019 fora do protocolo principal | **n/a** (nada encontrado referenciando ISIC2019 como ativo) | `docs/dataset_validation/report_allclasses.md:51-57` só discute ISIC2019 como implicação futura, não como experimento em execução. |
| Amostra congelada de 3.000 lesões, estratificada, mesmas imagens para todos sistemas | **ausente** | Nenhuma infraestrutura de amostragem/congelamento nesse tamanho encontrada; maior amostra localizada é 140 imagens (piloto exploratório). |
| CNN EfficientNetV2-S, ImageNet, threshold por Youden | **parcial** | ADR de seleção existe (`feature/classification_baseline`), mas o script real (`train_resnet50_ph2.py`, idêntico nessa branch e nesta) ainda treina **ResNet-50** com threshold fixo em 0.5 (`train_resnet50_ph2.py:241,431`), não Youden. |
| Segmentação: U-Net baseline, candidatas DeepLabV3+/EM-Net, nnU-Net opcional, factory+YAML intercambiável, 5 folds fixos | **parcial** | Em `exp/dataset_validation`: existe um padrão `SegmentationStrategy` (factory) com **apenas** a estratégia `unet` implementada (`src/segmentation/strategies/{base,unet}.py`); nenhuma estratégia DeepLabV3+/EM-Net/nnU-Net encontrada. `folds.py` faz k-fold estratificado, mas sobre PH2, não ISIC2018 Task1. Nada disso está nesta branch. |
| Multiagente ABCD+Menzies+SPCL+sumarizador | **implementado** | Seção 5.1–5.3. |
| Sem agente crítico/RAG ativo | **parcial — presente mas inerte** | Código existe e roda (`agents.py:475-507,572-587`), mas nunca altera o diagnóstico; RAG desativado nos pilotos já executados (ver seção 5.4). |
| Sem feature engineering | **implementado (conforme protocolo)** nesta branch | `src/features/` vazio aqui; código completo só em `feature/feat_extraction`, não mesclado, sem evidência de uso no grafo em nenhuma branch. |
| Sem o experimento de extração determinística de borda | **discrepância a assinalar** | `results/abcd_border_*/` (6 subpastas, não rastreadas) **já contém** um experimento de robustez geométrica do critério de Borda do ABCD (JSON/MD/PNG). A relação temporal/de escopo desse experimento com o protocolo aprovado não foi possível verificar — apenas registro de que ele já existe fisicamente na árvore de trabalho. |
| Prompts preservados até decisão explícita | **implementado (até onde foi possível comparar)** | Prompts em `agents.py` idênticos entre esta branch, `feature/segmentation` e `feature/classification_baseline` (mesmo arquivo, sem diff). Não comparado byte a byte com `exp/dataset_validation`/`feature/feat_extraction` (fora do orçamento desta inspeção). |
| Condições visuais S0/S1/S2 | **ausente** | Nenhuma lógica de contorno/condição visual encontrada em `agents.py`/`model.py`; a segmentação hoje só produz uma imagem mascarada (preenchimento preto fora da lesão), sem variantes de contorno. |
| Métricas: acurácia, precisão, sensibilidade (+ ao menos uma extra) | **parcial** | `evaluate.py` calcula acurácia/sensibilidade/especificidade (sem precisão explícita); `train_resnet50_ph2.py` calcula acurácia/acurácia balanceada/F1/AUC/sensibilidade/especificidade (também sem "precisão" nominal, embora F1/AUC cubram parte do requisito). Precisão/VPP aparece apenas nos scripts de auditoria pontual (`reports/diagnostic_audit_2026-10-19/`), não em código de produção. |
| Falhas: uma tentativa, sem retry, registro separado de falha técnica | **parcial/ausente** | Sem retry (confirmado, seção 5.6) ✅; mas também sem nenhum `try/except` no loop de inferência (`pipeline.py:49-82`) — uma falha técnica hoje interrompe a execução em vez de ser registrada separadamente. |
| Bootstrap 5.000 réplicas, cotas por categoria, IC95% percentil | **ausente nesta branch / existe parametrização diferente em auditoria pontual** | `reports/diagnostic_audit_2026-10-19` usa 10.000 réplicas (não 5.000) para um propósito de auditoria retrospectiva, não a infraestrutura experimental definitiva do protocolo. `src/dataset_validation/metrics.py` (branch `exp/dataset_validation`) implementa bootstrap estratificado genérico, reaproveitável, mas não parametrizado para 5.000/3.000 observado. |
| Grad-CAM qualitativo para CNN final | **ausente** | Nenhuma referência a Grad-CAM encontrada em nenhuma branch/arquivo pesquisado. |

---

## 8. Proposta de sequência de tarefas (não implementadas agora)

Ordem sugerida por dependência técnica, não por prioridade de pesquisa —
cada item pressupõe o anterior já aceito/mesclado:

1. **Consolidar a credencial exposta** (revogação + remoção de histórico, se
   decidido) — bloqueador de qualquer compartilhamento mais amplo do
   repositório. *Critério de aceitação:* chave antiga revogada confirmada no
   provedor; nenhuma chave em texto puro em árvore de trabalho ou histórico
   ativo (branches em uso).
2. **Decidir e documentar o merge/integração das branches divergentes**
   (`exp/dataset_validation`, `feature/feat_extraction`,
   `feature/classification_baseline`) antes de qualquer trabalho novo sobre
   `feature/inspection`/`main`. *Critério de aceitação:* ADR ou decisão
   registrada sobre ordem e estratégia de integração (rebase vs. merge vs.
   reescrita), sem conflitos não resolvidos.
3. **Resolver a dependência externa de segmentação** (`Image_Segmentation.code.network`):
   vendorizar, empacotar via `requirements.txt`/git submodule, ou substituir
   pela função equivalente dentro de `src/segmentation/strategies/unet.py`.
   *Critério de aceitação:* `segmentation.mode="model"` executa sem
   `ModuleNotFoundError` em ambiente limpo.
4. **Obter/validar checkpoint de segmentação e dataset ISIC2018 Task1** (dev)
   antes de qualquer treino de segmentação alinhado ao protocolo. *Critério
   de aceitação:* checkpoint e dataset presentes e íntegros nos caminhos
   declarados em config, com hash registrado.
5. **Portar a infraestrutura de folds/factory de segmentação para ISIC2018
   Task1** (hoje em `exp/dataset_validation` ela opera sobre PH2). *Critério
   de aceitação:* 5 folds fixos e reproduzíveis sobre ISIC2018 Task1,
   persistidos com manifesto (padrão já usado em `folds.py` para PH2).
6. **Implementar estratégias DeepLabV3+/EM-Net como candidatas no mesmo
   factory**, sem alterar o contrato de `SegmentationStrategy`. *Critério de
   aceitação:* ao menos uma estratégia adicional registrada e testada com os
   mesmos folds do item 5.
7. **Substituir ResNet-50 por EfficientNetV2-S no script de treino da CNN**,
   migrando dev/eval para BCN20000/HAM10000 e implementando threshold por
   Youden na validação interna. *Critério de aceitação:* checkpoint e
   threshold congelados e registrados antes de qualquer avaliação em
   HAM10000.
8. **Definir e implementar a amostra congelada de 3.000 lesões** (sementes
   fixas, estratificação, mesmas imagens entre sistemas/condições). *Critério
   de aceitação:* manifesto de amostra com hash, reproduzível, usado por
   todos os sistemas/condições subsequentes.
9. **Implementar condições visuais S0/S1/S2** no pipeline de segmentação →
   agentes. *Critério de aceitação:* as três variantes geradas para a mesma
   imagem, com padronização de cor/espessura/resolução documentada.
10. **Adicionar registro de falha técnica separado de indeterminação
    diagnóstica** no loop de `run_pipeline`, sem introduzir retry. *Critério
    de aceitação:* uma falha simulada (ex.: erro de rede) é registrada em
    log/estrutura própria e a execução continua para as próximas entradas.
11. **Decidir explicitamente o destino do agente crítico/RAG** (remover do
    grafo, ou torná-lo capaz de de fato alterar `validation`, com evidência
    de benefício). *Critério de aceitação:* decisão registrada em ADR; se
    mantido, código altera `state.validation` de forma testável.
12. **Implementar bootstrap de 5.000 réplicas com cotas por categoria e IC95%
    percentil**, parametrizável para a amostra de 3.000, reaproveitando o
    utilitário genérico já existente em `src/dataset_validation/metrics.py`.
    *Critério de aceitação:* saída reproduzível (mesma seed → mesmo IC) sobre
    um conjunto de teste fixo.
13. **Adicionar Grad-CAM qualitativo à CNN final**, só após o item 7 estar
    concluído. *Critério de aceitação:* mapas gerados para uma amostra de
    casos, uso explicitamente rotulado como qualitativo em qualquer relatório.

Nenhuma dessas tarefas foi iniciada nesta inspeção.

---

## 9. Separação explícita de naturezas de trabalho

Para evitar confundir estas três coisas em tarefas futuras:

**(a) Infraestrutura de implementação (código) já existente** — por branch:
- Esta branch (`feature/inspection`/`main`): pipeline multiagente completo
  (agentes, grafo, RAG inerte, avaliação simples), loader PH2/ISIC genérico,
  treino ResNet-50 em PH2.
- `exp/dataset_validation`: módulo `dataset_validation` (amostragem, métricas
  com bootstrap, interpretação, relatórios) + módulo `segmentation` (factory
  de estratégias, folds, engine, comparação, relatório) — ambos com testes.
- `feature/feat_extraction`: módulo `features` (ABCD/Menzies/7-point
  computacionais) + testes + ADRs extensos de reprodutibilidade científica.
- `feature/classification_baseline`: só o ADR de escolha de arquitetura
  (EfficientNetV2-S); zero código novo.

**(b) Trabalho de auditoria/análise já executado (não é implementação, não é
experimento da dissertação)** — `reports/diagnostic_audit_2026-10-19/`
(auditoria retrospectiva dos resultados de 140 imagens já existentes:
`01_executive_summary.md` a `06_recommended_experimental_design.md`), e
`docs/dataset_validation/report_allclasses.md` (relatório descritivo do
mesmo piloto de 140 imagens). Ambos **analisam dados já coletados**; não
treinam, não chamam LLMs, não geram novas inferências.

**(c) Execução experimental real da dissertação (ainda não realizada)** —
nenhum dos itens do protocolo aprovado (amostra de 3.000 lesões congelada,
segmentação dev=ISIC2018 Task1/eval=HAM10000, CNN dev=BCN20000/eval=HAM10000
com EfficientNetV2-S, multiagente nas condições S0/S1/S2 sobre a amostra
final, bootstrap de 5.000 réplicas) foi executado. O que existe sob os nomes
"experimento"/"piloto" até aqui (140 imagens HAM10000, border experiments do
ABCD, treino PH2) é **exploratório/preparatório**, não a execução do
protocolo principal.

---

## 10. Limitações desta inspeção

- Não li o conteúdo completo de todos os módulos em `exp/dataset_validation`
  e `feature/feat_extraction` linha a linha (ex.: `src/dataset_validation/allclasses.py`,
  `inference.py`, `sampling.py`, todos os extratores em `src/features/`) —
  confirmei sua existência, nomes, e alguns trechos-chave via `git show`/`grep`,
  mas uma auditoria de corretude desses módulos exigiria leitura completa,
  fora do orçamento desta tarefa.
- Não executei nenhum código, não instalei dependências, não tentei importar
  `Image_Segmentation`, não validei integridade de `PH2/` nem dos dados
  HAM10000 — apenas confirmei presença/ausência de caminhos.
- Não comparei byte a byte os prompts de `agents.py` entre **todas** as
  branches (comparei apenas contra `feature/segmentation` e
  `feature/classification_baseline`, que são diffs vazios); não confirmei se
  `exp/dataset_validation` ou `feature/feat_extraction` alteraram os prompts.
- A busca por segredos usou padrões de regex conhecidos (OpenAI `sk-proj-`,
  Google `AIza`, tokens Slack, chaves privadas PEM) sobre arquivos de texto e
  o histórico completo do git; não abri `allclasses_bundle.zip`/`config.zip`
  em profundidade (apenas listei nomes de arquivo), então não há garantia de
  que não haja segredos dentro desses arquivos compactados.
- `.pytest_tmp/` não pôde ser listado (permissão negada no Windows) — não
  verificado.
- A proveniência exata dos arquivos não rastreados (seção 2.2) é uma
  inferência baseada em correspondência de conteúdo com outras branches, não
  uma confirmação via histórico de versionamento — pode estar incompleta ou
  incorreta.
- Não investiguei `.claude/settings.local.json` (ignorado pelo git) além de
  confirmar sua existência.
