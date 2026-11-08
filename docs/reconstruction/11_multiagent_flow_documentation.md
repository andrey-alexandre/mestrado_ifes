# Fluxo Real do Sistema Multiagente — Documentação (Etapa 5, A05/P012) e Remoção do Crítico/RAG (A36)

Documentação do que o código **faz hoje**, lida diretamente em
`src/agents/agents.py`/`schemas.py`/`src/pipeline/pipeline.py` nesta
sessão (2026-11-08), não reconstruída de memória ou de versões antigas.
Onde a inspeção da Tarefa 1 (há ~1 mês) já tinha registrado algo, cada
ponto foi reconferido linha a linha; nenhuma observação antiga foi
copiada sem reverificação.

## 1. A36 — Remoção do crítico/RAG do fluxo ativo

### Estado ANTES (confirmado nesta sessão, antes da edição)

- `agents.py:572-575` registrava o nó `critical_review`
  (`CriticalReviewAgent(retriever, model_name).validate_diagnosis`).
- `agents.py:586-587` ligava `summary → critical_review → END`.
- `CriticalReviewAgent.validate_diagnosis` (`agents.py:487-507`): lê
  `state.validation`, chama `self.retriever.invoke(str(val_res))`, e
  devolve `{"final_report": f"Confirmação baseada em literatura médica: {retrieved_docs[0].page_content[:200]}..."}`.
  Ele **lê** `state.validation` mas **nunca escreve nele** — não altera o
  diagnóstico consolidado, só anexa um campo `final_report` novo ao
  estado.

### Estado DEPOIS (edição feita nesta entrega)

- `agents.py`: o registro do nó `critical_review` (`add_node`) foi
  **removido**, não só a aresta. A classe `CriticalReviewAgent`
  (`agents.py:475-507`) **não foi apagada** — continua no arquivo,
  intocada, só não é mais instanciada/registrada dentro de `build_graph`.
- A aresta final agora é `summary → END` diretamente
  (`agents.py`, trecho "A36: fluxo termina no sumarizador").
- **Decisão de implementação, fora do que foi sugerido no escopo**: a
  instrução original permitia só desconectar a aresta, mantendo o nó
  registrado. Optei por remover o registro (`add_node`) inteiro, não só a
  aresta, porque o LangGraph (`StateGraph.compile()`) valida que todo nó
  registrado seja alcançável a partir de `START` — um nó registrado sem
  nenhuma aresta de entrada tem risco real de fazer `graph.compile()`
  falhar com um erro de nó inalcançável na primeira execução real do
  pesquisador. Como esta tarefa não executa nada, não dava para testar
  empiricamente qual comportamento o LangGraph teria; removido o registro
  inteiro é a opção que elimina esse risco com certeza, ao custo de não
  ser literalmente "só desconectar a aresta" como primeiro sugerido. Se
  isso não for aceitável, é reversível: basta registrar o nó de novo e
  testar se `compile()` aceita um nó órfão.
- O fluxo completo agora é:
  `START → segmentation → [diagnostic_abcd ‖ diagnostic_menzies ‖ diagnostic_spcl] → summary → END`.

### Achado: pontos de entrada ainda exigem RAG incondicionalmente

`scripts/run_pipeline.py:147-152` e `scripts/run_train.py:192-197` chamam
`build_rag(...)` **incondicionalmente**, sempre, e passam o resultado
como `retriever` para `build_graph(retriever=retriever, ...)` —
independente de o crítico estar ou não no caminho ativo do grafo.
`build_rag` (`src/data/preprocess.py:64-105`) exige que
`medical_texts.txt` já exista (ou seja gerado por `prepare_medical_texts`,
que por sua vez exige `RAG.zip`), baixa/usa o modelo de embeddings
SentenceTransformer, e constrói um índice FAISS — custo real de
setup/execução que hoje **não tem mais nenhum consumidor** dentro de
`build_graph`, já que o parâmetro `retriever` deixou de ser usado
internamente (ver docstring atualizada de `build_graph`).

**Isto não foi corrigido nesta entrega** — mudar a assinatura de
`build_graph` (tornar `retriever` opcional) e os dois pontos de entrada
(parar de construir o RAG incondicionalmente) é uma mudança de estrutura
maior que o escopo autorizado ("se for uma mudança pequena e segura...
pode ajustar — mas se for maior, só registre"). **Pendência explícita
para decisão**: hoje, rodar `run_pipeline.py` ou `run_train.py` sem
`--skip-preprocess` (ou sem `medical_texts.txt` já existente) ainda falha
ou gasta recursos construindo um RAG que nenhum nó ativo consome.

### Evidência histórica: o crítico rodou de verdade nos experimentos antigos?

Distinção exigida pelo pesquisador: "o nó existe no código" (fato de
código, confirmado acima) é diferente de "há evidência de que rodou"
(fato empírico). Busquei por evidência **nos dados**, não no código:

- `llm_result_gpt-5.4-nano_20260609.json` e
  `llm_result_gemini-3.1-flash-lite_20260609.json`: **TÊM** o campo
  `"final_report"` populado com o texto literal
  `"Confirmação baseada em literatura médica: An Bras Dermatol. 2016;91(1):49-58...`
  (mesma citação/trecho em todas as entradas verificadas do arquivo
  `gpt-5.4-nano`, o que por si só é uma observação lateral — sugere baixa
  diversidade de recuperação do FAISS, não investigado mais a fundo
  aqui). Esse texto só pode ter sido gerado por
  `CriticalReviewAgent.validate_diagnosis` (é o literal exato de
  `agents.py:503-506`) — **isto é evidência real de que o crítico/RAG
  rodou** nesses dois arquivos de resultado (datados de 09/06).
- `llm_result_gpt-5.4-nano_20260606.json` e
  `llm_result_gemini-3.1-flash-lite_20260606.json`: **NÃO** têm o campo
  `final_report` em nenhuma entrada (busca pela chave retornou zero
  ocorrências) — **sem evidência de execução do crítico** nesses dois
  arquivos (datados de 06/06, anteriores aos de 09/06).
- `results.csv` (raiz do repo, parece ser o experimento PH2 citado no
  texto da dissertação): colunas são só
  `validation.diagnostico,estimated_Value,real_value` — **não tem nenhuma
  coluna relacionada a crítico/RAG**. Isso é consistente com "não rodou
  nesse experimento específico", mas não é prova definitiva sozinha — a
  ausência da coluna também poderia (em tese) refletir um script de
  exportação que simplesmente não incluiu esse campo, mesmo que tivesse
  sido computado. Não teve como verificar isso sem reexecutar nada, o que
  não é permitido nesta tarefa. `results_stefani.csv` não foi aberto por
  falta de tempo/orçamento desta entrega — pendência de inspeção, não
  uma afirmação sobre seu conteúdo.

**Conclusão registrada**: o crítico/RAG tem evidência empírica real de
execução em pelo menos dois arquivos de resultado (09/06), e evidência de
ausência em pelo menos dois outros (06/06) e no `results.csv` do PH2. Não
é uniforme — não dá para afirmar "sempre rodou" nem "nunca rodou" nos
experimentos antigos; os dois estados coexistem em arquivos diferentes.

### Pendências de documentação (não atualizadas nesta entrega)

- `README.md`: **vazio (0 linhas)** — nada para atualizar, não descreve
  nenhum fluxo.
- `docs/reconstruction/01_inspection_report.md`: menciona o crítico como
  parte do grafo — é um relatório de inspeção **datado e histórico**
  (registra o estado de um mês atrás), não uma especificação viva do
  fluxo atual. Não foi alterado (preserva o registro histórico, como
  pedido), e não deveria ser lido como descrição do estado atual a partir
  de agora — este novo documento (`11_...md`) é a referência atualizada.
- Nenhum diagrama (Mermaid, imagem, etc.) do fluxo foi encontrado em
  nenhum lugar do repositório na busca feita.

## 2. Fluxo real — agentes, schemas, sumarizador

### 2.1 Os três agentes diagnósticos

| Agente | Classe / método | `agents.py` | Entrada (de `GraphState`) | Schema de saída (`schemas.py`) |
|---|---|---|---|---|
| ABCD | `ABCDDiagnosticAgent.analyze_lesion` | 202-268 | `state.seg_image_data` (imagem segmentada, base64) | `ABCDDiagnosticAnswer` (`schemas.py:7-14`): `assimetria`, `bordas`, `cor`, `estruturas`, `explicacao` — todos `str` livres (não campos numéricos tipados) |
| Menzies | `MenziesDiagnosticAgent.analyze_lesion` | 275-338 | idem | `MenziesDiagnosticAnswer` (`schemas.py:36-45`): `positivas` (`MenziesPositiveAnswer`, 7 campos `str`, linhas 17-26), `negativas` (`MenziesNegativeAnswer`, 2 campos `str`, linhas 29-33), `explicacao` |
| SPCL (Seven-Point) | `SPCLDiagnosticAgent.analyze_lesion` | 345-404 | idem | `SPCLDiagnosticAnswer` (`schemas.py:48-58`): 7 campos de critério (`str`) + `explicacao` |

Cada agente constrói seu próprio `HumanMessage` multimodal
(`_image_message`, `agents.py:92-108`: imagem como `image_url` base64 +
texto do prompt) e chama `_invoke` (`agents.py:72-89`), que usa
`llm.with_structured_output(output_format).invoke(messages)` para
modelos OpenAI/Gemini, ou `llm.invoke(messages).content` (texto livre,
sem schema forçado) para modelos Ollama (`_is_ollama`, linha 32-34: nome
contém `:`).

**Observação sobre os schemas**: todos os campos de critério são `str`
livres com instrução textual ("Valor de 0 a 2 e motivo", "Sim ou não e
motivo"), não tipos numéricos/booleanos estritos — o schema Pydantic
garante a *estrutura* (quais campos existem), não que o *conteúdo* de
cada campo seja parseável como número. Isso significa que o cálculo
final descrito no prompt (soma ponderada do ABCD, contagem Menzies/SPCL)
é instrução textual para o LLM seguir, não verificado/recalculado em
código Python em nenhum lugar — reconfirmado: não há nenhuma lógica de
parsing/soma desses campos em `agents.py`.

### 2.2 O sumarizador (`SummaryAgent.summarize`, `agents.py:411-468`)

**Entrada**: os três diagnósticos (`state.diagnosis_abcd`,
`diagnosis_menzies`, `diagnosis_spcl`), serializados via
`_serialize_diagnosis` (`agents.py:111-115`: `model_dump()` + `json.dumps`
se for Pydantic, senão `str()` direto) e embutidos como texto no prompt,
mais `state.seg_image_data` (a mesma imagem segmentada, de novo, como
`image_url`).

**Prompt de consolidação — texto exato** (`agents.py:443-464`):

```
<-- Identidade do Agente -->
Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.
Não forneça um diagnóstico ou julgamento clínico.

Você recebe a imagem segmentada da lesão acima, que realça os contornos e características internas da lesão.

<-- Instruções Específicas -->
Você está atuando como um especialista que recebeu três pareceres técnicos distintos sobre uma mesma lesão cutânea, baseados nos seguintes algoritmos:
1️⃣ ABCD – {abcd_str}
2️⃣ Menzies – {menzies_str}
3️⃣ SPCL (Seven Point Checklist) – {spcl_str}

Sua função é integrar os três pareceres fornecidos, comparando seus resultados e justificativas, para construir um **diagnóstico consolidado e fundamentado**.

Considere:
- O grau de concordância ou conflito entre os algoritmos;
- A presença de padrões de alto risco recorrentes nos pareceres;
- A confiabilidade dos critérios observados (por exemplo, múltiplas cores, bordas irregulares e estruturas atípicas);
- A gravidade potencial com base em critérios combinados.
- Caso haja divergências significativas entre os pareceres, utilize a imagem disponibilizada para avaliar e decidir qual está mais condizente com a lesão
```
(`{abcd_str}`/`{menzies_str}`/`{spcl_str}` são os três pareceres
serializados, interpolados no f-string.)

**Confirmado: julgamento livre do LLM, sem regra determinística.** Não há,
em nenhum lugar de `summarize` ou do prompt acima, soma ponderada,
votação, contagem de concordância, nem limiar numérico combinando os três
escores — a integração é inteiramente a interpretação qualitativa do LLM
sobre o texto serializado dos três pareceres mais a própria imagem. A
chamada final (`agents.py:467`) usa
`llm.with_structured_output(SummaryAnswer)`.

**Saída**: `SummaryAnswer` (`schemas.py:61-77`): `diagnostico` (`str`
livre — a descrição sugere "Suspeita de Melanoma / Lesão Benigna / Lesão
Indeterminada", mas **o campo aceita qualquer string**, não é um Enum
Pydantic restrito), `justificativa`, `recomendacoes`.

### 2.3 Tratamento de resposta parcial/inválida — o que o código FAZ

Reconfirmado, não presumido:

- **Nenhum `try/except` existe** em nenhum método de nenhum agente
  (`analyze_lesion`, `summarize`) nem em `_invoke`
  (`agents.py:72-89`). Se `with_structured_output(...).invoke(...)`
  lançar uma exceção (ex. a saída do LLM não bate com o schema Pydantic,
  erro de rede, erro de autenticação), a exceção **propaga sem ser
  capturada**.
- `src/pipeline/pipeline.py:60` (`for event in compiled_graph.stream(...)`)
  também **não tem `try/except`** ao redor do `stream`. Uma exceção em
  qualquer nó, para qualquer imagem, interrompe o `for j, (...)` inteiro
  em `run_pipeline` — a execução para ali, sem processar as imagens
  restantes da lista, sem registrar a falha num campo separado, sem
  retry.
- **Nenhuma lógica de retry existe** em nenhum arquivo (`agents.py`,
  `pipeline.py`, `_build_llm`) — reconfirmado nesta sessão, busca direta
  no texto dos três arquivos.
- **Nenhuma conversão de ausência de diagnóstico em "não melanoma" existe
  em `src/agents/`/`src/pipeline/`** — como não há tratamento de erro
  nenhum nesses arquivos, não existe um "else" ou "fallback" que
  produziria esse valor; o comportamento real diante de falha é a
  interrupção do pipeline, não uma substituição silenciosa do
  diagnóstico. (`src/evaluation/evaluate.py` tem uma regra de
  pós-processamento para métricas que mapeia texto para binário, incluindo
  tratar "lesão indeterminada" como 1 — mas isso é só para cálculo de
  métricas sobre resultados já existentes, não altera o que o pipeline
  grava como diagnóstico; não investigado mais a fundo aqui, fora do
  escopo desta entrega.)

### 2.4 Ausência silenciosa de parecer (A05 — correção 2026-11-08)

**Defeito encontrado ao verificar os resultados reais do piloto** (não
hipotético): no cenário `agente_nao_responde` (seção 4), quando o SPCL
retorna `None`, o sumarizador passava a receber só 2 dos 3 pareceres
(`summary_prompt_marker_count` cai de 3 para 2), e a execução terminava
normalmente (`ok: true`) **sem nenhum registro, em nenhum lugar do
estado, de que um parecer estava ausente**. O detector de divergências
do piloto (`_detect_divergences`) não pegava isso porque só verificava
retry e conversão para "não melanoma" — não ausência silenciosa.

**Correção implementada** (decisão do pesquisador, não reaberta aqui):

- `schemas.py`: nova classe `AgentStatus` (`status: str`, `causa:
  Optional[str]`) e três novos campos em `GraphState`:
  `status_abcd`, `status_menzies`, `status_spcl` — um por agente, não um
  dict único compartilhado, porque os três nós diagnósticos rodam em
  paralelo (fan-out de `segmentation`) e o LangGraph exigiria um reducer
  explícito para múltiplas escritas concorrentes na mesma chave.
- `agents.py`: cada um dos três `analyze_lesion` (ABCD/Menzies/SPCL)
  passou a checar `if output is None` após `_invoke(...)` e retornar,
  além do campo de diagnóstico já existente, `status_<agente> =
  AgentStatus(status="sem_resposta", causa="desconhecida")` nesse caso,
  ou `AgentStatus(status="ok")` no caso normal. **Não é um
  `try/except`** — é um `if/else` simples sobre o valor já retornado por
  `_invoke`; nenhuma lógica de chamada ao LLM foi alterada, nenhum
  retry foi introduzido, nenhuma causa é inferida além de "desconhecida"
  quando não há mais informação disponível.
- `pipeline.py`: os três novos campos (`AgentStatus`, objetos Pydantic)
  foram adicionados à lista de campos serializados via `model_dump()`
  antes do `json.dump` — sem isso, uma execução real quebraria ao
  tentar serializar um objeto Pydantic não tratado. Nenhuma outra
  mudança em `pipeline.py`.
- O sumarizador **não foi alterado**: continua recebendo exatamente os
  pareceres disponíveis no estado (hoje 2 de 3 nesse cenário), sem
  parecer fictício, sem mudança de prompt, sem segunda chamada ao
  agente ausente.

**Dois indicadores separados, calculados pelo piloto** (função pura
`_compute_consolidation`, `scripts/run_multiagent_flow_pilot.py`), não
escritos no `GraphState` em si:

- **Cobertura diagnóstica**: `True` se o sumarizador produziu um
  `validation.diagnostico` não nulo — independe de quantos pareceres
  chegaram até ele. Se o protocolo conta o caso como cobertura mesmo
  com consolidação parcial, essa regra não muda.
- **Consolidação parcial**: `pareceres_recebidos`/`pareceres_esperados`
  (ex. "2/3") e `agentes_ausentes` (lista de identificadores, calculada
  genericamente iterando os três agentes — não hardcoded para o SPCL,
  que foi só o agente ausente no cenário testado).
- **Status final**: `"determinado"` se há diagnóstico do sumarizador,
  `"indeterminado"` caso contrário — independente dos status
  individuais por agente (um caso pode ter todos os três pareceres
  presentes e ainda assim o sumarizador não produzir diagnóstico válido;
  os dois fatos são registrados separadamente).

**Ressalva explícita do pesquisador, preservada aqui**: o diagnóstico
fixo (`"Lesão Benigna"`) que aparece nesses cenários vem do dublê do
piloto, não de um LLM real — não demonstra nada sobre viés ou
necessidade de nova regra no sumarizador real. Nenhuma regra nova foi
introduzida no sumarizador por causa desta correção.

**Pendência registrada, não implementada nesta correção**: os
indicadores de cobertura/consolidação parcial/status final hoje só
existem no piloto (`scripts/run_multiagent_flow_pilot.py`), calculados
a partir do estado final do grafo. Se o pesquisador quiser esses mesmos
indicadores persistidos também nas execuções reais (`run_pipeline.py` /
`results.csv`), isso não foi feito aqui — está fora do escopo desta
correção pontual (que é sobre o piloto simulado, A05).

## 3. O que NÃO foi feito nesta entrega (reforço explícito)

Não foi introduzida nenhuma votação, limiar numérico, Feature Engineering
nem fallback diagnóstico novo em nenhum arquivo. A única mudança de
comportamento de execução é a remoção do nó `critical_review` do grafo
(seção 1) — tudo mais neste documento é descrição do que já existia.

## 4. Piloto local com respostas simuladas (verificação executável)

**STATUS: implementado, aguardando execução do pesquisador.** Não
executado nesta sessão (diretiva explícita de não rodar nenhum código).

- Script: `scripts/run_multiagent_flow_pilot.py`.
- Comando exato: `python scripts/run_multiagent_flow_pilot.py`.
- Sem credenciais, sem chamadas de rede: `_build_llm` é substituído por
  um dublê (`FakeLLM`/`FakeStructuredRunnable`) que devolve respostas
  pré-construídas; `PrecomputedSegmentationAgent.segment_image` também é
  substituído para não depender de arquivo de imagem/máscara real. O
  grafo executado é o **grafo real pós-edição** (`build_graph`, seção 1),
  não uma reimplementação.
- Saídas, em `experiments/multiagent/` (diretório criado pelo próprio
  script se não existir):
  - `multiagent_pilot_report.md` — resumo legível, tabela por cenário.
  - `multiagent_pilot_results.json` — resultado estruturado completo.
  - `multiagent_pilot.log` — log da execução.
- Cenários cobertos:
  1. `caminho_normal` — ABCD/Menzies/SPCL simulados com respostas
     válidas; confirma (inspecionando o texto real recebido pelo dublê
     do sumarizador, contando as ocorrências do marcador presente nos
     três pareceres simulados) que o sumarizador de fato recebe os três
     pareceres, e que a execução termina com `final_report == None` no
     estado final (a chave sempre existe — `GraphState.final_report` é
     declarado com default `None` em `schemas.py:93` — mas seu valor só
     deixaria de ser `None` se `CriticalReviewAgent.validate_diagnosis`
     tivesse rodado, o único lugar do código que o preenche).
  2. `agente_lanca_excecao` — ABCD simulado lançando uma exceção.
  3. `agente_fora_do_schema` — Menzies simulado devolvendo um `dict` cru
     em vez da instância Pydantic esperada.
  4. `agente_nao_responde` — SPCL simulado devolvendo `None`.
- Para os cenários 2–4, o piloto registra o **número real de chamadas**
  por agente (via contador no dublê) e o conteúdo do campo de diagnóstico
  final, e sinaliza como `"DIVERGÊNCIA ENCONTRADA — requer decisão"`,
  com arquivo:linha, qualquer um dos dois casos abaixo caso realmente
  ocorram ao rodar:
  - um agente é chamado mais de uma vez num cenário de falha (indicaria
    retry automático, o que contradiz a ausência de `try/except`/retry
    confirmada na seção 2);
  - o campo de diagnóstico final contiver uma variação de "não melanoma"
    num cenário de falha simulada (indicaria conversão silenciosa, o que
    contradiz a ausência de fallback confirmada na seção 2).
- **(A05 — correção 2026-11-08)** Além do acima, o piloto agora calcula
  e registra, por cenário (via `_compute_consolidation`, ver seção 2.4):
  `pareceres_recebidos`/`pareceres_esperados`, `agentes_ausentes`,
  `diagnostic_coverage` e `status_final`. No cenário `caminho_normal`,
  espera-se 3/3 sem ausentes; no `agente_nao_responde`, espera-se 2/3
  com `agentes_ausentes == ["diagnostic_spcl"]`, preservando a saída
  fixa do dublê do sumarizador. O cenário `agente_fora_do_schema`
  continua travando com `ValidationError` do próprio `GraphState`
  (comportamento inalterado por esta correção).
- Este fork **não executou o script**; não há, portanto, resultado real
  de PASSOU/FALHOU a reportar aqui — apenas a implementação e a lógica de
  detecção, prontas para o pesquisador rodar. Se a execução real revelar
  qualquer divergência, ela aparecerá marcada no `.md`/`.json` gerados,
  não deve ser corrigida silenciosamente antes de uma decisão.
- Testes sintéticos (dados fabricados, não reais) da lógica pura
  extraída (`_detect_divergences`, dublês de comportamento,
  `_render_report_md`): `tests/multiagent/test_pilot_logic.py`. Comando
  exato: `pytest tests/multiagent/test_pilot_logic.py -v`. Não cobrem o
  grafo real em isolamento (isso é papel do próprio piloto rodado pelo
  pesquisador); não executados nesta sessão.
