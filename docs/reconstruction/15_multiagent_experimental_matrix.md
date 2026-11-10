# 15 — Matriz experimental do multiagente (Etapa 5) e levantamento de modelos/configuração

Sessão de documentação/levantamento (2026-11-09), sem execução de código,
testes ou chamadas a modelo (`feedback_no_autonomous_execution`). Tudo
abaixo foi lido diretamente do código, da configuração e dos arquivos de
resultado existentes no repositório nesta sessão — nenhuma lacuna foi
preenchida por memória. Onde não havia como confirmar algo sem executar
ou sem acesso a informação externa verificável, isso está marcado como
pendência, não como fato.

## 1. Matriz aprovada (registro, não decidida nesta entrega)

Aprovada pelo pesquisador antes desta entrega. Todas as condições usam as
**3.000 lesões do manifesto HAM10000 congelado** (seed `20261106`,
SHA-256 `32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`,
ver `10_frozen_sample_registry.md`), uma imagem por lesão. Qualquer
execução futura deve reconferir esse hash antes de usar o manifesto.

| Sistema | Condição | Entrada | Dependência para execução |
|---|---|---|---|
| EfficientNetV2-S (CNN) | única | Imagem original, sem contorno | Checkpoint definitivo (treino ainda não concluído — ver §4) e limiar escolhido na validação interna do BCN20000 (ver §4). Dev/val exclusivamente BCN20000; sem segunda CNN treinada com segmentação. |
| VLM direto | S0 | Imagem original | Modelo a definir (ver §5) |
| Multiagente | S0 | Imagem original | Modelo a definir (ver §5); ABCD+Menzies+SPCL+sumarizador, sem Feature Engineering, sem crítico/RAG (fluxo A36, `11_multiagent_flow_documentation.md`) |
| Multiagente | S1 | Original + contorno da **máscara de referência** (Tschandl/HAM10000) | Mesma forma de desenho de contorno usada em S2 (a definir, ver §6.4) |
| Multiagente | S2 | Original + contorno da **máscara predita pelo segmentador final** | Aguarda arquitetura de segmentação final e máscaras preditas sobre as 3.000 imagens (ver §4) |

Em S1/S2 o fluxo multiagente preserva ABCD, Menzies, SPCL e sumarizador
com os prompts/regras vigentes (texto exato documentado em
`11_multiagent_flow_documentation.md §2.2`) — nenhuma mudança de prompt
está implícita nesta matriz.

## 2. Comparações aprovadas (registro)

1. **Principal — CNN (imagem original) × Multiagente S2.** Cada pipeline
   com a entrada prevista para seu uso real. S2 é a condição principal
   mesmo que inferior a S0 — não há cláusula de substituição por "melhor
   resultado". CNN × Multiagente S0 **não** integra a matriz principal.
2. **A34 — VLM direto S0 × Multiagente S0.** Mesmo VLM, mesmas imagens,
   mesma definição de tarefa; saída tratada de forma idêntica para
   decisão binária/indeterminação/recusa/falha (ver gap de schema no
   §6.1 — isso ainda não está pronto hoje). Configuração do sumarizador
   documentada separadamente (ela só existe no lado multiagente; o VLM
   direto não tem sumarizador).
3. **Efeito do contorno dentro do multiagente**: S0×S1 (efeito do
   contorno de referência), S1×S2 (efeito de trocar referência por
   predição), S0×S2 (efeito líquido de adicionar contorno automático).
   Modelo, prompts e parâmetros constantes entre as três condições.

**Preservado, não resolvido aqui**: a regra para comparar as 5 métricas
binárias quando os subconjuntos com diagnóstico binário diferem entre
condições/sistemas continua pendente — não decidida por padrão do
código nesta entrega. `src/evaluation/evaluate.py` e
`src/evaluation/stage6_metrics` já separam cobertura binária, taxa de
saída válida (inclui indeterminado) e acurácia sobre o total (ver
§6.2), mas isso não resolve a comparação entre sistemas quando o
conjunto de casos com decisão binária não é o mesmo nos dois lados —
isso é uma decisão de protocolo, não de implementação.

A **política de tentativa única** já está preservada estruturalmente:
`11_multiagent_flow_documentation.md §2.3` reconfirma que não existe
`try/except` nem lógica de retry em nenhum agente, `_invoke` ou no loop
do pipeline — uma falha interrompe o caso, não é reexecutada
silenciosamente. Nada nesta entrega altera isso.

## 3. O que já está fechado e não é reaberto aqui

- Remoção do crítico/RAG do grafo ativo (A36) e RAG opt-in nos dois
  pontos de entrada via `--build-rag` — fechado, `12_entrypoint_rag_removal.md`.
- Rastreamento de ausência por agente (`status_abcd/menzies/spcl`,
  `AgentStatus`) e separação cobertura/consolidação parcial — fechado,
  A05.
- U-Net como baseline de segmentação confirmado; 5 folds fixos do
  ISIC2018 Task1 (seed `20261108`) congelados e reutilizáveis — fechado,
  `09`/stage 4.
- DeepLabV3+-ResNet34 (`output_stride=32`, ASPP (3,6,9)), métricas
  (Dice/IoU/Thresholded Jaccard/HD95 com `border_value=0`) e inicialização
  aleatória nas duas arquiteturas para a execução científica — fechado,
  `13_segmentation_deeplabv3plus_pilot.md` + decisões do pesquisador em
  `14_segmentation_nested_cv_consolidation.md §5`.
- Grade restrita da validação aninhada: só arquitetura varia; LR
  (1e-4, AdamW), lote (4, `drop_last=True`) fixos; teto de 20 épocas +
  early stopping (paciência 5) — fechado, `14 §5`. Implementação do
  early stopping e do modo de sondagem de rodada única ainda não
  entregue (ver `14 §6`, pendência separada desta matriz).

## 4. Dependências por sistema/modelo/papel/condição

| Sistema | Papel | Modelo/arquitetura | Status | Dependência bloqueante |
|---|---|---|---|---|
| CNN | Classificador único | EfficientNetV2-S, ImageNet init | Split BCN20000 gerado e verificado (seed `20261107`, `apenas_train`, val 20%); infraestrutura de treino smoke-testada | **Checkpoint definitivo não existe.** Uma tentativa real de treino completo (seed 42, 20 épocas, CPU) crashou por `STATUS_ACCESS_VIOLATION` antes da época 1 (provável exaustão de memória, não bug de código) — pausado pelo pesquisador. Nenhum checkpoint/histórico de treino existe hoje (`experiments/cnn_baseline/training/checkpoints/` vazio). Limiar de decisão também pendente (seleção por Youden só pode ocorrer depois de um treino real, exclusivamente no `val` do BCN20000, nunca no HAM10000). |
| Multiagente | Segmentação (entrada dos 3 agentes) | U-Net (padrão aprovado) OU DeepLabV3+-ResNet34 — seleção ainda não feita | U-Net e DeepLabV3+ implementados e testados isoladamente (pilotos); validação aninhada real (5×4) ainda não executada | Regra de seleção de arquitetura (Dice médio, empate <0,005 por taxa de falha ISIC depois HD95 depois U-Net) está **definida** (`13 §10`) mas não **aplicada** — depende do early stopping (não implementado, `14 §6`) e das 5×4 rodadas reais (não executadas). Sem isso, não há "segmentador final" para gerar as máscaras de S2. |
| Multiagente | ABCD / Menzies / SPCL / Sumarizador | 1 único `model_name` compartilhado pelos 4 papéis (`cfg["llm"]["model_name"]`, `config/config.yaml:46`, repassado igual a todos em `agents.py:build_graph`, linhas 583-595) | Dispatch por provedor implementado (`_is_ollama`/`_is_gemini`/OpenAI, `agents.py:33-67`) | Não há, hoje, suporte a modelo diferente por papel sem alterar código/CLI — se a matriz exigir isso, é trabalho novo, não configuração. |
| VLM direto | Classificador único | A definir — hoje não existe nenhum script `run_vlm_direct.py` ou equivalente no repositório (busca feita, nenhum arquivo encontrado) | Não implementado | É uma condição nova, não uma reutilização de código existente — precisa de uma implementação própria (chamada única ao VLM com a imagem original e uma instrução de tarefa equivalente à do multiagente, sem os 3 agentes/sumarizador). |

## 5. Levantamento de modelos — o que o código/config/dados mostram

**Distinção mantida em todo este levantamento**: "configuração histórica
encontrada" (nomes de modelo que aparecem em arquivos de resultado ou no
`config.yaml`) ≠ "disponibilidade atual verificada" (se esse modelo
pode ser chamado agora, com as credenciais/pacotes presentes neste
ambiente).

### 5.1 Configuração histórica encontrada

- `config/config.yaml:46`: `llm.model_name: "gpt-5.4-nano"` — valor
  atualmente commitado como padrão do pipeline.
- Arquivos de resultado datados (nomes de arquivo, não conteúdo
  inventado): `llm_result_gpt-5.4-nano_20260606.csv`,
  `llm_result_gpt-5.4-nano_20260609.json`,
  `llm_result_gemini-3.1-flash-lite_20260606.csv`,
  `llm_result_gemini-3.1-flash-lite_20260609.json` — dois identificadores
  de modelo usados historicamente para qualificação: `gpt-5.4-nano`
  (provedor OpenAI, via `ChatOpenAI`, ramo `else` de `_build_llm`) e
  `gemini-3.1-flash-lite` (provedor Google, via `_is_gemini` →
  `ChatGoogleGenerativeAI`, `agents.py:38-40,60-65`).
- Conteúdo real do JSON `_20260609` inspecionado diretamente (não só o
  nome do arquivo): confirma que o mesmo `model_name` é usado para os 4
  papéis (campos `diagnosis_abcd`/`diagnosis_menzies`/`diagnosis_spcl`/
  `validation` populados na mesma entrada), consistente com o código
  atual.
- `results.csv` e `results_stefani.csv` (raiz do repo): mesmo schema de
  colunas (`validation.diagnostico,estimated_Value,real_value`), **sem**
  coluna `final_report`. Isso fecha a pendência **P014** registrada na
  entrega anterior: `results_stefani.csv` foi aberto nesta sessão — não
  tem o campo `final_report` em nenhuma linha. Resultado registrado como
  "execução histórica do crítico **não comprovada** neste arquivo" (a
  ausência da coluna é consistente com "não rodou", mas — como já
  registrado em `11_multiagent_flow_documentation.md` para `results.csv`
  — não é prova definitiva isolada, pois também poderia refletir um
  script de exportação que omite o campo). Nenhum arquivo de resultado
  pendente de inspeção permanece.
- Nenhuma menção a "Qwen" em nenhum arquivo de código, configuração,
  `requirements.txt` ou dado do repositório (busca direta, zero
  ocorrências fora de `.venv`).

### 5.2 Disponibilidade atual verificada

- Pacotes instalados no `.venv` confirmados por inspeção direta de
  `site-packages`: `langchain-openai`, `langchain-google-genai`,
  `langchain-ollama==1.1.0`, `ollama==0.6.2` (cliente Python do Ollama).
  `requirements.txt` lista as três integrações LangChain de provedor
  (OpenAI/Gemini/Ollama) — nenhuma dependência de SDK Qwen nativo (ex.
  `dashscope`) está presente.
- Dispatch de provedor em `agents.py:33-67` reconhece exatamente três
  famílias: nome contendo `:` → Ollama local; prefixo `gemini` → Google;
  qualquer outro nome → OpenAI. **Não existe um quarto ramo para Qwen
  nativo** (ex. via API DashScope/Alibaba Cloud) — se a intenção for usar
  a API comercial do Qwen diretamente, isso é código novo, não só
  configuração.
- Verificação externa real feita nesta sessão (consulta à biblioteca
  oficial do Ollama, não memória): os seguintes modelos Qwen-VL existem
  hoje como tags publicadas no Ollama —
  `qwen2.5vl:{3b,7b,32b,72b}` (requer Ollama ≥ 0.7.0) e
  `qwen3-vl:{2b,4b,8b,30b,32b,235b,235b-cloud}` (requer Ollama ≥ 0.12.7).
  Ambas as famílias declaram suporte a entrada de imagem.
- **Não verificável nesta sessão** (exigiria executar comandos no
  ambiente do pesquisador, fora do permitido): versão do Ollama
  instalada localmente, se algum modelo Qwen-VL já foi puxado
  (`ollama list`), e se `ChatOllama` (`langchain_ollama==1.1.0`) envia
  corretamente uma imagem para esses modelos no fluxo multimodal usado
  por `_image_message` (`agents.py:92-108`). Ver verificação proposta no
  §6.3.

## 6. O que precisa de decisão antes da próxima implementação

### 6.1 Gap de schema para a comparação A34 (achado novo, não resolvido)

A saída real do multiagente hoje (`SummaryAnswer.diagnostico`,
`schemas.py:61-77`) é texto livre, sem Enum/campo estruturado de
decisão — a classificação em
positivo/negativo/indeterminado/ausente/não-reconhecido só existe
depois, em `src/evaluation/case_status.classify_case_status` (consumida
por `evaluate.py`/`stage6_metrics`), nunca no momento da geração. Para a
comparação A34 ("a saída deve permitir o mesmo tratamento de decisão
binária, indeterminação, recusa e falha"), **o VLM direto precisa
produzir uma saída compatível com esse mesmo classificador pós-hoc**, ou
o classificador precisa ser estendido para aceitar o formato do VLM
direto. Isso não está implementado nem decidido — registrado aqui como
pendência nova, não presente nas entregas anteriores.

### 6.2 Indicadores já implementados, reaproveitáveis sem mudança

`compute_metrics`/`stage6_metrics` (stage 6, `evaluate.py:82-179`) já
implementam exatamente a separação pedida: 5 métricas binárias (só
positivo/negativo), cobertura binária, taxa de saída válida (inclui
indeterminado) e acurácia sobre o total — qualquer métrica sem
denominador retorna `None`, nunca `0.0`. Isso cobre os quatro
indicadores da matriz sem precisar de código novo, **desde que** a saída
de cada condição já tenha passado por `classify_case_status` (ver 6.1
para o VLM direto).

### 6.3 Verificação proposta para Qwen-VL (para o pesquisador rodar, não autônoma)

Script sugerido, não escrito nesta entrega (fora do escopo pedido —
"preparar uma verificação local para eu rodar" seria a próxima tarefa, se
aprovada): confirmar `ollama --version` ≥ requisito da tag escolhida,
`ollama pull <tag>`, e um teste manual mínimo de `ChatOllama(model=
"<tag>").invoke(...)` com uma mensagem multimodal (mesmo formato de
`_image_message`) para confirmar que a imagem é de fato processada (não
só o texto) antes de qualquer execução real da matriz.

### 6.4 Forma de desenho do contorno (S1/S2)

Não encontrada nenhuma implementação de "desenhar contorno sobre a
imagem original a partir de uma máscara" em `src/` nesta busca — nem no
código atual do multiagente, nem nos módulos de segmentação. Precisa ser
implementada (função única, reutilizada idêntica entre S1 e S2, conforme
já exigido pela matriz aprovada) — registrado como trabalho pendente,
não como decisão.

## 7. Recomendações (separadas do que está aprovado — sujeitas a aprovação)

**Recomendação 1 — configuração multiagente de referência para a
comparação principal**: usar `gpt-5.4-nano` (já é o padrão commitado em
`config/config.yaml`, já tem o maior volume de dados históricos
reais — `llm_result_gpt-5.4-nano_20260609.json`, 447 KB vs. 265 KB do
Gemini — e já percorreu os dois provedores suportados sem mudança de
código). Esta é uma recomendação **operacional** (menor atrito, maior
precedente de execução real), não uma afirmação sobre qual modelo
responde melhor — não há dados de desempenho comparativo entre os dois
nesta base que permitam recomendar por qualidade de resposta. Alternativa
igualmente verificável: `gemini-3.1-flash-lite`, mesmo caminho de código
(`_is_gemini`), menos volume histórico. **O que falta confirmar**: que
`gpt-5.4-nano` ainda está disponível/acessível com a chave atual do
pesquisador (não verificável nesta sessão) e se esse é de fato o
identificador real do modelo que o pesquisador pretende cobrar na
qualificação (nome não reconhecido como um ID OpenAI publicamente
documentado até o conhecimento desta sessão — pode ser um alias
interno/placeholder; se for, precisa ser substituído pelo ID real antes
da execução).

**Recomendação 2 — versão do Qwen-VL**: `qwen2.5vl:7b` via Ollama local
(rota já suportada pelo código existente, `_is_ollama`, sem mudança em
`agents.py`) como primeira opção — família mais madura, requisito de
versão do Ollama mais baixo (≥ 0.7.0), tamanho (6 GB) compatível com
execução local sem GPU dedicada de grande porte. Alternativa:
`qwen3-vl:8b` (mais novo, contexto maior, requer Ollama ≥ 0.12.7) se o
pesquisador preferir a geração mais recente. **Ambas as opções herdam a
mesma limitação do caminho Ollama já documentada no código**
(`_invoke`, `agents.py:73-89`): sem `with_structured_output`, a resposta
vem como texto livre, não como objeto Pydantic validado — isso afeta
diretamente o gap do §6.1 (A34) e precisa de uma estratégia de parsing
decidida antes da execução, não só a escolha da tag do modelo.

## 8. O que esta entrega NÃO fez

Não foi implementado nenhum script de execução, parsing, geração de
contorno ou chamada a modelo. Não foi alterado nenhum arquivo de código
existente. Não foi decidida a arquitetura de segmentação final, o
checkpoint/limiar da CNN, a configuração definitiva do multiagente, a
versão final do Qwen-VL, nem a regra de comparação das 5 métricas sob
subconjuntos binários distintos — todos permanecem como pendências
explícitas de decisão do pesquisador, listadas na seção 6.
