# 21 — Fechamento da correção dos controles de síntese + revisão de fidelidade (Etapa 5)

Resumo para o planejador central: fechamento da tarefa "corrigir os quatro
controles de síntese do Estágio 1 e preparar sua execução manual"
(`feature/inspection`, commits `a3c9f48` em diante), com resultado real já
executado pelo pesquisador e revisão de fidelidade concluída.

## 1. Problema que originou a tarefa

Na 3ª execução real do Estágio 1 do piloto local
(`qwen3-vl:8b-instruct-q4_K_M`), o controle de síntese `um_parecer_ausente`
falhou porque o placeholder usado até então (`_fixed_control_image()`, 64×64
cinza sólido, sem lesão) foi corretamente identificado pelo modelo como
vazio — ele recusou aplicar os algoritmos diagnósticos. O sumarizador real
(`SummaryAgent.summarize`) sempre envia uma imagem dermatoscópica real; era
o controle que quebrava essa fidelidade, não o modelo nem o parser.

## 2. Correção aplicada

- Imagem placeholder substituída, **somente nos 4 controles de síntese**,
  por uma imagem HAM10000 real (`lesion_id=HAM_0000124`,
  `image_id=ISIC_0025091`, `dx=nv`), selecionada e verificada como externa
  à amostra congelada de 3.000 lesões por `lesion_id` **e** `image_id`
  (nova semente `20270901`, independente do piloto de 12 lesões).
  Registro completo: `experiments/multiagent/synthesis_control_real_image_registry.json`.
- Implementação aditiva (`build_synthesis_control_state_real_image`/
  `run_synthesis_control_real_image` em `src/multiagent/vlm_direct.py`) —
  zero linhas alteradas nas funções originais; qualquer reexecução futura
  do Estágio 1 padrão via `run_local_vlm_pilot.py` permanece idêntica.
- `src/evaluation/case_status_v2.py` (parser A34 v2, congelado), o gate do
  Estágio 1 e os prompts reais de `src/agents/agents.py` **não foram
  tocados** nesta tarefa.
- Executor novo e isolado (`scripts/run_synthesis_controls_real_image.py`):
  exatamente 8 chamadas (4 cenários × 2 modelos, sem retry), gravadas em
  `experiments/multiagent/synthesis_controls_real_image/` — artefatos das
  3 execuções históricas anteriores preservados intactos.

## 3. Resultado real (executado pelo pesquisador, 2026-11-14)

Todas as 8 chamadas completaram com `call_status=ok` — **nenhuma recusa
por imagem vazia**, confirmando que o problema era estrutural (placeholder),
não o modelo nem o parser.

| Modelo | Cenário | Categoria esperada | v2 (oficial, congelado) | Bateu? |
|---|---|---|---|---|
| qwen3-vl | unanime_positivo | positivo | `nao_reconhecido` | Não (vocabulário) |
| qwen3-vl | unanime_negativo | negativo | negativo | Sim |
| qwen3-vl | conflitante | (sem gabarito único, por desenho) | positivo | — |
| qwen3-vl | um_parecer_ausente | negativo | negativo | Sim |
| medgemma1.5 | unanime_positivo | positivo | `nao_reconhecido` | Não (vocabulário) |
| medgemma1.5 | unanime_negativo | negativo | negativo | Sim |
| medgemma1.5 | conflitante | (sem gabarito único, por desenho) | negativo | — (ver §5) |
| medgemma1.5 | um_parecer_ausente | negativo | negativo | Sim |

Os dois `nao_reconhecido` foram registrados **sem** ampliar o vocabulário do
v2 nem recorrer ao v1 como desempate, por instrução explícita desta tarefa.

## 4. Revisão de fidelidade (concluída, preenchida em `results.json`/`report.md`)

- **Autoria preservada em todos os 8 casos** — achados sempre atribuídos ao
  algoritmo correto (ABCD/Menzies/SPCL), sem trocas.
- **Nenhum parecer fictício inventado** em nenhum caso.
- **Distinção "não avaliado" vs "ausente" preservada corretamente** nos
  casos aplicáveis — destaque para `qwen3-vl/conflitante`, que lista
  literalmente "Vasos atípicos: Não avaliado" e atribui a suspeita
  especificamente ao Menzies (não alega consenso), preservando a
  divergência real entre os 3 algoritmos — considerada a melhor resposta
  das 8 em fidelidade.
- Um alerta da heurística lexical de triagem
  (`likely_misrepresented_absence_as_lesion_lacks_feature=True`, em
  `qwen3-vl/um_parecer_ausente` e no mesmo cenário do medgemma) foi lido
  manualmente no texto e avaliado como **falso positivo** — a frase
  sinalizada descreve um achado visual real da lesão, não confunde
  ausência de parecer com ausência de característica.
- Dois casos marcados `faithful_but_unrecognized_by_v2=true`
  (`qwen3-vl/unanime_positivo`, `medgemma1.5/unanime_positivo`) — conteúdo
  claramente fiel e positivo, mas o veredito final usa uma variante fora do
  vocabulário fechado do v2 ("alto risco para melanoma" /
  "suspeita **por** melanoma", em vez de "suspeita **de** melanoma").
  Registrados, não corrigidos nesta tarefa.

## 5. Achado técnico novo — vazamento de template no medgemma1.5 (para decisão do planejador)

As 4 respostas de `medgemma1.5:4b-it-q4_K_M` vêm com um bloco de
cadeia-de-pensamento interna entre marcadores `<unused94>thought` e
`<unused95>` **concatenado antes** da resposta final, sem filtragem pelo
pipeline atual (`ChatOllama`/`_invoke`). Em 3 dos 4 casos isso não
comprometeu a classificação (a resposta final é curta e consistente com o
raciocínio). Mas em **`medgemma1.5/conflitante`**, a resposta final
preserva o conflito sem resolver para um lado — e o v2 classificou como
`negativo` mesmo assim. Como o bloco de pensamento (bem mais longo que a
resposta real) menciona "benigno" várias vezes, é plausível que a
heurística de "metade final do documento" do v2 tenha capturado texto do
*raciocínio*, não da resposta efetivamente entregue. Isto não é um caso de
"fiel mas não reconhecido" — é um caso de **classificação do v2
potencialmente não confiável** para texto com esse vazamento de template.

Fora do escopo desta tarefa (parser/prompts congelados) — decisão proposta
ao planejador: avaliar se vale a pena pré-processar/descartar o bloco
`<unused94>...<unused95>` antes de qualquer classificação automática sobre
saídas do medgemma1.5, ou se a matriz de 12 lesões deve tratar esse modelo
com um cuidado adicional de parsing.

## 6. Status e escopo preservado

- Matriz das 12 lesões (`experiments/multiagent/pilot12_manifest.csv`):
  **fora desta execução**, como instruído — não tocada, não usada para
  escolher nada aqui.
- Critério de ≥95% de respostas válidas: pertence ao piloto de 12 lesões,
  **não estimado** a partir destes 8 controles.
- `src/evaluation/case_status_v2.py`, o gate do Estágio 1
  (`scripts/run_local_vlm_pilot.py`) e os prompts reais de `agents.py`:
  confirmados intocados.

**Pronto para decisão do planejador central**: próximos passos possíveis —
(a) tratar o vazamento de template do medgemma1.5 antes de prosseguir com
esse modelo; (b) decidir se os controles de síntese corrigidos autorizam
reabrir o Estágio 1 completo (`run_local_vlm_pilot.py`) com a imagem real
em vez do placeholder (mudança não feita nesta tarefa, que ficou
deliberadamente restrita aos 4 controles isolados); (c) qualquer outra
priorização.
