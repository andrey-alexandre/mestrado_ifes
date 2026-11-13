# 20 — Piloto local de VLMs via Ollama (Etapa 5, segunda entrega)

STATUS: implementado, aguardando execução do pesquisador. Nada deste
pacote foi executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) — nenhum `python`, `pytest`, `git`
ou `ollama` foi chamado. Tudo abaixo é entregue como script para o
pesquisador rodar na própria máquina (Ollama + modelos locais).

Zero chamadas a OpenAI/Gemini nesta entrega — só Ollama. O módulo
`src/multiagent/presentation.py` (primeira entrega da Etapa 5) não foi
alterado; só importado/reusado.

## 1. Arquivos criados/alterados

Criados:
- `scripts/run_pilot12_sample.py` — seleciona as 12 lesões fora da amostra congelada.
- `scripts/run_s2_mask_generation_pilot12.py` — gera máscaras S2 para as 12 (mesmo checkpoint/limiar congelados, nunca lê a amostra de 3.000).
- `scripts/run_pilot12_image_preparation.py` — constrói S0/S1/S2 para as 12 (reusa `src/multiagent/presentation.py`).
- `src/multiagent/vlm_direct.py` — camada de chamada a VLM (VLM-direto + multiagente-sobre-Ollama), controles, abstenção, classificação A34.
- `scripts/run_local_vlm_pilot.py` — driver único, uma execução por tag de modelo.
- `tests/multiagent/test_vlm_direct.py` — testes com Ollama/LLM mockados (sem rede).

Alterados:
- `requirements.txt` — duas dependências novas documentadas (`psutil`, `ollama`).

Não alterados (só importados/reusados): `src/multiagent/presentation.py`, `src/agents/agents.py`, `src/agents/schemas.py`, `src/evaluation/case_status.py`, `src/data_audit/sampling.py`, `scripts/run_s2_mask_generation.py`.

## 2. Sequência exata de comandos

Diretório de trabalho para todos: `C:\Projetos\IFES\mestrado_ifes`.

```
# 0. Dependências novas
pip install psutil ollama

# 1. Amostragem dos 12 casos fora da amostra congelada
python scripts/run_pilot12_sample.py

# 2. Máscaras S2 para os 12 (mesmo checkpoint/limiar congelados)
python scripts/run_s2_mask_generation_pilot12.py --device mps \
    --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
# (ajuste --device para cpu/cuda/mps conforme a máquina; --ham10000-images-dir já é o default no Windows)

# 3. Imagens de apresentação S0/S1/S2 para os 12
python scripts/run_pilot12_image_preparation.py

# 4. Piloto de VLM -- uma vez por modelo (Ollama já rodando localmente, modelos já baixados)
python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M
python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M

# 4b. SOMENTE SE necessário (ver seção 7 abaixo): rodar o modelo de contingência
python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M
```

## 3. Comando de testes

```
python -m pytest tests/multiagent/test_vlm_direct.py -v
```

(Os testes já existentes `tests/multiagent/test_s0_s1_s2_preparation.py` e `tests/multiagent/test_pilot_logic.py` não são alterados por esta entrega; rodar `python -m pytest tests/multiagent -v` executa os três arquivos juntos.)

## 4. Arquivos/relatórios a ler de volta

Em ordem de execução:

1. `experiments/multiagent/pilot12_manifest.csv` + `experiments/multiagent/pilot12_sampling_report.md` + `experiments/multiagent/pilot12_sampling_results.json` — confirma as 12 lesões, zero sobreposição por `lesion_id` E `image_id`, e os 4 designados para S0/S1.
2. `experiments/multiagent/pilot12_s2_mask_generation_report.md` + `..._results.json` — confirma geração das máscaras S2 dos 12 (OK/vazia/falha), e que a amostra de 3.000 não foi lida.
3. `experiments/multiagent/pilot12_image_preparation_report` (CSV `pilot12_image_preparation_manifest.csv` + `..._results.json`) — confirma disponibilidade de S0/S1/S2 por caso.
4. Por modelo, em `experiments/multiagent/local_vlm_pilot/<model_tag>/`:
   - `call_log.jsonl` — log por chamada (todas, incl. controles): modelo, papel, condição, lesion_id/image_id, timestamp, `elapsed_s`, memória/swap antes+depois, texto bruto, status da chamada, causa de falha, status classificado.
   - `summary.json` — contagens planejadas/executadas/não-executadas por papel×condição, taxa de validade na 1ª tentativa, cobertura binária, rastro dos contadores de abstenção, resultado dos controles, crescimento de swap.
   - `report.md` — versão legível do `summary.json`, incluindo se a execução parou antes da matriz principal (controle reprovado) e por quê.

## 5. Interpretações de design que precisam de confirmação explícita do pesquisador ANTES de qualquer execução real

1. **Texto exato da instrução de tarefa do VLM-direto** (`vd.VLM_DIRECT_TASK_INSTRUCTION`, em `src/multiagent/vlm_direct.py`) — pede explicitamente o mesmo vocabulário de 3 categorias do sumarizador ("Suspeita de Melanoma"/"Lesão Benigna"/"Lesão Indeterminada"), decisão necessária para a comparação A34 ser justa, mas o texto de justificativa/estilo em si é uma escolha deste assistente, não uma transcrição de spec prévia.
2. **Desenho dos controles de envio de imagem** (`vd.IMAGE_CONTROL_SPECS`, `vd.IMAGE_CONTROL_PROMPT`) — 3 imagens sintéticas (quadrado colorido em quadrante fixo sobre fundo escuro) e o prompt que pede quadrante+cor. Critério de aprovação (`vd.evaluate_image_sending_controls`): cada resposta identifica corretamente SUA própria imagem E as respostas não são todas idênticas entre si. Não havia especificação prévia no projeto para isto.
3. **Os 4 inputs fixos dos controles de síntese** (`vd.FIXED_SYNTHESIS_SCENARIOS`) — pareceres ABCD/Menzies/SPCL sintéticos fixos para os cenários unânime-positivo, unânime-negativo, conflitante e um-parecer-ausente. O conteúdo textual exato de cada parecer fictício é uma escolha deste assistente para produzir os 4 padrões pedidos, não uma transcrição.
4. **Heurística de "achado clínico novo"** nos controles de síntese (`vd._vocabulary_terms_not_in_inputs`, vocabulário fechado de termos dos 3 prompts reais) — é uma checagem lexical (substring), não semântica; não prova ausência real de alucinação clínica, só torna visível um sinal objetivo e reproduzível para revisão manual.
5. **Heurística de detecção de recusa explícita** (`vd._REFUSAL_MARKERS`) — lista curta de frases em português/inglês; pode não cobrir todas as formas reais de recusa de um VLM local.
6. **Definição de "crescimento sustentado de swap"** (`vd.detect_sustained_swap_growth`) — adotada como sequência monotônica não-decrescente de amostras de swap ao longo de todas as chamadas, acima de um piso (`floor_bytes`, default 0 — valor neutro, não uma recomendação). Marcada no próprio resultado (`definition_needs_researcher_signoff: True`) — não deve ser tratada como critério de aprovação automática sem aval.
7. **Regra dos "4 de 12" designados para multiagente S0/S1** (`scripts/run_pilot12_sample.py`) — os 4 primeiros `lesion_id` em ordem alfabética entre os 12 sorteados. Determinística e reproduzível, mas é uma escolha de convenção, não uma regra ditada previamente.
8. **Semente do piloto de 12** (`20270203`, distinta de `20261106`) — nova, escolhida só para ser claramente distinta e documentada; não há significado além disso.
9. **Checagem de "papel sem nenhum parecer conclusivo"** (passo 8/relatório) — restrita, na implementação, aos papéis `vlm_direto` e `summary` (únicos com um campo de diagnóstico categorizável por `classify_case_status`); para ABCD/Menzies/SPCL essa checagem seria sempre "zero" por construção (eles não emitem "positivo/negativo/indeterminado" diretamente) e foi deliberadamente omitida do relatório para não gerar um falso alarme — mas isso é uma leitura deste assistente da intenção do pesquisador, não uma confirmação.
10. **Ordem de execução dos 3 agentes diagnósticos dentro de cada caso** — sequencial (ABCD, depois Menzies, depois SPCL) em vez de paralela, já que os 3 são logicamente independentes entre si no grafo real (fan-out da mesma `segmentation`); a sequencialização aqui é só para simplificar o driver manual (ver seção 6), sem efeito no conteúdo de cada parecer.

## 6. Por que foi necessário um driver manual (não dá para reusar `build_graph()` compilado caso a caso)

Confirmado por leitura de `src/agents/agents.py::build_graph`: o grafo compilado tem uma ÚNICA aresta de entrada, `START -> "segmentation"`, e o nó `"segmentation"` sempre recalcula `seg_image_data` a partir de `state.image_path`/`state.seg_path` (modo "model" ou "precomputed" do próprio `agents.py`) — não há um jeito de invocar o grafo compilado "pulando" esse nó para injetar um `seg_image_data` já pronto de uma condição S0/S1/S2 pré-computada por `src/multiagent/presentation.py`. Portanto `src/multiagent/vlm_direct.py` NÃO modifica `agents.py`/`build_graph()`; instancia diretamente `ABCDDiagnosticAgent`/`MenziesDiagnosticAgent`/`SPCLDiagnosticAgent`/`SummaryAgent` (as mesmas classes, os mesmos prompts internos, a mesma função `_invoke`/`_image_message`) com um `GraphState` real já populado com o `seg_image_data` da condição sob teste — ver `vd.run_multiagent_role_call`/`vd.build_graph_state`.

## 7. Lógica de seleção de modelo (Passo 7)

`scripts/run_local_vlm_pilot.py` roda UM modelo por execução (`--model-tag`). Depois de rodar `qwen3-vl:8b-instruct-q4_K_M` e `medgemma1.5:4b-it-q4_K_M` (cada um grava seu próprio `summary.json`/`report.md` em `experiments/multiagent/local_vlm_pilot/<tag>/`):

- Se **ambos** passaram nos controles (`summary.json["controls"]["image_sending"]["overall_pass"]` e cada `summary.json["controls"]["synthesis"][i]["passed"]` = `True` nos dois `summary.json`, e `stopped_before_main_matrix` = `False` em ambos): os dois seguem para a matriz completa (já rodada nessa mesma execução) — nenhum é "primário", o modelo de contingência (`qwen2.5vl:7b-q4_K_M`) NÃO é necessário.
- Se **só um** passou: rode uma TERCEIRA vez, `python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M`, seguindo o mesmo procedimento de controles+matriz.
- Se **nenhum** passou: nenhum modelo tem matriz executada por este script (cada execução já parou sozinha na etapa de controle que reprovou, ver `report.md` de cada um); decidir os próximos passos é do pesquisador, não automatizado aqui.

Este branching é decidido pelo pesquisador entre execuções, lendo os `summary.json`/`report.md` reais — o assistente que escreveu este pacote não pode executar nada para descobrir ao vivo qual modelo passa.

## 8. Decisões já fechadas, implementadas sem re-derivação

- Timeout de 120s por chamada, sem retry (`vd.TIMEOUT_S`, `vd.call_with_timeout`) — nota de implementação: a thread da chamada que excedeu o timeout continua rodando em segundo plano (limitação do `ThreadPoolExecutor`, documentada em `call_with_timeout`); o resultado atrasado nunca é usado.
- 3 contadores de abstenção por papel (falha técnica / sem resposta / recusa explícita), reiniciados em qualquer sucesso incluindo diagnóstico indeterminado válido (`vd.AbstentionCounters`/`vd.AbstentionTracker`).
- Casos inalcançáveis por pausa marcados `"nao_executado"` (`vd.STATUS_NOT_EXECUTED`, `vd.not_executed_record`) — nunca omitidos.
- Matriz principal com exatamente 92 chamadas planejadas, verificado por `assert` em `vd.build_planned_calls` (não só no docstring).
- A34 (`src.evaluation.case_status.classify_case_status`) reusada idêntica para VLM-direto e multiagente-sumarizador, nunca uma classificação paralela.
- Texto bruto de toda resposta preservado verbatim no `call_log.jsonl`, mesmo quando classificado como `nao_reconhecido`/`ausente`.
