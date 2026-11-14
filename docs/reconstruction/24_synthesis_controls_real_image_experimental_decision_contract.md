# 24 — Controles de síntese com imagem real usando o contrato de decisão experimental

Resumo para o planejador central: segunda entrega do protótipo descrito
em `docs/reconstruction/23_explicit_decision_contract_prototype.md`,
pedida verbatim pelo pesquisador após validar o parser (8/8 PASS na
verificação manual do doc 23): "repetir os quatro controles de síntese
com imagem real, uma chamada por cenário para Qwen3-VL e MedGemma, sem
retry".

**Status: implementado, aguardando execução do pesquisador.** Nada
deste script foi executado por este agente (nem `python`, nem
`pytest`, nem git, nem `ollama`) — `feedback_no_autonomous_execution`.

Esta rodada é uma decisão SEPARADA (confirmado verbatim pelo
pesquisador na entrega anterior) e **não aprova automaticamente** a
troca do critério de avanço do Estágio 1 — `scripts/run_local_vlm_pilot.py`
(gate oficial via `classify_case_status_v2`) não foi modificado nem é
lido por este script novo.

## 1. O que foi construído

### 1.1 Caminho de chamada experimental (novo, paralelo, isolado)

`src/multiagent/experimental_synthesis_control_real_image.py`:

- `run_experimental_synthesis_control_real_image(model_name,
  scenario_name, image, timeout_s)` — UMA chamada (sem retry) ao
  sumarizador, usando:
  - os MESMOS pareceres fixos de `vd.FIXED_SYNTHESIS_SCENARIOS`
    (`src/multiagent/vlm_direct.py`, intocado);
  - a instrução de tarefa EXPERIMENTAL do sumarizador
    (`build_experimental_summary_task_instruction`, doc 23) em vez do
    prompt oficial (`SummaryAgent.summarize`, não chamado aqui);
  - classificação SÓ pelo contrato de decisão explícito
    (`extract_decisao_final`, doc 23) — `classify_case_status`/
    `classify_case_status_v2` NÃO são importados nem chamados neste
    módulo.
- Reusa, sem modificar, as primitivas de infraestrutura já existentes
  em `src/multiagent/vlm_direct.py`: `call_with_timeout` (timeout
  120s, sem retry), `classify_call_outcome` (recusa/falha técnica/sem
  resposta), `capture_memory_snapshot`, `TIMEOUT_S` — mesma convenção
  do piloto real, não uma segunda implementação divergente.
- `ExperimentalDecisionCallRecord` — registro paralelo a `vd.CallRecord`,
  com campos do contrato de decisão (`decision_status`,
  `decisao_final`, `matched_raw_values`,
  `template_extraction_status`) em vez de
  `classified_status`/`classified_status_v2`.

### 1.2 Script driver (novo)

`scripts/run_synthesis_controls_real_image_experimental_decision_contract.py`:

- Reusa a MESMA imagem real já selecionada/verificada por
  `scripts/run_synthesis_control_real_image_selection.py` (mesmo
  registry lido por `scripts/run_synthesis_controls_real_image.py`,
  NÃO modificado) — nenhuma nova seleção de imagem é feita; a mesma
  pré-checagem sem inferência (hash do manifesto congelado, exclusão
  por ambas as chaves, imagem carrega) roda antes de qualquer chamada
  real, e aborta sem chamar nenhum modelo se falhar.
- 4 cenários x 2 modelos (`qwen3-vl:8b-instruct-q4_K_M`,
  `medgemma1.5:4b-it-q4_K_M`) = 8 chamadas, uma por par, sem
  repetição/retry.
- Escreve em
  `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/`
  — NUNCA em `experiments/multiagent/synthesis_controls_real_image/`
  (preserva os 3 artefatos históricos reais já lá, incluindo a
  execução anterior com o sumarizador oficial).

**Comando exato para o pesquisador rodar** (requer Ollama já em
execução, os 2 modelos já baixados, e o registry da imagem real já
existente — mesmo pré-requisito de `run_synthesis_controls_real_image.py`):

```
python scripts/run_synthesis_controls_real_image_experimental_decision_contract.py \
    --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
```

**Saídas esperadas:** `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/results.json`,
`report.md`, e um `call_log.jsonl` por modelo — 8 registros no total
(4 cenários x 2 modelos), cada um com `call_status`,
`template_extraction_status`, `decision_status`, `decisao_final` e o
texto bruto verbatim. Nenhum campo é preenchido automaticamente como
"aprovado"/"reprovado" — a leitura humana dos 8 textos brutos e das
decisões declaradas é o objetivo desta rodada, não um gate automático.

## 2. O que NÃO foi feito nesta tarefa

- Nenhuma execução real (sem Ollama, sem chamada a qwen3-vl/medgemma).
- Nenhuma mudança em `src/agents/agents.py`,
  `src/multiagent/vlm_direct.py`,
  `src/evaluation/case_status.py`/`case_status_v2.py` ou
  `scripts/run_local_vlm_pilot.py` (critério de avanço do Estágio 1).
- Nenhuma comparação automática com `classified_status_v2` desta
  rodada (poderia ser feita numa tarefa futura, comparando os 2
  arquivos `results.json` lado a lado — não pedido nesta entrega).
- Qualquer decisão de trocar o critério de avanço do Estágio 1 pelo
  contrato explícito permanece pendente, a cargo exclusivo do
  pesquisador, após revisão dos resultados reais desta rodada.

## 3. Arquivos alterados/criados nesta entrega

- `src/multiagent/experimental_synthesis_control_real_image.py` (novo)
- `scripts/run_synthesis_controls_real_image_experimental_decision_contract.py` (novo)
- `docs/reconstruction/24_synthesis_controls_real_image_experimental_decision_contract.md` (novo, este arquivo)

Nenhum arquivo existente foi modificado.
