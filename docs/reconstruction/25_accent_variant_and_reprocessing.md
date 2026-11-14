# 25 — Variante acentuada de `nao_melanoma` + reprocessamento offline

Resumo para o planejador central: resposta à 1ª execução real dos 4
controles de síntese com imagem real via contrato de decisão
experimental (doc 24,
`experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/results.json`,
8 chamadas reais, todas `call_status: "ok"`).

**Achado real:** 6/8 chamadas classificadas `decision_status: "ok"`
(qwen3-vl 4/4; medgemma1.5 2/4). As 2 chamadas restantes do
medgemma1.5 (`unanime_negativo`, `conflitante`) vieram
`invalid_value`, porque o modelo escreveu `decisao_final: não_melanoma`
(grafia correta em português, com "ã") em vez do token ASCII
`nao_melanoma` pedido literalmente na instrução experimental
(`matched_raw_values: ["não_melanoma"]` nos dois casos). Não é erro do
parser — é a comparação estritamente ASCII original (desenho
documentado no doc 23) rejeitando uma decisão inequívoca e
corretamente grafada.

**Decisão do pesquisador (verbatim desta tarefa):** no protótipo
experimental, aceitar `"não_melanoma"` como variante EXPLÍCITA de
`"nao_melanoma"`, só dentro do campo `decisao_final`, aplicada
IDENTICAMENTE ao VLM-direto e ao sumarizador (mesmo parser). Manter os
demais valores estritos e preservar a rejeição de campo ausente/
repetido/inválido. Não alterar prompts, parser v2 nem o critério
oficial de avanço. Preservar o resultado original da 1ª execução
(2/8 `invalid_value`) e preparar reprocessamento offline das 8
respostas já salvas, sem novas inferências.

**Status: implementado, aguardando execução do pesquisador.** Nada
deste reprocessamento foi executado por este agente
(`feedback_no_autonomous_execution`).

## 1. O que foi alterado

### 1.1 Parser (único arquivo de comportamento alterado)

`src/evaluation/explicit_decision_contract.py`:

- Nova tabela fechada e só-observada `_VALUE_VARIANTS` (e seu mapa
  inverso `_VARIANT_TO_CANONICAL`): `DECISAO_NAO_MELANOMA` ganha a
  variante literal `"não_melanoma"`, observada nas 2 chamadas reais
  citadas acima. `"melanoma"`/`"indeterminado"` não têm variante
  conhecida (nenhuma observada ainda).
- `extract_decisao_final` agora resolve o valor capturado (após
  `.strip().lower()`) via `_VARIANT_TO_CANONICAL.get(...)` em vez de
  checar membership direto em `ALL_DECISIONS` — `decisao_final`
  devolvido é SEMPRE o valor CANÔNICO (`"nao_melanoma"`), nunca a
  variante literal, mesmo quando a variante foi a forma aceita.
  `matched_raw_values` continua preservando a string literal
  capturada (`"não_melanoma"`) para auditoria.
- Docstring do módulo documenta a exceção explicitamente como FECHADA
  — qualquer outra grafia (de qualquer um dos 3 valores) que não
  esteja na tabela continua `invalid_value`. Isto NÃO é uma
  normalização geral de acentos (ex.: via `unicodedata`); é uma
  entrada literal única, no mesmo espírito do vocabulário fechado de
  `_KNOWN_TEMPLATE_DELIMITER_PAIRS` em `response_template_extraction.py`.
- A EXCEÇÃO se aplica automaticamente aos dois papéis (VLM-direto e
  sumarizador) porque ambos chamam a mesma função
  `extract_decisao_final` — nenhuma duplicação de lógica entre papéis.
- `src/multiagent/experimental_decision_contract.py` (as duas
  instruções de tarefa experimentais) **não foi modificado** — ainda
  pede literalmente o token ASCII `nao_melanoma`; a variante aceita é
  uma rede de segurança do PARSER para uma grafia real observada, não
  uma mudança do que é pedido ao modelo.

### 1.2 Verificação manual (ampliada, ainda sem inferência)

`scripts/verify_explicit_decision_contract.py` — 2 cenários novos
(total agora 10, todos sem chamar modelo):

6. `decisao_final: não_melanoma` isolado → esperado `ok`/`nao_melanoma`
   (confirma a variante aceita, normalizada para o valor canônico).
7. a palavra `não_melanoma` aparecendo na JUSTIFICATIVA (corpo do
   texto), SEM a linha `decisao_final` declarada → esperado
   `field_absent` (confirma que a exceção de grafia vale só para o
   valor capturado pela linha do campo, nunca reabre a leitura do
   corpo do texto — pedido explícito desta tarefa).

**Comando exato para o pesquisador rodar:**

```
python scripts/verify_explicit_decision_contract.py
```

**Saída esperada:** 10 linhas `[PASS]` seguidas de
`Resumo: 10 PASS, 0 FAIL, 10 cenários no total.`

### 1.3 Reprocessamento offline das 8 chamadas já salvas (novo, zero inferência)

`scripts/run_reprocess_experimental_decision_contract_accent_variant.py`:

- Lê `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/results.json`
  (as 8 chamadas reais já salvas pelo doc 24) — **não modifica esse
  arquivo**.
- Para cada uma das 8, reaplica `extract_decisao_final` (já com a
  variante aceita) sobre o MESMO `raw_response_text` salvo — zero
  chamada nova a modelo. Registra "antes" (valor já salvo) e "depois"
  (recomputado agora) lado a lado, e um campo `changed` (bool).
- Grava, em arquivos NOVOS (nunca sobrescrevendo `results.json`/
  `report.md` originais):
  - `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/accent_variant_reprocessing_results.json`
  - `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/accent_variant_reprocessing_report.md`

**Comando exato para o pesquisador rodar:**

```
python scripts/run_reprocess_experimental_decision_contract_accent_variant.py
```

**Saída esperada:** stdout reportando "Reprocessadas 8 chamadas" e
"Mudaram de decision_status/decisao_final: 2" (os 2 casos do
medgemma1.5 citados acima, de `invalid_value`/`None` para
`ok`/`"nao_melanoma"`); os outros 6 devem reportar `changed: False`
(nenhuma regressão nos que já estavam `ok`). Os dois arquivos novos
listados acima, com os 8 registros antes/depois.

## 2. O que NÃO foi feito nesta tarefa

- Nenhuma execução real (nenhuma chamada nova a Ollama/qwen3-vl/
  medgemma) — o reprocessamento é só sobre texto já salvo em disco.
- Nenhuma mudança em `src/multiagent/experimental_decision_contract.py`
  (instruções de tarefa), `src/agents/agents.py`,
  `src/multiagent/vlm_direct.py`,
  `src/evaluation/case_status.py`/`case_status_v2.py`, ou
  `scripts/run_local_vlm_pilot.py`.
- Nenhuma generalização da aceitação de acento (ex.: via
  `unicodedata.normalize`) — só a entrada literal única observada.
- Nenhuma decisão sobre trocar o critério de avanço do Estágio 1 pelo
  contrato de decisão explícito — continua pendente, a cargo do
  pesquisador.

## 3. Arquivos alterados/criados nesta entrega

- `src/evaluation/explicit_decision_contract.py` (modificado --
  variante aceita)
- `scripts/verify_explicit_decision_contract.py` (modificado -- 2
  cenários novos)
- `scripts/run_reprocess_experimental_decision_contract_accent_variant.py` (novo)
- `docs/reconstruction/25_accent_variant_and_reprocessing.md` (novo,
  este arquivo)

Nenhum outro arquivo existente foi modificado.
