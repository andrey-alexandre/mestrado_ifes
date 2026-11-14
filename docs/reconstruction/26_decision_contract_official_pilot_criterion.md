# 26 — Contrato de decisão explícito como critério oficial do piloto local

Resumo para o planejador central: decisão do pesquisador, após os 8/8
`ok` do reprocessamento do doc 25, de **substituir o parser v2 pelo
contrato `decisao_final` como critério de leitura do veredito no
piloto local de VLMs**, para os papéis VLM-direto e sumarizador.
`classify_case_status`/`classify_case_status_v2` (v1/v2) são
preservados, só para rastreabilidade, nunca como fallback.
qwen3-vl/medgemma1.5 seguem como candidatos.

**Status: integração implementada; verificação real do VLM-direto
preparada, aguardando execução do pesquisador.** Nada foi executado
por este agente (`feedback_no_autonomous_execution`).

## 1. Integração ao executor do piloto (`src/multiagent/vlm_direct.py`)

- `CallRecord` ganha dois campos ADITIVOS, OFICIAIS a partir desta
  entrega: `decision_status`/`decisao_final` (saída de
  `extract_decisao_final`). `classified_status`/`classified_status_v2`
  continuam computados, no mesmo gate (`status == STATUS_OK`), só para
  rastreabilidade.
- `VLM_DIRECT_TASK_INSTRUCTION` (instrução PADRÃO do papel VLM-direto)
  passa a ser a mesma instrução do contrato de decisão já validada nos
  controles de síntese (docs 23-25) -- `EXPERIMENTAL_VLM_DIRECT_TASK_INSTRUCTION`,
  reusada sem duplicar texto. O texto ORIGINAL de 3 categorias
  textuais fica preservado, verbatim, em
  `VLM_DIRECT_TASK_INSTRUCTION_V1_HISTORICAL` -- não mais usado como
  padrão, mantido só para rastreabilidade.
- `run_vlm_direct_call` agora computa `decision_status`/`decisao_final`
  (aditivo, mesmo gate de v1/v2).
- Papel sumarizador: NOVA função
  `run_summary_role_call_official_decision_contract` -- usa a MESMA
  rota/configuração já validada pelos 8 controles de síntese reais
  (docs 23-25): `build_experimental_summary_task_instruction` +
  chamada direta ao `ChatOllama`, em vez de `SummaryAgent.summarize()`
  (prompt oficial de `src/agents/agents.py`, que continua INTOCADO e
  continua sendo usado por `build_graph()`/produção e pelos controles
  de síntese HISTÓRICOS `run_synthesis_control`/
  `run_synthesis_control_real_image`, que continuam chamando
  `run_multiagent_role_call` SEM alteração -- "não há motivo para
  repetir os oito controles de síntese", verbatim do pesquisador).
- `run_multiagent_role_call` **não foi modificado** -- continua sendo
  o caminho do papel `summary` para qualquer chamador que não seja a
  matriz principal do piloto.
- Novo helper `classify_binary_coverage_from_decisao_final` (análogo a
  `classify_binary_coverage`, mas sobre `decisao_final`).

## 2. Integração ao driver do piloto (`scripts/run_local_vlm_pilot.py`)

- Dispatch da matriz principal: papel `summary` agora chama
  `run_summary_role_call_official_decision_contract`; `vlm_direto`
  continua chamando `run_vlm_direct_call` (que já usa a nova
  instrução por padrão); ABCD/Menzies/SPCL continuam usando
  `run_multiagent_role_call`, sem mudança.
- `binary_coverage` (relatório oficial) passa a vir de
  `classify_binary_coverage_from_decisao_final`. v1/v2 ficam em
  `binary_coverage_v1_historical_informational`/
  `binary_coverage_v2_historical_informational` -- rastreabilidade só.
- `roles_with_zero_conclusive_opinion` passa a checar
  `decisao_final in (melanoma, nao_melanoma, indeterminado)`.
- NOVO campo oficial `decision_contract_valid_rate_official`: fração de
  chamadas executadas de vlm_direto/summary com `decision_status ==
  "ok"` -- `indeterminado` explícito CONTA como válido (decisão
  verbatim do pesquisador), só não entra na cobertura binária. Este é
  o critério de "≥95% de respostas válidas" do piloto a partir desta
  entrega. `valid_on_first_attempt_rate` (todos os papéis, sucesso de
  chamada) continua reportado, sem mudança de definição.
- `official_classification_parser_version` passa a ser
  `PARSER_VERSION` de `src.evaluation.explicit_decision_contract`
  (novo, `"explicit_decision_contract_v1_accented_nao_melanoma_variant_2026_11_14"`),
  não mais `case_status_v2.PARSER_VERSION`.
- **NÃO alterado**: o gate de parseabilidade do ESTÁGIO 1
  (`run_synthesis_control`, imagem placeholder, "SE a saída não for
  sequer PARSEÁVEL pela A34, PARA aqui") continua lendo
  `classify_case_status_v2` -- é um gate de parseabilidade
  estruturalmente anterior e separado do critério de leitura do
  veredito da matriz, fora do escopo desta troca.

## 3. Verificação real do VLM-direto (nova, não executada)

`scripts/run_vlm_direto_real_image_smoke_test.py`:

- Reusa a MESMA imagem real já selecionada/verificada pelo doc 24
  (`experiments/multiagent/synthesis_control_real_image_registry.json`),
  já confirmada externa à amostra congelada de 3.000. ADICIONALMENTE
  verifica exclusão contra `experiments/multiagent/pilot12_manifest.csv`
  -- se esse arquivo ainda não existir (não gerado até agora), registra
  isso explicitamente, nunca assume.
- 1 chamada por modelo (`qwen3-vl:8b-instruct-q4_K_M`,
  `medgemma1.5:4b-it-q4_K_M`) = 2 chamadas totais, sem retry, via
  `vd.run_vlm_direct_call` (já usando a instrução oficial por padrão).
- Verifica 3 coisas, SEM avaliar acerto diagnóstico: (1) payload
  multimodal construído com a imagem certa
  (`verify_image_control_payload_sent_correct_image`, reusado); (2)
  texto bruto não vazio preservado; (3) `decision_status == "ok"`
  (campo lido com sucesso -- qualquer uma das 3 decisões conta como
  sucesso desta verificação, não há categoria esperada).

**Comando exato para o pesquisador rodar:**

```
python scripts/run_vlm_direto_real_image_smoke_test.py \
    --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
```

**Saída esperada:** `experiments/multiagent/local_vlm_pilot_vlm_direto_smoke_test/results.json`
com 2 entradas (uma por modelo), `all_integration_checks_passed: true`
se as 3 checagens passarem nos 2 modelos. Qualquer `false` indica um
problema de INTEGRAÇÃO (imagem não chega, texto vazio, ou campo não
lido), não um erro de diagnóstico do modelo.

## 4. O que NÃO foi feito nesta tarefa

- Nenhuma execução real (nenhuma chamada a Ollama/qwen3-vl/medgemma).
- Nenhuma repetição dos 8 controles de síntese reais (docs 24/25) --
  a integração reusa a mesma rota/configuração já validada lá.
- Nenhuma mudança em `src/agents/agents.py` (prompts oficiais da
  produção/`build_graph()`), `run_synthesis_control`/
  `run_synthesis_control_real_image` (controles históricos),
  `run_multiagent_role_call` (dispatch genérico, continua servindo
  qualquer chamador fora da matriz principal), ou o gate de
  parseabilidade do Estágio 1.
- Nenhuma liberação do piloto das 12 lesões -- continua explicitamente
  condicionada ao resultado real desta verificação, decisão do
  pesquisador.

## 5. Arquivos alterados/criados nesta entrega

- `src/evaluation/explicit_decision_contract.py` (modificado -- novo
  `PARSER_VERSION`)
- `src/multiagent/vlm_direct.py` (modificado -- integração do contrato)
- `scripts/run_local_vlm_pilot.py` (modificado -- dispatch + relatório
  oficial trocado)
- `scripts/run_vlm_direto_real_image_smoke_test.py` (novo)
- `docs/reconstruction/26_decision_contract_official_pilot_criterion.md`
  (novo, este arquivo)
