# 27 — Checagem prévia sem inferência antes do piloto das 12 lesões

Resumo para o planejador central: o pesquisador autorizou seguir para
o piloto das 12 lesões com Qwen3-VL e MedGemma, após o smoke test real
do VLM-direto (doc 26) ter passado nos 2 modelos
(`all_integration_checks_passed: true`). Antes de entregar o `.py` e
os comandos de execução real, foi pedida uma checagem prévia SEM
INFERÊNCIA.

**Status: checagem preparada, aguardando execução do pesquisador.**
Nada foi executado por este agente (`feedback_no_autonomous_execution`).

## 1. O que a checagem confirma (sem chamar nenhum modelo)

`scripts/run_pilot12_preflight_check.py`:

1. `pilot12_manifest.csv` tem exatamente 12 lesões distintas (12
   `lesion_id` distintos, 12 `image_id` distintos, uma linha por
   lesão) -- reusa `src.data_audit.sampling
   .verify_lesion_and_image_uniqueness`, já usada em outras
   verificações de amostragem do projeto, não uma segunda cópia.
2. ZERO sobreposição, por `lesion_id` E por `image_id`, com a amostra
   congelada de 3.000 (`experiments/sampling/sample_manifest.csv`) --
   SHA-256 desse manifesto conferido primeiro contra o valor
   registrado (mesmo literal já usado em
   `scripts/run_pilot12_sample.py`/
   `scripts/run_synthesis_control_real_image_selection.py`).
3. As entradas S0/S1/S2 previstas -- recomputa
   `src.multiagent.vlm_direct.build_planned_calls` (a MESMA função
   que `scripts/run_local_vlm_pilot.py` usa de verdade) sobre o
   manifesto e os 4 `image_id` designados para S0/S1 (resolvidos pela
   MESMA lógica de `run_local_vlm_pilot._resolve_designated_image_ids`,
   reusada, não duplicada) -- confirma 92 chamadas totais: 12
   vlm_direto/S0, 48 multiagente/S2, 32 multiagente/{S0,S1}
   designados.
4. Registra a configuração fixa desta rodada: os 2 modelos
   candidatos (`qwen3-vl:8b-instruct-q4_K_M`,
   `medgemma1.5:4b-it-q4_K_M`), a versão exata do parser do contrato
   de decisão (`src.evaluation.explicit_decision_contract.PARSER_VERSION`)
   e a regra de execução (uma tentativa por chamada, sem retry, sem
   fallback para v1/v2, sem ajustes de vocabulário durante a
   execução) -- para rastreabilidade no relatório final do piloto.

Se qualquer checagem falhar, o relatório registra
`ready_for_pilot12_execution: false` e a lista de problemas
explicitamente -- o script nunca decide por conta própria seguir para
a execução real.

**Comando exato para o pesquisador rodar:**

```
python scripts/run_pilot12_preflight_check.py
```

**Saída esperada:** `experiments/multiagent/local_vlm_pilot_preflight_check/results.json`
com `ready_for_pilot12_execution: true` (problems vazio) se tudo
estiver certo.

## 2. Comandos para a execução real do piloto (se a checagem passar)

`scripts/run_local_vlm_pilot.py` já está integrado ao contrato de
decisão (doc 26) -- uma tentativa por chamada, sem retry (já era
assim desde a 2ª entrega da Etapa 5), sem fallback para v1/v2 (ver
doc 26, `decision_status`/`decisao_final` são a fonte oficial), e sem
nenhum ajuste de vocabulário feito durante a execução (o parser está
congelado nesta entrega). Dois estágios, por modelo, um modelo por
vez:

```
# qwen3-vl -- Estágio 1 (controles + template de revisão manual)
python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M
# (pesquisador revisa e preenche experiments/multiagent/local_vlm_pilot/
#  qwen3-vl_8b-instruct-q4_K_M/synthesis_manual_review_template.json)
# qwen3-vl -- Estágio 2 (matriz principal, 92 chamadas)
python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M \
    --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/synthesis_manual_review_template.json

# medgemma1.5 -- mesmo fluxo de 2 estágios
python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M
python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M \
    --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/medgemma1.5_4b-it-q4_K_M/synthesis_manual_review_template.json
```

O relatório final de cada modelo (`summary.json`/`report.md`, em
`experiments/multiagent/local_vlm_pilot/<model_tag>/`) agora separa
explicitamente:

- `valid_on_first_attempt_rate` -- sucesso de CHAMADA, todos os
  papéis (sem mudança de definição).
- `decision_contract_valid_rate_official` -- o critério de ≥95% de
  respostas válidas (só vlm_direto/summary): `indeterminado` explícito
  CONTA como válido, só não entra em `binary_coverage`.
  `decision_contract_valid_count`/`decision_contract_applicable_count`
  são os denominadores explícitos (numerador/total).
- `decision_contract_failures_detail` -- NOVO nesta entrega: lista de
  toda chamada vlm_direto/summary que não contou como válida, com
  papel/condição/lesão/imagem/causa explícitos (1 modelo por execução
  do script -- "por modelo" já é o próprio arquivo de saída).
- `binary_coverage` -- positivo/negativo, OFICIAL via `decisao_final`.
  v1/v2 ficam em `binary_coverage_v1_historical_informational`/
  `binary_coverage_v2_historical_informational`, só rastreabilidade.

## 3. O que NÃO foi feito nesta tarefa

- Nenhuma execução real (nem a checagem prévia, nem o piloto).
- Esta liberação é só para as 12 lesões -- a decisão sobre quais
  modelos avançam para as 3.000 continua pendente, após revisão do
  piloto pelo pesquisador.

## 4. Arquivos alterados/criados nesta entrega

- `scripts/run_pilot12_preflight_check.py` (novo)
- `scripts/run_local_vlm_pilot.py` (modificado -- denominadores
  explícitos e `decision_contract_failures_detail` no relatório)
- `docs/reconstruction/27_pilot12_preflight_check.md` (novo, este
  arquivo)
