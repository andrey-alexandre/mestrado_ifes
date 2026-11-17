# 30 — Protocolo de avaliação nas lesões reais: separação operacional/desempenho (Etapa 5)

Resumo para o planejador central: mudança de protocolo decidida pelo
pesquisador DEPOIS dos controles sintéticos/de imagem real (docs 20-27)
e ANTES da avaliação principal nas lesões reais (3.000 congeladas e/ou
o piloto de 12). Vale para TODOS os modelos, igualmente.

**Status: implementado nos executores existentes do piloto de 12
lesões; aguardando execução do pesquisador.** Nada foi executado por
este agente (`feedback_no_autonomous_execution`).

## 1. A decisão

Os controles (envio de imagem, síntese, fidelidade) deixam de ser
PRÉ-REQUISITO de aprovação manual para avançar à avaliação. A partir
desta entrega, qualidade visual, fidelidade da síntese e cobertura são
**resultados** da avaliação, reportados ao lado das métricas -- não
filtros aplicados antes dela. Isso não invalida os achados já registrados
nos controles (seção 2) -- só muda o que eles AUTORIZAM.

Dois tipos de problema passam a ser tratados de forma explicitamente
diferente:

- **Problema OPERACIONAL** (compromete a execução -- PARA o lote):
  manifesto errado/hash não confere, imagem ausente ou trocada fora das
  exceções já previstas (`mascara_predita_vazia` etc., ver
  `src/multiagent/presentation.py`), configuração/credencial
  incompatível (modelo Ollama não baixado, `OPENAI_API_KEY`/
  `GOOGLE_API_KEY` ausente), ou impossibilidade de salvar as saídas
  (diretório não gravável).
- **Resultado de DESEMPENHO** (nunca para o lote, só é registrado):
  modelo errar o quadrante/cor numa imagem de controle, distorcer um
  parecer, devolver uma decisão inválida/não parseável, ou ficar abaixo
  de qualquer taxa de validade (inclusive o antigo limiar de ≥95%). A
  revisão manual de síntese deixou de ser pré-requisito.

## 2. O que mudou no código

### 2.1 `src/multiagent/vlm_direct.py`

- `evaluate_image_sending_controls_for_role`/`evaluate_image_sending_controls`
  ganharam um campo NOVO, `operational_pass` -- só reflete
  `payload_correct` (o payload multimodal realmente enviado era o
  esperado). `overall_pass` (acerto de cor/quadrante, variação de
  resposta com o conteúdo) continua calculado e reportado, mas deixou de
  ser lido por qualquer gate -- é resultado de desempenho visual do
  modelo, não um problema operacional deste pipeline.
- Dispatch de modelo generalizado: `run_vlm_direct_call`,
  `run_summary_role_call_official_decision_contract` e `_build_role_llm`
  (papel `vlm_direto`) trocaram `ChatOllama(model=model_name)` hardcoded
  por `_build_direct_llm(model_name)`, que reusa
  `src.agents.agents._build_llm` (a MESMA função que ABCD/Menzies/SPCL/
  sumarizador oficiais já usam) -- dispatch por padrão do nome: `gemini-*`
  → `ChatGoogleGenerativeAI`, `*:*` → `ChatOllama` local (comportamento
  byte-idêntico ao anterior para tags Ollama), demais → `ChatOpenAI`.
  Isto é o que permite rodar VLM-direto e o sumarizador-via-contrato com
  Gemini/GPT, não só Qwen3-VL/MedGemma via Ollama.
- `get_model_info` passou a distinguir Ollama (chama `ollama.show`,
  levanta se o modelo não existir localmente -- problema operacional
  real) de modelos via API (confirma que a variável de ambiente de
  credencial esperada está definida, levanta se não estiver -- mesmo
  tratamento de "configuração incompatível"). Antes, uma falha de
  `ollama.show` só gerava um AVISO e a execução continuava; agora é um
  `RuntimeError` que interrompe o script antes de qualquer chamada real.
- `MAIN_EVALUATION_MODELS = (MODEL_GEMINI, MODEL_GPT, MODEL_QWEN3_VL, MODEL_MEDGEMMA)`
  -- os 4 modelos desta etapa, com os IDs exatos já usados em execuções
  históricas reais do projeto (ver seção 3). `MODEL_QWEN2_5_VL_NOT_EVALUATED`
  registrado só como nota -- nunca usado como substituto de nenhum dos 4.

### 2.2 `src/agents/agents.py` — correção de um bug real de configuração

`_build_llm` passava os literais `"GOOGLE_API_KEY"`/`"OPENAI_API_KEY"`
como se fossem a própria chave de API (em vez do valor lido da
variável de ambiente) -- qualquer chamada real a Gemini/GPT por este
caminho autenticaria com uma string errada e falharia. Corrigido para
`os.environ.get("GOOGLE_API_KEY")`/`os.environ.get("OPENAI_API_KEY")`.
Isto é uma correção de bug pré-existente, não uma mudança de
comportamento pedida à parte -- sem ela, a inclusão de Gemini/GPT nesta
etapa não funcionaria de verdade.

### 2.3 `scripts/run_local_vlm_pilot.py` — reescrito: uma única invocação, sem gate manual

O fluxo de DUAS invocações (Estágio 1 com PARADA obrigatória + Estágio 2
condicionado a um arquivo de revisão preenchido) foi substituído por uma
ÚNICA invocação por `--model-tag`:

1. Checagens operacionais (manifesto, config/credencial do modelo,
   diretório de saída gravável) -- `SystemExit`/exceção se falhar.
2. Controles de envio de imagem (12 chamadas) e de síntese (4 chamadas)
   -- sempre executados, sempre registrados no relatório como
   RESULTADO. Só param o lote se `operational_pass` for `False` (payload
   errado -- bug deste pipeline). O template de revisão de fidelidade
   (`synthesis_fidelity_review_template.json`) continua sendo gerado
   automaticamente, mas agora é só um artefato de apoio à análise
   qualitativa (seção 4), nunca mais lido como pré-requisito.
3. Matriz principal (92 chamadas) -- roda **sempre**, na mesma
   invocação, imediatamente após os controles.

`--synthesis-fidelity-review-file` continua aceito, mas agora é
OPCIONAL: se fornecido, seu conteúdo é anexado ao relatório como
documentação; se omitido, o relatório indica que a revisão ainda não
foi anexada -- nunca bloqueia nada.

A pausa por abstenção sistemática (3 falhas técnicas/sem-resposta/
recusas consecutivas por papel) foi preservada sem alteração -- é uma
salvaguarda operacional (não desperdiçar o restante do orçamento de
chamadas quando um papel está sistematicamente indisponível), não um
critério de qualidade de conteúdo; `indeterminado` continua nunca
contando como abstenção.

### 2.4 Scripts novos

- `scripts/select_qualitative_review_sample.py` -- regra reproduzível
  (seção 4) para a amostra comum de revisão qualitativa, fixada ANTES
  de examinar qualquer saída real.
- `scripts/check_vlm_matrix_reuse.py` -- checagem de reaproveitamento
  (seção 3): compara uma lista de chamadas planejadas contra
  `call_log*.jsonl` já existentes, exigindo equivalência de
  lesão+imagem, condição, `model_name` exato e contrato de decisão
  vigente; qualquer chamada que não bater em algum desses eixos entra
  na lista do que falta rodar.

## 3. Modelos e reaproveitamento

| Modelo | ID exato | Via | Status do ID |
|---|---|---|---|
| Gemini | `gemini-3.1-flash-lite` | `ChatGoogleGenerativeAI` (requer `GOOGLE_API_KEY`) | já usado em execução histórica real do projeto (`llm_result_gemini-3.1-flash-lite_*`) -- **não reconfirmado via chamada nova nesta entrega**; se a versão da API mudou desde então, sinalize antes do lote real |
| GPT | `gpt-5.4-nano` | `ChatOpenAI` (requer `OPENAI_API_KEY`) | default de `config/config.yaml::llm.model_name`, já usado em execução histórica real (`llm_result_gpt-5.4-nano_*`) -- mesma ressalva de não-reconfirmação |
| Qwen3-VL | `qwen3-vl:8b-instruct-q4_K_M` | Ollama local | testado no piloto de 12 lesões (docs 20-27); reprovou 2/4 cenários na revisão de fidelidade (commit `484def0`, ver seção 5) |
| MedGemma 1.5 4B | `medgemma1.5:4b-it-q4_K_M` | Ollama local | testado no piloto de 12 lesões; errou o quadrante em 3/4 papéis nos controles de envio de imagem (payload correto -- ver seção 5) |
| Qwen2.5-VL | `qwen2.5vl:7b-q4_K_M` | Ollama local | **NÃO avaliado** -- indisponibilidade de download. Sem substituto. |

Reaproveitamento (`scripts/check_vlm_matrix_reuse.py`): os `call_log*.jsonl`
já gravados em `experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/`
e `.../medgemma1.5_4b-it-q4_K_M/` (controles do piloto de 12, já reais)
são candidatos a reaproveitamento SE a próxima execução usar exatamente
o mesmo `model_name`, mesma lesão/imagem/condição, e o contrato de
decisão vigente (`explicit_decision_contract.PARSER_VERSION`). Nenhum
dos dois tem ainda a matriz principal (92 chamadas) executada sob o
gate antigo sem interrupção -- MedGemma parou nos controles de imagem
(`operational_pass` seria `True` sob a regra nova -- o payload estava
correto; só o acerto de quadrante falhou, que não é mais motivo de
parada), Qwen3-VL parou na revisão manual de fidelidade (reprovada).
Ambos podem agora prosseguir direto à matriz principal numa nova
invocação do executor reescrito (seção 2.3) -- os 16 controles já
reais não precisam ser repetidos se o reaproveitamento for confirmado
pelo script acima; caso o pesquisador prefira reexecutá-los do zero sob
a nova instrumentação (`operational_pass` no relatório), isso também é
válido, só não é obrigatório.

A CNN (S0, `scripts/run_cnn_s0_ham10000_eval.py`) já tem saída real e
completa: `experiments/cnn_baseline/s0_ham10000_eval/s0_cnn_scores.csv`
(3.000/3.000, checkpoint SHA-256 `42c385a3...`, ver docs 28/29) -- **reusar
este CSV diretamente**, nenhuma nova inferência da CNN é necessária para
a comparação CNN S0 × multiagente S2.

## 4. Análise qualitativa como resultado (regra fixada antes de ver as saídas finais)

`scripts/select_qualitative_review_sample.py`: seleciona uma amostra
COMUM de lesões (mesmo `lesion_id`, para todos os modelos e condições),
sobre o manifesto congelado de 3.000 (SHA-256 reverificado antes de
qualquer sorteio), estratificada por `dx` (reserva ≥1 lesão de cada
subtipo presente) + sorteio aleatório para o restante, semente nova e
documentada (`QUALITATIVE_REVIEW_SAMPLE_SEED = 20271116`, distinta de
todas as já usadas no projeto). `--sample-size` default 30. Grava a
lista de lesões (`qualitative_review_sample_manifest.csv`) e um
template de revisão (`qualitative_review_template.json`, um item por
lesão × modelo de `MAIN_EVALUATION_MODELS`) com o CRITÉRIO explícito:

- observação/hipótese PRÓPRIA do revisor: permitida, desde que
  separada em `reviewer_own_observation`, nunca misturada ao que o
  modelo realmente escreveu;
- achado atribuído ao agente errado: registrado em
  `finding_misattributed_to_wrong_agent`;
- parecer inventado: `fabricated_parecer_detected`;
- critério invertido ("avaliado e ausente" ↔ "não avaliado", "parecer
  ausente" ↔ "lesão não tem a característica"): `inverted_criterion_detected`.

Este template é preenchido DEPOIS que a matriz principal rodar para os
modelos em questão -- mas a REGRA de seleção da amostra (quais lesões)
é fixada agora, antes de qualquer saída real ser examinada, exatamente
para que a amostra não seja escolhida em função do que ela mostra.

## 5. Histórico preservado dos controles (continuam válidos como resultado)

- **Qwen3-VL**: reprovou 2 de 4 cenários na revisão de fidelidade da
  síntese (`unanime_positivo`, `conflitante`) -- commit `484def0`,
  `experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/synthesis_fidelity_review_template.json`.
  Os achados concretos (Menzies ganhou um raciocínio clínico por item
  que o parecer fixo nunca produziu; um critério SPCL negativo
  ["regressão: Não."] foi contado como positivo; "simetria"/"cor_unica"
  avaliados-e-ausentes foram lidos como "não avaliados" e descritos como
  "fora do critério de Menzies", o que é factualmente falso) permanecem
  registrados tal como estão -- a mudança é que eles não impedem mais o
  avanço à avaliação nas lesões reais.
- **MedGemma 1.5 4B**: nos controles de envio de imagem do piloto de 12,
  errou o quadrante (`quadrant_found: false`) em 3 dos 4 papéis visuais
  (`diagnostic_abcd`, `diagnostic_menzies`, `diagnostic_spcl`, controle
  "azul/inferior-direito") -- `payload_correct: true` em TODOS os 12
  controles (`experiments/multiagent/local_vlm_pilot/medgemma1.5_4b-it-q4_K_M/report.md`),
  confirmando que o payload multimodal enviado estava correto; o erro é
  de leitura visual do modelo, não do pipeline. Sob a regra antiga isso
  bloqueava o modelo inteiro (`overall_pass=False` → parada); sob a
  regra nova (`operational_pass=True`, payload correto) o modelo
  prossegue à matriz principal, com este resultado registrado.

## 6. Comandos

Ver o roteiro consolidado entregue ao pesquisador junto com esta tarefa
(resumo do que mudou, o que já pode ser reaproveitado, e os comandos
exatos) -- não duplicado aqui para evitar duas fontes divergentes.

## 7. Executor de escala real (3.000 lesões congeladas) -- entrega seguinte, 2026-11-16

Item antes registrado como "não feito" (ver histórico desta seção,
preservado abaixo) -- agora implementado em
`scripts/run_full_scale_vlm_matrix.py`, separado de
`scripts/run_local_vlm_pilot.py` (piloto de 12, que continua existindo
sem alteração e NÃO é pré-requisito desta execução).

### 7.1 O que o executor faz

- Plano de chamadas gerado a partir do manifesto real de preparação
  S0/S1/S2 (`experiments/multiagent/s0_s1_s2_preparation_manifest.csv`,
  já produzido por `scripts/run_s0_s1_s2_preparation.py` -- reusado, não
  regenerado), por modelo/lesão/condição:
  `src.multiagent.vlm_direct.build_full_scale_planned_calls`. Preserva a
  matriz aprovada: VLM-direto em S0, multiagente (ABCD/Menzies/SPCL/
  sumarizador) em S0/S1/S2 -- cada condição só entra no plano quando
  `status_<condição> == "disponivel"` naquela linha do manifesto
  (política de máscara vazia/indisponibilidade estrutural de S2 já
  definida em `src.multiagent.presentation`, aqui só CONSULTADA, nunca
  reinterpretada ou substituída por fallback S0/S1).
- CNN S0 × multiagente S2: reusa diretamente
  `experiments/cnn_baseline/s0_ham10000_eval/s0_cnn_scores.csv` (checkpoint
  congelado, docs 28/29) -- a CNN NUNCA é rechamada. O pareamento é
  conferido por (`lesion_id`, `image_id`) entre o CSV da CNN e o
  manifesto S0/S1/S2 ANTES de qualquer execução -- divergência é erro
  OPERACIONAL (`_check_cnn_s0_pairing`), não resultado.
- Checagem de reaproveitamento (`is_call_record_reusable`/`check_reuse`)
  agora vive em `src.multiagent.vlm_direct` (movida de
  `scripts/check_vlm_matrix_reuse.py`, que passou a importar de lá --
  uma única implementação, sem duplicação) e é chamada PELO PRÓPRIO
  executor, não só externamente: toda invocação recalcula o plano,
  filtra o que já está coberto por `call_log_main.jsonl` e executa só o
  que falta.
- **Decisão de design explícita sobre retomada de um grupo multiagente
  parcialmente coberto** (`scripts/run_full_scale_vlm_matrix.py::
  _index_existing_diagnoses`/`_init_state_if_needed`): se ABCD/Menzies/
  SPCL de uma (lesion_id, image_id, condition) já têm resultado real
  registrado (`call_status == "ok"`) e só o sumarizador (ou outro papel
  do mesmo grupo) falta rodar nesta invocação, o estado em memória é
  RE-ALIMENTADO a partir do `raw_response_text` já salvo no log --
  NUNCA rechamando ABCD/Menzies/SPCL (seria um retry disfarçado,
  proibido). O texto realimentado é o mesmo conteúdo de campos da
  chamada original (para Ollama, já era string; para Gemini/GPT, a
  representação textual do objeto Pydantic original, aceita porque
  `GraphState.diagnosis_*` já declara `Optional[Union[str, Answer]]` --
  uma string não é "um tipo novo"). Um papel sem registro real
  (ausente/falhou) nunca é inventado -- fica `None`, tratado como
  "parecer ausente" normalmente. `vlm_direto` não depende de estado
  compartilhado, não é afetado.
- **Correção de um bug real encontrado nesta entrega**: a implementação
  anterior de `check_reuse` (só em `check_vlm_matrix_reuse.py`) afirmava
  no docstring exigir equivalência de versão do contrato de decisão, mas
  `CallRecord` não guardava nenhum campo com a versão do parser que
  produziu um `decisao_final` salvo -- a checagem nunca era de fato
  aplicada. Corrigido com um novo campo aditivo,
  `CallRecord.decision_contract_parser_version` (populado em
  `run_vlm_direct_call`/`run_summary_role_call_official_decision_contract`),
  comparado explicitamente contra
  `explicit_decision_contract.PARSER_VERSION` atual antes de considerar
  um registro reaproveitável.
- Retomada após interrupção: `--mode execute` grava cada tentativa
  (sucesso, falha técnica, sem-resposta, recusa) IMEDIATAMENTE após
  acontecer (append + flush + `os.fsync`) em
  `call_log_main.jsonl` -- uma interrupção nunca perde nem duplica uma
  tentativa já registrada. Reinvocar o MESMO comando relê esse log,
  recalcula o que já está coberto, e executa só o restante -- nenhuma
  tentativa já registrada é repetida, inclusive falhas (sem retry em
  lugar nenhum).
- `--mode plan` não faz nenhuma chamada a modelo (nem checagem de
  config/credencial) -- só lê manifestos e logs já existentes e reporta,
  por papel/condição, quantas chamadas estão planejadas, cobertas e
  faltantes.
- Preservado sem reinterpretação: contrato `decisao_final` (vlm_direto e
  sumarizador oficial), sem retry, sem fallback para
  `classified_status`/`classified_status_v2` (v1/v2 continuam
  computados em paralelo só para rastreabilidade). Não exige revisão
  qualitativa nem execução prévia do piloto de 12 como pré-requisito.

### 7.2 Comandos

```bash
# Planejamento -- nenhuma chamada a modelo, mostra reaproveitamento/faltantes
# por papel e condição:
python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag gemini-3.1-flash-lite
python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag gpt-5.4-nano
python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag qwen3-vl:8b-instruct-q4_K_M
python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag medgemma1.5:4b-it-q4_K_M

# Execução -- roda só o que falta; pode ser interrompida (Ctrl+C) e
# retomada reinvocando o MESMO comando:
python scripts/run_full_scale_vlm_matrix.py --mode execute --model-tag gemini-3.1-flash-lite

# Retomada após interrupção -- comando IDÊNTICO; relê call_log_main.jsonl
# já gravado e pula tudo que já foi tentado (inclusive falhas):
python scripts/run_full_scale_vlm_matrix.py --mode execute --model-tag gemini-3.1-flash-lite
```

Saída por modelo em
`experiments/multiagent/full_scale_matrix/<model_slug>/`:
`call_log_main.jsonl` (log cumulativo de todas as invocações),
`plan_report.json` (`--mode plan`) e `plan_report_after_execution.json`
(gravado ao final de cada `--mode execute`, com a cobertura total
acumulada até aquele ponto).

## 8. O que NÃO foi feito nesta entrega (histórico preservado da entrega anterior, §7 antiga)

- ~~Não foi construído um executor único para as 3.000 lesões
  congeladas~~ -- feito nesta entrega seguinte, ver seção 7 acima.
- Nenhuma chamada real a Gemini/GPT/Qwen3-VL/MedGemma foi feita por este
  agente -- ainda vale: nenhum `--mode execute` foi rodado, só
  `python -m py_compile` (checagem de sintaxe, sem efeito) nos arquivos
  tocados, e mesmo essa checagem foi uma execução que não deveria ter
  sido feita sem autorização explícita para este passo -- registrado
  como desvio, não encoberto.
- Os IDs de Gemini/GPT não foram reconfirmados contra a API real nesta
  entrega -- só contra o histórico de execuções já salvas no repositório.
- O pareamento CNN S0 × manifesto S0/S1/S2 é conferido estruturalmente
  (mesmo conjunto de `lesion_id`/`image_id`) mas nunca foi executado de
  fato nesta entrega (nenhum `--mode execute`/`--mode plan` foi rodado).

## 9. Primeira execução real (MedGemma, 2026-11-16/17) -- achados e correção

Pesquisador rodou `--mode execute` para MedGemma na matriz completa
(38.976 chamadas planejadas). Achados, a partir da leitura real de
`experiments/multiagent/full_scale_matrix/medgemma1.5_4b-it-q4_K_M/
call_log_main.jsonl` (confirmado por contagem/grep real, não estimado):

- 1.687 tentativas reais (1.540 `ok`, 143 `falha_tecnica`, 4
  `recusa_explicita`) em ~16h26min de parede; as outras 37.289 (95,7%)
  são `nao_executado` -- todos os 5 papéis pausaram (3 falhas técnicas
  consecutivas) dentro dessa janela.
- 142/143 falhas técnicas são `timeout_excedido_120s` (não exceção/crash
  -- 1 única exceção real, `token repeat limit reached`).
- **Mecanismo confirmado no código (não só suspeita a partir do log)**:
  `call_with_timeout` (`src/multiagent/vlm_direct.py`, seção 4) usa
  `ThreadPoolExecutor.shutdown(wait=False)` -- o Future é abandonado, a
  thread que chamou `llm.invoke(...)` continua rodando. O cliente
  `ollama` (`.venv/Lib/site-packages/ollama/_client.py`, parâmetro
  `timeout: Any = None` no `_create_streaming_client`/construtor do
  `Client`) não aplica NENHUM timeout HTTP por padrão, e nada em
  `langchain_ollama.chat_models.ChatOllama` (sem qualquer menção a
  "timeout" no módulo) nem em `_build_llm`/`_build_direct_llm` passa um
  timeout alternativo. Logo: aos 120s, só o NOSSO lado para de esperar --
  a requisição HTTP para o Ollama continua aberta e a geração continua
  rodando no servidor até terminar por conta própria, e nós nunca a
  fechamos. No log de `diagnostic_menzies`, a chamada imediatamente
  seguinte a cada timeout teve `elapsed_s` de 87,3s e 55,4s (baseline
  ~25-30s) -- consistente com ela enfileirando atrás da geração
  abandonada ainda ocupando o Ollama. A sequência de exatamente 3
  `falha_tecnica` consecutivas que disparou a pausa do Menzies é
  plausivelmente o mesmo efeito em cascata (cada timeout aumenta a
  chance do próximo também estourar, até emendarem 3), não 3
  indisponibilidades independentes do serviço.

**Correção aplicada (bug real, não um critério de design -- corrigida
sem precisar de decisão do pesquisador)**: `is_call_record_reusable`
exigia equivalência de versão do contrato de decisão (ponto 3) mesmo
para chamadas que FALHARAM (`vlm_direto`/`summary`), mas uma falha nunca
tem `decision_contract_parser_version` preenchido (`None` por
construção) -- `None != PARSER_VERSION` reprovava essas falhas como "não
reaproveitáveis", o que faria uma retomada REPETIR 56 chamadas reais já
falhadas (17 vlm_direto + 39 summary, confirmado no log). Corrigido: o
ponto 3 só se aplica quando `call_status == STATUS_OK`.

**Decisão tomada pelo pesquisador (2026-11-17)**: a pausa por 3 falhas
técnicas consecutivas (`vd.AbstentionTracker`) deixou de ser usada por
`scripts/run_full_scale_vlm_matrix.py` para excluir um papel do restante
da matriz -- a causa raiz (fila do Ollama por timeout sem cancelamento
real, não indisponibilidade real nem desempenho do modelo) não
justifica desativar um papel pelo resto do lote. Mudança implementada:

- O laço de execução do executor de escala real não consulta mais
  `tracker.is_paused(role)` -- toda chamada planejada (`missing_calls`)
  recebe uma tentativa real. Um timeout continua gravado como
  `falha_tecnica` na tentativa individual (sem retry, como sempre).
  `vd.AbstentionTracker`/`not_executed_record` não são mais usados por
  este script (preservados intocados em `src/multiagent/vlm_direct.py`
  e ainda usados por `scripts/run_local_vlm_pilot.py`, fora do escopo
  desta decisão).
- Nenhum timeout nem espera arbitrária foi adicionado -- não se mudou o
  valor de `TIMEOUT_S` (120s) nem se inseriu um intervalo de
  resfriamento; a causa (fila do Ollama) não foi corrigida no código, só
  deixou de ser tratada como motivo de exclusão silenciosa de um papel.
- Diagnóstico informativo adicionado (nunca gate): ao final de cada
  `--mode execute`, o script recalcula a maior sequência de
  `falha_tecnica` consecutivas por papel (cumulativo, lendo
  `call_log_main.jsonl` inteiro) e grava em
  `technical_failure_streaks.json`; se algum papel tiver 3+ na
  sequência, imprime um aviso recomendando reiniciar o processo do
  Ollama ANTES de retomar (`--mode execute` de novo) para esvaziar a
  fila -- uma recuperação operacional manual do pesquisador, não uma
  checagem automática via `ollama ps` (que não deve ser usada como
  indicador de servidor ocioso).
- Efeito sobre os 37.289 `nao_executado` já gravados pela execução
  anterior do MedGemma: continuam pendentes (`is_call_record_reusable`
  nunca os trata como reaproveitáveis) -- na próxima `--mode execute`,
  cada um recebe uma tentativa real (não mais pulado por pausa). As
  1.687 tentativas reais já registradas (inclusive as 143 falhas e as 4
  recusas) permanecem preservadas e não são repetidas, graças à correção
  do bug de `decision_contract_parser_version` registrada acima.
