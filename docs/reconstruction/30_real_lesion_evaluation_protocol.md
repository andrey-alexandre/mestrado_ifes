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
  cada um recebe uma tentativa real (não mais pulado por pausa). Dos
  1.687 registros reais já existentes: os 1.540 sucessos e as 4 recusas
  permanecem preservados e NÃO são repetidos (correção do bug de
  `decision_contract_parser_version`, acima). As 143 falhas técnicas,
  porém, foram produzidas pelo mecanismo ANTIGO (geração nunca cancelada
  de verdade) -- ver seção 10 abaixo: a partir da correção do
  cancelamento real, essas 143 ESPECIFICAMENTE voltam a ser tentadas de
  verdade (não são "repetição", são a primeira tentativa sob a
  configuração corrigida -- ver ponto 4 de `is_call_record_reusable`).

## 10. Cancelamento real do timeout (2026-11-17) -- causa raiz e correção

Pesquisador apontou, antes de qualquer retomada do lote: a correção da
seção 9 removeu a PAUSA, mas não resolveu a causa (o timeout ainda só
abandona a espera, sem fechar a chamada) -- correto, confirmado no
código, não só no log.

**Mecanismo real do problema**: `langchain_ollama.ChatOllama._generate`
(usado por `.invoke()`) sempre delega para `_chat_params` com
`"stream": kwargs.pop("stream", True)` -- streaming interno MESMO numa
chamada "não-streaming". Isso significa que um timeout HTTP por LEITURA
(`client_kwargs={"timeout": N}` no `ollama.Client`/httpx) nunca dispara
enquanto o modelo continuar emitindo QUALQUER token dentro da janela --
útil para travamentos reais (nenhum byte chega), inútil para uma geração
lenta mas contínua que passa de 120s no TOTAL (exatamente o padrão
observado no MedGemma). Configurar só esse timeout, portanto, não prova
nada sobre o caso que importa.

**Correção implementada** (`src/agents/agents.py`):
- `_build_llm(model_name, temperature, timeout_s=None)` -- novo
  parâmetro opcional, só afeta o ramo Ollama
  (`ChatOllama(model=model_name, client_kwargs={"timeout": timeout_s})`)
  -- cobre o caso "travou antes do primeiro byte"; `None` (default)
  preserva o comportamento histórico exato para qualquer chamador que
  não passe isto.
- `_invoke_ollama_stream_with_deadline(llm, messages, timeout_s)`
  (nova): consome `llm.stream(messages)` (não `.invoke()`) ela mesma,
  verificando o tempo TOTAL decorrido a cada chunk recebido -- cobre o
  caso "geração lenta mas contínua, só o total excede". Ao exceder,
  fecha explicitamente o gerador (`stream_iter.close()`), propagando um
  `GeneratorExit` até o `with self._client.stream(...) as r:` da
  biblioteca `ollama`, fechando de verdade a conexão HTTP/TCP -- e
  levanta `OllamaStreamDeadlineExceeded` (classe própria, NÃO
  `TimeoutError` builtin -- desde Python 3.11
  `concurrent.futures.TimeoutError` é um alias do `TimeoutError`
  builtin, então usar `TimeoutError` aqui seria capturado pelo primeiro
  `except` de `call_with_timeout` e perderia a mensagem/distinção).
- `_invoke(..., timeout_s=None)` e os 4 agentes
  (`ABCDDiagnosticAgent`/`MenziesDiagnosticAgent`/
  `SPCLDiagnosticAgent`/`SummaryAgent`) ganharam um parâmetro opcional
  `ollama_timeout_s` (default `None`, mesma garantia de compatibilidade)
  que propaga até `_invoke_ollama_stream_with_deadline` -- cobre os 5
  papéis (antes só vlm_direto/summary tinham QUALQUER timeout
  configurável neste nível; ABCD/Menzies/SPCL dependiam só do
  `call_with_timeout` externo, que nunca cancela nada).
- `src/multiagent/vlm_direct.py`: `run_vlm_direct_call`,
  `run_summary_role_call_official_decision_contract` e
  `run_multiagent_role_call` passam `ollama_timeout_s=timeout_s` (só
  para modelos Ollama) na construção do LLM/agente -- os 5 papéis agora
  usam o mesmo mecanismo de cancelamento real.
- **Identificação da configuração preservada** (pedido do pesquisador):
  novo campo `CallRecord.call_mechanism_version` (constante
  `CALL_MECHANISM_VERSION`), gravado em TODO registro. Não é uma versão
  de contrato de decisão -- é a identificação de qual mecanismo de
  cancelamento produziu a tentativa. `is_call_record_reusable` ganhou um
  4º ponto de equivalência: uma FALHA TÉCNICA só é reaproveitável se
  `call_mechanism_version` bater com o atual -- os 143 registros de
  falha do MedGemma (mecanismo antigo, sem este campo) deixam de contar
  como "já tentados" e voltam a ser tentados de verdade sob o mecanismo
  corrigido (não é repetição -- é a primeira tentativa comparável).
  Sucesso/sem-resposta/recusa não exigem este campo.

**Verificação empírica (não executada por este agente -- requer Ollama
real)**: `scripts/verify_ollama_real_cancellation.py`. Mede, por
latência (NUNCA por `ollama ps`/estado interno do servidor -- decisão
explícita do pesquisador): (1) baseline de um prompt rápido sem prazo;
(2) força o corte de um prompt deliberadamente longo com um prazo curto
via `_invoke_ollama_stream_with_deadline`, confirmando que o corte
dispara perto do prazo pedido (não que o prompt terminou por conta
própria); (3) repete o prompt rápido imediatamente após o corte. Se a
latência pós-corte for próxima do baseline, é evidência de cancelamento
real; se for muito maior, é evidência de que o servidor ainda está
ocupado pela geração abandonada. Resultado é evidência circunstancial,
reportado como tal (`interpretation_caveat`), nunca um veredito
definitivo automático.

**Recomendação operacional antes de retomar o lote** (pedido do
pesquisador, não implementado como gate automático): encerrar o
processo do executor e reiniciar o Ollama para limpar qualquer geração
abandonada acumulada ANTES de rodar `--mode execute` de novo. Depois
disso, a retomada continua só os itens ainda não tentados (`missing`),
sem repetir nenhuma tentativa já registrada -- exceto as 143 falhas
técnicas do mecanismo antigo, que são tentadas de verdade pela primeira
vez sob a configuração corrigida (ver acima).

### Comandos

```bash
# Verificação empírica do cancelamento real (fazer ANTES de retomar o lote):
python scripts/verify_ollama_real_cancellation.py --model-tag medgemma1.5:4b-it-q4_K_M

# Reiniciar o Ollama manualmente (fora deste repositório) para limpar
# qualquer geração abandonada acumulada da execução anterior.

# Retomar -- só os itens ainda não tentados; as 143 falhas antigas do
# mecanismo anterior são tentadas de verdade agora, nada mais é repetido:
python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag medgemma1.5:4b-it-q4_K_M
python scripts/run_full_scale_vlm_matrix.py --mode execute --model-tag medgemma1.5:4b-it-q4_K_M
```

## 11. Bug real de representação de conteúdo (Gemini, 2026-11-18) -- correção + reprocessamento sem nova chamada à API

Execução completa do Gemini (38.976/38.976, ver §12 para a reconciliação
de contagens) revelou: **100% das chamadas `vlm_direto`/`summary` com
sucesso (11.992 no total) tinham `decision_status: "field_absent"`** --
o contrato oficial `decisao_final` nunca foi lido com sucesso para este
modelo, apesar do texto correto estar presente na resposta real.

### 11.1 Causa raiz (confirmada no código, não só no log)

`ChatGoogleGenerativeAI` (com metadados de "thought signature" anexados)
devolve `.content` como uma LISTA de blocos --
`[{'type': 'text', 'text': '...texto real...\n\ndecisao_final: indeterminado', 'extras': {'signature': '...'}}]`
-- não uma string simples. O código anterior (`_invoke_direct_llm_text`
em `src/multiagent/vlm_direct.py`, caminho não-Ollama) fazia
`llm.invoke([message]).content` e tratava o resultado como se já fosse
string. Esse valor (uma LISTA, não string) seguia até
`classify_call_outcome` (`vlm_direct.py`), que fazia `text =
str(raw_text)` -- `str()` de uma lista Python ESCAPA as quebras de linha
reais da string interna em `\n` LITERAL (dois caracteres, não um
newline de verdade). O regex de `extract_decisao_final`
(`^decisao_final:` com `re.MULTILINE`) exige newline REAL antes da
âncora `^` -- nunca bate contra `\n` literal. Resultado:
`field_absent` sempre, mesmo quando o modelo escreveu a linha exatamente
no formato certo.

Confirmado que ABCD/Menzies/SPCL (26.976 chamadas `ok`, zero
contaminação) NÃO são afetados -- esses papéis passam por
`with_structured_output()` (`src.agents.agents._invoke`, ramo não-Ollama),
que nunca toca `.content` bruto.

### 11.2 Correção prospectiva (código, afeta só chamadas futuras)

- `src/agents/agents.py`: nova `extract_text_from_llm_content(content)`
  -- aceita `str` (devolvida idêntica, sem nenhuma transformação), `None`
  (idem), ou `list` de blocos (extrai texto dos blocos `{"type": "text",
  "text": ...}` OU strings puras, NA ORDEM ORIGINAL, concatenados sem
  separador -- blocos de outros tipos, como metadados/"extras"/
  assinatura/"thinking", são deliberadamente ignorados, nunca incluídos
  no texto). Qualquer outro tipo levanta `TypeError` explícito -- nunca
  mais um `str()` silencioso de algo desconhecido.
- `src/multiagent/vlm_direct.py`: `_invoke_direct_llm_text` (papéis
  `vlm_direto`/`summary`, ramo Gemini/GPT) e `run_image_sending_control_call`
  (controles do piloto de 12) agora normalizam `.content` com essa
  função antes de devolver o texto.
- `_invoke_ollama_stream_with_deadline` (correção de timeout, seção 10)
  também passou a usar a mesma normalização por chunk, defensivamente --
  nunca observado contaminado no caminho Ollama real, mas sem custo
  adicional de manter consistente.
- Testes: `tests/agents/test_content_normalization.py` (string, lista de
  blocos de texto, lista com metadados/"extras", múltiplos blocos,
  bloco de tipo inesperado, tipo totalmente inesperado -- inclusive
  reproduzindo byte-a-byte a amostra real do log do Gemini).

### 11.3 Reprocessamento dos dados já coletados (NUNCA uma nova chamada à API)

Esta é uma correção de REPRESENTAÇÃO do texto já salvo, não do contrato
diagnóstico -- `extract_decisao_final` não foi alterado em nenhuma
regra. `scripts/reprocess_decisao_final.py`:

- Lê `call_log_main.jsonl` em modo SOMENTE LEITURA -- nunca modifica o
  arquivo original.
- Para cada registro `vlm_direto`/`summary` com `call_status == "ok"`:
  tenta recuperar `raw_response_text` via
  `src.evaluation.content_recovery.recover_raw_response_text`, que usa
  `ast.literal_eval` -- NUNCA `eval()` -- para reconstruir a lista a
  partir do texto já corrompido (`ast.literal_eval` só aceita literais
  Python puros: strings/números/listas/dicts/tuplas/sets/
  True/False/None; nunca executa chamadas de função ou código, mesmo
  com entrada hostil/malformada -- testado explicitamente em
  `tests/evaluation/test_content_recovery.py::
  test_never_uses_eval_malicious_input_never_executes_code`). Se
  `literal_eval` tiver sucesso e o resultado for uma lista, reusa a
  MESMA `extract_text_from_llm_content` da correção prospectiva (nunca
  uma segunda implementação). Casos sem sucesso: distinguidos entre
  "nunca foi contaminado, é prosa normal" (`literal_eval` falha e o
  texto não parece sintaxe de lista) e "estava contaminado mas ficou
  irrecuperável" (`literal_eval` falha mas o texto começa com `[{`, OU
  `literal_eval` recupera uma lista com estrutura de bloco inesperada)
  -- sempre sinalizado explicitamente no campo `recovery_status`, nunca
  omitido ou adivinhado.
- Reaplica `extract_decisao_final` (inalterado) sobre o texto recuperado
  (ou o texto original, se já estava limpo) -- fonte oficial do
  veredito continua sendo exatamente a mesma função, sem fallback para
  `classified_status`/`classified_status_v2` (v1/v2).
- Grava um arquivo DERIVADO (`call_log_main.decisao_final_reprocessed.jsonl`,
  nunca sobrescreve o original) com, por registro: identificação da
  chamada (`model_name`/`role`/`condition`/`lesion_id`/`image_id`/
  `attempt_timestamp_utc`), status/decisão ANTERIOR, `recovery_status`,
  texto recuperado, e status/decisão REPROCESSADOS -- os dois lados
  sempre visíveis, nunca um sobrescrevendo o outro.
- Testes: `tests/evaluation/test_content_recovery.py` -- cobre
  exatamente os 4 cenários pedidos (string já limpa, lista de blocos
  contaminada real, lista com metadados/"extras", campo realmente
  ausente mesmo após recuperação bem-sucedida), mais casos de
  texto truncado/irrecuperável e confirmação de que `eval()` nunca é
  usado.

### 11.4 Comandos

```bash
# 1. Testes (confirmar a normalização e a recuperação antes de reprocessar o log real):
python -m pytest tests/agents/test_content_normalization.py tests/evaluation/test_content_recovery.py -v

# 2. Reprocessamento do log real do Gemini (NENHUMA chamada à API):
python scripts/reprocess_decisao_final.py \
    --call-log experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/call_log_main.jsonl \
    --output-dir experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/reprocessing
```

## 12. Reconciliação de contagens: tentativas históricas x resultado final por chave

Ao avaliar o Gemini, uma soma aparentemente inconsistente foi observada:
38.968 `ok` + 72 `falha_tecnica` = 39.040, mais que as 38.976 chamadas
planejadas. Isto NÃO é um erro de dado -- é a diferença entre duas
contagens que não são a mesma coisa:

- **Tentativas históricas**: toda linha de `call_log_main.jsonl`,
  incluindo tentativas SUPERSEDIDAS por uma invocação posterior (ex.:
  um `nao_executado` de quando a pausa existia, seguido por uma
  tentativa real depois que a pausa foi removida). O log do Gemini tem
  56.055 linhas no total -- mais que as 38.976 chamadas planejadas,
  porque passou por múltiplas invocações antes e depois das correções
  desta entrega.
- **Resultado final por chave planejada**: um único veredito por
  `(role, condition, lesion_id, image_id)` -- por construção, soma
  exatamente a 38.976 (o número de chamadas planejadas), nunca mais,
  nunca menos.

`scripts/reprocess_decisao_final.py::reconcile_historical_vs_final`
calcula as duas, nunca somando a primeira como se fosse a segunda.
Verificado separadamente (contagem manual, antes do script existir):
**zero chaves têm 2+ entradas `ok`** -- nenhuma chamada bem-sucedida foi
repetida -- mas isso, por si só, não resolve a contagem de 39.040: a
causa é a presença de linhas supersedidas (`nao_executado` seguido de
tentativa real, ou falha do mecanismo antigo seguida de tentativa real
sob o mecanismo corrigido), que inflam a contagem de linhas sem inflar
o número de vereditos finais.

**Registrado explicitamente, nunca escondido do histórico**: chaves com
mais de uma tentativa REAL (não `nao_executado`) são reportadas em 3
categorias distintas no relatório do reprocessamento:
  (a) `exception_old_mechanism_retried` -- falha do mecanismo antigo de
      cancelamento, seguida de uma tentativa real sob o mecanismo
      corrigido (seção 10) -- **EXCEÇÃO OPERACIONAL DOCUMENTADA** à
      regra de uma tentativa, por mudança de configuração, nunca um
      retry disfarçado da mesma tentativa;
  (b) `historical_duplicate_bug_pattern` -- as duas tentativas são do
      mecanismo antigo (bug já corrigido do `decision_contract_parser_version`
      exigido mesmo para falhas, seção 9);
  (c) `unexplained_duplicate_same_mechanism` -- qualquer caso que não se
      encaixe em (a) nem (b) (duas tentativas reais sob o MESMO
      mecanismo não-nulo) -- reportado para investigação manual, NUNCA
      descartado silenciosamente.

Resultado real do reprocessamento do log do Gemini (confirmado, não
estimado): 64 chaves com múltiplas tentativas reais -- **47**
`exception_old_mechanism_retried` + **17** `historical_duplicate_bug_pattern`
+ **0** `unexplained_duplicate_same_mechanism`. Todas as 64 sequências
são `[falha_tecnica, ok]` (nunca mais que 2 tentativas reais por chave).

## 13. Fechamento da inferência do Gemini (2026-11-18) -- 8 falhas residuais, correção de `call_mechanism_version`

### 13.1 Decisão do pesquisador

As 11.992 decisões recuperadas pelo reprocessamento (seção 11) são a
fonte oficial dos vereditos `decisao_final` do Gemini a partir desta
entrega -- `experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/
reprocessing/call_log_main.decisao_final_reprocessed.jsonl`, NUNCA mais
`call_log_main.jsonl` original (que mantém `decision_status:
"field_absent"` para essas 11.992, preservado intocado para auditoria).

A inferência do Gemini está FECHADA: resultado final por chave =
38.968 `ok` + **8 `falha_tecnica`** = 38.976 (reconciliação exata, ver
seção 12). As 8 falhas são erros de rede REAIS (não relacionados à
cascata do Ollama -- Gemini nunca passa por
`_invoke_ollama_stream_with_deadline`), confirmadas por leitura direta
do log:

| role | condition | lesion_id | image_id | causa |
|---|---|---|---|---|
| diagnostic_abcd | s2 | HAM_0006808 | ISIC_0027438 | `timeout_excedido_120s` |
| diagnostic_abcd | s2 | HAM_0006944 | ISIC_0030762 | `RemoteProtocolError: Server disconnected without sending a response` |
| diagnostic_abcd | s2 | HAM_0007325 | ISIC_0026267 | `RemoteProtocolError: Server disconnected without sending a response` |
| diagnostic_menzies | s2 | HAM_0006939 | ISIC_0031675 | `ReadError: [Errno 32] Broken pipe` |
| diagnostic_spcl | s1 | HAM_0006846 | ISIC_0027199 | `RemoteProtocolError: Server disconnected without sending a response` |
| diagnostic_spcl | s1 | HAM_0006903 | ISIC_0031487 | `timeout_excedido_120s` |
| summary | s1 | HAM_0006969 | ISIC_0028007 | `RemoteProtocolError: Server disconnected without sending a response` |
| summary | s1 | HAM_0007325 | ISIC_0026267 | `RemoteProtocolError: Server disconnected without sending a response` |

**Decisão explícita do pesquisador: NÃO forçar retentativa.** São falhas
reais (desconexão/timeout da API), e a regra de não-retry do protocolo
vale igualmente para qualquer provedor -- continuam registradas como
`falha_tecnica`, definitivamente, sem nenhum novo veredito pendente.
Isto deixa, do lado dos papéis diagnósticos (ABCD/Menzies/SPCL -- não
têm `decisao_final` próprio): 4 lesões/condições com um dos 3 pareceres
ausente (`HAM_0006808`-s2, `HAM_0006944`-s2, `HAM_0007325`-s2,
`HAM_0006939`-s2, `HAM_0006846`-s1, `HAM_0006903`-s1 -- 6 ocorrências em
4 lesões distintas, ver seção 14.2 sobre o efeito disso no sumarizador);
e do lado do sumarizador, 2 lesões (`HAM_0006969`-s1, `HAM_0007325`-s1)
sem NENHUM `decisao_final` multiagente para aquela condição -- essas 2
são as únicas exclusões reais do denominador de S1 (seção 14.2).

### 13.2 Correção de `call_mechanism_version` -- distinção por provedor

**Bug real encontrado e corrigido**: `call_mechanism_version` era
gravado incondicionalmente em TODO `CallRecord`, inclusive para
Gemini/GPT -- mas o mecanismo que esse campo identifica
(`_invoke_ollama_stream_with_deadline`, a correção do timeout com
cancelamento real, seção 10) é ESPECÍFICO do Ollama; para modelos via
API, nada sobre cancelamento mudou nesta entrega (SDK próprio). Isso
tinha DUAS consequências:

1. **Rótulo historicamente impreciso** (não um erro de dado, um erro de
   rótulo): as 8 falhas do Gemini acima aparecem no log com
   `call_mechanism_version: "ollama_stream_deadline_real_cancel_2026_11_17"`,
   como se o mecanismo de cancelamento do Ollama se aplicasse a elas --
   nunca se aplicou. **Preservado como está no arquivo histórico**
   (`call_log_main.jsonl`, nunca reescrito) -- documentado aqui como
   imprecisão conhecida, não corrigido retroativamente nos dados.
2. **Bug funcional real em `is_call_record_reusable`** (ponto 4): a
   checagem de equivalência deste campo era aplicada a QUALQUER
   `falha_tecnica`, inclusive de API -- mas como Gemini/GPT nunca têm
   esse campo com significado real, uma correção futura deste campo
   faria com que falhas de API ficassem PERMANENTEMENTE "não
   reaproveitáveis", forçando retentativa a cada nova invocação
   (violando a própria política de não-retry).

**Correção aplicada** (`src/multiagent/vlm_direct.py`): `CallRecord`
agora só grava `call_mechanism_version` para modelos Ollama
(`CALL_MECHANISM_VERSION if _is_ollama(model_name) else None`) nas 3
funções que produzem `CallRecord` (`run_vlm_direct_call`,
`run_summary_role_call_official_decision_contract`,
`run_multiagent_role_call`). `is_call_record_reusable` (ponto 4) só
aplica a checagem de equivalência quando `_is_ollama(model_name)` é
verdadeiro -- para API, uma falha técnica real é SEMPRE reaproveitável
(nunca retentada), independente do valor histórico deste campo.
**Nenhum dado histórico foi reescrito** -- só o código prospectivo
(registros futuros de Gemini/GPT nunca mais recebem este rótulo
Ollama-específico).

## 14. Correções ao relatório analítico (2026-11-18) -- cobertura x taxa de sucesso, denominadores de S1/S2

### 14.1 99,98% é taxa de sucesso de CHAMADA, não cobertura diagnóstica binária

Relatado anteriormente como "99,98% com decisão válida" -- impreciso.
Correção: 99,98% (38.968/38.976) é a taxa de SUCESSO DE CHAMADA (a
tentativa completou e produziu um `decisao_final` lido com sucesso,
incluindo `indeterminado`). As **5 métricas binárias** planejadas
(sensibilidade/especificidade/etc., melanoma vs não-melanoma) só se
aplicam a decisões categóricas -- `melanoma`/`nao_melanoma` -- NUNCA a
`indeterminado`, que fica estruturalmente FORA das 5 métricas binárias,
não é um erro nem uma lacuna, é a terceira categoria do contrato por
desenho (ver `explicit_decision_contract.py`).

Das 38.968 chamadas bem-sucedidas, a divisão real é:

| decisao_final | N | % das chamadas ok |
|---|---|---|
| nao_melanoma | 5.812 | 48,5% |
| indeterminado | 4.114 | 34,3% |
| melanoma | 2.066 | 17,2% |

As **4.114 indeterminações ficam fora das 5 métricas binárias** -- são
reportadas SEPARADAMENTE, por sistema (papel) e por condição, nunca
diluídas no denominador binário nem tratadas como "negativo" por
default:

| role | condition | n_ok_total | melanoma | nao_melanoma | indeterminado | % indeterminado |
|---|---|---|---|---|---|---|
| vlm_direto | s0 | 3.000 | 1.201 | 676 | 1.123 | 37,4% |
| summary | s0 | 3.000 | 340 | 1.694 | 966 | 32,2% |
| summary | s1 | 2.998 | 270 | 1.721 | 1.007 | 33,6% |
| summary | s2 | 2.994 | 255 | 1.721 | 1.018 | 34,0% |

(Contado diretamente de `call_log_main.decisao_final_reprocessed.jsonl`
-- cada linha soma exatamente ao `n_ok_total` da sua condição/papel,
confirmado. O relatório consolidado automático --
`decisao_final_reprocessing_report.json` -- ainda agrega
`reprocessed_decisao_final_breakdown_among_status_ok` de forma GLOBAL,
não por condição/papel; `build_report` em
`scripts/reprocess_decisao_final.py` precisa dessa quebra adicional
como próximo passo de código -- a tabela acima foi obtida por leitura
direta do arquivo derivado já existente, sem nenhuma nova inferência.)

**Observação preliminar, sem interpretação de significância**:
VLM-direto tem uma taxa de `melanoma` bruta (40,0%) visivelmente maior
que o sumarizador multiagente em qualquer condição (11,3% em S0 a 8,5%
em S2) -- essa
diferença é exatamente o tipo de achado que a comparação VLM-direto S0
× multiagente S0 (seção 14.4) precisa quantificar, mas só depois que a
regra pendente de subconjuntos binários diferentes for decidida -- não
interpretado aqui como conclusão.

### 14.2 Denominadores por condição -- S1 e S2 têm causas DIFERENTES, nunca confundidas

| condição | planejado | resultado final `ok` | diferença | causa |
|---|---|---|---|---|
| S0 (vlm_direto) | 3.000 | 3.000 | 0 | -- |
| S0 (summary) | 3.000 | 3.000 | 0 | -- |
| S1 (summary) | 3.000 | **2.998** | **2** | 2 falhas técnicas REAIS do papel `summary` (seção 13.1: `HAM_0006969`, `HAM_0007325`) -- NUNCA um problema estrutural de máscara; confirmado no manifesto `s0_s1_s2_preparation_manifest.csv`: **zero** lesões têm `status_s1 == "indisponivel"` (as 6 ocorrências de `"indisponivel"` no arquivo são TODAS `status_s2`) -- S1 está estruturalmente disponível para as 3.000 lesões, sem exceção. |
| S2 (summary) | 2.994 | 2.994 | 0 (do planejado) | 6 lesões estruturalmente indisponíveis (`mascara_predita_vazia`, já conhecidas desde a Etapa 4/stage 4 -- `HAM_0005998`, `HAM_0004545`, `HAM_0003874`, `HAM_0005250`, `HAM_0002639`, `HAM_0007614`) nunca chegam a ser PLANEJADAS para S2 -- confirmado: nenhuma das 2 falhas de S1 nem das 4 falhas restantes em S2 (diagnostic_abcd/menzies) coincide com essas 6 lesões (zero sobreposição verificada). |

**Confirmação de independência das duas causas**: as 4 lesões com falha
técnica real em condição S2 (`HAM_0006808`, `HAM_0006944`, `HAM_0007325`,
`HAM_0006939` -- papéis ABCD/Menzies, não `summary`) são DIFERENTES das
6 lesões com máscara vazia -- confirmado por comparação direta das
`lesion_id`/`image_id` contra o manifesto S0/S1/S2. Essas 4 falhas NÃO
reduzem o denominador de `summary`-S2 (2.994/2.994 -- o sumarizador
rodou e teve sucesso para essas 4 lesões mesmo com um parecer
diagnóstico ausente) -- são, em vez disso, casos de **consolidação
parcial** (ver abaixo), não de decisão final ausente.

**Denominadores corretos para as métricas binárias por condição**
(excluindo `indeterminado`, que fica fora por desenho -- ver 14.1):
  - S0: denominador de chamadas válidas = 3.000 (vlm_direto) / 3.000 (summary).
  - S1: denominador de chamadas válidas = **2.998** (não 3.000) -- as 2
    falhas técnicas reais NUNCA entram no denominador binário nem são
    tratadas como qualquer categoria de decisão.
  - S2: denominador de chamadas válidas = **2.994** (não 3.000) -- as 6
    lesões estruturalmente indisponíveis nunca entram no denominador
    (nunca foram chamadas).

### 14.3 Nota sobre consolidação parcial (ABCD/Menzies/SPCL) -- não é o mesmo problema do sumarizador

As 6 ocorrências de falha técnica real em papéis diagnósticos (ABCD×3,
Menzies×1, SPCL×2, distribuídas em 4 lesões distintas -- ver tabela da
seção 13.1) significam que, para essas 4 lesões/condições, o
sumarizador recebeu 2 de 3 pareceres (não 3) -- mas o sumarizador EM SI
teve sucesso e produziu um `decisao_final` válido (confirmado: nenhuma
dessas 4 lesões está entre as 2 falhas de `summary`). Isto é
"consolidação parcial" (mesmo conceito da primeira entrega da Etapa 5,
ver [[project_multiagent_stage5]]) -- não reduz nenhum denominador
binário, mas é uma nota de qualidade de dado relevante para quem for
interpretar os vereditos multiagente dessas 4 lesões/condições
especificamente.

### 14.4 Próximo passo -- SEM nova inferência

Com as contagens reconciliadas e as decisões recuperadas, o próximo
passo são as métricas por condição (S0/S1/S2) usando os denominadores
corretos acima -- nenhuma chamada nova a modelo necessária. As
diferenças PAREADAS entre as 5 métricas (comparação VLM-direto S0 ×
multiagente S0, CNN S0 × multiagente S2) continuam dependendo da regra
ainda pendente (não resolvida por default de código, decisão do
pesquisador) para os casos em que os sistemas comparados produzem
decisões binárias sobre SUBCONJUNTOS DIFERENTES de lesões (ex.: uma
lesão com `indeterminado` num sistema e `melanoma`/`nao_melanoma` no
outro, ou uma falha técnica num sistema e sucesso no outro) -- essa
regra era e continua pendente, não deve ser resolvida por um default de
código (ver [[project_multiagent_stage5]], "comparação cross-system de
5 métricas sob subconjuntos binários diferentes -- explicitamente
deixado pendente pelo pesquisador").

## 15. Redução de escopo da avaliação local por viabilidade computacional (2026-11-18)

### 15.1 A decisão

**Decisão OPERACIONAL do pesquisador, enquanto aguarda retorno do
orientador -- APROVAÇÃO DO ORIENTADOR AINDA PENDENTE, não resolvida por
este registro.** Motivada por tempo de processamento real observado
(MedGemma: ~16h26min de parede para 1.687 tentativas reais na primeira
execução, projeção de dias para a matriz completa de 3.000 -- ver §9),
NUNCA por seleção/exclusão baseada em desempenho dos modelos.

Qwen3-VL e MedGemma 1.5 4B passam a ser avaliados em **duas camadas**:
  - **Avaliação principal**: inalterada no desenho (matriz completa
    aprovada, 3.000 lesões, VLM-direto S0 + multiagente S0/S1/S2) --
    continua existindo como destino de longo prazo, não descontinuada,
    só momentaneamente inviável em tempo.
  - **Avaliação complementar**: NOVA, reduzida -- só multiagente em S2
    (ABCD/Menzies/SPCL/sumarizador), sobre um subconjunto de **300**
    lesões distintas das 3.000 congeladas. Sem VLM-direto, sem S0, sem
    S1.

Gemini e GPT permanecem na matriz principal de 3.000, sem alteração. A
CNN já foi avaliada (docs 28/29) -- não é rechamada em nenhum ponto
desta entrega.

**Perdas explícitas decorrentes da redução** (registradas, não
escondidas):
  - Para Qwen3-VL/MedGemma, as comparações S0×S1×S2 (efeito da condição
    de apresentação) e VLM-direto×multiagente (mesma lesão, dois modos)
    NÃO são possíveis na avaliação complementar -- só existem, quando
    existirem, na avaliação principal (ainda pendente de viabilidade).
  - **Maior incerteza estatística esperada** na avaliação complementar
    em relação à principal, por dois motivos combinados: (a) n=300 é
    10x menor que n=3.000; (b) a cota de Hamilton para a categoria
    `mel` (melanoma) sobre a composição real da amostra congelada
    (247/3.000 lesões, 8,23%) projeta **25 lesões de melanoma em 300**
    (8,33%) -- um número pequeno em termos absolutos, que amplia
    especialmente o intervalo de confiança da **sensibilidade**
    (depende só dos verdadeiros positivos entre essas ~25 lesões, não
    das ~275 restantes). Isto é uma limitação estrutural do tamanho
    reduzido, não um defeito da amostragem.
  - Saídas locais já existentes da matriz principal (parciais, ver §9)
    são PRESERVADAS sem alteração -- nunca movidas, apagadas ou
    mescladas no novo recorte. O que estiver fora do novo recorte de
    300 lesões continua no histórico como execução parcial adicional da
    avaliação principal, disponível para quando ela for retomada.

### 15.2 Seleção do subconjunto de 300 -- `scripts/select_local_300_lesion_s2_subset.py`

Reusa `src.data_audit.sampling.compute_quotas_largest_remainder`
(método de Hamilton/maiores restos, MESMA função já usada na amostra
congelada de 3.000 -- nunca reimplementada) e
`src.data_audit.sampling.ELIGIBLE_DX_CATEGORIES` (as 7 categorias
diagnósticas do HAM10000). Cotas calculadas sobre a composição REAL da
amostra de 3.000 (não da população HAM10000 completa, já que a seleção
é um subconjunto DELA):

| categoria (dx) | n elegível (de 3.000) | cota exata | cota final (Hamilton) |
|---|---|---|---|
| nv | 2.170 | 217,0 | 217 |
| bkl | 292 | 29,2 | 29 |
| mel | 247 | 24,7 | **25** |
| bcc | 131 | 13,1 | 13 |
| akiec | 92 | 9,2 | 9 |
| vasc | 39 | 3,9 | 4 |
| df | 29 | 2,9 | 3 |
| **total** | **3.000** | | **300** |

(Calculado manualmente para confirmação cruzada a partir da composição
real de `experiments/sampling/sample_manifest.csv`, 2026-11-18 --
`scripts/select_local_300_lesion_s2_subset.py` recalcula isso via
código, nunca lê esta tabela.)

Sorteio (semente `20261118`, nova e distinta de todas as já usadas no
projeto): dentro de cada categoria, `random.Random(seed).sample()` sobre
o conjunto ORDENADO de `lesion_id` elegíveis -- determinístico,
reproduzível. **A imagem de cada lesão sorteada é EXATAMENTE a já
existente no manifesto de 3.000** -- nunca um novo sorteio de imagem
(diferente de `src.data_audit.sampling.sample_lesions`, que sorteia uma
imagem por lesão a partir da população bruta -- função nova e
necessária aqui, não uma duplicação indevida, porque o formato de
entrada é diferente: lesões já emparelhadas 1:1 com imagem, não a
população multi-imagem do HAM10000 bruto).

**Nunca consulta predições, status de execução, latência ou qualidade
de máscara** -- a seleção lê só `lesion_id`/`dx` do manifesto congelado.
Lesões com máscara predita vazia (`status_s2 == "indisponivel"`, as 6
já conhecidas desde a Etapa 4) podem ser sorteadas normalmente -- a
política vigente se aplica no PLANO DE CHAMADAS do executor (nunca na
seleção), preservando-as no denominador de 300 (não substituídas, não
excluídas).

Registro (`local_300_s2_subset_registry.json`): semente, tabela de
cálculo de Hamilton completa, SHA-256 do manifesto de origem (conferido
contra `src.multiagent.presentation.REGISTERED_MANIFEST_SHA256`, o
mesmo identificador oficial do projeto), SHA-256 do subconjunto gerado,
composição final por `dx`. Nunca sobrescreve silenciosamente um
subconjunto já registrado -- compara hashes antes de gravar, exige
`--force` explícito para substituir um conteúdo diferente.

Verificação independente: `scripts/verify_local_300_s2_subset.py` --
REGERA a seleção do zero (mesma semente) e compara byte a byte contra o
arquivo em disco (reprodutibilidade real, não só contagens), confirma
`image_id` idêntico ao manifesto de origem lesão a lesão, unicidade,
SHA-256, e cotas de Hamilton recalculadas.

### 15.3 Executor adaptado -- `scripts/run_full_scale_vlm_matrix.py --scope local_s2_subset`

Generalização (não duplicação): `src.multiagent.vlm_direct.
build_full_scale_planned_calls` ganhou dois parâmetros opcionais
(`conditions`, `include_vlm_direto`), default = comportamento histórico
exato (`("s0","s1","s2")`, `True`) -- nenhum chamador existente
(Gemini/GPT) é afetado. O executor ganhou `--scope {full,
local_s2_subset}` (default `full`, inalterado) -- `local_s2_subset`
chama `build_full_scale_planned_calls(rows_300, conditions=("s2",),
include_vlm_direto=False)`, pula a checagem/pareamento de CNN (não
aplicável -- CNN já avaliada, nunca rechamada neste escopo), e grava em
`experiments/multiagent/local_300_s2_subset_matrix/` -- diretório
SEMPRE distinto do diretório da matriz de 3.000
(`full_scale_matrix/`), com uma checagem operacional que IMPEDE
`--output-dir` de apontar para lá neste escopo (nunca retoma
acidentalmente a matriz de 3.000).

Até **1.200 chamadas por modelo** (300 lesões × 4 papéis multiagente),
descontadas as condições S2 estruturalmente indisponíveis já previstas
pelo protocolo (máscara vazia). Preservados sem alteração: contrato
`decisao_final`, parâmetros/prompts/timeout, extração de template do
MedGemma, mecanismo de cancelamento real do timeout Ollama (seção 10).

**Reaproveitamento automático de resultados anteriores**: o executor
neste escopo lê, além do seu próprio log novo, o log JÁ EXISTENTE da
matriz completa (`full_scale_matrix/<modelo>/call_log_main.jsonl`) --
SÓ LEITURA, nunca escrito nem alterado -- para que qualquer lesão do
subconjunto de 300 que já tenha um resultado real compatível (mesmo
modelo, mesma lesão/imagem/condição S2, mesmo mecanismo de execução --
`is_call_record_reusable`, seção 13, pontos 1-4, sem nenhuma regra
nova) seja tratada como já coberta, sem reexecução. Placeholders
`nao_executado` continuam pendentes (nunca reaproveitados). Falhas do
mecanismo antigo seguem a MESMA exceção operacional já documentada
(seção 10/13) -- nenhuma regra de retentativa nova foi criada.

**Dependência da síntese preservada por construção** (nenhuma lógica
nova necessária): a função de seeding `_index_existing_diagnoses` já
indexa por `(lesion_id, image_id, condition)` exato e só a partir de
registros `call_status == "ok"` reais -- um sumarizador retomado nunca
recebe pareceres de uma lesão/condição/execução diferente, porque a
chave de busca é sempre exatamente a mesma da chamada em questão. Uma
vez que um sumarizador tenha `ok` registrado, ele NUNCA é re-rodado
(mesmo que um ABCD/Menzies/SPCL da mesma lesão seja retentado depois com
sucesso) -- o resultado já fechado nunca é silenciosamente "atualizado"
com pareceres de uma tentativa posterior, prevenindo exatamente a
mistura que o pesquisador pediu para evitar.

**Checagens preservadas, nenhuma nova reintroduzida**: só problemas
operacionais (manifesto, imagem, config/credencial, diretório gravável,
e agora também a restrição de escopo/diretório) interrompem a execução
-- revisão qualitativa, erros de controle visual, indeterminação e
percentual de respostas válidas continuam sem ser gates, exatamente como
decidido em 2026-11-16 (seção 1).

### 15.4 Exportação pareada -- `scripts/export_local_300_subset_comparison.py`

Produz uma linha por lesão (300 linhas) com o status de CADA sistema --
CNN S0 (já avaliada, só filtrada ao subconjunto), Gemini multiagente S2
(fonte OFICIAL: o arquivo REPROCESSADO de `decisao_final`, seção 11 --
nunca o log original, que mantém `field_absent`; rastreabilidade ao log
original preservada pelos mesmos campos de identificação), GPT
multiagente S2 (lido do escopo `full` quando existir), Qwen3-VL e
MedGemma multiagente S2 (união SOMENTE-LEITURA do escopo
`local_s2_subset` com o escopo `full`). Nenhuma chamada nova a
modelo/API. Resultado ainda não existente para algum sistema/lesão é
marcado explicitamente `"pendente"` -- nunca omitido, nunca convertido
em "falha" ou "negativo".

**Restrição deliberada**: os 5 sistemas são TODOS filtrados ao MESMO
subconjunto de 300 -- esta exportação nunca mistura métricas locais de
300 com métricas de outros sistemas calculadas nas 3.000 (decisão
explícita do pesquisador). Preserva a distinção entre resposta válida
(`call_status`), decisão binária (`melanoma`/`nao_melanoma`),
indeterminação (`indeterminado`, sempre distinto de negativo), falha, e
ausência de execução (`pendente`). **A regra pendente para comparar as
5 métricas binárias quando os sistemas produzem decisões sobre
subconjuntos diferentes continua NÃO resolvida** -- nenhum default de
código decide isso; o script só exporta os dados pareados com a
classificação completa preservada.

### 15.5 Comandos

```bash
# 1. Gerar e verificar o subconjunto de 300 lesões:
python scripts/select_local_300_lesion_s2_subset.py
python scripts/verify_local_300_s2_subset.py

# 2. Planejamento (sem chamadas a modelo) -- mostra reaproveitamento/faltantes:
python scripts/run_full_scale_vlm_matrix.py --mode plan --scope local_s2_subset --model-tag qwen3-vl:8b-instruct-q4_K_M
python scripts/run_full_scale_vlm_matrix.py --mode plan --scope local_s2_subset --model-tag medgemma1.5:4b-it-q4_K_M

# 3. Execução -- separada por modelo, pode ser interrompida (Ctrl+C):
python scripts/run_full_scale_vlm_matrix.py --mode execute --scope local_s2_subset --model-tag qwen3-vl:8b-instruct-q4_K_M
python scripts/run_full_scale_vlm_matrix.py --mode execute --scope local_s2_subset --model-tag medgemma1.5:4b-it-q4_K_M

# 4. Retomada -- comando IDÊNTICO, só os itens ainda pendentes:
python scripts/run_full_scale_vlm_matrix.py --mode execute --scope local_s2_subset --model-tag medgemma1.5:4b-it-q4_K_M

# 5. Exportação pareada (depois que houver algum resultado real --
#    pode ser rodada em qualquer ponto, resultados ausentes ficam "pendente"):
python scripts/export_local_300_subset_comparison.py
```
