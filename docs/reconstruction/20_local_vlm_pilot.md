# 20 — Piloto local de VLMs via Ollama (Etapa 5, segunda entrega — REWORK)

STATUS: implementado, aguardando execução do pesquisador. Nada deste
pacote foi executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) — nenhum `python`, `pytest`, `git`
ou `ollama` foi chamado. Tudo abaixo é entregue como script para o
pesquisador rodar na própria máquina (Ollama + modelos locais).

Este é o REWORK da segunda entrega original (commit `b097037` em
`feature/inspection`), incorporando as 10 decisões que o pesquisador
fechou sobre o primeiro rascunho (ver seção 5). Mudanças de arquitetura
relevantes (não só cosméticas) estão listadas na seção 5 e no topo de
`src/multiagent/vlm_direct.py`.

Zero chamadas a OpenAI/Gemini nesta entrega — só Ollama. O módulo
`src/multiagent/presentation.py` (primeira entrega da Etapa 5) não foi
alterado; só importado/reusado.

## 1. Arquivos criados/alterados

Criados/alterados nesta entrega (REWORK):
- `src/multiagent/vlm_direct.py` — camada de chamada a VLM (VLM-direto + multiagente-sobre-Ollama), controles POR PAPEL VISUAL, abstenção, classificação A34, nova regra de swap, template de revisão manual.
- `scripts/run_local_vlm_pilot.py` — driver reescrito para o fluxo de DOIS ESTÁGIOS (controles+template → revisão manual → matriz principal).
- `scripts/run_pilot12_sample.py` — designação dos "4 de 12" reescrita (estratificada por grupo diagnóstico + validade de máscara de referência).
- `tests/multiagent/test_vlm_direct.py` — testes reescritos/estendidos para todas as funções alteradas (os testes antigos de swap monotônico, de 3 controles compartilhados, e de designação alfabética foram removidos — testavam comportamento substituído, não só estendido).

Não alterados nesta entrega (só importados/reusados): `src/multiagent/presentation.py`, `src/agents/agents.py`, `src/agents/schemas.py`, `src/evaluation/case_status.py`, `src/data_audit/sampling.py`, `scripts/run_s2_mask_generation.py`, `scripts/run_s2_mask_generation_pilot12.py`, `scripts/run_pilot12_image_preparation.py`.

Alterados na primeira entrega (mantidos): `requirements.txt` — `psutil`, `ollama`.

Entrega adicional pós-execução real do Estágio 1 (2026-11-13) — parser v2
"conclusão final", ver seção 9 abaixo: `src/evaluation/case_status_v2.py`
(novo), `scripts/run_reclassify_pilot_with_v2_parser.py` (novo),
`tests/evaluation/test_case_status_v2.py` (novo), e alterações aditivas em
`src/multiagent/vlm_direct.py`/`scripts/run_local_vlm_pilot.py` (campo
`classified_status_v2` ao lado de `classified_status`, nunca o
substituindo). `src/evaluation/case_status.py` continua intocado.

## 2. Sequência exata de comandos

Diretório de trabalho para todos: `C:\Projetos\IFES\mestrado_ifes`.

```
# -1. Pré-requisito de infraestrutura (NENHUM script deste pacote faz isso
#     por você -- ChatOllama/ollama.show() só se conectam a um servidor já
#     rodando com o modelo já baixado; nunca sobem o servidor nem puxam o
#     modelo sozinhos):
ollama serve                                      # se o Ollama não estiver rodando como serviço já
ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama pull medgemma1.5:4b-it-q4_K_M
ollama list                                        # confirme que a tag aparece EXATAMENTE como em --model-tag
# (só se necessário, modelo de contingência -- ver seção 7)
ollama pull qwen2.5vl:7b-q4_K_M

# 0. Dependências novas
pip install psutil ollama

# 1. Amostragem dos 12 casos fora da amostra congelada + designação estratificada dos 4 para S0/S1
python scripts/run_pilot12_sample.py \
    --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000 \
    --ham10000-ref-masks-dir "D:/Andrey/Mestrado/dataverse_files/HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl"

# 2. Máscaras S2 para os 12 (mesmo checkpoint/limiar congelados)
python scripts/run_s2_mask_generation_pilot12.py --device mps \
    --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
# (ajuste --device para cpu/cuda/mps conforme a máquina; --ham10000-images-dir já é o default no Windows)

# 3. Imagens de apresentação S0/S1/S2 para os 12
python scripts/run_pilot12_image_preparation.py

# 4. Piloto de VLM -- DOIS ESTÁGIOS por modelo (Ollama já rodando localmente, modelos já baixados)

# 4a. ESTÁGIO 1 -- controles de envio de imagem (12 chamadas, 4 papéis x 3) +
#     controles de síntese (4 chamadas) + template de revisão manual.
#     PARA SEMPRE ao final (nunca prossegue à matriz na mesma invocação).
python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M

# 4b. Pesquisador revisa manualmente os 4 cenários e preenche o veredito em:
#     experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/synthesis_manual_review_template.json
#     (campo "manual_review_verdict": "sem_achado_novo" ou "achado_novo_encontrado"
#     para CADA um dos 4 cenários -- strings exatas, ver seção 5).

# 4c. ESTÁGIO 2 -- matriz principal (92 chamadas), só prossegue se os 4
#     vereditos forem "sem_achado_novo" E os controles de imagem já tiverem passado.
python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M \
    --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/synthesis_manual_review_template.json

# Repita 4a-4c para o segundo modelo:
python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M
# (revisar e preencher o template)
python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M \
    --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/medgemma1.5_4b-it-q4_K_M/synthesis_manual_review_template.json

# 4d. SOMENTE SE necessário (ver seção 7): modelo de contingência, mesmo fluxo de 2 estágios
python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M
python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M \
    --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/qwen2.5vl_7b-q4_K_M/synthesis_manual_review_template.json
```

Total de chamadas planejadas por modelo (ESTÁGIO 1 + ESTÁGIO 2 combinados)
= **108** = 92 (matriz, `assert` inalterado em
`vd.build_planned_calls`) + 16 de controle (12 de envio de imagem + 4 de
síntese). **108 é um TETO OPERACIONAL de 4h para a execução completa de
um modelo** (os dois estágios), não uma meta de desempenho — passar do
teto não estabelece por si só viabilidade do modelo para os 3.000 casos
reais (ver a projeção de duração do `report.md` do ESTÁGIO 2, seção 8
abaixo).

## 3. Comando de testes

```
python -m pytest tests/multiagent/test_vlm_direct.py -v
```

(Os testes já existentes `tests/multiagent/test_s0_s1_s2_preparation.py` e `tests/multiagent/test_pilot_logic.py` não são alterados por esta entrega; rodar `python -m pytest tests/multiagent -v` executa os três arquivos juntos.)

## 4. Arquivos/relatórios a ler de volta

Em ordem de execução:

1. `experiments/multiagent/pilot12_manifest.csv` + `experiments/multiagent/pilot12_sampling_report.md` + `experiments/multiagent/pilot12_sampling_results.json` — confirma as 12 lesões, zero sobreposição por `lesion_id` E `image_id`, e os 4 designados para S0/S1 por grupo diagnóstico + validade de máscara (`designation_detail` no JSON: `reserved_by_group`, `random_draw_lesion_ids`, `random_draw_seed`, `restrictions_found`, `mask_validity_by_lesion_id`).
2. `experiments/multiagent/pilot12_s2_mask_generation_report.md` + `..._results.json` — confirma geração das máscaras S2 dos 12 (OK/vazia/falha), e que a amostra de 3.000 não foi lida.
3. `experiments/multiagent/pilot12_image_preparation_report` (CSV `pilot12_image_preparation_manifest.csv` + `..._results.json`) — confirma disponibilidade de S0/S1/S2 por caso.
4. Por modelo, em `experiments/multiagent/local_vlm_pilot/<model_tag>/`, APÓS O ESTÁGIO 1:
   - `call_log.jsonl` — log das chamadas de CONTROLE (envio de imagem + síntese).
   - `synthesis_manual_review_template.json` — **requer edição manual do pesquisador** antes do ESTÁGIO 2 (preencher `manual_review_verdict`/`reviewer_notes` por cenário).
   - `summary.json`/`report.md` — mostram em qual estágio parou (`stopped_at_stage`); se `aguardando_revisao_manual_sintese`, os controles passaram e falta só a revisão manual.
5. Por modelo, APÓS O ESTÁGIO 2 (mesmo diretório, mesmos `summary.json`/`report.md` agora sobrescritos com o resultado final):
   - `call_log_matriz_principal.jsonl` — log por chamada da matriz principal (92 chamadas).
   - `summary.json` — contagens planejadas/executadas/não-executadas por papel×condição, taxa de validade na 1ª tentativa, cobertura binária (com nota de proxy "negativo"="não_melanoma"), relatório informativo ABCD/Menzies/SPCL, candidatos a recusa, rastro dos contadores de abstenção, resultado dos controles + revisão manual, crescimento de swap (nova regra), projeção de duração em escala real.
   - `report.md` — versão legível do `summary.json`, com seção dedicada "candidatos a recusa para revisão manual" e a projeção de duração claramente separada dos critérios de pass/fail do piloto.

## 5. Decisões fechadas pelo pesquisador (substituem as interpretações abertas do primeiro rascunho)

Todas as 10 decisões abaixo foram fechadas explicitamente pelo pesquisador
sobre o primeiro rascunho (commit `b097037`) e estão implementadas em
`src/multiagent/vlm_direct.py`/`scripts/run_local_vlm_pilot.py`/
`scripts/run_pilot12_sample.py` sem re-derivação:

1. **Instrução de tarefa do VLM-direto** (`vd.VLM_DIRECT_TASK_INSTRUCTION`) — mesmas 3 categorias/mesmo significado do sumarizador, texto livre (NÃO JSON), sem pedir ABCD/Menzies/SPCL ao VLM-direto. `classify_case_status` reusada sem modificação para VLM-direto e sumarizador. **Registro de proxy**: o status `"negativo"` de `classify_case_status` vem da palavra-chave literal "lesão benigna" — nesta comparação A34 ele é usado como PROXY de `binary_target == "não_melanoma"` (convenção já existente do projeto, não introduzida por esta instrução). Sempre que a cobertura binária for discutida no relatório, "negativo" deve ser lido como esse proxy, nunca como diagnóstico literal de benignidade.
2. **Controles de envio de imagem — redesenhados, POR PAPEL VISUAL** (`vd.VISUAL_ROLES`, `vd.IMAGE_CONTROL_SPECS_BY_ROLE`, `vd.NO_IMAGE_CONTROL_PROMPT`): para CADA um dos 4 papéis (`vlm_direto`, `diagnostic_abcd`, `diagnostic_menzies`, `diagnostic_spcl`), 3 controles — 2 imagens sintéticas que diferem em cor E quadrante (`vd.build_image_control_specs_for_role`) + 1 controle "sem imagem" (mensagem só texto, nenhum `image_url` no payload, `vd._text_only_control_message`). Total = 4 x 3 = **12 chamadas de controle de imagem** (não mais 3 compartilhadas). Cada controle passa pelo caminho de construção de LLM PRÓPRIO do papel (`vd._build_role_llm`: `ChatOllama` direto para `vlm_direto`, `.llm` do agente real — `ABCDDiagnosticAgent`/etc. — para os 3 papéis diagnósticos), usando o prompt/imagem de CONTROLE, nunca o prompt clínico real (que fica intocado). O payload de cada controle de imagem é decodificado e comparado pixel-a-pixel com a imagem de origem (`vd.verify_image_control_payload_sent_correct_image`, reusa `images_are_effectively_equal` de `presentation.py`); o controle "sem imagem" é confirmado por ausência literal de entrada `image_url` (`vd.verify_no_image_control_payload_has_no_image`). Nenhum metadado/nome de arquivo revela a resposta esperada ao modelo. **Interpretação de design NOVA #11** (ainda sem confirmação): a fraseologia aceita como reconhecimento de "não há imagem" (`vd._NO_IMAGE_ACK_MARKERS`) — heurística de substring, pode não cobrir todas as formas de um VLM local dizer isso.
3. **Controles de síntese — mesmos 4 cenários, 2 mudanças de conteúdo**: o cenário "conflitante" agora tem `vasos_atipicos` do SPCL com fraseologia inequívoca de **"não avaliado"** (`"Não avaliado: este critério não foi avaliado nesta análise."`, nunca "Sim."/"Não." — ver `vd.FIXED_SYNTHESIS_SCENARIOS["conflitante"]`), para poder testar se o sumarizador preserva a distinção entre "não avaliado" e "avaliado e ausente" (`vd._check_not_evaluated_not_misrepresented_as_absent`, heurística lexical APROXIMADA, flagueada como tal). Para "um_parecer_ausente", adicionada checagem equivalente contra confundir "parecer ausente" com "lesão não tem a característica" (`vd._check_absence_not_misrepresented_as_lesion_lacks_feature`, igualmente aproximada).
4. **"Achado clínico novo" — deixou de ser pass/fail automático**: `vd._vocabulary_terms_not_in_inputs` agora é um sinal de TRIAGEM — `run_synthesis_control()["passed"]` reflete só se a saída é parseável pela A34, nunca se há "achado novo". Fluxo de controle redesenhado em DOIS ESTÁGIOS: ESTÁGIO 1 roda os 12 controles de imagem + 4 de síntese, escreve `synthesis_manual_review_template.json` (um item por cenário: `justificativa`, `novel_clinical_terms_candidates`, as 2 checagens aproximadas de "não avaliado", e `manual_review_verdict: null`/`reviewer_notes: ""` para o pesquisador preencher), e PARA — nunca prossegue à matriz principal na mesma invocação, mesmo com zero achados novos na heurística lexical. ESTÁGIO 2 (`--synthesis-manual-review-file <caminho>`) lê o template preenchido; só prossegue à matriz se TODOS os 4 cenários tiverem `manual_review_verdict == "sem_achado_novo"` (string exata; a outra opção válida é `"achado_novo_encontrado"`, que bloqueia) E os controles de envio de imagem já tiverem passado na invocação anterior. Qualquer veredito ausente/inválido/`"achado_novo_encontrado"` PARA a execução com mensagem clara (`vd.evaluate_manual_review_file`).
5. **Marcadores de recusa — classificação mais estrita** (`vd.classify_call_outcome`): um marcador de `vd._REFUSAL_MARKERS` só classifica a resposta como `STATUS_REFUSAL` se o texto NÃO contiver também um dos 3 rótulos categóricos válidos (`vd._contains_valid_parecer`, que importa o vocabulário de `src.evaluation.case_status._KEYWORD_TO_STATUS` — não uma segunda cópia hardcoded). Se marcador de recusa + parecer válido coexistirem, a resposta é `STATUS_OK` e classificada normalmente. Revisado e confirmado: "não consigo distinguir pela imagem" não estava e continua NÃO estando em `_REFUSAL_MARKERS` (é linguagem legítima de parecer indeterminado, roteada por `classify_case_status` para `"indeterminado"`). **Visibilidade de relatório**: toda chamada classificada `STATUS_REFUSAL` (não só a 3ª consecutiva que dispara pausa) é surfaceada com seu texto bruto na seção "candidatos a recusa para revisão manual" do `report.md` do ESTÁGIO 2 — como a arquitetura é de execução em lote único (sem checkpoint interativo ao vivo), esta seção é a interpretação prática de "revise os casos ambíguos antes de pausar": revisão pós-hoc no relatório, não uma pausa interativa real.
6. **Regra de swap — substituída por completo** (`vd.detect_sustained_swap_growth`): baseline explícito capturado pelo driver IMEDIATAMENTE após controles completos e ANTES da primeira chamada da matriz principal (parâmetro obrigatório `baseline_swap_bytes`, nunca derivado da própria lista de amostras). Amostras da MATRIZ PRINCIPAL (nunca dos controles) particionadas em janelas não-sobrepostas de 5; mediana por janela. `sustained_growth_detected=True` só se existirem 2 janelas CONSECUTIVAS com ambas as medianas `>= baseline + 512 MiB` (`vd.SWAP_GROWTH_THRESHOLD_BYTES = 512*1024*1024` — constante OPERACIONAL deste piloto, não um limite universal de máquina) E a segunda mediana estritamente maior que a primeira. Um padrão "pico então estável" (uma janela cruza o limiar, a próxima fica no/acima do limiar mas não supera a mediana anterior) é reportado separadamente em `spike_then_stable_detected`, nunca contado como crescimento sustentado. `definition_needs_researcher_signoff` agora é `False` (decisão já fechada), mas as janelas/medianas brutas continuam sempre reportadas.
7. **Designação dos "4 de 12" — estratificada + filtrada por máscara válida** (`scripts/run_pilot12_sample.py::designate_4_for_s0_s1`): 3 grupos diagnósticos (melanoma `dx=="mel"`, nevo `dx=="nv"`, não-melanocítica — demais), 1 lesão reservada de CADA grupo entre as com máscara de referência válida (reusa `load_binary_mask`/`load_original_image`/`shapes_match` de `presentation.py`, nunca reimplementado; tiebreak determinístico: primeiro `lesion_id` ascendente). A 4ª vaga é um sorteio aleatório real com sub-semente NOVA (`FOURTH_SLOT_SEED = 20270204`, documentada e distinta de `PILOT_SEED`) entre as lesões com máscara válida restantes. Se um grupo não tiver NENHUMA lesão com máscara válida: a restrição é registrada explicitamente (`restrictions_found`: grupo, candidatos, confirmação de zero válidas) e a vaga é redistribuída — o sorteio aleatório passa a sortear 1 lesão extra por grupo vazio, nunca uma substituição silenciosa. O resultado é persistido em `pilot12_sampling_results.json["designated_s0_s1_lesion_ids"]` e reusado identicamente pelos dois modelos.
8. **Semente do piloto de 12** (`PILOT_SEED = 20270203`, inalterada) — governa SÓ a seleção das 12 lesões, nunca o determinismo das respostas do VLM (inferência local via Ollama não é seedada por este valor). A nova sub-semente do sorteio da 4ª vaga (`FOURTH_SLOT_SEED = 20270204`, decisão #7) é documentada SEPARADAMENTE, nunca reusando `PILOT_SEED` silenciosamente para um segundo sorteio.
9. **"Zero parecer conclusivo" — mantido restrito a `vlm_direto`/`summary`** (aprovado como estava). NOVO (extensão): para `diagnostic_abcd`/`diagnostic_menzies`/`diagnostic_spcl`, o `report.md`/`summary.json` do ESTÁGIO 2 passam a reportar (informativo, NUNCA pass/fail) validade de resposta (via `call_status`), recusas explícitas, e uma contagem lexical de frases "não avaliado"/"não considerado" (`vd.count_not_evaluated_phrases`, `vd.summarize_diagnostic_role_reporting`) — esses 3 papéis nunca têm campo de diagnóstico categórico, então o flag "zero parecer conclusivo" nunca se aplica a eles (documentado explicitamente no próprio campo `note` do relatório, para não ser confundido com o flag real).
10. **Execução sequencial dos 4 papéis** (ABCD → Menzies → SPCL → sumarizador, mantida, conteúdo inalterado). NOVO (extensão): o driver soma `elapsed_s` das 4 chamadas sequenciais por lesão+condição (`vd.compute_per_lesion_condition_elapsed_totals`) e projeta (NUNCA promete) uma duração aproximada para uma execução hipotética de 3.000 lesões (`vd.project_full_scale_duration`, fórmula = média dos totais por lesão+condição DO PILOTO × N hipotético), com caveat explícito de que a projeção extrapola só dos tempos por chamada do piloto (modelo já carregado, 12 lesões, máquina do pesquisador) — não é promessa de tempo real de uma execução de 3.000, que pode diferir por hardware/estado de carregamento/contenção de recursos/variação do próprio modelo local.

### Novo parâmetro fechado: 108 chamadas / teto de 4h

4 horas por modelo é um TETO OPERACIONAL para a execução COMPLETA de um
modelo (ESTÁGIO 1 + ESTÁGIO 2 combinados), não uma meta de desempenho.
Total de chamadas planejadas por modelo = **108** = 92 (matriz, `assert`
inalterado) + 16 de controle (12 de envio de imagem + 4 de síntese).
Passar do teto de 4h NÃO estabelece por si só viabilidade do modelo para
os 3.000 casos reais — isso requer a projeção de duração do item 10
acima, a partir dos tempos medidos por condição do piloto com o modelo já
carregado.

### Interpretações de design NOVAS introduzidas neste rework (distintas das 10 já fechadas)

11. Fraseologia aceita como reconhecimento de "não há imagem" no controle sem imagem (`vd._NO_IMAGE_ACK_MARKERS`) — ver decisão #2.
12. As 2 combinações cor+quadrante dos controles de imagem são reusadas IDÊNTICAS para os 4 papéis visuais (só o nome é prefixado pelo papel) — não 8 combinações distintas; escolha de conveniência deste rework.
13. O texto exato da sub-semente do 4º sorteio de `run_pilot12_sample.py` (`FOURTH_SLOT_SEED = 20270204`) e o mecanismo exato de redistribuição de vaga quando um grupo diagnóstico não tem máscara válida (sortear 1 lesão extra por grupo vazio) — ver decisão #7.

## 6. Por que foi necessário um driver manual (não dá para reusar `build_graph()` compilado caso a caso)

Confirmado por leitura de `src/agents/agents.py::build_graph`: o grafo compilado tem uma ÚNICA aresta de entrada, `START -> "segmentation"`, e o nó `"segmentation"` sempre recalcula `seg_image_data` a partir de `state.image_path`/`state.seg_path` (modo "model" ou "precomputed" do próprio `agents.py`) — não há um jeito de invocar o grafo compilado "pulando" esse nó para injetar um `seg_image_data` já pronto de uma condição S0/S1/S2 pré-computada por `src/multiagent/presentation.py`. Portanto `src/multiagent/vlm_direct.py` NÃO modifica `agents.py`/`build_graph()`; instancia diretamente `ABCDDiagnosticAgent`/`MenziesDiagnosticAgent`/`SPCLDiagnosticAgent`/`SummaryAgent` (as mesmas classes, os mesmos prompts internos, a mesma função `_invoke`/`_image_message`) com um `GraphState` real já populado com o `seg_image_data` da condição sob teste — ver `vd.run_multiagent_role_call`/`vd.build_graph_state`. Os controles de envio de imagem (decisão #2) reusam o mesmo princípio: `vd._build_role_llm` instancia a classe de agente real e usa o `.llm` já construído no `__init__` (mesmo `_build_llm` interno de `agents.py`), sem chamar `.analyze_lesion` (que monta sempre o prompt clínico real).

## 7. Lógica de seleção de modelo (Passo 7)

`scripts/run_local_vlm_pilot.py` roda UM modelo por par de invocações (ESTÁGIO 1 + ESTÁGIO 2), `--model-tag`. Depois de rodar os dois estágios para `qwen3-vl:8b-instruct-q4_K_M` e `medgemma1.5:4b-it-q4_K_M` (cada um grava seu próprio `summary.json`/`report.md` final em `experiments/multiagent/local_vlm_pilot/<tag>/`):

- Se **ambos** completaram o ESTÁGIO 2 (`summary.json["stopped_before_main_matrix"] == False` nos dois): nenhum é "primário", o modelo de contingência (`qwen2.5vl:7b-q4_K_M`) NÃO é necessário.
- Se **só um** completou o ESTÁGIO 2: rode o mesmo fluxo de 2 estágios com `--model-tag qwen2.5vl:7b-q4_K_M`.
- Se **nenhum** completou (algum parou em `controle_de_envio_de_imagem`/`controle_de_sintese`, ou a revisão manual encontrou achado novo/veredito ausente): decidir os próximos passos é do pesquisador, não automatizado aqui.

Este branching é decidido pelo pesquisador entre execuções, lendo os `summary.json`/`report.md` reais — o assistente que escreveu este pacote não pode executar nada para descobrir ao vivo qual modelo passa.

## 8. Decisões já fechadas, implementadas sem re-derivação (herdadas da primeira entrega, inalteradas)

- Timeout de 120s por chamada, sem retry (`vd.TIMEOUT_S`, `vd.call_with_timeout`) — nota de implementação: a thread da chamada que excedeu o timeout continua rodando em segundo plano (limitação do `ThreadPoolExecutor`, documentada em `call_with_timeout`); o resultado atrasado nunca é usado.
- 3 contadores de abstenção por papel (falha técnica / sem resposta / recusa explícita), reiniciados em qualquer sucesso incluindo diagnóstico indeterminado válido (`vd.AbstentionCounters`/`vd.AbstentionTracker`).
- Casos inalcançáveis por pausa marcados `"nao_executado"` (`vd.STATUS_NOT_EXECUTED`, `vd.not_executed_record`) — nunca omitidos.
- Matriz principal com exatamente 92 chamadas planejadas, verificado por `assert` em `vd.build_planned_calls` (não só no docstring) — inalterado; só a contagem "total da execução do modelo" (controles + matriz) passou de 92 para 108.
- A34 (`src.evaluation.case_status.classify_case_status`) reusada idêntica para VLM-direto e multiagente-sumarizador, nunca uma classificação paralela.
- Texto bruto de toda resposta preservado verbatim nos logs (`call_log.jsonl` dos controles, `call_log_matriz_principal.jsonl` da matriz), mesmo quando classificado como `nao_reconhecido`/`ausente`.

## 9. Parser v2 "conclusão final" (pós-execução real do Estágio 1, 2026-11-13)

STATUS: implementado, aguardando execução do pesquisador. Nada deste
pacote foi executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) — nenhum `python`/`pytest`/`git`
foi chamado.

### 9.1 O que aconteceu (dado real, não hipotético)

O pesquisador executou o Estágio 1 do piloto para
`qwen3-vl:8b-instruct-q4_K_M`. Controles de envio de imagem: 4/4 papéis
passaram. Controles de síntese: 3/4 cenários parsearam corretamente via
`classify_case_status` (A34). O 4º cenário (`um_parecer_ausente`) NÃO
parseou — `classified_status: "nao_reconhecido"` — porque o texto bruto
real do modelo (preservado em
`experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/call_log.jsonl`
e `report.md`) nunca escreveu a frase literal "lesão benigna":
  - no meio do texto: `"Diagnóstico Consolidado: Lesão Cutânea Benigna — Pode ser caracterizada como um nevo..."` (a palavra "Cutânea" quebra o substring, e isto nem é o veredito final);
  - no veredito final (depois de um separador `---`, prefixado por `✅`): `"✅ **Diagnóstico Consolidado: BENIGNO (Nevus com características típicas)**"` — usa "BENIGNO" isolado, sem a palavra "lesão".

Isto confirma o gap já registrado em
`docs/reconstruction/15_multiagent_experimental_matrix.md §6.1`: modelos
locais via Ollama (sem `with_structured_output`) parafraseiam e não
batem sempre a frase literal exigida pela A34.

### 9.2 Decisão do pesquisador (implementada sem reinterpretação)

NÃO modificar `src/evaluation/case_status.py::classify_case_status` (usado
em avaliações históricas/stage6 — "mantenha as avaliações históricas
intactas"). Também não alterados: os prompts reais
(`VLM_DIRECT_TASK_INSTRUCTION` em `src/multiagent/vlm_direct.py`,
`src/agents/agents.py`, `src/agents/schemas.py`,
`src/multiagent/presentation.py`). Em vez disso, foi construído um
classificador NOVO, separadamente versionado, aplicado ADITIVAMENTE.

### 9.3 Arquivos novos/alterados nesta entrega

Novos:
- `src/evaluation/case_status_v2.py` — classificador v2
  (`classify_case_status_v2`/`classify_case_status_v2_detailed`,
  `PARSER_VERSION = "v2_conclusion_line_2026_11_13"`). Reusa o MESMO
  vocabulário de status de `case_status` (import direto das constantes
  `POSITIVO`/`NEGATIVO`/`INDETERMINADO`/`AUSENTE`/`NAO_RECONHECIDO`,
  nunca redefinidas).
- `scripts/run_reclassify_pilot_with_v2_parser.py` — reclassificação
  OFFLINE de um `--output-dir` já executado (lê `call_log.jsonl`
  e/ou `call_log_matriz_principal.jsonl`, nunca faz nova chamada ao
  modelo, nunca sobrescreve os arquivos originais).
- `tests/evaluation/test_case_status_v2.py` — cobre o texto bruto REAL
  (literal, copiado de `call_log.jsonl`) dos cenários `um_parecer_ausente`
  e `conflitante`, negação sintética, múltiplas categorias no corpo com
  um veredito final claro, e confirmação de que indeterminado/recusa-
  adjacente/ausente nunca caem em positivo/negativo.

Alterados (aditivamente, sem tocar prompts/vocabulário/gate de
pausa-abstenção):
- `src/multiagent/vlm_direct.py` — `CallRecord` ganhou o campo
  `classified_status_v2` (default `None`), calculado em paralelo a
  `classified_status` (v1, intocado) em `run_vlm_direct_call`,
  `run_multiagent_role_call` e `run_synthesis_control`. Confirmado por
  leitura de código antes de editar: a lógica de pausa/abstenção
  (`AbstentionTracker.register`, em `scripts/run_local_vlm_pilot.py`) lê
  só `record.call_status` (saída de `classify_call_outcome` — recusa/
  falha técnica/sem resposta), NUNCA `classified_status` — portanto não
  foi tocada e não depende do novo campo.
- `scripts/run_local_vlm_pilot.py` — importa `case_status_v2`; o log por
  chamada no console agora mostra `classificado_v1=`/`classificado_v2=`
  lado a lado; `summary.json` ganhou um bloco aditivo/informativo
  `binary_coverage_v2_informational` (mesma contagem de
  `binary_coverage`, mas a partir de `classified_status_v2`) — o bloco
  `binary_coverage` original (v1) não foi alterado e continua sendo o
  campo histórico/real do relatório.

Não alterados: `src/evaluation/case_status.py`, `src/agents/agents.py`,
`src/agents/schemas.py`, `src/multiagent/presentation.py`,
`VLM_DIRECT_TASK_INSTRUCTION` e os demais prompts reais em
`vlm_direct.py`.

### 9.4 Heurística do v2 (APROXIMADA — ver docstring completa em `case_status_v2.py`)

1. **Extração da "conclusão final"** (`_extract_final_conclusion_region`):
   separa o texto em parágrafos (por linha em branco); procura marcadores
   (`"conclusão final"`, `"diagnóstico consolidado"`, `"avaliação final"`,
   `"conclusão integrada"`, `"veredito final"`, `"resumo final"`) só na
   METADE FINAL do documento (por índice de parágrafo) — isto evita
   ancorar num título/cabeçalho que repete a mesma palavra no início
   (ex.: o cenário real `conflitante` começa com "Diagnóstico
   Consolidado: ..." como TÍTULO, não como veredito). A região de
   conclusão final vai do primeiro marcador qualificado até o fim do
   documento. Sem marcador: cai para a última linha que começa com `✅`;
   sem isso, cai para o último parágrafo não vazio.
2. **Mapeamento de variantes** (`LABEL_VARIANTS_V2`, dict/tupla
   inspecionável): positivo/indeterminado usam EXATAMENTE as mesmas
   frases literais de `case_status._KEYWORD_TO_STATUS` (nenhuma variante
   nova — não foi pedido e é mais arriscado generalizar sem dados).
   Negativo ganha, ALÉM de `"lesão benigna"`/`"lesao benigna"`: `"lesão
   cutânea benigna"`/`"lesao cutanea benigna"` e as palavras isoladas
   `"benigno"`/`"benigna"` (match de PALAVRA EXATA — não casa o plural
   `"benignos"`/`"benignas"`, decisão deliberada: no cenário real
   `conflitante`, "resultados benignos dos algoritmos ABCD e SPCL" é um
   adjetivo sobre OUTROS algoritmos, não um veredito).
3. **Negação** (`_is_negated`): procura "não"/"nao" nas últimas 4
   palavras IMEDIATAMENTE antes do rótulo casado, restrito à cláusula
   atual (nunca atravessa `. ! ? ; , : \n`) — evita que uma negação de
   uma cláusula anterior "vaze" para a próxima (ex.: "não é benigna, é
   suspeita de melanoma" não nega "suspeita de melanoma"). HEURÍSTICA
   APROXIMADA, não NLP real — não entende negação distribuída, dupla
   negação, "nunca"/"jamais", etc.
4. **Resolução de múltiplos candidatos**: dentro da região de conclusão
   final, entre todos os rótulos casados NÃO negados, vence o mais
   próximo do FIM da região — aproxima "o último veredito declarado
   vence" sem reimplementar um parser de discurso real.

**LIMITAÇÃO CONHECIDA, AINDA SEM CONFIRMAÇÃO EXPLÍCITA DO PESQUISADOR**:
esta é uma heurística posicional/lexical sobre os estilos de resposta
observados neste piloto (qwen3-vl:8b-instruct-q4_K_M, 4 cenários fixos).
Pode não cobrir todos os estilos possíveis de um VLM local (ex.: um
modelo que declare o veredito real ANTES de uma longa análise posterior,
sem repeti-lo ao final, quebraria a heurística de "metade final do
documento").

### 9.5 Confirmação real (cenário `um_parecer_ausente`, mesmo texto bruto do piloto)

- v1 (`classify_case_status`, intocado): `"nao_reconhecido"` — inalterado.
- v2 (`classify_case_status_v2`): `"negativo"` — variante casada:
  `"benigno"`, na região de conclusão final extraída (último parágrafo:
  `"✅ Diagnóstico Consolidado: BENIGNO (Nevus com características típicas)"`),
  sem negação.

### 9.6 Comando de reclassificação (sem repetir chamadas ao modelo)

```
python scripts/run_reclassify_pilot_with_v2_parser.py \
    --output-dir experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M
```

Lê `call_log.jsonl` (e `call_log_matriz_principal.jsonl`, se existir)
já gravados por uma execução anterior de `run_local_vlm_pilot.py`.
NUNCA sobrescreve/muta `call_log.jsonl`, `call_log_matriz_principal.jsonl`,
`summary.json` ou `report.md`. Escreve, no MESMO `--output-dir`:
- `reclassification_v2_results.json`
- `reclassification_v2_report.md` (inclui uma seção dedicada confirmando
  o resultado do item 9.5 acima, e, por registro, texto bruto + v1
  histórico + v1 recomputado + v2 + concordância).

### 9.7 Comando de testes

```
python -m pytest tests/evaluation/test_case_status_v2.py -v
```

(`tests/evaluation/test_case_status.py` — já existente, testa só o v1 —
não é alterado por esta entrega.)

### 9.8 Interpretações de design novas introduzidas nesta entrega (sem confirmação explícita)

14. O conjunto exato de marcadores de "conclusão final"
    (`_CONCLUSION_MARKERS`) e a regra "só na metade final do documento,
    por índice de parágrafo" — cobre os 2 cenários reais deste piloto
    (`um_parecer_ausente`, `conflitante`), mas não foi confirmado contra
    outros estilos de resposta/modelos ainda não testados.
15. A exclusão deliberada do plural (`"benignos"`/`"benignas"`) do
    mapeamento de variantes — resolve o caso real observado no cenário
    `conflitante`, mas é uma escolha de design, não uma regra
    linguística geral.
16. A janela de negação de 4 tokens restrita à cláusula atual (até a
    pontuação mais próxima) — heurística aproximada, não confirmada
    contra todos os padrões de negação possíveis em português.
