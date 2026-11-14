# 22 — Extração da resposta final antes do v2 (preprocessamento, Etapa 5)

Resumo para o planejador central: implementação da extração de resposta
final pedida verbatim pelo pesquisador em resposta ao achado técnico
registrado em `docs/reconstruction/21_synthesis_control_fidelity_review_closure.md §5`
(vazamento de template `<unused94>thought ... <unused95>` nas 4 respostas
brutas reais de `medgemma1.5:4b-it-q4_K_M`, e a suspeita de que
`medgemma1.5/conflitante` foi classificado `negativo` pelo v2 por um
artefato da heurística de "região de conclusão final", não pelo
conteúdo real da resposta).

**Status: implementada, aguardando execução do pesquisador.** Nenhum
script foi executado por este agente — nem `python`, nem `pytest`, nem
git. Tudo abaixo que descreve "o que o script vai produzir" é uma
**previsão manual**, feita por rastreamento linha-a-linha da lógica real
de `case_status_v2.py` contra os 8 textos brutos reais, não uma execução.

## 1. O que foi construído

### 1.1 Módulo de extração (novo, reusável, importável)

`src/evaluation/response_template_extraction.py` —
`extract_final_answer(raw_text) -> ExtractionResult`
(`final_answer_text`, `template_removed`, `status`,
`delimiter_pair_used`).

- Vocabulário fechado e SÓ-OBSERVADO de delimitadores:
  `("<unused94>thought", "<unused95>")` — citação literal confirmada nas
  4 respostas brutas reais de `medgemma1.5:4b-it-q4_K_M` em
  `experiments/multiagent/synthesis_controls_real_image/results.json`.
  Exemplo literal (início de `unanime_positivo`): `"<unused94>thought\nMy
  thinking process to arrive at the consolidated diagnosis goes like
  this:..."`; exemplo literal da transição real (mesmo caso):
  `"...as per instructions).<unused95>Análise Integrada dos
  Pareceres:\n\n..."`.
- Flag explícita no docstring do módulo: esta lista é fechada e cobre
  **só** o que foi observado neste piloto. Outros modelos/quantizações/
  versões de template podem usar marcadores diferentes (ex.:
  `<think>`/`</think>`), não cobertos — cairiam em
  `status="no_template_detected"`, tratado como "sem template", não
  como erro.
- Quatro status possíveis: `extracted_ok`, `no_template_detected`,
  `incomplete_block_no_closing_delimiter`, `empty_after_extraction`. Os
  dois últimos são tratados como **problema registrado** — o módulo
  nunca devolve texto parcial para classificação nesses casos
  (`final_answer_text=None`), por instrução explícita do pesquisador.
- Não importa nada de `case_status_v2.py`/`case_status.py` — é uma
  etapa estruturalmente anterior e separada da classificação.

### 1.2 Script de reprocessamento offline (novo, zero chamadas de modelo)

`scripts/run_reprocess_synthesis_controls_final_answer_extraction.py` —
lê `experiments/multiagent/synthesis_controls_real_image/results.json`
(as 8 chamadas já executadas), para cada caso:

1. Aplica `extract_final_answer` ao `raw_response_text` salvo.
2. Classifica com `classify_case_status_v2_detailed`
   (`src.evaluation.case_status_v2`, **não modificado**) duas vezes:
   ANTES (texto bruto original — verificado contra `classified_status_v2`
   já salvo em `results.json`, não assumido) e DEPOIS (resposta final
   extraída — só quando a extração não reportou problema).
3. Grava dois arquivos **novos** (nunca sobrescreve `results.json`/
   `report.md`, que são o registro histórico já revisado por
   fidelidade):
   - `experiments/multiagent/synthesis_controls_real_image/final_answer_extraction_reprocessing_results.json`
   - `experiments/multiagent/synthesis_controls_real_image/final_answer_extraction_reprocessing_report.md`

Para `medgemma1.5/conflitante`, o script nunca compara contra um rótulo
esperado (`expected_category` é `None` por desenho) — só mostra lado a
lado a região de conclusão final extraída e o rótulo casado em cada
passada, para leitura humana.

**Comando exato para o pesquisador rodar:**

```
python scripts/run_reprocess_synthesis_controls_final_answer_extraction.py
```

**Comando exato de teste:**

```
python -m pytest tests/evaluation/test_response_template_extraction.py -v
```

### 1.3 Testes

`tests/evaluation/test_response_template_extraction.py`:
- excerto literal real de `medgemma1.5/unanime_positivo` → `extracted_ok`,
  resposta final começa logo após `<unused95>`, nenhum fragmento do
  bloco de pensamento presente;
- texto bruto real completo de `qwen3-vl/um_parecer_ausente` (sem
  nenhum marcador) → `no_template_detected`, texto inalterado;
- caso sintético de bloco incompleto (abertura sem fechamento) →
  `incomplete_block_no_closing_delimiter`, `final_answer_text=None`;
- caso sintético de texto vazio após extração → `empty_after_extraction`,
  `final_answer_text=None`;
- texto vazio/`None` → `no_template_detected` (passagem segura).

`tests/evaluation/test_case_status_v2.py` **não foi tocado**.

## 2. O que NÃO foi feito nesta tarefa (deferido, por instrução explícita)

O pesquisador disse: *"Primeiro eu rodaria o reprocessamento local e
revisaria o resultado"* — passo 3 do pedido (ligar a extração a
`run_vlm_direct_call`/`run_multiagent_role_call`, os caminhos de chamada
AO VIVO em `src/multiagent/vlm_direct.py`) está **explicitamente
deferido**. Confirmações de leitura (nenhuma modificação feita):

- `src/multiagent/vlm_direct.py` — `run_vlm_direct_call` (linhas
  495–532), `run_multiagent_role_call` (linhas 562–634), `CallRecord`
  (linhas 438–460), `run_synthesis_control_real_image` — **não
  modificados**. Lidos para confirmar exatamente onde `classify_case_status_v2`
  é chamado hoje (`classified_v2 = classify_case_status_v2({"validation":
  raw_text}) if status == STATUS_OK else None`, idêntico nos dois
  pontos de chamada), para que a futura troca por extração + classificação
  seja um diff mínimo e isolado quando/se aprovada.
- `src/evaluation/case_status_v2.py` — **não modificado**. Reusado
  apenas via `classify_case_status_v2_detailed`, importado sem alteração.
- `src/agents/agents.py` — **não lido/modificado** (fora do escopo; os
  prompts reais não são tocados por este preprocessamento).

O módulo novo foi desenhado para ser **importável e reusável** sem
qualquer acoplamento aos caminhos ao vivo, exatamente para que a ligação
futura (passo 3) seja trivial depois da aprovação do pesquisador.

## 3. Previsão manual do resultado (NÃO EXECUTADO) — os 8 casos

Rastreamento feito à mão, texto real por texto real, contra a lógica de
`_extract_final_conclusion_region`/`_find_last_valid_label_match`/
`_find_all_label_matches` em `case_status_v2.py`. **Confiança alta** para
os 4 casos do qwen3-vl (texto idêntico ANTES/DEPOIS, trivial). **Confiança
alta também** para os 4 casos do medgemma1.5 na trilha DEPOIS (textos
curtos, poucos parágrafos, enumerados à mão abaixo) — **confiança média**
na trilha ANTES para o medgemma (textos muito longos com o bloco de
pensamento completo; a lógica foi aplicada mentalmente mas não
executada, e o próprio script existe precisamente para confirmar/
refutar isto com certeza).

### 3.1 qwen3-vl:8b-instruct-q4_K_M (as 4 respostas confirmadas, por leitura
integral, SEM nenhum marcador `<unused94>`/`<unused95>`)

Extração: `no_template_detected` nos 4 casos → `final_answer_text` ==
`raw_response_text` byte-a-byte → ANTES e DEPOIS são o **mesmo texto**
alimentado duas vezes ao mesmo classificador → previsão: **ANTES ==
DEPOIS trivialmente, para os 4**, idênticos aos valores já salvos em
`classified_status_v2`:

| Cenário | `classified_status_v2` salvo | Previsão ANTES | Previsão DEPOIS |
|---|---|---|---|
| unanime_positivo | `nao_reconhecido` | `nao_reconhecido` | `nao_reconhecido` |
| unanime_negativo | `negativo` | `negativo` | `negativo` |
| conflitante | `positivo` | `positivo` | `positivo` |
| um_parecer_ausente | `negativo` | `negativo` | `negativo` |

Nenhuma mudança esperada para este modelo — o preprocessamento é um
no-op para ele por construção (nunca houve template para remover).

### 3.2 medgemma1.5:4b-it-q4_K_M

#### `unanime_positivo`

Resposta final (após `<unused95>`) tem 4 parágrafos (separados por
`\n\n`): título, frase de abertura, lista 1/2/3 (ABCD/Menzies/SPCL) como
um único parágrafo, e o parágrafo final `"**Conclusão:** ... indicam um
**diagnóstico de alta suspeita por melanoma**."`. Com `n=4`,
`final_half_start_index = 4 - (4+1)//2 = 2` → metade final = os 2 últimos
parágrafos. Nenhum deles contém um marcador de `_CONCLUSION_MARKERS`
("conclusão:" sozinho não é "conclusão final"/"diagnóstico consolidado")
→ cai no fallback `last_paragraph` → região = só o último parágrafo.
Nesse parágrafo: a frase literal `"suspeita de melanoma"` NÃO aparece
(o texto usa **"suspeita POR melanoma"**); o padrão regex
`_AFFIRMATIVE_SUSPEITA_MELANOMA_PATTERN` exige literalmente `"...de\s+
melanoma"` após "suspeita" + qualificadores — `"por"` não é `"de"`, não
bate. Nenhuma variante NEGATIVO/INDETERMINADO presente. **Previsão
DEPOIS: `nao_reconhecido`** — igual ao `classified_status_v2` já salvo
(`nao_reconhecido`). Consistente com o achado já registrado em
`results.json` (`faithful_but_unrecognized_by_v2: true`): é
genuinamente fiel e positivo, só fora do vocabulário fechado (mesmo
problema de "suspeita por" vs. "suspeita de" já conhecido, **não**
causado pelo template leakage — a extração confirma isso, já que o
mesmo resultado se mantém depois de remover o bloco de pensamento).

#### `unanime_negativo`

Resposta final tem 9 parágrafos. `final_half_start_index =
9-(9+1)//2 = 4` → metade final = parágrafos 4–8. Nenhum marcador de
`_CONCLUSION_MARKERS` bate em nenhum deles ("conclusão geral"/"conclusão:"
sozinhos não contam) → fallback `last_paragraph` → região = último
parágrafo: `"Com base na análise conjunta dos três pareceres e da
imagem, a lesão cutânea é classificada como **benigna**, sem suspeita
de malignidade."`. Match válido: `"benigna"` (palavra isolada, `\b`,
sem negação na cláusula — o `"não"` mais próximo não existe nesta
frase). `"sem suspeita de malignidade"` não dispara POSITIVO (a palavra
é "malignidade", não "melanoma"). **Previsão DEPOIS: `negativo`** —
igual ao salvo. Bem suportado pelo conteúdo: a frase final afirma
benignidade de forma direta e inequívoca.

#### `conflitante` — caso central desta tarefa

Resposta final tem 7 parágrafos: (0) abertura, (1) `"**Análise dos
Pareceres:**"`, (2) lista 1/2/3 ABCD/Menzies/SPCL, (3)
`"**Integração e Diagnóstico Consolidado:**"`, (4) parágrafo de
divergência, (5) parágrafo mais longo repetindo a divergência, (6)
`"**Conclusão:** A lesão apresenta características que são
discordantemente interpretadas pelos diferentes algoritmos, com
resultados que variam de **benigno (ABCD)** a melanoma (Menzies) e
borderline (SPCL). A presença de múltiplas características de alto
risco (Menzies) e um achado borderline (SPCL) é um ponto crítico."`

`final_half_start_index = 7-(7+1)//2 = 3` → metade final = parágrafos
3–6. O parágrafo 3, `"**Integração e Diagnóstico Consolidado:**"`,
**contém literalmente "diagnóstico consolidado"** → bate em
`_CONCLUSION_MARKERS` → `marker_in_final_half`, região = parágrafos
3–6 inteiros (não só o último).

Dentro dessa região, `_find_last_valid_label_match` varre TODAS as
ocorrências válidas (não negadas) de variantes e devolve a de maior
índice (mais próxima do fim). Candidatas NEGATIVO (`"benigno"`/
`"benigna"`, palavra isolada) encontradas, em ordem de posição:
1. `"...o critério ABCD interpreta a lesão como **benigna**, enquanto..."` (parágrafo 4)
2. `"...juntamente com a interpretação **benigna** (ABCD)..."` (parágrafo 5)
3. `"...com resultados que variam de **benigno** (ABCD) a melanoma (Menzies)..."` (parágrafo 6, a própria frase de conclusão)

Nenhuma é negada (nenhum "não"/"nao" nos 4 tokens anteriores, dentro da
mesma cláusula, para nenhuma das três). Nenhuma ocorrência de
`"suspeita de melanoma"` (literal ou via padrão) existe em lugar nenhum
da região — as únicas aparições de "suspeita" estão nas palavras
"suspeitas"/"suspeitos" (plural), que não casam com o padrão (exige
`\s` logo depois de "suspeita", mas o próximo caractere é a letra "s"
do plural). Portanto o match de MAIOR posição é a 3ª ocorrência —
**dentro da própria frase de conclusão**, mas a frase inteira é:
*"resultados que variam de **benigno** (ABCD) a melanoma (Menzies) e
borderline (SPCL)"* — isto **enumera os três resultados divergentes**,
não afirma um veredito final de benignidade. O parágrafo de conclusão
explicitamente preserva o conflito ("são discordantemente
interpretadas") e NÃO resolve para um lado.

**Previsão DEPOIS: `negativo`** — mesmo valor já salvo em
`classified_status_v2` (`negativo`). **Isto confirma, por rastreamento
manual da resposta final já limpa do bloco de pensamento, a suspeita
registrada no fechamento anterior: o v2 continua devolvendo `negativo`
mesmo sobre a resposta final genuína, sem o bloco de pensamento — então
o problema NÃO se limita ao vazamento de template.** O mecanismo real é
a regra "match de maior posição vence" de `_find_last_valid_label_match`
combinada com uma frase de conclusão que lista literalmente as 3
categorias (benigno/melanoma/borderline) sem resolver — a heurística
posicional pega a palavra de vocabulário fechado que aparece mais à
direita no texto, não a que representa o veredito efetivo (que, neste
caso, é "nenhum veredito único — conflito preservado", exatamente o
comportamento que o cenário foi desenhado para testar, não um status
binário). Portanto: **o `negativo` devolvido pelo v2 para este caso NÃO
tem respaldo literal na resposta final como afirmação de benignidade**
— é um artefato de uma frase que descreve o espectro de divergência, não
uma conclusão. Isto é reportado aqui como achado, não corrigido (v2
está congelado nesta tarefa).

Nota de confiança: esta é a trilha DEPOIS, sobre um texto curto (7
parágrafos) que enumerei e verifiquei match-a-match à mão — confiança
alta. A trilha ANTES (sobre o texto bruto completo, com o bloco de
pensamento de ~900 palavras antes da resposta final) seria mais sujeita
a erro de contagem de parágrafos manual; o valor já SALVO em
`results.json` (`negativo`) é usado diretamente como a previsão ANTES,
e o script vai *confirmar* (não *assumir*) essa consistência via
`classify_case_status_v2_detailed` chamado de fato sobre o texto
completo.

#### `um_parecer_ausente`

Resposta final tem múltiplos parágrafos; o último é: *"Portanto, a
análise combinada aponta fortemente para uma lesão benigna."* — contém
a frase literal `"lesão benigna"` (variante NEGATIVO explícita, par
idade com v1), não negada. Mesmo que a região de conclusão capturada
inclua parágrafos anteriores (ex. `"**Conclusão Consolidada:**"` já
bate em nenhum marcador fechado tipo "diagnóstico consolidado" — na
verdade o cabeçalho real é `"**Conclusão Consolidada:**"`, que **não**
é um dos literais de `_CONCLUSION_MARKERS` como está escrito — então
provavelmente cai no fallback `last_paragraph` de novo), o match de
maior posição na região (seja ela o parágrafo inteiro final ou mais)
ainda será `"lesão benigna"` no último parágrafo, por ser a ocorrência
mais ao final do texto. **Previsão DEPOIS: `negativo`** — igual ao
salvo, e aqui **bem suportado pelo conteúdo**: a frase final afirma
diretamente benignidade, sem ambiguidade de enumerar múltiplas
categorias (ao contrário do caso `conflitante` acima).

### 3.3 Tabela-resumo da previsão (não executada)

| Modelo | Cenário | Extração | ANTES (previsto) | DEPOIS (previsto) | Mudou? | Suporte real no texto final? |
|---|---|---|---|---|---|---|
| qwen3-vl | unanime_positivo | no_template_detected | nao_reconhecido | nao_reconhecido | não | sim (vocabulário, não fidelidade) |
| qwen3-vl | unanime_negativo | no_template_detected | negativo | negativo | não | sim |
| qwen3-vl | conflitante | no_template_detected | positivo | positivo | não | sim (atribuído a Menzies, não consenso) |
| qwen3-vl | um_parecer_ausente | no_template_detected | negativo | negativo | não | sim |
| medgemma1.5 | unanime_positivo | extracted_ok | nao_reconhecido | nao_reconhecido | não | sim (vocabulário, "suspeita por" vs "de") |
| medgemma1.5 | unanime_negativo | extracted_ok | negativo | negativo | não | sim |
| medgemma1.5 | **conflitante** | extracted_ok | negativo | negativo | não | **NÃO — artefato posicional, ver §3.2** |
| medgemma1.5 | um_parecer_ausente | extracted_ok | negativo | negativo | não | sim |

**Conclusão da previsão**: em nenhum dos 8 casos a extração do bloco de
template muda o rótulo devolvido pelo v2 (nenhuma "mudou? sim" na
tabela). O vazamento de template não estava, pela minha análise manual,
causando os `nao_reconhecido`/`negativo` observados — a extração só
torna a entrada do classificador mais limpa e auditável, e **confirma**
(não corrige) o achado já suspeitado: `medgemma1.5/conflitante`
continua `negativo` sem respaldo semântico real, por um motivo
estrutural do v2 (regra de posição), independente do template.

## 4. Interpretações de design abertas (registradas, não resolvidas aqui)

- `template_removed` nos status de problema: decidi `False` para
  `incomplete_block_no_closing_delimiter` (nenhuma remoção de fato
  concluída) e `True` para `empty_after_extraction` (a remoção
  ocorreu, só não restou texto). Não foi pedido explicitamente qual
  valor usar nesses dois casos — documentado no docstring do módulo
  como decisão desta tarefa, não verbatim do pesquisador.
- O achado de §3.2 (`medgemma1.5/conflitante`) é reportado como
  evidência, não como correção — `case_status_v2.py` permanece
  congelado. Se o pesquisador confirmar a previsão rodando o script,
  a decisão de ajustar (ou não) a heurística de "match de maior
  posição" para esse tipo de frase enumerativa fica para uma tarefa
  futura, fora deste escopo.
- O passo 3 do pedido original (ligar a extração a `run_vlm_direct_call`/
  `run_multiagent_role_call`) está pronto para ser implementado (o
  módulo é importável e autocontido), mas **não foi feito** — aguarda
  revisão do resultado real do reprocessamento pelo pesquisador, como
  pedido explicitamente.
