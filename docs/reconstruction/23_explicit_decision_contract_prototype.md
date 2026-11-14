# 23 — Protótipo de contrato de decisão final explícito (Etapa 5, sem inferência)

Resumo para o planejador central: resposta ao achado confirmado em
`docs/reconstruction/22_final_answer_extraction_preprocessing.md §3`
(o classificador v2, `src.evaluation.case_status_v2.classify_case_status_v2`,
pode capturar "benigno" dentro de uma frase que ENUMERA pareceres
conflitantes como veredito negativo, mesmo quando o sumarizador não
tomou essa decisão — achado independente do vazamento de template já
corrigido pela extração de resposta final do doc 22).

**Status: entrega preparada, SEM inferência real.** Nenhum script foi
executado por este agente (nem `python`, nem `pytest`, nem git) —
aguardando o pesquisador rodar a verificação e aprovar antes de
qualquer passo seguinte.

## 1. O que foi construído

### 1.1 Parser do contrato (novo, puro, isolado)

`src/evaluation/explicit_decision_contract.py` —
`extract_decisao_final(raw_text) -> DecisionContractResult`.

- Sintaxe única documentada no módulo: uma linha dedicada
  `decisao_final: <valor>`, onde `<valor>` é EXATAMENTE um de
  `"melanoma"`, `"nao_melanoma"`, `"indeterminado"` (comparação
  case-insensitive só quanto à caixa, nunca fuzzy — "melanoma." com
  ponto, ou "suspeita de melanoma" como valor, são inválidos por
  desenho).
- Roda ESTRUTURALMENTE APÓS `src.evaluation.response_template_extraction
  .extract_final_answer` (reusado sem modificação, doc 22) — o campo é
  procurado SÓ na resposta final já livre do bloco de template, nunca
  no texto bruto completo. É isto que garante que um rótulo mencionado
  dentro de um bloco de pensamento (`<unused94>thought ... <unused95>`)
  nunca seja lido como a decisão declarada.
- Cinco status possíveis: `ok`, `extraction_failed` (etapa anterior
  falhou — nunca escaneia texto parcial), `field_absent`,
  `field_repeated` (inclusive quando os valores repetidos são iguais
  entre si), `invalid_value`. `decisao_final` só é preenchido quando
  `decision_status == "ok"` — nunca um valor "melhor esforço" nos
  demais casos.
- Nunca importa `case_status.py`/`case_status_v2.py` — via de
  classificação separada e paralela, nunca um substituto ou fallback
  um do outro.

### 1.2 Instruções de tarefa experimentais (novo, isolado)

`src/multiagent/experimental_decision_contract.py`:

- `EXPERIMENTAL_VLM_DIRECT_TASK_INSTRUCTION` — preserva a tarefa
  melanoma-versus-não-melanoma a partir só da imagem; troca as 3
  categorias textuais fixas do `VLM_DIRECT_TASK_INSTRUCTION` oficial
  (`src/multiagent/vlm_direct.py`, não modificado) pela linha
  `decisao_final`.
- `build_experimental_summary_task_instruction(diagnosis_abcd,
  diagnosis_menzies, diagnosis_spcl)` — preserva a tarefa atual do
  sumarizador oficial (`SummaryAgent.summarize`,
  `src/agents/agents.py:441-484`, não modificado): avaliar a imagem
  dermatoscópica e os pareceres disponíveis. Diferença deliberada: um
  parecer `None` (cenário "parecer ausente") é OMITIDO da lista
  numerada, não serializado como um item "– None" (comportamento do
  prompt oficial, não alterado lá).
- As duas instruções compartilham o MESMO bloco de contrato
  (`_DECISION_CONTRACT_INSTRUCTION_BLOCK`) — garante a mesma sintaxe,
  lida pelo mesmo parser, nos dois papéis.
- Nenhum dos dois módulos novos é importado por `agents.py`,
  `vlm_direct.py`, `case_status_v2.py` ou
  `scripts/run_local_vlm_pilot.py` — zero mudança de comportamento em
  qualquer caminho de produção ou critério de avanço do Estágio 1.

### 1.3 Verificação manual (sem inferência)

`scripts/verify_explicit_decision_contract.py` — 8 cenários com texto
BRUTO SIMULADO (escrito à mão, nunca uma resposta real de modelo),
cobrindo exatamente o pedido:

1. justificativa cita "benigno" numa enumeração de pareceres
   conflitantes, sem declarar `decisao_final` → esperado
   `field_absent`;
2. mesmo desacordo, com `decisao_final: indeterminado` declarado →
   esperado `ok`/`indeterminado` (nenhuma categoria binária imposta);
3. decisão válida (`melanoma`) que diverge do parecer isolado do ABCD
   (`benigna`, mencionado no corpo) → esperado `ok`/`melanoma`, prova
   de que o corpo do texto nunca é lido para decidir;
4. quatro subcenários de falha de formato — ausente (4a), valor
   inválido "suspeito" (4b), repetido com valores diferentes (4c),
   repetido com valores IGUAIS (4d, ainda assim `field_repeated`);
5. bloco de template menciona "melanoma" informalmente no raciocínio
   interno; a resposta final (pós-template) declara
   `decisao_final: nao_melanoma` → esperado `ok`/`nao_melanoma`, só o
   rótulo pós-extração é lido.

**Comando exato para o pesquisador rodar:**

```
python scripts/verify_explicit_decision_contract.py
```

**Saída esperada:** 8 linhas `[PASS] <nome_do_cenário>` (uma por
cenário, ver lista acima) seguidas do resumo final
`Resumo: 8 PASS, 0 FAIL, 8 cenários no total.` — qualquer `[FAIL]`
indica que o parser não está lendo a linha exatamente como
documentado.

## 2. O que NÃO foi feito nesta tarefa (explicitamente fora do escopo)

- Nenhuma mudança em `src/agents/agents.py` (prompts oficiais),
  `src/multiagent/vlm_direct.py` (`VLM_DIRECT_TASK_INSTRUCTION`,
  `run_vlm_direct_call`, `run_multiagent_role_call`,
  `run_synthesis_control`/`run_synthesis_control_real_image`),
  `src/evaluation/case_status_v2.py` ou
  `scripts/run_local_vlm_pilot.py` (critério de avanço do Estágio 1,
  gate "parseable"/"passed" via v2) — todos lidos para confirmar os
  pontos de acoplamento, nenhum escrito.
- Nenhuma execução real: sem `pytest`, sem chamada a Ollama/qwen3-vl/
  medgemma, sem o piloto das 12 lesões, sem a amostra congelada de
  3.000 lesões.
- Nenhuma ligação do parser/instruções experimentais a um caminho de
  chamada real (`run_vlm_direct_call`/`run_multiagent_role_call`) —
  os dois módulos novos só existem como protótipo importável, análogo
  ao passo 3 deferido do doc 22.
- A próxima entrega (repetir os 4 controles de síntese com imagem
  real, uma chamada por cenário, Qwen3-VL e MedGemma, sem retry) é uma
  decisão SEPARADA, explicitamente não aprovada automaticamente por
  este protótipo — aguarda decisão própria do pesquisador.

## 3. Arquivos alterados/criados nesta entrega

- `src/evaluation/explicit_decision_contract.py` (novo)
- `src/multiagent/experimental_decision_contract.py` (novo)
- `scripts/verify_explicit_decision_contract.py` (novo)
- `docs/reconstruction/23_explicit_decision_contract_prototype.md` (novo, este arquivo)

Nenhum arquivo existente foi modificado.
