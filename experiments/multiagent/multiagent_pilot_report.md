# Piloto do Fluxo Multiagente — Respostas Simuladas (Etapa 5)

**Sem credenciais, sem chamadas de rede.** Camada de LLM inteiramente substituída por um dublê determinístico. Não é inferência real, não usa HAM10000, não integra checkpoint de CNN, não executa S2 com máscara real.

## Crítico/RAG nunca roda (campo `final_report` permanece None em todos os cenários): True

## Nenhuma divergência encontrada nos cenários testados

## Resultado por cenário

| Cenário | Terminou sem erro? | Chaves no estado final | `final_report` presente? | Chamadas por agente |
|---|---|---|---|---|
| `caminho_normal` | True | ['diagnosis_abcd', 'diagnosis_menzies', 'diagnosis_spcl', 'image_data', 'image_path', 'lesion_size', 'seg_image_data', 'seg_path', 'status_abcd', 'status_menzies', 'status_spcl', 'validation'] | False | {'diagnostic_abcd': 1, 'diagnostic_spcl': 1, 'diagnostic_menzies': 1, 'summary': 1} |
| `agente_lanca_excecao` | False | None | None | {'diagnostic_abcd': 1, 'diagnostic_spcl': 1, 'diagnostic_menzies': 1} |
| `agente_fora_do_schema` | False | None | None | {'diagnostic_abcd': 1, 'diagnostic_spcl': 1, 'diagnostic_menzies': 1} |
| `agente_nao_responde` | True | ['diagnosis_abcd', 'diagnosis_menzies', 'diagnosis_spcl', 'image_data', 'image_path', 'lesion_size', 'seg_image_data', 'seg_path', 'status_abcd', 'status_menzies', 'status_spcl', 'validation'] | False | {'diagnostic_abcd': 1, 'diagnostic_spcl': 1, 'diagnostic_menzies': 1, 'summary': 1} |

### Detalhe — `caminho_normal`

- Erro observado: `None`
- `validation` presente no estado final: True
- Campo `diagnostico` final (se houver): `Lesão Benigna`
- Sumarizador foi invocado: True
- Marcadores dos 3 pareceres encontrados no prompt do sumarizador: 3 (esperado 3 no caminho normal)
- Consolidação: 3/3 pareceres (agentes ausentes: [])
- Cobertura diagnóstica (sumarizador produziu diagnóstico válido, independente de consolidação parcial): True
- Status final: determinado

### Detalhe — `agente_lanca_excecao`

- Erro observado: `RuntimeError: falha simulada de API`
- `validation` presente no estado final: None
- Campo `diagnostico` final (se houver): `None`
- Sumarizador foi invocado: None
- Marcadores dos 3 pareceres encontrados no prompt do sumarizador: None (esperado 3 no caminho normal)
- Consolidação: None/None pareceres (agentes ausentes: None)
- Cobertura diagnóstica (sumarizador produziu diagnóstico válido, independente de consolidação parcial): None
- Status final: None

### Detalhe — `agente_fora_do_schema`

- Erro observado: `ValidationError: 4 validation errors for GraphState
diagnosis_menzies.str
  Input should be a valid string [type=string_type, input_value={'campo_inesperado': 'valor'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/string_type
diagnosis_menzies.MenziesDiagnosticAnswer.positivas
  Field required [type=missing, input_value={'campo_inesperado': 'valor'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
diagnosis_menzies.MenziesDiagnosticAnswer.negativas
  Field required [type=missing, input_value={'campo_inesperado': 'valor'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
diagnosis_menzies.MenziesDiagnosticAnswer.explicacao
  Field required [type=missing, input_value={'campo_inesperado': 'valor'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing`
- `validation` presente no estado final: None
- Campo `diagnostico` final (se houver): `None`
- Sumarizador foi invocado: None
- Marcadores dos 3 pareceres encontrados no prompt do sumarizador: None (esperado 3 no caminho normal)
- Consolidação: None/None pareceres (agentes ausentes: None)
- Cobertura diagnóstica (sumarizador produziu diagnóstico válido, independente de consolidação parcial): None
- Status final: None

### Detalhe — `agente_nao_responde`

- Erro observado: `None`
- `validation` presente no estado final: True
- Campo `diagnostico` final (se houver): `Lesão Benigna`
- Sumarizador foi invocado: True
- Marcadores dos 3 pareceres encontrados no prompt do sumarizador: 2 (esperado 3 no caminho normal)
- Consolidação: 2/3 pareceres (agentes ausentes: ['diagnostic_spcl'])
- Cobertura diagnóstica (sumarizador produziu diagnóstico válido, independente de consolidação parcial): True
- Status final: determinado
