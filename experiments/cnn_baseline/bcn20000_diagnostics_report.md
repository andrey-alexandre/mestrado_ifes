# Diagnóstico das Partições Oficiais do BCN20000 (preparação da Tarefa de divisão treino/validação)

**Isto é só diagnóstico — nenhuma divisão treino/validação foi gerada nesta execução.** O HAM10000 não foi lido nem referenciado; é reservado exclusivamente para avaliação externa final.

## Partição `train`

- Linhas no CSV: 12413
- Imagens encontradas no zip local: 12413
- Imagens do CSV ausentes do zip local: 0
- Arquivos no zip sem linha correspondente no CSV: 0
- Linhas com rótulo utilizável (`diagnosis` válido): 12413
- Linhas com `diagnosis` vazio: 0
- Linhas com `diagnosis` inesperado: 0 (valores: [])
- Linhas sem `lesion_id`: 0
- `lesion_id` com `diagnosis` conflitante entre suas linhas: 0
- Lesões distintas (com `lesion_id` válido): 3576

| Categoria | Linhas | Lesões |
|---|---|---|
| `AK` | 737 | 236 |
| `BCC` | 2809 | 983 |
| `BKL` | 1138 | 353 |
| `DF` | 124 | 40 |
| `MEL` | 2857 | 524 |
| `NV` | 4206 | 1281 |
| `SCC` | 431 | 122 |
| `VASC` | 111 | 37 |

- Registros inelegíveis (qualquer motivo): 0

## Partição `test`

- Linhas no CSV: 6533
- Imagens encontradas no zip local: 6501
- Imagens do CSV ausentes do zip local: 32 (['BCN_0000000705.jpg', 'BCN_0000000790.jpg', 'BCN_0000001537.jpg', 'BCN_0000001982.jpg', 'BCN_0000002709.jpg']...)
- Arquivos no zip sem linha correspondente no CSV: 0
- Linhas com rótulo utilizável (`diagnosis` válido): 0
- Linhas com `diagnosis` vazio: 6533
- Linhas com `diagnosis` inesperado: 0 (valores: [])
- Linhas sem `lesion_id`: 0
- `lesion_id` com `diagnosis` conflitante entre suas linhas: 0
- Lesões distintas (com `lesion_id` válido): 1864

| Categoria | Linhas | Lesões |
|---|---|---|
| `AK` | 0 | 0 |
| `BCC` | 0 | 0 |
| `BKL` | 0 | 0 |
| `DF` | 0 | 0 |
| `MEL` | 0 | 0 |
| `NV` | 0 | 0 |
| `SCC` | 0 | 0 |
| `VASC` | 0 | 0 |

- Registros inelegíveis (qualquer motivo): 6533

## Possível sobreposição de lesão entre `train` e `test` (heurística, NÃO confirmada)

- Método: heuristic: strip non-digit characters, compare as integer (removes underscore and zero-padding differences)
- `lesion_id` distintos em treino: 3576
- `lesion_id` distintos em teste: 1864
- Colisões após normalização: 0
- **Coincidência numérica após normalização NÃO confirma que seja a mesma lesão -- os dois formatos podem usar faixas de numeração não relacionadas. Não usar este resultado para decidir exclusão nem para afirmar sobreposição confirmada.**

## Aviso

Este relatório é só diagnóstico. A recomendação de política de partições (usar só `train` ou incorporar `test` rotulado) é apresentada separadamente para decisão do pesquisador — não está implementada como padrão silencioso em nenhum lugar do código.
