# Divisão Treino/Validação do BCN20000 — CNN Baseline (preparação de dados)

**Isto é preparação de dados, não treino.** Nenhum modelo foi treinado, nenhum limiar foi selecionado, nenhuma avaliação no HAM10000 foi feita. O HAM10000 não foi lido nem referenciado nesta execução.

- Política de partição (`--partition-policy`): `apenas_train`
- Proporção de validação solicitada (`--val-fraction`): 0.2
- Semente (`--seed`): 20261107
- Registros excluídos antes da divisão (sem lesion_id, diagnosis inutilizável, ou lesion_id conflitante): 0

## Verificação de não sobreposição de lesão

- Nenhum `lesion_id` em treino E validação simultaneamente: True

## Composição por split

| Split | Imagens | Lesões | Melanoma (img) | Não-melanoma (img) | Proporção melanoma (img) | Melanoma (lesão) | Proporção melanoma (lesão) |
|---|---|---|---|---|---|---|---|
| `train` | 9912 | 2861 | 2281 | 7631 | 0.2301 | 419 | 0.1465 |
| `val` | 2501 | 715 | 576 | 1925 | 0.2303 | 105 | 0.1469 |

- Fração de validação alcançada (por imagem): 0.2015 (solicitada: 0.2, desvio: 0.0015)
- Fração de validação alcançada (por lesão): 0.1999 (solicitada: 0.2, desvio: 0.0001)

## Detalhe por classe (ajuste estratificado)

| Classe | Lesões | Val | Train |
|---|---|---|---|
| `melanoma` | 524 | 105 | 419 |
| `nao_melanoma` | 3052 | 610 | 2442 |

## Aviso

Este manifesto é preparação de dados para a etapa 3 (baseline CNN). Seleção de limiar (índice de Youden) acontece depois, exclusivamente na validação interna aqui gerada. O manifesto da amostra do HAM10000 permanece intacto e reservado à avaliação externa final.
