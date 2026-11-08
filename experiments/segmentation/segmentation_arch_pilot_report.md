# Piloto — comparação de arquiteturas de segmentação (U-Net vs DeepLabV3+-ResNet34)

**Verificação técnica mecânica, não treino real.** Não seleciona arquitetura, métrica final nem limiar de máscara definitivos — essas decisões ficam para a execução científica completa (5 folds externos x 4 internos, não implementada aqui). HAM10000 não foi lido nesta execução.

## Resultado geral: TODAS AS VERIFICACOES MECANICAS PASSARAM

- Dispositivo solicitado: `mps` / efetivo: `mps`
- Resolução de treino: 256x256
- Threshold de demonstração: 0.5 (NÃO é o limiar escolhido por validação)
- Política de inicialização: Ambas arquiteturas usam inicializacao aleatoria neste piloto (weights=None, sem download) -- decisao do piloto, nao decide a politica de pre-treinamento do experimento definitivo.
- Arquivo de folds: `/Users/andrey.basilio/Projetos/master/mestrado_ifes/experiments/segmentation/isic2018_task1_folds.json`
- Chave de agrupamento dos folds: image:{image_id} (sem lesion_id -- ver src/segmentation/folds.py)
- Fold externo: 0 (519 imagens, NUNCA usado neste piloto)
- Fold de validação interna: 4 (518 imagens no fold; amostra demo: 3)
- Folds de treino piloto: [1, 2, 3] (1557 imagens; amostra piloto: 12)
- HAM10000 tocado: False

## Correspondência imagem↔máscara

- Imagens: 2594 / Máscaras: 2594 / Pareadas: 2594 / Correspondência perfeita: True

## Arquitetura: `unet`

| Verificação mecânica | Resultado |
|---|---|
| `leitura_alinhamento` | PASSOU |
| `atualizacao_peso` | PASSOU |
| `checkpoint_save_reload` | PASSOU |
| `mascara_predita` | PASSOU |

- Tempo de execução: 3.34s
- Imagens de treino piloto usadas: 12
- Imagens de validação interna usadas nas métricas-demo: 3
- Medição de tempo (1 época, amostra de piloto): 0.42s para 12 imagens (0.035s/imagem) — Isto e so medicao de tempo (1 epoca sobre a amostra de piloto, nao o dataset inteiro) para planejar a execucao aninhada real -- nao e treino real nem selecao de limiar/arquitetura.

### Métricas de demonstração (threshold=0.5)

- Casos: total=3 ok=3 por status={'ok': 3}
- Dice médio (ok): 0.0
- IoU médio (ok): 0.0
- Thresholded Jaccard médio (ok): 0.0
- Taxa de falha ISIC (ok): 1.0
- HD95 mediana (valores finitos): None / indefinidos: 3
- HD95 mediana normalizada pela diagonal (valores finitos): None

| image_id | status | dimensões originais | dice | iou | thresholded_jaccard | hd95 | hd95_norm_diag | falha_isic |
|---|---|---|---|---|---|---|---|---|
| `ISIC_0000009` | ok | [767, 1022] | 0.0 | 0.0 | 0.0 | None | None | True |
| `ISIC_0000012` | ok | [767, 1022] | 0.0 | 0.0 | 0.0 | None | None | True |
| `ISIC_0000015` | ok | [767, 1022] | 0.0 | 0.0 | 0.0 | None | None | True |

## Arquitetura: `deeplabv3plus`

| Verificação mecânica | Resultado |
|---|---|
| `leitura_alinhamento` | PASSOU |
| `atualizacao_peso` | PASSOU |
| `checkpoint_save_reload` | PASSOU |
| `mascara_predita` | PASSOU |

- Tempo de execução: 3.66s
- Imagens de treino piloto usadas: 12
- Imagens de validação interna usadas nas métricas-demo: 3
- Medição de tempo (1 época, amostra de piloto): 0.28s para 12 imagens (0.023s/imagem) — Isto e so medicao de tempo (1 epoca sobre a amostra de piloto, nao o dataset inteiro) para planejar a execucao aninhada real -- nao e treino real nem selecao de limiar/arquitetura.

### Métricas de demonstração (threshold=0.5)

- Casos: total=3 ok=3 por status={'ok': 3}
- Dice médio (ok): 0.2808732370275
- IoU médio (ok): 0.16981428137685395
- Thresholded Jaccard médio (ok): 0.0
- Taxa de falha ISIC (ok): 1.0
- HD95 mediana (valores finitos): None / indefinidos: 3
- HD95 mediana normalizada pela diagonal (valores finitos): None

| image_id | status | dimensões originais | dice | iou | thresholded_jaccard | hd95 | hd95_norm_diag | falha_isic |
|---|---|---|---|---|---|---|---|---|
| `ISIC_0000009` | ok | [767, 1022] | 0.37580471713946495 | 0.2313790226490482 | 0.0 | None | None | True |
| `ISIC_0000012` | ok | [767, 1022] | 0.09582179468493417 | 0.0503218629524643 | 0.0 | None | None | True |
| `ISIC_0000015` | ok | [767, 1022] | 0.37099319925810087 | 0.2277419585290493 | 0.0 | None | None | True |
