# Sondagem de rodada única (descartável)

**SONDAGEM DESCARTAVEL -- nao e uma das rodadas reais da validacao aninhada; serve so para validar a regra de parada e medir tempo antes da execucao completa (5x4 rodadas). NAO rodou busca de limiar, retreino ou avaliacao externa.**

- Dispositivo solicitado: `mps` / efetivo: `mps`
- Fold externo: 0 (não usado nesta sondagem)
- Índice da rodada de sondagem: 0 (fold de validação interna: 1)
- max_epochs: 20 / patience: 5
- Threshold de monitoramento: 0.5 / lr: 0.0001
- HAM10000 tocado: False

## Arquitetura: `unet`

- Melhor época: 20 (Dice monitor: 0.8484)
- **Motivo da parada**: teto_atingido
- Épocas efetivamente rodadas: 20 de até 20
- Tempo total: 2607.9s / tempo médio por época: 130.4s
- Checkpoint: `/Users/andrey.basilio/Projetos/master/mestrado_ifes/experiments/segmentation/nested_cv_round_unet_valfold1.pt`

| época | dice_monitor |
|---|---|
| 1 | 0.7040 |
| 2 | 0.7299 |
| 3 | 0.7191 |
| 4 | 0.7930 |
| 5 | 0.8206 |
| 6 | 0.8024 |
| 7 | 0.8166 |
| 8 | 0.8084 |
| 9 | 0.8241 |
| 10 | 0.8275 |
| 11 | 0.8197 |
| 12 | 0.8263 |
| 13 | 0.8355 |
| 14 | 0.7928 |
| 15 | 0.8328 |
| 16 | 0.8059 |
| 17 | 0.8456 |
| 18 | 0.8423 |
| 19 | 0.8361 |
| 20 | 0.8484 |

## Arquitetura: `deeplabv3plus`

- Melhor época: 17 (Dice monitor: 0.8719)
- **Motivo da parada**: teto_atingido
- Épocas efetivamente rodadas: 20 de até 20
- Tempo total: 2343.5s / tempo médio por época: 117.2s
- Checkpoint: `/Users/andrey.basilio/Projetos/master/mestrado_ifes/experiments/segmentation/nested_cv_round_deeplabv3plus_valfold1.pt`

| época | dice_monitor |
|---|---|
| 1 | 0.7751 |
| 2 | 0.8206 |
| 3 | 0.8287 |
| 4 | 0.8534 |
| 5 | 0.8323 |
| 6 | 0.8477 |
| 7 | 0.8484 |
| 8 | 0.8561 |
| 9 | 0.8424 |
| 10 | 0.8601 |
| 11 | 0.8510 |
| 12 | 0.8688 |
| 13 | 0.8663 |
| 14 | 0.8230 |
| 15 | 0.8569 |
| 16 | 0.8618 |
| 17 | 0.8719 |
| 18 | 0.8714 |
| 19 | 0.8495 |
| 20 | 0.8603 |
