# Fixacao de epoca/limiar final -- deeplabv3plus

- Epocas fixadas: **17** (mediana round-half-up de [18, 15, 16, 20, 17])
- Limiar escolhido (busca out-of-fold, 5 treinos novos): **0.5**
- Empate: False / Empate simetrico residual: False

| limiar | dice medio out-of-fold |
|---|---|
| 0.3 | 0.8545116312028292 |
| 0.35 | 0.8606056273170446 |
| 0.4 | 0.8647649007578837 |
| 0.45 | 0.8672879551991902 |
| 0.5 | 0.8683257409912402 |
| 0.55 | 0.8678702929726541 |
| 0.6 | 0.8660041394821592 |
| 0.65 | 0.8627015135218381 |
| 0.7 | 0.857749006763831 |

## Procedimento (registrado, nao reaberto)

chosen_epochs = mediana (round-half-up) dos chosen_epochs das 5 rodadas externas reais (nao recomputado, lido dos arquivos fonte). chosen_threshold = busca de limiar out-of-fold: 5 treinos novos (nao os 50 originais), cada um com chosen_epochs fixo, sem early stopping, no conjunto de 4 folds; Dice por imagem no quinto fold (nunca visto por aquele treino) pooled nas 5 repeticoes e avaliado na grade 0.30-0.70. Os limiares das 5 rodadas externas originais NAO foram promediados nem reutilizados.

HAM10000 tocado: False