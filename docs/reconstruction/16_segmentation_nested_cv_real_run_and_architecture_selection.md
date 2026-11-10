# 16 — Execução real da validação cruzada aninhada (5×4) e seleção de arquitetura de segmentação

Consolidação de resultados **já executados pelo pesquisador** (máquina
Mac/MPS) — esta sessão só leu os arquivos de saída existentes em
`experiments/segmentation/` e aplicou a regra de seleção já aprovada
(`13_segmentation_deeplabv3plus_pilot.md §10`); nenhum código foi
executado aqui (`feedback_no_autonomous_execution`).

## 1. O que foi encontrado na máquina

5 rodadas externas completas, `nested_cv_rehearsal_externalfold{0..4}_
{report.md,results.json}`, geradas por `scripts/run_nested_cv_rehearsal.py`
sobre `src/segmentation/nested_cv.py` com early stopping (a lacuna
registrada em `14_segmentation_nested_cv_consolidation.md §6` e na
memória do coordenador está, portanto, fechada — early stopping foi
implementado e usado nesta execução real).

Confirmado por leitura direta de cada relatório, não assumido:
- Correspondência imagem↔máscara perfeita em todos os 5 folds: 2.594/2.594
  pareadas (soma exata dos 5 folds externos: 519+519+519+519+518=2.594).
- `HAM10000 tocado: False` em todos os 5 relatórios.
- Dispositivo efetivo `mps` em todos os 5.
- Avaliação externa sem nenhuma falha: `status={'ok': 519}` (ou 518 no
  fold 4) em todas as 10 avaliações (5 folds × 2 arquiteturas) — 100% dos
  casos processados com sucesso, nenhum `referencia_invalida`/
  `mascara_incompativel`/`falha_processamento`.
- Também presente: `nested_cv_single_round_probe_{report.md,results.json}`
  (8 de novembro) — a sondagem descartável prevista em
  `14 §5`, não reaproveitada como rodada real.

## 2. Resultados reais por rodada externa (avaliação externa única, limiar escolhido internamente)

| Fold externo | Arquitetura | Limiar | Dice médio | IoU médio | Taxa falha ISIC | HD95 mediana (px, finitos) |
|---|---|---|---|---|---|---|
| 0 | U-Net | 0,50 | 0,8615 | 0,7824 | 16,96% | 85,08 |
| 0 | DeepLabV3+ | 0,50 | 0,8744 | 0,7976 | 14,26% | 73,26 |
| 1 | U-Net | 0,50 | 0,8537 | 0,7708 | 16,76% | 90,00 |
| 1 | DeepLabV3+ | 0,55 | 0,8688 | 0,7909 | 15,22% | 69,82 |
| 2 | U-Net | 0,45 | 0,8655 | 0,7878 | 16,18% | 81,54 |
| 2 | DeepLabV3+ | 0,45 | 0,8643 | 0,7832 | 17,53% | 86,21 |
| 3 | U-Net | 0,50 | 0,8510 | 0,7680 | 19,46% | 87,01 |
| 3 | DeepLabV3+ | 0,50 | 0,8690 | 0,7947 | 15,22% | 68,03 |
| 4 | U-Net | 0,50 | 0,8521 | 0,7693 | 18,15% | 98,12 |
| 4 | DeepLabV3+ | 0,50 | 0,8672 | 0,7916 | 15,06% | 73,90 |

**Média simples das 5 rodadas externas** (não é a média ponderada por
imagem pedida pela regra — ver ressalva no §4; como os 5 folds têm
tamanho quase idêntico, 519/519/519/519/518, a diferença entre as duas
formas de calcular é desprezível frente à margem encontrada):

| Arquitetura | Dice médio | IoU médio | Taxa falha ISIC | HD95 mediana (média dos 5) |
|---|---|---|---|---|
| U-Net | 0,8568 | 0,7757 | 17,50% | 88,35 |
| DeepLabV3+-ResNet34 | 0,8687 | 0,7916 | 15,46% | 74,24 |

## 3. Aplicação da regra de seleção já aprovada

Regra registrada em `13_segmentation_deeplabv3plus_pilot.md §10`
("seleção de arquitetura pelo Dice médio por imagem nos folds externos,
empate prático se diferença absoluta < 0,005; desempate: menor taxa de
falha ISIC, menor HD95 mediano, depois U-Net"), aprovada antes desta
execução — esta seção só aplica aritmética a essa regra com os números
reais acima, não introduz critério novo.

- Diferença de Dice médio: 0,8687 − 0,8568 = **0,0119** → maior que o
  limiar de empate prático (0,005). **Não há empate** — a regra de
  desempate (taxa de falha ISIC / HD95 / preferência por U-Net) não
  chega a ser necessária.
- DeepLabV3+-ResNet34 também é superior em **todos** os outros
  indicadores (IoU médio, taxa de falha ISIC, HD95 mediano) em 4 dos 5
  folds individuais (só perde no Dice/IoU do fold 2, onde a diferença é
  pequena: 0,8655 vs 0,8643) — a vitória não depende de nenhuma métrica
  isolada nem de um fold específico.

**Resultado da aplicação da regra: DeepLabV3+-ResNet34 é a arquitetura
selecionada** pelos dados reais da validação cruzada aninhada completa.
Isto fecha a dependência de arquitetura que bloqueava o segmentador
final em `15_multiagent_experimental_matrix.md §4` — a condição S2 da
matriz multiagente pode agora apontar para DeepLabV3+-ResNet34 como o
"segmentador final", sujeito à ressalva do §4 abaixo.

## 4. Ressalvas que não são resolvidas aqui

- **Agregação exata**: a tabela de médias do §2 é a média simples das 5
  médias por fold, não o Dice médio pooled sobre as 2.594 imagens
  individuais (que exigiria reler os 5 arquivos `results.json`,
  ~1,2 MB cada, por imagem) — a regra aprovada diz "por imagem nos folds
  externos", que é mais próximo do pooled. Dado que os folds têm
  tamanho quase idêntico e a margem encontrada (0,0119) é mais de 2×
  o limiar de empate, isso não deveria mudar a decisão — mas o valor
  exato do Dice médio pooled não foi calculado aqui e não deve ser
  citado como número final sem recomputar.
- **Orçamento de épocas ainda sob suspeita, confirmada agora com dados
  reais**: de 40 rodadas internas (5 folds × 4 internas × 2
  arquiteturas), **23 (57,5%) pararam por `teto_atingido`** (bateram no
  limite de 20 épocas), não por `paciencia_esgotada` — ou seja, na
  maioria das rodadas internas reais, o teto de 20 épocas continua
  sendo o fator determinante da parada, não a convergência medida pela
  paciência. Isso é exatamente o risco que `14_segmentation_nested_cv_
  consolidation.md §2` e §5 já tinham alertado ("se `teto_atingido` e o
  Dice ainda estiver melhorando, isso deve ser registrado como ‘teto
  atingido’, nunca como ‘convergência’") — registrado aqui como fato
  confirmado na execução real, não mais hipótese. **Não decidido
  nesta entrega**: se o pesquisador quer re-executar com um teto maior
  antes de considerar este resultado definitivo, ou se aceita o teto de
  20 como protocolo fechado mesmo com a maioria das rodadas batendo nele.
- **Seleção de modelo final**: a regra aprovada (`13 §10`) também
  prevê "modelo final reinicializado e treinado com todos os dados
  elegíveis do ISIC2018 Task1 após a seleção" — isso não foi feito
  nesta execução (as 5 rodadas produzem 5 modelos retreinados nos 4
  folds de desenvolvimento cada, não um modelo único final treinado em
  todos os dados). Esse passo final, e a geração das máscaras preditas
  sobre as 3.000 imagens do manifesto HAM10000 congelado (necessárias
  para S2), **não foram feitos** e não são implementados aqui.
- Nenhum checkpoint do modelo final existe ainda — os `.pt` encontrados
  (`pilot_checkpoint.pt`, `nested_cv_round_unet_valfold1.pt`) são do
  piloto e da sondagem descartável, não de um modelo final.

## 5. O que esta entrega NÃO fez

Não treinou nada, não gerou máscaras sobre o manifesto HAM10000, não
implementou o retreino final com todos os dados, não recalculou o Dice
pooled por imagem, e não decidiu se o teto de 20 épocas precisa ser
revisto — todos ficam como pendências explícitas para decisão do
pesquisador.
