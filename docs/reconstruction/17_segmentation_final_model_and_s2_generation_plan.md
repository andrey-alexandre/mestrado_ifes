# 17 — Decisão de arquitetura confirmada, detalhe do orçamento de épocas, e plano para o modelo final + máscaras de S2

Sessão de implementação estática (2026-11-10) — nenhum script foi
executado (`feedback_no_autonomous_execution`). Três scripts novos foram
escritos para o pesquisador rodar no Mac, nesta ordem.

## 1. Decisão confirmada pelo pesquisador

**DeepLabV3+-ResNet34 é o segmentador final do protocolo.** Dice médio
0,869 vs. 0,857 do U-Net nos 5 folds externos reais (doc 16); diferença
0,012 > margem de empate de 0,005 — a regra aprovada (`13 §10`) resolve
a seleção sem recorrer aos desempates. Registrado aqui como decisão
fechada, não reaberta.

## 2. Orçamento de épocas — duas estatísticas distintas, por arquitetura

O pesquisador pediu que estas duas informações fossem mantidas separadas
(não são a mesma coisa): quantas rodadas bateram no teto operacional de
20 épocas, e quantas delas tiveram a melhor época exatamente na última
(evidência mais forte — "ainda melhorando até o fim" — do que só
"bateu no teto sem esgotar a paciência"). Contagem real, lida diretamente
das 5 tabelas de rodadas internas (`nested_cv_rehearsal_externalfold{0..4}_
report.md`), 20 rodadas por arquitetura (5 folds × 4 rodadas internas):

| Arquitetura | Rodadas com `teto_atingido` | Rodadas com melhor época = 20 (a própria última) |
|---|---|---|
| U-Net | 11/20 (55%) | 2/20 (10%) |
| DeepLabV3+-ResNet34 | 12/20 (60%) | 2/20 (10%) |
| **Combinado** | **23/40 (57,5%)** | **4/40 (10%)** |

**Leitura correta, não a mesma conclusão que uma leitura apressada da
primeira coluna sugeriria**: na maioria das rodadas que bateram no teto
(19 das 23), o Dice de monitoramento já tinha parado de melhorar antes
da época 20, mas ainda não tinha acumulado 5 épocas consecutivas sem
melhora (paciência) quando o teto chegou — ou seja, essas rodadas não
estavam necessariamente "subindo até o fim", só não tinham disparado a
paciência ainda. Apenas 4 das 40 rodadas (10%, 2 por arquitetura) tiveram
o pico exatamente na última época permitida, a evidência mais direta de
"ainda melhorando quando parou".

**Conclusão, nos termos definidos pelo pesquisador**: a DeepLabV3+ venceu
sob a receita e o orçamento de treino definidos (20 épocas, paciência 5,
LR/otimizador/lote fixos, inicialização aleatória nas duas) — **a
convergência não foi demonstrada** (nem para a arquitetura vencedora, nem
para a perdedora; a maioria das rodadas de ambas terminou por teto, não
por esgotamento de paciência, mesmo que só 10% tenha o pico exatamente
na borda). O resultado vale como comparação sob orçamento fixo, não como
afirmação de que qualquer uma das duas chegou ao seu melhor Dice
possível.

**Decisão do pesquisador, registrada aqui**: não repetir os 50 treinos
(nem para re-estimar o orçamento, nem para re-selecionar arquitetura).
Esse teto permanece como protocolo fechado para fins de seleção de
arquitetura.

## 3. Plano para o modelo final e as máscaras de S2 — três scripts, nesta ordem

Nenhum deles foi executado nesta sessão. Todos reusam funções já
existentes e testadas de `src/segmentation/nested_cv.py` e
`scripts/run_nested_cv_rehearsal.py` — nada da lógica de treino/limiar
foi reimplementado do zero.

### 3.1 `scripts/run_segmentation_final_config_fit.py`

Fixa época e limiar para a configuração DeepLabV3+-ResNet34 selecionada,
**sem repetir os 50 treinos**:

- **Épocas**: mediana (round-half-up, mesma regra já testada em
  `nested_cv.select_epochs_via_median`) dos 5 `chosen_epochs` reais já
  gravados em `nested_cv_rehearsal_externalfold{0..4}_results.json` para
  `deeplabv3plus` (18, 15, 16, 20, 17 → **mediana = 17**, recomputada
  pelo próprio script a partir dos arquivos-fonte, não hardcoded).
- **Limiar — decisão de design explícita, sujeita à sua confirmação
  antes de rodar**: o pesquisador rejeitou promediar os 5 limiares
  externos (0,50/0,55/0,45/0,50/0,50) porque cada um foi escolhido com
  predições de modelos em números de época DIFERENTES (o próprio
  conjunto de melhores épocas internas de cada rodada externa), não na
  época final fixa (17). A interpretação implementada: busca de limiar
  **out-of-fold em 5 TREINOS NOVOS** (não 50) — para cada um dos 5 folds
  do ISIC2018 Task1, treina do zero um DeepLabV3+ por exatamente 17
  épocas (sem early stopping, já que a época está fixa) nos outros 4
  folds, e coleta o Dice por imagem na grade 0,30–0,70 no fold restante
  (nunca visto por aquele treino). Os Dices dos 5 folds são agrupados
  (as 2.594 imagens, cada uma usada exatamente uma vez como "fora do
  treino") e o limiar é escolhido pelo mesmo critério já testado
  (`nested_cv.select_threshold_from_internal_dice`). **Se esta não for a
  leitura pretendida de "predições fora do treino dessa configuração",
  avise antes de rodar** — é a interpretação mais próxima do texto que
  evita reusar os 5 limiares originais e evita os 50 treinos completos,
  mas é uma escolha de design desta entrega, não uma instrução literal
  sua.
- Saída: `experiments/segmentation/final_config_fit_{results.json,
  report.md,log}` + 5 checkpoints intermediários/descartáveis
  (`nested_cv_retrain_deeplabv3plus_extfoldoof{0..4}.pt`) — explicitamente
  marcados como não-finais no relatório.

### 3.2 `scripts/run_segmentation_final_retrain.py`

Lê época (17) e limiar do passo 3.1, reinicializa o DeepLabV3+-ResNet34 e
treina do zero em **todas as 2.594 imagens elegíveis** do ISIC2018 Task1
(sem held-out), pelas 17 épocas fixas, sem early stopping. Verifica antes
de treinar que há exatamente 2.594 imagens elegíveis (aborta se não).
Salva o checkpoint único (`final_model_deeplabv3plus.pt`), calcula seu
SHA-256 e tamanho em bytes, e grava `final_model_frozen_config.json` com
tudo congelado antes de qualquer leitura do HAM10000: arquitetura,
épocas, limiar, caminho/hash do checkpoint, pré-processamento (resize
256×256 bilinear na imagem, `ToTensor` para [0,1], sem normalização
adicional — mesma convenção já usada em todo o estágio 4) e
pós-processamento (sigmoid → restaurar para a resolução nativa da imagem
de entrada via interpolação bilinear → binarizar no limiar congelado,
restauração sempre antes da binarização). HAM10000 não é importado nem
referenciado neste script — confirmado por inspeção do código, não só
por convenção.

### 3.3 `scripts/run_s2_mask_generation.py`

Gera as máscaras preditas sobre as 3.000 imagens do manifesto HAM10000
congelado:

- **Primeiro passo, incondicional**: recalcula o SHA-256 do
  `sample_manifest.csv` e compara com o hash registrado
  (`32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`,
  `10_frozen_sample_registry.md`). Divergência → `SystemExit(1)` com
  mensagem explícita, nenhuma continuação silenciosa com outro arquivo.
- Carrega o checkpoint final + limiar congelado do passo 3.2 (não
  recalcula nada).
- Para cada um dos 3.000 casos: carrega a imagem HAM10000 nativa, aplica
  o mesmo pipeline de pré/pós-processamento congelado no passo 3.2
  (resize 256 → modelo → sigmoid → restaurar para a resolução nativa da
  própria imagem HAM10000 → binarizar no limiar), salva a máscara
  predita em PNG e grava uma linha no CSV de saída — **sempre uma linha
  por caso**, inclusive quando a máscara sai vazia (0 pixels positivos,
  registrado como `mask_empty=True`, nunca convertido/omitido) e
  inclusive quando o caso falha (ex. imagem ausente — registrado como
  `status="falha"` com a mensagem de erro, nunca descartado, nunca
  reexecutado, nunca substituído pela máscara de referência Tschandl).
  A máscara de referência do HAM10000 **não é lida em nenhum ponto deste
  script** — confirmado por inspeção, não só por convenção; S2 é
  estritamente a predição do segmentador, nunca a verdade de referência
  usada por S1.
- Ao final, valida que o número de linhas escritas é exatamente 3.000
  (`RuntimeError` se não for — nenhuma linha pode desaparecer
  silenciosamente) e imprime a contagem de OK/vazias/falhas.
- Saída: `experiments/segmentation/s2_mask_generation_{manifest.csv,
  results.json,report.md}` + uma máscara PNG por caso bem-sucedido em
  `experiments/segmentation/s2_masks/`.

## 4. O que libera a próxima entrega

Conforme definido pelo pesquisador: o checkpoint final congelado (passo
3.2, com SHA-256 registrado) e as 3.000 imagens do manifesto
contabilizadas na geração de S2 (passo 3.3, com a linha de cada caso,
inclusive vazias/falhas). A partir disso, a preparação conjunta de
S0/S1/S2 pode começar — ainda sem nenhuma chamada a VLM, conforme
combinado.

## 5. O que esta entrega NÃO fez

Não executou nenhum dos três scripts. Não gerou nenhuma máscara real.
Não implementou a função de desenho de contorno sobre a imagem original
(pendência registrada em `15_multiagent_experimental_matrix.md §6.4`,
necessária só na preparação de S1/S2, não nesta entrega). Não decidiu
nada sobre o VLM-direto, Qwen-VL, ou a CNN (stage 3, continua pausada,
pista separada).
