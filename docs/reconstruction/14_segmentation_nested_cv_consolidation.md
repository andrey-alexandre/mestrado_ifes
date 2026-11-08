# 14 — Consolidação pós-ensaio: política de inicialização e grade restrita antes da execução científica completa (Etapa 4)

Este documento consolida o que o ensaio de uma rodada externa (seção 2) comprovou de fato, e lista exatamente os pontos que ainda precisam de uma decisão explícita do pesquisador antes de iniciar a execução científica completa (5 rodadas externas × 4 internas, por arquitetura). Não decide nenhum desses pontos por conta própria — cada um está marcado como pendente, com uma recomendação e a razão por trás dela, para aprovação ou ajuste.

## 1. O que o ensaio comprovou (evidência real, não hipotética)

Execução real em escala completa, 2026-11-08, Mac/MPS, fold externo índice 0 (519 imagens), `experiments/segmentation/nested_cv_rehearsal_results.json` (committed, `616b98d`):

- **Executor correto de ponta a ponta**: 4 rodadas internas por arquitetura (treino real, seleção de melhor época por monitoramento), mediana das épocas, busca de limiar na grade 0,30–0,70 pelas predições internas na resolução original, retreino nos 4 folds de desenvolvimento, avaliação externa única — sem nenhum erro, nas duas arquiteturas.
- **Sem vazamento do fold externo**: por construção do código (`src/segmentation/nested_cv.py`), confirmado também pelos testes unitários (`tests/segmentation/test_nested_cv.py`).
- **Curvas de limiar sãs, sem degenerescência**: pico suave e unimodal na grade de 0,30–0,70 para as duas arquiteturas (U-Net em 0,50; DeepLabV3+ em 0,55) — nenhum sinal de máscara saturada/vazia nesta execução.
- **HD95 bem definido na quase totalidade dos casos** (3 de 519 indefinidos em cada arquitetura) — confirma que a correção de convenção (`border_value=0`, entrega anterior) está funcionando como esperado em escala real.
- **Resultados por arquitetura na avaliação externa** (519 imagens, 2 épocas de ensaio, limiar escolhido internamente):

| | U-Net | DeepLabV3+-ResNet34 |
|---|---|---|
| Época escolhida (mediana) | 2 | 2 |
| Limiar escolhido | 0,50 | 0,55 |
| Dice médio | 0,780 | 0,821 |
| IoU médio | 0,678 | 0,727 |
| Taxa de falha ISIC | 34,7% | 27,7% |
| HD95 mediana (px) | 195,4 | 106,5 |
| HD95 mediana (norm. diagonal) | 0,051 | 0,034 |
| Tempo por rodada interna | ~255–264s | ~232–243s |
| Tempo de retreino | 273,5s | 232,1s |
| Tempo de avaliação externa | 306,3s | 303,5s |

**Estes números NÃO decidem arquitetura nem configuração** — são de um ensaio de 2 épocas em 1 de 5 rodadas externas. Servem só para confirmar que o executor produz resultados plausíveis em dados reais, não números degenerados.

## 2. Achado metodológico importante: o teto de épocas do ensaio não revela o orçamento real

Nas 8 rodadas internas (4 por arquitetura), a **melhor época foi sempre a época 2** — ou seja, `--max-epochs 2` (valor de ensaio) foi sempre o teto artificial, nunca um pico interno real. Isso significa que o ensaio comprova que o executor funciona, mas **não dá nenhuma evidência sobre qual é o orçamento de épocas correto** para a execução científica — só sabemos que o modelo ainda estava melhorando quando o ensaio parou de treinar.

## 3. Pontos fechados (não reabertos aqui)

- Arquitetura DeepLabV3+-ResNet34 com `output_stride=32` (limitação confirmada do `BasicBlock` do torchvision) — fechado na entrega anterior.
- Resolução de entrada 256×256 para as duas arquiteturas — fechado desde a primeira entrega.
- Chave de agrupamento dos folds: `image:{image_id}`, sem `lesion_id` — limitação registrada, não há correção disponível.
- Convenção do HD95 (`border_value=0`) e diagnóstico de saturação (`pred_positive_pixels`/`ref_positive_pixels`) — fechado na entrega anterior.
- Regra de seleção de limiar (grade 0,30–0,70, empate por proximidade a 0,5, empate simétrico sinalizado) e de época (mediana das 4 melhores, arredondamento round-half-up) — implementadas e testadas, comprovadas no ensaio.
- `drop_last=True` nos DataLoaders de treino (corrige o crash de BatchNorm do ASPP com lote de tamanho 1) — corrigido e re-validado neste mesmo ensaio.

## 4. Pontos abertos — decisão do pesquisador antes da execução completa

### 4.1 Política de inicialização

U-Net não tem nenhum caminho de pré-treinamento aplicável (arquitetura encoder-decoder treinada do zero, sem backbone de classificação — já registrado em `src/segmentation/unet_model.py`). Isso significa que a única forma de manter a comparação entre arquiteturas livre do confundimento mencionado no escopo original ("não atribua diferenças exclusivamente à arquitetura se a inicialização... diferirem") é usar **inicialização aleatória nas duas arquiteturas** também na execução científica — a mesma política já usada no ensaio, agora proposta como definitiva, não mais "só do piloto".

**Recomendação**: manter inicialização aleatória para as duas arquiteturas na execução completa. Alternativa descartada: ResNet34 pré-treinado em ImageNet para o DeepLabV3+ — introduziria uma vantagem estrutural que não isola o efeito da arquitetura.

### 4.2 Orçamento real de épocas por rodada interna

O ensaio não revelou onde o Dice de monitoramento realmente estabiliza (seção 2). Antes de travar um valor para as 5×4 rodadas reais, sugiro um **segundo ensaio, mais barato**: rodar `--max-epochs` bem maior (ex. 15–20) em **apenas uma rodada interna** (não as 4×5 completas) para observar onde o Dice de monitoramento para de melhorar, e travar o orçamento real a partir disso. Custo estimado por época observado no ensaio: ~128s/época (U-Net), ~118s/época (DeepLabV3+) — 20 épocas em uma única rodada ≈ 40–43 minutos por arquitetura, ordem de grandeza bem menor que repetir isso nas 20 rodadas internas da execução completa (5×4) sem saber o teto certo.

### 4.3 Hiperparâmetros fixos (taxa de aprendizado, otimizador, tamanho de lote)

O ensaio usou `AdamW`, `lr=1e-4`, lote 4 (com `drop_last=True`), idênticos para as duas arquiteturas. O protocolo descrito menciona apenas busca de limiar (grade) e escolha de época (mediana) como decisões via dados internos — não há, até agora, nenhuma busca de taxa de aprendizado implementada.

**Pergunta direta**: a "grade restrita de configurações" que você mencionou inclui variar algo além de arquitetura × limiar × época (ex. taxa de aprendizado, otimizador)? Se sim, preciso saber o que exatamente entra nessa grade antes de estender o executor — hoje ele não busca nada além de limiar e época.

## 5. Decisões do pesquisador (2026-11-08, fecham a seção 4)

- **4.1 Inicialização**: aleatória nas duas arquiteturas na execução completa. Fechado.
- **4.3 Grade restrita**: só a arquitetura varia. Taxa de aprendizado (1e-4, AdamW), lote (4, `drop_last=True`) e demais hiperparâmetros ficam fixos e idênticos nas duas arquiteturas. Fechado.
- **4.2 Orçamento de épocas** — protocolo fixado ANTES de qualquer nova execução (para não deixar a regra ser ajustada depois de ver resultados, o que violaria o desenho aninhado):
  - **Teto de 20 épocas** por rodada interna e por retreino — limite operacional, não estimativa da época ótima.
  - **Early stopping**: interrompe a rodada após 5 épocas consecutivas sem melhora do Dice de monitoramento interno; salva o checkpoint de maior Dice observado (mesma regra de "melhor época" já implementada, agora com corte antecipado).
  - Cada rodada registra o **motivo da parada**: `paciencia_esgotada` (5 épocas sem melhora) ou `teto_atingido` (chegou a 20 sem disparar a paciência) — se `teto_atingido` e o Dice ainda estiver melhorando, isso deve ser registrado como "teto atingido", nunca como "convergência".
  - Antes de rodar as 5×4 rodadas completas: **uma rodada interna por arquitetura**, no Mac, com a configuração real prevista, só para validar a regra de parada e medir tempo por época/rodada — não para reajustar o teto ou a paciência depois de ver o resultado. O retreino de cada rodada externa continua usando a mediana das melhores épocas das 4 rodadas internas, sem early stopping (regra já aprovada, inalterada).
  - Cautela registrada pelo pesquisador: a sondagem de uma rodada não deve ser reaproveitada como se fosse uma das rodadas reais da validação aninhada — é descartável, só para viabilidade/tempo.

## 6. Implementação necessária a partir desta decisão

O executor atual (`src/segmentation/nested_cv.py` + `scripts/run_nested_cv_rehearsal.py`) treina até `--max-epochs` sem nenhuma lógica de parada antecipada. Para a seção 5 funcionar é preciso:
1. Early stopping real (paciência configurável) na rodada interna, com `stopped_reason` explícito no resultado.
2. Um modo de sondagem de rodada única (uma rodada interna, uma arquitetura por vez ou ambas, sem busca de limiar/retreino/avaliação externa) para a validação de tempo/regra pedida antes das 5×4 rodadas completas.

Status: implementação delegada, aguardando execução do pesquisador (ver RETORNO separado quando a entrega estiver pronta).
