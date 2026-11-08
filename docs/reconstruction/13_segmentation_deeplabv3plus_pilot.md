# 13 — DeepLabV3+(ResNet34), métricas aprovadas e piloto comparativo de arquiteturas (Etapa 4)

Sessão de implementação estática (2026-11-08) — não executada pelo assistente/fork
que escreveu este documento (restrição explícita do pesquisador: nenhum
`.py`, teste ou instalação roda nesta sessão). O pesquisador executa tudo
no Mac depois.

Continuação direta de [`09_sampling_infrastructure_report.md`] e do piloto
U-Net já validado (ver `project_segmentation_stage4` na memória do
coordenador): U-Net continua intocada (`src/segmentation/unet_model.py`,
`src/segmentation/folds.py`, `scripts/run_isic2018_unet_pilot.py` —
nenhum destes três arquivos foi modificado nesta entrega).

## 1. Ambiente confirmado

- `torch==2.14.0`, `torchvision==0.29.0` instalados
  (`.venv/Lib/site-packages/torch-2.14.0.dist-info`,
  `.venv/Lib/site-packages/torchvision-0.29.0.dist-info`) — confirmado por
  leitura direta dos metadados do ambiente, não assumido.
- `segmentation_models_pytorch` NÃO está instalado (confirmado por busca
  no `site-packages`) — não pode ser usado nesta entrega (exigiria
  instalação, fora do que esta sessão pode fazer). A DeepLabV3+ foi
  implementada diretamente em `torch`/`torchvision`, do mesmo jeito que a
  U-Net foi implementada standalone na entrega anterior (mesma motivação:
  remover uma dependência externa não instalável agora).

## 2. Achado confirmado: `BasicBlock` do torchvision não aceita `dilation > 1`

Antes de decidir o `output_stride`, o código-fonte do torchvision
instalado foi lido diretamente (não assumido de memória/documentação):

`.venv/Lib/site-packages/torchvision/models/resnet.py:78-79`:
```python
if dilation > 1:
    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
```

`BasicBlock` é o bloco usado por `resnet18`/`resnet34`; `Bottleneck` (usado
por `resnet50`/`101`/`152`) não tem essa restrição — é por isso que
`torchvision.models.segmentation.deeplabv3_resnet50` consegue usar
`replace_stride_with_dilation` para obter `output_stride=16`, e o
ResNet34 não consegue.

**Decisão adotada**: `output_stride=32` para o encoder ResNet34 — forward
padrão, sem nenhuma stage dilatada
(`src/segmentation/deeplabv3plus_model.py:142-148`). Isto é a resposta
correta a uma limitação confirmada da biblioteca, não uma substituição de
arquitetura/encoder: ResNet34 + DeepLabV3+ continua sendo a combinação
implementada, exatamente como aprovado. Nenhuma troca para ResNet50/101
foi feita.

Taxas do ASPP ajustadas de (6, 12, 18) — valores do artigo original para
`output_stride=16` — para **(3, 6, 9)**
(`src/segmentation/deeplabv3plus_model.py:61`), uma redução proporcional
pela metade (mesma proporção da mudança de stride 16→32), para manter o
campo receptivo das convoluções atrous compatível com o mapa de features
menor (8x8 para entrada 256x256 em stride 32, vs. 16x16 em stride 16).

## 3. Arquitetura DeepLabV3+-ResNet34 implementada

`src/segmentation/deeplabv3plus_model.py`:

- **Encoder** (`DeepLabV3PlusResNet34.__init__`, linhas 130-148):
  `torchvision.models.resnet34(weights=None)` — inicialização aleatória,
  sem download (ver seção 4). Features de baixo nível capturadas após
  `layer1` (stride 4, **64 canais** — `LOW_LEVEL_IN_CHANNELS`, linha 64);
  features de alto nível após `layer4` (stride 32, **512 canais** —
  `HIGH_LEVEL_IN_CHANNELS`, linha 65; não 2048, contagem que só se
  aplicaria a ResNet50/101 via `Bottleneck`, `expansion=4`).
- **ASPP** (`ASPP`, linhas 99-122): 1 branch 1x1 + 3 branches atrous
  (rates 3, 6, 9) + 1 branch de pooling global
  (`_ASPPPoolingBranch`, linhas 82-96), concatenados (5×256=1280 canais) e
  projetados para 256 canais via 1x1 conv + BN + ReLU + dropout
  (linhas 112-118).
- **Decoder** (linhas 152-169, `forward` linhas 171-190): projeta as
  features de baixo nível para 48 canais (`LOW_LEVEL_PROJECT_CHANNELS`,
  linha 63, via 1x1 conv — igual ao artigo original), faz upsample
  bilinear da saída do ASPP até o tamanho espacial das features de baixo
  nível (linha 182), concatena (304 canais), duas convoluções 3x3
  (256→256, com dropout) e uma 1x1 conv final para `output_ch` canais
  (linha 169), seguida de upsample bilinear até o tamanho de entrada
  original (linha 189).
- `build_deeplabv3plus(img_ch=3, output_ch=1)` (linhas 193-196): retorna
  logits brutos — mesma convenção de `build_unet` (sigmoid fica de fora,
  nunca dentro do `forward`).

## 4. Política de inicialização do piloto

Ambas as arquiteturas usam **inicialização aleatória** neste piloto:
U-Net já era assim (sem pesos pré-treinados, arquitetura treinada do
zero); DeepLabV3+ usa `resnet34(weights=None)` — nenhum peso ImageNet é
baixado (`src/segmentation/deeplabv3plus_model.py:9-14,132`). Isto é
necessário nesta sessão (sem acesso à rede) e é também a política adotada
*para o piloto*, registrada aqui explicitamente conforme pedido — **não
decide a política de pré-treinamento do experimento definitivo**. Essa
decisão (ImageNet-pretrained vs. do-zero para o encoder do DeepLabV3+)
continua aberta para quando o treino real for planejado.

## 5. Interface comum / factory

`src/segmentation/model_factory.py` — `build_segmentation_model(architecture, img_ch, output_ch)`
roteia para `build_unet` ou `build_deeplabv3plus` sem duplicar lógica;
`ValueError` claro para nome desconhecido.

## 6. Métricas aprovadas (`src/segmentation/metrics.py`)

Resumo da implementação (ver docstrings do próprio módulo para a
especificação completa linha a linha):

- **Dice por imagem** (`dice_per_image`) — métrica primária, nunca
  agregada globalmente por pixel entre imagens.
- **IoU/Jaccard** (`iou_jaccard`) — complementar.
- **Taxa de falha ISIC** (`is_isic_failure`) — `iou < 0.65`.
- **Thresholded Jaccard** (`thresholded_jaccard`) — `0.0` se
  `iou < 0.65` (corte estrito), senão preserva o valor do IoU
  (`iou == 0.65` preserva).
- **HD95 simétrico** (`hd95_symmetric`) — contorno via erosão binária
  (`scipy.ndimage.binary_erosion`, `border_value=1` — escolha deliberada
  documentada no código: uma máscara totalmente preenchida fica sem
  contorno, pois os pixels fora da imagem são tratados como tendo o mesmo
  valor da borda da máscara); distâncias direcionais via
  `scipy.ndimage.distance_transform_edt`; HD95 = máximo dos dois
  percentis 95 (A→B e B→A). Retorna `None` (indefinido) se qualquer uma
  das máscaras não tiver pixels de contorno — nunca convertido para
  `0.0`. `pixel_spacing` multiplica o resultado final (variante em
  unidade física).
- **Variante normalizada pela diagonal** (correção 2026-11-08): calculada
  em `compute_image_metrics` (não em `hd95_symmetric`), como
  `hd95_normalizado_diagonal = hd95 / sqrt(h**2 + w**2)`, com `h, w` da
  resolução original da máscara de referência. `None` sempre que `hd95`
  for `None` (mesma regra de nunca converter indefinido em zero), e
  `None` em todos os status de erro (`referencia_invalida`,
  `mascara_incompativel`, `falha_processamento`), igual aos demais campos
  de métrica. Esta era uma omissão da entrega anterior (a variante estava
  documentada como "a cargo de quem chama", mas nenhum chamador a
  calculava) — agora é computada e registrada em todos os pontos: no
  dict por imagem, na agregação e no relatório do piloto (seção 8).
- **Orquestração** (`compute_image_metrics`): recebe o mapa de
  probabilidade JÁ restaurado à resolução original (ver seção 7) e a
  máscara de referência na resolução original; binariza só depois da
  restauração. Status explícitos, nunca omitidos ou convertidos
  silenciosamente: `"ok"` (inclui o caso predição vazia + referência
  válida: Dice/IoU=0.0, HD95=`None`, `hd95_normalizado_diagonal=None`,
  `is_isic_failure=True` — uma métrica válida com valor zero, não uma
  falha), `"referencia_invalida"` (referência sem nenhum pixel de lesão —
  todas as métricas `None`, não tratado como "predição vazia correta"),
  `"mascara_incompativel"` (formas diferentes), `"falha_processamento"`
  (exceção inesperada, capturada e registrada, nunca propagada
  silenciosamente).
- **Agregação** (`aggregate_metrics`): contagem por status; para os
  `"ok"`: Dice médio, IoU médio, taxa de falha ISIC, Thresholded Jaccard
  médio, e para HD95 — mediana dos valores finitos **separada** da
  contagem de indefinidos (nunca entram juntos na mediana, nunca tratados
  como zero) — e a mesma separação para `hd95_normalizado_diagonal`
  (`hd95_normalizado_diagonal_median_finite`, reaproveitando a contagem de
  indefinidos de HD95, já que um HD95 indefinido implica o normalizado
  também indefinido).

## 7. Restauração à resolução original (`src/segmentation/isic2018_dataset.py`)

`load_original_mask(masks_zip, image_id)` (nova função, inserida após
`_resize_mask_nearest`) lê a máscara de referência na resolução NATIVA do
arquivo — sem nenhum resize, nunca interpolação contínua. O comportamento
de treino existente (`ISIC2018Task1Dataset.__getitem__`, que devolve o
par 256×256 para treino) não foi alterado.

A restauração do mapa de probabilidade (contínuo, saída do sigmoid em
256×256) para a resolução da máscara de referência usa
`torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`
(`src/segmentation/metrics.py`, `restore_probability_to_original_resolution`)
— só então o resultado é binarizado em `compute_image_metrics`. A ordem
restaurar→binarizar é verificada por teste controlado (ver seção 9).

## 8. Piloto (`scripts/run_segmentation_arch_pilot.py`)

Script novo, não modifica `scripts/run_isic2018_unet_pilot.py` (reaproveita
`_ensure_folds` dele via import). Reaproveita o arquivo de folds já
congelado (`experiments/segmentation/isic2018_task1_folds.json`, seed
`20261108`) — **chave de agrupamento confirmada**: `image:{image_id}`
(sem `lesion_id` — ISIC2018 Task1 não tem metadado de lesão, limitação já
documentada em `src/segmentation/folds.py:5-18`); este piloto não
regenera o arquivo.

- `--device {cpu,cuda,mps}`: seleção explícita, sem fallback silencioso —
  se o dispositivo pedido não estiver disponível, levanta `RuntimeError`
  nomeando o que foi pedido e o que está disponível
  (`_resolve_device`, linhas ~62-80).
- Fold externo (`--external-fold`, default 0) nunca é usado em nenhuma
  amostra deste piloto. Dos 4 folds de desenvolvimento restantes, o
  último (por índice) é usado como "validação interna" (só para a
  demonstração de métricas); os outros 3 são "treino piloto" — divisão
  estrutural simples, não é a validação cruzada aninhada real (não
  implementada aqui).
- 4 checks mecânicos por arquitetura (leitura/alinhamento com IDs e
  dimensões rastreáveis, atualização de peso real após 1 passo,
  salvar/recarregar checkpoint, forma da máscara predita) + cálculo de
  métricas de demonstração (`--demo-threshold`, default 0.5 — threshold
  de demonstração, não o limiar escolhido por validação) sobre a amostra
  de validação interna, restaurando a probabilidade antes de binarizar.
- `--run-timing-epoch` (opcional): mede o tempo de 1 época sobre a
  amostra de piloto (não o dataset inteiro) — só medição de tempo, não
  treino real.
- Relatório em `experiments/segmentation/segmentation_arch_pilot_{report.md,results.json,log}`.

## 9. Testes controlados (`tests/segmentation/`)

- `test_metrics.py`: máscara perfeita (Dice=IoU=1.0, HD95=0.0,
  `hd95_normalizado_diagonal` próximo de 0.0), predição vazia + referência
  válida (Dice=IoU=0.0, HD95=`None`, `hd95_normalizado_diagonal=None`,
  status `"ok"`), referência inválida (status `"referencia_invalida"`,
  tudo `None` incluindo `hd95_normalizado_diagonal`), corte do
  Thresholded Jaccard (abaixo/igual/acima de 0.65), restauração antes de
  binarizar (prova que o resultado difere de
  binarizar-antes-de-redimensionar), `aggregate_metrics` separando status
  e HD95 (e seu normalizado) indefinido.
- `test_model_factory.py`: roteamento correto, `ValueError` para nome
  desconhecido.
- `test_deeplabv3plus_model.py`: forward `(1,3,256,256)→(1,1,256,256)`;
  confirma 512 canais em `layer4` (não 2048); confirma que o modelo
  retorna logits brutos, não probabilidades.

## 10. O que esta entrega NÃO implementa (protocolo científico futuro)

Descrito e entendido, para não ser esquecido, mas **não implementado
aqui**: validação cruzada real com 5 folds externos × 4 internos por
rodada externa; hiperparâmetros/checkpoint/limiar escolhidos só nos dados
internos; retreino nos 4 folds de desenvolvimento usando a mediana das
melhores épocas internas, sem early stopping no fold externo; grade de
limiar 0,30–0,70 em passos de 0,05 (empate: valor mais próximo de 0,5,
sinalizando empate simétrico residual); seleção de arquitetura pelo Dice
médio por imagem nos folds externos, empate prático se diferença absoluta
< 0,005 (desempate: menor taxa de falha ISIC, menor HD95 mediano, depois
U-Net); modelo final reinicializado e treinado com todos os dados
elegíveis do ISIC2018 Task1 após a seleção.

## 11. Fora de escopo / preservado

- `src/segmentation/unet_model.py`, `src/segmentation/folds.py`,
  `scripts/run_isic2018_unet_pilot.py` e o arquivo de folds congelado:
  intocados.
- HAM10000: não lido nem referenciado em nenhum arquivo novo desta
  entrega.
- Etapa 3 (CNN) e etapa 5 (multiagente): não tocadas.
