# Experimento de segmentação pós-qualificação (ISIC 2018 → PH²)

Código: `src/segmentation/` · CLI: `scripts/run_segmentation_experiment.py` ·
Config: seção `segmentation_experiment` de `config/config.yaml` · Testes: `tests/segmentation/`.

## Etapas e comandos

| Etapa | Comando | Artefatos principais (`output.root_dir`) |
|---|---|---|
| 1. Treino ISIC 2018 | `train-isic` | `checkpoints/isic2018/best.pt`, `results/isic_training/*` |
| 2. Folds do PH² | `make-folds` | `results/folds_manifest.{json,csv}` |
| 3. Transferência direta | `direct-transfer` | `results/direct_transfer/per_image.csv`, `predictions/direct_transfer/*.png` |
| 4. Fine-tuning (5 folds) | `finetune` | `results/finetune/*`, `predictions/finetune/*.png`, `checkpoints/finetune/fold_k.pt` |
| 5. Comparação pareada | `compare` | `results/comparison/{summary,selection,examples}.json`, `paired_differences.csv` |
| 6. Multiagente C0 × C1 | `multiagent [--run-vlm]` | `results/multiagent/*` |
| 7. Modelo final | `final-model` | `checkpoints/final/final_segmenter.pt`, `results/final_model/provenance.json` |
| Relatório | `report` | `reports/report.md`, `reports/figures/*` |
| Tudo | `all` | — |

Cada etapa grava um marcador com o fingerprint da configuração e dos insumos
(`results/_stages/`). Rodar de novo com artefatos compatíveis pula a etapa; com
configuração diferente é recusado, a menos que `--force` seja passado. O fine-tuning
retoma por fold.

## Decisões metodológicas

- **Rótulo do PH²**: `Clinical Diagnosis == 2` (40 melanomas). A regra legada
  (`"melanoma"` em *Histological Diagnosis*, em `src/data/dataset.py`) produz 28
  positivos porque a coluna histológica está vazia em 159/200 linhas. O manifesto de
  folds e o relatório registram a discrepância; `PH2Dataset` não foi alterado.
- **Transferência direta** é uma avaliação externa de um único modelo, não cinco treinos.
- **Fine-tuning**: cada rodada instancia uma estratégia nova e recarrega o checkpoint do
  ISIC; a validação interna (early stopping/checkpoint) é estratificada sobre os 4 folds
  de treino; o fold de teste só é lido para predição, depois do treino.
- **Limiar** fixo no config (`segmentation.threshold`), nunca ajustado com o PH².
- **Seleção**: métrica primária e regra declaradas no config (`higher_mean` |
  `mean_with_ci`). A seleção usa o próprio PH² e portanto é sujeita a viés de seleção.
- **Casos degenerados**: Dice/IoU = 0 com uma máscara vazia; todas indefinidas (NaN,
  contadas em `n_undefined`) com ambas vazias; HD95 indefinido com qualquer máscara vazia.
- **Multiagente**: C0 e C1 usam o mesmo grafo em `segmentation_mode="precomputed"`;
  só `seg_path` muda. As features são recalculadas por condição. Sem `--run-vlm` não há
  chamadas a VLMs. Métrica diagnóstica = parecer final do agente de síntese (mapeamento
  legado de `src/evaluation/evaluate.py`; ROC-AUC usa escore ordinal 0 / 0.5 / 1).
- **Modelo final**: `direct_transfer` preserva o checkpoint ISIC; `finetune` retreina nas
  200 imagens por nº fixo de épocas (mediana das melhores épocas do CV) e **não**
  substitui as predições fora da amostra.

## Dependências externas ao repositório

- `Image_Segmentation.code.network` (U-Net) — já exigido por `src/models/model.py`.
- Extração de features: `src.features.extractor.build_feature_package`, que está na
  branch `feature/feat_extraction`. Sem ela, a etapa 6 registra "features indisponíveis".

## Extensão

Nova arquitetura = subclasse de `SegmentationStrategy` (`build_model`, `block_groups`)
registrada com `@register_strategy`; selecione-a em `segmentation.strategy`.
Treino, avaliação e inferência não mudam.
