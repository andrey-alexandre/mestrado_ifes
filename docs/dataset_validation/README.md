# Investigação PH² × HAM10000 — plano experimental

Branch: `exp/dataset_validation` · Código: `src/dataset_validation/` · CLI:
`scripts/run_dataset_validation.py` · Config: seção `dataset_validation` de
`config/config.yaml` · Testes: `tests/dataset_validation/` · Artefatos (não versionados):
`experiments/dataset_validation/`.

Documentos: [auditoria de equivalência](audit.md) ·
[risco metodológico do ISIC 2019](isic2019_risk.md) · [relatório e limitações](report.md).

## Escopo e restrições respeitadas

Experimento independente, reversível e exploratório. Não altera agentes, prompts, U-Net,
thresholds, nem os experimentos principais (PH²/ISIC 2019); não há Feature Engineering,
fine-tuning, download automático nem imagens/pesos no git (`experiments/` é ignorado).
Toda seleção aleatória usa semente fixa (42). Chamadas pagas só com `infer --run-vlm`.

## Hipóteses (não mutuamente excludentes)

| | Hipótese | O que o piloto consegue (e não consegue) mostrar |
|---|---|---|
| H1 | PH² tem condições de aquisição mais favoráveis (resolução, nitidez, contraste, iluminação, artefatos, área ocupada pela lesão) | Mede indicadores *computacionais*; não mede visibilidade clínica de estruturas |
| H2 | HAM10000 tem maior diversidade/dificuldade morfológica | Só há dx, tipo de confirmação, localização, idade/sexo; PH² tem anotações dermatoscópicas que o HAM não tem → comparação direta de dificuldade é **indeterminada** |
| H3 | Prevalência, seleção e definição dos grupos explicam parte da diferença | Controlável por desenho (20/20 por dataset, mesma regra de leitura) e por estratificação |

Diferenças arquiteturais **não** são hipótese (arquitetura declarada igual); verifica-se
apenas a equivalência de configuração/entrada/critério de avaliação (ver auditoria).

## Protocolo do piloto

- **Amostra (Etapa A)**: PH² 20 melanomas (`Clinical Diagnosis == 2`) + 20 nevos (`0` comum ou
  `1` atípico, mesma definição de "não-melanoma" da dissertação); HAM10000 20 `mel` + 20 `nv`.
  Sorteio `random.Random(f"{seed}:{dataset}:{classe}").sample(sorted(unidades), n)`; no HAM
  a unidade é `lesion_id` (uma imagem por lesão). Manifesto com id, dataset, classe, rótulo
  original, fonte do rótulo, caminho, semente, fluxo, tamanho do pool, ordem do sorteio e
  hashes SHA-256 de imagem e máscara.
- **Entrada**: mesma de PH² nos dois datasets — imagem original × máscara de referência
  (>128), resolução nativa (`apply_precomputed_mask`), só a imagem mascarada vai aos 4 agentes.
- **Qualidade (Etapa B)**: ~35 indicadores (resolução, geometria da lesão, exposição,
  contraste, nitidez nativa e em largura de referência de 512 px, ruído, proxies de artefato),
  comparados com Mann–Whitney bilateral, δ de Cliff e correção de Holm — no total e dentro de
  cada classe. Inspeção humana: `quality/visual_sample.png` (primeiros 6 do sorteio por grupo).
- **Inferência (Etapa C)**: pipeline original (`build_graph` + `run_pipeline`), 1 inferência por
  imagem, modelos `gpt-5.4-nano` e `gemini-3.1-flash-lite`, `temperature` do `config.yaml`.
- **Avaliação (Etapa D)**: matriz de confusão, sensibilidade, especificidade, precisão, F1,
  acurácia balanceada, acurácia; IC de Wilson (proporções) e bootstrap percentil estratificado
  (F1, acurácia balanceada; B=10000, seed 42). Diferença PH² − HAM: Newcombe + Fisher
  (exploratório). **ROC-AUC não é calculada**: saída categórica, sem escore contínuo comparável.
- **Interpretação das respostas** (mesma regra nos dois datasets; resposta original sempre
  preservada): (1) substrings idênticas a `src/evaluation/evaluate.py`; (2) só se não
  reconhecido, palavras-chave determinísticas (tier registrado); categorias
  `SUSPEITA/INDETERMINADA/BENIGNA/NAO_AVALIAVEL`. Política primária `indeterminate_as_positive`
  (repositório); sensibilidade `indeterminate_as_negative`; não avaliáveis excluídos e, em
  sensibilidade, contados como erro.

## Execução da inferência em outra máquina (autorizada em 2026-10-14)

`scripts/run_inference_pilot.py` — 2 modelos (`gpt-5.4-nano`, `gemini-3.1-flash-lite`) × 80 imagens
× 4 chamadas = 640 chamadas; sem reuso de inferências antigas. Reconstrói o manifesto a partir dos
dados da máquina e **aborta se o fingerprint diferir** do aprovado
(`e577fe88ee66…`); compara hashes com `pilot_manifest_reference.csv`; `--dry-run` valida tudo sem
custo; retoma sem repetir chamadas; gera `pilot_inference_outputs_*.zip`. Ao voltar, copiar
`inference/` para `experiments/dataset_validation/inference/` e rodar `evaluate` e `report`
(o pareamento usa `sample_id`, independente dos caminhos da máquina).

## Como reproduzir

```bash
python scripts/run_dataset_validation.py local      # audit + sample + quality + estimate + report (sem custo)
python scripts/run_dataset_validation.py estimate   # estimativa de chamadas/tokens
# somente após aprovar custo e definir OPENAI_API_KEY / GOOGLE_API_KEY:
python scripts/run_dataset_validation.py infer --run-vlm [--models gpt-5.4-nano]
python scripts/run_dataset_validation.py evaluate && python scripts/run_dataset_validation.py report
python -m pytest tests/dataset_validation -q
```

Artefatos por etapa: `audit/equivalence_audit.json`, `audit/legacy_ph2_reevaluation.json`,
`audit/legacy_ph2_rows.csv`; `manifest/manifest.csv`, `manifest/manifest_meta.json`;
`quality/{indicators.csv,comparison.json,composition.json,visual_sample.png,distributions.png}`;
`inference/{cost_estimate.json,results_<modelo>.json,results_<modelo>.meta.json,raw/}`;
`evaluation/{predictions_<modelo>.csv,metrics.json,hypothesis_analysis.json}`;
`reports/report.md`. A proveniência (commit, branch, dependências, semente) é gravada em
`manifest_meta.json`, `equivalence_audit.json` e `results_*.meta.json`.
