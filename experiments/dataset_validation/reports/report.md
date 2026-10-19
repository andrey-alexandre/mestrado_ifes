# Investigação PH² × HAM10000 — relatório gerado

> Exploratório, n pequeno, sem inferência causal. Gerado por `scripts/run_dataset_validation.py report`; interpretação em `docs/dataset_validation/`.

## 1. Estado da execução

| etapa | estado |
|---|---|
| auditoria/legado | não gerada nesta pasta (rode a etapa correspondente) |
| amostragem | executada (80 imagens) |
| qualidade | não gerada nesta pasta (rode a etapa correspondente) |
| inferência VLM | executada |
| avaliação | executada |

## 2. Amostra

- semente: **42**; n por classe: **20**; imagens: **80**; fingerprint do manifesto: `e577fe88ee6668cb…`

**ph2** — fonte de aquisição × classe: `{'pedro_hispano_hospital': {'melanoma': 20, 'nevus': 20}}`; tipo de confirmação × classe: `{}`; subtipo do nevo (PH²): `{'atypical': 10, 'common': 10}`

**ham10000** — fonte de aquisição × classe: `{'rosendahl': {'melanoma': 9}, 'vidir_modern': {'melanoma': 10, 'nevus': 3}, 'vidir_molemax': {'melanoma': 1, 'nevus': 16}, 'vienna_dias': {'nevus': 1}}`; tipo de confirmação × classe: `{'consensus': {'nevus': 1}, 'follow_up': {'nevus': 16}, 'histo': {'melanoma': 20, 'nevus': 3}}`; subtipo do nevo (PH²): `{}`

## 3. Evidências legadas do PH² (recuperadas dos arquivos; sem reconstrução)

_Não gerado nesta pasta: rode `audit`._
| modelo | cenário | TN/FP/FN/TP | sens. | espec. |
|---|---|---|---|---|

## 4. Qualidade computacional das imagens

_Indicadores de sinal; não medem adequação clínica nem dificuldade diagnóstica._

_Não gerado nesta pasta (exige as imagens): rode `quality`._

## 5. Inferência multiagente e avaliação


### gpt-5.4-nano — política primária `indeterminate_as_positive`

| dataset | TN/FP/FN/TP | n aval. | sens. | espec. | prec. | F1 | acc. bal. | acc. |
|---|---|---|---|---|---|---|---|---|
| ph2 | 2/18/1/19 | 40 | 0.95 [0.76–0.99] | 0.10 [0.03–0.30] | 0.51 [0.36–0.67] | 0.67 [0.61–0.71] | 0.53 [0.45–0.60] | 0.53 [0.37–0.67] |
| ham10000 | 1/17/2/18 | 38 | 0.90 [0.70–0.97] | 0.06 [0.01–0.26] | 0.51 [0.36–0.67] | 0.65 [0.58–0.71] | 0.48 [0.40–0.56] | 0.50 [0.35–0.65] |

Sensibilidade (política `indeterminate_as_negative`):

| dataset | TN/FP/FN/TP | n aval. | sens. | espec. | prec. | F1 | acc. bal. | acc. |
|---|---|---|---|---|---|---|---|---|
| ph2 | 17/3/18/2 | 40 | 0.10 [0.03–0.30] | 0.85 [0.64–0.95] | 0.40 [0.12–0.77] | 0.16 [0.00–0.37] | 0.47 [0.38–0.57] | 0.47 [0.33–0.63] |
| ham10000 | 18/0/19/1 | 38 | 0.05 [0.01–0.24] | 1.00 [0.82–1.00] | 1.00 [0.21–1.00] | 0.10 [0.00–0.26] | 0.53 [0.50–0.57] | 0.50 [0.35–0.65] |

| métrica | PH² | HAM10000 | diferença [IC Newcombe] | p Fisher |
|---|---|---|---|---|
| sensitivity | 0.95 | 0.90 | 0.05 [-0.15–0.26] | 1.000 |
| specificity | 0.10 | 0.06 | 0.04 [-0.17–0.25] | 1.000 |

### gemini-3.1-flash-lite — política primária `indeterminate_as_positive`

| dataset | TN/FP/FN/TP | n aval. | sens. | espec. | prec. | F1 | acc. bal. | acc. |
|---|---|---|---|---|---|---|---|---|
| ph2 | 19/1/3/17 | 40 | 0.85 [0.64–0.95] | 0.95 [0.76–0.99] | 0.94 [0.74–0.99] | 0.89 [0.78–0.98] | 0.90 [0.80–0.97] | 0.90 [0.77–0.96] |
| ham10000 | 16/4/2/18 | 40 | 0.90 [0.70–0.97] | 0.80 [0.58–0.92] | 0.82 [0.61–0.93] | 0.86 [0.75–0.95] | 0.85 [0.73–0.95] | 0.85 [0.71–0.93] |

Sensibilidade (política `indeterminate_as_negative`):

| dataset | TN/FP/FN/TP | n aval. | sens. | espec. | prec. | F1 | acc. bal. | acc. |
|---|---|---|---|---|---|---|---|---|
| ph2 | 20/0/10/10 | 40 | 0.50 [0.30–0.70] | 1.00 [0.84–1.00] | 1.00 [0.72–1.00] | 0.67 [0.46–0.82] | 0.75 [0.65–0.85] | 0.75 [0.60–0.86] |
| ham10000 | 20/0/13/7 | 40 | 0.35 [0.18–0.57] | 1.00 [0.84–1.00] | 1.00 [0.65–1.00] | 0.52 [0.26–0.71] | 0.68 [0.57–0.78] | 0.68 [0.52–0.80] |

| métrica | PH² | HAM10000 | diferença [IC Newcombe] | p Fisher |
|---|---|---|---|---|
| sensitivity | 0.85 | 0.90 | -0.05 [-0.27–0.17] | 1.000 |
| specificity | 0.95 | 0.80 | 0.15 [-0.07–0.37] | 0.342 |
