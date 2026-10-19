# Investigação PH² × HAM10000 — relatório gerado

> Exploratório, n pequeno, sem inferência causal. Gerado por `scripts/run_dataset_validation.py report`; interpretação em `docs/dataset_validation/`.

## 1. Estado da execução

| etapa | estado |
|---|---|
| auditoria/legado | executada |
| amostragem | executada (80 imagens) |
| qualidade | executada |
| inferência VLM | executada |
| avaliação | executada |

## 2. Amostra

- semente: **42**; n por classe: **20**; imagens: **80**; fingerprint do manifesto: `e577fe88ee6668cb…`

**ph2** — fonte de aquisição × classe: `{'pedro_hispano_hospital': {'melanoma': 20, 'nevus': 20}}`; tipo de confirmação × classe: `{}`; subtipo do nevo (PH²): `{'atypical': 10, 'common': 10}`

**ham10000** — fonte de aquisição × classe: `{'rosendahl': {'melanoma': 9}, 'vidir_modern': {'melanoma': 10, 'nevus': 3}, 'vidir_molemax': {'melanoma': 1, 'nevus': 16}, 'vienna_dias': {'nevus': 1}}`; tipo de confirmação × classe: `{'consensus': {'nevus': 1}, 'follow_up': {'nevus': 16}, 'histo': {'melanoma': 20, 'nevus': 3}}`; subtipo do nevo (PH²): `{}`

## 3. Evidências legadas do PH² (recuperadas dos arquivos; sem reconstrução)

| modelo | cenário | TN/FP/FN/TP | sens. | espec. |
|---|---|---|---|---|
| gpt-5.4-nano | thesis_estimated_value__legacy_labels | 17/8/8/17 | 0.68 [0.48–0.83] | 0.68 [0.48–0.83] |
| gpt-5.4-nano | thesis_estimated_value__clinical_labels | 16/7/9/18 | 0.67 [0.48–0.81] | 0.70 [0.49–0.84] |
| gpt-5.4-nano | programmatic_indet_positive__legacy_labels | 0/25/0/25 | 1.00 [0.87–1.00] | 0.00 [0.00–0.13] |
| gpt-5.4-nano | programmatic_indet_positive__clinical_labels | 0/23/0/27 | 1.00 [0.88–1.00] | 0.00 [0.00–0.14] |
| gpt-5.4-nano | programmatic_indet_negative__clinical_labels | 23/0/25/2 | 0.07 [0.02–0.23] | 1.00 [0.86–1.00] |
| gemini-3.1-flash-lite | thesis_estimated_value__legacy_labels | 21/4/8/17 | 0.68 [0.48–0.83] | 0.84 [0.65–0.94] |
| gemini-3.1-flash-lite | thesis_estimated_value__clinical_labels | 21/2/8/19 | 0.70 [0.52–0.84] | 0.91 [0.73–0.98] |
| gemini-3.1-flash-lite | programmatic_indet_positive__legacy_labels | 20/5/3/22 | 0.88 [0.70–0.96] | 0.80 [0.61–0.91] |
| gemini-3.1-flash-lite | programmatic_indet_positive__clinical_labels | 20/3/3/24 | 0.89 [0.72–0.96] | 0.87 [0.68–0.95] |
| gemini-3.1-flash-lite | programmatic_indet_negative__clinical_labels | 22/1/11/16 | 0.59 [0.41–0.75] | 0.96 [0.79–0.99] |

- gpt-5.4-nano: positivos pelo rótulo legado = 25, pelo diagnóstico clínico = 27; imagens com rótulo divergente: ['IMD061', 'IMD065']

- gemini-3.1-flash-lite: positivos pelo rótulo legado = 25, pelo diagnóstico clínico = 27; imagens com rótulo divergente: ['IMD061', 'IMD065']

## 4. Qualidade computacional das imagens

_Indicadores de sinal; não medem adequação clínica nem dificuldade diagnóstica._

### PH² × HAM10000 (todas as 40 + 40)

| indicador | grupo | ph2 mediana (IQR) | ham10000 mediana (IQR) | δ Cliff | p (MW) | p Holm | direção |
|---|---|---|---|---|---|---|---|
| megapixels | resolution | 0.442 (0.441–0.442) | 0.270 (0.270–0.270) | 1.00 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lesion_area_ratio | lesion_geometry | 0.464 (0.161–0.727) | 0.250 (0.139–0.373) | 0.34 | 0.008 | 0.062 | ph2 > ham10000 (médio) |
| lesion_equiv_diameter_ref_px | lesion_geometry | 341.324 (200.550–426.815) | 250.367 (186.197–305.641) | 0.35 | 0.008 | 0.062 | ph2 > ham10000 (médio) |
| lum_mean | exposure | 153.380 (127.436–164.944) | 152.319 (145.396–167.168) | -0.14 | 0.296 | 1.000 | sem diferença apreciável |
| lum_std | contrast | 38.096 (34.286–42.891) | 25.530 (21.708–30.821) | 0.73 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lum_dynamic_range | contrast | 162.705 (137.369–175.323) | 115.898 (103.656–134.417) | 0.68 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| contrast_lesion_skin_dE | contrast | 24.475 (18.548–28.840) | 22.295 (18.031–28.216) | 0.06 | 0.648 | 1.000 | sem diferença apreciável |
| lap_var_global_ref | sharpness | 461.733 (282.338–628.421) | 62.546 (43.148–135.246) | 0.93 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lap_var_lesion_ref | sharpness | 445.507 (333.581–710.961) | 116.649 (87.197–158.085) | 0.86 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| edge_strength_boundary_ref | sharpness | 5.097 (3.912–6.065) | 2.765 (2.404–3.481) | 0.73 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| noise_sigma_skin | noise | 2.236 (1.880–2.514) | 0.971 (0.864–1.138) | 0.93 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| vignetting_ratio | artifact | 0.635 (0.528–0.763) | 1.183 (1.001–1.353) | -0.77 | 0.000 | 0.000 | ham10000 > ph2 (grande) |
| hair_proxy_frac | artifact | 0.063 (0.038–0.086) | 0.009 (0.002–0.033) | 0.71 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| glare_frac_global | artifact | 0.000 (0.000–0.002) | 0.000 (0.000–0.000) | 0.71 | 0.000 | 0.000 | ph2 > ham10000 (grande) |

### PH² × HAM10000 — somente melanoma

| indicador | grupo | ph2 mediana (IQR) | ham10000 mediana (IQR) | δ Cliff | p (MW) | p Holm | direção |
|---|---|---|---|---|---|---|---|
| megapixels | resolution | 0.442 (0.442–0.442) | 0.270 (0.270–0.270) | 1.00 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lesion_area_ratio | lesion_geometry | 0.689 (0.460–0.818) | 0.299 (0.179–0.444) | 0.70 | 0.000 | 0.003 | ph2 > ham10000 (grande) |
| lesion_equiv_diameter_ref_px | lesion_geometry | 415.262 (339.415–452.567) | 273.761 (211.206–333.314) | 0.70 | 0.000 | 0.003 | ph2 > ham10000 (grande) |
| lum_mean | exposure | 133.772 (114.915–156.507) | 155.776 (146.243–163.271) | -0.54 | 0.004 | 0.040 | ham10000 > ph2 (grande) |
| lum_std | contrast | 35.669 (32.562–42.639) | 26.375 (22.688–30.684) | 0.70 | 0.000 | 0.003 | ph2 > ham10000 (grande) |
| lum_dynamic_range | contrast | 141.218 (126.401–168.840) | 119.667 (111.241–136.311) | 0.56 | 0.002 | 0.026 | ph2 > ham10000 (grande) |
| contrast_lesion_skin_dE | contrast | 18.633 (15.663–24.987) | 19.766 (12.764–27.695) | -0.08 | 0.684 | 1.000 | sem diferença apreciável |
| lap_var_global_ref | sharpness | 521.944 (428.062–966.240) | 66.428 (53.658–127.670) | 0.95 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lap_var_lesion_ref | sharpness | 561.939 (346.713–1,126) | 129.127 (96.105–202.571) | 0.83 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| edge_strength_boundary_ref | sharpness | 5.379 (4.336–5.921) | 3.016 (2.466–3.966) | 0.65 | 0.001 | 0.007 | ph2 > ham10000 (grande) |
| noise_sigma_skin | noise | 2.236 (1.942–2.503) | 0.985 (0.864–1.138) | 0.94 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| vignetting_ratio | artifact | 0.661 (0.608–0.791) | 1.152 (0.893–1.301) | -0.74 | 0.000 | 0.001 | ham10000 > ph2 (grande) |
| hair_proxy_frac | artifact | 0.072 (0.041–0.089) | 0.015 (0.006–0.034) | 0.68 | 0.000 | 0.004 | ph2 > ham10000 (grande) |
| glare_frac_global | artifact | 0.000 (0.000–0.002) | 0.000 (0.000–0.000) | 0.68 | 0.000 | 0.001 | ph2 > ham10000 (grande) |

### PH² × HAM10000 — somente nevus

| indicador | grupo | ph2 mediana (IQR) | ham10000 mediana (IQR) | δ Cliff | p (MW) | p Holm | direção |
|---|---|---|---|---|---|---|---|
| megapixels | resolution | 0.441 (0.439–0.442) | 0.270 (0.270–0.270) | 1.00 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lesion_area_ratio | lesion_geometry | 0.191 (0.114–0.465) | 0.235 (0.119–0.312) | 0.03 | 0.903 | 1.000 | sem diferença apreciável |
| lesion_equiv_diameter_ref_px | lesion_geometry | 217.237 (169.355–341.954) | 242.303 (172.799–279.241) | 0.03 | 0.903 | 1.000 | sem diferença apreciável |
| lum_mean | exposure | 163.704 (144.794–178.442) | 151.671 (141.865–167.426) | 0.26 | 0.172 | 1.000 | ph2 > ham10000 (pequeno) |
| lum_std | contrast | 39.997 (35.990–42.891) | 24.723 (20.677–30.891) | 0.70 | 0.000 | 0.003 | ph2 > ham10000 (grande) |
| lum_dynamic_range | contrast | 169.380 (160.517–176.486) | 108.436 (99.598–131.092) | 0.76 | 0.000 | 0.001 | ph2 > ham10000 (grande) |
| contrast_lesion_skin_dE | contrast | 27.808 (24.420–33.849) | 24.023 (19.535–32.010) | 0.29 | 0.114 | 1.000 | ph2 > ham10000 (pequeno) |
| lap_var_global_ref | sharpness | 330.096 (270.945–530.026) | 52.876 (35.974–135.246) | 0.92 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| lap_var_lesion_ref | sharpness | 377.026 (333.581–456.794) | 108.985 (45.540–133.279) | 0.91 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| edge_strength_boundary_ref | sharpness | 4.744 (3.665–6.192) | 2.681 (2.325–3.345) | 0.83 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| noise_sigma_skin | noise | 2.201 (1.806–2.503) | 0.948 (0.864–1.185) | 0.91 | 0.000 | 0.000 | ph2 > ham10000 (grande) |
| vignetting_ratio | artifact | 0.530 (0.442–0.725) | 1.243 (1.115–1.420) | -0.75 | 0.000 | 0.001 | ham10000 > ph2 (grande) |
| hair_proxy_frac | artifact | 0.057 (0.038–0.080) | 0.004 (0.001–0.024) | 0.76 | 0.000 | 0.001 | ph2 > ham10000 (grande) |
| glare_frac_global | artifact | 0.000 (0.000–0.002) | 0.000 (0.000–0.000) | 0.72 | 0.000 | 0.000 | ph2 > ham10000 (grande) |

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
