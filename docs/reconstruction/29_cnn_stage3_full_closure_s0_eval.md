# 29 — Fechamento completo da Etapa 3: checkpoint congelado + avaliação S0 real no HAM10000

Consolidação de resultados **já executados pelo pesquisador** (Mac/MPS,
2026-11-15) — esta sessão só leu os arquivos de saída reais retornados
(`run_cnn_s0_ham10000_eval.py`); nenhum script foi executado aqui
(`feedback_no_autonomous_execution`). Fecha as duas pendências deixadas
em `28_cnn_stage3_threshold_closure.md`.

## 1. Checkpoint congelado

- **SHA-256: `42c385a36845bdfeb33c308862973dc55b1d92c0a1779bdf3193ebd9e6ede0c3`**
  (época 19, `efficientnet_v2_s`, treinado no BCN20000) — identificador
  real e verificável a partir de agora; qualquer uso futuro deste
  checkpoint deve bater com este hash. Calculado pelo próprio script no
  momento da inferência (`s0_cnn_eval_results.json::summary.checkpoint_sha256`)
  e confirmado pelo pesquisador — valores idênticos.
- Limiar aplicado junto ao checkpoint: `0,08107660710811615` (fixado na
  validação interna do BCN20000, índice de Youden — ver doc 28).

## 2. Avaliação S0 real (`run_cnn_s0_ham10000_eval.py`, Mac/MPS)

- Manifesto verificado: SHA-256 bateu com o registro congelado
  (`32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`,
  `10_frozen_sample_registry.md`).
- **3.000/3.000 casos contabilizados**: OK=3.000, falhas=0. Nenhum caso
  descartado ou substituído. HAM10000 não usado para ajustar o limiar
  (`ham10000_used_to_tune_threshold: false`).
- Tempo total de inferência: 47,0s (3.000 imagens, MPS).
- Composição real do rótulo verdadeiro: 247 melanoma / 2.753
  não-melanoma (confere com a amostra congelada).
- Composição da predição: 404 melanoma / 2.596 não-melanoma.

**Matriz de confusão real** (recomputada nesta sessão a partir de
`s0_cnn_scores.csv`, não apenas lida do resumo):

| | Predito melanoma | Predito não-melanoma |
|---|---|---|
| **Real melanoma** (247) | TP = 126 | FN = 121 |
| **Real não-melanoma** (2.753) | FP = 278 | TN = 2.475 |

- **Sensibilidade (S0): 0,510121** (126/247)
- **Especificidade (S0): 0,899019** (2.475/2.753)

## 3. Comparação com a validação interna do BCN20000 (doc 28, não recomputada aqui)

| | Validação interna BCN20000 | S0 (HAM10000, transferência externa) |
|---|---|---|
| Sensibilidade | 0,71875 | 0,510121 |
| Especificidade | 0,8477922077922078 | 0,899019 |

Queda de sensibilidade real e esperada ao transferir o limiar fixado no
BCN20000 para o HAM10000 (domínios diferentes: equipamento,
iluminação, distribuição de diagnósticos) — registrado como fato
observado, não investigado mais a fundo aqui; não é um defeito do
pipeline (0 falhas, 0 casos descartados, limiar nunca reajustado com
HAM10000).

## 4. Status da Etapa 3 — FECHADA

Split congelado, treino real, seleção de checkpoint, limiar por Youden,
SHA-256 do checkpoint e avaliação S0 real: todos com resultado real,
verificado, não auto-relatado pelo assistente. `s0_cnn_scores.csv`
(score + decisão por lesão) está pronto para a comparação pareada
posterior com os VLMs/segmentação, assim que essa frente for retomada.
