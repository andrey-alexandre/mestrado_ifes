# 28 — Fechamento parcial: treino real da CNN (BCN20000) e limiar fixado por Youden

Consolidação de resultados **já executados pelo pesquisador** (Mac/MPS,
2026-11-15) para o treino, mais uma computação determinística feita
nesta sessão diretamente sobre `val_scores.csv` real já retornado
(seleção de limiar — não é um treino, não é inferência, não há
auto-relato de "funcionou"; os números abaixo são reproduzíveis rodando
`scripts/select_bcn20000_threshold.py`). Checkpoint `.pt` permanece
apenas no Mac (`feedback_commit_for_mac_sync`: binário não vai para o
git; só o SHA-256 é registrado).

## 1. Divergência de hash do manifesto — resolvida e verificada

O `config.json` da execução real no Mac registrou
`manifest_sha256 = 66ccaead92e97b5690f1ab9c7a41c924444da5a238b46f62b54cd75d60675ab0`,
diferente do hash do mesmo arquivo neste repositório Windows
(`643203902ba21ab68e0db5dcaefc4201edb38c12efc9209cc2fe4e3057134351`).

**Verificado nesta sessão, não apenas aceito por afirmação**: removendo
`\r` (CRLF→LF) do `bcn20000_split_manifest.csv` local e recalculando o
SHA-256, o resultado é exatamente `66ccaead92e97b5690f1ab9c7a41c924444da5a238b46f62b54cd75d60675ab0`
— idêntico ao hash do Mac, mesma contagem de linhas (12.414, header +
12.413 linhas de dados). **Os dois arquivos têm exatamente os mesmos
registros; a única diferença é o final de linha (CRLF no Windows, LF no
Mac).** Não há mudança de conteúdo, não há violação da regra de split
congelado (seed `20261107`). `config.json` não foi alterado — o hash lá
gravado (`66ccaead...`) é o hash real do arquivo efetivamente usado no
treino, que é o que deve ficar registrado.

## 2. Treino real (`train_bcn20000_efficientnet.py`, Mac/MPS, seed 42)

- `epochs=20`, `batch_size=32`, `lr=0.0001`, `device=mps`,
  `n_train_images=9912`, `n_val_images=2501` — conforme protocolo
  aprovado (split por lesão, seed de dados `20261107`, `apenas_train`).
- 20/20 épocas completas, sem interrupção (`training_history.csv`).
- `val_auc` oscila entre 0,826 e 0,857 ao longo das épocas (ruído
  esperado em 2.501 imagens / 715 lesões de validação).
- **Checkpoint selecionado: época 19** (`val_auc = 0,8568916847041848`),
  pela regra pré-aprovada (maior AUC-ROC; empate → época mais cedo) —
  confere com o máximo real da coluna `val_auc` (`checkpoint_selection.json`).
- `val_scores.csv`: 2.501 linhas, 715 lesões distintas, 576
  melanoma / 1.925 não-melanoma — bate exatamente com a composição do
  split de validação congelado.

## 3. Limiar fixado por índice de Youden (`select_bcn20000_threshold.py`)

Calculado **somente** sobre `val_scores.csv` (validação interna do
BCN20000, 2.501 linhas, 576 melanoma / 1.925 não-melanoma). HAM10000
não foi lido nem referenciado nesta etapa.

- **Limiar: `0,08107660710811615`**
- **Sensibilidade: `0,71875`**
- **Especificidade: `0,8477922077922078`**
- **Índice de Youden (J): `0,5665422077922078`**
- Regra: maior J na curva ROC; empate → menor limiar (determinístico,
  documentado em `src/cnn_baseline/threshold_selection.py`).
- Saída: `experiments/cnn_baseline/training/threshold_selection.{json,md}`.
- 4/4 testes sintéticos novos passaram (`tests/cnn_baseline/test_threshold_selection.py`).

## 4. Pendência real — SHA-256 do checkpoint

`best_checkpoint.pt` (época 19) não existe nesta máquina Windows — fica
no Mac, por decisão já combinada (não commitar o binário). Falta rodar,
no Mac, e colar o resultado de volta:

```bash
shasum -a 256 experiments/cnn_baseline/training/best_checkpoint.pt
```

Sem esse hash, o checkpoint usado no treino real não tem identificador
verificável registrado (mesmo papel que o hash do segmentador em
`18_s2_final_model_closure.md §2`).

## 5. Próximo passo — avaliação S0 no HAM10000 congelado (script pronto, não executado)

`scripts/run_cnn_s0_ham10000_eval.py` (revisado por leitura nesta
sessão, não executado — `feedback_no_autonomous_execution`): aplica o
checkpoint de época 19 + o limiar fixado no passo 3 às 3.000 lesões do
manifesto congelado (`experiments/sampling/sample_manifest.csv`, SHA-256
verificado no início, igual ao padrão de `run_s2_mask_generation.py`),
usando as imagens ORIGINAIS do HAM10000 (não as máscaras de S2). Nunca
toca o rótulo verdadeiro para ajustar o limiar — só registra `dx`/
`binary_target` ao lado da predição, para a comparação pareada
posterior. Nenhum caso é descartado; falhas (ex.: imagem ausente) são
registradas, não substituídas.

Comando (Mac, substituir os caminhos reais):

```bash
python scripts/run_cnn_s0_ham10000_eval.py --device mps \
  --checkpoint-path experiments/cnn_baseline/training/best_checkpoint.pt \
  --ham10000-images-dir /Volumes/Andrey/Mestrado/dataverse_files/HAM10000
```

Saída esperada: `experiments/cnn_baseline/s0_ham10000_eval/s0_cnn_scores.csv`
(uma linha por lesão: `lesion_id, image_id, dx, true_binary_target,
predicted_score, predicted_binary_target, status, error`) +
`.json`/`.md` de resumo. Esta execução é independente da frente de VLMs
locais (pausada) e pode seguir em paralelo.

## 6. Status da Etapa 3

Fechado: split, treino real, seleção de checkpoint, limiar por Youden.
**Pendente, não bloqueante para o item 5**: SHA-256 do checkpoint
(precisa ser colado de volta do Mac). **Pendente, é o próximo passo
real**: executar `run_cnn_s0_ham10000_eval.py` no Mac e retornar
`s0_cnn_scores.csv` para a comparação pareada com os VLMs/segmentação.
