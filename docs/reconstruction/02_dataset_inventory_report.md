# Inventário e Validação Estrutural dos Datasets — 2026-11-04

Inventário estrutural (sem amostragem, sem treino, sem chamadas a LLM/VLM, sem extração de zips grandes) dos 4 datasets do protocolo experimental: PH2, HAM10000, ISIC2018 Task1 e BCN20000.

## PH2

**Status: completo**

- Diretórios de lesão encontrados: 200
- Linhas de metadados (`PH2_dataset.txt`): 200

### Pareamento

- **Imagem × metadados**: 200 pareados, 0 só no primeiro conjunto, 0 só no segundo.
- **Máscara × metadados**: 200 pareados, 0 só no primeiro conjunto, 0 só no segundo.
- **ROI × metadados**: 27 pareados, 0 só no primeiro conjunto, 173 só no segundo.
- **Imagem × máscara**: 200 pareados, 0 só no primeiro conjunto, 0 só no segundo.

### Critérios de diagnóstico (lado a lado, sem decidir qual é a correta)

- `Histological Diagnosis` não vazio: 41 lesões.
- `Clinical Diagnosis == 2`: 40 lesões.

> **Nota de revisão (coordenação, 2026-11-04, verificada manualmente, não recalculada pelo script):**
> "não vazio" conta qualquer biópsia registrada, não apenas melanoma. Dos 41
> registros histológicos não vazios, os valores distintos são: `Melanoma`
> (26), `Nodular Melanoma` (2), `Dysplastic Nevus` (5), `Lentigo Maligna` (5),
> `Intradermal Nevus` (2), `Blue Nevus` (1). O número de melanoma confirmado
> por texto histológico é **28** (26+2), não 41 — os outros 13 são biópsias
> de lesões não melanoma. A comparação metodologicamente relevante é
> **28 (histológico/melanoma) × 40 (clínico==2)**, não 41×40. Esta nota não
> altera `src/data_audit/adapters.py`; decidido manter o código como entregue
> e apenas registrar esta ressalva (decisão do pesquisador, 2026-11-04).

## HAM10000

**Status: completo**

- Linhas de metadados: 10015
- Arquivos de imagem (`.jpg`) encontrados: 10015
- Arquivos de máscara (`.png`) encontrados: 10015
- Lesões distintas (`lesion_id`): 7470

### Pareamento (3 direções)

- **Metadados × imagens**: 10015 pareados, 0 só no primeiro conjunto, 0 só no segundo.
- **Metadados × máscaras**: 10015 pareados, 0 só no primeiro conjunto, 0 só no segundo.
- **Imagens × máscaras**: 10015 pareados, 0 só no primeiro conjunto, 0 só no segundo.

### `lesion_id` → `dx`

- Linhas sem `lesion_id`: 0
- `lesion_id` com `dx` conflitante entre linhas: 0

### Distribuição por `dx`

- `akiec`: 327
- `bcc`: 514
- `bkl`: 1099
- `df`: 115
- `mel`: 1113
- `nv`: 6705
- `vasc`: 142

### Distribuição de imagens por lesão (nº de imagens → nº de lesões com esse nº)

- 1 imagem(ns): 5514 lesões
- 2 imagem(ns): 1423 lesões
- 3 imagem(ns): 490 lesões
- 4 imagem(ns): 34 lesões
- 5 imagem(ns): 5 lesões
- 6 imagem(ns): 4 lesões

> Nota: hashing sha256 de ~20 mil arquivos (imagens + máscaras) foi executado nesta rodada; pode levar minutos — é esperado, não é falha.

## ISIC2018 Task1

**Status: parcial**

- Entradas totais no zip de imagens: 2597
- Entradas `.jpg` (imagens): 2594
- Outras entradas no zip:
  - `ISIC2018_Task1-2_Training_Input/` (0 bytes)
  - `ISIC2018_Task1-2_Training_Input/ATTRIBUTION.txt` (426 bytes)
  - `ISIC2018_Task1-2_Training_Input/LICENSE.txt` (7048 bytes)
- Máscaras de referência (`C:\Projetos\IFES\mestrado_ifes\data\ISIC2018_Task1_Training_GroundTruth`): not found

> Imagens disponíveis (zip não extraído); máscaras ausentes — pareamento do conjunto de desenvolvimento da segmentação não pode ser validado.

## BCN20000

**Status: parcial**

### Entradas do zip externo (não extraídas)

- `BCN_20k_test.zip` (3327235230 bytes)
- `BCN_20k_train.zip` (6368901290 bytes)
- `bcn_20k_test.csv` (446244 bytes)
- `bcn_20k_train.csv` (945681 bytes)

- Linhas no CSV de treino: 12413 (lesões distintas: 3576)
- Linhas no CSV de teste: 6533 (lesões distintas: 1864)

### `lesion_id` → `diagnosis` (treino)

- Linhas sem `lesion_id`: 0
- `lesion_id` com `diagnosis` conflitante entre linhas: 0

### Distribuição por `diagnosis` (treino)

- `AK`: 737
- `BCC`: 2809
- `BKL`: 1138
- `DF`: 124
- `MEL`: 2857
- `NV`: 4206
- `SCC`: 431
- `VASC`: 111

- `diagnosis` não vazio no CSV de teste: 0/6533

### Observações estruturais

- diagnosis vazio em 6533/6533 linhas de teste (rótulo retido na versão pública); não é dado ausente por erro, é retenção intencional.
- Formato de lesion_id difere entre treino (ex.: 'BCN_0003884', com underscore e zero-padding) e teste (ex.: 'BCN4657', sem underscore/padding) — mesma fonte, formatos diferentes. Apenas documentado; não normalizado nem presumido equivalente a identificadores de outros datasets.
- Imagens aninhadas em 'BCN_20k_train.zip'/'BCN_20k_test.zip' (dentro do zip externo) não foram listadas nem hasheadas nesta tarefa — inventário em nível de arquivo de imagem fica para quando o treino da CNN for implementado.
