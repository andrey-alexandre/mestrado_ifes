# Piloto Complementar — pHash 64 vs. 256 bits + ORB Independente do pHash — 2026-11-05

Extensão da Tarefa 8. A Tarefa 8 (pHash com bloco DCT 16x16/256 bits) obteve ZERO sobreviventes nos 2.000 pares amostrados, incluindo os 400 pares difíceis reais conhecidos — resultado consistente tanto com "não há duplicata real na amostra" quanto com "o pHash de 256 bits descarta positivos reais antes da etapa geométrica", e aquele piloto não conseguiu distinguir as duas hipóteses. Esta tarefa compara diretamente pHash de 64 bits (bloco DCT 8x8) contra 256 bits no mesmo protocolo, e quebra a dependência geométrica rodando ORB+RANSAC diretamente nos candidatos reais, independente de qualquer filtro de pHash.

**Este é um PILOTO CURTO E DELIMITADO**: reaproveita exatamente a amostra de 2.000 pares da Tarefa 8 (mesma semente), não roda a busca completa nos 421.196 candidatos, e não testa nenhum tamanho de bloco DCT além de 8x8 e 16x16. Os artefatos da Tarefa 8 (`visual_duplicate_pilot_index.json`, `05_visual_duplicate_pilot_report.md`) permanecem intactos. **Isto é piloto/detecção, não confirmação: nenhuma exclusão de lesão/imagem foi decidida ou implementada em nenhuma etapa.**

## 1. Comparação pHash 64 vs. 256 bits no mesmo protocolo real

Amostra de imagens reais para variantes controladas (semente=42): 30 imagens (HAM10000=10, ISIC2018 Task1=10, BCN20000=10). Ampliada em relação à Tarefa 8 (que usou 8 imagens, todas do HAM10000) para cobrir as três fontes que efetivamente compõem os pares candidatos da Tarefa 7.

### Distância pHash-64 por variante controlada (normalizada = distância ÷ 64)

| Variante | n | min | média | max | min/norm | média/norm | max/norm |
|---|---|---|---|---|---|---|---|
| Cópia idêntica | 30 | 0 | 0.0 | 0 | 0.000 | 0.000 | 0.000 |
| Recompressão JPEG (qualidade 40) | 30 | 0 | 0.4 | 4 | 0.000 | 0.006 | 0.062 |
| Redimensionamento (50%) | 30 | 0 | 0.4 | 2 | 0.000 | 0.006 | 0.031 |
| Recorte pequeno (5% da borda) | 30 | 4 | 11.4 | 20 | 0.062 | 0.178 | 0.312 |
| Rotação 90° | 30 | 20 | 30.1 | 36 | 0.312 | 0.470 | 0.562 |
| Espelhamento horizontal | 30 | 26 | 30.7 | 36 | 0.406 | 0.480 | 0.562 |
| Recorte significativo (30% da borda) | 30 | 24 | 31.0 | 40 | 0.375 | 0.484 | 0.625 |

### Distância pHash-256 por variante controlada (normalizada = distância ÷ 256)

| Variante | n | min | média | max | min/norm | média/norm | max/norm |
|---|---|---|---|---|---|---|---|
| Cópia idêntica | 30 | 0 | 0.0 | 0 | 0.000 | 0.000 | 0.000 |
| Recompressão JPEG (qualidade 40) | 30 | 0 | 3.7 | 12 | 0.000 | 0.014 | 0.047 |
| Redimensionamento (50%) | 30 | 0 | 2.5 | 10 | 0.000 | 0.010 | 0.039 |
| Recorte pequeno (5% da borda) | 30 | 48 | 76.2 | 118 | 0.188 | 0.298 | 0.461 |
| Rotação 90° | 30 | 104 | 124.1 | 142 | 0.406 | 0.485 | 0.555 |
| Espelhamento horizontal | 30 | 108 | 125.3 | 134 | 0.422 | 0.489 | 0.523 |
| Recorte significativo (30% da borda) | 30 | 108 | 126.1 | 144 | 0.422 | 0.493 | 0.562 |

### Pares difíceis reais (candidatos dHash distância 0–2 da Tarefa 7, n=400)

- **pHash-64**: n=400, min=6 (norm=0.094), mediana=24.0, média=23.6, max=42 (norm=0.656)
- **pHash-256**: n=400, min=58 (norm=0.227), mediana=112.0, média=112.7, max=152 (norm=0.594)

### Limiar derivado — pHash-64

Separação real observada: variantes limpas ['identical_copy', 'jpeg_recompress_q40', 'resize_50pct'] com máximo 4 contra mínimo real dos pares difíceis 6 — gap de 2 bits. Variantes não separadas: ['crop_5pct_border', 'rotate_90', 'mirror_horizontal', 'crop_30pct_border'].
**Limiar escolhido: distância ≤ 5** (de 64 bits).

### Limiar derivado — pHash-256

Separação real observada: variantes limpas ['identical_copy', 'jpeg_recompress_q40', 'resize_50pct'] com máximo 12 contra mínimo real dos pares difíceis 58 — gap de 46 bits. Variantes não separadas: ['crop_5pct_border', 'rotate_90', 'mirror_horizontal', 'crop_30pct_border'].
**Limiar escolhido: distância ≤ 35** (de 256 bits).

## 2. Cobertura do dHash original (NOVO)

Para as mesmas imagens/variantes do item 1, distância **dHash** (não pHash) de cada variante ao original, e quantas ficariam dentro do limiar ≤10 da Tarefa 7 (o limiar que gerou os 421.196 candidatos). Uma duplicata que nunca entra no pool de candidatos aqui não pode ser resgatada por nenhum filtro posterior, por melhor que seja.

| Variante | n | recuperadas (≤10) | taxa | min | max |
|---|---|---|---|---|---|
| Cópia idêntica | 30 | 30 | 100.00% | 0 | 0 |
| Recompressão JPEG (qualidade 40) | 30 | 30 | 100.00% | 0 | 2 |
| Redimensionamento (50%) | 30 | 30 | 100.00% | 0 | 2 |
| Recorte pequeno (5% da borda) | 30 | 23 | 76.67% | 3 | 16 |
| Rotação 90° | 30 | 2 | 6.67% | 1 | 42 |
| Espelhamento horizontal | 30 | 3 | 10.00% | 2 | 36 |
| Recorte significativo (30% da borda) | 30 | 0 | 0.00% | 22 | 44 |

**Achado**: as transformações Recorte pequeno (5% da borda), Rotação 90°, Espelhamento horizontal, Recorte significativo (30% da borda) não são 100% recuperadas pelo dHash nesta amostra — qualquer duplicata real sob essas transformações pode nunca ter entrado no pool de candidatos da Tarefa 7.

## 3. ORB+RANSAC independente do pHash

### Positivos controlados (imagem real original × cada variante, SEM qualquer filtro de pHash)

| Variante | n | avaliados | forte | fraco | não avaliável | inlier_ratio médio (avaliados) |
|---|---|---|---|---|---|---|
| Cópia idêntica | 30 | 30 | 29 | 1 | 0 | 1.000 |
| Recompressão JPEG (qualidade 40) | 30 | 29 | 28 | 1 | 1 | 0.993 |
| Redimensionamento (50%) | 30 | 29 | 24 | 5 | 1 | 0.899 |
| Recorte pequeno (5% da borda) | 30 | 30 | 29 | 1 | 0 | 0.985 |
| Rotação 90° | 30 | 30 | 29 | 1 | 0 | 0.999 |
| Espelhamento horizontal | 30 | 10 | 0 | 10 | 20 | 0.469 |
| Recorte significativo (30% da borda) | 30 | 27 | 23 | 4 | 3 | 0.832 |

ORB é projetado para ter alguma invariância rotacional/de espelhamento, diferente de dHash/pHash — a tabela acima mede isso diretamente em vez de presumir o resultado.

### Amostra real dos 2.000 pares (ORB rodado em TODOS, sem filtrar por sobrevivência de pHash)

- Pares avaliados: 2.000
- Erros de leitura de imagem: 0
- Limiar pHash-64 usado para a flag de sobrevivência (não para gatear o ORB): 5 — sobreviventes: 0 (0.00%)
- Limiar pHash-256 usado para a flag de sobrevivência (não para gatear o ORB): 35 — sobreviventes: 0 (0.00%)
- Avaliados geometricamente (ORB+RANSAC tentado): 498
  - suporte **forte**: 3
  - suporte **fraco**: 495
- Não avaliáveis geometricamente: 1.502
- Imagens de evidência renderizadas: 300 de 498 avaliados (limite: 300)
- Tempo medido para rodar ORB nos 2.000 pares: 129.2s

| Bucket dHash | n | forte | fraco | não avaliável | sobreviventes pHash-64 | sobreviventes pHash-256 |
|---|---|---|---|---|---|---|
| 0-2 | 400 | 1 | 113 | 286 | 0 | 0 |
| 3-7 | 800 | 1 | 175 | 624 | 0 | 0 |
| 8-10 | 800 | 1 | 207 | 592 | 0 | 0 |

## 4. Análise cruzada e recomendação

### Tabela cruzada (bucket dHash conhecido × pHash-64 × pHash-256 × força geométrica)

| Bucket dHash | n | sobrev. pHash-64 | sobrev. pHash-256 | forte | forte sobrev. 64 | forte sobrev. 256 | fraco | não avaliável |
|---|---|---|---|---|---|---|---|---|
| 0-2 | 400 | 0 | 0 | 1 | 0 | 0 | 113 | 286 |
| 3-7 | 800 | 0 | 0 | 1 | 0 | 0 | 175 | 624 |
| 8-10 | 800 | 0 | 0 | 1 | 0 | 0 | 207 | 592 |

- Total de pares com suporte geométrico 'forte' na amostra de 2.000: 3
- Desses, sobreviveriam ao pHash-64: 0 (rejeitados: 3)
- Desses, sobreviveriam ao pHash-256: 0 (rejeitados: 3)
- Pares difíceis conhecidos (bucket 0-2, n=400) retidos como falso candidato — pHash-64: 0; pHash-256: 0

### Recomendação

**Descartar a etapa de pHash — usar dHash → ORB direto**

- Pares com suporte geométrico 'forte' observados na amostra de 2.000: 3.
- Desses, sobreviveriam ao pHash-64: 0 (3 seriam rejeitados antes da etapa geométrica).
- Desses, sobreviveriam ao pHash-256: 0 (3 seriam rejeitados antes da etapa geométrica).
- Pares difíceis conhecidos (bucket dHash 0-2, n=400) retidos como falso candidato pelo pHash-64: 0; pelo pHash-256: 0.

**Ver a seção "Revisão manual dos pares com suporte geométrico 'forte'" abaixo antes de agir sobre esta recomendação.**

## Revisão manual dos pares com suporte geométrico 'forte'

Revisão humana direta das 3 imagens de evidência (`drawMatches`) renderizadas para os pares classificados 'forte' nesta execução — mesma prática de inspeção visual que a Tarefa 7 aplicou aos pares de distância dHash 0: nenhuma conclusão sobre duplicata real é tirada de um rótulo numérico sem abrir a imagem.

| # | path_a (HAM10000) | path_b | n_inliers | inlier_ratio | evidência |
|---|---|---|---|---|---|
| 0 | `ISIC_0033458.jpg` | `ISIC2018_Task1-2_Training_Input/ISIC_0000388.jpg` | 23 | 0.548 | `experiments/data_audit/visual_duplicate_pilot_v2_evidence/00002.png` |
| 1 | `ISIC_0024675.jpg` | `ISIC2018_Task1-2_Training_Input/ISIC_0000384.jpg` | 25 | 0.806 | `experiments/data_audit/visual_duplicate_pilot_v2_evidence/00000.png` |
| 2 | `ISIC_0033901.jpg` | `bcn_20k_train/BCN_0000015589.jpg` | 17 | 0.567 | `experiments/data_audit/visual_duplicate_pilot_v2_evidence/00001.png` |

**Achado da revisão manual (não derivável dos números sozinhos): nenhum dos 3 pares classificados 'forte' nesta execução é, visualmente, a mesma lesão.** As imagens de evidência mostram lesões/pele claramente distintas entre o lado HAM10000 e o lado ISIC2018 Task1/BCN20000. Em todos os casos, as linhas de correspondência do lado B convergem para um único ponto (ou um agrupamento muito pequeno) em vez de se espalharem pela imagem — um padrão consistente com um ajuste de homografia degenerado (pontos quase colineares/coincidentes admitem múltiplas homografias com erro de reprojeção baixo, inflando artificialmente a contagem de inliers) e não com correspondência estrutural genuína entre duas fotos da mesma lesão. É o mesmo problema estrutural que inflou os candidatos de dHash/pHash nas Tarefas 7/8 (lesão escura centralizada sobre fundo de pele genérico), agora enganando também o ORB+RANSAC, não apenas os hashes globais.

**Implicação direta para a recomendação acima**: a recomendação automática escolhe a configuração que retém os pares 'forte' encontrados, sob a premissa de que suporte geométrico forte é um proxy razoável para duplicata real. A revisão manual destes pares específicos não confirma essa premissa — nenhum é uma duplicata confirmada. Isto não invalida a lógica da recomendação (continua sendo verdade, e medido, que o pHash nos dois limiares derivados rejeitaria os únicos candidatos com suporte geométrico que este piloto encontrou — um motivo válido, por si só, para não confiar apenas no pHash), mas significa que **esta amostra de 2.000 pares não produziu nenhuma duplicata visual confirmada por revisão humana, em nenhuma das duas tarefas (8 e 9)**. A classificação 'forte'/'fraco' do ORB, do jeito que está implementada atualmente (`STRONG_MIN_INLIERS`/`STRONG_MIN_INLIER_RATIO` em `geometric_verification.py`), não filtra homografias degeneradas e portanto pode superestimar confiança geométrica em imagens de baixa variabilidade estrutural como estas — uma limitação adicional, real e observada nesta tarefa, do classificador geométrico herdado da Tarefa 8, não apenas do pHash.

### Limitação de escopo explícita

Todos os testes de transformação controlada (item 1 e item 3-controlado) avaliam **recuperação de duplicata literal** — o mesmo arquivo de imagem reprocessado (recompressão, redimensionamento, recorte, rotação, espelhamento). Eles **NÃO** demonstram capacidade de reconhecer duas fotos originalmente diferentes da mesma lesão física (ângulo ou sessão de captura diferente). Isso não foi testado nesta tarefa nem em nenhuma anterior, e permanece como lacuna registrada, não resolvida.

### Estimativa para uma execução completa futura sob a configuração recomendada

- Candidatos totais da Tarefa 7 (universo): 421.196
- Sobreviventes estimados por extrapolação linear da taxa medida na amostra — pHash-64: 0; pHash-256: 0
- Tempo medido de ORB+RANSAC nos 2.000 pares da amostra: 129.2s
- Tempo estimado (extrapolação linear) para rodar ORB+RANSAC sem qualquer gate de pHash nos 421.196 candidatos completos: ~7.6h

Esta é uma extrapolação linear simples (tempo/par medido × total de candidatos), não uma medição direta — não considera eventuais efeitos de cache de disco/zip em escala maior.

## Aviso

**Isto é piloto/detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi decidida ou implementada nesta tarefa, em nenhuma etapa do funil, mesmo para pares com suporte geométrico forte. Qualquer par listado no índice completo deste piloto (`experiments/data_audit/visual_duplicate_pilot_v2_index.json`) requer revisão humana antes de qualquer decisão sobre uso, exclusão ou reclassificação de dados. Os artefatos da Tarefa 8 permanecem intactos e válidos como registro da tentativa anterior.
