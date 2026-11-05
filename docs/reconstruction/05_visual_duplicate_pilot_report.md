# Piloto de Filtragem em Etapas (pHash + Verificação Geométrica) — 2026-11-05

Extensão da Tarefa 7 (detecção de candidatos por dHash). A Tarefa 7 encontrou 421.196 candidatos nos dois pares prioritários e seu relatório concluiu que a maioria seria ruído — **essa conclusão foi um exagero e está corrigida aqui**: os dois pares de distância 0 inspecionados na Tarefa 7 mostraram imagens diferentes, o que prova que dHash idêntico não confirma duplicata visual, mas **não prova** que os demais candidatos sejam ruído, nem que os pares inspecionados sejam de pacientes diferentes (aparência visual não é metadado de identidade de paciente), nem que o modelo nulo de bits aleatórios usado na Tarefa 7 seja a distribuição de referência correta para imagens dermatoscópicas reais. Este piloto não assume nenhuma dessas três afirmações.

**Este é um PILOTO**: valida o método (auditoria de pré-processamento, pHash, verificação geométrica) numa amostra estratificada, sem rodar a busca completa nos 421.196 candidatos. A Tarefa 7 não foi sobrescrita — seus artefatos (`visual_candidates_index.json`, `visual_candidates/`, `04_visual_duplicate_candidates_report.md`) permanecem intactos como registro da tentativa anterior. **Isto é piloto/detecção, não confirmação: nenhuma exclusão de lesão/imagem foi decidida ou implementada, em nenhuma etapa, mesmo para pares com score geométrico forte.**

## 1. Auditoria dirigida de pré-processamento

### Verificação de comportamento do `PIL.Image.open` frente a orientação EXIF

Verificação comportamental (não por amostra): imagem sintética com orientação EXIF conhecida (tag `0x0112`=6) codificada e reaberta. Tamanho bruto esperado: `(100, 60)`. Tamanho reportado por `PIL.Image.open()` (sem aplicar `ImageOps.exif_transpose`): `(100, 60)`.

**Resultado: `PIL.Image.open()` aplica orientação EXIF automaticamente? NÃO** (confirmado nos bytes reais acima, não assumido da documentação).

### Distribuição real da tag de orientação EXIF por fonte (amostra)

- **HAM10000**: amostrados=200, processados=200, ilegíveis=0
  - valores de orientação observados: {1: 200}; não-padrão (!=1): 0
- **ISIC2018 Task1**: amostrados=200, processados=200, ilegíveis=0
  - valores de orientação observados: {1: 200}; não-padrão (!=1): 0
- **BCN20000**: amostrados=200, processados=200, ilegíveis=0
  - valores de orientação observados: {1: 200}; não-padrão (!=1): 0

### Dimensões nativas por fonte (amostra)

- **HAM10000**: largura [600, 600, 600] (min/média/max), altura [450, 450, 450], razão de aspecto [1.333, 1.333, 1.333] (alvo do grid dHash 9x8 = 1.125; desvio absoluto médio = 0.208)
- **ISIC2018 Task1**: largura [576, 3089, 6708] (min/média/max), altura [542, 2111, 4461], razão de aspecto [0.750, 1.416, 1.518] (alvo do grid dHash 9x8 = 1.125; desvio absoluto médio = 0.295)
- **BCN20000**: largura [1024, 1024, 1024] (min/média/max), altura [1024, 1024, 1024], razão de aspecto [1.000, 1.000, 1.000] (alvo do grid dHash 9x8 = 1.125; desvio absoluto médio = 0.125)

### Imagens degeneradas (quase uniformes / próximas de preto ou branco)

- **HAM10000**: 0/200 sinalizadas (limiar std<5.0, margem extrema de média 10.0)
- **ISIC2018 Task1**: 0/200 sinalizadas (limiar std<5.0, margem extrema de média 10.0)
- **BCN20000**: 0/200 sinalizadas (limiar std<5.0, margem extrema de média 10.0)

### Checagem de "hub" — valores de dHash distintos respondendo por fração desproporcional dos candidatos

- Total de candidatos (Tarefa 7): 421.196
- Valores de dHash distintos envolvidos nos candidatos: 17.396
- Participações (soma, um hash pode contar 2x no mesmo candidato se aparecer dos dois lados) dos 20 hashes mais frequentes: 26.118.
- Candidatos distintos que tocam pelo menos um desses 20 hashes: 25.716 (6.11% do total de candidatos).

| dHash (int) | nº de candidatos em que aparece |
|---|---|
| `1085142330213338895` | 3.000 |
| `1085102747793952527` | 2.340 |
| `1085102747796311823` | 1.572 |
| `1085102747793428239` | 1.572 |
| `1085142330212814607` | 1.492 |
| `1085142330214911751` | 1.354 |
| `1085102609818128143` | 1.351 |
| `1085102765040930575` | 1.192 |
| `1085146728326172431` | 1.168 |
| `508646393539399439` | 1.132 |

## 2. Validação do pHash em dados reais

Imagens reais usadas (amostra HAM10000, semente=42): 8.

### Distância pHash por variante controlada (sobre as imagens reais, não a sintética)

| Variante | min | média | max | Recuperada (max ≤ limiar)? |
|---|---|---|---|---|
| Cópia idêntica | 0 | 0.0 | 0 | sim |
| Recompressão JPEG (qualidade 40) | 2 | 6.5 | 12 | sim |
| Redimensionamento (50%) | 0 | 4.2 | 10 | sim |
| Recorte pequeno (5% da borda) | 52 | 69.8 | 88 | não |
| Rotação 90° | 116 | 127.0 | 142 | não |
| Espelhamento horizontal | 124 | 128.2 | 134 | não |
| Recorte significativo (30% da borda) | 118 | 130.2 | 144 | não |

### Pares difíceis reais (candidatos dHash distância 0–2 da Tarefa 7, n=400)

Estes são os falsos positivos reais já confirmados (par com dHash colidindo em imagens diferentes) — o benchmark de falso-positivo que já temos à mão, reaproveitado do `visual_candidates_index.json` da Tarefa 7, não um conjunto inventado.

- Distância pHash: min=58, mediana=112.0, média=112.7, max=152

### Separação observada e limiar escolhido

Separação real observada: as variantes Cópia idêntica, Recompressão JPEG (qualidade 40), Redimensionamento (50%) ficaram estritamente abaixo da distância mínima observada nos pares difíceis reais (máximo 12 vs. mínimo real de pares difíceis 58) — gap real de 46 bits. **Achado real, não esperado pelo comportamento do dHash na Tarefa 7**: Recorte pequeno (5% da borda), Rotação 90°, Espelhamento horizontal, Recorte significativo (30% da borda) NÃO ficaram separadas dos pares difíceis reais (sua distância máxima observada cai dentro ou acima da faixa de distância dos pares difíceis) — ao contrário do dHash, este pHash (DCT global de imagem inteira) não é tolerante a recorte, mesmo pequeno: um recorte desloca o conteúdo espacial o bastante para alterar substancialmente os coeficientes DCT de baixa frequência. Isto é reportado como limitação real, não escondido.

**Limiar de candidatura pHash escolhido: distância ≤ 35** (de 256 bits).

## 3. Funil do piloto

- Candidatos totais da Tarefa 7 (universo de onde a amostra foi tirada): 421.196
- Pares na amostra estratificada do piloto: 2.000
  - distância 0-2: 400 amostrados de 400 candidatos reais nessa faixa (todos)
  - distância 3-7: 800 amostrados de 57.614 candidatos reais nessa faixa (semente=42)
  - distância 8-10: 800 amostrados de 363.182 candidatos reais nessa faixa (semente=43)
- Sobreviventes do filtro pHash (distância ≤ 35): 0 (0.0% da amostra)
- Avaliados geometricamente (ORB+RANSAC tentado): 0
  - suporte **forte**: 0
  - suporte **fraco**: 0
- Não avaliáveis geometricamente (poucos keypoints/matches, não "sem suporte"): 0
- Imagens de evidência (drawMatches) renderizadas: 0 de 0 avaliados (limite: 300)

## 4. Reafirmação corrigida da limitação de interpretação

Hash idêntico (dHash ou pHash) — mesmo nos dois casos de distância dHash 0 inspecionados na Tarefa 7 — **não confirma duplicata visual nem identidade de paciente**. Os casos inspecionados confirmaram apenas que, visualmente, as imagens mostravam lesões diferentes; isso não foi cruzado com nenhum metadado de paciente/lesão, porque nenhum dos datasets usados fornece esse cruzamento direto entre HAM10000, ISIC2018 Task1 e BCN20000. Da mesma forma, suporte geométrico forte (ORB+RANSAC) neste piloto é evidência substancialmente mais específica que coincidência de hash global, mas **também não é confirmação automática** — depende de quantos keypoints uma imagem realmente tem e de quão genérica é sua textura (ver limitação em `geometric_verification.py`: imagens pouco texturizadas, comuns em pele lisa, produzem poucos keypoints e ficam `não avaliáveis`, não "sem suporte"). Toda conclusão sobre duplicata real requer revisão humana da evidência (imagem lado a lado + imagem de correspondência geométrica) par a par.

## 5. Recomendação técnica para uma eventual execução completa futura

**Esta é uma recomendação para revisão — não uma decisão tomada nem uma execução completa.** A execução completa (todos os 421.196 candidatos) NÃO foi feita nesta tarefa.

Recomendação técnica (para revisão do pesquisador, não decidida nem executada aqui):

- **Manter o pipeline em 3 etapas** (dHash -> pHash -> ORB+RANSAC) para a execução completa futura, na mesma ordem: cada etapa é mais cara e mais discriminativa que a anterior, e aplicar a mais cara (geométrica) só aos sobreviventes das etapas baratas é o que torna uma execução completa nos 421.196 candidatos (e não só numa amostra) viável em tempo razoável.
- **Achado real deste piloto que precisa ser resolvido antes da execução completa: ZERO dos 2.000 pares amostrados (incluindo os 400 pares difíceis reais de distância dHash 0-2 completos) sobreviveram ao filtro pHash** (limiar 35; a menor distância pHash observada em toda a amostra foi bem acima do limiar, não um quase-empate). A etapa geométrica (item 3) não chegou a ser exercida em nenhum par real neste piloto como consequência direta disso -- só em dados sintéticos/teste de integração. Antes de rodar a execução completa com esta mesma configuração, **recomenda-se validar diretamente com ORB+RANSAC uma amostra de candidatos dHash de baixa distância SEM passar pelo filtro pHash**, para checar se o limiar pHash (que não recupera recorte, ver achado do item 2) está descartando positivos reais antes que a verificação geométrica tenha a chance de avaliá-los -- o resultado atual é consistente tanto com "não há duplicatas reais na amostra" quanto com "o filtro pHash está rejeitando duplicatas reais baseadas em recorte antes da etapa geométrica", e este piloto não consegue distinguir as duas.
- **Rodar o filtro pHash nos candidatos completos da Tarefa 7** (não recalcular dHash; reaproveitar o índice e os manifestos já existentes), usando o limiar derivado neste piloto a partir de dados reais -- reavaliar esse limiar nos números completos antes de finalizar, já que 400 pares difíceis e 8 imagens reais é uma base pequena para fixar um valor definitivo.
- **Documentar explicitamente, na execução completa, que recorte (mesmo pequeno) não é recuperado pelo pHash nesta implementação** (achado real deste piloto, item 2) -- se recorte for uma preocupação relevante do protocolo, a etapa pHash não deve ser tratada como suficiente sozinha para descartar candidatos de baixa distância dHash que o pHash classificar como não-candidatos; quem decidir o uso final dos dados precisa saber dessa lacuna.
- **Rodar a verificação geométrica em todos os sobreviventes do pHash na execução completa**, não apenas numa amostra -- o custo computacional por par (ORB + RANSAC) é pequeno comparado ao hashing em massa; o gargalo esperado é o número de sobreviventes do pHash, não o método em si.
- **Revisar os limiares de classificação "forte"/"fraco"** (`STRONG_MIN_INLIERS`, `STRONG_MIN_INLIER_RATIO` em `geometric_verification.py`) contra a distribuição real observada na execução completa -- os valores atuais são um ponto de partida razoável, não validados contra um conjunto rotulado.
- Para qualquer par classificado como suporte "forte" na execução completa: **isto continua sendo evidência para revisão humana, não confirmação automática de duplicata nem base para exclusão de dado**.

## Aviso

**Isto é piloto/detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi decidida ou implementada nesta tarefa, em nenhuma etapa do funil, mesmo para pares com score geométrico forte. Qualquer par listado no índice completo do piloto (`experiments/data_audit/visual_duplicate_pilot_index.json`) requer revisão humana antes de qualquer decisão sobre uso, exclusão ou reclassificação de dados. Os artefatos da Tarefa 7 permanecem intactos e válidos como registro da tentativa anterior.
