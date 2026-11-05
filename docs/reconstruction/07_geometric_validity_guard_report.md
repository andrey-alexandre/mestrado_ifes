# Guarda Contra Homografia Degenerada — Validada com Conjunto Separado — 2026-11-05

Extensão da Tarefa 9. A revisão manual da Tarefa 9 encontrou que os 3 pares classificados 'forte' pelo classificador geométrico existente (`classify_geometric_support`, `src/data_audit/geometric_verification.py`) não eram, visualmente, a mesma lesão: em todos os 3, as correspondências inlier do lado B convergiam para um único ponto ou agrupamento muito pequeno em vez de se espalharem pela imagem — padrão de homografia degenerada, que o classificador atual (baseado só em contagem/proporção de inliers) não detecta. O pesquisador rejeitou a recomendação da Tarefa 9 (descartar pHash em favor de dHash→ORB direto) por motivo explícito: recuperar os 3 candidatos que depois se confirmaram falso-positivos não demonstra vantagem desse fluxo. Corrigir apenas os 3 exemplos conhecidos também não seria suficiente — esta tarefa constrói um critério geométrico de validade adicional (dispersão espacial dos pontos inlier), ajustado em um subconjunto e validado em outro, nunca visto durante o ajuste.

**Piloto curto e delimitado**: não roda o pipeline completo (nem os 2.000 pares da Tarefa 9, nem os 421.196 candidatos da Tarefa 7) — apenas o conjunto pequeno de ajuste+validação descrito abaixo. Nenhuma exclusão de lesão/imagem é decidida ou implementada em nenhuma etapa. Os artefatos das Tarefas 7–9 não são modificados.

## 1. Métrica de dispersão espacial e guarda de degenerescência

**Métrica escolhida**: área do fecho convexo (`cv2.convexHull` + `cv2.contourArea`) dos pontos inlier de cada lado do par, normalizada pela área da imagem correspondente (`src/data_audit/geometric_degeneracy.py::convex_hull_normalized_area`). Justificativa: os 3 casos conhecidos da Tarefa 9 mostram dois padrões de degenerescência — pontos convergindo para um agrupamento pequeno (clustering) e pontos quase colineares — e o fecho convexo captura ambos com um único escalar (tende a 0 em qualquer um dos dois casos, só cresce quando os pontos se espalham por uma extensão 2-D real da imagem). Um desvio-padrão ou bounding-box isolado captura clustering mas não colinearidade (uma linha fina de pontos pode ter bounding-box grande). Menos de 3 pontos não pode delimitar área alguma — reportado como `0.0` (máxima degenerescência), nunca como indefinido.

**Classificação degenerada**: uma configuração é degenerada quando a dispersão de QUALQUER um dos dois lados fica abaixo do limiar (`is_degenerate`) — um lado colapsado já invalida a correspondência, pois correspondência estrutural genuína exige estrutura real espalhada nos dois lados, não em só um (nos 3 casos conhecidos, o colapso ocorreu sempre do lado HAM10000, nunca nos dois lados simultaneamente, o que exclui uma regra baseada só na média dos dois lados).

**Integração**: nova função `classify_with_degeneracy_guard` (não substitui `classify_geometric_support`, envolve-a). Um par classificado 'forte' pelo critério existente E degenerado pela guarda recebe o rótulo distinto e explícito `"forte_degenerada"` — nunca vira 'fraco' silenciosamente. Pares 'fraco'/'not_evaluable' passam pela guarda sem mudança de rótulo (a guarda só atua sobre 'forte', que é o rótulo que esta tarefa existe para corrigir), mas a dispersão é computada e registrada mesmo para eles, para visibilidade diagnóstica.

## 2. Composição dos conjuntos de ajuste e validação

### Positivos (duplicata real, controlada — split por imagem)

| Fonte | imagens ajuste | imagens validação |
|---|---|---|
| HAM10000 | 3 | 3 |
| ISIC2018 Task1 | 3 | 3 |
| BCN20000 | 3 | 3 |

Cada imagem gera 7 variantes (mesmo conjunto das Tarefas 8/9: cópia idêntica, recompressão JPEG, redimensionamento, recorte pequeno, recorte significativo, rotação 90°, espelhamento horizontal), avaliadas par a par (base × variante) com ORB+RANSAC.
- Pares positivos usados no ajuste (restritos a `base_strength=="forte"` — ver seção 3 para a justificativa dessa restrição): 44

### Negativos (configuração degenerada conhecida ou provável)

- 3 pares 'forte' falso-positivo já conhecidos da Tarefa 9 (caminhos exatos reaproveitados) — **reservados só para validação, nunca usados no ajuste**.
- 10 pares adicionais minerados de `experiments/data_audit/visual_duplicate_pilot_v2_index.json`: classificados 'fraco', status 'evaluated', mais próximos do limiar atual de suporte forte (`min(n_inliers/STRONG_MIN_INLIERS, inlier_ratio/STRONG_MIN_INLIER_RATIO)` descendente) — os casos mais difíceis disponíveis nesta amostra. Cada par vem do pool de candidatos da Tarefa 7, sobre o qual a revisão manual da Tarefa 9 concluiu que esta amostra específica de 2.000 pares não produziu nenhuma duplicata visual confirmada — essa constatação de nível populacional, não uma revisão manual par-a-par renovada, é a convenção adotada aqui para tratá-los como negativos conhecidos (limitação registrada na seção 7).
  - Divididos por interleaving de dificuldade: 5 para ajuste, 5 para validação.
- Pares negativos avaliáveis no conjunto de ajuste (minerados apenas, os 3 conhecidos ficam fora do ajuste): 5

## 3. Ajuste do limiar (conjunto de ajuste apenas)

**Convenção adotada**: o grupo "positivo" usado para ajustar o limiar é restrito a pares genuinamente duplicados (variante controlada) que a linha de base (`classify_geometric_support`) já classificou como `"forte"` — a guarda só é aplicada a pares `"forte"`, então um par genuíno que a linha de base já rejeita como `"fraco"` (ex.: 0 inliers RANSAC sobreviveram para uma transformação por outro lado válida) nunca chegaria à guarda na prática; incluí-lo no grupo positivo poluiria a distribuição com dispersão=0 trivial (poucos pontos para medir qualquer coisa), não com o padrão de colapso estrutural que o limiar precisa separar dos negativos. Os negativos minerados permanecem sem essa restrição (são deliberadamente 'fraco' por construção — ver seção 2).

- Dispersão mínima observada entre positivos de ajuste: 0.001255
- Dispersão máxima observada entre negativos de ajuste: 0.061005
- Gap limpo observado: False
- **Limiar de dispersão escolhido: 0.061005**
- **Sem separação limpa observada no conjunto de ajuste**: o limiar acima é o fallback (máximo dos negativos de ajuste), não uma separação medida — risco residual explícito, ver seção 6.

## 4. Validação (conjunto de validação apenas, nunca visto no ajuste)

### Os 3 casos originais (reservados só para validação)

| path_a (HAM10000) | path_b | suporte base | dispersão A | dispersão B | classificação final |
|---|---|---|---|---|---|
| `ISIC_0033458.jpg` | `ISIC2018_Task1-2_Training_Input/ISIC_0000388.jpg` | forte | 0.021747 | 0.002165 | forte_degenerada |
| `ISIC_0024675.jpg` | `ISIC2018_Task1-2_Training_Input/ISIC_0000384.jpg` | forte | 0.138981 | 0.000000 | forte_degenerada |
| `ISIC_0033901.jpg` | `bcn_20k_train/BCN_0000015589.jpg` | forte | 0.041563 | 0.000030 | forte_degenerada |

**3/3 dos casos originais foram corretamente reclassificados como `"forte_degenerada"` pela guarda, no conjunto de validação (limiar nunca ajustado olhando para estes 3 pares).**

### Negativos minerados adicionais de validação

- 5/5 permanecem fora de 'forte'/'forte_degenerada' (todos eram 'fraco' na linha de base, logo nenhum é promovido incorretamente a suporte forte pela guarda).
- 5/5 têm a dispersão flagada como degenerada mesmo assim (diagnóstico: confirma que o padrão de colapso espacial também ocorre em casos que o classificador de contagem/proporção já rejeitava por outro motivo, não apenas nos 3 casos 'forte').

### Positivos de validação, por tipo de transformação

| Transformação | n | avaliáveis | 'forte' na linha de base | permanece 'forte' | reclassificado 'forte_degenerada' |
|---|---|---|---|---|---|
| Cópia idêntica | 9 | 9 | 9 | 9 | 0 |
| Recompressão JPEG (qualidade 40) | 9 | 9 | 9 | 9 | 0 |
| Redimensionamento (50%) | 9 | 9 | 7 | 7 | 0 |
| Recorte pequeno (5% da borda) | 9 | 9 | 9 | 8 | 1 |
| Rotação 90° | 9 | 9 | 9 | 9 | 0 |
| Espelhamento horizontal | 9 | 2 | 0 | 0 | 0 |
| Recorte significativo (30% da borda) | 9 | 7 | 5 | 2 | 3 |

Leitura: a guarda não deve reduzir, no conjunto de validação, a contagem de positivos reais classificados 'forte' na linha de base (coluna 'permanece forte' deveria igualar 'forte na linha de base' quando a guarda funciona como pretendido; qualquer diferença indica que a guarda penalizou indevidamente uma duplicata real sob aquela transformação — ver números acima para o resultado medido, não assumido).

## 5. Alcance proposto de uma eventual execução completa (para revisão, não decisão)

A lacuna de cobertura do dHash original (Tarefa 9: recorte pequeno 76,67%, rotação 6,67%, espelhamento 10%, recorte significativo 0% recuperados no limiar ≤10) **não é resolvida por esta tarefa** — a guarda geométrica só pode atuar sobre pares que já entraram no funil como candidatos; não resgata nenhuma duplicata que o dHash nunca apresentou como candidato. Com base nas taxas de avaliabilidade/suporte medidas nas Tarefas 9 e 10 para cada transformação, o alcance proposto de um funil dHash → ORB+guarda completo, por transformação, é:

- **Cópia idêntica / recompressão / redimensionamento / recorte pequeno**: cobertura dHash alta (76–100%, Tarefa 9) e, entre os pares que chegam à etapa geométrica, suporte 'forte' consistente (ver tabela da seção 4) — risco residual baixo, mas não medido na escala completa.
- **Rotação 90° / recorte significativo**: cobertura dHash muito baixa (0–6,67%, Tarefa 9) — mesmo com a guarda funcionando perfeitamente na etapa geométrica, a maioria das duplicatas reais sob essas transformações nunca chegaria a essa etapa. Risco residual alto e não mitigado por esta tarefa.
- **Espelhamento horizontal**: cobertura dHash baixa (10%, Tarefa 9) E baixa avaliabilidade geométrica mesmo quando o par chega à etapa ORB (Tarefa 9: 10/30 avaliáveis, 0 'forte' na amostra controlada) — risco residual duplo, da entrada no funil e da etapa geométrica em si.
- Nenhuma extrapolação de contagem de sobreviventes para os 421.196 candidatos é feita aqui — esta tarefa não mediu suporte geométrico em nenhuma amostra nova desses candidatos, apenas reavaliou 3 pares já conhecidos + pares minerados do índice da Tarefa 9.

## 6. Recomendação final de procedimento e limitações

A guarda de dispersão corrigiu 3/3 dos casos conhecidos no conjunto de validação, usando um limiar derivado inteiramente de um conjunto de ajuste disjunto — ou seja, a correção generaliza para casos não vistos durante o ajuste, não apenas para os 3 exemplos que motivaram a tarefa. **Isso é a evidência central a favor da guarda.**

**Mas a evidência não sustenta adoção sem reserva**: o conjunto de ajuste não produziu separação limpa entre dispersão de positivos genuínos e de negativos (seção 3) — a dispersão mínima entre positivos 'forte' de ajuste (0.001255) ficou ABAIXO da dispersão máxima entre negativos de ajuste (0.061005), e o limiar usado é o fallback (máximo dos negativos), não uma separação medida. O preço disso aparece diretamente na validação: 3/5 positivos genuínos sob recorte significativo (30%) e 1/9 sob recorte pequeno (5%) foram reclassificados 'forte_degenerada' incorretamente — um recorte agressivo restringe estruturalmente a região de correspondência real, produzindo dispersão baixa por motivo legítimo, não degenerado, e a guarda atual não distingue os dois casos.

**Recomendação**: usar `classify_with_degeneracy_guard` como um sinalizador adicional para revisão humana sobre pares 'forte' (reduz a carga de revisão ao apontar os casos mais suspeitos primeiro), **não** como critério automático de reclassificação/exclusão — o limiar atual (derivado sem gap limpo) tem demonstravelmente falsos positivos conhecidos sob recorte agressivo. Antes de qualquer uso que trate 'forte_degenerada' como equivalente a 'fraco' em decisões posteriores, valeria revisar se recorte deveria ter um limiar próprio (ou ser excluído da guarda), o que esta tarefa não testou.

**Limitações explícitas:**
- Amostra de ajuste/validação pequena (poucas dezenas de pares no total) — suficiente para mostrar generalização além dos 3 casos conhecidos, não para estimar uma taxa de erro com precisão estatística.
- Os negativos minerados além dos 3 conhecidos não passaram por revisão manual individual — apoiam-se na constatação de nível populacional da Tarefa 9 (ver seção 2), não em confirmação par-a-par.
- Esta tarefa não produziu nenhuma duplicata real confirmada entre os pares HAM10000×ISIC2018 Task1/BCN20000 — todo o sinal de 'positivo real' usado para ajustar/validar o limiar vem de pares controlados (mesma imagem reprocessada), nunca de duas fotos originalmente diferentes da mesma lesão física. Essa lacuna, já registrada na Tarefa 9, permanece não resolvida.
- Espelhamento/rotação continuam com baixa avaliabilidade geométrica (ver seção 4) — limitação herdada do ORB, não introduzida nem corrigida por esta guarda.

## Aviso

**Isto é piloto/detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi decidida ou implementada nesta tarefa. Os artefatos das Tarefas 7–9 permanecem intactos e válidos. O índice completo desta tarefa está em `experiments/data_audit/geometric_degeneracy_guard_pilot_index.json`.
