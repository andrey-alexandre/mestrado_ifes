# Candidatos a Duplicata Visual entre Datasets — 2026-11-05

Extensão da detecção de sobreposição da Tarefa 6: SHA-256 só encontra arquivos byte-idênticos, então a mesma foto de lesão reenviada com resolução, recorte, rotação, espelhamento ou compressão diferentes não é detectada por hash criptográfico. Esta tarefa implementa a detecção de **candidatos** a essa duplicata visual via hash perceptual (dHash). **Isto é detecção, não confirmação: nenhuma exclusão de lesão/imagem foi decidida ou implementada aqui. Todo candidato listado requer revisão humana antes de qualquer decisão sobre uso dos dados.**

## Método

**dHash (difference hash)**: a imagem é decodificada, convertida para escala de cinza e redimensionada para 9×8 pixels; cada bit do hash de 64 bits compara um pixel com seu vizinho horizontal (bit 1 se o pixel atual é mais claro que o próximo). Esse método foi escolhido porque não exige treinar nem carregar nenhum modelo (usa só Pillow+numpy, já disponíveis no ambiente), é determinístico, e é tolerante a exatamente o tipo de reprocessamento esperado entre releases do mesmo dataset (recompressão, redimensionamento, recorte pequeno) — porque codifica gradientes grosseiros de brilho, não valores exatos de pixel. Em contrapartida, não é tolerante a rotação, espelhamento ou recorte significativo, porque essas transformações mudam quais gradientes ficam adjacentes a quais — ver seção de limitações abaixo, com números medidos, não apenas teoria.

**Limiar de candidatura escolhido: distância de Hamming ≤ 10** (de 64 bits). Derivado dos testes controlados abaixo: variantes que são genuinamente a mesma imagem reprocessada (cópia idêntica, recompressão JPEG, redimensionamento, recorte pequeno de 5% da borda) mediram distância máxima 6; variantes com alteração estrutural (rotação 90°, espelhamento horizontal, recorte significativo de 30% da borda) mediram distância mínima 16. O limiar escolhido (10) fica estritamente entre os dois agrupamentos medidos, com margem de 4 abaixo do caso real mais alto e margem de 6 acima do caso estrutural mais baixo — não foi escolhido arbitrariamente.

## Testes controlados (exemplos sintéticos, não imagens de paciente)

Base: uma imagem sintética (gradiente + formas geométricas, gerada via `PIL.ImageDraw`) e um conjunto fixo de variantes dela. Números medidos pela mesma função usada pelos testes automatizados (`src/data_audit/visual_duplicates.py::measure_controlled_variant_distances`), não copiados manualmente:

| Variante | Distância de Hamming observada | Candidato (distância ≤ 10)? |
|---|---|---|
| Cópia idêntica | 0 | sim |
| Recompressão JPEG (qualidade 40) | 1 | sim |
| Redimensionamento (50%) | 2 | sim |
| Recorte pequeno (5% da borda) | 6 | sim |
| Rotação 90° | 16 | não |
| Espelhamento horizontal | 36 | não |
| Recorte significativo (30% da borda) | 17 | não |

## Limitações explícitas (confirmadas pelos testes acima, não apenas em teoria)

- **Sensível a rotação**: uma rotação de 90° mediu distância 16 (acima do limiar 10) — **não detectada** como candidato por este método.
- **Sensível a espelhamento**: um espelhamento horizontal mediu distância 36 (acima do limiar 10) — **não detectada** como candidato por este método.
- **Sensível a recorte significativo**: um recorte de 30% da borda mediu distância 17 (acima do limiar 10) — **não detectada** como candidato por este método.
- Um candidato com distância baixa não é uma duplicata confirmada: imagens diferentes, mas visualmente simples/parecidas (ex.: lesão redonda centrada em pele lisa) podem coincidir por acaso. Cada candidato requer inspeção humana da imagem lado a lado antes de qualquer conclusão.

## Cobertura por dataset (dHash)

- **HAM10000**: expected=10015, processed=10015, unreadable=0
- **ISIC2018 Task1**: expected=2594, processed=2594, unreadable=0
- **BCN20000 (treino+teste)**: expected=18914, processed=18914, unreadable=0

## HAM10000 x ISIC2018 Task1

- Imagens comparadas: 10015 x 2594 (25.978.910 pares possíveis)
- Limiar de distância usado: 10
- **Total de candidatos encontrados: 136761**
- Contagem esperada sob um modelo nulo de hashes uniformemente aleatórios e independentes (ou seja, SE as imagens não compartilhassem nenhuma estrutura): ~0.26
- Imagens lado a lado renderizadas: 200 (as de menor distância primeiro; limite por par: 200)
- Pasta das imagens lado a lado: `experiments/data_audit/visual_candidates/ham10000_x_isic2018_task1`
- Índice completo (todos os 136761 candidatos, renderizados ou não): `experiments/data_audit/visual_candidates_index.json`

**Nota**: 136761 candidatos encontrados excede o limite de renderização (200); apenas os 200 de menor distância foram renderizados como imagem — a contagem total real está acima, não escondida pela amostra, e está disponível completa no índice.

O total observado (136761) é muitas ordens de grandeza maior do que o esperado sob o modelo nulo (~0.26) — ver seção "Achado real" abaixo para a interpretação disso.

## HAM10000 x BCN20000

- Imagens comparadas: 10015 x 18914 (189.423.710 pares possíveis)
- Limiar de distância usado: 10
- **Total de candidatos encontrados: 284435**
- Contagem esperada sob um modelo nulo de hashes uniformemente aleatórios e independentes (ou seja, SE as imagens não compartilhassem nenhuma estrutura): ~1.89
- Imagens lado a lado renderizadas: 200 (as de menor distância primeiro; limite por par: 200)
- Pasta das imagens lado a lado: `experiments/data_audit/visual_candidates/ham10000_x_bcn20000`
- Índice completo (todos os 284435 candidatos, renderizados ou não): `experiments/data_audit/visual_candidates_index.json`

**Nota**: 284435 candidatos encontrados excede o limite de renderização (200); apenas os 200 de menor distância foram renderizados como imagem — a contagem total real está acima, não escondida pela amostra, e está disponível completa no índice.

O total observado (284435) é muitas ordens de grandeza maior do que o esperado sob o modelo nulo (~1.89) — ver seção "Achado real" abaixo para a interpretação disso.

## Achado real: alta taxa de falso-positivo em imagens dermatoscópicas

A contagem observada de candidatos em ambos os pares é muitas ordens de grandeza maior do que a esperada sob um modelo nulo de hashes uniformemente aleatórios (ver `Contagem esperada sob um modelo nulo` em cada seção acima). Isso não indica um bug — indica que imagens dermatoscópicas de datasets diferentes **não são visualmente aleatórias entre si**: elas compartilham fortemente a mesma convenção de aquisição (lesão pigmentada centralizada, fundo de pele razoavelmente uniforme, iluminação e enquadramento do dermatoscópio, por vezes régua/pelos), o que já basta para produzir um dHash (9×8, grosseiro) muito parecido entre imagens de **pacientes e lesões completamente diferentes**.

Inspeção manual direta (local, nunca publicada — ver restrições desta tarefa) dos candidatos de menor distância possível (distância 0, o caso mais forte que este método pode produzir) confirmou isso: em `experiments/data_audit/visual_candidates/ham10000_x_isic2018_task1/00000.png` (`ISIC_0026029.jpg` x `ISIC2018_Task1-2_Training_Input/ISIC_0010220.jpg`) e em `experiments/data_audit/visual_candidates/ham10000_x_bcn20000/00000.png` (`ISIC_0025653.jpg` x `bcn_20k_test/BCN_0000005436.jpg`), as duas imagens de cada par mostram claramente lesões diferentes — não a mesma imagem reenviada. A semelhança é de gênero fotográfico, não de conteúdo.

**Implicação prática**: o volume de candidatos (incluindo os de distância 0) não deve ser lido como "dezenas de milhares de duplicatas prováveis". É a taxa de falso-positivo esperada deste método de baixa resolução aplicado a um domínio visualmente homogêneo — e reforça, em vez de enfraquecer, a exigência de que cada candidato seja revisado individualmente por um humano antes de qualquer conclusão sobre duplicata real.

## Aviso

**Isto é detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi decidida ou implementada nesta tarefa, para nenhum candidato, mesmo os de distância muito baixa. Qualquer candidato listado no índice completo (`experiments/data_audit/visual_candidates_index.json`) requer revisão humana — da imagem lado a lado e do contexto de cada dataset — antes de qualquer decisão sobre uso, exclusão ou reclassificação de dados.
