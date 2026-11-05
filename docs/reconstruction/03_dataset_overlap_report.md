# Detecção de Sobreposição/Duplicata entre Datasets — 2026-11-05

Primeira verificação de sobreposição entre datasets exigida pelo protocolo experimental ("Arquivos idênticos por SHA-256", "Identificadores das imagens... correspondências entre bases"). **Isto é detecção e relato, não decisão de exclusão.** Encontrar uma sobreposição aqui não autoriza remover nada automaticamente; qualquer achado abaixo é um candidato que precisa de decisão do pesquisador.

## PH2 x HAM10000 (comparação por conteúdo, sha256)

Comparação por conteúdo (sha256), combinando imagens+máscaras de cada lado num único manifesto por dataset (PH2 tem imagem+máscara por lesão; HAM10000 idem). Hash idêntico implica arquivo byte-idêntico, independente de nome/formato de id.

**Resultado: nenhum arquivo byte-idêntico encontrado entre PH2 (imagens+máscaras) e HAM10000 (imagens+máscaras).**

## ISIC2018 Task1 x HAM10000 (comparação por identificador `ISIC_xxxxxxx`)

Comparação por identificador `ISIC_xxxxxxx` entre os ids image×mask confirmados do ISIC2018 Task1 (2.594) e os ids metadata×image confirmados do HAM10000 (10.015).

**Resultado: nenhum id `ISIC_xxxxxxx` em comum entre o ISIC2018 Task1 (pareamento imagem×máscara) e o HAM10000 (pareamento metadado×imagem).**

## ISIC2018 Task1 x HAM10000 (comparação por conteúdo, sha256)

Comparação por conteúdo (sha256) entre as imagens do ISIC2018 Task1 hasheadas diretamente do zip (sem extração) e o manifesto imagem+máscara do HAM10000. Esta é uma checagem ADICIONAL à comparação por identificador ISIC_xxxxxxx já existente acima — id e conteúdo são checagens diferentes: ids iguais não implicam bytes iguais, e vice-versa.

**Resultado: nenhum arquivo byte-idêntico encontrado entre ISIC2018 Task1 e HAM10000.**

## BCN20000 x HAM10000 (comparação por conteúdo, sha256)

Comparação por conteúdo (sha256) entre as imagens do BCN20000 (splits treino+teste, materializados em D: e hasheados diretamente dos zips internos, sem extrair imagens individuais) e o manifesto imagem+máscara do HAM10000.

**Resultado: nenhum arquivo byte-idêntico encontrado entre BCN20000 e HAM10000.**

**Limitação do SHA-256 nesta comparação**: SHA-256 só detecta arquivos byte-idênticos. Duas imagens da mesma lesão distribuídas com resolução, recorte, compressão (ex.: JPEG com qualidade diferente) ou formato de arquivo diferentes produzem hashes diferentes mesmo sendo clinicamente 'a mesma imagem' — isso não é uma limitação desta implementação, é uma propriedade do hashing criptográfico, e é a motivação da próxima tarefa (detecção de candidatos a duplicata visual/perceptual, fora do escopo desta).

## Pares não verificáveis nesta tarefa

### PH2 x ISIC2018 Task1

Esquemas de identificador incompatíveis: PH2 usa `IMDxxx`, ISIC2018 Task1 usa `ISIC_xxxxxxx` — não há campo em comum para comparação por id. Comparação por conteúdo (hash) não foi feita nesta tarefa por não ser um dos dois pares prioritários (ISIC2018 Task1 x HAM10000 e BCN20000 x HAM10000); o manifesto de conteúdo do ISIC2018 Task1 já existe (`isic2018_task1_content_manifest.json`, gerado na Tarefa 6 sem extrair o zip) e pode ser reaproveitado para este par numa tarefa futura, caso priorizado.

### PH2 x BCN20000

Esquemas de identificador incompatíveis: PH2 usa `IMDxxx`, BCN20000 usa `BCN_xxx`/`BCNxxx` — não há campo em comum. Comparação por conteúdo (hash) não foi feita nesta tarefa por não ser um dos dois pares prioritários; o manifesto de conteúdo do BCN20000 já existe (`bcn20000_content_manifest.json`, gerado na Tarefa 6) e pode ser reaproveitado para este par numa tarefa futura, caso priorizado.

### ISIC2018 Task1 x BCN20000

Esquemas de identificador incompatíveis: ISIC2018 Task1 usa `ISIC_xxxxxxx`, BCN20000 usa `BCN_xxx`/`BCNxxx`. Comparação por conteúdo (hash) não foi feita nesta tarefa por não ser um dos dois pares prioritários; os manifestos de conteúdo de ambos já existem (gerados na Tarefa 6) e podem ser reaproveitados para este par numa tarefa futura, caso priorizado.
