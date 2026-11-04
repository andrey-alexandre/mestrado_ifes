# Detecção de Sobreposição/Duplicata entre Datasets — 2026-11-04

Primeira verificação de sobreposição entre datasets exigida pelo protocolo experimental ("Arquivos idênticos por SHA-256", "Identificadores das imagens... correspondências entre bases"). **Isto é detecção e relato, não decisão de exclusão.** Encontrar uma sobreposição aqui não autoriza remover nada automaticamente; qualquer achado abaixo é um candidato que precisa de decisão do pesquisador.

## PH2 x HAM10000 (comparação por conteúdo, sha256)

Comparação por conteúdo (sha256), combinando imagens+máscaras de cada lado num único manifesto por dataset (PH2 tem imagem+máscara por lesão; HAM10000 idem). Hash idêntico implica arquivo byte-idêntico, independente de nome/formato de id.

**Resultado: nenhum arquivo byte-idêntico encontrado entre PH2 (imagens+máscaras) e HAM10000 (imagens+máscaras).**

## ISIC2018 Task1 x HAM10000 (comparação por identificador `ISIC_xxxxxxx`)

Comparação por identificador `ISIC_xxxxxxx` entre os ids image×mask confirmados do ISIC2018 Task1 (2.594) e os ids metadata×image confirmados do HAM10000 (10.015).

**Resultado: nenhum id `ISIC_xxxxxxx` em comum entre o ISIC2018 Task1 (pareamento imagem×máscara) e o HAM10000 (pareamento metadado×imagem).**

## Pares não verificáveis nesta tarefa

### PH2 x ISIC2018 Task1

Esquemas de identificador incompatíveis: PH2 usa `IMDxxx`, ISIC2018 Task1 usa `ISIC_xxxxxxx` — não há campo em comum para comparação por id. Comparação por conteúdo (hash) não foi feita porque exigiria extrair o zip grande do ISIC2018 Task1 para obter hashes por arquivo, o que esta tarefa não autoriza.

### PH2 x BCN20000

Esquemas de identificador incompatíveis: PH2 usa `IMDxxx`, BCN20000 usa `BCN_xxx`/`BCNxxx` — não há campo em comum. Além disso, as imagens do BCN20000 nunca foram extraídas/hasheadas (decisão da Tarefa 2/3: zips internos não abertos), então comparação por conteúdo também não é possível com os manifestos atuais.

### ISIC2018 Task1 x BCN20000

Esquemas de identificador incompatíveis: ISIC2018 Task1 usa `ISIC_xxxxxxx`, BCN20000 usa `BCN_xxx`/`BCNxxx`. As imagens do BCN20000 nunca foram extraídas/hasheadas, então comparação por conteúdo também não é possível com os manifestos atuais.

### BCN20000 x HAM10000

As imagens do BCN20000 nunca foram extraídas/hasheadas (decisão da Tarefa 2/3: zips internos `BCN_20k_train.zip`/`BCN_20k_test.zip` não abertos), então nem comparação por conteúdo (hash) nem por id de imagem são possíveis com os manifestos atuais. O `lesion_id` do BCN20000 (`BCN_xxx`) também não é comparável ao `lesion_id` do HAM10000 (`HAM_xxx`) — formatos de fontes diferentes, sem relação conhecida.
