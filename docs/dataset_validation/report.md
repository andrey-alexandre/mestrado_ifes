# Relatório — investigação PH² × HAM10000 (piloto exploratório)

**Estado: etapas locais concluídas; inferência VLM e avaliação diagnóstica NÃO executadas**
(aguardam autorização de custo e credenciais: 640 chamadas, 2 modelos × 80 imagens × 4 agentes).
Tabelas completas: `experiments/dataset_validation/reports/report.md`.
Tudo abaixo é exploratório (n = 20 por grupo), sem inferência causal.

## 1. Evidências recuperadas (PH² legado, sem reconstrução)

- Matrizes da dissertação reproduzidas exatamente (GPT 17/8/8/17; Gemini 21/4/8/17).
- **Dois melanomas (IMD061, IMD065) constam como negativos** nas 50 imagens legadas (rótulo
  legado); com o diagnóstico clínico a amostra é 27/23. Re-lendo as mesmas respostas com o
  rótulo clínico: GPT sens 0,67 / espec 0,70; Gemini sens 0,70 [0,52–0,84] / espec 0,91 [0,73–0,98]
  (leitura manual da dissertação). Não substitui a tabela publicada; é uma verificação.
- A **regra de leitura decide o resultado**: aplicada programaticamente (indeterminada→positiva,
  regra do repositório), o GPT-5.4-nano vira 100 % positivo (48/50 "indeterminada"), enquanto o
  Gemini fica em sens 0,89 / espec 0,87. A tabela da dissertação depende de leitura manual.
- Duas execuções legadas do mesmo modelo/amostra discordam em 6/50 (GPT) e 5/50 (Gemini)
  categorias: o ruído execução-a-execução é da mesma ordem de grandeza das diferenças que o piloto
  quer detectar.

## 2. Qualidade das imagens (Etapa B; 40 PH² × 40 HAM10000)

Direção consistente nos dois estratos de classe (Holm p < 0,05; δ de Cliff grande):

| Indicador | PH² (med.) | HAM (med.) | Leitura cautelosa |
|---|---|---|---|
| Megapixels nativos | 0,44 | 0,27 | PH² ~1,6× mais pixels (δ = 1,0) |
| Contraste de luminância (DP) | 38 | 26 | PH² maior |
| Faixa dinâmica p99−p1 | 163 | 116 | PH² maior |
| Laplaciano global (512 px) | 462 | 63 | PH² maior — **confundido com ruído** |
| Ruído estimado na pele | 2,2 | 1,0 | PH² maior (BMP sem compressão × JPEG) |
| Força de borda na faixa da lesão | 5,1 | 2,8 | PH² maior |
| Vinheta (cantos/centro) | 0,64 | 1,18 | PH² com cantos escuros |
| Proxy de pelos | 6,3 % | 0,9 % | PH² maior |
| Contraste lesão × pele (ΔE) | 24 | 22 | sem diferença apreciável |
| Luminância média | 153 | 152 | sem diferença apreciável |

Pontos que **não** devem ser lidos como "PH² é melhor": nitidez por Laplaciano e ruído
sobem juntos (a compressão JPEG do HAM atenua alta frequência *e* ruído); vinheta e pelos são
artefatos e estão *maiores* no PH²; indicadores de sinal não medem visibilidade clínica.

Achados mais relevantes para H1/H3 (dentro de cada dataset):

- **PH²**: melanomas ocupam 69 % do quadro × 19 % dos nevos (δ = 0,73, Holm p = 0,003) e são mais
  escuros (δ = −0,64) e de menor contraste lesão/pele (δ = −0,63). Há, portanto, pistas de
  enquadramento/exposição associadas à classe — um modelo (ou VLM) poderia usá-las sem análise
  dermatoscópica.
- **HAM10000**: sem diferenças apreciáveis de enquadramento entre classes (δ ≈ 0,2, n.s.), mas a
  **fonte de aquisição está associada à classe** (16/20 nevos e 1/20 melanomas do `vidir_molemax`).

## 3. Situação das hipóteses

| Hipótese | Evidência até aqui | Falta |
|---|---|---|
| **H1** qualidade | *Apoia parcialmente*: PH² tem mais pixels, maior contraste/faixa dinâmica e melhor definição de borda; *enfraquece/ambíguo*: nitidez ligada a ruído, mais artefatos no PH² | Relação dos indicadores com os erros (exige inferência) |
| **H2** complexidade | **Indeterminada**: HAM não tem anotações dermatoscópicas; nevos PH² são 50 % atípicos, nevos HAM 80 % *follow-up* na amostra. Dificuldade clínica não é inferível da imagem | Erros por critério/estrutura (PH² tem anotações; HAM só dx/localização) |
| **H3** amostra | *Apoia*: origem do rótulo diferente; 12/40 melanomas PH² fora do rótulo legado; confusão fonte × classe no HAM; regra de leitura altera o resultado. Prevalência já controlada (20/20) | Resultado do piloto estratificado por fonte |

## 4. Limitações

n = 20 por grupo (ICs largos); 1 inferência por imagem; ruído execução-a-execução ≈ 10 %; leitura
manual da dissertação não reproduzível (política programática declarada, com sensibilidade);
indicadores computacionais ≠ adequação clínica; máscaras de origens distintas; imagens HAM lidas
da cópia ISIC 2019; versões de modelo/temperatura do legado não verificáveis; sem ground truth do
ISIC 2019 local; resultado HAM10000 prévio não documentado.

## 5. Decisão pendente do usuário

Autorizar (ou não) a inferência: 640 chamadas LLM (320 por modelo; estimativa de ≈ 0,69 M tokens
de entrada e 0,12 M de saída por modelo, premissas em `cost_estimate.json`), definir
`OPENAI_API_KEY`/`GOOGLE_API_KEY`, e informar os preços para o cálculo em US$. Alternativa de
custo menor: um só modelo (`--models gpt-5.4-nano`).
