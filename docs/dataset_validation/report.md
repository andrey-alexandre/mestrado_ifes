# Relatório — investigação PH² × HAM10000 (piloto exploratório)

**Estado: concluído.** Inferência executada em 2026-10-18 (macOS, commit `bb11e66`, `agents.py`
idêntico ao do repositório; 2 modelos × 80 imagens, 640 chamadas, 0 falhas; manifesto com
fingerprint e hashes iguais ao aprovado). Avaliação refeita nesta máquina com os indicadores de qualidade.
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

## 3. Resultados diagnósticos (política primária: "indeterminada" = positiva)

| Modelo | Dataset | TN/FP/FN/TP | Sens. [IC95] | Espec. [IC95] | F1 | Acc. bal. |
|---|---|---|---|---|---|---|
| Gemini-3.1-flash-lite | PH² | 19/1/3/17 | 0,85 [0,64–0,95] | 0,95 [0,76–0,99] | 0,89 | 0,90 |
| Gemini-3.1-flash-lite | HAM10000 | 16/4/2/18 | 0,90 [0,70–0,97] | 0,80 [0,58–0,92] | 0,86 | 0,85 |
| GPT-5.4-nano | PH² | 2/18/1/19 | 0,95 [0,76–0,99] | 0,10 [0,03–0,30] | 0,67 | 0,53 |
| GPT-5.4-nano | HAM10000¹ | 1/17/2/18 | 0,90 [0,70–0,97] | 0,06 [0,01–0,26] | 0,65 | 0,48 |

¹ 2 respostas do GPT (nevos do HAM) ficaram `NAO_AVALIAVEL` ("padrão predominantemente benigno …
baixa suspeição" contém termos benignos e de suspeição) e foram excluídas; lidas como benignas
virariam 2 TN, sem mudar a conclusão.

**Diferença PH² − HAM10000** (exploratória; Newcombe/Fisher; bootstrap estratificado para acc. bal.):

| Modelo | Δ sens. | Δ espec. | Δ acc. bal. |
|---|---|---|---|
| Gemini | −0,05 [−0,27; 0,17], p = 1,0 | 0,15 [−0,07; 0,37], p = 0,34 | 0,05 [−0,10; 0,20] |
| GPT | 0,05 [−0,15; 0,26], p = 1,0 | 0,04 [−0,17; 0,25], p = 1,0 | 0,05 [−0,06; 0,17] |

Com a política alternativa ("indeterminada" = negativa) o Gemini fica com espec. 1,00 nos dois e
sens. 0,50 (PH²) × 0,35 (HAM) (Δ 0,15 [−0,15; 0,41], p = 0,52); o GPT perde quase toda a
sensibilidade (0,10 × 0,05). **Nenhuma diferença PH² × HAM é detectável neste piloto.**

### Reprodutibilidade
- Gemini: nas 17 imagens PH² que também estavam na execução legada (09/06), a categoria final
  coincide em **17/17**. GPT: 12/17 categorias e 16/17 na decisão binária.
- O comportamento legado é reproduzido: o GPT-5.4-nano responde "indeterminada" em ~80 % das imagens
  das duas classes nos dois datasets (sem poder discriminativo; acc. bal. ≈ 0,5); o Gemini discrimina.

### Onde estão os erros (exploratório)
- **Gemini, nevos do HAM**: 2/16 erros em `vidir_molemax` (quase todos *follow-up*) × 2/4 nas
  demais fontes; por confirmação, 2/16 *follow-up* × 2/3 histopatológicos. A especificidade no HAM
  vem sobretudo de nevos de acompanhamento, os presumivelmente mais banais. No PH²: 1/10 erro em
  nevos atípicos, 0/10 em comuns.
- **Critérios dos agentes (melanomas)**: ABCD e 7-Point dão escores parecidos nos dois datasets
  (Gemini ABCD 5,94 × 5,66; SPCL 4,65 × 4,80). A diferença está nas **estruturas do Menzies**: o
  Gemini aponta véu azul-esbranquiçado em 35 % dos melanomas PH² × 10 % HAM, múltiplas cores 25 % × 0 %,
  e nenhum pseudópode/streaming/despigmentação/pontos periféricos no HAM (média de características
  positivas 1,5 × 0,4; GPT 1,05 × 0,25). No HAM mais melanomas terminam "indeterminada" (11/20 × 7/20),
  o que só não vira perda de sensibilidade porque a política primária conta "indeterminada" como positiva.
- **Qualidade × erro**: Gemini — nenhum indicador associado ao erro após Holm (PH² só 4 erros, HAM 6).
  GPT — as associações (área da lesão, luminância etc.) são espúrias: como ele classifica quase tudo
  como positivo, "erro" ≈ "ser nevo", e os indicadores refletem a diferença de classe, não de qualidade.

## 4. Situação das hipóteses

| Hipótese | Evidência | Conclusão |
|---|---|---|
| **H1** qualidade | PH² com mais pixels, contraste e definição de borda (Etapa B). Melanomas HAM têm menos estruturas dermatoscópicas *reportadas* pelos agentes (véu, cores, Menzies). Nenhum indicador de qualidade se associa aos erros | **Fraca/indeterminada**: compatível com menor visibilidade de estruturas, mas sem efeito mensurável no desempenho final |
| **H2** complexidade | Erros do Gemini concentrados em nevos histopatológicos/não-*follow-up* do HAM e em nevo atípico do PH² (n mínimos). HAM sem anotações dermatoscópicas | **Indeterminada**; sinal compatível de que nevos excisados são mais difíceis |
| **H3** amostra/protocolo | Com prevalência, classes (mel × nv), entrada (máscara GT) e regra de leitura iguais, a diferença desaparece dentro dos ICs. No legado: rótulo PH² errado, leitura manual, entrada possivelmente diferente | **Mais apoiada**: a diferença relatada antes provavelmente vinha do protocolo (composição de classes, entrada/segmentação, regra de leitura), não das imagens do HAM em si |

## 5. Limitações

n = 20 por grupo (ICs largos); 1 inferência por imagem; ruído execução-a-execução ≈ 10 %; leitura
manual da dissertação não reproduzível (política programática declarada, com sensibilidade);
indicadores computacionais ≠ adequação clínica; máscaras de origens distintas; imagens HAM lidas
da cópia ISIC 2019; versões de modelo/temperatura do legado não verificáveis; sem ground truth do
ISIC 2019 local; resultado HAM10000 prévio não documentado.

## 6. Recomendação para o ISIC 2019

**Avançar**, com três condições (ver `isic2019_risk.md`):
1. Usar **só o Gemini-3.1-flash-lite** como sistema diagnóstico de referência. O GPT-5.4-nano não
   discrimina melanoma × nevo em nenhum dataset; mantê-lo só como comparação qualitativa.
2. Reportar métricas **por fonte** (BCN_20000 / HAM10000 e subfontes / MSK) e, no HAM, por tipo de
   confirmação: a especificidade observada se apoia em nevos de *follow-up* que podem não
   representar os nevos do ISIC 2019.
3. Fixar de antemão a política de leitura das respostas "indeterminadas" e reportar as duas: ela
   move a sensibilidade do Gemini de ~0,9 para ~0,4–0,5.

O piloto não mostra degradação no HAM10000 para melanoma × nevo, mas com n = 20 só exclui
diferenças grandes (≳ 0,3); diferenças moderadas (~0,15 em especificidade) seguem possíveis.
