# Risco metodológico do ISIC 2019 (avaliação documental, sem inferência)

Dados: `audit/equivalence_audit.json → isic_composition`. Sem inferência em massa e sem
alterar o protocolo da comparação com a EfficientNetV2-S.

## Fatos verificados em disco

| Fato | Valor |
|---|---|
| Imagens no `ISIC_2019_Training_Input` local | 25 331 |
| Fontes declaradas (`ATTRIBUTION.txt`) | BCN_20000 (Hospital Clínic de Barcelona), HAM10000 (ViDIR/Viena), MSK |
| IDs do HAM10000 contidos no ISIC 2019 | 10 015 / 10 015 (**39,5 %** do ISIC 2019) |
| ISIC 2019 fora do HAM10000 (BCN + MSK, sem separação local) | 15 316 |
| ISIC 2018 Task 1 (treino) | 2 594 imagens; **772** também estão no ISIC 2019; 0 no HAM10000 |
| `ISIC_2019_Training_GroundTruth.csv` / `Metadata.csv` | **ausentes** (composição por classe e por fonte BCN × MSK não é verificável localmente) |
| HAM10000: fonte de aquisição × classe | fortemente associadas (ver `audit.md` §3) |

Observações sobre BCN_20000/MSK (resolução, localizações difíceis, recortes) vêm da
literatura, **não** de arquivos locais, e não devem ser citadas como verificadas.

## Perguntas

**O desempenho no HAM10000 ameaça a validade do experimento principal no ISIC 2019?**
Não é possível afirmar com os dados atuais: (i) não há resultado HAM10000 documentado deste
pipeline; (ii) o piloto de inferência ainda não foi executado. Estruturalmente o risco é
real porque 39,5 % do ISIC 2019 é HAM10000 e o pipeline foi validado só no PH² (uma fonte,
rótulo clínico, imagens de alta resolução). O risco é de **validade externa e de
confundimento por fonte**, não de erro de implementação.

**Quais fontes podem se comportar de forma diferente?**
HAM10000 (e suas 4 subfontes internas, sobretudo `vidir_molemax`, quase só nevos), BCN_20000 e
MSK. Qualquer diferença entre classes que coincida com diferenças de fonte é indistinguível de
sinal diagnóstico sem estratificação.

**Quais análises por origem seriam necessárias?**
1. Resultados (sens., espec., F1, acurácia balanceada, matriz) **por fonte** (BCN/HAM/MSK e subfonte
   HAM) e por classe, com ICs; relatar também a composição de classes por fonte.
2. Análise intra-fonte (melanoma × nevo dentro da mesma fonte) para separar sinal diagnóstico de
   sinal de aquisição; reportar fontes com uma só classe como não informativas.
3. Agrupar por `lesion_id` (HAM10000 tem múltiplas imagens por lesão) em qualquer divisão/IC.
4. Sinalizar/excluir da avaliação do segmentador as 772 imagens do ISIC 2018 Task 1 presentes
   no ISIC 2019, se a U-Net foi treinada nelas (vazamento de treino de segmentação).
5. Obter `GroundTruth` e `Metadata` oficiais do ISIC 2019 e registrar a regra mel × nevo
   (MEL × NV) antes de qualquer execução.

**O piloto oferece evidência suficiente para avançar?**
Hoje **não**: o piloto de inferência não foi executado e, quando for, com 20 por classe os ICs de
Wilson têm largura da ordem de ±0,2, só distinguindo diferenças grandes (≳0,3). Recomendação:
**avançar condicionado a** (a) executar o piloto autorizado e (b) fixar de antemão a análise por
fonte acima. Um resultado PH² ≈ HAM dentro dos ICs não prova equivalência; uma queda nítida no
HAM exigiria, antes do ISIC 2019, a verificação intra-fonte (item 2).
