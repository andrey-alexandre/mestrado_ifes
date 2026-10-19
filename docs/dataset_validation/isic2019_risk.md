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
Não pelos dados do piloto: sob protocolo controlado o Gemini não mostrou queda no HAM10000 (melanoma × nevo), e o resultado HAM10000 anterior não está documentado. Estruturalmente o risco é
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

**O piloto oferece evidência suficiente para avançar?** (atualizado após a inferência, 2026-10-19)
Sim, para avançar com condições. Sob protocolo controlado, o Gemini-3.1-flash-lite teve acurácia
balanceada 0,90 (PH²) × 0,85 (HAM10000), Δ 0,05 [−0,10; 0,20]; nenhuma diferença foi detectável. O
GPT-5.4-nano não discrimina em nenhum dos dois (≈ 0,5). Ressalvas: n = 20 por classe só exclui
diferenças grandes; a especificidade no HAM se apoia em nevos de *follow-up* da fonte `vidir_molemax`
(2/16 erros × 2/4 nas demais fontes); os agentes reportam menos estruturas dermatoscópicas nos
melanomas do HAM. As análises por fonte acima continuam obrigatórias no ISIC 2019. Detalhes em
`report.md`.
