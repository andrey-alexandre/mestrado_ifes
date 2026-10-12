# ADR-001 — Seleção da arquitetura e do protocolo experimental do baseline supervisionado

**Status:** Aceito  
**Data:** 12/10/2026  
**Projeto:** Dissertação de Mestrado — Detecção e Diagnóstico de Melanoma com Sistema Multiagente Baseado em VLMs  
**Categoria:** Decisão arquitetural e metodológica

## 1. Contexto

A dissertação investiga o desempenho e a consistência diagnóstica de um sistema multiagente baseado em modelos de visão e linguagem (VLMs) para identificação de melanoma em imagens dermatoscópicas.

A metodologia original utilizava uma ResNet-50 como baseline supervisionado, treinada com imagens do dataset PH², composto por apenas 200 imagens, incluindo 40 melanomas.

Essa abordagem recebeu críticas em razão do tamanho reduzido da amostra de treinamento, do risco de sobreajuste e da capacidade limitada de generalização.

Adicionalmente, foi questionada a representatividade da ResNet-50 como referência contemporânea de classificação supervisionada de melanoma.

Diante dessas limitações, decidiu-se revisar a arquitetura e o protocolo experimental do baseline, preservando o escopo principal da dissertação: comparar um classificador supervisionado especializado com o sistema multiagente, utilizando as mesmas imagens e os mesmos rótulos de referência.

## 2. Decisão

**Adotar a EfficientNetV2-S como arquitetura definitiva do baseline supervisionado, treinada no ISIC 2019 para classificação binária de melanoma versus não melanoma.**

O experimento utilizará uma única divisão fixa do dataset, reservando aproximadamente 20 mil imagens para desenvolvimento e 5 mil para avaliação final.

O mesmo conjunto de avaliação será utilizado pelo classificador supervisionado e pelo sistema multiagente.

Não serão realizados experimentos adicionais de comparação entre arquiteturas, transferência direta entre datasets ou fine-tuning no PH².

A EfficientNetV2-S será inicializada com pesos pré-treinados no ImageNet e adaptada à tarefa binária mediante treinamento supervisionado no conjunto de desenvolvimento do ISIC 2019.

## 3. Fundamentação teórica

### 3.1. Limitações da ResNet-50

He et al. (2016) introduziram a arquitetura ResNet, demonstrando a eficácia das conexões residuais para treinamento de redes profundas.

Embora permaneça uma arquitetura amplamente reconhecida e utilizada como baseline, a ResNet-50 não representa, isoladamente, o estado da arte contemporâneo em classificação de lesões dermatoscópicas.

Estudos posteriores demonstraram resultados competitivos de arquiteturas EfficientNet, EfficientNetV2 e Vision Transformers.

### 3.2. Evidências em classificação dermatoscópica

Gessert et al. (2020) apresentaram uma solução baseada em ensembles de EfficientNet que obteve o primeiro lugar na tarefa de classificação por imagens do ISIC 2019.

Combalia et al. (2022) analisaram modelos submetidos ao desafio ISIC 2019 e identificaram o uso de ensembles de EfficientNet entre os sistemas mais bem classificados, destacando também questões de generalização.

Pacal et al. (2024) compararam arquiteturas como ResNet-50, EfficientNetV2-M e Swin Transformer no ISIC 2019. Os resultados reportados incluem acurácia de 75,10% para ResNet-50, 82,73% para EfficientNetV2-M e 84,92% para Swin-Base, considerando a classificação multiclasse.

Esses resultados demonstram que existem alternativas competitivas à ResNet-50. Entretanto, não devem ser interpretados como estimativas de desempenho para a tarefa binária proposta nesta dissertação.

### 3.3. Justificativa da EfficientNetV2-S

Tan e Le (2021) propuseram a família EfficientNetV2 com foco em maior eficiência de treinamento e melhor relação entre desempenho e custo computacional.

A variante EfficientNetV2-S foi selecionada pelos seguintes fatores:

- Arquitetura contemporânea e consolidada na literatura de visão computacional.
- Relação favorável entre capacidade de representação e custo computacional.
- Disponibilidade de implementações e pesos pré-treinados.
- Viabilidade de treinamento supervisionado sem ensembles ou buscas extensivas de arquitetura.
- Coerência com as evidências de competitividade da família EfficientNet em classificação dermatoscópica.

A seleção da variante S constitui uma decisão de engenharia fundamentada em eficiência e viabilidade, não uma demonstração de superioridade empírica sobre todas as demais arquiteturas.

A EfficientNetV2-S será descrita como um **baseline supervisionado moderno e competitivo**, e não como o estado da arte absoluto.

## 4. Dataset e definição da tarefa

Será utilizado o ISIC 2019, contendo 25.331 imagens dermatoscópicas de treinamento, distribuídas em oito categorias diagnósticas.

A tarefa será definida como classificação binária:

| Rótulo | Definição |
|---|---|
| 1 | Melanoma (MEL) |
| 0 | Qualquer diagnóstico diferente de melanoma |

Todas as categorias não melanoma serão agrupadas na classe negativa, incluindo outras neoplasias malignas.

Consequentemente, a classificação não será denominada *maligno versus benigno*, uma vez que a classe negativa poderá conter outras condições malignas.

A tarefa será denominada **melanoma versus não melanoma**.

Essa definição deverá ser aplicada de maneira idêntica ao baseline supervisionado e ao sistema multiagente.

## 5. Protocolo experimental

### 5.1. Particionamento

O ISIC 2019 será dividido em três subconjuntos:

| Subconjunto | Finalidade |
|---|---|
| Treinamento | Ajuste dos parâmetros da EfficientNetV2-S |
| Validação | Seleção de hiperparâmetros, épocas e limiar de classificação |
| Teste | Avaliação final compartilhada entre os sistemas |

O conjunto de teste conterá aproximadamente 5 mil imagens. As demais serão destinadas ao desenvolvimento do modelo supervisionado.

O particionamento deverá:

- Ser estratificado pelo diagnóstico binário.
- Evitar que imagens de uma mesma lesão apareçam em subconjuntos distintos, quando identificadores estiverem disponíveis.
- Utilizar semente aleatória fixa e registrar os identificadores das imagens.
- Permanecer imutável após sua definição.

O tamanho exato de cada subconjunto será documentado após a divisão.

### 5.2. Treinamento supervisionado

A EfficientNetV2-S utilizará pesos pré-treinados no ImageNet.

A camada de classificação será adaptada para produzir a probabilidade da classe melanoma.

O treinamento ocorrerá exclusivamente com imagens do subconjunto de treinamento.

O subconjunto de validação será utilizado para seleção de hiperparâmetros, controle de sobreajuste, seleção de checkpoint e definição do limiar de classificação.

O desbalanceamento entre classes deverá ser considerado durante o treinamento e documentado na metodologia.

Nenhuma informação do conjunto de teste poderá ser utilizada para ajuste do modelo.

### 5.3. Avaliação do sistema multiagente

O sistema multiagente será executado uma única vez sobre cada imagem do conjunto de teste.

Serão mantidos os agentes diagnósticos baseados nos critérios ABCD, Menzies e Seven-Point Checklist, além do agente sumarizador.

Quando necessária, a segmentação será realizada pelo módulo previamente definido na metodologia, sem utilizar máscaras ground truth indisponíveis no conjunto de teste.

As mesmas imagens e os mesmos rótulos de referência serão utilizados por ambos os sistemas.

As saídas individuais dos agentes e a classificação consolidada deverão ser armazenadas para análise posterior.

### 5.4. Métricas de avaliação

A comparação principal utilizará:

- Sensibilidade para melanoma.
- Especificidade para não melanoma.
- Precisão.
- F1-score.
- Balanced accuracy.
- ROC-AUC.
- Matriz de confusão.

Serão calculados intervalos de confiança de 95%, preferencialmente por bootstrap estratificado e pareado sobre as predições armazenadas.

Também serão analisados os erros por categoria diagnóstica original, especialmente para identificar possíveis limitações do sistema multiagente diante de lesões não melanocíticas.

A avaliação da consistência e da qualidade das justificativas dos agentes permanecerá uma dimensão separada da comparação de desempenho classificatório.

Uma única execução por imagem permite avaliar concordância entre agentes e modelos distintos, mas não permite estimar a repetibilidade intra-modelo em execuções independentes.

## 6. Alternativas consideradas

| Alternativa | Decisão | Justificativa |
|---|---|---|
| ResNet-50 treinada no PH² | Rejeitada | Conjunto de treinamento reduzido e risco de sobreajuste |
| ResNet-50 treinada no ISIC 2019 | Não selecionada | Arquitetura consolidada, mas existem alternativas contemporâneas competitivas |
| EfficientNet-B4 | Não selecionada | Alternativa válida, porém optou-se pela geração EfficientNetV2 |
| EfficientNetV2-S | **Selecionada** | Equilíbrio entre modernidade, eficiência e viabilidade |
| Swin Transformer | Não selecionada | Arquitetura competitiva, mas sem benefício comprovado para justificar sua adoção neste escopo |
| PanDerm e outros modelos fundacionais | Fora do escopo | Introduziriam diferenças de pré-treinamento e possíveis riscos de sobreposição com dados de avaliação |
| Ensemble de classificadores | Fora do escopo | Aumentaria a complexidade sem atender a uma necessidade central da dissertação |

## 7. Consequências da decisão

### Benefícios esperados

1. Redução da dependência de um dataset com apenas 200 imagens.
2. Comparação direta entre o baseline supervisionado e o sistema multiagente.
3. Utilização de uma arquitetura supervisionada contemporânea.
4. Eliminação de experimentos adicionais de adaptação entre PH² e ISIC 2019.
5. Ampliação da amostra de avaliação e maior precisão estatística das estimativas.
6. Maior reprodutibilidade mediante particionamento fixo e armazenamento das predições.

### Limitações e riscos

1. A EfficientNetV2-S não representa necessariamente o melhor classificador supervisionado disponível.
2. O ISIC 2019 apresenta desbalanceamento entre categorias diagnósticas.
3. A classe não melanoma reúne condições clínicas heterogêneas.
4. Os critérios ABCD, Menzies e Seven-Point Checklist foram desenvolvidos principalmente para avaliação de lesões melanocíticas, o que pode limitar sua aplicabilidade às demais categorias.
5. O conjunto de teste pertence à mesma coleção de origem do treinamento, de modo que o experimento não demonstra generalização para uma instituição ou dataset externo.
6. A ausência de máscaras ground truth para todas as imagens pode introduzir erros provenientes do módulo de segmentação.
7. A execução única dos agentes não fornece evidência de estabilidade entre múltiplas execuções do mesmo modelo.

## 8. Critérios de aceitação

A implementação será considerada conforme esta decisão quando:

- [ ] A EfficientNetV2-S estiver implementada com pesos pré-treinados e saída binária.
- [ ] O particionamento do ISIC 2019 estiver documentado e versionado.
- [ ] Não houver sobreposição identificada de lesões entre treinamento, validação e teste.
- [ ] O treinamento e a seleção de parâmetros não utilizarem imagens ou rótulos do teste.
- [ ] A classificação binária for idêntica para os dois sistemas.
- [ ] A EfficientNetV2-S e o sistema multiagente forem avaliados nas mesmas imagens.
- [ ] As predições, probabilidades e identificadores forem armazenados.
- [ ] As métricas e intervalos de confiança forem calculados sobre as predições do conjunto de teste.
- [ ] As limitações de generalização e escopo clínico forem explicitadas na dissertação.

## 9. Referências

1. HE, K. et al. Deep Residual Learning for Image Recognition. *CVPR*, 2016. https://doi.org/10.1109/CVPR.2016.90

2. GESSERT, N. et al. Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data. *MethodsX*, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7150512/

3. TAN, M.; LE, Q. EfficientNetV2: Smaller Models and Faster Training. *ICML*, 2021. https://arxiv.org/abs/2104.00298

4. COMBALIA, M. et al. Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images: the 2019 International Skin Imaging Collaboration Grand Challenge. *The Lancet Digital Health*, 2022. https://doi.org/10.1016/S2589-7500(22)00021-8

5. PACAL, I. et al. Enhancing Skin Cancer Diagnosis Using Swin Transformer: A Comparative Study of Advanced Neural Network Models. *Journal of Digital Imaging*, 2024. https://doi.org/10.1007/s10278-024-01140-8

6. ISIC. ISIC 2019 Challenge — Skin Lesion Analysis Towards Melanoma Detection. https://challenge.isic-archive.com/landing/2019/

---

## 10. Decisão final

**Fica estabelecida a EfficientNetV2-S como arquitetura definitiva do baseline supervisionado da dissertação.**

O treinamento utilizará o ISIC 2019, com classificação binária melanoma versus não melanoma.

A avaliação será realizada em um conjunto fixo de aproximadamente 5 mil imagens, compartilhado integralmente com o sistema multiagente.

A comparação terá como finalidade investigar o desempenho classificatório relativo dos sistemas, preservando a análise de consistência diagnóstica e de justificativas como contribuições específicas da abordagem multiagente.

Não serão realizadas comparações adicionais entre arquiteturas supervisionadas, salvo se forem identificadas falhas metodológicas que comprometam a validade do baseline escolhido.