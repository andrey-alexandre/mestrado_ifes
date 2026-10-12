# Diagnóstico de Lesões Cutâneas com Multiagentes, RAG e Baseline CNN

Este repositório contém um pipeline experimental para diagnóstico de lesões cutâneas utilizando:

- segmentação de imagem com U-Net;
- suporte a datasets ISIC e PH2;
- uso de agentes de IA em LangGraph para análise multimodal;
- recuperação de conhecimento médico via RAG (FAISS + textos extraídos de PDFs);
- avaliação automática dos resultados por métricas comparando o diagnóstico final com o rótulo real;
- baseline de referência baseado em ResNet-50 treinado no PH2.

A estrutura foi organizada para permitir execução do pipeline completo, avaliação dos resultados e treinamento de um modelo clássico de classificação.

## Objetivo do projeto

O projeto visa explorar um fluxo de diagnóstico assistido por IA para lesões cutâneas, combinando:

1. pré-processamento e segmentação da lesão;
2. análise clínica por múltiplos agentes especializad os (ABCD, Menzies, SPCL e validação final);
3. recuperação de contexto médico em uma base documental por RAG;
4. comparação com um baseline profundo em PyTorch.

## Estrutura do repositório

```text
.
├── config/
│   └── config.yaml                 # configurações gerais do pipeline e do treinamento
├── PH2/                           # dataset PH2 em estrutura de diretórios
├── scripts/
│   ├── run_pipeline.py             # executa o pipeline completo
│   ├── run_train.py                # wrapper para execução do pipeline com overrides
│   ├── run_eval.py                 # avalia um arquivo JSON de resultados
│   └── train_resnet50_ph2.py      # treinamento do baseline ResNet-50 no PH2
├── src/
│   ├── agents/                     # agentes LangGraph para análise da lesão
│   ├── data/                       # carregamento de datasets e preprocessamento
│   ├── evaluation/                 # leitura e métricas dos resultados
│   ├── models/                     # segmentação U-Net e helpers de imagem
│   └── pipeline/                   # execução do pipeline de inferência
├── analyze_llm_results.py          # scripts de análise agregada dos resultados de LLM
├── requirements.txt                # dependências Python
├── README.md                       # documentação do projeto
├── PH2_dataset.txt                 # metadados do PH2
├── results.csv                     # resultados agregados
├── results_stefani.csv             # resultados de referência/experimentais
├── analysis_*.csv                  # resumos e detalhamentos de análise
└── LangChain.ipynb                 # notebook de prototipagem/experimentos
```

## Como funciona o fluxo principal

### 1) Pré-processamento e RAG

O pipeline inicia com a criação da base de conhecimento médica:

- extração de textos a partir de PDFs em um zip definido em `config/config.yaml`;
- concatenação desses textos em `medical_texts.txt`;
- divisão em chunks e indexação com FAISS via embeddings SentenceTransformer;
- armazenamento em um retriever que será consultado pelos agentes durante a análise.

A lógica está em:

- `src/data/preprocess.py`
- `scripts/run_pipeline.py`

### 2) Carregamento do dataset

O projeto aceita dois cenários:

- `isic`: usa CSV de metadados e imagens `.jpg` em diretório local;
- `ph2`: usa a estrutura de diretórios do PH2 com imagens `.bmp` e máscaras de lesão.

Os loaders estão em:

- `src/data/dataset.py`

### 3) Segmentação da lesão

No modo `model`, o pipeline carrega um modelo U-Net pré-treinado e segmenta a imagem em tempo real.

No modo `precomputed`, ele usa a máscara já existente no dataset PH2.

Arquivo relevante:

- `src/models/model.py`
- `src/agents/agents.py`

### 4) Agentes de diagnóstico

O grafo LangGraph executa múltiplos agentes sobre a imagem segmentada:

- agente ABCD;
- agente Menzies;
- agente SPCL;
- agente de validação/finalização integrado ao RAG.

Esses agentes montam prompts multimodais com a imagem e contexto médico recuperado.

Arquivos relevantes:

- `src/agents/agents.py`
- `src/agents/schemas.py`

### Extração determinística de evidências

O pacote de Feature Extraction inclui componentes determinísticos e evidências brutas cientificamente delimitadas. Para ABCD-B, o método ativo é uma operacionalização autoral guiada por Kasmi e Mokrani (2016): máscara binária GT, contorno 2D ordenado, setores de 45 graus alinhados ao eixo principal da lesão, spline cúbica suavizada adimensional, RMSE geométrico, normalização por `L_smooth` e limiar fixo `E_smooth > 0.05`. A implementação é uma proxy geométrica de irregularidade de borda, não uma reprodução exata do artigo nem uma medida direta de transição pigmento-pele.

Detalhes metodológicos: `docs/abcd_border_methodology.md`.

### 5) Avaliação

Após a execução, o JSON produzido pelo pipeline é lido por `src/evaluation/evaluate.py`, que:

- imprime os diagnósticos de cada imagem;
- calcula métricas binárias como acurácia, sensibilidade e especificidade;
- converte o diagnóstico final em label binário para comparação com o target real.

## Configuração

O arquivo principal de configuração está em [config/config.yaml](config/config.yaml).

Ele controla:

- caminhos de datasets e saída;
- configuração da U-Net;
- escolha do modelo LLM;
- parâmetros do RAG;
- dataset em uso (`isic` ou `ph2`);
- opções de treinamento do baseline ResNet-50.

Principais seções:

- `paths`: diretórios e arquivos de entrada/saída;
- `segmentation`: modo e parâmetros da segmentação;
- `llm`: modelo de linguagem usado;
- `rag`: configuração do FAISS e split dos documentos;
- `pipeline`: número de imagens, seed, dataset e balanceamento;
- `training`: hiperparâmetros do ResNet-50.

## Requisitos

O projeto depende de:

- Python 3.10+;
- bibliotecas listadas em [requirements.txt](requirements.txt);
- chaves de API para OpenAI ou Google, dependendo do modelo LLM escolhido;
- opcionalmente, execução local com Ollama para modelos do tipo `llava:13b` ou similares.

Instalação:

```bash
pip install -r requirements.txt
```

Se for usar modelos cloud, configure as variáveis de ambiente antes da execução:

```bash
export OPENAI_API_KEY="sua_chave_openai"
export GOOGLE_API_KEY="sua_chave_google"
```

## Execução do pipeline

### Pipeline completo

```bash
python scripts/run_pipeline.py --skip-preprocess
```

### PH2 com máscara pré-computada

```bash
python scripts/run_pipeline.py --dataset ph2 --segmentation-mode precomputed --skip-preprocess
```

### Limitando o número de imagens

```bash
python scripts/run_pipeline.py --dataset isic --max-images 20 --balanced
```

### Troca do modelo LLM

```bash
python scripts/run_pipeline.py --model-name gemini-1.5-pro
```

### Ignorando etapa de RAG e usando arquivos já existentes

```bash
python scripts/run_pipeline.py --skip-preprocess
```

## Execução do pipeline com overrides rápidos

Existe também um wrapper mais flexível em [scripts/run_train.py](scripts/run_train.py), que aceita argumentos como:

```bash
python scripts/run_train.py --dataset ph2 --segmentation-mode precomputed --max-images 20 --model-name gpt-4o-mini
```

## Avaliação dos resultados

Após a geração do arquivo JSON de saída, use o script de avaliação:

```bash
python scripts/run_eval.py --results results/llm_result_gpt-5.4-nano_20260609.json --verbose
```

Esse comando lê o JSON, imprime cada diagnóstico e calcula métricas de desempenho.

## Baseline ResNet-50 no PH2

Além do pipeline multiagente, o repositório inclui o treinamento de um baseline CNN para classificação binária do PH2:

```bash
python scripts/train_resnet50_ph2.py
```

Esse script:

- carrega imagens BMP do PH2;
- detecta melanoma vs demais casos;
- faz split estratificado 70/30;
- treina ResNet-50 com augmentations e normalização;
- salva métricas de treino e teste em diretório de saída.

Todos os hiperparâmetros são lidos de [config/config.yaml](config/config.yaml), sob a seção `training`.

## Saída esperada

Os scripts produzem arquivos JSON em diretórios de saída, geralmente em `results/`, com nome no formato:

```text
llm_result_<model_name>_<YYYYMMDD>.json
```

Esses arquivos contêm:

- imagem processada;
- diagnóstico ABCD;
- diagnóstico Menzies;
- diagnóstico SPCL;
- validação final;
- relatório RAG;
- label real da imagem.

## Observações importantes

- O projeto foi concebido para pesquisa e experimentação; não substitui avaliação clínica profissional.
- O uso do modelo de segmentação exige um checkpoint U-Net em `paths.model_path` no modo `model`.
- Para PH2, o arquivo `PH2_dataset.txt` deve conter diagnósticos histológicos em formato pipe-delimited.
- O pipeline é sensível ao ambiente de execução e às chaves de API do provedor de LLM escolhido.

## Fluxo recomendado para começar

1. Instale as dependências.
2. Ajuste os caminhos e modelos em [config/config.yaml](config/config.yaml).
3. Rode um teste rápido do pipeline em um subconjunto de imagens.
4. Examine a saída JSON e calcule as métricas com `scripts/run_eval.py`.
5. Se desejar, treine o baseline ResNet-50 com `scripts/train_resnet50_ph2.py`.

## Referências internas

Principais arquivos de execução:

- [scripts/run_pipeline.py](scripts/run_pipeline.py)
- [scripts/run_train.py](scripts/run_train.py)
- [scripts/run_eval.py](scripts/run_eval.py)
- [scripts/train_resnet50_ph2.py](scripts/train_resnet50_ph2.py)
- [src/data/dataset.py](src/data/dataset.py)
- [src/agents/agents.py](src/agents/agents.py)
- [src/evaluation/evaluate.py](src/evaluation/evaluate.py)
- [src/data/preprocess.py](src/data/preprocess.py)
