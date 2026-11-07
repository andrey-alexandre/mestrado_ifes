# Infraestrutura de Amostragem das 3.000 Lesões (HAM10000) — Tarefa 12

**Status: implementada, aguardando execução do pesquisador.** Nenhum código,
teste ou script desta tarefa foi executado pelo assistente — nem o
script de amostragem, nem `pytest`, nem qualquer comando git. Tudo abaixo é
o que foi escrito, pronto para o pesquisador rodar.

## 1. O que foi implementado

- `src/data_audit/sampling.py` — módulo puro de amostragem (sem I/O de
  execução além de ler o CSV de metadados e, opcionalmente, um arquivo de
  exclusões):
  - `load_ham10000_metadata` — mesmo parsing (`core.parse_csv_text`) usado
    por `adapters.audit_ham10000`.
  - `check_lesion_population` — reaproveita `core.find_missing_or_conflicting`
    (não reimplementa); também reporta valores de `dx` fora das 7 categorias
    esperadas (akiec, bcc, bkl, df, mel, nv, vasc), sem excluir nada por
    conta própria.
  - `build_lesion_table` — colapsa linhas por imagem em uma entrada por
    `lesion_id` (`dx` + lista de `image_id`s). Só deve ser chamada depois de
    `check_lesion_population` reportar `ok=True`.
  - `apply_exclusions` — remove `lesion_id`s de uma lista antes da
    amostragem; reporta quantos foram efetivamente removidos vs. ids
    desconhecidos (não lança erro se um id não existir, mas também não o
    esconde).
  - `compute_quotas_largest_remainder` — método dos maiores restos
    (Hamilton): cota exata, piso, resto, distribuição das unidades extras
    por maior resto (empate por nome de categoria). Levanta `ValueError` se
    `target_n` exceder a população total — **não** reduz o alvo
    silenciosamente.
  - `sample_lesions` — sorteio sem reposição por categoria (`random.Random(seed).sample`)
    e, para cada lesão sorteada, sorteio determinístico de uma única imagem
    dela. Levanta `ValueError` se uma cota exceder o pool da categoria.
  - `build_sampling_summary_markdown` — renderiza `sampling_summary.md`
    (formatação pura, sem lógica de amostragem).

- `scripts/run_ham10000_sampling.py` — CLI executável (ver seção 3).

- `tests/sampling/test_ham10000_sample.py` — 16 testes com população
  sintética fabricada (50 lesões fictícias em 4 categorias, `nv=28 mel=12
  bcc=6 bkl=4`, escolhida deliberadamente para não dividir exatamente por
  10, forçando o caminho de distribuição de unidades extras do método dos
  maiores restos). Cobre: integridade da população (`lesion_id` ausente,
  `dx` conflitante, `dx` inesperado), tabela de lesões, exclusões (ids
  conhecidos/desconhecidos/lista vazia), cálculo de cotas (caso exato e
  caso com resto, mais o `ValueError` de alvo > população), amostragem
  (determinismo por seed, respeito às cotas, sem repetir lesão, imagem
  sorteada pertence à própria lesão, alvo binário derivado corretamente,
  `ValueError` de cota > pool).

## 2. Comando de instalação de dependências

Nenhuma dependência nova foi introduzida — `sampling.py` e o script usam
apenas a biblioteca padrão (`random`, `csv`, `json`, `argparse`, `pathlib`)
e `src.data_audit.core`, já presente no projeto. Nenhum comando de
instalação é necessário.

## 3. Comando exato dos testes

```
python -m pytest tests/sampling/test_ham10000_sample.py -v
```

Executar da raiz do repositório (`C:\Projetos\IFES\mestrado_ifes`).
**Use `python -m pytest`, não o comando `pytest` isolado**: o repositório
não tem `pytest.ini`/`pyproject.toml` com configuração de testes (já
registrado na Tarefa 1), então nada garante que a raiz do repositório
esteja no `sys.path` para resolver `from src.data_audit import ...`.
`python -m pytest` insere o diretório atual (a raiz, se for daí que você
rodar o comando) no início do `sys.path` automaticamente; o executável
`pytest` isolado não faz isso. Isso vale para todos os testes existentes
em `tests/data_audit/`, não é uma particularidade desta tarefa.

## 4. Comando exato do script

```
python scripts/run_ham10000_sampling.py --output-dir experiments/sampling
```

- Diretório de execução: raiz do repositório.
- `--metadata-csv` (opcional): default `D:/Andrey/Mestrado/dataverse_files/HAM10000_metadata`
  (mesmo caminho real já usado em `config/dataset_inventory.yaml`).
- `--target-n` (opcional): default `3000`.
- `--seed` (opcional): default `20261106` (data de hoje como inteiro; valor
  fixo documentado, qualquer outro inteiro fixo funciona igual).
- `--exclusions-file` (opcional): CSV (coluna/linha `lesion_id`) ou JSON
  (lista de `lesion_id`) de lesões a excluir antes da amostragem. **Não
  informar** nesta execução — não há exclusão aprovada hoje (Tarefa 11
  encerrada sem confirmar nenhuma duplicata; ver
  `docs/reconstruction/08_visual_duplicate_audit_closure_report.md`).
- `--output-dir` (opcional): default `experiments/sampling/`.

**Entradas exigidas**: o arquivo `HAM10000_metadata` precisa existir no
caminho informado (ou no default).

**Saídas esperadas** (em `experiments/sampling/`):
- `sample_manifest.csv` — colunas `lesion_id,image_id,dx,binary_target`,
  uma linha por lesão sorteada (até 3.000, se a população permitir).
- `sampling_summary.md` — população elegível total, contagem por
  categoria, tabela completa do cálculo de cotas (maiores restos), e
  observações.
- `sampling_results.json` — tudo acima em JSON, mais `status` (
  `concluido_teste_nao_definitivo`, `populacao_insuficiente`, ou
  `populacao_invalida`), para leitura programática.

**Duração estimada**: poucos segundos (leitura de um CSV de ~10 mil linhas
e amostragem determinística em memória, sem rede nem imagens).

## 5. Arquivos a devolver para análise

1. `experiments/sampling/sampling_summary.md`
2. `experiments/sampling/sampling_results.json`
3. Saída completa do comando de testes (stdout do `python -m pytest -v`,
   ou o arquivo se preferir redirecionar:
   `python -m pytest tests/sampling/test_ham10000_sample.py -v > tests_output.txt`)
4. Saída completa (stdout) da execução do script.

Não há `sampling.log` separado — o script imprime progresso em stdout
(população lida, exclusões aplicadas, cotas finais, confirmação de
gravação de cada arquivo); redirecione para um arquivo se quiser devolvê-lo
como texto (`python scripts/run_ham10000_sampling.py --output-dir experiments/sampling > sampling_run_output.txt`).

## 6. Limitações, ambiguidades e decisões de convenção registradas

- **Semente default (`20261106`)**: é a data de hoje (2026-11-06) como
  inteiro, só para ter um valor fixo e documentado — não há significado
  estatístico nela. Qualquer semente fixa serve; o importante é que fique
  registrada (está, no `--seed` e no `sampling_summary.md`).
- **"Uma imagem por lesão" quando há mais de uma**: o sorteio da imagem
  usa um `random.Random(seed)` **separado** do usado para sortear as
  lesões (mesma semente, instâncias diferentes), para que mudar a ordem de
  chamadas de um sorteio não perturbe o outro. Isso é uma decisão de
  implementação, não um requisito explícito do protocolo — documentado
  aqui para que fique visível caso precise ser revisado.
- **Ids de exclusão desconhecidos**: `apply_exclusions` reporta (não
  levanta erro) quando um `lesion_id` da lista de exclusão não existe na
  população — decisão de não travar a execução por um id digitado errado,
  mas o relatório expõe isso explicitamente (`unknown_ids`) para não
  esconder o problema.
- **`dx` fora das 7 categorias esperadas**: `check_lesion_population`
  reporta (`unexpected_dx_values`) e marca a população como **inválida**
  (`ok=False`) se isso ocorrer — o script então NÃO amostra e grava
  `status: "populacao_invalida"` em `sampling_results.json`. Isso é uma
  decisão de segurança (não assumir que um valor inesperado é seguro de
  ignorar), não um comportamento pedido literalmente na tarefa — registrada
  aqui para revisão.
- **População insuficiente (< 3.000 lesões elegíveis)**: por requisito
  explícito da tarefa, o script não completa com outras imagens das
  mesmas lesões, nem reduz o alvo — grava `status: "populacao_insuficiente"`
  e para, sem gerar `sample_manifest.csv`.
- **Esta execução não congela a amostra definitiva**: tanto o script
  quanto `sampling_summary.md` repetem esse aviso explicitamente, por
  instrução direta da tarefa.
- **Teste de integração real não escrito**: os testes usam população
  100% sintética/fabricada (nenhum teste lê o `HAM10000_metadata` real) —
  por design, já que a tarefa pede dados "sintéticos/controlados" para os
  testes e reserva a leitura do CSV real para a execução do script pelo
  pesquisador.
