# 12 — Remoção da inicialização incondicional de RAG dos entry points (A36)

Sessão de verificação real (2026-11-08), não reconstruída de memória. Esta é
a parte de A36 que fecha a remoção do crítico/RAG no nível dos pontos de
entrada — a remoção no grafo (`src/agents/agents.py::build_graph`) já havia
sido feita e verificada em
[11_multiagent_flow_documentation.md](11_multiagent_flow_documentation.md),
mas aquela entrega deixou registrado como pendência que `scripts/run_pipeline.py`
e `scripts/run_train.py` ainda construíam o índice RAG incondicionalmente.

## 1. O que existia antes

**`scripts/run_pipeline.py`** (antes desta correção, linhas ~136-152):

```python
logger.info("=== ETAPA 1: Pré-processamento ===")
if args.skip_preprocess:
    logger.info("Pulando extração de PDFs (--skip-preprocess ativo).")
else:
    prepare_medical_texts(...)

logger.info("=== ETAPA 2: Construção do RAG ===")
retriever = build_rag(...)
```

**`scripts/run_train.py`** (antes desta correção, linhas ~191-197):

```python
# 2. Construir RAG
retriever = build_rag(...)
```

Em ambos, `build_rag()` (`src/data/preprocess.py`) carrega um modelo de
embeddings (`SentenceTransformerEmbeddings`) e constrói um índice FAISS real
a partir de `medical_texts.txt` — isso rodava sempre, mesmo que o grafo ativo
não tivesse mais nenhum nó que consumisse esse retriever (o único consumidor,
`CriticalReviewAgent`, deixou de ser registrado em `build_graph()` desde a
remoção do A36 no grafo).

## 2. O que foi alterado

Em cada script, a construção foi extraída para uma função `_resolve_retriever()`
e tornada **opt-in via a nova flag `--build-rag`**:

- **Sem `--build-rag` (caminho oficial, padrão):** `_resolve_retriever()`
  retorna `None` imediatamente, sem chamar `build_rag()` nem
  `prepare_medical_texts()` — nenhum índice, embedding, extração de PDF ou
  credencial desse componente é construído ou importado. Este `None` é
  passado para `build_graph(retriever=None, ...)`, exatamente como o piloto
  `scripts/run_multiagent_flow_pilot.py` já havia confirmado ser seguro (o
  parâmetro `retriever` não é mais usado dentro de `build_graph` após a
  remoção do nó `critical_review`).

- **Com `--build-rag` (caminho histórico, preservado):** `_resolve_retriever()`
  chama `prepare_medical_texts()` (salvo `--skip-preprocess`, em
  `run_pipeline.py`) e `build_rag()` exatamente como antes, construindo um
  índice FAISS real. Nada do código de `src/data/preprocess.py` foi alterado
  ou removido — a função ficou disponível, só deixou de ser chamada por
  padrão.

Nenhum fallback/try-except foi introduzido: a ausência de RAG no caminho
oficial é uma omissão limpa (a função simplesmente não é chamada), não um
"tentar construir, falhar e seguir sem".

## 3. Verificação com respostas simuladas

`scripts/run_entrypoint_rag_pilot.py` (sem rede, sem credenciais) confirma,
para `run_pipeline` e `run_train`:

1. **Caminho oficial não constrói nada relacionado a RAG** — prova
   estrutural: `build_rag`/`prepare_medical_texts` são monkey-patchados para
   lançar `AssertionError` se forem chamados; o piloto reporta `ok=True` e
   `retriever_is_none=True` apenas se essas funções nunca forem invocadas no
   caminho oficial.
2. **Caminho histórico (`--build-rag`) continua funcional** — mesmas funções
   monkey-patchadas para devolver um sentinela fixo (não um índice real);
   confirma que `--build-rag` ainda as invoca.
3. **O grafo real termina no sumarizador com o `retriever=None` resolvido** —
   reaproveita a infra de dublê de LLM já validada em
   `run_multiagent_flow_pilot.py` (`_run_scenario`, `_ok`) para rodar
   `build_graph()` de ponta a ponta com esse valor e confirmar
   `final_report` ausente (`critical_review` nunca roda) e o sumarizador
   efetivamente invocado.

Testes unitários da lógica pura de `_resolve_retriever()` (sem importar o
grafo, sem I/O real) estão em `tests/entrypoints/test_rag_resolution.py`.

## 4. O que esta entrega NÃO cobre

- `results_stefani.csv` — tratado separadamente em **P014**: se o arquivo
  trouxer evidência explícita de execução do crítico, isso será registrado
  lá; caso contrário, o status será "execução histórica não comprovada".
  Esta correção dos entry points não depende desse resultado.
- Inferência real, HAM10000, checkpoint da CNN, S2 com máscaras reais,
  modelos/versões de VLM, matriz final de condições — fora de escopo, como
  nas entregas anteriores da etapa 5.
