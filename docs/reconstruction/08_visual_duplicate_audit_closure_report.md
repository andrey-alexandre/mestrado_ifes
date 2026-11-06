# Fechamento da Auditoria de Duplicatas Visuais — Revisão Manual Parcial — 2026-11-06

Encerramento da Tarefa 11 por decisão explícita do pesquisador: a revisão manual foi interrompida deliberadamente antes de cobrir a fila de 1.034 pares dimensionada, e este relatório documenta a cobertura REAL alcançada — não a cobertura planejada. Nenhuma revisão adicional foi feita depois dessa decisão; nenhuma exclusão de lesão ou imagem foi decidida ou implementada em nenhuma etapa desta tarefa.

## 1. Cobertura real da revisão manual, por categoria

| Categoria | Selecionados na fila | Revisados | Cobertura | Julgamento humano |
|---|---|---|---|---|
| `forte` | 34 | **34** | **100%** | completo |
| `forte_degenerada` | 500 | **250** | **50%** | parcial |
| `fraco` | 400 | **0** | **0%** | **nenhum** |
| `not_evaluable` | 100 | **0** | **0%** | **nenhum** |
| **Total da fila** | **1.034** | **284** | **27,5%** | — |

A fila de 1.034 pares foi **dimensionada por completo** (a amostragem — quais pares específicos entrariam na fila, com seed fixa `20261105`, reproduzível — está concluída e gravada em `experiments/data_audit/full_audit_review_queue.json`). A **revisão manual dos itens da fila não foi concluída**: 750 dos 1.034 pares selecionados (todo `fraco`, todo `not_evaluable`, e 250/500 de `forte_degenerada`) nunca foram abertos para julgamento humano.

## 2. Resultado da revisão (só os 284 pares efetivamente julgados)

| Categoria revisada | n julgados | `mesma_lesao` | `lesoes_diferentes` | `indeterminado` |
|---|---|---|---|---|
| `forte` | 34 | 0 | 34 | 0 |
| `forte_degenerada` | 250 | 0 | 248 | 2 |
| **Total** | **284** | **0** | **282** | **2** |

Fonte: `review_judgment` gravado por par em `experiments/data_audit/full_audit_review_queue.json` (campo preenchido apenas nos pares efetivamente abertos; `None` nos 750 pares selecionados e ainda não revisados — ver seção 3 sobre essa distinção).

## 3. Status dos pares não revisados — duas categorias distintas, não uma

Esta distinção é deliberada no código (`src/data_audit/visual_duplicate_full_run.py::build_review_queue`) e deve ser preservada neste relatório:

- **Pares selecionados para a fila de 1.034 mas não abertos para julgamento (750 pares: 250 `forte_degenerada` + 400 `fraco` + 100 `not_evaluable`)**: a seleção os marcou como prioritários justamente por suspeita (suporte geométrico total ou parcial, ou amostra estratificada por distância dHash). Eles **continuam sem nenhum julgamento humano** — `review_judgment = None` no registro. Não são "presumidamente distintos"; são "selecionados para revisão, mas não revisados".
- **Pares que nunca entraram na fila de 1.034 (420.162 dos 421.196 candidatos)**: nunca foram sequer selecionados para revisão. Só esses recebem a rotulagem `presumidamente_distinto` gravada automaticamente pelo código no momento da seleção — convenção que o próprio código distingue explicitamente de qualquer confirmação ("nunca a rotulagem 'confirmado'", testado em `test_build_review_queue_unselected_rows_are_presumed_distinct_never_confirmed`).

## 4. Conclusão sobre duplicatas

**Nenhuma duplicata confirmada nos 284 pares efetivamente revisados** (0 `mesma_lesao` em 284 julgamentos). Isto não é evidência de que `fraco` ou `not_evaluable` estejam livres de duplicatas — nenhum par dessas duas categorias foi aberto para julgamento, nem parcialmente (ver seção 5).

## 5. Composição completa dos 421.196 candidatos (Tarefa 7, reaproveitados sem regenerar)

Passada ORB+RANSAC + guarda de degenerescência (Tarefas 8/9/10) aplicada a TODOS os 421.196 candidatos dHash, sem nenhum gate de pHash:

| `final_strength` | n | % do total |
|---|---|---|
| `not_evaluable` | 320.167 | 76,03% |
| `fraco` | 99.951 | 23,73% |
| `forte_degenerada` | 1.044 | 0,248% |
| `forte` | 34 | 0,008% |
| **Total** | **421.196** | **100%** |

Por par de datasets:

| Par | `not_evaluable` | `fraco` | `forte_degenerada` | `forte` | Total |
|---|---|---|---|---|---|
| HAM10000 × ISIC2018 Task1 | 104.754 | 31.692 | 305 | 10 | 136.761 |
| HAM10000 × BCN20000 | 215.413 | 68.259 | 739 | 24 | 284.435 |

Nenhum erro de leitura de imagem ocorreu (`image_read_error` = 0 em toda a execução); os quatro status acima somam exatamente os 421.196 candidatos.

## 6. Alcance declarado do método (Tarefas 9/10) e da guarda de degenerescência

- O método ORB+RANSAC herda as limitações de cobertura já medidas nas Tarefas 9/10 por tipo de transformação (recorte significativo e rotação 90° com baixa avaliabilidade geométrica; espelhamento com baixa avaliabilidade mesmo quando chega à etapa ORB) — esta tarefa não remediu essas limitações, apenas rodou o mesmo método, inalterado, na escala completa.
- A guarda de degenerescência (limiar de dispersão `0.061005`, Tarefa 10) **muda apenas a prioridade/rotulagem de pares já classificados `forte` pelo critério geométrico de base** — não é, e não foi apresentada como, independência absoluta do critério original; o próprio limiar foi derivado sem separação limpa entre positivos e negativos de ajuste (Tarefa 10, seção 3) e tem falsos positivos conhecidos sob recorte agressivo (Tarefa 10, seção 6). `forte_degenerada` continua sendo um rótulo de suspeita elevada, não de descarte automático.
- Nenhuma exclusão de lesão/imagem foi decidida ou implementada em nenhuma das Tarefas 7–11.

## 7. Risco residual da revisão incompleta

As categorias `fraco` e `not_evaluable` — juntas, **99,74% de todos os 421.196 candidatos** (420.118 de 421.196) — têm **zero verificação humana**, em qualquer grau:

| Categoria | População completa | Selecionados na amostra | Revisados |
|---|---|---|---|
| `fraco` | 99.951 | 400 | **0** |
| `not_evaluable` | 320.167 | 100 | **0** |
| **Soma** | **420.118 (99,74% do total)** | **500** | **0** |

Esta é a maior limitação do fechamento desta auditoria, e não deve ser minimizada pelo fato de a categoria `forte` — a mais suspeita por suporte geométrico, mas apenas 0,008% da população — ter sido 100% revisada e não ter produzido nenhuma duplicata confirmada. Um resultado "zero duplicatas" nas categorias de maior suporte geométrico não permite nenhuma inferência sobre as categorias de suporte fraco ou não avaliável, que nunca foram abertas para julgamento humano, nem na amostra estratificada dimensionada para isso. Adicionalmente, 250 dos 500 pares `forte_degenerada` selecionados (50%) também permanecem sem julgamento.

Qualquer uso futuro deste resultado como "auditoria completa" ou "ausência de duplicatas confirmada na população" seria uma leitura incorreta do que foi de fato executado: a amostragem foi dimensionada por completo, a revisão manual foi interrompida em 284/1.034 pares da fila (27,5%), por decisão do pesquisador de fechar a tarefa nesse ponto.

## Aviso

**Isto é fechamento de uma auditoria com revisão manual parcial, não confirmação de ausência de duplicatas na população.** Nenhuma exclusão de lesão ou imagem foi decidida ou implementada em nenhuma etapa das Tarefas 7–11. Os artefatos completos desta tarefa estão em `experiments/data_audit/full_audit_checkpoint.jsonl` (421.196 linhas, passada completa), `experiments/data_audit/full_audit_population_summary.json` (composição completa) e `experiments/data_audit/full_audit_review_queue.json` (fila de 1.034 pares com `review_judgment` por par, `None` nos 750 ainda não revisados).
