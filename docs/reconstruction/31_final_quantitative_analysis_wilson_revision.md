# 31 — Análise quantitativa final: desenho (Etapa 6, continuação) e revisão metodológica (Wilson)

Este documento cobre duas entregas da análise quantitativa final da dissertação, na branch `feature/inspection`:

1. A infraestrutura original (v1) que estende o avaliador estatístico da Etapa 6 já existente para as seis análises aprovadas, sobre os artefatos reais de 3.000/300 lesões.
2. A revisão metodológica (v2, 2026-11-19): **IC95% de Wilson para as proporções individuais; bootstrap preservado para F1 e para todas as diferenças pareadas.**

As duas versões coexistem no repositório. **v1 não foi alterada** — qualquer execução em andamento daquela versão não é afetada por esta entrega. v2 é um script e um diretório de saída separados.

## 1. Desenho v1 (reuso da Etapa 6)

Módulos reutilizados sem modificação: `src/evaluation/stage6_metrics.py`, `stage6_bootstrap.py`, `stage6_pairing.py`, `case_status.py`, `explicit_decision_contract.py`, `src/multiagent/vlm_direct.is_call_record_reusable`.

Módulos novos (v1):
- `src/evaluation/stage6_operational_status.py` — 8 status operacionais mutuamente exclusivos (positivo/negativo/indeterminado/recusa/resposta_invalida/falha_tecnica/ausencia_execucao/mascara_vazia), com ponte explícita (`STAGE6_STATUS_MAP`) para o vocabulário de 5 status de `case_status`.
- `src/evaluation/stage6_sources.py` — leitura dos artefatos reais (manifesto 3.000, subconjunto 300, `s0_cnn_scores.csv`, `call_log_main.jsonl` de cada modelo, reprocessamento oficial do Gemini); seleção da tentativa oficial por chave reusando `is_call_record_reusable`, mais recente por timestamp em caso de múltiplas tentativas reais (cego ao resultado, registrado em `multi_attempt_keys`).
- `src/evaluation/stage6_pairwise.py` — recomputa as 5 métricas de duas configurações sobre a interseção fixa dos casos binários de ambas.
- `src/evaluation/stage6_bootstrap_ci.py` — IC95% por bootstrap (percentis), métricas individuais e diferenças pareadas, regra dos ≥4.750/5.000 réplicas válidas.
- `src/evaluation/stage6_report.py` — montagem de tabelas e escritores CSV/Markdown/JSON.
- `scripts/run_final_quantitative_analysis.py` — orquestrador v1 (bootstrap para todas as métricas, inclusive as proporções individuais).

Seis análises (protocolo original, seção 5), universo 3.000 (análises 1–5, pares Gemini/GPT) e universo 300 (análise 6, 10 pares todos-contra-todos entre CNN/Gemini/GPT/Qwen3-VL/MedGemma, ordem fixa CNN→Gemini→GPT→Qwen3-VL→MedGemma).

## 2. Revisão metodológica v2 (2026-11-19) — decisão do pesquisador

**Wilson substitui bootstrap só para as proporções individuais listadas abaixo.** Bootstrap continua para F1 individual e para toda diferença pareada (as 5 métricas na interseção do par + cobertura binária/acertos sobre o total no manifesto inteiro).

### 2.1 Tabela de aplicação de Wilson

| Indicador | Sucessos | Denominador |
|---|---|---|
| Acurácia | TP+TN | TP+TN+FP+FN |
| Precisão | TP | TP+FP |
| Sensibilidade | TP | TP+FN |
| Especificidade | TN | TN+FP |
| Cobertura binária | decisões binárias válidas | total do manifesto |
| Acertos sobre o total | decisões binárias corretas | total do manifesto |

Implementado em `src/evaluation/stage6_wilson.py::wilson_score_interval` (fórmula fechada de Wilson, bilateral, **sem** correção de continuidade) e `wilson_results_for_indicators` (aplica a tabela acima a um `Stage6Indicators` já calculado, sem recontar nada). Único uso de dependência externa: `scipy.stats.norm.ppf` para o quantil normal — `scipy` já é dependência do projeto (`src/data_audit/phash.py`), nenhuma dependência nova foi adicionada (`statsmodels` foi considerado e descartado por ser mais pesado do que o necessário só para este cálculo).

**Não é Wald** (não tem o termo `z²/n` que desloca o centro), **não é Agresti-Coull** (não usa `n` ajustado) e **não é Clopper-Pearson** (não é exato via Beta/F) — as três descartadas explicitamente.

A estimativa pontual devolvida é sempre a proporção observada (`successes/n`); o centro de Wilson (deslocado) nunca é usado como ponto. Denominador zero → `None` (nunca 0.0). Zero sucessos ou todos os sucessos são calculados pela mesma fórmula, sem caso especial (o formato fechado é bem-definido e permanece em `[0,1]` nos dois extremos).

**Rótulo obrigatório em toda saída**: estes são **intervalos binomiais aproximados para as proporções observadas** — não reproduzem o condicionamento por categoria diagnóstica do bootstrap estratificado, nem são IC ajustados ao desenho amostral da amostra congelada. Não devem ser lidos como equivalentes aos IC bootstrap.

### 2.2 Onde o bootstrap permanece (protocolo original intacto)

- F1 individual: `bootstrap_single_system_metric_cis(..., metric_names=("f1",))` — mesma função de v1, agora chamada só para F1 pelo orquestrador v2 (assinatura estendida com `metric_names`, default inalterado — v1 não é afetada).
- Diferenças pareadas das 5 métricas: `bootstrap_paired_difference_cis` (inalterada), interseção binária fixa do par, dentro de cada réplica.
- **Novo**: diferenças pareadas de cobertura binária e de acertos sobre o total: `bootstrap_full_manifest_paired_difference_cis` — mesma réplica (manifesto inteiro, nunca a interseção) reusada nos dois lados.
- 5.000 réplicas, semente 42, reamostragem estratificada por `dx`, mesmos índices para os dois lados de cada par, ≥4.750 réplicas válidas para reportar qualquer IC bootstrap. Esta regra **não se aplica a Wilson** (que não depende de réplicas/semente).

### 2.3 Novos/alterados arquivos

- **Novo**: `src/evaluation/stage6_wilson.py`.
- **Estendido, retrocompatível**: `stage6_bootstrap_ci.py` (parâmetro `metric_names` opcional em `bootstrap_single_system_metric_cis`, default inalterado; nova função `bootstrap_full_manifest_paired_difference_cis`).
- **Estendido, retrocompatível**: `stage6_report.py` (`build_pairwise_row` agora itera as chaves do dicionário recebido, não uma lista fixa — mesmo resultado para os chamadores existentes com 5 chaves; novas funções `build_main_table_row_wilson`/`main_table_markdown_row_wilson`/`format_wilson_markdown`/`wilson_to_dict`, aditivas — `build_main_table_row`/`main_table_markdown_row` originais **intocadas**).
- **Novo orquestrador**: `scripts/run_final_quantitative_analysis_v2_wilson.py` — cópia adaptada de `run_final_quantitative_analysis.py` (v1, intocado), grava em `experiments/evaluation/final_quantitative_analysis_v2_wilson/` (diretório novo, separado de `experiments/evaluation/final_quantitative_analysis/`).

### 2.4 Tabelas e metadados

Tabela principal v2 (`tabela_principal.csv/.md`): coluna `metodologia` e nota explícita por linha identificando Wilson (acurácia/precisão/sensibilidade/especificidade/cobertura binária/acertos sobre o total) vs. bootstrap (F1). Cada coluna Wilson expõe `_method`, `_numerator`, `_denominator`; F1 expõe `_method="bootstrap"`, `_ci_reported`, `_n_valid_replicas`. Tabela pareada suplementar: toda diferença expõe `diff_<metrica>_method="bootstrap"` (nunca derivado de subtração de limites de Wilson). `metadados_reproducao.json` registra `methodology_version`, o bloco `wilson` (método/confiança/indicadores aplicáveis) e o bloco `bootstrap` (semente/réplicas/mínimo válido/indicadores aplicáveis) separadamente.

## 3. Comandos para o pesquisador

Testes (nenhum executado por mim — regra de não-execução autônoma):
```
python -m pytest tests/evaluation/test_stage6_wilson.py tests/evaluation/test_stage6_bootstrap_ci.py tests/evaluation/test_stage6_report.py -v
```

Execução oficial v2 (não interfere com uma execução em andamento de `run_final_quantitative_analysis.py`, script e diretório de saída diferentes):
```
python scripts/run_final_quantitative_analysis_v2_wilson.py --n-replicas 200 --output-dir experiments/evaluation/final_quantitative_analysis_v2_wilson_smoke   # smoke test, nao oficial
python scripts/run_final_quantitative_analysis_v2_wilson.py   # oficial, 5000 replicas/seed 42 (defaults)
```

## 4. Limites preservados

Nenhuma amostra, predição, contrato de decisão, seleção de tentativa oficial, modelo, prompt ou limiar foi alterado nesta revisão. Nenhuma inferência nova foi feita. A pendência real de dados já registrada (matriz de 3.000 do GPT-5.4-nano incompleta no momento desta entrega) é preservada e continua marcada `provisorio=True` em v2 exatamente como em v1.
