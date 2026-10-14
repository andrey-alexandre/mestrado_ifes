# Auditoria de equivalência e evidências recuperadas

Fonte dos números: `experiments/dataset_validation/audit/equivalence_audit.json` e
`legacy_ph2_reevaluation.json` (gerados por `run_dataset_validation.py audit`; HEAD `a557ba4`,
branch `exp/dataset_validation`, árvore com alterações não commitadas).

## 1. O que está comprovado em arquivo × o que não está

### Resultados HAM10000 do pipeline multiagente VLM: **AUSENTES**

Não foi encontrado, no repositório nem nas pastas locais pesquisadas (`C:\Projetos\Mestrado`,
`Downloads`), nenhum resultado do *este* pipeline (4 agentes VLM + sumarizador) sobre
HAM10000, nem matriz de confusão, nem composição de amostra. Em particular:

- `resultados.tex` e a qualificação (`Qualificação.pdf`, v3) reportam **apenas PH²**; HAM10000
  aparece só como referência bibliográfica e como "trabalho futuro" (`conclusao.tex`).
- `Multiagent - CAD/llm_result_gpt-4o-mini_20250711.json` (30 imagens) e `Dissertação -
  CP/llm_result.json` usam ids `ISIC_6767172…` (padrão do ISIC 2024) e `ISIC_0015104…` (fora do intervalo `ISIC_0024306–0034320` do HAM10000), não HAM10000.
- `Downloads/Mestrado_V3/outputs_*.zip` contém execuções sobre HAM10000 (70 imagens, 7 classes, 10 por
  classe) de **outra implementação** (indicadores determinísticos + agentes heurísticos, `agents.summary_agent`).
  Não é este pipeline, não é melanoma × nevo e não é comparável; foi apenas catalogada.

→ A premissa "desempenho muito inferior no HAM10000" **não está documentada nos arquivos
disponíveis**. Peça a origem (Drive/Colab) se existir; até lá é relato, não evidência.

### Resultados PH² (comprovados)

| Arquivo | Conteúdo |
|---|---|
| `llm_result_{gpt-5.4-nano,gemini-3.1-flash-lite}_20260609.json` | 50 imagens (25/25 pelo rótulo legado), respostas originais, `seg_path` GT |
| `llm_result_*_20260606.csv` | `diagnostico`, `real_value`, `estimated_value` (sem id de imagem) — é a tabela da dissertação |
| `analysis_*_{detail,summary}.csv` | análise de 5 eixos de consistência |
| `resultados.tex` | Tab. LLM: Gemini acc .76/prec .81/sens .68/F1 .74; GPT .68/.68/.68/.68; ResNet-50 (60 img) |

Reprodução exata: a matriz do CSV de 06/06 é GPT TN/FP/FN/TP = 17/8/8/17 e Gemini 21/4/8/17,
idêntica à tabela da dissertação (testada em `test_metrics_with_ci_reproduces_thesis_table`).

## 2. Equivalência de configuração, entrada e critério

| Item | Achado | Status |
|---|---|---|
| Prompts/agentes | `src/agents/agents.py` tem **um único commit** (`3fe1e2d`); o JSON legado tem exatamente o esquema de saída de `run_pipeline` (`seg_path`, `lesion_size`…). Compatibilidade de formato ≠ prova de mesma versão | parcial |
| Imagem enviada às VLMs | Os 4 agentes recebem **apenas** `seg_image_data` (imagem mascarada); `image_data` nunca é enviado (4 ocorrências, 0 referências). **A dissertação descreve "pares de imagens (original e segmentada)" — divergente do código** | **divergência comprovada** |
| Modo de segmentação | PH² legado: `precomputed` (GT, resolução nativa). Default do `config.yaml`: `model` (U-Net, redimensiona p/ ~256 px de largura, máscara *prevista*, limiar 0,4, resize aleatório 300–320). Não há registro do modo usado para HAM10000 | **indeterminado** (se foi `model`, a entrada difere materialmente) |
| `lesion_size` | Não entra em nenhum prompt (0 interpolações) — irrelevante | comprovado |
| Temperatura | `config.yaml` = 0; `build_graph` usa 0,7 se omitido; JSON legado não registra | indeterminado |
| Versão dos modelos | Apenas o nome no arquivo (`gpt-5.4-nano`, `gemini-3.1-flash-lite`); sem versão/fingerprint do provedor | indeterminado |
| Rótulo PH² | Regra legada (`Histological Diagnosis` ⊃ "melanoma"): **28** positivos; diagnóstico clínico: **40**. 12 melanomas clínicos ficam "negativos" (7 sem histologia, 5 *Lentigo Maligna*). Na amostra de 50, **IMD061 e IMD065 (melanomas) estão rotulados 0**: são 27/23, não 25/25 | **divergência comprovada** |
| Regra de leitura | `evaluate.py` mapeia "lesão indeterminada"→1 e descarta texto não reconhecido. A dissertação usa **leitura manual semântica**. Aplicada programaticamente ao JSON legado: GPT → 48/50 "indeterminada" ⇒ todos positivos (espec. 0,00); Gemini sens. 0,89 / espec. 0,87 (rótulo clínico) | **não reproduzível automaticamente** |
| Execuções legadas | CSV (06/06) e JSON (09/06) são **execuções distintas**: textos idênticos em 0/50 (GPT) e 38/50 (Gemini); mesma `real_value` em 50/50 por ordem; categorias coincidem em 44/50 e 45/50 | comprovado (ruído execução-a-execução ≈ 10–12 %) |
| Credenciais | `_build_llm` passa strings literais `"OPENAI_API_KEY"`/`"GOOGLE_API_KEY"` como chave; reuso em Colab provavelmente as substituía | ambiente |
| RAG | O nó crítico (`final_report`) não afeta o diagnóstico; construir RAG baixa `all-MiniLM-L6-v2` | ambiente |
| `Image_Segmentation` | Pacote externo da U-Net ausente do repo (checkpoint `U_Net-100-0.0005-70-0.6667.pkl` está em `Dissertação - CP/`) | ambiente |

### Desvios do piloto em relação ao legado (todos documentados no código)

1. **Interpretação**: regra programática de duas camadas e duas políticas (acima) — substitui
   a leitura manual, que não é reprodutível nem aplicável sem viés a 80 novas respostas.
2. **Crítico RAG → recuperador nulo**; U-Net → stub inerte; chaves lidas do ambiente. Nenhum
   altera modelo, prompt, temperatura ou saída diagnóstica.
3. **HAM10000**: imagens lidas da cópia do ISIC 2019 (ids `ISIC_0024306–0034320`; 10015/10015
   presentes, 600×450) e máscaras do conjunto de segmentações distribuído com o HAM10000
   (`archive.zip`). Não há como comparar byte a byte com o pacote original do HAM10000.
4. **Reuso de inferências legadas: não adotado.** 17 das 40 imagens PH² do piloto estão
   entre as 50 legadas, mas a equivalência de prompts/versões/temperatura não é comprovável
   (e as duas execuções legadas já divergem entre si). O PH² será reinferido no mesmo lote
   que o HAM10000 — também elimina deriva temporal entre os datasets.

## 3. Confundidores estruturais descobertos (relevantes para H1/H3)

- **Origem do rótulo**: PH² = diagnóstico clínico consensual; HAM10000 `mel` = 100 % histopatologia,
  `nv` = 3704 follow-up + 2498 histo + 503 consenso (pool completo). Na amostra: 16/20 nevos são *follow-up*.
- **Fonte de aquisição × classe no HAM10000** (pool): `vidir_molemax` tem 3720 nevos e 24 melanomas;
  `vidir_modern` 1832/680; `rosendahl` 803/342; `vienna_dias` 350/67. Na amostra: 16/20 nevos e 1/20
  melanomas vêm de `vidir_molemax`. Fonte e classe estão associadas — um sinal de aquisição
  poderia imitar sinal diagnóstico. No PH² há uma única fonte.
- **Nevos do PH²**: 50 % atípicos (displásicos) — negativos presumivelmente mais difíceis; no HAM `nv` não há essa separação.
- **Origem das máscaras**: PH² manual por especialista; HAM10000 distribuídas com o dataset
  (processo de criação não verificável nos arquivos locais).
- **Chave de API em texto claro** em `LangChain.ipynb` (comentário da célula 14, não rastreado):
  recomenda-se revogar a chave e removê-la do notebook antes de qualquer commit.
