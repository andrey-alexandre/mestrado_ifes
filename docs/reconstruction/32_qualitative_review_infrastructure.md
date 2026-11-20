# 32 — Infraestrutura de análise qualitativa dos VLMs (sínteses multiagente S2)

Na branch `feature/inspection`. **É uma análise dos resultados — não constitui critério de aprovação de modelo nem autoriza excluir casos.** Nenhuma amostra, predição ou contrato de decisão foi alterado; nenhuma inferência nova de VLM foi feita.

## 1. Revisão qualitativa das sínteses multiagente (S2, 300 lesões)

População: exatamente as 300 lesões já congeladas em `experiments/multiagent/local_300_s2_subset/local_300_s2_subset_manifest.csv` — nenhum novo sorteio. Até 1.200 sínteses (300 × 4 modelos: Gemini, GPT, Qwen3-VL, MedGemma), cada uma com os 3 pareceres (ABCD/Menzies/SPCL) pareados por lesão/condição/modelo via a mesma seleção oficial de tentativa já usada na análise quantitativa (`src.evaluation.stage6_sources.select_official_call_records` — reusado, não reimplementado).

**Novos módulos:**
- `src/evaluation/qualitative_review_schema.py` — os 10 critérios + 1 marcador separado (observação/hipótese própria, nunca tratado como problema por si), vocabulário de status (`presente`/`ausente`/`nao_avaliavel`/pendente), mapeamento de identificadores neutros (seed `20261120`, permutação global fixa — documentado como NÃO sendo cegamento garantido).
- Extensão aditiva em `src/evaluation/stage6_sources.py`: `load_gemini_recovered_text_by_full_key` (só leitura do texto já recuperado oficialmente do Gemini, para exibição — a análise quantitativa não usa esta função, nada nela foi alterado).

**Scripts:**
- `scripts/prepare_qualitative_review_materials.py` — gera 300 páginas HTML (`experiments/evaluation/qualitative_review/pages/<lesion_id>.html`) com a imagem S2, os 3 pareceres e a síntese de cada modelo (sob "Modelo A/B/C/D"), indicadores de agente ausente e consolidação parcial, e destaque heurístico leve (nomes de agente, marcadores de hipótese) — nunca preenche veredito. Usa `extract_final_answer` (já validada) para exibir o texto pós-template; para Gemini usa a fonte oficial reprocessada. Gera `review_template.csv` (formato longo, uma linha por lesão×modelo×critério, pendente por padrão — só marcado `nao_avaliavel` automaticamente quando a síntese está literalmente sem texto, nunca por presunção de conteúdo). O diagnóstico real e a chave de modelos ficam em arquivos separados, com `_CONFIDENCIAL` no nome, nunca linkados do HTML.
- `scripts/consolidate_qualitative_reviews.py` — lê `review_template.csv` preenchido, agrega por modelo×critério (revisado/presente/ausente/não avaliável/pendente/denominador avaliável/taxa). Não decide aprovação de nada.

## 2. Comandos para o pesquisador (nenhum executado nesta sessão)

```
python -m pytest tests/evaluation/test_qualitative_review_schema.py -v

python scripts/prepare_qualitative_review_materials.py
# preencha experiments/evaluation/qualitative_review/review_template.csv
python scripts/consolidate_qualitative_reviews.py
```

Dependências: todas já existentes (`Pillow` indireta via HTML/imagens) — nenhuma nova.

## 3. Limites preservados

Nenhuma amostra/predição/contrato de decisão foi alterada. Nenhuma inferência de VLM foi feita nesta entrega. A análise quantitativa em desenvolvimento (`run_final_quantitative_analysis*.py`, `stage6_*`) não foi tocada além da função aditiva `load_gemini_recovered_text_by_full_key`, que ela não usa.
