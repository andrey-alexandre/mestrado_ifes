# Reprocessamento offline -- variante acentuada de `nao_melanoma`

ZERO chamadas novas de modelo -- só reaplica `extract_decisao_final` (já com `não_melanoma` aceito como variante de `nao_melanoma`) sobre o `raw_response_text` já salvo das 8 chamadas reais da 1ª execução (ver `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/results.json`, não modificado por este script).

- Total de chamadas reprocessadas: 8 (esperado 8)
- Chamadas cujo `decision_status`/`decisao_final` MUDOU: 2

## Chamadas (uma seção por cenário x modelo)

### modelo=`qwen3-vl:8b-instruct-q4_K_M` cenário=`unanime_positivo`

- `changed`: **False**
- ANTES: `decision_status='ok'` `decisao_final='melanoma'` `matched_raw_values=['melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='melanoma'` `matched_raw_values=['melanoma']`

---

### modelo=`qwen3-vl:8b-instruct-q4_K_M` cenário=`unanime_negativo`

- `changed`: **False**
- ANTES: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['nao_melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['nao_melanoma']`

---

### modelo=`qwen3-vl:8b-instruct-q4_K_M` cenário=`conflitante`

- `changed`: **False**
- ANTES: `decision_status='ok'` `decisao_final='melanoma'` `matched_raw_values=['melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='melanoma'` `matched_raw_values=['melanoma']`

---

### modelo=`qwen3-vl:8b-instruct-q4_K_M` cenário=`um_parecer_ausente`

- `changed`: **False**
- ANTES: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['nao_melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['nao_melanoma']`

---

### modelo=`medgemma1.5:4b-it-q4_K_M` cenário=`unanime_positivo`

- `changed`: **False**
- ANTES: `decision_status='ok'` `decisao_final='melanoma'` `matched_raw_values=['melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='melanoma'` `matched_raw_values=['melanoma']`

---

### modelo=`medgemma1.5:4b-it-q4_K_M` cenário=`unanime_negativo`

- `changed`: **True**
- ANTES: `decision_status='invalid_value'` `decisao_final=None` `matched_raw_values=['não_melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['não_melanoma']`

---

### modelo=`medgemma1.5:4b-it-q4_K_M` cenário=`conflitante`

- `changed`: **True**
- ANTES: `decision_status='invalid_value'` `decisao_final=None` `matched_raw_values=['não_melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['não_melanoma']`

---

### modelo=`medgemma1.5:4b-it-q4_K_M` cenário=`um_parecer_ausente`

- `changed`: **False**
- ANTES: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['nao_melanoma']`
- DEPOIS: `decision_status='ok'` `decisao_final='nao_melanoma'` `matched_raw_values=['nao_melanoma']`

---
