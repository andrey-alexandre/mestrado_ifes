import json
import re
import csv
import sys
from pathlib import Path

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ── EIXO 1 ──────────────────────────────────────────────────────────────────
ROOT_FIELDS = ['image_path','lesion_size','seg_path','diagnosis_abcd',
               'diagnosis_menzies','diagnosis_spcl','validation','final_report','real_value']
ABCD_FIELDS = ['assimetria','bordas','cor','estruturas','explicacao']
MENZIES_POS = ['veu_azul_branco','multiplos_pontos_marrons','pseudopodes',
               'streaming_radial','despigmentacao_tipo_cicatriz',
               'pontos_pretos_perifericos','multiplas_cores']
MENZIES_NEG = ['simetria','cor_unica']
SPCL_FIELDS = ['padrao_de_pigmentacao_atipico','veu_azul_branco_irregular',
               'vasos_atipicos','padrao_de_reticulo_irregular',
               'padrao_de_globulos_irregulares','hiperpigmentacao_localizada',
               'regressao','explicacao']
VALIDATION_FIELDS = ['diagnostico','justificativa','recomendacoes']

TOTAL_EXPECTED = (len(ROOT_FIELDS) + len(ABCD_FIELDS) +
                  len(MENZIES_POS) + len(MENZIES_NEG) +
                  len(SPCL_FIELDS) + len(VALIDATION_FIELDS))

def check_e1(record):
    missing = []
    present = 0
    for f in ROOT_FIELDS:
        if f in record: present += 1
        else: missing.append(f)
    abcd = record.get('diagnosis_abcd', {})
    for f in ABCD_FIELDS:
        if f in abcd: present += 1
        else: missing.append(f'abcd.{f}')
    menz = record.get('diagnosis_menzies', {})
    pos = menz.get('positivas', {})
    neg = menz.get('negativas', {})
    for f in MENZIES_POS:
        if f in pos: present += 1
        else: missing.append(f'menzies.positivas.{f}')
    for f in MENZIES_NEG:
        if f in neg: present += 1
        else: missing.append(f'menzies.negativas.{f}')
    spcl = record.get('diagnosis_spcl', {})
    for f in SPCL_FIELDS:
        if f in spcl: present += 1
        else: missing.append(f'spcl.{f}')
    val = record.get('validation', {})
    for f in VALIDATION_FIELDS:
        if f in val: present += 1
        else: missing.append(f'validation.{f}')
    pct = round(present / TOTAL_EXPECTED * 100, 2)
    conforme = 1 if present == TOTAL_EXPECTED else 0
    return pct, conforme, '|'.join(missing)

# ── EIXO 2 ──────────────────────────────────────────────────────────────────
ABCD_WEIGHTS = {'assimetria': 1.3, 'bordas': 0.1, 'cor': 0.5, 'estruturas': 0.5}

def extract_score(text):
    """Extract leading number from string. Returns (value, imprecise_flag)"""
    if not text:
        return None, False
    text = str(text).strip()
    # range like "7/8"
    m = re.match(r'^(\d+(?:\.\d+)?)\s*/\s*\d', text)
    if m:
        return float(m.group(1)), True
    # leading number
    m = re.match(r'^(\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1)), False
    return None, False

def extract_declared_total(explicacao):
    if not explicacao:
        return None
    # Find line with "Soma ponderada total" and take the LAST = NUMBER on that line
    for line in explicacao.splitlines():
        if re.search(r'[Ss]oma ponderada total', line):
            matches = re.findall(r'=\s*([\d]+(?:[.,]\d+)?)', line)
            if matches:
                try:
                    return float(matches[-1].replace(',', '.'))
                except:
                    pass
    # Gemini style: "... = 6.7. " — last = NUMBER followed by . and non-digit or end
    m = re.search(r'=\s*([\d]+[.,]\d+)\.(?:\s|$)', explicacao)
    if m:
        try:
            return float(m.group(1).replace(',', '.'))
        except:
            pass
    # Gemini single-decimal style "= 6.7" at end of math chain
    all_eq = re.findall(r'=\s*([\d]+(?:[.,]\d+)?)', explicacao)
    if all_eq:
        try:
            return float(all_eq[-1].replace(',', '.'))
        except:
            pass
    return None

def check_e2(record):
    abcd = record.get('diagnosis_abcd', {})
    scores = {}
    imprecise = []
    for field in ['assimetria','bordas','cor','estruturas']:
        val, imp = extract_score(abcd.get(field, ''))
        scores[field] = val
        if imp:
            imprecise.append(field)
    a, b, c, d = scores['assimetria'], scores['bordas'], scores['cor'], scores['estruturas']
    if any(v is None for v in [a,b,c,d]):
        total_rec = None
        aritmetica = None
        diferenca = None
    else:
        total_rec = round(a*1.3 + b*0.1 + c*0.5 + d*0.5, 2)
        total_decl = extract_declared_total(abcd.get('explicacao',''))
        if total_decl is None:
            aritmetica = None
            diferenca = None
        else:
            diferenca = round(abs(total_rec - total_decl), 4)
            aritmetica = 1 if diferenca <= 0.05 else 0
    total_decl2 = extract_declared_total(abcd.get('explicacao',''))
    return (a, b, c, d, total_rec,
            total_decl2, aritmetica,
            diferenca if diferenca is not None else None,
            '|'.join(imprecise) if imprecise else '')

# ── EIXO 3 ──────────────────────────────────────────────────────────────────
def classify_abcd(total):
    if total is None: return 'indefinido'
    if total < 4.75: return 'benigna'
    if total <= 5.45: return 'suspeita'
    return 'maligna'

def classify_validation(diagnostico):
    if not diagnostico: return 'indefinido'
    d = diagnostico.lower()
    # Check for malign without negation
    if re.search(r'\bmaligna?\b', d):
        # check negation nearby
        if not re.search(r'(não|nao|sem)\s+\w{0,10}\s*maligna?', d):
            return 'maligna'
    if re.search(r'alto grau de suspei[cç]|alta suspei[cç]', d):
        return 'suspeita'
    if re.search(r'suspei[cç]', d) and not re.search(r'baixa\s+suspei[cç]', d):
        return 'suspeita'
    if re.search(r'indeterminad', d):
        return 'indeterminada'
    if re.search(r'benigna?|benigno', d):
        return 'benigna'
    return 'indefinido'

def risk_level(cat):
    if cat in ('suspeita','maligna'): return 'alto'
    if cat in ('benigna',): return 'baixo'
    return 'parcial'

def check_e3(total_rec, diagnostico_text):
    abcd_cat = classify_abcd(total_rec)
    val_cat = classify_validation(diagnostico_text)
    abcd_risk = risk_level(abcd_cat)
    val_risk = risk_level(val_cat)
    if abcd_risk == val_risk:
        aderencia = 1.0
    elif val_risk == 'parcial':
        aderencia = 0.5
    else:
        aderencia = 0.0
    return abcd_cat, val_cat, aderencia

# ── EIXO 4 ──────────────────────────────────────────────────────────────────
def classify_menzies(record):
    menz = record.get('diagnosis_menzies', {})
    pos = menz.get('positivas', {})
    neg = menz.get('negativas', {})
    n_pos = sum(1 for v in pos.values() if str(v).strip().startswith('1'))
    sim = str(neg.get('simetria','')).strip().startswith('1')
    cor_u = str(neg.get('cor_unica','')).strip().startswith('1')
    ambos_neg = sim and cor_u
    if n_pos >= 1 and not ambos_neg:
        return 'suspeita_melanoma'
    return 'benigna'

SPCL_MAJOR = ['padrao_de_pigmentacao_atipico','veu_azul_branco_irregular','vasos_atipicos']
SPCL_MINOR = ['padrao_de_reticulo_irregular','padrao_de_globulos_irregulares',
              'hiperpigmentacao_localizada','regressao']

def extract_spcl_score(record):
    spcl = record.get('diagnosis_spcl', {})
    expl = spcl.get('explicacao','')
    patterns = [
        r'[Pp]ontua[cç][aã]o total\s*[=:]\s*(\d+)',
        r'[Tt]otal\s*[=:]\s*(\d+)',
        r'[Tt]otal estimado\s*[=:]*\s*(\d+)',
    ]
    for p in patterns:
        m = re.search(p, expl)
        if m:
            return int(m.group(1))
    # manual sum
    score = 0
    for f in SPCL_MAJOR:
        if str(spcl.get(f,'')).strip().lower().startswith('sim'):
            score += 2
    for f in SPCL_MINOR:
        if str(spcl.get(f,'')).strip().lower().startswith('sim'):
            score += 1
    return score

def check_e4(record, abcd_cat):
    menz_dec = classify_menzies(record)
    spcl_score = extract_spcl_score(record)
    spcl_dec = 'suspeita' if spcl_score >= 3 else 'benigna'

    high_risk = 0
    for dec, cat in [(abcd_cat, abcd_cat), (menz_dec, menz_dec), (spcl_dec, spcl_dec)]:
        if dec in ('suspeita','maligna','suspeita_melanoma'):
            high_risk += 1

    consenso = 1 if high_risk in (0, 3) else 0

    val_cat = classify_validation(record.get('validation',{}).get('diagnostico',''))
    val_risk = risk_level(val_cat)

    if consenso:
        if high_risk == 3:
            val_concorda = 1 if val_risk == 'alto' else 0
        else:
            val_concorda = 1 if val_risk == 'baixo' else 0
    else:
        val_concorda = None

    return menz_dec, spcl_score, spcl_dec, high_risk, consenso, val_concorda

# ── EIXO 5 ──────────────────────────────────────────────────────────────────
def check_e5(val_cat, real_value):
    cat = val_cat
    if cat in ('suspeita','maligna'):
        pred = 1
    elif cat == 'benigna':
        pred = 0
    else:
        pred = None
    if pred is None:
        correto = None
    else:
        correto = 1 if pred == real_value else 0
    return cat, pred, correto

# ── MAIN ─────────────────────────────────────────────────────────────────────
def analyze(filepath):
    data = load_json(filepath)
    rows = []

    for img_path, record in data.items():
        real_value = record.get('real_value', None)

        # E1
        e1_pct, e1_conf, e1_missing = check_e1(record)

        # E2
        a, b, c, d, total_rec, total_decl, arit, dif, imprecise = check_e2(record)

        # E3
        abcd_cat, val_cat, aderencia = check_e3(total_rec,
            record.get('validation',{}).get('diagnostico',''))

        # E4
        menz_dec, spcl_score, spcl_dec, high_risk, consenso, val_concorda = check_e4(record, abcd_cat)

        # E5
        e5_cat, e5_pred, e5_correto = check_e5(val_cat, real_value)

        rows.append({
            'imagem': img_path,
            'real_value': real_value,
            'e1_campos_presentes_pct': e1_pct,
            'e1_conforme': e1_conf,
            'e1_campos_faltantes': e1_missing,
            'e2_a': a, 'e2_b': b, 'e2_c': c, 'e2_d': d,
            'e2_total_recomputado': total_rec,
            'e2_total_declarado': total_decl,
            'e2_aritmetica_correta': arit,
            'e2_diferenca': dif,
            'e2_scores_imprecisos': imprecise,
            'e3_classificacao_abcd': abcd_cat,
            'e3_classificacao_validacao': val_cat,
            'e3_aderencia_limiar': aderencia,
            'e4_menzies_decisao': menz_dec,
            'e4_spcl_score': spcl_score,
            'e4_spcl_decisao': spcl_dec,
            'e4_metodos_alto_risco': high_risk,
            'e4_consenso_total': consenso,
            'e4_validacao_concorda_consenso': val_concorda,
            'e5_categoria_diagnostica': e5_cat,
            'e5_predicao_binaria': e5_pred,
            'e5_correto': e5_correto,
        })

    return rows

def build_summary(rows):
    n = len(rows)
    # E1
    e1_full = sum(1 for r in rows if r['e1_conforme'] == 1)
    e1_avg = round(sum(r['e1_campos_presentes_pct'] for r in rows) / n, 2) if n else 0

    # E2
    e2_valid = [r for r in rows if r['e2_aritmetica_correta'] is not None]
    e2_corr = sum(1 for r in e2_valid if r['e2_aritmetica_correta'] == 1)
    e2_imp = sum(1 for r in rows if r['e2_scores_imprecisos'])

    # E3
    e3_plena = sum(1 for r in rows if r['e3_aderencia_limiar'] == 1.0)
    e3_parc = sum(1 for r in rows if r['e3_aderencia_limiar'] == 0.5)

    # E4
    e4_cons = sum(1 for r in rows if r['e4_consenso_total'] == 1)
    e4_cons_rows = [r for r in rows if r['e4_consenso_total'] == 1]
    e4_val_conc = sum(1 for r in e4_cons_rows if r['e4_validacao_concorda_consenso'] == 1)
    dist = [sum(1 for r in rows if r['e4_metodos_alto_risco'] == i) for i in range(4)]

    # E5
    e5_def = [r for r in rows if r['e5_predicao_binaria'] is not None]
    e5_acc = sum(1 for r in e5_def if r['e5_correto'] == 1)
    e5_ind = sum(1 for r in rows if r['e5_predicao_binaria'] is None)

    cats = ['benigna','suspeita','maligna','indeterminada']
    cat_stats = {}
    for cat in cats:
        sub = [r for r in rows if r['e5_categoria_diagnostica'] == cat]
        n_cat = len(sub)
        pct_mel = round(sum(1 for r in sub if r['real_value'] == 1) / n_cat * 100, 1) if n_cat else 0
        cat_stats[cat] = (n_cat, pct_mel)

    def fmt(x, tot, pct_dec=1):
        p = round(x/tot*100, pct_dec) if tot else 0
        return f"{x}/{tot} ({p}%)"

    summary = [
        ("Eixo 1 – Conformidade Estrutural", "Respostas com 100% dos campos presentes", fmt(e1_full, n)),
        ("Eixo 1 – Conformidade Estrutural", "Média de campos presentes por resposta", f"{e1_avg}%"),
        ("Eixo 2 – Consistência Aritmética ABCD", "Soma ponderada correta (tolerância ±0,05)", fmt(e2_corr, len(e2_valid))),
        ("Eixo 2 – Consistência Aritmética ABCD", "Respostas com scores imprecisos (ex.: \"7/8\")", fmt(e2_imp, n)),
        ("Eixo 3 – Aderência ao Limiar Declarado", "Validação concorda plenamente com ABCD", fmt(e3_plena, n)),
        ("Eixo 3 – Aderência ao Limiar Declarado", "Divergência parcial (indeterminada vs. benigna ABCD)", fmt(e3_parc, n)),
        ("Eixo 4 – Concordância Inter-Métodos", "Casos com consenso total (3 métodos)", fmt(e4_cons, n)),
        ("Eixo 4 – Concordância Inter-Métodos", "Validação concorda com consenso total", fmt(e4_val_conc, len(e4_cons_rows))),
        ("Eixo 4 – Concordância Inter-Métodos", "Distribuição: 0/1/2/3 métodos em alto risco",
         f"{dist[0]}; {dist[1]}; {dist[2]}; {dist[3]}"),
        ("Eixo 5 – Calibração da Incerteza", "Acurácia nos casos definitivos (excluindo indeterminados)", fmt(e5_acc, len(e5_def))),
        ("Eixo 5 – Calibração da Incerteza", "Proporção de diagnósticos indeterminados", fmt(e5_ind, n)),
        ("Eixo 5 – Calibração da Incerteza", "Benigna: n casos / % melanoma real",
         f"n={cat_stats['benigna'][0]} / {cat_stats['benigna'][1]}%"),
        ("Eixo 5 – Calibração da Incerteza", "Suspeita: n casos / % melanoma real",
         f"n={cat_stats['suspeita'][0]} / {cat_stats['suspeita'][1]}%"),
        ("Eixo 5 – Calibração da Incerteza", "Maligna: n casos / % melanoma real",
         f"n={cat_stats['maligna'][0]} / {cat_stats['maligna'][1]}%"),
        ("Eixo 5 – Calibração da Incerteza", "Indeterminada: n casos / % melanoma real",
         f"n={cat_stats['indeterminada'][0]} / {cat_stats['indeterminada'][1]}%"),
    ]
    return summary

DETAIL_COLS = [
    'imagem','real_value',
    'e1_campos_presentes_pct','e1_conforme','e1_campos_faltantes',
    'e2_a','e2_b','e2_c','e2_d','e2_total_recomputado','e2_total_declarado',
    'e2_aritmetica_correta','e2_diferenca','e2_scores_imprecisos',
    'e3_classificacao_abcd','e3_classificacao_validacao','e3_aderencia_limiar',
    'e4_menzies_decisao','e4_spcl_score','e4_spcl_decisao',
    'e4_metodos_alto_risco','e4_consenso_total','e4_validacao_concorda_consenso',
    'e5_categoria_diagnostica','e5_predicao_binaria','e5_correto',
]

def save_csvs(rows, summary, base_name):
    detail_path = f"{base_name}_detail.csv"
    summary_path = f"{base_name}_summary.csv"

    with open(detail_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS, delimiter=';')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,'') for k in DETAIL_COLS})

    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(['eixo','metrica','valor'])
        for eixo, metrica, valor in summary:
            w.writerow([eixo, metrica, valor])

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    return detail_path, summary_path

files = [
    ('llm_result_gemini-3.1-flash-lite_20260609.json', 'analysis_gemini'),
    ('llm_result_gpt-5.4-nano_20260609.json', 'analysis_gpt'),
]

for fname, base in files:
    print(f"\n=== Processing {fname} ===")
    rows = analyze(fname)
    summary = build_summary(rows)
    save_csvs(rows, summary, base)
    print(f"Total records: {len(rows)}")
    # Print summary to stdout
    print("\nSUMMARY:")
    for eixo, metrica, valor in summary:
        print(f"  {metrica}: {valor}")
