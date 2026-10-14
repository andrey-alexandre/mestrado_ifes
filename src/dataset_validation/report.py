"""Gera reports/report.md a partir dos artefatos (tabelas e achados automáticos).

O texto interpretativo longo vive em docs/dataset_validation/*.md (escrito a partir dos
números gerados). Aqui só entram fatos extraídos dos artefatos, sem inferência causal.
"""
from pathlib import Path
from typing import Dict, List, Optional

KEY_QUALITY = [
    "megapixels", "lesion_area_ratio", "lesion_equiv_diameter_ref_px", "lum_mean", "lum_std",
    "lum_dynamic_range", "contrast_lesion_skin_dE", "lap_var_global_ref", "lap_var_lesion_ref",
    "edge_strength_boundary_ref", "noise_sigma_skin", "vignetting_ratio", "hair_proxy_frac",
    "glare_frac_global",
]


def _f(x, nd=3) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:,.0f}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _ci(m: Dict, key: str) -> str:
    c = m.get(key) or {}
    if c.get("estimate") is None:
        return "—"
    return f"{_f(c['estimate'], 2)} [{_f(c.get('ci_low'), 2)}–{_f(c.get('ci_high'), 2)}]"


def metrics_table(by_dataset: Dict, policy: str) -> str:
    rows = []
    for ds in ("ph2", "ham10000"):
        m = by_dataset[ds][policy]
        rows.append([ds, f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}", str(m["n_evaluable"]),
                     _ci(m, "sensitivity"), _ci(m, "specificity"), _ci(m, "precision"),
                     _ci(m, "f1"), _ci(m, "balanced_accuracy"), _ci(m, "accuracy")])
    return _table(["dataset", "TN/FP/FN/TP", "n aval.", "sens.", "espec.", "prec.", "F1", "acc. bal.", "acc."], rows)


def quality_table(comp: List[Dict], label_a: str, label_b: str, only: Optional[List[str]] = None) -> str:
    rows = []
    for r in comp:
        if only and r["indicator"] not in only:
            continue
        rows.append([r["indicator"], r["group"],
                     f"{_f(r[f'median_{label_a}'])} ({_f(r[f'q1_{label_a}'])}–{_f(r[f'q3_{label_a}'])})",
                     f"{_f(r[f'median_{label_b}'])} ({_f(r[f'q1_{label_b}'])}–{_f(r[f'q3_{label_b}'])})",
                     _f(r["cliffs_delta"], 2), _f(r["mannwhitney_p"], 3), _f(r["holm_p"], 3), r["direction"]])
    return _table(["indicador", "grupo", f"{label_a} mediana (IQR)", f"{label_b} mediana (IQR)",
                   "δ Cliff", "p (MW)", "p Holm", "direção"], rows)


def build_report(art: Dict) -> str:
    """`art`: manifest_meta, composition, quality_summary, legacy, evaluation, estimate, audit."""
    L: List[str] = []
    meta = art["manifest_meta"]
    L.append("# Investigação PH² × HAM10000 — relatório gerado\n")
    L.append("> Exploratório, n pequeno, sem inferência causal. Gerado por "
             "`scripts/run_dataset_validation.py report`; interpretação em "
             "`docs/dataset_validation/`.\n")
    L.append("## 1. Estado da execução\n")
    L.append(_table(["etapa", "estado"], [[k, v] for k, v in art["status"].items()]))
    L.append("\n## 2. Amostra\n")
    L.append(f"- semente: **{meta['seed']}**; n por classe: **{meta['n_per_class']}**; "
             f"imagens: **{meta['n_images']}**; fingerprint do manifesto: `{meta['fingerprint'][:16]}…`")
    comp = art["composition"]
    for ds in ("ph2", "ham10000"):
        L.append(f"\n**{ds}** — fonte de aquisição × classe: `{comp[ds]['acquisition_source_x_class']}`; "
                 f"tipo de confirmação × classe: `{comp[ds]['dx_type_x_class']}`; "
                 f"subtipo do nevo (PH²): `{comp[ds]['ph2_nevus_subtype']}`")
    L.append("\n## 3. Evidências legadas do PH² (recuperadas dos arquivos; sem reconstrução)\n")
    leg = art["legacy"]
    rows = []
    for model, res in leg.items():
        if not res.get("present"):
            rows.append([model, "arquivo ausente", "", "", ""])
            continue
        for scen, m in res["scenarios"].items():
            if not m.get("n"):
                continue
            rows.append([model, scen, f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}",
                         _ci(m, "sensitivity"), _ci(m, "specificity")])
    L.append(_table(["modelo", "cenário", "TN/FP/FN/TP", "sens.", "espec."], rows))
    for model, res in leg.items():
        if res.get("present"):
            L.append(f"\n- {model}: positivos pelo rótulo legado = {res['n_positive_legacy']}, "
                     f"pelo diagnóstico clínico = {res['n_positive_clinical']}; "
                     f"imagens com rótulo divergente: {[d['image_id'] for d in res['label_disagreement']]}")
    L.append("\n## 4. Qualidade computacional das imagens\n")
    L.append("_Indicadores de sinal; não medem adequação clínica nem dificuldade diagnóstica._\n")
    q = art["quality_summary"]
    L.append("### PH² × HAM10000 (todas as 40 + 40)\n")
    L.append(quality_table(q["ph2_vs_ham10000_all"], "ph2", "ham10000", KEY_QUALITY))
    for cls in ("melanoma", "nevus"):
        L.append(f"\n### PH² × HAM10000 — somente {cls}\n")
        L.append(quality_table(q[f"ph2_vs_ham10000_{cls}"], "ph2", "ham10000", KEY_QUALITY))
    L.append("\n## 5. Inferência multiagente e avaliação\n")
    ev = art.get("evaluation")
    if not ev:
        L.append("**NÃO EXECUTADA** — aguarda autorização de custo/credenciais. "
                 "Estimativa em `inference/cost_estimate.json`:\n")
        est = art["estimate"]
        L.append(_table(["modelo", "chamadas", "tokens entrada (est.)", "tokens saída (est.)",
                         "custo US$ (est.)", "credencial no ambiente"],
                        [[m, str(v["llm_calls"]), _f(v["input_tokens_est"]), _f(v["output_tokens_est"]),
                          _f(v["cost_usd_est"]) if v["cost_usd_est"] is not None else "preço não informado",
                          "sim" if v["credential_present"] else "NÃO"]
                         for m, v in est["per_model"].items()]))
    else:
        for model, res in ev.items():
            prim = res["primary_policy"]
            L.append(f"\n### {model} — política primária `{prim}`\n")
            L.append(metrics_table(res["by_dataset"], prim))
            other = [p for p in res["policies"] if p != prim][0]
            L.append(f"\nSensibilidade (política `{other}`):\n")
            L.append(metrics_table(res["by_dataset"], other))
            c = res["ph2_vs_ham10000_primary_policy"]
            L.append("\n" + _table(["métrica", "PH²", "HAM10000", "diferença [IC Newcombe]", "p Fisher"],
                                   [[k, _f(v["p1"], 2), _f(v["p2"], 2),
                                     f"{_f(v['difference'], 2)} [{_f(v['ci_low'], 2)}–{_f(v['ci_high'], 2)}]",
                                     _f(v["fisher_exact_p_two_sided"], 3)] for k, v in c.items()]))
    return "\n".join(L) + "\n"


def write_report(path: Path, art: Dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(art), encoding="utf-8")
    return str(path)
