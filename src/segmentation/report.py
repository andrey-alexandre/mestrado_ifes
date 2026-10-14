"""Relatório científico em Markdown a partir dos artefatos estruturados.

Não importa código de treino. Cada seção declara explicitamente quando seus
insumos não existem; nenhum valor é inventado ou estimado.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.segmentation import figures
from src.segmentation.artifacts import read_csv, read_json, write_json

logger = logging.getLogger(__name__)

UNAVAILABLE = "_Resultados indisponíveis: a etapa **{stage}** não foi executada (artefato ausente)._"


def fmt(x: Any, nd: int = 4) -> str:
    if x is None or x == "":
        return "n/d"
    if isinstance(x, float):
        return "n/d" if x != x else f"{x:.{nd}f}"
    return str(x)


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def ci(d: Dict) -> str:
    return f"{fmt(d['estimate'])} [{fmt(d['ci_low'])}; {fmt(d['ci_high'])}]"


def _opt(path: Path) -> Optional[Any]:
    return read_json(str(path)) if path.exists() else None


class _Inputs:
    def __init__(self, results: Path, cfg: Dict):
        self.cfg = cfg
        self.isic = _opt(results / "isic_training" / "summary.json")
        self.manifest = _opt(Path(cfg["output"]["folds_manifest"]))
        self.direct = _opt(results / "direct_transfer" / "summary.json")
        self.tuned = _opt(results / "finetune" / "summary.json")
        self.comp = _opt(results / "comparison" / "summary.json")
        self.sel = _opt(results / "comparison" / "selection.json")
        self.examples = _opt(results / "comparison" / "examples.json")
        self.final = _opt(results / "final_model" / "provenance.json")
        self.ma = _opt(results / "multiagent" / "summary.json")
        self.diag = _opt(results / "multiagent" / "diagnostic_metrics.json")
        self.direct_rows = read_csv(str(results / "direct_transfer" / "per_image.csv")) \
            if (results / "direct_transfer" / "per_image.csv").exists() else None
        self.tuned_rows = read_csv(str(results / "finetune" / "per_image.csv")) \
            if (results / "finetune" / "per_image.csv").exists() else None
        self.diffs = read_csv(str(results / "comparison" / "paired_differences.csv")) \
            if (results / "comparison" / "paired_differences.csv").exists() else None
        self.feat = read_csv(str(results / "multiagent" / "feature_comparison.csv")) \
            if (results / "multiagent" / "feature_comparison.csv").exists() else None


def _seg_table(summary: Dict, metrics: List[str]) -> str:
    rows = []
    for m in metrics:
        s = summary["overall"][m]
        rows.append([m.upper(), s["mean"], s["median"], s["std"], s["min"], s["max"],
                     f"{s['n_valid']}/{s['n']}"])
    return md_table(["Métrica", "Média", "Mediana", "DP", "Mín", "Máx", "válidas/total"], rows)


def _per_fold_table(summary: Dict, metrics: List[str]) -> str:
    rows = []
    for f in sorted(summary["per_fold"], key=int):
        blk = summary["per_fold"][f]
        rows.append([f, blk["n_images"]] + [blk[m]["mean"] for m in metrics])
    return md_table(["Fold", "n"] + [f"{m.upper()} médio" for m in metrics], rows)


def build_report(cfg: Dict) -> str:
    """Gera `report.md` (+ figuras) em `output.reports_dir`. Retorna o caminho do relatório."""
    results = Path(cfg["output"]["results_dir"])
    reports = Path(cfg["output"]["reports_dir"])
    fig_dir = reports / "figures"
    inp = _Inputs(results, cfg)
    ev = cfg["evaluation"]
    primary = ev["primary_metric"]
    metrics = [primary] + list(ev["secondary_metrics"])
    available = {k: v is not None for k, v in {
        "isic_training": inp.isic, "folds": inp.manifest, "direct_transfer": inp.direct,
        "finetune": inp.tuned, "comparison": inp.comp, "multiagent": inp.ma,
        "final_model": inp.final}.items()}
    S: List[str] = []

    # 1. Resumo executivo ────────────────────────────────────────────────────
    S.append("# Relatório experimental — impacto da segmentação no diagnóstico de melanoma\n")
    S.append("## 1. Resumo executivo\n")
    S.append("Etapas com resultados disponíveis: " + ", ".join(
        f"**{k}** {'✔' if v else '✘'}" for k, v in available.items()) + ".\n")
    if inp.comp and inp.sel:
        o = inp.comp["overall"][primary]
        S.append(
            f"- Métrica primária: **{primary.upper()}** (n = {inp.comp['n_images']} imagens do PH²).\n"
            f"- Transferência direta: média {fmt(o['direct_transfer']['mean'])}; "
            f"fine-tuning: média {fmt(o['finetune']['mean'])}; "
            f"diferença pareada (fine-tuning − direta): {ci(o['paired_diff_mean'])}.\n"
            f"- Estratégia selecionada pela regra `{inp.sel['rule']}`: **{inp.sel['selected']}**.\n"
            f"- ⚠ {inp.sel['selection_bias_note']}\n")
    else:
        S.append("- Comparação entre estratégias indisponível.\n")
    if inp.diag and inp.diag.get("available"):
        S.append(f"- Impacto no sistema multiagente: ver seção 11 (n = {inp.diag['n_images']}).\n")
    else:
        S.append("- Avaliação diagnóstica multiagente: indisponível (VLMs não executados ou etapa ausente).\n")

    # 2. Objetivo ─────────────────────────────────────────────────────────────
    S.append("\n## 2. Objetivo e hipóteses experimentais\n")
    S.append(
        "**Objetivo.** Avaliar o impacto da segmentação automática no desempenho diagnóstico do sistema "
        "multiagente baseado em VLMs, e investigar a transferência entre datasets dermatoscópicos "
        "(ISIC 2018 Task 1 → PH²).\n\n"
        "**Hipóteses (declaradas antes da execução).**\n"
        "- H1: o fine-tuning no PH² (validação cruzada em 5 folds) melhora a métrica primária de "
        "segmentação em relação à transferência direta.\n"
        "- H2: o uso de máscaras previstas (C1) em vez da referência (C0) altera as decisões do sistema "
        "multiagente em magnitude pequena; a diferença nas métricas diagnósticas é estimada com IC "
        "bootstrap pareado, sem teste formal de não-inferioridade.\n")

    # 3. Datasets ─────────────────────────────────────────────────────────────
    S.append("\n## 3. Descrição dos datasets\n")
    if inp.isic:
        d = inp.isic["data_summary"]
        S.append(f"**ISIC 2018 Task 1.** {inp.isic['n_train']} imagens de treino e {inp.isic['n_val']} de "
                 f"validação; {d.get('n_excluded_overlap', 0)} excluídas por sobreposição declarada; "
                 f"metadados de lesão disponíveis: {'sim' if d.get('group_info_available') else 'não'}.\n")
    else:
        S.append("**ISIC 2018 Task 1.** " + UNAVAILABLE.format(stage="isic_training") + "\n")
    if inp.manifest and "data_summary" in inp.manifest:
        ds = inp.manifest["data_summary"]
        S.append(
            f"**PH².** {ds['n_images']} imagens; {ds['n_melanoma']} melanomas "
            f"(rótulo: `{ds['label_source']}`). Classes: "
            + ", ".join(f"{k}={v}" for k, v in ds["class_counts"].items()) + ".\n\n"
            f"Observação: a regra legada do repositório (`\"melanoma\"` em *Histological Diagnosis*) "
            f"rotularia {ds['legacy_histological_rule_positives']} positivos, contra "
            f"{ds['clinical_rule_positives']} pela *Clinical Diagnosis*; "
            f"{len(ds['label_discrepancy_ids'])} imagens divergem entre as regras.\n")
    else:
        S.append("**PH².** " + UNAVAILABLE.format(stage="folds") + "\n")

    # 4. Arquitetura ──────────────────────────────────────────────────────────
    S.append("\n## 4. Arquitetura e configuração dos modelos\n")
    seg = cfg["segmentation"]
    S.append(f"- Estratégia: `{seg['strategy']}` — parâmetros `{json.dumps(seg['strategy_params'])}`\n"
             f"- Resolução de entrada: {seg['image_size']}×{seg['image_size']}; limiar: {seg['threshold']} "
             f"(fixado a priori)\n- Loss: `{seg['loss']}`; otimizador: `{seg['optimizer']}` "
             f"(lr {seg['learning_rate']}); scheduler: `{seg['scheduler']['name']}`\n"
             f"- Seed: {seg['seed']}; device: `{seg['device']}`; mixed precision: {seg['mixed_precision']}\n"
             f"- Fine-tuning: blocos do encoder congelados `{cfg['transfer_learning']['frozen_encoder_blocks']}`, "
             f"política `{cfg['transfer_learning']['unfreeze_policy']}`, lr "
             f"{cfg['transfer_learning']['learning_rate']}\n")
    prov = (inp.isic or inp.direct or {}).get("provenance")
    if prov:
        S.append("- Versões: " + ", ".join(f"{k} {v}" for k, v in prov["versions"].items()) + "\n")

    # 5. Protocolo de treinamento ─────────────────────────────────────────────
    S.append("\n## 5. Protocolo de treinamento\n")
    if inp.isic:
        S.append(
            f"Treino supervisionado no ISIC 2018; melhor época **{inp.isic['best_epoch']}** pelo monitor "
            f"`{inp.isic['monitor']}` (= {fmt(inp.isic['best_monitor'])}); épocas executadas "
            f"{inp.isic['epochs_run']} (early stopping: {inp.isic['stopped_early']}). Checkpoint "
            f"`{Path(inp.isic['checkpoint']).name}` — SHA-256 `{inp.isic['checkpoint_sha256'][:16]}…`. "
            "O PH² não foi usado em nenhuma decisão deste treino.\n")
    else:
        S.append(UNAVAILABLE.format(stage="isic_training") + "\n")

    # 6. Validação cruzada ────────────────────────────────────────────────────
    S.append("\n## 6. Protocolo de validação cruzada\n")
    if inp.manifest:
        m = inp.manifest
        counts: Dict[int, List[int]] = {}
        for im in m["images"]:
            c = counts.setdefault(im["fold"], [0, 0])
            c[0] += 1
            c[1] += im["label"]
        S.append(f"{m['n_folds']} folds estratificados (`{m['stratify_by']}`), seed {m['seed']}; "
                 f"hash dos ids `{m['ids_sha256'][:16]}…`. Os mesmos folds foram usados na transferência "
                 "direta e no fine-tuning; em cada rodada, a validação interna (early stopping/seleção "
                 "de checkpoint) usa apenas os 4 folds de treino.\n\n")
        S.append(md_table(["Fold", "imagens", "melanomas"],
                          [[f, c[0], c[1]] for f, c in sorted(counts.items())]) + "\n")
    else:
        S.append(UNAVAILABLE.format(stage="folds") + "\n")

    # 7. Transferência direta ─────────────────────────────────────────────────
    S.append("\n## 7. Resultados da transferência direta\n")
    if inp.direct:
        S.append("Avaliação externa sem adaptação (um único modelo; não são cinco treinamentos).\n\n")
        S.append(_seg_table(inp.direct, metrics) + "\n\n" + _per_fold_table(inp.direct, metrics) + "\n")
        S.append(f"\nPesos inalterados durante a inferência: {inp.direct['weights_unchanged']}.\n")
    else:
        S.append(UNAVAILABLE.format(stage="direct_transfer") + "\n")

    # 8. Fine-tuning ──────────────────────────────────────────────────────────
    S.append("\n## 8. Resultados do fine-tuning\n")
    if inp.tuned:
        S.append("Predições fora da amostra (cada imagem predita por modelo que não a viu).\n\n")
        S.append(_seg_table(inp.tuned, metrics) + "\n\n" + _per_fold_table(inp.tuned, metrics) + "\n\n")
        S.append(md_table(["Rodada", "n adapt.", "n val. interna", "n teste", "melhor época", "épocas"],
                          [[r["round"], r["n_adapt_train"], r["n_adapt_val"], r["n_test"],
                            r["best_epoch"], r["epochs_run"]] for r in inp.tuned["rounds"]]) + "\n")
        S.append(f"\nTodas as rodadas partiram dos mesmos pesos ISIC 2018: "
                 f"{inp.tuned['all_rounds_started_from_same_weights']}.\n")
    else:
        S.append(UNAVAILABLE.format(stage="finetune") + "\n")

    # 9. Comparação ───────────────────────────────────────────────────────────
    S.append("\n## 9. Comparação estatística pareada\n")
    if inp.comp:
        S.append(f"Mesmas {inp.comp['n_images']} imagens. Convenção: {inp.comp['diff_convention']}. "
                 f"IC bootstrap pareado ({ev['bootstrap']['n_resamples']} reamostras, "
                 f"{int(100 * ev['bootstrap']['confidence'])}%).\n\n")
        rows = []
        for m in metrics:
            o = inp.comp["overall"][m]
            rows.append([m.upper(), o["direct_transfer"]["mean"], o["finetune"]["mean"],
                         ci(o["paired_diff_mean"]), ci(o["paired_diff_median"]),
                         f"{o['paired_diff_mean']['n_pairs']} (excl. {o['paired_diff_mean']['n_excluded_undefined']})"])
        S.append(md_table(["Métrica", "Média direta", "Média fine-tuning", "Δ média [IC]",
                           "Δ mediana [IC]", "pares (indefinidos)"], rows) + "\n\n")
        S.append("**Por fold** (Δ média da métrica primária):\n\n")
        S.append(md_table(["Fold", "n", "Δ média [IC]"],
                          [[f, b["n_images"], ci(b[primary]["paired_diff_mean"])]
                           for f, b in sorted(inp.comp["per_fold"].items(), key=lambda t: int(t[0]))]) + "\n\n")
        S.append("**Por classe diagnóstica:**\n\n")
        S.append(md_table(["Classe", "n", "Média direta", "Média fine-tuning", "Δ média [IC]"],
                          [[c, b["n_images"], b[primary]["direct_transfer"]["mean"],
                            b[primary]["finetune"]["mean"], ci(b[primary]["paired_diff_mean"])]
                           for c, b in sorted(inp.comp["per_diagnosis"].items())]) + "\n\n")
        figs = []
        if inp.direct_rows and inp.tuned_rows:
            figs.append(figures.plot_metric_distribution(inp.direct_rows, inp.tuned_rows, primary,
                                                         str(fig_dir / f"dist_{primary}.png")))
            figs.append(figures.plot_per_fold(inp.comp, primary, str(fig_dir / f"per_fold_{primary}.png")))
        if inp.diffs:
            figs.append(figures.plot_paired_difference(inp.diffs, primary, str(fig_dir / f"diff_{primary}.png")))
        for f in figs:
            S.append(f"![{Path(f).stem}](figures/{Path(f).name})\n")
    else:
        S.append(UNAVAILABLE.format(stage="comparison") + "\n")

    # 10. Seleção ─────────────────────────────────────────────────────────────
    S.append("\n## 10. Estratégia selecionada e justificativa\n")
    if inp.sel:
        s = inp.sel
        S.append(
            f"Regra pré-declarada: `{s['rule']}` sobre **{s['primary_metric'].upper()}**. "
            f"Média direta {fmt(s['direct_transfer_mean'])}, fine-tuning {fmt(s['finetune_mean'])}; "
            f"ganho do fine-tuning {fmt(s['improvement_of_finetune_over_direct'])} "
            f"(IC [{fmt(s['improvement_ci'][0])}; {fmt(s['improvement_ci'][1])}]). "
            f"**Selecionada: {s['selected']}.** Ambos os conjuntos de predições foram preservados.\n\n"
            f"> ⚠ {s['selection_bias_note']}\n")
    else:
        S.append(UNAVAILABLE.format(stage="comparison") + "\n")
    if inp.final:
        S.append(f"\nModelo final preparado: `{inp.final['selected_strategy']}`, checkpoint SHA-256 "
                 f"`{inp.final['checkpoint_sha256'][:16]}…`. "
                 + inp.final["data_provenance"].get("in_sample_warning", "") + "\n")

    # 11. Multiagente ─────────────────────────────────────────────────────────
    S.append("\n## 11. Impacto da segmentação no sistema multiagente\n")
    if inp.ma:
        S.append(f"Segmentador: `{inp.ma['selected_segmenter']}`; C0 = máscara de referência, C1 = máscara "
                 f"prevista fora da amostra. Imagens sinalizadas (predição vazia): {inp.ma['n_flagged']}.\n\n")
        if inp.ma["features"].get("available"):
            S.append(f"Features mascara-dependentes recalculadas em cada condição "
                     f"({inp.ma['features']['n_features']} variáveis numéricas; falhas C0/C1: "
                     f"{inp.ma['features']['n_errors_c0']}/{inp.ma['features']['n_errors_c1']}).\n\n")
            if inp.feat:
                def key(r):
                    v = r["mean_abs_diff"]
                    return -float(v) if v not in ("", None) else 0.0
                top = sorted(inp.feat, key=key)[:10]
                S.append(md_table(["Feature", "n", "|Δ| médio", "ρ(|Δ|, 1−Dice)", "ρ(|Δ|, HD95)"],
                                  [[r["feature"], r["n_pairs"], r["mean_abs_diff"],
                                    r["spearman_absdiff_vs_1_minus_dice"], r["spearman_absdiff_vs_hd95"]]
                                   for r in top]) + "\n\n")
        else:
            S.append(f"Comparação de features indisponível: {inp.ma['features'].get('reason')}\n\n")
        if inp.diag and inp.diag.get("available"):
            d = inp.diag
            S.append(md_table(["Métrica", "C0 (GT)", "C1 (automática)", "Δ C1−C0 [IC]"],
                              [[m, d["C0"][m], d["C1"][m], ci(d["paired_diff_c1_minus_c0"][m])]
                               for m in ("sensitivity", "specificity", "precision", "f1", "accuracy")]
                              + [["ROC-AUC (ordinal)", d["C0"]["roc_auc_ordinal"],
                                  d["C1"]["roc_auc_ordinal"], "n/d"]]) + "\n\n")
            S.append(f"Cohen's κ (C0 vs C1): {fmt(d['cohen_kappa'])}; taxa de mudança de decisão: "
                     f"{fmt(d['change_rate'])} ({d['n_changed']}/{d['n_images']}).\n\n"
                     f"Escopo da decisão: {d['decision_scope']}\n")
            S.append(f"![diag](figures/{Path(figures.plot_diagnostic_metrics(d, str(fig_dir / 'diagnostic.png'))).name})\n")
        else:
            S.append("Métricas diagnósticas: **indisponíveis** (VLMs não executados nesta rodada).\n")
    else:
        S.append(UNAVAILABLE.format(stage="multiagent") + "\n")

    # 12. Exemplos ────────────────────────────────────────────────────────────
    S.append("\n## 12. Exemplos qualitativos\n")
    if inp.examples and inp.direct_rows and inp.tuned_rows:
        for tag, rows in (("direct_transfer", inp.direct_rows), ("finetune", inp.tuned_rows)):
            by_id = {r["image_id"]: r for r in rows}
            for kind in ("best", "worst"):
                ids = inp.examples[tag][kind]
                titles = {i: f"{i} ({primary} {float(by_id[i][primary]):.3f})" for i in ids}
                p = figures.plot_examples(by_id, ids, titles, str(fig_dir / f"examples_{tag}_{kind}.png"),
                                          f"{figures.APPROACH_LABEL[tag]} — {'melhores' if kind == 'best' else 'piores'}")
                if p:
                    S.append(f"![{tag}_{kind}](figures/{Path(p).name})\n")
    else:
        S.append(UNAVAILABLE.format(stage="comparison") + "\n")

    # 13. Limitações ──────────────────────────────────────────────────────────
    S.append("\n## 13. Limitações e ameaças à validade\n")
    lim = [
        "Avaliação em um único dataset de teste (PH², 200 imagens, uma instituição, 40 melanomas): "
        "ICs são largos e a generalização para outras populações/dispositivos não é suportada.",
        "Viés de seleção: a escolha da estratégia usa o PH² (ver seção 10).",
        "A transferência direta é uma avaliação externa de um único modelo; seus 5 folds não são réplicas independentes.",
        "O treino e a validação interna do fine-tuning usam poucas imagens (~136 por rodada); a variabilidade "
        "entre sementes não foi caracterizada.",
        "A referência (máscaras do PH²) é uma anotação única, sem variabilidade inter-observador.",
        "HD95 em pixels (sem calibração de espaçamento) e indefinido quando uma das máscaras é vazia.",
        "Rótulo de melanoma do PH² vem de *Clinical Diagnosis*; ver divergência com a regra legada na seção 3.",
        "As decisões dos agentes ABCD/Menzies/SPCL não são binárias no esquema atual; métricas diagnósticas "
        "usam o parecer final do agente de síntese, com mapeamento ordinal para ROC-AUC.",
        "Respostas de VLMs são estocásticas e dependem de versões de API; uma execução por condição.",
    ]
    if inp.direct and inp.direct["overall"]["hd95"]["n_undefined"]:
        lim.append(f"Transferência direta: {inp.direct['overall']['hd95']['n_undefined']} imagens com HD95 indefinido.")
    if inp.ma and inp.ma["n_flagged"]:
        lim.append(f"{inp.ma['n_flagged']} imagens com predição vazia tratadas pela política "
                   f"`{cfg['multiagent']['empty_mask_policy']}` na avaliação multiagente.")
    S.append("\n".join(f"- {x}" for x in lim) + "\n")

    # 14. Conclusões ──────────────────────────────────────────────────────────
    S.append("\n## 14. Conclusões sustentadas pelos resultados\n")
    concl: List[str] = []
    if inp.comp:
        o = inp.comp["overall"][primary]["paired_diff_mean"]
        verdict = ("o IC exclui zero" if (o["ci_low"] > 0 or o["ci_high"] < 0)
                   else "o IC inclui zero (diferença não distinguível de zero neste conjunto)")
        concl.append(f"**Segmentação.** Δ{primary.upper()} médio (fine-tuning − direta) = {ci(o)}; {verdict}.")
    else:
        concl.append("**Segmentação.** Sem conclusão: comparação indisponível.")
    if inp.diag and inp.diag.get("available"):
        d = inp.diag
        concl.append("**Diagnóstico.** Diferenças C1−C0: " + "; ".join(
            f"{m} {ci(d['paired_diff_c1_minus_c0'][m])}" for m in ("sensitivity", "specificity", "f1"))
            + f". κ = {fmt(d['cohen_kappa'])}.")
    else:
        concl.append("**Diagnóstico.** Sem conclusão: avaliação multiagente com VLMs não executada.")
    concl.append("**Generalização.** Os resultados descrevem a transferência ISIC 2018 → PH² nas condições "
                 "descritas; não se estendem a outros datasets (nenhuma inferência no ISIC 2019 foi realizada).")
    S.append("\n".join(f"- {c}" for c in concl) + "\n")

    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "report.md"
    path.write_text("\n".join(S), encoding="utf-8")
    write_json(str(reports / "report_index.json"), {"available_stages": available, "report": str(path)})
    logger.info("Relatório gerado em %s", path)
    return str(path)
