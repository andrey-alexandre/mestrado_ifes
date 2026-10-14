"""Entry point CLI da investigação PH² × HAM10000 (piloto exploratório, 80 imagens).

Nada consome API paga sem `infer --run-vlm`. Cada etapa grava artefatos em
`dataset_validation.output.root_dir` (padrão: experiments/dataset_validation, ignorado
pelo git) e pode ser repetida de forma idempotente.

Uso:
    python scripts/run_dataset_validation.py audit      # equivalência + evidências legadas
    python scripts/run_dataset_validation.py sample     # manifesto determinístico (seed fixa)
    python scripts/run_dataset_validation.py quality    # indicadores de qualidade + figuras
    python scripts/run_dataset_validation.py estimate   # estimativa de custo (sem chamadas)
    python scripts/run_dataset_validation.py local      # audit + sample + quality + estimate + report
    python scripts/run_dataset_validation.py infer --run-vlm [--models gpt-5.4-nano]   # PAGO
    python scripts/run_dataset_validation.py evaluate   # métricas + hipóteses (usa inference/results_*.json)
    python scripts/run_dataset_validation.py report
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_validation import analysis, audit as audit_mod, config as cfgmod
from src.dataset_validation import figures, inference, labels as L, legacy, quality, report, sampling
from src.dataset_validation.io_utils import (
    provenance, read_csv, read_json, write_csv, write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("dataset_validation")

COMMANDS = ["audit", "sample", "quality", "estimate", "infer", "evaluate", "report", "local"]


def parse_args():
    p = argparse.ArgumentParser(description="Investigação PH² × HAM10000 (piloto exploratório).")
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--run-vlm", action="store_true", help="Autoriza chamadas pagas às VLMs (infer).")
    p.add_argument("--models", nargs="+", default=None, help="Sobrescreve inference.models.")
    p.add_argument("--n-per-class", type=int, default=None, help="Sobrescreve sample.n_per_class.")
    p.add_argument("--output-root", default=None)
    p.add_argument("--ph2-dir", default=None)
    p.add_argument("--ham-image-dir", default=None)
    return p.parse_args()


def make_cfg(args):
    over = {}
    if args.n_per_class:
        over.setdefault("sample", {})["n_per_class"] = args.n_per_class
    if args.output_root:
        over.setdefault("output", {})["root_dir"] = args.output_root
    if args.models:
        over.setdefault("inference", {})["models"] = args.models
    if args.ph2_dir:
        over.setdefault("datasets", {}).setdefault("ph2", {})["dataset_dir"] = args.ph2_dir
    if args.ham_image_dir:
        over.setdefault("datasets", {}).setdefault("ham10000", {})["image_dir"] = args.ham_image_dir
    return cfgmod.load_config(args.config, over)


def _need(path, what):
    if not path or not Path(path).exists():
        raise SystemExit(f"{what} não encontrado: {path!r}. Ajuste a seção dataset_validation do config.yaml.")
    return path


def load_inputs(cfg):
    ph2 = cfg["datasets"]["ph2"]
    ham = cfg["datasets"]["ham10000"]
    _need(ph2["metadata_txt"], "PH2_dataset.txt")
    _need(ph2["dataset_dir"], "Diretório do PH²")
    _need(ham["metadata_csv"], "Metadado HAM10000")
    _need(ham["image_dir"], "Diretório de imagens HAM10000")
    return L.parse_ph2_metadata(ph2["metadata_txt"]), L.parse_ham_metadata(ham["metadata_csv"])


def cmd_audit(cfg, dirs):
    ph2_records, ham_meta = load_inputs(cfg)
    first_json = next(iter(cfg["legacy"]["ph2_results_json"].values()), None)
    a = audit_mod.build_audit(cfg, ph2_records, ham_meta,
                              str(cfgmod.ROOT / first_json) if first_json else None)
    a["provenance"] = provenance(cfg["seed"])
    write_json(dirs["audit"] / "equivalence_audit.json", a)
    leg = legacy.load_legacy_rows(cfg, ph2_records)
    summary = legacy.summarize_legacy(leg, cfg)
    write_json(dirs["audit"] / "legacy_ph2_reevaluation.json", summary)
    flat = [{k: v for k, v in r.items() if k != "event"} for b in leg.values() for r in b["rows"]]
    write_csv(dirs["audit"] / "legacy_ph2_rows.csv", flat)
    logger.info("Auditoria gravada em %s", dirs["audit"])
    return a, summary


def cmd_sample(cfg, dirs):
    ph2_records, ham_meta = load_inputs(cfg)
    rows = sampling.build_manifest(cfg, ph2_records, ham_meta,
                                   mask_cache_dir=dirs["inputs"] / "ham10000_masks")
    write_csv(dirs["manifest"] / "manifest.csv", rows, sampling.MANIFEST_FIELDS)
    meta = {"seed": cfg["seed"], "n_per_class": cfg["sample"]["n_per_class"], "n_images": len(rows),
            "fingerprint": sampling.manifest_fingerprint(rows),
            "sampling": "random.Random(f'{seed}:{dataset}:{classe}').sample(sorted(unidades), n); "
                        "HAM10000: unidade = lesion_id (1 imagem por lesão, escolhida com semente derivada); "
                        "PH²: unidade = caso; sem seleção manual",
            "label_sources": {"ph2": "Clinical Diagnosis (2=melanoma; 0,1=nevo)",
                              "ham10000": "dx (mel=melanoma; nv=nevo melanocítico)"},
            "provenance": provenance(cfg["seed"])}
    write_json(dirs["manifest"] / "manifest_meta.json", meta)
    logger.info("Manifesto: %d imagens (fingerprint %s)", len(rows), meta["fingerprint"][:12])
    return rows


def _manifest(dirs):
    p = dirs["manifest"] / "manifest.csv"
    if not p.exists():
        raise SystemExit("Manifesto ausente: execute `sample` antes.")
    rows = read_csv(p)
    for r in rows:
        r["binary_label"] = int(r["binary_label"])
        r["draw_rank"] = int(r["draw_rank"])
    return rows


def cmd_quality(cfg, dirs):
    manifest = _manifest(dirs)
    rows = []
    for r in manifest:
        ind = quality.indicators_for_row(r, cfg["quality"])
        rows.append({"sample_id": r["sample_id"], "dataset": r["dataset"], "class_label": r["class_label"],
                     "image_id": r["image_id"], "acquisition_source": r["acquisition_source"], **ind})
    write_csv(dirs["quality"] / "indicators.csv", rows)
    summ = analysis.quality_summary(rows)
    write_json(dirs["quality"] / "comparison.json", summ)
    write_json(dirs["quality"] / "composition.json", analysis.source_composition(manifest))
    figures.contact_sheet(manifest, dirs["quality"] / "visual_sample.png")
    figures.distribution_plots(rows, dirs["quality"] / "distributions.png")
    logger.info("Qualidade: %d imagens; figuras em %s", len(rows), dirs["quality"])
    return rows, summ


def cmd_estimate(cfg, dirs):
    manifest = _manifest(dirs)
    est = inference.estimate_cost(cfg, len(manifest))
    write_json(dirs["inference"] / "cost_estimate.json", est)
    return est


def cmd_infer(cfg, dirs, run_vlm):
    if not run_vlm:
        raise SystemExit("Inferência paga NÃO executada: passe --run-vlm após aprovar a estimativa "
                         "(`estimate`). Nenhuma chamada foi feita.")
    manifest = _manifest(dirs)
    est = cmd_estimate(cfg, dirs)
    for m in cfg["inference"]["models"]:
        logger.info("Executando %s (%d chamadas estimadas)", m, est["per_model"][m]["llm_calls"])
        path = inference.run_model(cfg, manifest, m, dirs["inference"])
        logger.info("Resultados: %s", path)


def cmd_evaluate(cfg, dirs):
    manifest = _manifest(dirs)
    ph2_records, _ = load_inputs(cfg)
    quality_rows = read_csv(dirs["quality"] / "indicators.csv") if (dirs["quality"] / "indicators.csv").exists() else []
    out, hyp = {}, {}
    prim = cfg["evaluation"]["primary_policy"]
    for m in cfg["inference"]["models"]:
        rp = dirs["inference"] / f"results_{inference.safe_name(m)}.json"
        if not rp.exists():
            logger.warning("Sem resultados para %s (%s): pulando.", m, rp)
            continue
        interp = analysis.interpret_results(manifest, read_json(rp))
        write_csv(dirs["evaluation"] / f"predictions_{inference.safe_name(m)}.csv", interp)
        out[m] = analysis.evaluate_model(interp, cfg["evaluation"])
        hyp[m] = {
            "quality_vs_error": analysis.quality_vs_error(interp, quality_rows, prim) if quality_rows else None,
            "error_by_acquisition_source": analysis.error_by_group(interp, prim, "acquisition_source"),
            "error_by_dx_type": analysis.error_by_group(interp, prim, "dx_type"),
            "error_by_ph2_nevus_subtype": analysis.error_by_group(interp, prim, "ph2_nevus_subtype"),
            "ph2_annotation_errors": analysis.ph2_annotation_errors(interp, ph2_records, prim),
            "agent_profiles": analysis.agent_profiles(interp),
        }
    write_json(dirs["evaluation"] / "metrics.json", out)
    write_json(dirs["evaluation"] / "hypothesis_analysis.json", hyp)
    return out, hyp


def cmd_report(cfg, dirs):
    manifest = _manifest(dirs)
    meta = read_json(dirs["manifest"] / "manifest_meta.json")
    art = {
        "manifest_meta": meta,
        "composition": read_json(dirs["quality"] / "composition.json"),
        "quality_summary": read_json(dirs["quality"] / "comparison.json"),
        "legacy": read_json(dirs["audit"] / "legacy_ph2_reevaluation.json"),
        "estimate": read_json(dirs["inference"] / "cost_estimate.json"),
        "evaluation": (read_json(dirs["evaluation"] / "metrics.json") or None)
        if (dirs["evaluation"] / "metrics.json").exists() else None,
    }
    art["status"] = {
        "auditoria/legado": "executada", "amostragem": f"executada ({len(manifest)} imagens)",
        "qualidade": "executada",
        "inferência VLM": "executada" if art["evaluation"] else "NÃO executada (aguarda autorização)",
        "avaliação": "executada" if art["evaluation"] else "pendente da inferência",
    }
    return report.write_report(dirs["reports"] / "report.md", art)


def main():
    args = parse_args()
    cfg = make_cfg(args)
    dirs = cfgmod.output_dirs(cfg)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cmd = args.command
    if cmd == "audit":
        cmd_audit(cfg, dirs)
    elif cmd == "sample":
        cmd_sample(cfg, dirs)
    elif cmd == "quality":
        cmd_quality(cfg, dirs)
    elif cmd == "estimate":
        cmd_estimate(cfg, dirs)
    elif cmd == "infer":
        cmd_infer(cfg, dirs, args.run_vlm)
    elif cmd == "evaluate":
        cmd_evaluate(cfg, dirs)
    elif cmd == "report":
        print(cmd_report(cfg, dirs))
    elif cmd == "local":
        cmd_audit(cfg, dirs)
        cmd_sample(cfg, dirs)
        cmd_quality(cfg, dirs)
        cmd_estimate(cfg, dirs)
        print(cmd_report(cfg, dirs))


if __name__ == "__main__":
    main()
