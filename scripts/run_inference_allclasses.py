"""Extensão do piloto: HAM10000, melanoma × TODAS as classes (PH² fora) — pago, rodar na máquina com os dados.

Amostra (140 imagens, semente 42, uma por lesão): 20 melanoma + 20 nevo (as MESMAS do piloto, cujas
respostas são reaproveitadas, sem novas chamadas) + 20 de cada: akiec, bcc, bkl, df, vasc (100 novas).
Chamadas pagas novas: 100 imagens × 4 agentes × 2 modelos = 800. Mesmos agentes/prompts/regra de leitura.

PRÉ-REQUISITOS: os mesmos de scripts/run_inference_pilot.py (dependências, OPENAI_API_KEY e
GOOGLE_API_KEY, dados HAM10000: imagens, HAM10000_metadata e máscaras em pasta OU zip) — o PH² NÃO é
necessário — mais os resultados do piloto em `reuse/` (results_<modelo>.json + .meta.json) ou em
`--reuse-dir`. Informe os caminhos em LOCAL_PATHS ou por flags.

USO
    python scripts/run_inference_allclasses.py --dry-run     # valida tudo, sem chamadas
    python scripts/run_inference_allclasses.py               # executa (retoma se interrompido)
    python scripts/run_inference_allclasses.py --evaluate-only   # reavalia sem imagens nem chamadas

SAÍDA: <output_root>/{inference,evaluation,manifest}/ e `pilot_inference_outputs_*.zip`.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LOCAL_PATHS = {
    "ham_image_dir": None, "ham_metadata_csv": None, "ham_mask_dir": None, "ham_mask_zip": None,
    "output_root": None, "reuse_dir": None,
}
MODELS = ["gpt-5.4-nano", "gemini-3.1-flash-lite"]
# Fingerprint do manifesto de 140 imagens (semente 42; independe de caminhos).
EXPECTED_FINGERPRINT = "a6fbe0abdc9fab94590289b860d77c8d656794bd45b3ed0cc94efe3cf5a09e11"

logger = logging.getLogger("inference_allclasses")


def parse_args():
    p = argparse.ArgumentParser(description="HAM10000: melanoma × todas as classes (pago).")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Smoke test: N imagens novas por modelo.")
    p.add_argument("--allow-fingerprint-mismatch", action="store_true")
    p.add_argument("--allow-reuse-mismatch", action="store_true",
                   help="Reaproveita respostas do piloto mesmo com agents.py/schemas.py diferentes (não recomendado).")
    for k in LOCAL_PATHS:
        p.add_argument("--" + k.replace("_", "-"), default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import run_inference_pilot as base
    base.preflight_imports()
    from src.dataset_validation import allclasses as A, analysis, config as cfgmod, inference, labels as L
    from src.dataset_validation.io_utils import provenance, read_csv, read_json, write_csv, write_json
    from src.dataset_validation import sampling

    paths = {k: (getattr(args, k) or LOCAL_PATHS[k]) for k in LOCAL_PATHS}
    ham: dict = {}
    for src_k, key in (("ham_image_dir", "image_dir"), ("ham_metadata_csv", "metadata_csv"),
                       ("ham_mask_dir", "mask_dir"), ("ham_mask_zip", "mask_zip")):
        if paths[src_k]:
            ham[key] = paths[src_k]
    if paths["ham_mask_dir"] and not paths["ham_mask_zip"]:
        ham["mask_zip"] = None
    over = {"inference": {"models": list(args.models)}, "datasets": {"ham10000": ham}}
    if paths["output_root"]:
        over["allclasses"] = {"output_root": paths["output_root"]}
    if paths["reuse_dir"]:
        over.setdefault("allclasses", {})["reuse_results_dir"] = paths["reuse_dir"]
    cfg = cfgmod.load_config(args.config, over)
    cfg["output"]["root_dir"] = cfg["allclasses"]["output_root"]
    dirs = cfgmod.output_dirs(cfg)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(dirs["inference"] / "run.log")])

    # 1) manifesto
    hcfg = cfg["datasets"]["ham10000"]
    meta = None
    if args.evaluate_only:
        mp = dirs["manifest"] / "manifest.csv"
        if not mp.exists():
            sys.exit(f"--evaluate-only exige {mp} (gerado numa execução anterior).")
        manifest = read_csv(mp)
        for r in manifest:
            r["binary_label"] = int(r["binary_label"])
            r["draw_rank"] = int(r["draw_rank"])
        ham_meta = L.parse_ham_metadata(hcfg["metadata_csv"]) if hcfg["metadata_csv"] and Path(hcfg["metadata_csv"]).exists() else None
    else:
        need = [("HAM10000 imagens", hcfg["image_dir"]), ("HAM10000_metadata", hcfg["metadata_csv"])]
        bad = [f"  - {n}: {p!r}" for n, p in need if not p or not Path(p).exists()]
        if not ((hcfg["mask_dir"] and Path(hcfg["mask_dir"]).is_dir()) or (hcfg["mask_zip"] and Path(hcfg["mask_zip"]).exists())):
            bad.append(f"  - máscaras: mask_dir={hcfg['mask_dir']!r} / mask_zip={hcfg['mask_zip']!r}")
        if bad:
            sys.exit("Caminhos não encontrados (ajuste LOCAL_PATHS ou use flags):\n" + "\n".join(bad))
        ham_meta = L.parse_ham_metadata(hcfg["metadata_csv"])
        manifest = A.build_manifest(cfg, ham_meta, mask_cache_dir=dirs["inputs"] / "ham10000_masks")
        fp = A.manifest_fingerprint(manifest)
        logger.info("Manifesto all-classes: %d imagens, fingerprint %s", len(manifest), fp[:16])
        if fp != EXPECTED_FINGERPRINT and not args.allow_fingerprint_mismatch:
            sys.exit(f"Fingerprint difere do esperado ({fp[:16]}… ≠ {EXPECTED_FINGERPRINT[:16]}…). Nada foi chamado.")
        write_csv(dirs["manifest"] / "manifest.csv", manifest, sampling.MANIFEST_FIELDS)
        write_json(dirs["manifest"] / "manifest_meta.json", {
            "seed": cfg["seed"], "n_per_class": cfg["allclasses"]["n_per_class"], "n_images": len(manifest),
            "fingerprint": fp, "expected_fingerprint": EXPECTED_FINGERPRINT,
            "classes": list(A.CLASS_ORDER), "provenance": provenance(cfg["seed"])})

    # 2) reuso do piloto + estimativa de chamadas novas
    reuse_dir = Path(args.reuse_dir or LOCAL_PATHS["reuse_dir"] or
                     (ROOT / "reuse" if (ROOT / "reuse").is_dir() else cfg["allclasses"]["reuse_results_dir"]))
    agents_py, schemas_py = ROOT / "src/agents/agents.py", ROOT / "src/agents/schemas.py"
    reuse_log, new_counts = {}, {}
    if not args.evaluate_only:
        for model in args.models:
            raw = dirs["inference"] / "raw" / inference.safe_name(model)
            reused = A.seed_reused_results(reuse_dir, model, raw, manifest, agents_py, schemas_py,
                                           allow_mismatch=args.allow_reuse_mismatch)
            reuse_log[model] = reused
            done = sum(1 for r in manifest if (raw / f"{r['dataset']}__{r['image_id']}.json").exists())
            new_counts[model] = len(manifest) - done
            logger.info("%s: %d reaproveitadas do piloto (%s), %d já feitas/novas pendentes → %d chamadas LLM",
                        model, len(reused), reuse_dir, new_counts[model], 4 * new_counts[model])
        write_json(dirs["inference"] / "reuse_log.json", {"reuse_dir": str(reuse_dir), "reused": reuse_log})
        if any(len(v) != 40 for v in reuse_log.values()):
            logger.warning("Esperado reaproveitar 40 imagens por modelo; veja reuse_log.json "
                           "(sem results_<modelo>.json em %s o piloto seria REINFERIDO, com custo).", reuse_dir)
        if args.dry_run:
            logger.info("Dry-run concluído. Chamadas pagas previstas: %d.", 4 * sum(new_counts.values()))
            return 0
        missing_keys = sorted({base.KEY_VARS.get(m) or inference.provider_env_var(m) for m in args.models
                               if (base.KEY_VARS.get(m) or inference.provider_env_var(m))
                               and not os.environ.get(base.KEY_VARS.get(m) or inference.provider_env_var(m) or "")})
        if missing_keys:
            sys.exit("Variáveis de ambiente ausentes: " + ", ".join(missing_keys))

    # 3) inferência (só imagens pendentes) + avaliação
    status = 0
    counts = A.pool_counts(ham_meta) if ham_meta else None
    all_eval = {}
    for model in args.models:
        safe = inference.safe_name(model)
        if not args.evaluate_only:
            logger.info("=== %s ===", model)
            path = inference.run_model(cfg, manifest, model, dirs["inference"], limit=args.limit)
            m = read_json(Path(path).with_suffix(".meta.json"))
            if m["failures"]:
                status = 1
                logger.error("%s: %d falhas: %s", model, len(m["failures"]), list(m["failures"]))
        else:
            path = dirs["inference"] / f"results_{safe}.json"
            if not Path(path).exists():
                logger.warning("Sem %s", path)
                continue
        interp = analysis.interpret_results(manifest, read_json(path))
        write_csv(dirs["evaluation"] / f"predictions_allclasses_{safe}.csv", interp)
        if counts is None:
            logger.warning("HAM10000_metadata indisponível: sem especificidade ponderada por prevalência.")
            counts = {c: 1 for c in A.CLASS_ORDER}
        ev = A.evaluate_allclasses(interp, counts, cfg["evaluation"])
        write_json(dirs["evaluation"] / f"metrics_allclasses_{safe}.json", ev)
        all_eval[model] = ev
        prim = ev[cfg["evaluation"]["primary_policy"]]
        for cls, c in prim["per_class"].items():
            rate = c.get("sensitivity") if cls == "melanoma" else c.get("specificity")
            logger.info("%s | %-8s | n=%d | %s=%.2f | categorias=%s", model, cls, c["n"],
                        "sens" if cls == "melanoma" else "espec", rate if rate is not None else float("nan"),
                        c["categories"])
    if not args.evaluate_only:
        logger.info("Concluído. Traga de volta: %s", base.zip_outputs(dirs))
    return status


if __name__ == "__main__":
    sys.exit(main())
