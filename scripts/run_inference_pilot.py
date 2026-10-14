"""Piloto de inferência PH² × HAM10000 — executar na máquina com acesso aos dados.

Roda o pipeline multiagente ORIGINAL (ABCD + Menzies + 7-Point + sumarizador) sobre as 80
imagens do piloto (20 melanomas + 20 nevos por dataset) com `gpt-5.4-nano` e
`gemini-3.1-flash-lite`: 2 modelos × 80 imagens × 4 chamadas = 640 chamadas pagas.
Nenhuma inferência antiga é reaproveitada.

PRÉ-REQUISITOS
  1. Repositório (branch exp/dataset_validation) com `src/`, `config/`, `scripts/` e
     `docs/dataset_validation/pilot_manifest_reference.csv`; dependências de requirements.txt
     (inclui torch/torchvision, importados pelos agentes, e Pillow/scipy/PyYAML).
  2. Variáveis de ambiente:  OPENAI_API_KEY  e  GOOGLE_API_KEY.
  3. Dados locais: PH² (pasta + PH2_dataset.txt), HAM10000 (imagens `ISIC_xxxxxxx.jpg`,
     `HAM10000_metadata` e máscaras — pasta com `ISIC_xxxxxxx_segmentation.png` OU o zip
     `archive.zip` com `HAM10000_segmentations_lesion_tschandl/`).
     Informe os caminhos em LOCAL_PATHS (abaixo), por flags, ou em
     `dataset_validation.datasets` do config.yaml.

USO
    python scripts/run_inference_pilot.py --dry-run     # valida tudo, sem chamadas pagas
    python scripts/run_inference_pilot.py               # executa os 2 modelos (retoma se interrompido)
    python scripts/run_inference_pilot.py --models gpt-5.4-nano

SAÍDA: <output_root>/inference/results_<modelo>.json (+ .meta.json e raw/ por imagem),
<output_root>/evaluation/{predictions_<modelo>.csv,metrics.json} e um zip
`pilot_inference_outputs_*.zip` para trazer de volta. Reexecutar não repete chamadas já feitas.
"""
import argparse
import importlib
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Edite uma vez na máquina de execução (None = usa o config.yaml) ───────────────────────
LOCAL_PATHS = {
    "ph2_dir": None,            # pasta com IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp ...
    "ph2_metadata_txt": None,   # PH2_dataset.txt
    "ham_image_dir": None,      # pasta com ISIC_xxxxxxx.jpg
    "ham_metadata_csv": None,   # HAM10000_metadata
    "ham_mask_dir": None,       # pasta com ISIC_xxxxxxx_segmentation.png (ou use o zip)
    "ham_mask_zip": None,       # archive.zip (HAM10000_segmentations_lesion_tschandl/)
    "output_root": None,        # padrão: experiments/dataset_validation
}
MODELS = ["gpt-5.4-nano", "gemini-3.1-flash-lite"]
# Fingerprint do manifesto gerado em 2026-10-12 (semente 42; independe de caminhos).
EXPECTED_FINGERPRINT = "e577fe88ee6668cb2c59e2f2a9b7bf96b1b857b4dd98f7f808f8b44021cc2fe3"
REFERENCE_MANIFEST = ROOT / "docs" / "dataset_validation" / "pilot_manifest_reference.csv"

REQUIRED_MODULES = {
    "yaml": "pyyaml", "PIL": "Pillow", "numpy": "numpy", "scipy": "scipy", "torch": "torch",
    "torchvision": "torchvision", "langgraph": "langgraph", "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai", "langchain_google_genai": "langchain-google-genai",
    "langchain_ollama": "langchain-ollama", "pydantic": "pydantic",
}
KEY_VARS = {"gpt-5.4-nano": "OPENAI_API_KEY", "gemini-3.1-flash-lite": "GOOGLE_API_KEY"}

logger = logging.getLogger("inference_pilot")


def parse_args():
    p = argparse.ArgumentParser(description="Piloto de inferência PH² × HAM10000 (pago).")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--dry-run", action="store_true", help="Valida ambiente, dados e manifesto; sem chamadas.")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test: processa só N imagens por modelo (a execução completa retoma depois).")
    p.add_argument("--allow-fingerprint-mismatch", action="store_true",
                   help="Continua mesmo se o manifesto reconstruído difere do esperado (não recomendado).")
    for k in LOCAL_PATHS:
        p.add_argument("--" + k.replace("_", "-"), default=None)
    return p.parse_args()


def preflight_imports() -> None:
    missing = []
    for mod, pkg in REQUIRED_MODULES.items():
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            missing.append(pkg)
    if missing:
        sys.exit("Dependências ausentes: " + ", ".join(missing) +
                 "\nInstale com: pip install " + " ".join(missing))


def build_overrides(args) -> dict:
    paths = {k: (getattr(args, k) or LOCAL_PATHS[k]) for k in LOCAL_PATHS}
    over: dict = {"inference": {"models": list(args.models)}, "datasets": {"ph2": {}, "ham10000": {}}}
    ph2, ham = over["datasets"]["ph2"], over["datasets"]["ham10000"]
    for src, dst, key in (("ph2_dir", ph2, "dataset_dir"), ("ph2_metadata_txt", ph2, "metadata_txt"),
                          ("ham_image_dir", ham, "image_dir"), ("ham_metadata_csv", ham, "metadata_csv"),
                          ("ham_mask_dir", ham, "mask_dir"), ("ham_mask_zip", ham, "mask_zip")):
        if paths[src]:
            dst[key] = paths[src]
    if paths["ham_mask_dir"] and not paths["ham_mask_zip"]:
        ham["mask_zip"] = None
    if paths["output_root"]:
        over["output"] = {"root_dir": paths["output_root"]}
    return over


def check_inputs(cfg) -> None:
    ph2, ham = cfg["datasets"]["ph2"], cfg["datasets"]["ham10000"]
    need = [("PH² (pasta)", ph2["dataset_dir"]), ("PH2_dataset.txt", ph2["metadata_txt"]),
            ("HAM10000 imagens", ham["image_dir"]), ("HAM10000_metadata", ham["metadata_csv"])]
    bad = [f"  - {n}: {p!r}" for n, p in need if not p or not Path(p).exists()]
    masks_ok = (ham["mask_dir"] and Path(ham["mask_dir"]).is_dir()) or (ham["mask_zip"] and Path(ham["mask_zip"]).exists())
    if not masks_ok:
        bad.append(f"  - máscaras HAM10000: mask_dir={ham['mask_dir']!r} / mask_zip={ham['mask_zip']!r}")
    if bad:
        sys.exit("Caminhos não encontrados (ajuste LOCAL_PATHS no topo do script ou use flags):\n" + "\n".join(bad))


def compare_reference(manifest) -> dict:
    """Compara hashes de imagem/máscara com o manifesto de referência (aviso, não bloqueia)."""
    from src.dataset_validation.io_utils import read_csv
    out = {"reference_present": REFERENCE_MANIFEST.exists(), "image_hash_mismatch": [], "mask_hash_mismatch": []}
    if not REFERENCE_MANIFEST.exists():
        return out
    ref = {r["sample_id"]: r for r in read_csv(REFERENCE_MANIFEST)}
    for r in manifest:
        e = ref.get(r["sample_id"])
        if not e:
            out["image_hash_mismatch"].append(r["sample_id"] + " (ausente na referência)")
            continue
        if e["image_sha256"] != r["image_sha256"]:
            out["image_hash_mismatch"].append(r["sample_id"])
        if e["mask_sha256"] != r["mask_sha256"]:
            out["mask_hash_mismatch"].append(r["sample_id"])
    return out


def zip_outputs(dirs) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = dirs["root"] / f"pilot_inference_outputs_{stamp}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("inference", "evaluation", "manifest"):
            for f in sorted((dirs["root"] / sub).rglob("*")):
                if f.is_file() and "_tmp" not in f.parts:
                    z.write(f, f.relative_to(dirs["root"]).as_posix())
    return str(path)


def main() -> int:
    args = parse_args()
    preflight_imports()
    from src.dataset_validation import analysis, config as cfgmod, inference, labels as L, sampling
    from src.dataset_validation.io_utils import provenance, write_csv, write_json

    cfg = cfgmod.load_config(args.config, build_overrides(args))
    dirs = cfgmod.output_dirs(cfg)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(dirs["inference"] / "run.log")])

    # 1) ambiente
    unknown = [m for m in args.models if m not in KEY_VARS]
    if unknown:
        logger.warning("Modelos fora do piloto original: %s", unknown)
    missing_keys = sorted({KEY_VARS.get(m) or inference.provider_env_var(m) for m in args.models
                           if (KEY_VARS.get(m) or inference.provider_env_var(m))
                           and not os.environ.get(KEY_VARS.get(m) or inference.provider_env_var(m))})
    if missing_keys and not args.dry_run:
        sys.exit("Variáveis de ambiente ausentes: " + ", ".join(missing_keys))
    if missing_keys:
        logger.warning("(dry-run) credenciais ausentes: %s", ", ".join(missing_keys))

    # 2) dados e manifesto determinístico (semente fixa) + verificação contra o esperado
    check_inputs(cfg)
    ph2_records = L.parse_ph2_metadata(cfg["datasets"]["ph2"]["metadata_txt"])
    ham_meta = L.parse_ham_metadata(cfg["datasets"]["ham10000"]["metadata_csv"])
    manifest = sampling.build_manifest(cfg, ph2_records, ham_meta,
                                       mask_cache_dir=dirs["inputs"] / "ham10000_masks")
    fp = sampling.manifest_fingerprint(manifest)
    logger.info("Manifesto: %d imagens, fingerprint %s", len(manifest), fp[:16])
    if fp != EXPECTED_FINGERPRINT and not args.allow_fingerprint_mismatch:
        sys.exit(f"Fingerprint do manifesto difere do esperado ({fp[:16]}… ≠ {EXPECTED_FINGERPRINT[:16]}…).\n"
                 "Os metadados/dados não são os mesmos do piloto aprovado. Nada foi chamado.")
    ref = compare_reference(manifest)
    if ref["image_hash_mismatch"] or ref["mask_hash_mismatch"]:
        logger.warning("Hashes diferem do manifesto de referência (imagens: %d, máscaras: %d) — "
                       "registrado em manifest_meta.json.", len(ref["image_hash_mismatch"]),
                       len(ref["mask_hash_mismatch"]))
    write_csv(dirs["manifest"] / "manifest.csv", manifest, sampling.MANIFEST_FIELDS)
    write_json(dirs["manifest"] / "manifest_meta.json", {
        "seed": cfg["seed"], "n_per_class": cfg["sample"]["n_per_class"], "n_images": len(manifest),
        "fingerprint": fp, "expected_fingerprint": EXPECTED_FINGERPRINT,
        "reference_hash_check": ref, "provenance": provenance(cfg["seed"])})

    est = inference.estimate_cost(cfg, len(manifest))
    write_json(dirs["inference"] / "cost_estimate.json", est)
    logger.info("Chamadas LLM previstas: %d (%s)", est["total_llm_calls"],
                ", ".join(f"{m}: {v['llm_calls']}" for m, v in est["per_model"].items()))
    if args.dry_run:
        logger.info("Dry-run concluído: ambiente, dados e manifesto OK. Nenhuma chamada realizada.")
        return 0

    # 3) inferência + avaliação por modelo
    status = 0
    for model in args.models:
        logger.info("=== %s ===", model)
        path = inference.run_model(cfg, manifest, model, dirs["inference"], limit=args.limit)
        from src.dataset_validation.io_utils import read_json
        meta = read_json(Path(path).with_suffix(".meta.json"))
        if meta["failures"]:
            status = 1
            logger.error("%s: %d imagens falharam: %s", model, len(meta["failures"]), list(meta["failures"]))
        interp = analysis.interpret_results(manifest, read_json(path))
        safe = inference.safe_name(model)
        write_csv(dirs["evaluation"] / f"predictions_{safe}.csv", interp)
        ev = analysis.evaluate_model(interp, cfg["evaluation"])
        write_json(dirs["evaluation"] / f"metrics_{safe}.json", ev)
        prim = cfg["evaluation"]["primary_policy"]
        for ds in ("ph2", "ham10000"):
            m = ev["by_dataset"][ds][prim]
            logger.info("%s | %s | TN/FP/FN/TP=%d/%d/%d/%d | sens=%.2f espec=%.2f | não avaliáveis=%d",
                        model, ds, m["tn"], m["fp"], m["fn"], m["tp"], m["sensitivity"]["estimate"],
                        m["specificity"]["estimate"], m["n_not_evaluable"])
    out_zip = zip_outputs(dirs)
    logger.info("Concluído. Traga de volta: %s", out_zip)
    return status


if __name__ == "__main__":
    sys.exit(main())
