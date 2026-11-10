"""Fixes the FINAL epoch count and mask threshold for the selected
segmentation configuration (stage 4) -- DeepLabV3+-ResNet34, selected by
the researcher from the real 5x4 nested-CV results (see
``docs/reconstruction/16_segmentation_nested_cv_real_run_and_architecture_
selection.md``: Dice medio 0.8687 vs 0.8568 do U-Net, diferenca 0.0119 >
margem de empate de 0.005 -- selecao sem desempate).

This does NOT repeat the 50 nested-CV trainings. It does exactly two
things, both explicitly requested by the researcher instead of reusing
the 5 external-fold thresholds as-is:

1. ``chosen_epochs`` = median of the 5 external folds' already-real
   retrain epoch counts for ``deeplabv3plus`` (read directly from
   ``nested_cv_rehearsal_externalfold{0..4}_results.json``'s
   ``architectures.deeplabv3plus.chosen_epochs`` -- NOT recomputed from
   memory, NOT hardcoded). Uses the same round-half-up median already
   implemented and tested in ``src.segmentation.nested_cv.
   select_epochs_via_median``.

2. ``chosen_threshold`` -- the researcher explicitly rejected averaging
   the 5 external folds' own thresholds (0.50/0.55/0.45/0.50/0.50),
   because each of those was selected using predictions from models at
   DIFFERENT epoch counts (each external fold's own internal-round best
   epochs), not the fixed, final ``chosen_epochs`` from step 1. Instead,
   this script performs a FRESH 5-fold out-of-fold (OOF) threshold
   search AT THE FIXED EPOCH COUNT: for each of the 5 ISIC2018 Task1
   folds (the same frozen ``isic2018_task1_folds.json`` used throughout
   stage 4), train a freshly-initialized DeepLabV3+-ResNet34 for EXACTLY
   ``chosen_epochs`` epochs (no early stopping -- the epoch count is
   already fixed) on the OTHER 4 folds, then collect per-image Dice at
   every threshold in the 0.30-0.70 grid on the held-out fold (a fold
   this specific model never trained on). Dice values are pooled across
   all 5 held-out folds (every one of the 2,594 images contributes
   exactly once, as an out-of-training prediction) and the threshold is
   selected via ``nested_cv.select_threshold_from_internal_dice`` (same
   tie-break rule already used throughout: ties -> closest to 0.5;
   residual symmetric tie flagged, never hidden).

   This is 5 NEW trainings (not 50, not 1) -- the minimum needed for a
   methodologically defensible out-of-fold Dice estimate at the fixed
   epoch count, reusing ``_retrain_final``/``_collect_threshold_dice``
   from ``scripts/run_nested_cv_rehearsal.py`` as-is (not duplicated).
   The resulting 5 per-fold checkpoints are INTERMEDIATE/DISCARDABLE --
   they exist only to produce the pooled Dice-by-threshold table, and
   must never be reused as "the final model" (that is a separate,
   single retrain on ALL 2,594 images -- see
   ``scripts/run_segmentation_final_retrain.py``).

HAM10000 is never read or referenced here.

STATUS: implementada, aguardando execucao do pesquisador (Mac/MPS). Nao
executado pelo assistente que escreveu este script.

Usage:
    python scripts/run_segmentation_final_config_fit.py --device mps \\
        --images-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip \\
        --masks-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.segmentation import folds as folds_mod  # noqa: E402
from src.segmentation import nested_cv  # noqa: E402
from src.segmentation.isic2018_dataset import verify_image_mask_correspondence  # noqa: E402

from run_isic2018_unet_pilot import _ensure_folds  # noqa: E402
from run_segmentation_arch_pilot import _resolve_device  # noqa: E402
from run_nested_cv_rehearsal import (  # noqa: E402
    _filter_folds_data_to_matched,
    _retrain_final,
    _collect_threshold_dice,
)

ARCHITECTURE = "deeplabv3plus"  # unica arquitetura aprovada para esta etapa -- ver docstring
DEFAULT_IMAGES_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip"
DEFAULT_MASKS_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation")
DEFAULT_FOLDS_PATH = str(REPO_ROOT / "experiments" / "segmentation" / "isic2018_task1_folds.json")
DEFAULT_FOLD_RESULTS_PATTERN = str(
    REPO_ROOT / "experiments" / "segmentation" / "nested_cv_rehearsal_externalfold{i}_results.json"
)
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_SEED = 42
N_EXTERNAL_FOLDS = 5


def _load_chosen_epochs_sources(pattern: str) -> list[dict[str, Any]]:
    """Reads the 5 real external-fold result files and pulls out
    ``architectures.deeplabv3plus.chosen_epochs`` from each -- the exact
    epoch count each external fold's own internal rounds produced (via
    median-of-4, already real, already computed). Fails loudly (does not
    skip/substitute) if any of the 5 expected files is missing or
    malformed -- the median in step 1 requires all 5."""
    sources = []
    for i in range(N_EXTERNAL_FOLDS):
        path = Path(pattern.format(i=i))
        if not path.exists():
            raise FileNotFoundError(
                f"Resultado da rodada externa {i} nao encontrado em {path} -- "
                "as 5 rodadas externas reais precisam existir antes de fixar a "
                "epoca final (ver docs/reconstruction/16_..._architecture_selection.md)."
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        arch_block = data["architectures"].get(ARCHITECTURE)
        if arch_block is None:
            raise KeyError(f"{path} nao tem resultados para a arquitetura '{ARCHITECTURE}'.")
        sources.append({
            "external_fold": i,
            "source_file": str(path),
            "chosen_epochs": arch_block["chosen_epochs"],
        })
    return sources


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-zip", default=DEFAULT_IMAGES_ZIP)
    parser.add_argument("--masks-zip", default=DEFAULT_MASKS_ZIP)
    parser.add_argument("--folds-path", default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--fold-results-pattern", default=DEFAULT_FOLD_RESULTS_PATTERN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (padrao: auto-detecta cuda>mps>cpu)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--seed", type=int, default=folds_mod.DEFAULT_SEED, help="semente do arquivo de folds")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        import torch as _torch
        args.device = "cuda" if _torch.cuda.is_available() else (
            "mps" if getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available() else "cpu"
        )
    device = _resolve_device(args.device)
    print(f"=== Fixacao de epoca/limiar final -- {ARCHITECTURE} === dispositivo={device}")

    print("[1/3] lendo chosen_epochs das 5 rodadas externas reais...")
    epoch_sources = _load_chosen_epochs_sources(args.fold_results_pattern)
    for s in epoch_sources:
        print(f"      fold {s['external_fold']}: chosen_epochs={s['chosen_epochs']} ({s['source_file']})")
    chosen_epochs = nested_cv.select_epochs_via_median([s["chosen_epochs"] for s in epoch_sources])
    print(f"      mediana (round-half-up) = {chosen_epochs} epocas -- FIXADO para o retreino final")

    print("[2/3] verificando correspondencia imagem<->mascara e carregando folds congelados...")
    correspondence = verify_image_mask_correspondence(args.images_zip, args.masks_zip)
    folds_data = _ensure_folds(args.images_zip, args.folds_path, folds_mod.N_FOLDS, args.seed)
    matched_ids = set(folds_data["image_to_fold"].keys()) - set(correspondence["only_in_images"])
    folds_data = _filter_folds_data_to_matched(folds_data, matched_ids)
    n_folds = folds_data["n_folds"]

    print(f"[3/3] busca de limiar out-of-fold (5 treinos novos, {chosen_epochs} epocas fixas cada, sem early stopping)...")
    combined_dice_by_threshold: dict[float, list[float]] = {t: [] for t in nested_cv.THRESHOLD_GRID}
    per_fold_oof: list[dict[str, Any]] = []

    for held_out_fold in range(n_folds):
        train_ids = nested_cv.build_retrain_ids(folds_data, held_out_fold)
        held_out_ids = nested_cv.build_external_fold_ids(folds_data, held_out_fold)
        print(f"      [fold OOF {held_out_fold + 1}/{n_folds}] treino={len(train_ids)} imagens, "
              f"held_out={len(held_out_ids)} imagens...")

        t0 = time.time()
        model, training_time_s, ckpt_path = _retrain_final(
            ARCHITECTURE, train_ids, args.images_zip, args.masks_zip, device,
            chosen_epochs=chosen_epochs, lr=args.lr, seed=args.train_seed + held_out_fold,
            output_dir=output_dir, external_fold=f"oof{held_out_fold}",
        )
        dice_by_threshold, n_skipped = _collect_threshold_dice(
            model, held_out_ids, args.images_zip, args.masks_zip, device
        )
        eval_time_s = time.time() - t0 - training_time_s
        for t, values in dice_by_threshold.items():
            combined_dice_by_threshold[t].extend(values)

        per_fold_oof.append({
            "held_out_fold": held_out_fold,
            "n_train": len(train_ids),
            "n_held_out": len(held_out_ids),
            "n_skipped_invalid_reference": n_skipped,
            "training_time_s": training_time_s,
            "eval_time_s": eval_time_s,
            "checkpoint_path": ckpt_path,
            "checkpoint_is_intermediate_discardable": True,
        })
        print(f"      [fold OOF {held_out_fold + 1}/{n_folds}] concluido em {training_time_s:.1f}s treino "
              f"+ {eval_time_s:.1f}s avaliacao")

    threshold_selection = nested_cv.select_threshold_from_internal_dice(combined_dice_by_threshold)
    print(f"\nchosen_threshold={threshold_selection['chosen_threshold']} "
          f"empate={threshold_selection['empate']} empate_simetrico={threshold_selection['empate_simetrico']}")

    report = {
        "architecture": ARCHITECTURE,
        "device_requested": args.device,
        "device_effective": str(device),
        "procedure": (
            "chosen_epochs = mediana (round-half-up) dos chosen_epochs das 5 rodadas "
            "externas reais (nao recomputado, lido dos arquivos fonte). chosen_threshold "
            "= busca de limiar out-of-fold: 5 treinos novos (nao os 50 originais), cada um "
            "com chosen_epochs fixo, sem early stopping, no conjunto de 4 folds; Dice por "
            "imagem no quinto fold (nunca visto por aquele treino) pooled nas 5 repeticoes "
            "e avaliado na grade 0.30-0.70. Os limiares das 5 rodadas externas originais "
            "NAO foram promediados nem reutilizados."
        ),
        "chosen_epochs": chosen_epochs,
        "chosen_epochs_sources": epoch_sources,
        "threshold_grid": list(nested_cv.THRESHOLD_GRID),
        "threshold_selection": threshold_selection,
        "oof_folds": per_fold_oof,
        "image_mask_correspondence": {
            k: v for k, v in correspondence.items() if k not in ("only_in_images", "only_in_masks")
        },
        "ham10000_touched": False,
        "note_intermediate_checkpoints": (
            "Os 5 checkpoints de held_out_fold acima sao INTERMEDIARIOS/DESCARTAVEIS "
            "(so para a busca de limiar) -- o modelo final usado no protocolo e um "
            "retreino separado em TODAS as 2594 imagens, ver "
            "scripts/run_segmentation_final_retrain.py."
        ),
    }

    results_path = output_dir / "final_config_fit_results.json"
    report_path = output_dir / "final_config_fit_report.md"
    log_path = output_dir / "final_config_fit.log"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md_lines = [
        f"# Fixacao de epoca/limiar final -- {ARCHITECTURE}",
        "",
        f"- Epocas fixadas: **{chosen_epochs}** (mediana round-half-up de "
        f"{[s['chosen_epochs'] for s in epoch_sources]})",
        f"- Limiar escolhido (busca out-of-fold, 5 treinos novos): **{threshold_selection['chosen_threshold']}**",
        f"- Empate: {threshold_selection['empate']} / Empate simetrico residual: {threshold_selection['empate_simetrico']}",
        "",
        "| limiar | dice medio out-of-fold |",
        "|---|---|",
    ] + [
        f"| {t} | {threshold_selection['mean_dice_by_threshold'].get(t, 'N/A')} |"
        for t in nested_cv.THRESHOLD_GRID
    ] + [
        "",
        "## Procedimento (registrado, nao reaberto)",
        "",
        report["procedure"],
        "",
        f"HAM10000 tocado: {report['ham10000_touched']}",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    print(f"\nresultados (json): {results_path}")
    print(f"relatorio (md): {report_path}")
    print(f"log: {log_path}")
    print("\nPROXIMO PASSO: scripts/run_segmentation_final_retrain.py (treino final em todas as 2594 imagens).")


if __name__ == "__main__":
    main()
