"""Executor + mechanical REHEARSAL of the nested cross-validation protocol
(stage 4, segmentation) -- one external fold, real training (not mocked).

Implements the researcher-approved protocol (quoted verbatim, not reopened
here): "A execucao cientifica tera cinco folds externos e quatro internos
por rodada externa. Hiperparametros, checkpoint e limiar serao escolhidos
somente nos dados internos. Em cada rodada, o retreino nos quatro folds de
desenvolvimento usara a mediana das melhores epocas internas, sem early
stopping no fold externo. O limiar sera escolhido pelo Dice medio das
predicoes internas na grade de 0,30 a 0,70, em passos de 0,05; em empate,
use o valor mais proximo de 0,5 e sinalize se ainda houver empate
simetrico." Pure selection logic lives in ``src/segmentation/nested_cv.py``
(unit-tested separately); this script is the real-training orchestrator.

SCOPE: this is a REHEARSAL of exactly ONE external round (fixed
--external-fold, default 0), run independently per architecture (U-Net and
DeepLabV3+ each run the same logic -- NO comparison or selection between
them here, that is a later decision gated on this rehearsal passing). It
does NOT run the full 5 external rounds, does NOT select a final
architecture/configuration, does NOT decide a narrowed final grid.
``--max-epochs`` (default 20) is an OPERATIONAL CEILING fixed by the
researcher before any new execution -- NOT an estimate of the optimal
epoch -- with real early stopping (``--patience``, default 5) governing
when a round actually stops; see "PROBE MODE" below and
docs/reconstruction/14_segmentation_nested_cv_consolidation.md section 5.

Per architecture, the flow is:
  1. Load the external fold's ids -- NEVER used before step 6.
  2. 4 internal rounds (each of the 4 development folds as internal
     validation in turn, the other 3 as internal training): train a fresh
     (randomly initialized) model for up to --max-epochs epochs (default
     20 -- an OPERATIONAL CEILING, not an estimate of the optimal epoch:
     the first rehearsal with --max-epochs 2 showed the best epoch was
     ALWAYS epoch 2 in all 8 internal rounds, i.e. 2 was never a real
     peak -- see docs/reconstruction/14_segmentation_nested_cv_consolidation.md
     section 5), with EARLY STOPPING after --patience (default 5)
     consecutive epochs without a strict improvement of the Dice of
     monitoring. After each epoch, evaluate Dice on the round's
     internal-validation set AT TRAINING RESOLUTION (256x256, fixed
     monitoring threshold 0.5) -- a deliberate, documented simplification
     for cheap per-epoch monitoring (the metrics that are actually
     reported in this rehearsal -- the threshold-grid search in step 4
     and the external evaluation in step 6 -- always use the full
     original-resolution pipeline). Track the best epoch (strict
     improvement; ties keep the FIRST epoch). Each round records
     ``stopped_reason``: ``"paciencia_esgotada"`` (patience exhausted) or
     ``"teto_atingido"`` (reached --max-epochs without patience firing --
     NEVER interpreted as convergence, even if Dice was still improving).
  3. chosen_epochs = median of the 4 rounds' best epochs (round-half-up,
     see src/segmentation/nested_cv.py).
  4. Using each round's best-epoch checkpoint, compute per-image Dice at
     the ORIGINAL resolution (restore -> binarize) for every threshold in
     the 0.30-0.70 grid, pooled across the 4 rounds; select the threshold
     (ties -> closest to 0.5; residual symmetric tie -> flagged, never
     silently resolved).
  5. Retrain: a NEW model (reinitialized), trained on the union of the 4
     development folds (never the external fold) for EXACTLY
     chosen_epochs epochs, no early stopping.
  6. External evaluation: the retrained model, run ONCE on the external
     fold, with the chosen threshold, through the full metrics pipeline
     (restore -> binarize -> compute_image_metrics -> aggregate_metrics).

PROBE MODE (--probe-single-round, added 2026-11-08): runs ONLY ONE
internal round (--probe-round-index, default 0, an index among the 4
development folds, NOT the raw fold index) per selected architecture --
NO threshold search, NO retrain, NO external evaluation. A DISCARDABLE
probe to validate the early-stopping rule and measure time/epoch before
committing to the full 5x4 rounds -- its report says so explicitly and it
must never be reused as if it were one of the real nested-CV rounds (its
checkpoint/results live in separate, distinctly-named output files from
the real rehearsal's). Reuses ``_run_internal_round`` (same early
stopping), ``nested_cv.build_internal_rounds``, and the same
device/folds-bootstrap logic -- not duplicated.

HAM10000 is never read or referenced here.

STATUS: implementada, aguardando execucao do pesquisador. Not run by the
assistant/fork that wrote it.

Usage:
    python scripts/run_nested_cv_rehearsal.py --device mps
    python scripts/run_nested_cv_rehearsal.py --architecture unet --max-epochs 20 --patience 5 \\
        --max-images-per-fold 20 --device mps
    python scripts/run_nested_cv_rehearsal.py \\
        --images-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip \\
        --masks-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip \\
        --device mps

    # Sondagem descartavel de uma unica rodada, antes da execucao completa:
    python scripts/run_nested_cv_rehearsal.py --probe-single-round --probe-round-index 0 --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.segmentation import folds as folds_mod  # noqa: E402
from src.segmentation import nested_cv  # noqa: E402
from src.segmentation.isic2018_dataset import (  # noqa: E402
    ISIC2018Task1Dataset,
    load_original_mask,
    verify_image_mask_correspondence,
)
from src.segmentation.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_image_metrics,
    dice_per_image,
    restore_probability_to_original_resolution,
)
from src.segmentation.model_factory import ARCHITECTURES, build_segmentation_model  # noqa: E402

# Reused, not duplicated -- same bootstrap/device logic already validated
# in the architecture-comparison pilot.
from run_segmentation_arch_pilot import (  # noqa: E402
    _best_available_device,
    _ensure_folds,
    _resolve_device,
)

DEFAULT_IMAGES_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip"
DEFAULT_MASKS_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation")
DEFAULT_FOLDS_PATH = str(REPO_ROOT / "experiments" / "segmentation" / "isic2018_task1_folds.json")
DEFAULT_EXTERNAL_FOLD = 0
# Researcher decision 2026-11-08 (docs/reconstruction/14_segmentation_nested_cv_consolidation.md,
# section 5): the first rehearsal used --max-epochs 2, and the best epoch
# was ALWAYS epoch 2 in all 8 internal rounds -- i.e. 2 was never a real
# peak, only the rehearsal's artificial ceiling. 20 is an OPERATIONAL
# CEILING fixed before any new execution (so the rule could not be tuned
# after seeing results), not an estimate of the optimal epoch.
DEFAULT_MAX_EPOCHS = 20
DEFAULT_PATIENCE = 5  # consecutive epochs without improvement before stopping
DEFAULT_MONITOR_THRESHOLD = 0.5
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_SEED = 42  # model init / shuffling seed -- distinct from the folds-assignment seed


def _filter_folds_data_to_matched(folds_data: dict[str, Any], matched_ids: set[str]) -> dict[str, Any]:
    filtered = {k: v for k, v in folds_data["image_to_fold"].items() if k in matched_ids}
    return {**folds_data, "image_to_fold": filtered}


def _truncate_folds_data(folds_data: dict[str, Any], max_per_fold: int | None) -> dict[str, Any]:
    """Caps each fold to its first ``max_per_fold`` ids (sorted,
    deterministic) -- applied consistently BEFORE any phase (external,
    internal rounds, retrain) is built, so the cap is respected everywhere,
    not just in training. ``None`` means no cap (use the full fold)."""
    if max_per_fold is None:
        return folds_data
    ids_by_fold: dict[int, list[str]] = {}
    for image_id, fold in folds_data["image_to_fold"].items():
        ids_by_fold.setdefault(fold, []).append(image_id)
    truncated: dict[str, int] = {}
    for fold, ids in ids_by_fold.items():
        for image_id in sorted(ids)[:max_per_fold]:
            truncated[image_id] = fold
    return {**folds_data, "image_to_fold": truncated}


def _train_one_epoch(model, loader, optimizer, criterion, device) -> None:
    model.train()
    for images, masks, _ids in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()


def _monitor_dice(model, dataset: ISIC2018Task1Dataset, device, threshold: float) -> float:
    """Mean Dice at TRAINING resolution (256x256), using the dataset's own
    resized mask as reference -- a deliberate, documented simplification
    for cheap per-epoch monitoring (see module docstring, step 2). NOT the
    metric that is actually reported anywhere in this rehearsal's output."""
    model.eval()
    dices = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, mask_tensor, _id = dataset[idx]
            logits = model(image_tensor.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            pred_binary = prob > threshold
            ref_binary = mask_tensor.squeeze(0).numpy().astype(bool)
            dices.append(dice_per_image(pred_binary, ref_binary))
    return float(np.mean(dices)) if dices else 0.0


def _run_internal_round(
    architecture: str,
    round_spec: dict[str, Any],
    images_zip: str,
    masks_zip: str,
    device,
    max_epochs: int,
    patience: int,
    lr: float,
    monitor_threshold: float,
    seed: int,
    output_dir: Path,
    external_fold: int,
    checkpoint_prefix: str = "nested_cv_round",
) -> dict[str, Any]:
    train_ids = round_spec["train_ids"]
    val_ids = round_spec["val_ids"]
    val_fold = round_spec["internal_val_fold"]

    if not train_ids or not val_ids:
        raise ValueError(
            f"Rodada interna (val_fold={val_fold}) com treino/validacao vazios -- "
            "verifique --images-zip/--masks-zip/--folds-path/--max-images-per-fold."
        )

    torch.manual_seed(seed)
    model = build_segmentation_model(architecture, img_ch=3, output_ch=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset_train = ISIC2018Task1Dataset(train_ids, images_zip, masks_zip)
    dataset_val = ISIC2018Task1Dataset(val_ids, images_zip, masks_zip)
    # drop_last=True: DeepLabV3+'s ASPP pooling branch (AdaptiveAvgPool2d(1)
    # -> BatchNorm2d) collapses the spatial dims to 1x1, so a batch of size 1
    # leaves BatchNorm2d exactly one value per channel over (N,H,W) in train
    # mode -- torch raises "Expected more than 1 value per channel when
    # training". A trailing partial batch of size 1 (whenever len(train_ids)
    # is not a multiple of the batch size) was triggering this for real on
    # the 4th internal round. batch_size=min(4, len(train_ids)) still
    # guarantees at least one batch survives dropping the remainder, since
    # batch_size never exceeds len(train_ids).
    loader = DataLoader(
        dataset_train, batch_size=min(4, len(train_ids)), shuffle=True, num_workers=0, drop_last=True,
    )

    best_dice = -1.0
    best_epoch = None
    best_state = None
    epoch_history = []
    epochs_without_improvement = 0
    stopped_reason = None

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        _train_one_epoch(model, loader, optimizer, criterion, device)
        monitor_dice = _monitor_dice(model, dataset_val, device, monitor_threshold)
        epoch_history.append({"epoch": epoch, "monitor_dice": monitor_dice})
        if monitor_dice > best_dice:  # strict -- ties keep the FIRST epoch (documented rule)
            best_dice = monitor_dice
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        stopped_reason = nested_cv.early_stopping_check(epoch, max_epochs, epochs_without_improvement, patience)
        if stopped_reason is not None:
            break  # stop IMMEDIATELY -- never keep training past the stop decision
    training_time_s = time.time() - t0

    # Namespaced by external_fold too (fix 2026-11-08): val_fold is an index
    # among the DEVELOPMENT folds, which repeats across different external
    # rounds (e.g. external_fold=2 and external_fold=3 can both have a
    # val_fold=0 internal round) -- without external_fold in the filename,
    # two different external rounds would overwrite each other's checkpoint.
    # checkpoint_prefix additionally separates the discardable single-round
    # PROBE's checkpoint ("nested_cv_probe_round") from a real round's
    # ("nested_cv_round") -- same architecture/external_fold/val_fold values
    # are plausible for both (e.g. probing external_fold=0 after already
    # running the real external_fold=0), so without a distinct prefix the
    # probe could silently overwrite a real round's checkpoint, or vice versa.
    ckpt_path = output_dir / f"{checkpoint_prefix}_{architecture}_extfold{external_fold}_valfold{val_fold}.pt"
    torch.save(
        {"model_state_dict": best_state, "architecture": architecture, "best_epoch": best_epoch},
        ckpt_path,
    )

    return {
        "internal_val_fold": val_fold,
        "train_folds": round_spec["train_folds"],
        "train_ids": train_ids,
        "val_ids": val_ids,
        "epoch_history": epoch_history,
        "best_epoch": best_epoch,
        "best_epoch_monitor_dice": best_dice,
        # "teto_atingido" never means convergence -- only that the Dice
        # curve was still improving (or at least not failing patience)
        # when the operational ceiling was hit.
        "stopped_reason": stopped_reason,
        "training_time_s": training_time_s,
        "checkpoint_path": str(ckpt_path),
    }


def _collect_threshold_dice(
    model, val_ids: list[str], images_zip: str, masks_zip: str, device
) -> tuple[dict[float, list[float]], int]:
    """Per-image Dice at the ORIGINAL resolution for every threshold in
    ``nested_cv.THRESHOLD_GRID``, for the given (already best-epoch)
    model, over ``val_ids``. Images whose original reference mask is
    empty are skipped (same validity rule as
    ``compute_image_metrics``'s ``"referencia_invalida"`` status) and
    counted separately -- never silently included as if valid."""
    model.eval()
    dataset_val = ISIC2018Task1Dataset(val_ids, images_zip, masks_zip)
    dice_by_threshold: dict[float, list[float]] = {t: [] for t in nested_cv.THRESHOLD_GRID}
    n_skipped_invalid_reference = 0

    with torch.no_grad():
        for idx, image_id in enumerate(val_ids):
            image_tensor, _mask_256, _id = dataset_val[idx]
            logits = model(image_tensor.unsqueeze(0).to(device))
            prob_256 = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            ref_mask_original = load_original_mask(masks_zip, image_id)
            if not ref_mask_original.any():
                n_skipped_invalid_reference += 1
                continue
            prob_original = restore_probability_to_original_resolution(prob_256, ref_mask_original.shape)
            for t in nested_cv.THRESHOLD_GRID:
                pred_binary = prob_original > t
                dice_by_threshold[t].append(dice_per_image(pred_binary, ref_mask_original))

    return dice_by_threshold, n_skipped_invalid_reference


def _retrain_final(
    architecture: str,
    retrain_ids: list[str],
    images_zip: str,
    masks_zip: str,
    device,
    chosen_epochs: int,
    lr: float,
    seed: int,
    output_dir: Path,
    external_fold: int,
):
    if not retrain_ids:
        raise ValueError("Conjunto de retreino vazio -- verifique a configuracao dos folds.")

    torch.manual_seed(seed)
    model = build_segmentation_model(architecture, img_ch=3, output_ch=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset_train = ISIC2018Task1Dataset(retrain_ids, images_zip, masks_zip)
    # drop_last=True -- same BatchNorm/batch-size-1 reason as _run_internal_round.
    loader = DataLoader(
        dataset_train, batch_size=min(4, len(retrain_ids)), shuffle=True, num_workers=0, drop_last=True,
    )

    t0 = time.time()
    for _epoch in range(1, chosen_epochs + 1):
        _train_one_epoch(model, loader, optimizer, criterion, device)
    training_time_s = time.time() - t0

    # Namespaced by external_fold too (fix 2026-11-08) -- otherwise the
    # retrain checkpoint of one external round overwrites another's.
    ckpt_path = output_dir / f"nested_cv_retrain_{architecture}_extfold{external_fold}.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "architecture": architecture, "epochs": chosen_epochs},
        ckpt_path,
    )
    return model, training_time_s, str(ckpt_path)


def _run_external_evaluation(
    model, external_ids: list[str], images_zip: str, masks_zip: str, device, chosen_threshold: float
) -> dict[str, Any]:
    model.eval()
    dataset_ext = ISIC2018Task1Dataset(external_ids, images_zip, masks_zip)
    per_image = []

    t0 = time.time()
    with torch.no_grad():
        for idx, image_id in enumerate(external_ids):
            image_tensor, _mask_256, _id = dataset_ext[idx]
            logits = model(image_tensor.unsqueeze(0).to(device))
            prob_256 = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            ref_mask_original = load_original_mask(masks_zip, image_id)
            prob_original = restore_probability_to_original_resolution(prob_256, ref_mask_original.shape)
            result = compute_image_metrics(prob_original, ref_mask_original, threshold=chosen_threshold)
            result["image_id"] = image_id
            result["original_shape"] = list(ref_mask_original.shape)
            per_image.append(result)
    evaluation_time_s = time.time() - t0

    return {
        "chosen_threshold": chosen_threshold,
        "per_image": per_image,
        "aggregate": aggregate_metrics(per_image),
        "evaluation_time_s": evaluation_time_s,
    }


def run_rehearsal_for_architecture(
    architecture: str, folds_data: dict[str, Any], images_zip: str, masks_zip: str, device, args, output_dir: Path
) -> dict[str, Any]:
    external_ids = nested_cv.build_external_fold_ids(folds_data, args.external_fold)
    internal_rounds_spec = nested_cv.build_internal_rounds(folds_data, args.external_fold)
    retrain_ids = nested_cv.build_retrain_ids(folds_data, args.external_fold)

    round_results = []
    combined_dice_by_threshold: dict[float, list[float]] = {t: [] for t in nested_cv.THRESHOLD_GRID}
    total_skipped_invalid_ref = 0

    for i, round_spec in enumerate(internal_rounds_spec):
        print(f"      [rodada interna {i + 1}/4] val_fold={round_spec['internal_val_fold']} ...")
        round_result = _run_internal_round(
            architecture, round_spec, images_zip, masks_zip, device,
            max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
            monitor_threshold=args.monitor_threshold,
            seed=args.train_seed + i, output_dir=output_dir,
            external_fold=args.external_fold,
        )
        round_results.append(round_result)
        print(
            f"      [rodada interna {i + 1}/4] melhor epoca={round_result['best_epoch']} "
            f"dice_monitor={round_result['best_epoch_monitor_dice']:.4f} "
            f"motivo_parada={round_result['stopped_reason']} "
            f"epocas_rodadas={len(round_result['epoch_history'])} "
            f"tempo={round_result['training_time_s']:.1f}s"
        )

        best_model = build_segmentation_model(architecture, img_ch=3, output_ch=1)
        ckpt = torch.load(round_result["checkpoint_path"], map_location="cpu")
        best_model.load_state_dict(ckpt["model_state_dict"])
        best_model.to(device)

        dice_by_threshold, n_skipped = _collect_threshold_dice(
            best_model, round_spec["val_ids"], images_zip, masks_zip, device
        )
        total_skipped_invalid_ref += n_skipped
        for t, values in dice_by_threshold.items():
            combined_dice_by_threshold[t].extend(values)

    chosen_epochs = nested_cv.select_epochs_via_median([r["best_epoch"] for r in round_results])
    threshold_selection = nested_cv.select_threshold_from_internal_dice(combined_dice_by_threshold)
    print(
        f"      chosen_epochs={chosen_epochs} chosen_threshold={threshold_selection['chosen_threshold']} "
        f"empate={threshold_selection['empate']} empate_simetrico={threshold_selection['empate_simetrico']}"
    )

    print(f"      [retreino] {len(retrain_ids)} imagens, {chosen_epochs} epocas, sem early stopping...")
    retrained_model, retrain_time_s, retrain_ckpt_path = _retrain_final(
        architecture, retrain_ids, images_zip, masks_zip, device,
        chosen_epochs=chosen_epochs, lr=args.lr, seed=args.train_seed + 100, output_dir=output_dir,
        external_fold=args.external_fold,
    )
    print(f"      [retreino] concluido em {retrain_time_s:.1f}s")

    print(f"      [avaliacao externa] {len(external_ids)} imagens, limiar={threshold_selection['chosen_threshold']}...")
    external_eval = _run_external_evaluation(
        retrained_model, external_ids, images_zip, masks_zip, device,
        chosen_threshold=threshold_selection["chosen_threshold"],
    )
    agg = external_eval["aggregate"]
    print(
        f"      [avaliacao externa] dice_medio={agg['mean_dice']} iou_medio={agg['mean_iou']} "
        f"falha_isic={agg['isic_failure_rate']} tempo={external_eval['evaluation_time_s']:.1f}s"
    )

    return {
        "architecture": architecture,
        "external_fold": args.external_fold,
        "external_fold_ids": external_ids,
        "internal_rounds": round_results,
        "n_skipped_invalid_reference_in_threshold_search": total_skipped_invalid_ref,
        "chosen_epochs": chosen_epochs,
        "threshold_selection": threshold_selection,
        "retrain_ids": retrain_ids,
        "retrain_time_s": retrain_time_s,
        "retrain_checkpoint_path": retrain_ckpt_path,
        "external_evaluation": external_eval,
    }


def run_rehearsal(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_epochs < 1:
        raise ValueError(f"--max-epochs deve ser >= 1, recebido {args.max_epochs}")

    is_real_round = args.max_images_per_fold is None
    print(
        f"=== {'RODADA EXTERNA REAL' if is_real_round else 'ENSAIO'} "
        f"(external_fold={args.external_fold}): executor da validacao cruzada aninhada ==="
    )
    if is_real_round:
        print(
            "Fold completo, teto/paciencia conforme o protocolo fixado -- esta execucao "
            "conta como UMA das 5 rodadas externas reais (nao mais um ensaio mecanico), "
            "ver docs/reconstruction/14_segmentation_nested_cv_consolidation.md. NAO "
            "seleciona arquitetura/configuracao definitiva por si so -- isso so acontece "
            "apos as 5 rodadas estarem completas. HAM10000 nao e tocado."
        )
    else:
        print(
            "LEMBRETE: --max-images-per-fold ativo -- isto e um ENSAIO/verificacao rapida "
            "com amostra reduzida, NAO conta como rodada externa real. HAM10000 nao e tocado."
        )
    device = _resolve_device(args.device)
    print(f"dispositivo solicitado={args.device} efetivo={device}")

    print("[fold] verificando correspondencia imagem<->mascara...")
    correspondence = verify_image_mask_correspondence(args.images_zip, args.masks_zip)
    print(
        f"      imagens={correspondence['n_images']} mascaras={correspondence['n_masks']} "
        f"pareadas={correspondence['n_matched']}"
    )

    folds_data = _ensure_folds(args.images_zip, args.folds_path, folds_mod.N_FOLDS, args.seed)
    matched_ids = set(folds_data["image_to_fold"].keys()) - set(correspondence["only_in_images"])
    folds_data = _filter_folds_data_to_matched(folds_data, matched_ids)
    folds_data = _truncate_folds_data(folds_data, args.max_images_per_fold)

    if not (0 <= args.external_fold < folds_data["n_folds"]):
        raise ValueError(
            f"--external-fold deve estar entre 0 e {folds_data['n_folds'] - 1}, recebido {args.external_fold}"
        )

    architectures = ARCHITECTURES if args.architecture == "both" else (args.architecture,)

    report: dict[str, Any] = {
        "device_requested": args.device,
        "device_effective": str(device),
        "external_fold": args.external_fold,
        "max_epochs_rehearsal": args.max_epochs,
        "patience": args.patience,
        "monitor_threshold": args.monitor_threshold,
        "train_seed": args.train_seed,
        "lr": args.lr,
        "max_images_per_fold": args.max_images_per_fold,
        "folds_file": str(args.folds_path),
        "fold_grouping_key": "image:{image_id} (sem lesion_id -- ver src/segmentation/folds.py)",
        "threshold_grid": list(nested_cv.THRESHOLD_GRID),
        "image_mask_correspondence": {
            k: v for k, v in correspondence.items() if k not in ("only_in_images", "only_in_masks")
        },
        "architectures": {},
        "ham10000_touched": False,
        "is_real_external_round": is_real_round,
        "aviso": (
            (
                "RODADA EXTERNA REAL (parte das 5 rodadas da execucao cientifica), "
                f"fold completo, teto de {args.max_epochs} epocas / paciencia "
                f"{args.patience} conforme o protocolo fixado -- ver "
                "docs/reconstruction/14_segmentation_nested_cv_consolidation.md. "
                "NAO seleciona arquitetura/configuracao definitiva por si so (isso "
                "so acontece apos as 5 rodadas estarem completas). HAM10000 nao foi tocado."
            ) if is_real_round else (
                "ENSAIO/verificacao rapida com --max-images-per-fold ativo (amostra "
                f"reduzida) por ate {args.max_epochs} epocas -- NAO conta como rodada "
                "externa real, NAO seleciona arquitetura ou configuracao definitiva. "
                "HAM10000 nao foi tocado."
            )
        ),
    }

    for arch in architectures:
        print(f"\n--- arquitetura: {arch} ---")
        report["architectures"][arch] = run_rehearsal_for_architecture(
            arch, folds_data, args.images_zip, args.masks_zip, device, args, output_dir
        )

    # Namespaced by external_fold (fix 2026-11-08): the researcher runs one
    # external round at a time (0, 1, 2, ...) -- fixed filenames would make
    # each new run overwrite the previous external round's report.
    ef = args.external_fold
    results_json_path = output_dir / f"nested_cv_rehearsal_externalfold{ef}_results.json"
    report_md_path = output_dir / f"nested_cv_rehearsal_externalfold{ef}_report.md"
    log_path = output_dir / f"nested_cv_rehearsal_externalfold{ef}.log"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(_render_report_md(report))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    print(f"\nresultados (json) gravados: {results_json_path}")
    print(f"relatorio (md) gravado: {report_md_path}")
    print(f"log gravado: {log_path}")
    print(
        f"\nLEMBRETE FINAL: {'rodada externa REAL' if is_real_round else 'ensaio/verificacao rapida'} "
        f"(external_fold={args.external_fold}). HAM10000 nao foi tocado nesta execucao."
    )


def run_single_round_probe(args: argparse.Namespace) -> None:
    """DISCARDABLE probe: trains exactly ONE internal round (with the same
    real training + early stopping as the full rehearsal), no threshold
    search, no retrain, no external evaluation. Writes to separate,
    distinctly-named output files -- never mixed with the real
    rehearsal's. See module docstring, "PROBE MODE"."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_epochs < 1:
        raise ValueError(f"--max-epochs deve ser >= 1, recebido {args.max_epochs}")

    print("=== SONDAGEM DE RODADA UNICA (DESCARTAVEL) ===")
    print(
        "AVISO: isto NAO e uma das rodadas reais da validacao aninhada -- serve "
        "so para validar a regra de parada e medir tempo antes da execucao "
        "completa (5x4 rodadas). HAM10000 nao e tocado."
    )
    device = _resolve_device(args.device)
    print(f"dispositivo solicitado={args.device} efetivo={device}")

    correspondence = verify_image_mask_correspondence(args.images_zip, args.masks_zip)
    folds_data = _ensure_folds(args.images_zip, args.folds_path, folds_mod.N_FOLDS, args.seed)
    matched_ids = set(folds_data["image_to_fold"].keys()) - set(correspondence["only_in_images"])
    folds_data = _filter_folds_data_to_matched(folds_data, matched_ids)
    folds_data = _truncate_folds_data(folds_data, args.max_images_per_fold)

    if not (0 <= args.external_fold < folds_data["n_folds"]):
        raise ValueError(
            f"--external-fold deve estar entre 0 e {folds_data['n_folds'] - 1}, recebido {args.external_fold}"
        )

    internal_rounds_spec = nested_cv.build_internal_rounds(folds_data, args.external_fold)
    if not (0 <= args.probe_round_index < len(internal_rounds_spec)):
        raise ValueError(
            f"--probe-round-index deve estar entre 0 e {len(internal_rounds_spec) - 1} "
            f"(indice entre os {len(internal_rounds_spec)} folds de desenvolvimento), "
            f"recebido {args.probe_round_index}"
        )
    round_spec = internal_rounds_spec[args.probe_round_index]

    architectures = ARCHITECTURES if args.architecture == "both" else (args.architecture,)

    report: dict[str, Any] = {
        "aviso": (
            "SONDAGEM DESCARTAVEL -- nao e uma das rodadas reais da validacao "
            "aninhada; serve so para validar a regra de parada e medir tempo "
            "antes da execucao completa (5x4 rodadas). NAO rodou busca de "
            "limiar, retreino ou avaliacao externa."
        ),
        "device_requested": args.device,
        "device_effective": str(device),
        "external_fold": args.external_fold,
        "probe_round_index": args.probe_round_index,
        "internal_val_fold": round_spec["internal_val_fold"],
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "monitor_threshold": args.monitor_threshold,
        "lr": args.lr,
        "ham10000_touched": False,
        "architectures": {},
    }

    for arch in architectures:
        print(f"\n--- arquitetura: {arch} (sondagem, val_fold={round_spec['internal_val_fold']}) ---")
        round_result = _run_internal_round(
            arch, round_spec, args.images_zip, args.masks_zip, device,
            max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
            monitor_threshold=args.monitor_threshold, seed=args.train_seed, output_dir=output_dir,
            external_fold=args.external_fold, checkpoint_prefix="nested_cv_probe_round",
        )
        n_epochs_run = len(round_result["epoch_history"])
        round_result["n_epochs_run"] = n_epochs_run
        round_result["mean_time_per_epoch_s"] = (
            round_result["training_time_s"] / n_epochs_run if n_epochs_run else None
        )
        print(
            f"      melhor_epoca={round_result['best_epoch']} motivo_parada={round_result['stopped_reason']} "
            f"epocas_rodadas={n_epochs_run} tempo_total={round_result['training_time_s']:.1f}s "
            f"tempo_medio_epoca={round_result['mean_time_per_epoch_s']:.1f}s"
        )
        report["architectures"][arch] = round_result

    results_json_path = output_dir / "nested_cv_single_round_probe_results.json"
    report_md_path = output_dir / "nested_cv_single_round_probe_report.md"
    log_path = output_dir / "nested_cv_single_round_probe.log"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(_render_probe_report_md(report))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    print(f"\nresultados (json) gravados: {results_json_path}")
    print(f"relatorio (md) gravado: {report_md_path}")
    print(f"log gravado: {log_path}")
    print(
        "\nLEMBRETE: sondagem descartavel -- NAO reaproveitar como se fosse uma "
        "rodada real da validacao aninhada."
    )


def _render_probe_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Sondagem de rodada única (descartável)",
        "",
        f"**{report['aviso']}**",
        "",
        f"- Dispositivo solicitado: `{report['device_requested']}` / efetivo: `{report['device_effective']}`",
        f"- Fold externo: {report['external_fold']} (não usado nesta sondagem)",
        f"- Índice da rodada de sondagem: {report['probe_round_index']} "
        f"(fold de validação interna: {report['internal_val_fold']})",
        f"- max_epochs: {report['max_epochs']} / patience: {report['patience']}",
        f"- Threshold de monitoramento: {report['monitor_threshold']} / lr: {report['lr']}",
        f"- HAM10000 tocado: {report['ham10000_touched']}",
        "",
    ]
    for arch, res in report["architectures"].items():
        lines.append(f"## Arquitetura: `{arch}`")
        lines.append("")
        lines.append(f"- Melhor época: {res['best_epoch']} (Dice monitor: {res['best_epoch_monitor_dice']:.4f})")
        lines.append(f"- **Motivo da parada**: {res['stopped_reason']}")
        lines.append(f"- Épocas efetivamente rodadas: {res['n_epochs_run']} de até {report['max_epochs']}")
        lines.append(
            f"- Tempo total: {res['training_time_s']:.1f}s / "
            f"tempo médio por época: {res['mean_time_per_epoch_s']:.1f}s"
        )
        lines.append(f"- Checkpoint: `{res['checkpoint_path']}`")
        lines.append("")
        lines.append("| época | dice_monitor |")
        lines.append("|---|---|")
        for e in res["epoch_history"]:
            lines.append(f"| {e['epoch']} | {e['monitor_dice']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        (
            f"# Rodada externa {report['external_fold']} — validação cruzada aninhada"
            if report.get("is_real_external_round")
            else f"# Ensaio — validação cruzada aninhada (rodada externa {report['external_fold']})"
        ),
        "",
        f"**{report['aviso']}**",
        "",
        f"- Dispositivo solicitado: `{report['device_requested']}` / efetivo: `{report['device_effective']}`",
        f"- Fold externo: {report['external_fold']}",
        f"- Teto operacional de épocas por rodada interna: {report['max_epochs_rehearsal']} "
        f"(early stopping com paciência={report['patience']} -- teto NÃO é estimativa de convergência)",
        f"- Threshold de monitoramento (época a época, resolução de treino): {report['monitor_threshold']}",
        f"- Grade de limiar (resolução original): {report['threshold_grid']}",
        f"- Seed de treino: {report['train_seed']} / lr: {report['lr']}",
        f"- Limite de imagens por fold: {report['max_images_per_fold']}",
        f"- Arquivo de folds: `{report['folds_file']}`",
        f"- Chave de agrupamento dos folds: {report['fold_grouping_key']}",
        f"- HAM10000 tocado: {report['ham10000_touched']}",
        "",
    ]
    corr = report["image_mask_correspondence"]
    lines += [
        "## Correspondência imagem↔máscara",
        "",
        f"- Imagens: {corr['n_images']} / Máscaras: {corr['n_masks']} / Pareadas: {corr['n_matched']} "
        f"/ Correspondência perfeita: {corr['ok']}",
        "",
    ]

    for arch, res in report["architectures"].items():
        lines.append(f"## Arquitetura: `{arch}`")
        lines.append("")
        lines.append(f"- Fold externo (NUNCA usado antes da avaliação final): {res['external_fold_ids']}")
        lines.append(
            f"- Imagens ignoradas por referência inválida na busca de limiar: "
            f"{res['n_skipped_invalid_reference_in_threshold_search']}"
        )
        lines.append("")
        lines.append("### Rodadas internas")
        lines.append("")
        lines.append(
            "| val_fold | train_folds | n_treino | n_val | épocas_rodadas | melhor_época | "
            "dice_monitor | motivo_parada | tempo(s) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in res["internal_rounds"]:
            lines.append(
                f"| {r['internal_val_fold']} | {r['train_folds']} | {len(r['train_ids'])} | "
                f"{len(r['val_ids'])} | {len(r['epoch_history'])} | {r['best_epoch']} | "
                f"{r['best_epoch_monitor_dice']:.4f} | {r['stopped_reason']} | "
                f"{r['training_time_s']:.1f} |"
            )
        lines.append("")
        lines.append(f"- IDs de treino/val por rodada (rastreabilidade completa):")
        for r in res["internal_rounds"]:
            lines.append(f"  - val_fold {r['internal_val_fold']}: val_ids={r['val_ids']}")
            lines.append(f"    train_ids={r['train_ids']}")
        lines.append("")
        lines.append(f"- **chosen_epochs** (mediana das melhores épocas, round-half-up): {res['chosen_epochs']}")
        lines.append("")
        ts = res["threshold_selection"]
        lines.append("### Seleção de limiar")
        lines.append("")
        lines.append(f"- Médias de Dice por limiar: {ts['mean_dice_by_threshold']}")
        lines.append(f"- Limiar escolhido: {ts['chosen_threshold']}")
        lines.append(f"- Empate: {ts['empate']} (limiares empatados: {ts['tied_thresholds']})")
        lines.append(f"- **Empate simétrico residual**: {ts['empate_simetrico']}")
        lines.append("")
        lines.append(f"### Retreino (4 folds de desenvolvimento, {res['chosen_epochs']} épocas, sem early stopping)")
        lines.append("")
        lines.append(f"- IDs de retreino: {res['retrain_ids']}")
        lines.append(f"- Tempo de retreino: {res['retrain_time_s']:.1f}s")
        lines.append(f"- Checkpoint: `{res['retrain_checkpoint_path']}`")
        lines.append("")
        ext = res["external_evaluation"]
        agg = ext["aggregate"]
        lines.append("### Avaliação externa (única, com o limiar escolhido)")
        lines.append("")
        lines.append(f"- Tempo de avaliação: {ext['evaluation_time_s']:.1f}s")
        lines.append(f"- Casos: total={agg['n_total']} ok={agg['n_ok']} por status={agg['counts_by_status']}")
        lines.append(f"- Dice médio: {agg['mean_dice']}")
        lines.append(f"- IoU médio: {agg['mean_iou']}")
        lines.append(f"- Thresholded Jaccard médio: {agg['mean_thresholded_jaccard']}")
        lines.append(f"- Taxa de falha ISIC: {agg['isic_failure_rate']}")
        lines.append(
            f"- HD95 mediana (finitos): {agg['hd95_median_finite']} / indefinidos: {agg['hd95_n_undefined']}"
        )
        lines.append(
            f"- HD95 normalizado pela diagonal, mediana (finitos): "
            f"{agg['hd95_normalizado_diagonal_median_finite']}"
        )
        lines.append("")
        lines.append(
            "| image_id | status | dim. original | dice | iou | hd95 | hd95_norm_diag | "
            "pred_pos_px | ref_pos_px |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for pr in ext["per_image"]:
            lines.append(
                f"| `{pr.get('image_id')}` | {pr['status']} | {pr.get('original_shape')} | "
                f"{pr.get('dice')} | {pr.get('iou')} | {pr.get('hd95')} | "
                f"{pr.get('hd95_normalizado_diagonal')} | {pr.get('pred_positive_pixels')} | "
                f"{pr.get('ref_positive_pixels')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", choices=["unet", "deeplabv3plus", "both"], default="both")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default=_best_available_device(),
        help="Se omitido, auto-detecta cuda > mps > cpu. Se passado explicitamente e "
        "o dispositivo nao estiver disponivel, levanta erro -- nunca cai silenciosamente para cpu.",
    )
    parser.add_argument(
        "--images-zip", default=DEFAULT_IMAGES_ZIP,
        help="Caminho do zip de imagens. Default e do Windows do pesquisador -- no Mac, passe o caminho real.",
    )
    parser.add_argument(
        "--masks-zip", default=DEFAULT_MASKS_ZIP,
        help="Caminho do zip de mascaras. Default e do Windows do pesquisador -- no Mac, passe o caminho real.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds-path", default=DEFAULT_FOLDS_PATH)
    parser.add_argument(
        "--seed", type=int, default=folds_mod.DEFAULT_SEED,
        help="Seed dos 5 folds fixos -- so usada se o arquivo de folds ainda nao existir "
        "(bootstrap). NAO regenera um arquivo de folds ja existente.",
    )
    parser.add_argument(
        "--train-seed", type=int, default=DEFAULT_TRAIN_SEED,
        help="Seed de inicializacao de modelo / embaralhamento do treino (distinta da seed dos folds).",
    )
    parser.add_argument(
        "--external-fold", type=int, default=DEFAULT_EXTERNAL_FOLD,
        help="Indice do fold (0-4) tratado como fold externo -- usado SO na avaliacao final (passo 6).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS,
        help="Teto OPERACIONAL de epocas por rodada interna (e do retreino, via mediana) -- "
        "NAO e uma estimativa da epoca otima (ver docs/reconstruction/"
        "14_segmentation_nested_cv_consolidation.md secao 5). Early stopping "
        "(--patience) normalmente para antes disso.",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help="Numero de epocas consecutivas sem melhora estrita do Dice de monitoramento "
        "antes de interromper a rodada interna (early stopping). Retreino final nao usa isso.",
    )
    parser.add_argument(
        "--monitor-threshold", type=float, default=DEFAULT_MONITOR_THRESHOLD,
        help="Threshold FIXO usado so no monitoramento epoca-a-epoca (resolucao de treino, 256x256) -- "
        "NAO e o limiar escolhido por validacao (esse vem da busca na grade, resolucao original).",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--max-images-per-fold", type=int, default=None,
        help="Limite opcional de imagens por fold (amostra determinística, ordem alfabética) "
        "para uma versao mais rapida do ensaio. Default: sem limite (fold completo).",
    )
    parser.add_argument(
        "--probe-single-round", action="store_true", default=False,
        help="Sondagem DESCARTAVEL: roda so UMA rodada interna (--probe-round-index), sem busca "
        "de limiar, retreino ou avaliacao externa. Nao e uma rodada real da validacao aninhada.",
    )
    parser.add_argument(
        "--probe-round-index", type=int, default=0,
        help="Indice (0-3) da rodada interna a sondar, entre os 4 folds de desenvolvimento -- "
        "NAO e o indice bruto do fold. So usado com --probe-single-round.",
    )
    args = parser.parse_args()
    if args.probe_single_round:
        run_single_round_probe(args)
    else:
        run_rehearsal(args)


if __name__ == "__main__":
    main()
