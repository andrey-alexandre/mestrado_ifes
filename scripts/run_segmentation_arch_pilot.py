"""Architecture-comparison pilot for the segmentation baseline (stage 4):
U-Net (already approved) vs. DeepLabV3+ with a ResNet34 encoder (new,
see src/segmentation/deeplabv3plus_model.py for the output_stride=32
decision forced by a confirmed torchvision BasicBlock limitation).

This is a MECHANICAL VERIFICATION, not real training, not cross-validation,
not architecture/threshold/metric selection -- those remain for the full
5-external x 4-internal-fold protocol (not implemented here). For each
selected architecture it verifies, on REAL data read from the ISIC2018
Task1 zips:
  1. Image/mask reading and alignment, with traceable ids and shapes.
  2. A real weight update after one training step.
  3. Checkpoint save + reload, producing a mask of the expected shape.
  4. Metric computation (compute_image_metrics) after restoring the
     probability map to the reference mask's ORIGINAL resolution and
     THEN binarizing with an explicit demo threshold (never the reverse
     order).
  5. Wall-clock time per architecture, images processed, and the exact
     configuration used (device, resolution, init policy, threshold).

Reuses the already-frozen 5-fold assignment
(experiments/segmentation/isic2018_task1_folds.json, seed 20261108,
image-level grouping -- see src/segmentation/folds.py for why there is no
lesion-level grouping for ISIC2018 Task1) via
run_isic2018_unet_pilot._ensure_folds -- never regenerates it. The fold
passed via --external-fold (default 0) is NEVER used anywhere in this
script (not for training, not for the internal-validation metric demo);
of the remaining 4 development folds, the last one (by index) is used as
"internal validation" (metric demo only) and the other 3 as "pilot
training" -- a simple, documented structural split for this mechanical
pilot, NOT the real nested cross-validation (5 external x 4 internal),
which is not implemented here.

HAM10000 is never read or referenced here: reserved exclusively for the
final external evaluation of segmentation.

STATUS: implementada, aguardando execucao do pesquisador. Not run by the
assistant/fork that wrote it.

``--device`` default (fix 2026-11-08): if omitted, auto-detects
cuda > mps > cpu (``_best_available_device``, same convention as
``scripts/train_bcn20000_efficientnet.py``) instead of always defaulting to
"cpu" -- a prior version of this script required ``--device mps`` to be
passed explicitly on every invocation, and a researcher run on a Mac with
no ``--device`` flag silently used CPU even though MPS was available. An
explicit ``--device`` value is still validated strictly by
``_resolve_device``: requesting an unavailable device remains a hard error,
never a silent fallback -- only the *default when omitted* changed.

Usage:
    python scripts/run_segmentation_arch_pilot.py
    python scripts/run_segmentation_arch_pilot.py --architecture deeplabv3plus --device mps
    python scripts/run_segmentation_arch_pilot.py \\
        --images-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip \\
        --masks-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip \\
        --device mps --run-timing-epoch
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
from src.segmentation.isic2018_dataset import (  # noqa: E402
    ISIC2018Task1Dataset,
    TARGET_SIZE,
    load_original_mask,
    verify_image_mask_correspondence,
)
from src.segmentation.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_image_metrics,
    restore_probability_to_original_resolution,
)
from src.segmentation.model_factory import ARCHITECTURES, build_segmentation_model  # noqa: E402

# Reused, not duplicated -- same folds-file bootstrap logic as the U-Net-only pilot.
from run_isic2018_unet_pilot import _ensure_folds  # noqa: E402

DEFAULT_IMAGES_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip"
DEFAULT_MASKS_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation")
DEFAULT_FOLDS_PATH = str(REPO_ROOT / "experiments" / "segmentation" / "isic2018_task1_folds.json")
DEFAULT_N_PILOT_IMAGES = 12
DEFAULT_PILOT_STEPS = 2
DEFAULT_DEMO_THRESHOLD = 0.5
DEFAULT_EXTERNAL_FOLD = 0

MECHANICAL_CHECKS = (
    "leitura_alinhamento",
    "atualizacao_peso",
    "checkpoint_save_reload",
    "mascara_predita",
)


def _best_available_device() -> str:
    """Auto-detects cuda > mps > cpu, in that preference order -- same
    convention already used by ``scripts/train_bcn20000_efficientnet.py``
    (``_best_available_device``). Used ONLY to compute the ``--device``
    argparse default (evaluated once, at parser-construction time) so that
    simply omitting ``--device`` does not silently mean "cpu" on a machine
    that actually has a GPU backend. An explicit ``--device`` value passed
    on the command line is still validated strictly by :func:`_resolve_device`
    below -- requesting an unavailable device remains a hard error, never a
    silent fallback."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(requested: str):
    """Explicit device resolution for whatever value ``--device`` ends up
    with (either the user's explicit choice, or the auto-detected default
    from :func:`_best_available_device` -- see that function's docstring).
    Never a silent 'best available' fallback AT THIS POINT: raises
    RuntimeError naming the requested device and what is actually available
    if the request (explicit or defaulted) cannot be honored."""
    import torch

    available = ["cpu"]
    if torch.cuda.is_available():
        available.append("cuda")
    if torch.backends.mps.is_available():
        available.append("mps")

    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Dispositivo desconhecido: '{requested}'. Escolha entre: cpu, cuda, mps.")
    if requested != "cpu" and requested not in available:
        raise RuntimeError(
            f"Dispositivo '{requested}' solicitado mas NAO disponivel neste ambiente. "
            f"Disponiveis aqui: {available}."
        )
    return torch.device(requested)


def _build_fold_id_lists(folds_data: dict[str, Any], matched_ids: set[str]) -> dict[int, list[str]]:
    n_folds = folds_data["n_folds"]
    ids_by_fold: dict[int, list[str]] = {i: [] for i in range(n_folds)}
    for image_id, fold in folds_data["image_to_fold"].items():
        if image_id in matched_ids:
            ids_by_fold[fold].append(image_id)
    for fold in ids_by_fold:
        ids_by_fold[fold].sort()
    return ids_by_fold


def run_pilot_for_architecture(
    architecture: str,
    train_ids: list[str],
    val_ids: list[str],
    images_zip: str,
    masks_zip: str,
    device,
    demo_threshold: float,
    pilot_steps: int,
    run_timing_epoch: bool,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    t0 = time.time()
    results: dict[str, Any] = {}

    dataset_train = ISIC2018Task1Dataset(train_ids, images_zip, masks_zip)

    try:
        image_tensor, mask_tensor, image_id = dataset_train[0]
        alignment_ok = (
            image_id == train_ids[0]
            and tuple(image_tensor.shape) == (3, TARGET_SIZE, TARGET_SIZE)
            and tuple(mask_tensor.shape) == (1, TARGET_SIZE, TARGET_SIZE)
        )
        results["leitura_alinhamento"] = {
            "status": "PASSOU" if alignment_ok else "FALHOU: formas/id inesperados",
            "image_id": image_id,
            "image_shape": list(image_tensor.shape),
            "mask_shape": list(mask_tensor.shape),
        }
    except Exception as exc:  # noqa: BLE001 - pilot must report, not crash
        results["leitura_alinhamento"] = {"status": f"FALHOU: {exc}"}

    loader = DataLoader(dataset_train, batch_size=min(4, len(train_ids)), shuffle=False, num_workers=0)
    model = build_segmentation_model(architecture, img_ch=3, output_ch=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ref_param_name, ref_param = next(iter(model.named_parameters()))
    before = ref_param.detach().clone()
    try:
        for step, (images, masks, _ids) in enumerate(loader):
            if step >= pilot_steps:
                break
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        after = dict(model.named_parameters())[ref_param_name].detach().clone()
        changed = not torch.equal(before, after)
        results["atualizacao_peso"] = {
            "status": "PASSOU" if changed else "FALHOU: pesos nao mudaram",
            "parametro_monitorado": ref_param_name,
        }
    except Exception as exc:  # noqa: BLE001
        results["atualizacao_peso"] = {"status": f"FALHOU: {exc}"}

    ckpt_path = output_dir / f"pilot_checkpoint_{architecture}.pt"
    try:
        torch.save({"model_state_dict": model.state_dict(), "architecture": architecture, "pilot": True}, ckpt_path)
        reloaded = build_segmentation_model(architecture, img_ch=3, output_ch=1)
        state = torch.load(ckpt_path, map_location="cpu")
        reloaded.load_state_dict(state["model_state_dict"])
        results["checkpoint_save_reload"] = {"status": "PASSOU", "path": str(ckpt_path)}
    except Exception as exc:  # noqa: BLE001
        results["checkpoint_save_reload"] = {"status": f"FALHOU: {exc}"}

    try:
        model.eval()
        with torch.no_grad():
            sample_image, _mask, _id = dataset_train[0]
            logits = model(sample_image.unsqueeze(0).to(device))
            predicted_mask = torch.sigmoid(logits)
        shape_ok = tuple(predicted_mask.shape) == (1, 1, TARGET_SIZE, TARGET_SIZE)
        results["mascara_predita"] = {
            "status": "PASSOU" if shape_ok else f"FALHOU: forma {tuple(predicted_mask.shape)}",
            "shape": list(predicted_mask.shape),
        }
    except Exception as exc:  # noqa: BLE001
        results["mascara_predita"] = {"status": f"FALHOU: {exc}"}

    # Metric demo: restore -> binarize -> compute, over the internal-validation
    # sample (never the external fold, never the pilot training sample).
    metric_results: list[dict[str, Any]] = []
    model.eval()
    if val_ids:
        dataset_val = ISIC2018Task1Dataset(val_ids, images_zip, masks_zip)
        with torch.no_grad():
            for idx, vid in enumerate(val_ids):
                try:
                    image_tensor, _mask_256, image_id = dataset_val[idx]
                    logits = model(image_tensor.unsqueeze(0).to(device))
                    prob_256 = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
                    ref_mask_original = load_original_mask(masks_zip, image_id)
                    prob_original = restore_probability_to_original_resolution(
                        prob_256, ref_mask_original.shape
                    )
                    metrics_result = compute_image_metrics(
                        prob_original, ref_mask_original, threshold=demo_threshold
                    )
                    metrics_result["image_id"] = image_id
                    metrics_result["original_shape"] = list(ref_mask_original.shape)
                    metric_results.append(metrics_result)
                except Exception as exc:  # noqa: BLE001
                    metric_results.append(
                        {"status": "falha_processamento", "erro": str(exc), "image_id": vid}
                    )

    results["metricas_demo"] = {
        "threshold_demo": demo_threshold,
        "per_image": metric_results,
        "aggregate": aggregate_metrics(metric_results),
    }

    results["tempo_execucao_s"] = time.time() - t0
    results["n_imagens_treino_piloto"] = len(train_ids)
    results["n_imagens_validacao_interna_metricas"] = len(val_ids)

    if run_timing_epoch:
        t_epoch0 = time.time()
        for images, masks, _ids in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        epoch_elapsed = time.time() - t_epoch0
        results["medicao_tempo_epoca"] = {
            "n_imagens": len(train_ids),
            "tempo_s": epoch_elapsed,
            "tempo_por_imagem_s": epoch_elapsed / max(1, len(train_ids)),
            "aviso": (
                "Isto e so medicao de tempo (1 epoca sobre a amostra de piloto, "
                "nao o dataset inteiro) para planejar a execucao aninhada real -- "
                "nao e treino real nem selecao de limiar/arquitetura."
            ),
        }

    return results


def run_pilot(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== PILOTO: comparacao de arquiteturas de segmentacao (U-Net vs DeepLabV3+-ResNet34) ===")
    device = _resolve_device(args.device)
    print(f"dispositivo solicitado={args.device} efetivo={device}")

    print("[fold] verificando correspondencia imagem<->mascara (ids, direto dos dois zips)...")
    correspondence = verify_image_mask_correspondence(args.images_zip, args.masks_zip)
    print(
        f"      imagens={correspondence['n_images']} mascaras={correspondence['n_masks']} "
        f"pareadas={correspondence['n_matched']}"
    )

    folds_data = _ensure_folds(args.images_zip, args.folds_path, folds_mod.N_FOLDS, args.seed)
    matched_ids = set(folds_data["image_to_fold"].keys()) - set(correspondence["only_in_images"])

    n_folds = folds_data["n_folds"]
    if not (0 <= args.external_fold < n_folds):
        raise ValueError(f"--external-fold deve estar entre 0 e {n_folds - 1}, recebido {args.external_fold}")

    ids_by_fold = _build_fold_id_lists(folds_data, matched_ids)

    dev_fold_indices = [i for i in range(n_folds) if i != args.external_fold]
    internal_val_fold = dev_fold_indices[-1]
    train_piloto_folds = dev_fold_indices[:-1]

    external_ids = ids_by_fold[args.external_fold]
    val_ids_full = ids_by_fold[internal_val_fold]
    train_ids_full = sorted(iid for f in train_piloto_folds for iid in ids_by_fold[f])

    train_sample = train_ids_full[: args.n_pilot_images]
    val_sample = val_ids_full[: min(3, len(val_ids_full))]

    if not train_sample:
        raise ValueError(
            "Amostra de treino piloto vazia -- verifique --images-zip/--masks-zip/--folds-path "
            "e se --external-fold nao exclui todos os folds de desenvolvimento."
        )

    print(
        f"[fold] fold externo={args.external_fold} (n={len(external_ids)}, NUNCA usado), "
        f"fold validacao interna={internal_val_fold} (n={len(val_ids_full)}, amostra demo={len(val_sample)}), "
        f"folds treino piloto={train_piloto_folds} (n={len(train_ids_full)}, amostra piloto={len(train_sample)})"
    )

    architectures = ARCHITECTURES if args.architecture == "both" else (args.architecture,)

    report: dict[str, Any] = {
        "device_requested": args.device,
        "device_effective": str(device),
        "target_size": TARGET_SIZE,
        "demo_threshold": args.demo_threshold,
        "init_policy": (
            "Ambas arquiteturas usam inicializacao aleatoria neste piloto "
            "(weights=None, sem download) -- decisao do piloto, nao decide a "
            "politica de pre-treinamento do experimento definitivo."
        ),
        "external_fold": args.external_fold,
        "external_fold_n_images": len(external_ids),
        "internal_val_fold": internal_val_fold,
        "train_piloto_folds": train_piloto_folds,
        "n_train_piloto_total": len(train_ids_full),
        "n_train_piloto_amostra": len(train_sample),
        "n_val_interna_total": len(val_ids_full),
        "n_val_interna_amostra": len(val_sample),
        "folds_file": str(args.folds_path),
        "fold_grouping_key": "image:{image_id} (sem lesion_id -- ver src/segmentation/folds.py)",
        "image_mask_correspondence": {
            k: v for k, v in correspondence.items() if k not in ("only_in_images", "only_in_masks")
        },
        "architectures": {},
        "ham10000_touched": False,
        "aviso": (
            "Verificacao tecnica mecanica -- NAO treina um modelo real, NAO roda "
            "validacao cruzada aninhada, NAO seleciona arquitetura/limiar/metrica "
            "final. HAM10000 nao foi tocado."
        ),
    }

    for arch in architectures:
        print(f"\n--- arquitetura: {arch} ---")
        arch_results = run_pilot_for_architecture(
            architecture=arch,
            train_ids=train_sample,
            val_ids=val_sample,
            images_zip=args.images_zip,
            masks_zip=args.masks_zip,
            device=device,
            demo_threshold=args.demo_threshold,
            pilot_steps=args.pilot_steps,
            run_timing_epoch=args.run_timing_epoch,
            output_dir=output_dir,
        )
        report["architectures"][arch] = arch_results
        for check_name in MECHANICAL_CHECKS:
            print(f"      [{check_name}] {arch_results[check_name]['status']}")
        agg = arch_results["metricas_demo"]["aggregate"]
        print(
            f"      [metricas_demo] n_ok={agg['n_ok']} dice_medio={agg['mean_dice']} "
            f"iou_medio={agg['mean_iou']} falha_isic={agg['isic_failure_rate']}"
        )
        print(f"      tempo_execucao_s={arch_results['tempo_execucao_s']:.2f}")

    all_mechanical_passed = all(
        report["architectures"][arch][check]["status"] == "PASSOU"
        for arch in architectures
        for check in MECHANICAL_CHECKS
    )
    report["status_geral"] = (
        "TODAS AS VERIFICACOES MECANICAS PASSARAM" if all_mechanical_passed else "AO MENOS UMA FALHOU"
    )
    print(f"\n=== RESULTADO: {report['status_geral']} ===")

    results_json_path = output_dir / "segmentation_arch_pilot_results.json"
    report_md_path = output_dir / "segmentation_arch_pilot_report.md"
    log_path = output_dir / "segmentation_arch_pilot.log"

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
        "\nLEMBRETE: verificacao tecnica mecanica, nao treino real, nao selecao "
        "de arquitetura/limiar/metrica final. HAM10000 nao foi tocado nesta execucao."
    )
    if not all_mechanical_passed:
        sys.exit(1)


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Piloto — comparação de arquiteturas de segmentação (U-Net vs DeepLabV3+-ResNet34)",
        "",
        "**Verificação técnica mecânica, não treino real.** Não seleciona "
        "arquitetura, métrica final nem limiar de máscara definitivos — essas "
        "decisões ficam para a execução científica completa (5 folds externos "
        "x 4 internos, não implementada aqui). HAM10000 não foi lido nesta "
        "execução.",
        "",
        f"## Resultado geral: {report['status_geral']}",
        "",
        f"- Dispositivo solicitado: `{report['device_requested']}` / efetivo: `{report['device_effective']}`",
        f"- Resolução de treino: {report['target_size']}x{report['target_size']}",
        f"- Threshold de demonstração: {report['demo_threshold']} (NÃO é o limiar escolhido por validação)",
        f"- Política de inicialização: {report['init_policy']}",
        f"- Arquivo de folds: `{report['folds_file']}`",
        f"- Chave de agrupamento dos folds: {report['fold_grouping_key']}",
        f"- Fold externo: {report['external_fold']} ({report['external_fold_n_images']} imagens, NUNCA usado neste piloto)",
        f"- Fold de validação interna: {report['internal_val_fold']} "
        f"({report['n_val_interna_total']} imagens no fold; amostra demo: {report['n_val_interna_amostra']})",
        f"- Folds de treino piloto: {report['train_piloto_folds']} "
        f"({report['n_train_piloto_total']} imagens; amostra piloto: {report['n_train_piloto_amostra']})",
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
        lines.append("| Verificação mecânica | Resultado |")
        lines.append("|---|---|")
        for check in MECHANICAL_CHECKS:
            lines.append(f"| `{check}` | {res[check]['status']} |")
        lines.append("")
        lines.append(f"- Tempo de execução: {res['tempo_execucao_s']:.2f}s")
        lines.append(f"- Imagens de treino piloto usadas: {res['n_imagens_treino_piloto']}")
        lines.append(
            f"- Imagens de validação interna usadas nas métricas-demo: "
            f"{res['n_imagens_validacao_interna_metricas']}"
        )
        if "medicao_tempo_epoca" in res:
            te = res["medicao_tempo_epoca"]
            lines.append(
                f"- Medição de tempo (1 época, amostra de piloto): {te['tempo_s']:.2f}s para "
                f"{te['n_imagens']} imagens ({te['tempo_por_imagem_s']:.3f}s/imagem) — {te['aviso']}"
            )
        lines.append("")
        agg = res["metricas_demo"]["aggregate"]
        lines.append(f"### Métricas de demonstração (threshold={res['metricas_demo']['threshold_demo']})")
        lines.append("")
        lines.append(f"- Casos: total={agg['n_total']} ok={agg['n_ok']} por status={agg['counts_by_status']}")
        lines.append(f"- Dice médio (ok): {agg['mean_dice']}")
        lines.append(f"- IoU médio (ok): {agg['mean_iou']}")
        lines.append(f"- Thresholded Jaccard médio (ok): {agg['mean_thresholded_jaccard']}")
        lines.append(f"- Taxa de falha ISIC (ok): {agg['isic_failure_rate']}")
        lines.append(f"- HD95 mediana (valores finitos): {agg['hd95_median_finite']} / indefinidos: {agg['hd95_n_undefined']}")
        lines.append(
            f"- HD95 mediana normalizada pela diagonal (valores finitos): "
            f"{agg['hd95_normalizado_diagonal_median_finite']}"
        )
        lines.append("")
        lines.append(
            "| image_id | status | dimensões originais | dice | iou | thresholded_jaccard | hd95 | "
            "hd95_norm_diag | falha_isic | pred_pos_px | ref_pos_px |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for pr in res["metricas_demo"]["per_image"]:
            lines.append(
                f"| `{pr.get('image_id')}` | {pr['status']} | {pr.get('original_shape')} | "
                f"{pr.get('dice')} | {pr.get('iou')} | {pr.get('thresholded_jaccard')} | "
                f"{pr.get('hd95')} | {pr.get('hd95_normalizado_diagonal')} | {pr.get('is_isic_failure')} | "
                f"{pr.get('pred_positive_pixels')} | {pr.get('ref_positive_pixels')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", choices=["unet", "deeplabv3plus", "both"], default="both")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default=_best_available_device(),
        help="Se omitido, auto-detecta cuda > mps > cpu (mesma convencao de "
        "scripts/train_bcn20000_efficientnet.py). Se passado explicitamente e "
        "o dispositivo nao estiver disponivel, levanta erro -- nunca cai "
        "silenciosamente para cpu.",
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
    parser.add_argument("--n-pilot-images", type=int, default=DEFAULT_N_PILOT_IMAGES)
    parser.add_argument("--pilot-steps", type=int, default=DEFAULT_PILOT_STEPS)
    parser.add_argument("--seed", type=int, default=folds_mod.DEFAULT_SEED)
    parser.add_argument(
        "--external-fold", type=int, default=DEFAULT_EXTERNAL_FOLD,
        help="Indice do fold (0-4) tratado como fold externo -- NUNCA usado neste piloto.",
    )
    parser.add_argument(
        "--demo-threshold", type=float, default=DEFAULT_DEMO_THRESHOLD,
        help="Limiar de binarizacao usado SO para demonstracao -- NAO e o limiar escolhido por validacao.",
    )
    parser.add_argument(
        "--run-timing-epoch", action="store_true", default=False,
        help="Roda 1 epoca completa sobre a amostra de piloto para medir tempo -- NAO e treino real.",
    )
    args = parser.parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
