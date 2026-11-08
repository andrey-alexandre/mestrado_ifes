"""Short pilot for the U-Net segmentation baseline on ISIC2018 Task1
(stage 4). Verifies the real pipeline mechanics on a small REAL subset
(not synthetic) -- image<->mask reading/alignment, a real weight update,
checkpoint save/reload, and predicted-mask shape -- it does NOT train a
real model, does NOT run cross-validation over the 5 folds, and does NOT
select/compare architectures, metrics, or a mask threshold. Those remain
for later, separately-approved decisions.

Also generates (once; reused on subsequent runs with the same seed) the
fixed, reusable 5-fold assignment for ISIC2018 Task1
(experiments/segmentation/isic2018_task1_folds.json) -- grouped by lesion
if metadata allows it; ISIC2018 Task1 has NO lesion-identifier metadata
(confirmed by direct inspection: neither zip nor the surrounding
directory has any CSV/metadata file beyond ATTRIBUTION.txt/LICENSE.txt),
so this falls back to one group per image. This limitation is recorded
in the generated report, not hidden.

HAM10000 is never read or referenced here: it is reserved exclusively
for the final external evaluation of segmentation.

STATUS: implementada, aguardando execucao do pesquisador. Not run by the
assistant/fork that wrote it.

Usage:
    python scripts/run_isic2018_unet_pilot.py
    python scripts/run_isic2018_unet_pilot.py --n-pilot-images 16 --seed 20261108
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation import folds as folds_mod  # noqa: E402
from src.segmentation.isic2018_dataset import (  # noqa: E402
    ISIC2018Task1Dataset,
    TARGET_SIZE,
    list_available_image_ids,
    verify_image_mask_correspondence,
)
from src.segmentation.unet_model import build_unet  # noqa: E402

DEFAULT_IMAGES_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip"
DEFAULT_MASKS_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation")
DEFAULT_FOLDS_PATH = str(REPO_ROOT / "experiments" / "segmentation" / "isic2018_task1_folds.json")
DEFAULT_N_PILOT_IMAGES = 12
DEFAULT_PILOT_STEPS = 2


def _ensure_folds(images_zip: str, folds_path: str, n_folds: int, seed: int) -> dict[str, Any]:
    """Reuse the folds file if it already exists (never silently
    regenerate a competing assignment); otherwise build it for real from
    the actual 2,594 image ids and write it once."""
    path = Path(folds_path)
    if path.is_file():
        print(f"folds ja existentes, reaproveitando: {path}")
        return folds_mod.load_folds_file(path)

    print("folds nao encontrados -- gerando pela primeira vez (real, nao sintetico)...")
    image_ids = list_available_image_ids(images_zip)
    print(f"image ids encontrados no zip de imagens: {len(image_ids)}")
    result = folds_mod.build_folds(image_ids, n_folds=n_folds, seed=seed, image_id_to_lesion_id=None)
    folds_mod.write_folds_file(result, path, dataset="isic2018_task1")
    print(
        f"folds gravados: {path} -- agrupamento por lesao disponivel: "
        f"{result['lesion_grouping_available']} (False esperado para ISIC2018 Task1, "
        "sem metadados de lesion_id -- ver limitacao no relatorio)"
    )
    return folds_mod.load_folds_file(path)


def run_pilot(args: argparse.Namespace) -> None:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    print("=== PILOTO: pipeline real U-Net no ISIC2018 Task1 ===")

    print("[0/4] verificando correspondencia imagem<->mascara (ids, direto dos dois zips)...")
    correspondence = verify_image_mask_correspondence(args.images_zip, args.masks_zip)
    print(
        f"      imagens={correspondence['n_images']} mascaras={correspondence['n_masks']} "
        f"pareadas={correspondence['n_matched']} so_em_imagens={correspondence['n_only_in_images']} "
        f"so_em_mascaras={correspondence['n_only_in_masks']}"
    )
    if not correspondence["ok"]:
        print(
            "      AVISO: correspondencia imperfeita -- ver 'only_in_images'/'only_in_masks' "
            "no relatorio. O piloto continua, mas qualquer id sem par sera pulado na amostra."
        )

    folds_data = _ensure_folds(args.images_zip, args.folds_path, folds_mod.N_FOLDS, args.seed)

    all_folded_ids = sorted(folds_data["image_to_fold"].keys())
    # Only ever sample ids confirmed paired above -- never hand the dataset
    # an id missing its mask just because it happened to be in the folds file.
    matched_ids = {iid for iid in all_folded_ids} - set(correspondence["only_in_images"])
    pilot_image_ids = sorted(matched_ids)[: args.n_pilot_images]
    print(
        f"[1/4] usando {len(pilot_image_ids)} pares reais imagem+mascara "
        "(os primeiros N ids em ordem alfabetica do arquivo de folds -- nao "
        "stratificado por fold, so uma amostra pequena e deterministica para o piloto)"
    )

    dataset = ISIC2018Task1Dataset(pilot_image_ids, args.images_zip, args.masks_zip)

    try:
        image_tensor, mask_tensor, image_id = dataset[0]
        alignment_ok = (
            image_id == pilot_image_ids[0]
            and tuple(image_tensor.shape) == (3, TARGET_SIZE, TARGET_SIZE)
            and tuple(mask_tensor.shape) == (1, TARGET_SIZE, TARGET_SIZE)
        )
        results["image_mask_alignment"] = "PASSOU" if alignment_ok else "FALHOU: formas/id inesperados"
        print(
            f"      imagem={tuple(image_tensor.shape)} mascara={tuple(mask_tensor.shape)} "
            f"id={image_id} (esperado {pilot_image_ids[0]})"
        )
    except Exception as exc:  # noqa: BLE001 - pilot must report, not crash
        results["image_mask_alignment"] = f"FALHOU: {exc}"
        print(f"      FALHOU: {exc}")
    print(f"[1/4] leitura e alinhamento imagem<->mascara: {results['image_mask_alignment']}")

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=min(4, len(pilot_image_ids)), shuffle=False, num_workers=0)
    device = torch.device("cpu")
    model = build_unet(img_ch=3, output_ch=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    before = model.out_conv.weight.detach().clone()
    try:
        for step, (images, masks, _ids) in enumerate(loader):
            if step >= args.pilot_steps:
                break
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        after = model.out_conv.weight.detach().clone()
        changed = not torch.equal(before, after)
        results["weight_update"] = "PASSOU" if changed else "FALHOU: pesos nao mudaram"
    except Exception as exc:  # noqa: BLE001
        results["weight_update"] = f"FALHOU: {exc}"
    print(f"[2/4] atualizacao de peso: {results['weight_update']}")

    ckpt_path = output_dir / "pilot_checkpoint.pt"
    try:
        torch.save({"model_state_dict": model.state_dict(), "pilot": True}, ckpt_path)
        reloaded = build_unet(img_ch=3, output_ch=1)
        state = torch.load(ckpt_path, map_location="cpu")
        reloaded.load_state_dict(state["model_state_dict"])
        results["checkpoint_save_reload"] = "PASSOU"
    except Exception as exc:  # noqa: BLE001
        results["checkpoint_save_reload"] = f"FALHOU: {exc}"
    print(f"[3/4] checkpoint (salvar+recarregar): {results['checkpoint_save_reload']}")

    try:
        model.eval()
        with torch.no_grad():
            sample_image, _mask, _id = dataset[0]
            logits = model(sample_image.unsqueeze(0).to(device))
            predicted_mask = torch.sigmoid(logits)
        shape_ok = tuple(predicted_mask.shape) == (1, 1, TARGET_SIZE, TARGET_SIZE)
        results["predicted_mask_shape"] = (
            "PASSOU" if shape_ok else f"FALHOU: forma {tuple(predicted_mask.shape)}"
        )
        print(f"      mascara predita: forma={tuple(predicted_mask.shape)} (esperado (1, 1, {TARGET_SIZE}, {TARGET_SIZE}))")
    except Exception as exc:  # noqa: BLE001
        results["predicted_mask_shape"] = f"FALHOU: {exc}"
    print(f"[4/4] geracao de mascara predita nas dimensoes esperadas: {results['predicted_mask_shape']}")

    all_passed = all(v == "PASSOU" for v in results.values())
    print(f"\n=== RESULTADO: {'TODAS AS VERIFICACOES PASSARAM' if all_passed else 'AO MENOS UMA FALHOU'} ===")

    report = {
        "status": "TODAS AS VERIFICACOES PASSARAM" if all_passed else "AO MENOS UMA FALHOU",
        "checks": results,
        "image_mask_correspondence": {
            **{k: v for k, v in correspondence.items() if k not in ("only_in_images", "only_in_masks")},
            "only_in_images_sample": correspondence["only_in_images"][:20],
            "only_in_masks_sample": correspondence["only_in_masks"][:20],
        },
        "n_pilot_images": len(pilot_image_ids),
        "pilot_image_ids": pilot_image_ids,
        "target_size": TARGET_SIZE,
        "unet_source": "implementado diretamente em src/segmentation/unet_model.py "
        "(Image_Segmentation.code.network nao esta instalado neste ambiente, "
        "confirmado por tentativa de import)",
        "folds_file": str(args.folds_path),
        "lesion_grouping_available": folds_data["lesion_grouping_available"],
        "lesion_grouping_limitation": (
            "ISIC2018 Task1 nao tem nenhum arquivo de metadados com lesion_id "
            "(confirmado por inspecao direta dos dois zips e do diretorio ao lado "
            "deles) -- os folds usam um grupo por imagem, nao por lesao."
            if not folds_data["lesion_grouping_available"]
            else "Agrupamento por lesao disponivel e usado."
        ),
        "ham10000_touched": False,
    }
    report_json_path = output_dir / "unet_pilot_results.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_md_path = output_dir / "unet_pilot_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(_render_report_md(report))

    print(f"relatorio (json) gravado: {report_json_path}")
    print(f"relatorio (md) gravado: {report_md_path}")
    print(
        "LEMBRETE: isto e um piloto de verificacao mecanica, nao treino real nem "
        "selecao de arquitetura/metrica/limiar. HAM10000 nao foi lido nesta execucao."
    )
    if not all_passed:
        sys.exit(1)


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Piloto U-Net — ISIC2018 Task1 (Etapa 4, segmentação)",
        "",
        "**Isto é um piloto de verificação mecânica, não treino real.** "
        "Não seleciona arquitetura, métrica final nem limiar de máscara — "
        "essas decisões ficam para discussão posterior. U-Net é o baseline "
        "já confirmado. HAM10000 não foi lido nesta execução.",
        "",
        f"## Resultado geral: {report['status']}",
        "",
        "| Verificação | Resultado |",
        "|---|---|",
    ]
    for name, value in report["checks"].items():
        lines.append(f"| `{name}` | {value} |")
    corr = report["image_mask_correspondence"]
    lines += [
        "",
        "## Correspondência imagem↔máscara (reaproveitando o inventário, recomputado por id de zip)",
        "",
        f"- Imagens: {corr['n_images']} / Máscaras: {corr['n_masks']} / Pareadas: {corr['n_matched']}",
        f"- Só em imagens: {corr['n_only_in_images']} / Só em máscaras: {corr['n_only_in_masks']}",
        f"- Correspondência perfeita: {corr['ok']}",
        "",
        f"- Imagens reais usadas no piloto: {report['n_pilot_images']}",
        f"- Resolução alvo (imagem e máscara): {report['target_size']}x{report['target_size']}",
        f"- Origem da U-Net: {report['unet_source']}",
        f"- Arquivo de folds: `{report['folds_file']}`",
        f"- Agrupamento por lesão disponível: {report['lesion_grouping_available']}",
        f"- {report['lesion_grouping_limitation']}",
        f"- HAM10000 tocado nesta execução: {report['ham10000_touched']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-zip", default=DEFAULT_IMAGES_ZIP)
    parser.add_argument("--masks-zip", default=DEFAULT_MASKS_ZIP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds-path", default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--n-pilot-images", type=int, default=DEFAULT_N_PILOT_IMAGES)
    parser.add_argument("--pilot-steps", type=int, default=DEFAULT_PILOT_STEPS)
    parser.add_argument(
        "--seed",
        type=int,
        default=folds_mod.DEFAULT_SEED,
        help=f"Semente dos 5 folds fixos (default documentado: {folds_mod.DEFAULT_SEED}). "
        "So usada na primeira geracao do arquivo de folds; execucoes seguintes reaproveitam o arquivo.",
    )
    args = parser.parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
