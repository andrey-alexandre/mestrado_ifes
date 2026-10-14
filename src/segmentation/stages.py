"""Etapas do pipeline experimental: treino ISIC, folds, transferência direta, fine-tuning
e preparação do modelo final.

Este módulo controla datasets, folds, treinamento, avaliação e persistência. Não
conhece detalhes de arquitetura (usa apenas a interface de `SegmentationStrategy`).
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

from src.segmentation import config as cfgmod
from src.segmentation.artifacts import (
    StageStore, fingerprint, provenance, read_json, sha256_file, write_csv, write_json,
)
from src.segmentation.data import (
    DataError, Record, SegmentationDataset, apply_groups_and_exclusions, assert_no_group_leak,
    discover_ph2, load_mask_bool, pair_images_and_masks, read_id_list, read_lesion_groups,
    save_mask_png, split_train_val,
)
from src.segmentation.folds import (
    assert_manifest_matches_records, fold_of_image, ids_fingerprint, load_manifest,
    make_folds, save_manifest,
)
from src.segmentation.metrics import segmentation_metrics, summarize
from src.segmentation.strategies import build_strategy, state_dict_sha256

logger = logging.getLogger(__name__)

PER_IMAGE_FIELDS = ["image_id", "fold", "label", "diagnosis", "dice", "iou", "hd95", "degenerate",
                    "pred_area", "gt_area", "image_path", "pred_mask_path", "gt_mask_path", "approach"]


class Layout:
    """Diretórios de saída resolvidos a partir de `segmentation_experiment.output`."""

    def __init__(self, cfg: Dict):
        o = cfg["output"]
        self.checkpoints, self.predictions = Path(o["checkpoints_dir"]), Path(o["predictions_dir"])
        self.results, self.reports = Path(o["results_dir"]), Path(o["reports_dir"])
        self.folds_manifest = Path(o["folds_manifest"])

    @property
    def isic_checkpoint(self) -> Path:
        return self.checkpoints / "isic2018" / "best.pt"

    def finetune_checkpoint(self, fold: int) -> Path:
        return self.checkpoints / "finetune" / f"fold_{fold}.pt"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoints / "final" / "final_segmenter.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de registros
# ─────────────────────────────────────────────────────────────────────────────

def load_isic_records(cfg: Dict, limit: Optional[int] = None):
    d = cfg["datasets"]["isic2018"]
    cfgmod.require(cfg, ["datasets.isic2018.image_dir", "datasets.isic2018.mask_dir"], "treino ISIC 2018")
    records = pair_images_and_masks(
        d["image_dir"], d["mask_dir"], d["mask_suffix"], d["image_extensions"],
        d["mask_extensions"], d["allow_unpaired"])
    groups = read_lesion_groups(d["lesion_groups_csv"]) if d["lesion_groups_csv"] else None
    exclude = read_id_list(d["overlap_ids_file"]) if d["overlap_ids_file"] else None
    records, summary = apply_groups_and_exclusions(records, groups, exclude)
    if limit:
        records = records[: int(limit)]
        summary["limit"] = int(limit)
    summary["n_records"] = len(records)
    return records, summary


def load_ph2_records(cfg: Dict):
    d = cfg["datasets"]["ph2"]
    cfgmod.require(cfg, ["datasets.ph2.dataset_dir", "datasets.ph2.metadata_txt"], "PH²")
    return discover_ph2(d["dataset_dir"], d["metadata_txt"], d["mask_dir"], d["label_source"])


def _seg_fingerprint_cfg(cfg: Dict) -> Dict:
    s = dict(cfg["segmentation"])
    s.pop("device", None)  # onde roda não altera o artefato científico
    s.pop("num_workers", None)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1 — treino no ISIC 2018 (não toca o PH²)
# ─────────────────────────────────────────────────────────────────────────────

def run_train_isic(cfg: Dict, store: StageStore, limit: Optional[int] = None) -> Dict:
    """Treina no ISIC 2018 Task 1 e salva o melhor checkpoint (por validação ISIC)."""
    lay = Layout(cfg)
    records, rec_summary = load_isic_records(cfg, limit)
    ids_fp = ids_fingerprint([r.image_id for r in records])
    fp = fingerprint(_seg_fingerprint_cfg(cfg), cfg["datasets"]["isic2018"], ids_fp, limit)
    stage = "isic_training"
    if store.is_complete(stage, fp) and lay.isic_checkpoint.exists():
        return read_json(str(lay.results / "isic_training" / "summary.json"))

    seed = cfg["segmentation"]["seed"]
    train_recs, val_recs = split_train_val(records, cfg["datasets"]["isic2018"]["val_fraction"], seed)
    assert_no_group_leak(train_recs, val_recs)
    write_json(str(lay.results / "isic_training" / "split.json"), {
        "seed": seed, "val_fraction": cfg["datasets"]["isic2018"]["val_fraction"],
        "data_summary": rec_summary,
        "train_ids": [r.image_id for r in train_recs], "val_ids": [r.image_id for r in val_recs],
    })

    strategy = build_strategy(cfg)
    train_ds = SegmentationDataset(train_recs, strategy.transform(train=True))
    val_ds = SegmentationDataset(val_recs, strategy.transform(train=False))
    logger.info("Treino ISIC 2018: %d treino / %d validação", len(train_ds), len(val_ds))
    result = strategy.train(train_ds, val_ds)

    meta = {"best_epoch": result.best_epoch, "best_monitor": result.best_monitor,
            "train_ids_sha256": ids_fingerprint([r.image_id for r in train_recs]),
            "val_ids_sha256": ids_fingerprint([r.image_id for r in val_recs])}
    ckpt_sha = strategy.save_checkpoint(str(lay.isic_checkpoint), metadata=meta)
    summary = {
        "checkpoint": str(lay.isic_checkpoint), "checkpoint_sha256": ckpt_sha,
        "best_epoch": result.best_epoch, "best_monitor": result.best_monitor,
        "monitor": cfg["segmentation"]["early_stopping"]["monitor"],
        "epochs_run": result.epochs_run, "stopped_early": result.stopped_early,
        "n_train": len(train_recs), "n_val": len(val_recs), "data_summary": rec_summary,
        "provenance": provenance(seed),
    }
    write_json(str(lay.results / "isic_training" / "config.json"), {
        "segmentation": cfg["segmentation"], "isic2018": cfg["datasets"]["isic2018"],
        "isic_limit": limit})
    write_json(str(lay.results / "isic_training" / "history.json"), result.history)
    write_json(str(lay.results / "isic_training" / "summary.json"), summary)
    store.mark_complete(stage, fp, {"checkpoint_sha256": ckpt_sha})
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2 — folds do PH²
# ─────────────────────────────────────────────────────────────────────────────

def run_make_folds(cfg: Dict, store: StageStore) -> Dict:
    lay = Layout(cfg)
    records, summary = load_ph2_records(cfg)
    ev, seed = cfg["evaluation"], cfg["segmentation"]["seed"]
    fp = fingerprint(ids_fingerprint([r.image_id for r in records]), ev["n_folds"], seed,
                     ev["stratify_by"], cfg["datasets"]["ph2"]["label_source"])
    if store.is_complete("folds", fp) and lay.folds_manifest.exists():
        manifest = load_manifest(str(lay.folds_manifest))
        assert_manifest_matches_records(manifest, records)
        return manifest
    manifest = make_folds(records, ev["n_folds"], seed, ev["stratify_by"],
                          cfg["datasets"]["ph2"]["label_source"])
    manifest["data_summary"] = summary
    save_manifest(manifest, str(lay.folds_manifest))
    write_csv(str(lay.results / "folds_manifest.csv"), manifest["images"])
    store.mark_complete("folds", fp)
    logger.info("Folds salvos em %s", lay.folds_manifest)
    return manifest


def _load_folds_and_records(cfg: Dict, lay: Layout):
    if not lay.folds_manifest.exists():
        raise DataError(f"Manifesto de folds ausente ({lay.folds_manifest}). Execute 'make-folds' antes.")
    records, _ = load_ph2_records(cfg)
    manifest = load_manifest(str(lay.folds_manifest))
    assert_manifest_matches_records(manifest, records)
    return manifest, {r.image_id: r for r in records}


# ─────────────────────────────────────────────────────────────────────────────
# Predição + métricas por imagem
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_score(strategy, records: Sequence[Record], fold_map: Dict[str, int],
                      out_dir: Path, approach: str) -> List[Dict]:
    """Prediz máscaras (sem atualizar pesos), salva PNGs e calcula métricas por imagem."""
    rows: List[Dict] = []
    for r in records:
        with Image.open(r.image_path) as im:
            rgb = im.convert("RGB")
        pred = strategy.predict_mask(rgb)
        gt = load_mask_bool(r.mask_path, size=rgb.size)
        pred_path = out_dir / f"{r.image_id}.png"
        save_mask_png(pred, str(pred_path))
        m = segmentation_metrics(pred, gt)
        rows.append({
            "image_id": r.image_id, "fold": fold_map[r.image_id], "label": r.label,
            "diagnosis": r.diagnosis, **{k: m[k] for k in ("dice", "iou", "hd95", "degenerate",
                                                           "pred_area", "gt_area")},
            "image_path": r.image_path, "pred_mask_path": str(pred_path),
            "gt_mask_path": r.mask_path, "approach": approach,
        })
    return rows


def aggregate_rows(rows: Sequence[Dict]) -> Dict:
    """Agregados globais e por fold das métricas por imagem."""
    def block(sub: Sequence[Dict]) -> Dict:
        out = {m: summarize([float("nan") if r[m] in (None, "") else float(r[m]) for r in sub])
               for m in ("dice", "iou", "hd95")}
        out["n_images"] = len(sub)
        out["degenerate_counts"] = {
            k: sum(1 for r in sub if r["degenerate"] == k)
            for k in sorted({r["degenerate"] for r in sub})}
        return out
    folds = sorted({int(r["fold"]) for r in rows})
    return {"overall": block(rows),
            "per_fold": {str(f): block([r for r in rows if int(r["fold"]) == f]) for f in folds}}


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 3 — transferência direta
# ─────────────────────────────────────────────────────────────────────────────

def run_direct_transfer(cfg: Dict, store: StageStore, checkpoint: Optional[str] = None) -> Dict:
    lay = Layout(cfg)
    ckpt = str(checkpoint or lay.isic_checkpoint)
    if not Path(ckpt).is_file():
        raise DataError(f"Checkpoint inicial não encontrado: {ckpt}. Execute 'train-isic' ou informe --checkpoint.")
    manifest, by_id = _load_folds_and_records(cfg, lay)
    fp = fingerprint(sha256_file(ckpt), manifest["ids_sha256"], _seg_fingerprint_cfg(cfg))
    out = lay.results / "direct_transfer"
    if store.is_complete("direct_transfer", fp) and (out / "summary.json").exists():
        return read_json(str(out / "summary.json"))

    strategy = build_strategy(cfg)
    info = strategy.load_checkpoint(ckpt)
    fold_map = fold_of_image(manifest)
    ordered = [by_id[im["image_id"]] for im in manifest["images"]]
    rows = predict_and_score(strategy, ordered, fold_map,
                             lay.predictions / "direct_transfer", "direct_transfer")
    after = state_dict_sha256(strategy.model)  # hash dos pesos após todas as predições
    summary = {
        "approach": "direct_transfer",
        "note": "Avaliação externa sem adaptação: um único modelo avaliado nos 5 folds (não são 5 treinos).",
        "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
        "weights_sha256": info["weights_sha256"], "weights_unchanged": info["weights_sha256"] == after,
        "folds_manifest_sha256": manifest["ids_sha256"], "threshold": cfg["segmentation"]["threshold"],
        **aggregate_rows(rows), "provenance": provenance(cfg["segmentation"]["seed"]),
    }
    write_csv(str(out / "per_image.csv"), rows, PER_IMAGE_FIELDS)
    write_json(str(out / "summary.json"), summary)
    store.mark_complete("direct_transfer", fp)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 4 — fine-tuning por fold
# ─────────────────────────────────────────────────────────────────────────────

def inner_split(train_recs: Sequence[Record], fraction: float, seed: int):
    """Validação interna estratificada, usando apenas os registros dos 4 folds de treino."""
    labels = np.array([int(r.label) for r in train_recs])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    tr, va = next(sss.split(np.zeros(len(train_recs)), labels))
    return [train_recs[i] for i in sorted(tr)], [train_recs[i] for i in sorted(va)]


def run_finetune_round(cfg: Dict, ckpt: str, round_def: Dict, by_id: Dict[str, Record],
                       fold_map: Dict[str, int], lay: Layout) -> Dict:
    """Uma rodada: recarrega o checkpoint ISIC, adapta com 4 folds e prediz o fold restante.

    Os registros do fold de teste só são lidos DEPOIS do treino, para predição; nenhuma
    decisão (early stopping, checkpoint, limiar) usa essas imagens.
    """
    k = round_def["round"]
    seed = cfg["segmentation"]["seed"] + 1000 + k
    train_recs = [by_id[i] for i in round_def["train_ids"]]
    inner_train, inner_val = inner_split(train_recs, cfg["transfer_learning"]["inner_val_fraction"], seed)
    if {r.image_id for r in inner_train + inner_val} & set(round_def["test_ids"]):
        raise DataError(f"Rodada {k}: dados de teste no conjunto de adaptação.")

    strategy = build_strategy(cfg)             # instância nova: sem estado da rodada anterior
    info = strategy.load_checkpoint(ckpt)      # pesos originais do ISIC 2018
    start_hash = info["weights_sha256"]
    train_ds = SegmentationDataset(inner_train, strategy.transform(train=True))
    val_ds = SegmentationDataset(inner_val, strategy.transform(train=False))
    result = strategy.fine_tune(train_ds, val_ds, cfg["transfer_learning"], seed=seed)

    ckpt_sha = strategy.save_checkpoint(str(lay.finetune_checkpoint(k)), metadata={
        "round": k, "best_epoch": result.best_epoch, "base_weights_sha256": start_hash})
    test_recs = [by_id[i] for i in round_def["test_ids"]]
    rows = predict_and_score(strategy, test_recs, fold_map, lay.predictions / "finetune", "finetune")
    return {
        "round": k, "n_adapt_train": len(inner_train), "n_adapt_val": len(inner_val),
        "n_test": len(test_recs), "base_weights_sha256": start_hash, "checkpoint_sha256": ckpt_sha,
        "best_epoch": result.best_epoch, "best_monitor": result.best_monitor,
        "epochs_run": result.epochs_run, "stopped_early": result.stopped_early,
        "history": result.history, "rows": rows,
    }


def run_finetune(cfg: Dict, store: StageStore, checkpoint: Optional[str] = None) -> Dict:
    if not cfg["transfer_learning"]["enabled"]:
        raise cfgmod.ConfigError("transfer_learning.enabled=false: fine-tuning desativado no config.yaml.")
    lay = Layout(cfg)
    ckpt = str(checkpoint or lay.isic_checkpoint)
    if not Path(ckpt).is_file():
        raise DataError(f"Checkpoint inicial não encontrado: {ckpt}. Execute 'train-isic' ou informe --checkpoint.")
    manifest, by_id = _load_folds_and_records(cfg, lay)
    fold_map = fold_of_image(manifest)
    base_fp = fingerprint(sha256_file(ckpt), manifest["ids_sha256"], _seg_fingerprint_cfg(cfg),
                          cfg["transfer_learning"])
    out = lay.results / "finetune"
    if store.is_complete("finetune", base_fp) and (out / "summary.json").exists():
        return read_json(str(out / "summary.json"))

    all_rows: List[Dict] = []
    rounds_meta: List[Dict] = []
    for rd in manifest["rounds"]:
        stage = f"finetune/fold_{rd['round']}"
        fold_fp = fingerprint(base_fp, rd["round"])
        part = out / f"round_{rd['round']}.json"
        if store.is_complete(stage, fold_fp) and part.exists():
            res = read_json(str(part))
        else:
            logger.info("Fine-tuning: rodada %d/%d", rd["round"] + 1, manifest["n_folds"])
            res = run_finetune_round(cfg, ckpt, rd, by_id, fold_map, lay)
            write_json(str(part), res)
            store.mark_complete(stage, fold_fp)
        all_rows.extend(res["rows"])
        rounds_meta.append({k: v for k, v in res.items() if k not in ("rows",)})

    order = {im["image_id"]: i for i, im in enumerate(manifest["images"])}
    all_rows.sort(key=lambda r: order[r["image_id"]])
    if [r["image_id"] for r in all_rows] != [im["image_id"] for im in manifest["images"]]:
        raise DataError("Fine-tuning não produziu predição fora da amostra para todas as imagens.")

    base_hashes = {m["base_weights_sha256"] for m in rounds_meta}
    summary = {
        "approach": "finetune",
        "note": "Predições fora da amostra: cada imagem foi predita pelo modelo cujo treino excluiu seu fold.",
        "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
        "all_rounds_started_from_same_weights": len(base_hashes) == 1,
        "folds_manifest_sha256": manifest["ids_sha256"], "threshold": cfg["segmentation"]["threshold"],
        "transfer_learning": cfg["transfer_learning"], "segmentation": cfg["segmentation"],
        "rounds": [{k: v for k, v in m.items() if k != "history"} for m in rounds_meta],
        **aggregate_rows(all_rows), "provenance": provenance(cfg["segmentation"]["seed"]),
    }
    write_csv(str(out / "per_image.csv"), all_rows, PER_IMAGE_FIELDS)
    write_json(str(out / "histories.json"), {str(m["round"]): m["history"] for m in rounds_meta})
    write_json(str(out / "summary.json"), summary)
    store.mark_complete("finetune", base_fp)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 7 — modelo final
# ─────────────────────────────────────────────────────────────────────────────

def prepare_final_model(cfg: Dict, store: StageStore, strategy_choice: Optional[str] = None,
                        checkpoint: Optional[str] = None) -> Dict:
    """Produz o checkpoint final sem tocar nas predições fora da amostra da avaliação."""
    lay = Layout(cfg)
    if strategy_choice is None:
        sel_path = lay.results / "comparison" / "selection.json"
        if not sel_path.exists():
            raise DataError("Seleção ausente. Execute 'compare' ou informe --strategy explicitamente.")
        strategy_choice = read_json(str(sel_path))["selected"]
    if strategy_choice not in ("direct_transfer", "finetune"):
        raise cfgmod.ConfigError(f"Estratégia final inválida: {strategy_choice}")

    ckpt = str(checkpoint or lay.isic_checkpoint)
    if not Path(ckpt).is_file():
        raise DataError(f"Checkpoint inicial não encontrado: {ckpt}")
    base_sha = sha256_file(ckpt)
    out = lay.results / "final_model"
    final_path = lay.final_checkpoint
    manifest = None
    epochs = None
    if strategy_choice == "finetune":
        manifest, by_id = _load_folds_and_records(cfg, lay)
        fe = cfg["transfer_learning"]["final_epochs"]
        if fe == "median_cv_best_epoch":
            ft = lay.results / "finetune" / "summary.json"
            if not ft.exists():
                raise DataError("final_epochs='median_cv_best_epoch' requer resultados do fine-tuning.")
            epochs = int(round(float(np.median([r["best_epoch"] for r in read_json(str(ft))["rounds"]]))))
        else:
            epochs = int(fe)
        epochs = max(epochs, 1)
    fp = fingerprint(strategy_choice, base_sha, _seg_fingerprint_cfg(cfg), cfg["transfer_learning"],
                     epochs, manifest["ids_sha256"] if manifest else None)
    if store.is_complete("final_model", fp) and final_path.exists():
        return read_json(str(out / "provenance.json"))

    strategy = build_strategy(cfg)
    history: List[Dict] = []
    if strategy_choice == "direct_transfer":
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt, final_path)
        info = strategy.load_checkpoint(str(final_path))
        data_prov = {"training_data": "ISIC 2018 Task 1 (checkpoint preservado)",
                     "ph2_used_for_training": False}
    else:
        strategy.load_checkpoint(ckpt)
        recs = [by_id[i] for i in sorted(by_id)]
        ds = SegmentationDataset(recs, strategy.transform(train=True))
        tl = dict(cfg["transfer_learning"], max_epochs=epochs,
                  early_stopping={"enabled": False, "patience": epochs, "min_delta": 0.0,
                                  "monitor": "val_dice"})
        result = strategy.fine_tune(ds, None, tl, seed=cfg["segmentation"]["seed"] + 9999)
        history = result.history
        strategy.save_checkpoint(str(final_path), metadata={
            "final_model": True, "epochs": epochs, "base_weights_sha256": base_sha,
            "n_train": len(recs)})
        info = strategy.load_checkpoint(str(final_path))
        data_prov = {"training_data": "ISIC 2018 Task 1 -> PH² (todas as imagens)",
                     "ph2_used_for_training": True, "n_ph2_images": len(recs),
                     "ph2_ids_sha256": manifest["ids_sha256"], "fixed_epochs": epochs,
                     "in_sample_warning": ("Este modelo viu as 200 imagens do PH²: qualquer métrica "
                                           "nele é em-amostra. A avaliação válida permanece nas "
                                           "predições fora da amostra de 'finetune'.")}

    prov = {
        "selected_strategy": strategy_choice, "data_provenance": data_prov,
        "checkpoint": str(final_path), "checkpoint_sha256": sha256_file(str(final_path)),
        "weights_sha256": info["weights_sha256"], "base_checkpoint": ckpt,
        "base_checkpoint_sha256": base_sha, "seed": cfg["segmentation"]["seed"],
        "config": {k: v for k, v in cfg.items() if not k.startswith("_") and k != "paths"},
        "training_history": history, "provenance": provenance(cfg["segmentation"]["seed"]),
        "isic2019_inference": "ISIC 2019: inferência não executada (fora do escopo desta etapa)",
    }
    write_json(str(out / "provenance.json"), prov)
    store.mark_complete("final_model", fp)
    return prov
