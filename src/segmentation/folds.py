"""Folds estratificados e reproduzíveis do PH² + manifesto persistido."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.segmentation.data import DataError, Record

logger = logging.getLogger(__name__)

SCHEMA = "ph2_folds_v1"


def ids_fingerprint(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def make_folds(
    records: Sequence[Record],
    n_folds: int = 5,
    seed: int = 42,
    stratify_by: str = "binary_melanoma",
    label_source: str = "clinical_diagnosis",
) -> Dict:
    """Cria o manifesto de folds. Cada imagem cai em exatamente um fold de teste.

    A ordenação por identificador torna o resultado independente da ordem do
    sistema de arquivos.
    """
    recs = sorted(records, key=lambda r: r.image_id)
    if any(r.label is None for r in recs):
        raise DataError("Todos os registros precisam de rótulo para estratificar os folds.")
    if stratify_by == "binary_melanoma":
        strata = np.array([int(r.label) for r in recs])
    elif stratify_by == "clinical_diagnosis":
        strata = np.array([r.diagnosis for r in recs])
    else:
        raise DataError(f"stratify_by inválido: {stratify_by}")
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < n_folds:
        raise DataError(f"A menor classe tem {counts.min()} amostras (< n_folds={n_folds}).")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.full(len(recs), -1, dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(recs)), strata)):
        fold_of[test_idx] = k

    ids = [r.image_id for r in recs]
    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "n_folds": n_folds,
        "stratify_by": stratify_by,
        "label_source": label_source,
        "n_images": len(recs),
        "n_positive": int(sum(r.label for r in recs)),
        "ids_sha256": ids_fingerprint(ids),
        "images": [
            {"image_id": r.image_id, "label": int(r.label), "diagnosis": r.diagnosis,
             "fold": int(f)} for r, f in zip(recs, fold_of)
        ],
        "rounds": [
            {
                "round": k,
                "test_fold": k,
                "train_ids": [i for i, f in zip(ids, fold_of) if f != k],
                "test_ids": [i for i, f in zip(ids, fold_of) if f == k],
            }
            for k in range(n_folds)
        ],
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Dict) -> None:
    """Verifica programaticamente a integridade dos folds (levanta DataError)."""
    if manifest.get("schema") != SCHEMA:
        raise DataError(f"Esquema de manifesto desconhecido: {manifest.get('schema')}")
    all_ids = [im["image_id"] for im in manifest["images"]]
    if len(set(all_ids)) != len(all_ids):
        raise DataError("Identificadores duplicados no manifesto de folds.")
    if manifest["ids_sha256"] != ids_fingerprint(all_ids):
        raise DataError("ids_sha256 do manifesto não confere com a lista de imagens.")
    rounds = manifest["rounds"]
    if len(rounds) != manifest["n_folds"]:
        raise DataError("Número de rodadas difere de n_folds.")
    seen_test: List[str] = []
    universe = set(all_ids)
    for rd in rounds:
        tr, te = set(rd["train_ids"]), set(rd["test_ids"])
        if tr & te:
            raise DataError(f"Rodada {rd['round']}: sobreposição treino/teste {sorted(tr & te)[:5]}")
        if tr | te != universe:
            raise DataError(f"Rodada {rd['round']}: treino ∪ teste não cobre todas as imagens.")
        if not te:
            raise DataError(f"Rodada {rd['round']}: fold de teste vazio.")
        seen_test.extend(rd["test_ids"])
    if len(seen_test) != len(set(seen_test)) or set(seen_test) != universe:
        raise DataError("Cada imagem deve pertencer a exatamente um fold de teste.")
    fold_map = {im["image_id"]: im["fold"] for im in manifest["images"]}
    for rd in rounds:
        if any(fold_map[i] != rd["test_fold"] for i in rd["test_ids"]):
            raise DataError(f"Rodada {rd['round']}: fold das imagens inconsistente com test_ids.")


def save_manifest(manifest: Dict, path: str) -> None:
    validate_manifest(manifest)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


def fold_of_image(manifest: Dict) -> Dict[str, int]:
    return {im["image_id"]: im["fold"] for im in manifest["images"]}


def assert_manifest_matches_records(manifest: Dict, records: Sequence[Record]) -> None:
    """Garante que o manifesto persistido corresponde às imagens atualmente descobertas."""
    if ids_fingerprint([r.image_id for r in records]) != manifest["ids_sha256"]:
        raise DataError(
            "O manifesto de folds não corresponde às imagens do PH² descobertas. "
            "Regenere os folds ou verifique os diretórios."
        )
