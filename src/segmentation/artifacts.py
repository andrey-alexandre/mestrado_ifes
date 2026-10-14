"""Persistência estruturada, fingerprints de configuração e retomada de etapas."""
import csv
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ArtifactMismatch(RuntimeError):
    """Artefato existente foi produzido com configuração/insumos diferentes."""


def clean(obj: Any) -> Any:
    """Converte tipos numpy e NaN/inf em equivalentes JSON-estritos (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str, obj: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    tmp.replace(p)
    return str(p)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict], fieldnames: Optional[Iterable[str]] = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [clean(r) for r in rows]
    cols = list(fieldnames) if fieldnames else (list(rows[0].keys()) if rows else [])
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return str(p)


def read_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fingerprint(*objs: Any) -> str:
    """SHA-256 estável de estruturas JSON-serializáveis."""
    blob = json.dumps([clean(o) for o in objs], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(seed: Optional[int] = None) -> Dict:
    """Versões de dependências, plataforma e commit git (melhor esforço)."""
    versions: Dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "torchvision", "numpy", "scipy", "sklearn", "PIL", "pandas", "yaml",
                "matplotlib"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "unavailable"
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return {"versions": versions, "git_commit": commit, "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat()}


class StageStore:
    """Marcadores de conclusão por etapa, com validação de compatibilidade.

    Cada etapa grava `{results_dir}/_stages/{stage}.json` com o fingerprint da
    configuração e dos insumos. Reexecutar com fingerprint igual pula a etapa;
    com fingerprint diferente exige `force=True` (nunca sobrescreve em silêncio).
    """

    def __init__(self, results_dir: str, force: bool = False):
        self.dir = Path(results_dir) / "_stages"
        self.force = force

    def _path(self, stage: str) -> Path:
        return self.dir / (stage.replace("/", "__") + ".json")

    def is_complete(self, stage: str, fp: str) -> bool:
        p = self._path(stage)
        if not p.exists():
            return False
        if self.force:
            logger.info("[%s] --force: reexecutando.", stage)
            return False
        meta = read_json(str(p))
        if meta.get("fingerprint") != fp:
            raise ArtifactMismatch(
                f"Etapa '{stage}' já concluída com configuração/insumos diferentes "
                f"({p}). Use --force para sobrescrever ou restaure a configuração original."
            )
        logger.info("[%s] artefatos compatíveis encontrados; etapa pulada.", stage)
        return True

    def mark_complete(self, stage: str, fp: str, extra: Optional[Dict] = None) -> None:
        write_json(str(self._path(stage)),
                   {"stage": stage, "fingerprint": fp, "completed_at":
                    datetime.now(timezone.utc).isoformat(), **(extra or {})})

    def read(self, stage: str) -> Optional[Dict]:
        p = self._path(stage)
        return read_json(str(p)) if p.exists() else None
