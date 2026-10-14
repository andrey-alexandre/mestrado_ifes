"""Persistência estruturada e proveniência (independente do experimento de segmentação)."""
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


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


def write_json(path: Any, obj: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp.replace(p)
    return str(p)


def read_json(path: Any) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Any, rows: List[Dict], fieldnames: Optional[Iterable[str]] = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [clean(r) for r in rows]
    if fieldnames:
        cols = list(fieldnames)
    else:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return str(p)


def read_csv(path: Any) -> List[Dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(*objs: Any) -> str:
    blob = json.dumps([clean(o) for o in objs], sort_keys=True, default=str, ensure_ascii=False)
    return sha256_text(blob)


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, cwd=str(ROOT))
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def provenance(seed: Optional[int] = None) -> Dict:
    """Versões de dependências, plataforma, commit e estado da árvore (melhor esforço)."""
    versions: Dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "PIL", "pandas", "sklearn", "yaml", "matplotlib", "torch",
                "langchain", "langchain_core", "langgraph", "langchain_openai",
                "langchain_google_genai"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "unavailable"
    status = _git("status", "--porcelain", "--", "src", "scripts", "config")
    return {
        "versions": versions,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_paths_in_src_scripts_config": (status or "").splitlines()[:50],
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
