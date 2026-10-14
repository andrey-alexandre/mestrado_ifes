"""Gráficos do relatório. Consome apenas artefatos estruturados (CSV/JSON), nunca o treino."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

APPROACH_LABEL = {"direct_transfer": "Transferência direta", "finetune": "Fine-tuning"}
COLORS = {"direct_transfer": "#2a6f97", "finetune": "#d1495b"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _vals(rows: Sequence[Dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) if r.get(key) not in (None, "", "None") else np.nan for r in rows])


def plot_metric_distribution(direct: Sequence[Dict], tuned: Sequence[Dict], metric: str, path: str) -> str:
    plt = _plt()
    a, b = _vals(direct, metric), _vals(tuned, metric)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    data = [a[~np.isnan(a)], b[~np.isnan(b)]]
    ax.boxplot(data, widths=0.5, showmeans=True)
    ax.set_xticks([1, 2], [APPROACH_LABEL["direct_transfer"], APPROACH_LABEL["finetune"]])
    ax.set_ylabel(metric.upper() + (" (px)" if metric == "hd95" else ""))
    ax.set_title(f"{metric.upper()} por imagem (n={len(a)})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_paired_difference(diffs: Sequence[Dict], metric: str, path: str) -> str:
    plt = _plt()
    d = _vals(diffs, f"{metric}_diff")
    d = d[~np.isnan(d)]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.hist(d, bins=min(30, max(5, len(d) // 4)), color="#6a4c93", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    if len(d):
        ax.axvline(float(d.mean()), color="#d1495b", ls="--", label=f"média = {d.mean():.4f}")
        ax.legend()
    ax.set_xlabel(f"Δ{metric.upper()} (fine-tuning − transferência direta)")
    ax.set_ylabel("Nº de imagens")
    ax.set_title("Distribuição das diferenças pareadas")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_per_fold(summary: Dict, metric: str, path: str) -> str:
    plt = _plt()
    folds = sorted(summary["per_fold"], key=int)
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, tag in enumerate(("direct_transfer", "finetune")):
        means = [summary["per_fold"][f][metric][tag]["mean"] for f in folds]
        ax.bar(x + (i - 0.5) * 0.38, means, width=0.38, label=APPROACH_LABEL[tag], color=COLORS[tag])
    ax.set_xticks(x, [f"fold {f}" for f in folds])
    ax.set_ylabel(f"{metric.upper()} médio")
    ax.set_title(f"{metric.upper()} por fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    from scipy import ndimage
    m = mask.astype(bool)
    edge = m & ~ndimage.binary_erosion(m, iterations=width)
    return edge


def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Contorno verde = referência; vermelho = predição."""
    out = image.copy()
    out[_boundary(gt)] = (0, 255, 0)
    out[_boundary(pred)] = (255, 0, 0)
    return out


def plot_examples(rows_by_id: Dict[str, Dict], ids: List[str], titles: Optional[Dict[str, str]],
                  path: str, suptitle: str, ncols: int = 4) -> Optional[str]:
    """Grade de sobreposições para os `ids`. Retorna None se `ids` for vazio."""
    from PIL import Image
    if not ids:
        return None
    plt = _plt()
    n = len(ids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, iid in zip(axes.ravel(), ids):
        r = rows_by_id[iid]
        with Image.open(r["image_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        with Image.open(r["gt_mask_path"]) as m:
            gt = np.asarray(m.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
        with Image.open(r["pred_mask_path"]) as m:
            pred = np.asarray(m.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
        ax.imshow(overlay(rgb, gt, pred))
        ax.set_title((titles or {}).get(iid, iid), fontsize=8)
    fig.suptitle(suptitle + " — verde: referência, vermelho: predição", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_diagnostic_metrics(diag: Dict, path: str) -> str:
    plt = _plt()
    metrics = ["sensitivity", "specificity", "precision", "f1", "accuracy"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, c in enumerate(("C0", "C1")):
        vals = [np.nan if diag[c][m] is None else diag[c][m] for m in metrics]
        ax.bar(x + (i - 0.5) * 0.38, vals, width=0.38, label={"C0": "C0 (GT)", "C1": "C1 (automática)"}[c])
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Desempenho diagnóstico do sistema multiagente")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
