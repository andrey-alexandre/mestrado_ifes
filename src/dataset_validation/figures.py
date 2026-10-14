"""Figuras: amostra visual para inspeção humana e distribuições comparativas."""
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.dataset_validation import labels as L

KEY_INDICATORS = [
    "megapixels", "lesion_area_ratio", "lesion_equiv_diameter_ref_px", "lum_mean", "lum_std",
    "lum_dynamic_range", "contrast_lesion_skin_dE", "lap_var_global_ref", "lap_var_lesion_ref",
    "edge_strength_boundary_ref", "noise_sigma_skin", "vignetting_ratio", "hair_proxy_frac",
    "glare_frac_global",
]


def contact_sheet(manifest: Sequence[Dict], out_path: Path, per_group: int = 6, thumb: int = 200) -> str:
    """Grade organizada por (dataset × classe): original e imagem mascarada lado a lado.

    Usa os primeiros `per_group` sorteados de cada grupo (ordem do sorteio; sem curadoria).
    """
    groups = [(ds, cls) for ds in ("ph2", "ham10000") for cls in (L.MELANOMA, L.NEVUS)]
    label_h = 18
    cell_w, cell_h = thumb * 2 + 6, thumb + label_h
    sheet = Image.new("RGB", (cell_w * per_group, (cell_h + 22) * len(groups)), "white")
    draw = ImageDraw.Draw(sheet)
    for gi, (ds, cls) in enumerate(groups):
        y0 = gi * (cell_h + 22)
        draw.text((4, y0 + 4), f"{ds.upper()} — {cls}  (primeiros {per_group} do sorteio, seed fixa)", fill="black")
        rows = [r for r in manifest if r["dataset"] == ds and r["class_label"] == cls]
        rows = sorted(rows, key=lambda r: r["draw_rank"])[:per_group]
        for ci, r in enumerate(rows):
            with Image.open(r["image_path"]) as im:
                im = im.convert("RGB")
                with Image.open(r["mask_path"]) as m:
                    m = m.convert("L").resize(im.size, Image.NEAREST)
                arr = np.asarray(im) * (np.asarray(m) > 128)[..., None]
                masked = Image.fromarray(arr.astype(np.uint8))
                for k, img in enumerate((im, masked)):
                    t = img.copy()
                    t.thumbnail((thumb, thumb))
                    sheet.paste(t, (ci * cell_w + k * (thumb + 6), y0 + 22))
            draw.text((ci * cell_w + 2, y0 + 22 + thumb), r["image_id"], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def distribution_plots(quality_rows: Sequence[Dict], out_path: Path,
                       indicators: Sequence[str] = KEY_INDICATORS) -> str:
    """Boxplots + pontos por (dataset × classe) para os indicadores principais."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("ph2", L.MELANOMA), ("ph2", L.NEVUS), ("ham10000", L.MELANOMA), ("ham10000", L.NEVUS)]
    names = ["PH² mel", "PH² nevo", "HAM mel", "HAM nevo"]
    colors = ["#c0392b", "#e59866", "#1f618d", "#85c1e9"]
    n = len(indicators)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n))
    for ax, ind in zip(np.ravel(axes), indicators):
        data = []
        for ds, cls in groups:
            v = [float(r[ind]) for r in quality_rows
                 if r["dataset"] == ds and r["class_label"] == cls and r.get(ind) not in (None, "")]
            data.append([x for x in v if not np.isnan(x)])
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=9, color=colors[i - 1], alpha=0.8)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_title(ind, fontsize=9)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Indicadores computacionais de qualidade (n=20 por grupo) — não são adequação clínica",
                 fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)
