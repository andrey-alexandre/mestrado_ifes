"""Indicadores computacionais objetivos de qualidade/aquisição das imagens.

IMPORTANTE: são medidas de sinal (nitidez, exposição, contraste, resolução, proxies de
artefato). NÃO equivalem a adequação clínica nem a dificuldade diagnóstica. Medidas de
nitidez dependem da escala; por isso as principais são reportadas também após
normalização de largura (`*_ref`). Áreas "lesão/pele" usam a máscara de referência
(PH²: manual; HAM10000: máscaras distribuídas com o dataset, a origem de ambas é distinta).
"""
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

NAN = float("nan")

INDICATOR_GROUPS: Dict[str, str] = {
    "width_px": "resolution", "height_px": "resolution", "megapixels": "resolution",
    "file_bytes": "resolution", "bits_per_pixel_file": "resolution",
    "lesion_area_px": "lesion_geometry", "lesion_area_ratio": "lesion_geometry",
    "lesion_equiv_diameter_px": "lesion_geometry", "lesion_equiv_diameter_ref_px": "lesion_geometry",
    "lesion_touches_border": "lesion_geometry", "lesion_centroid_offset": "lesion_geometry",
    "lum_mean": "exposure", "lum_std": "contrast", "lum_p1": "exposure", "lum_p99": "exposure",
    "lum_dynamic_range": "contrast", "clip_low_frac": "exposure", "clip_high_frac": "exposure",
    "lum_entropy_bits": "contrast",
    "contrast_lesion_skin_dE": "contrast", "contrast_lesion_skin_lum": "contrast",
    "lap_var_global": "sharpness", "tenengrad_global": "sharpness",
    "lap_var_global_ref": "sharpness", "tenengrad_global_ref": "sharpness",
    "lap_var_lesion_ref": "sharpness", "tenengrad_lesion_ref": "sharpness",
    "edge_strength_boundary_ref": "sharpness", "noise_sigma_skin": "noise",
    "vignetting_ratio": "artifact", "dark_corner_frac": "artifact", "black_border_frac": "artifact",
    "hair_proxy_frac": "artifact", "glare_frac_global": "artifact", "glare_frac_lesion": "artifact",
    "skin_rg_ratio": "color_cast", "skin_bg_ratio": "color_cast", "lesion_saturation_mean": "color",
}
INDICATORS = list(INDICATOR_GROUPS)


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_mask(path: str, shape) -> np.ndarray:
    """Binariza como em `apply_precomputed_mask` (> 128) e ajusta o tamanho (NEAREST)."""
    with Image.open(path) as m:
        m = m.convert("L")
        if m.size != (shape[1], shape[0]):
            m = m.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(m) > 128


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def srgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB (uint8, Nx3) -> CIELAB D65."""
    c = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _sharp(gray: np.ndarray, region: Optional[np.ndarray] = None) -> Dict[str, float]:
    lap = ndi.laplace(gray)
    gx, gy = ndi.sobel(gray, axis=1) / 8.0, ndi.sobel(gray, axis=0) / 8.0
    ten = gx ** 2 + gy ** 2
    if region is not None:
        if region.sum() < 25:
            return {"lap_var": NAN, "tenengrad": NAN, "edge": NAN}
        return {"lap_var": float(lap[region].var()), "tenengrad": float(ten[region].mean()),
                "edge": float(np.sqrt(ten[region]).mean())}
    return {"lap_var": float(lap.var()), "tenengrad": float(ten.mean()), "edge": float(np.sqrt(ten).mean())}


def _resize_gray(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape
    if w == width:
        return gray
    new_h = max(1, int(round(h * width / w)))
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((width, new_h), Image.LANCZOS), dtype=np.float32)


def _noise_sigma(gray: np.ndarray, region: np.ndarray) -> float:
    """Estimativa de ruído de Immerkaer restrita à região (pele)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = np.abs(ndi.convolve(gray, k, mode="reflect"))
    inner = region.copy()
    inner[:2, :] = inner[-2:, :] = False
    inner[:, :2] = inner[:, -2:] = False
    if inner.sum() < 100:
        return NAN
    return float(math.sqrt(math.pi / 2.0) * conv[inner].mean() / 6.0)


def compute_indicators(rgb: np.ndarray, mask: np.ndarray, qcfg: Dict, file_bytes: int = 0) -> Dict[str, float]:
    """Calcula todos os indicadores para uma imagem e sua máscara de referência."""
    h, w = rgb.shape[:2]
    npx = h * w
    out: Dict[str, float] = {k: NAN for k in INDICATORS}
    out.update(width_px=w, height_px=h, megapixels=npx / 1e6, file_bytes=file_bytes,
               bits_per_pixel_file=(file_bytes * 8 / npx) if file_bytes else NAN)
    ref_w = int(qcfg["reference_width_px"])
    scale = ref_w / w

    gray = luminance(rgb)
    area = int(mask.sum())
    out["lesion_area_px"] = area
    out["lesion_area_ratio"] = area / npx
    if area > 0:
        out["lesion_equiv_diameter_px"] = 2.0 * math.sqrt(area / math.pi)
        out["lesion_equiv_diameter_ref_px"] = out["lesion_equiv_diameter_px"] * scale
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean() / h, xs.mean() / w
        out["lesion_centroid_offset"] = float(math.hypot(cx - 0.5, cy - 0.5))
        out["lesion_touches_border"] = float(mask[0, :].any() or mask[-1, :].any()
                                             or mask[:, 0].any() or mask[:, -1].any())

    # exposição / contraste globais
    out["lum_mean"], out["lum_std"] = float(gray.mean()), float(gray.std())
    p1, p99 = np.percentile(gray, [1, 99])
    out["lum_p1"], out["lum_p99"], out["lum_dynamic_range"] = float(p1), float(p99), float(p99 - p1)
    out["clip_low_frac"] = float((gray <= 5).mean())
    out["clip_high_frac"] = float((gray >= 250).mean())
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256) / npx
    nz = hist[hist > 0]
    out["lum_entropy_bits"] = float(-(nz * np.log2(nz)).sum())

    # regiões lesão / pele (anel afastado da borda) / interior / faixa de borda
    have_lesion = 0 < area < npx
    skin = np.zeros_like(mask)
    interior = np.zeros_like(mask)
    band = np.zeros_like(mask)
    if have_lesion:
        d_out = ndi.distance_transform_edt(~mask)
        d_in = ndi.distance_transform_edt(mask)
        skin = d_out > qcfg["skin_ring_dilation_px"]
        interior = d_in > qcfg["lesion_interior_erosion_px"]
        b = qcfg["boundary_band_px"]
        band = ((d_in <= b) & mask) | ((d_out <= b) & ~mask)
    skin_ok = skin.sum() >= 0.01 * npx

    if have_lesion and skin_ok:
        les_lab = srgb_to_lab(rgb[mask]).mean(axis=0)
        skin_lab = srgb_to_lab(rgb[skin]).mean(axis=0)
        out["contrast_lesion_skin_dE"] = float(np.linalg.norm(les_lab - skin_lab))
        out["contrast_lesion_skin_lum"] = float(abs(gray[mask].mean() - gray[skin].mean()))
        sk = rgb[skin].astype(np.float64).mean(axis=0)
        out["skin_rg_ratio"] = float(sk[0] / max(sk[1], 1e-6))
        out["skin_bg_ratio"] = float(sk[2] / max(sk[1], 1e-6))
        out["noise_sigma_skin"] = _noise_sigma(gray, skin)

    # nitidez (nativa e em largura de referência)
    g = _sharp(gray)
    out["lap_var_global"], out["tenengrad_global"] = g["lap_var"], g["tenengrad"]
    gref = _resize_gray(gray, ref_w)
    gr = _sharp(gref)
    out["lap_var_global_ref"], out["tenengrad_global_ref"] = gr["lap_var"], gr["tenengrad"]
    if have_lesion:
        mref = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (gref.shape[1], gref.shape[0]), Image.NEAREST)) > 128
        d_in_r = ndi.distance_transform_edt(mref)
        d_out_r = ndi.distance_transform_edt(~mref)
        interior_r = d_in_r > max(1, qcfg["lesion_interior_erosion_px"] * scale)
        band_r = ((d_in_r <= qcfg["boundary_band_px"]) & mref) | ((d_out_r <= qcfg["boundary_band_px"]) & ~mref)
        li = _sharp(gref, interior_r)
        out["lap_var_lesion_ref"], out["tenengrad_lesion_ref"] = li["lap_var"], li["tenengrad"]
        out["edge_strength_boundary_ref"] = _sharp(gref, band_r)["edge"]
        out["lesion_saturation_mean"] = _mean_saturation(rgb[mask])

    # proxies de artefato
    s = int(max(2, round(qcfg["corner_fraction"] * min(h, w))))
    corners = [gray[:s, :s], gray[:s, -s:], gray[-s:, :s], gray[-s:, -s:]]
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    out["vignetting_ratio"] = float(np.mean([c.mean() for c in corners]) / max(center.mean(), 1e-6))
    out["dark_corner_frac"] = float(np.mean([(c < 40).mean() for c in corners]))
    t = max(1, int(round(0.03 * min(h, w))))
    frame = np.ones_like(mask)
    frame[t:-t, t:-t] = False
    out["black_border_frac"] = float((gray[frame] < 20).mean())
    sz = int(qcfg["hair_blackhat_size"])
    blackhat = ndi.grey_closing(gray, size=(sz, sz)) - gray
    nonles = ~mask
    out["hair_proxy_frac"] = float((blackhat[nonles] > qcfg["hair_threshold"]).mean()) if nonles.any() else NAN
    sat_white = rgb.min(axis=2) >= 240
    out["glare_frac_global"] = float(sat_white.mean())
    out["glare_frac_lesion"] = float(sat_white[mask].mean()) if area > 0 else NAN
    return out


def _mean_saturation(pixels: np.ndarray) -> float:
    p = pixels.astype(np.float64) / 255.0
    mx, mn = p.max(axis=1), p.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    return float(sat.mean())


def indicators_for_row(row: Dict, qcfg: Dict) -> Dict[str, float]:
    rgb = load_rgb(row["image_path"])
    mask = load_mask(row["mask_path"], rgb.shape[:2])
    return compute_indicators(rgb, mask, qcfg, file_bytes=Path(row["image_path"]).stat().st_size)
