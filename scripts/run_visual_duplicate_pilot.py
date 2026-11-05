"""Staged-filtering PILOT (task 8 of the reconstruction): pHash + geometric
(ORB+RANSAC) verification layered on top of task 7's dHash candidates.

Task 7 found 421,196 dHash candidates across HAM10000 x ISIC2018 Task1 and
HAM10000 x BCN20000 (threshold <=10, 64-bit dHash). Its report concluded
most were likely noise; the researcher judged that conclusion an
overreach — two inspected distance-0 pairs being visually different proves
dHash collision != duplicate, but proves nothing about the other ~421k
pairs, about patient identity (visual appearance is not patient metadata),
or that a uniform-random-bits null model is the right reference distribution
for real dermoscopy images. This script does not repeat those three claims.

This is a PILOT: it validates the staged method (directed preprocessing
audit, pHash, ORB+RANSAC) on a stratified SAMPLE of task 7's existing
candidates. It does NOT run a full dHash/pHash search over all pairs, does
NOT re-run task 7's candidate search, and does NOT touch any task 7
artifact (visual_candidates_index.json, visual_candidates/,
04_visual_duplicate_candidates_report.md are read-only inputs here).

No exclusion of any lesion/image is decided or implemented at any stage,
including for pairs with strong geometric support.

Usage:
    python scripts/run_visual_duplicate_pilot.py
    python scripts/run_visual_duplicate_pilot.py --seed 42 --mid-bucket-sample-size 800
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import geometric_verification as gv  # noqa: E402
from src.data_audit import phash  # noqa: E402
from src.data_audit import preprocessing_audit as pa  # noqa: E402
from src.data_audit import visual_duplicate_pilot_report as pilot_report  # noqa: E402
from src.data_audit import visual_duplicates as vd  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiments" / "data_audit"
PILOT_EVIDENCE_DIR = MANIFEST_DIR / "visual_duplicate_pilot_evidence"
PILOT_INDEX_PATH = MANIFEST_DIR / "visual_duplicate_pilot_index.json"
REPORT_PATH = REPO_ROOT / "docs" / "reconstruction" / "05_visual_duplicate_pilot_report.md"

DEFAULT_SEED = 42
N_SAMPLES_PER_SOURCE_PREPROCESSING = 200
N_REAL_IMAGES_FOR_PHASH_VALIDATION = 8
DEFAULT_MID_BUCKET_SAMPLE_SIZE = 800  # for distance 3-7 and distance 8-10 each
MAX_EVIDENCE_RENDERED = 300
GEOMETRIC_MAX_DIM = 800  # longer-side cap for geometric verification decode, keeps ORB fast
PHASH_VARIANT_LABELS = {
    "identical_copy": "Cópia idêntica",
    "jpeg_recompress_q40": "Recompressão JPEG (qualidade 40)",
    "resize_50pct": "Redimensionamento (50%)",
    "crop_5pct_border": "Recorte pequeno (5% da borda)",
    "rotate_90": "Rotação 90°",
    "mirror_horizontal": "Espelhamento horizontal",
    "crop_30pct_border": "Recorte significativo (30% da borda)",
}


class ImageSourceReader:
    """Dispatches byte reads across disk (HAM10000) and the already-
    materialized zips (ISIC2018 Task1, BCN20000 train/test) — the exact
    same files tasks 6/7 used; nothing is re-extracted or re-listed here.
    """

    def __init__(self, ham_dir: Path, isic_zip_path: Path, bcn_train_zip_path: Path, bcn_test_zip_path: Path):
        self.ham_dir = ham_dir
        self._isic_zf = zipfile.ZipFile(isic_zip_path)
        self._bcn_train_zf = zipfile.ZipFile(bcn_train_zip_path)
        self._bcn_test_zf = zipfile.ZipFile(bcn_test_zip_path)

    def read_ham10000(self, path: str) -> bytes:
        return (self.ham_dir / path).read_bytes()

    def read_other(self, path: str) -> bytes:
        """path_b from a task 7 candidate: dispatched by its own prefix,
        which already distinguishes ISIC2018 Task1 vs BCN20000 train/test."""
        if path.startswith("bcn_20k_train/"):
            return self._bcn_train_zf.read(path)
        if path.startswith("bcn_20k_test/"):
            return self._bcn_test_zf.read(path)
        return self._isic_zf.read(path)

    def close(self) -> None:
        self._isic_zf.close()
        self._bcn_train_zf.close()
        self._bcn_test_zf.close()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _br(n: int) -> str:
    """pt-BR thousands separator, applied to a single number in isolation
    (see the matching helper/docstring in visual_duplicate_pilot_report.py
    for why chaining .replace(",", ".") onto a multi-line f-string is a bug,
    not a style choice)."""
    return f"{n:,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Item 1: directed preprocessing audit
# ---------------------------------------------------------------------------


def run_preprocessing_audit(
    reader: ImageSourceReader,
    ham_manifest: list[dict[str, Any]],
    isic_manifest: list[dict[str, Any]],
    bcn_manifest: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    path_to_dhash_ham: dict[str, int],
    path_to_dhash_other: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    print("=== Item 1: auditoria dirigida de pré-processamento ===")

    exif_verify = pa.verify_open_does_not_apply_exif_orientation()
    print(f"PIL.Image.open aplica orientação EXIF automaticamente? {exif_verify['open_applies_orientation_automatically']}")

    sources = {
        "HAM10000": (ham_manifest, reader.read_ham10000),
        "ISIC2018 Task1": (isic_manifest, reader.read_other),
        "BCN20000": (bcn_manifest, reader.read_other),
    }

    exif_by_source: dict[str, Any] = {}
    dimensions_by_source: dict[str, Any] = {}
    degenerate_by_source: dict[str, Any] = {}

    for source_name, (manifest, read_fn) in sources.items():
        sampled_entries = pa.sample_paths(manifest, N_SAMPLES_PER_SOURCE_PREPROCESSING, seed)
        samples = [(e["path"], read_fn(e["path"])) for e in sampled_entries]

        exif_result = pa.exif_orientation_audit(samples)
        dim_result = pa.dimension_audit(samples)
        degen_result = pa.degenerate_image_audit(samples)

        exif_by_source[source_name] = exif_result
        dimensions_by_source[source_name] = dim_result
        degenerate_by_source[source_name] = degen_result

        print(
            f"[{source_name}] amostra={len(samples)} orientação não-padrão="
            f"{exif_result['n_non_default_orientation']} degeneradas={degen_result['n_flagged']}"
        )

    hub_result = pa.hub_hash_audit(candidates, path_to_dhash_ham, path_to_dhash_other, top_n=20)
    print(
        f"hub hashes: top-20 tocam {hub_result['n_candidates_touching_top_hashes']} "
        f"({hub_result['top_hashes_share_of_candidates']:.2%}) dos {hub_result['total_candidates']} candidatos"
    )

    return {
        "exif_verify": exif_verify,
        "exif_by_source": exif_by_source,
        "dimensions_by_source": dimensions_by_source,
        "degenerate_by_source": degenerate_by_source,
        "hub_hash_audit": hub_result,
    }


# ---------------------------------------------------------------------------
# Item 2: pHash validation on real data
# ---------------------------------------------------------------------------


def run_phash_validation(
    reader: ImageSourceReader,
    ham_manifest: list[dict[str, Any]],
    hard_pairs: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    print("=== Item 2: validação do pHash em dados reais ===")

    sampled_entries = pa.sample_paths(ham_manifest, N_REAL_IMAGES_FOR_PHASH_VALIDATION, seed)
    variant_distances: dict[str, list[int]] = {}
    for entry in sampled_entries:
        data = reader.read_ham10000(entry["path"])
        image = Image.open(io.BytesIO(data)).convert("RGB")
        variants = vd.generate_variants_of_image(image)
        base_hash = phash.phash_bytes(variants["base"])
        for name, variant_data in variants.items():
            if name == "base":
                continue
            distance = vd.hamming_distance(base_hash, phash.phash_bytes(variant_data))
            variant_distances.setdefault(name, []).append(distance)

    hard_pair_distances = []
    for cand in hard_pairs:
        data_a = reader.read_ham10000(cand["path_a_ham10000"])
        data_b = reader.read_other(cand["path_b"])
        hash_a = phash.phash_bytes(data_a)
        hash_b = phash.phash_bytes(data_b)
        hard_pair_distances.append(vd.hamming_distance(hash_a, hash_b))

    # Threshold derivation: the "clean" reprocessing cluster is whichever
    # named variants have max distance strictly below the hard-pairs'
    # minimum distance (an actual gap, measured -- not assumed to include
    # every variant). The threshold is the floor of the midpoint between
    # that cluster's max and the hard-pairs' min, same margin-on-both-sides
    # reasoning as task 7's dHash threshold.
    hard_min = min(hard_pair_distances)
    all_variant_names = list(variant_distances.keys())
    clean_variants = [
        name for name in all_variant_names if max(variant_distances[name]) < hard_min
    ]
    non_clean_variants = [name for name in all_variant_names if name not in clean_variants]

    if clean_variants:
        clean_max = max(max(variant_distances[name]) for name in clean_variants)
        threshold = (clean_max + hard_min) // 2
        gap_text = (
            f"Separação real observada: as variantes "
            f"{', '.join(PHASH_VARIANT_LABELS.get(n, n) for n in clean_variants)} ficaram "
            f"estritamente abaixo da distância mínima observada nos pares difíceis reais "
            f"(máximo {clean_max} vs. mínimo real de pares difíceis {hard_min}) — gap real de "
            f"{hard_min - clean_max} bits. "
        )
        if non_clean_variants:
            gap_text += (
                "**Achado real, não esperado pelo comportamento do dHash na Tarefa 7**: "
                f"{', '.join(PHASH_VARIANT_LABELS.get(n, n) for n in non_clean_variants)} NÃO "
                "ficaram separadas dos pares difíceis reais (sua distância máxima observada "
                "cai dentro ou acima da faixa de distância dos pares difíceis) — ao contrário "
                "do dHash, este pHash (DCT global de imagem inteira) não é tolerante a recorte, "
                "mesmo pequeno: um recorte desloca o conteúdo espacial o bastante para alterar "
                "substancialmente os coeficientes DCT de baixa frequência. Isto é reportado "
                "como limitação real, não escondido."
            )
        else:
            gap_text += "Todas as variantes testadas ficaram dentro do agrupamento limpo."
    else:
        # No variant cleanly separates from the hard pairs -- report this
        # explicitly rather than inventing a threshold; fall back to a
        # permissive threshold wide enough to keep identical_copy at least.
        threshold = max(variant_distances.get("identical_copy", [0])) + 1
        gap_text = (
            "**Achado real: nenhuma variante testada ficou claramente separada da distância "
            "mínima observada nos pares difíceis reais.** Não há uma lacuna limpa para derivar "
            "um limiar com margem em ambos os lados; isto é reportado explicitamente, não "
            "escondido atrás de um limiar escolhido sem base nos dados."
        )

    variant_stats = []
    for name in all_variant_names:
        dists = variant_distances[name]
        variant_stats.append(
            {
                "variant": name,
                "label": PHASH_VARIANT_LABELS.get(name, name),
                "min": min(dists),
                "mean": sum(dists) / len(dists),
                "max": max(dists),
                "recovered": max(dists) <= threshold,
            }
        )

    print(f"hard pairs (n={len(hard_pair_distances)}): min={hard_min} max={max(hard_pair_distances)}")
    print(f"pHash threshold derivado: {threshold}")

    return {
        "seed": seed,
        "n_real_images": len(sampled_entries),
        "variant_stats": variant_stats,
        "hard_pairs": {
            "n_pairs": len(hard_pair_distances),
            "min": hard_min,
            "max": max(hard_pair_distances),
            "mean": statistics.mean(hard_pair_distances),
            "median": statistics.median(hard_pair_distances),
        },
        "threshold": threshold,
        "threshold_derivation_text": gap_text,
        "hash_bits": phash.PHASH_LOW_FREQ_SIZE**2,
    }


# ---------------------------------------------------------------------------
# Item 4: stratified pilot sample, pHash filter, geometric verification
# ---------------------------------------------------------------------------


def _stratified_sample(candidates: list[dict[str, Any]], seed: int, mid_bucket_sample_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bucket_0_2 = [c for c in candidates if c["distance"] <= 2]
    bucket_3_7 = [c for c in candidates if 3 <= c["distance"] <= 7]
    bucket_8_10 = [c for c in candidates if c["distance"] >= 8]

    rng = random.Random(seed)
    sample_3_7 = rng.sample(bucket_3_7, min(mid_bucket_sample_size, len(bucket_3_7)))
    rng2 = random.Random(seed + 1)  # distinct draw from the 3-7 sample, same documented seed base
    sample_8_10 = rng2.sample(bucket_8_10, min(mid_bucket_sample_size, len(bucket_8_10)))

    sampled = bucket_0_2 + sample_3_7 + sample_8_10
    bucket_info = [
        {"label": "0-2", "n_pool": len(bucket_0_2), "n_sampled": len(bucket_0_2), "all": True, "seed": None},
        {"label": "3-7", "n_pool": len(bucket_3_7), "n_sampled": len(sample_3_7), "all": False, "seed": seed},
        {"label": "8-10", "n_pool": len(bucket_8_10), "n_sampled": len(sample_8_10), "all": False, "seed": seed + 1},
    ]
    return sampled, bucket_info


def _decode_grayscale_array(data: bytes, max_dim: int = GEOMETRIC_MAX_DIM) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("L")
    w, h = image.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return np.array(image)


def run_pilot_sample(
    reader: ImageSourceReader,
    candidates: list[dict[str, Any]],
    seed: int,
    mid_bucket_sample_size: int,
    phash_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("=== Item 4: amostra estratificada do piloto ===")
    sampled, bucket_info = _stratified_sample(candidates, seed, mid_bucket_sample_size)
    print(f"amostra total do piloto: {len(sampled)} pares")
    for b in bucket_info:
        print(f"  distância {b['label']}: {b['n_sampled']}/{b['n_pool']}")

    phash_cache: dict[tuple[str, str], int] = {}

    def cached_phash(side: str, path: str) -> int | None:
        key = (side, path)
        if key in phash_cache:
            return phash_cache[key]
        try:
            data = reader.read_ham10000(path) if side == "ham" else reader.read_other(path)
            h = phash.phash_bytes(data)
        except (OSError, zipfile.BadZipFile):
            phash_cache[key] = None
            return None
        phash_cache[key] = h
        return h

    array_cache: dict[tuple[str, str], np.ndarray] = {}

    def cached_array(side: str, path: str) -> np.ndarray | None:
        key = (side, path)
        if key in array_cache:
            return array_cache[key]
        try:
            data = reader.read_ham10000(path) if side == "ham" else reader.read_other(path)
            arr = _decode_grayscale_array(data)
        except (OSError, zipfile.BadZipFile):
            array_cache[key] = None
            return None
        array_cache[key] = arr
        return arr

    results: list[dict[str, Any]] = []
    n_survivors = 0
    evidence_rendered = 0
    PILOT_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate pHash for all sampled pairs first, so evidence rendering can
    # be prioritized by strongest geometric support rather than scan order.
    for cand in sampled:
        hash_a = cached_phash("ham", cand["path_a_ham10000"])
        hash_b = cached_phash("other", cand["path_b"])
        entry: dict[str, Any] = {
            "pair": cand["pair"],
            "path_a_ham10000": cand["path_a_ham10000"],
            "path_b": cand["path_b"],
            "dhash_distance": cand["distance"],
            "phash_distance": None,
            "phash_survivor": False,
            "geometric_status": "not_run_phash_filtered",
            "n_keypoints_a": None,
            "n_keypoints_b": None,
            "n_matches": None,
            "n_inliers": None,
            "inlier_ratio": None,
            "geometric_strength": "not_run",
            "evidence_image": None,
        }
        if hash_a is None or hash_b is None:
            entry["geometric_status"] = "phash_read_error"
            results.append(entry)
            continue

        phash_distance = vd.hamming_distance(hash_a, hash_b)
        entry["phash_distance"] = phash_distance
        entry["phash_survivor"] = phash_distance <= phash_threshold
        if entry["phash_survivor"]:
            n_survivors += 1
        results.append(entry)

    print(f"sobreviventes do filtro pHash (distância <= {phash_threshold}): {n_survivors}/{len(sampled)}")

    print("=== Item 3 aplicado aos sobreviventes: verificação geométrica (ORB+RANSAC) ===")
    survivors = [e for e in results if e["phash_survivor"]]
    # Render evidence for the pairs with the strongest signal first, capped
    # at MAX_EVIDENCE_RENDERED -- same "report true count, render a capped
    # sample" convention as task 7.
    n_evaluated = 0
    n_forte = 0
    n_fraco = 0
    n_not_evaluable = 0

    geometric_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in survivors:
        array_a = cached_array("ham", entry["path_a_ham10000"])
        array_b = cached_array("other", entry["path_b"])
        if array_a is None or array_b is None:
            entry["geometric_status"] = "image_read_error"
            continue
        result = gv.verify_geometric_match(array_a, array_b)
        entry["geometric_status"] = result["status"]
        entry["n_keypoints_a"] = result["n_keypoints_a"]
        entry["n_keypoints_b"] = result["n_keypoints_b"]
        entry["n_matches"] = result["n_matches"]
        entry["n_inliers"] = result["n_inliers"]
        entry["inlier_ratio"] = result["inlier_ratio"]
        strength = gv.classify_geometric_support(result)
        entry["geometric_strength"] = strength

        if result["status"] == "evaluated":
            n_evaluated += 1
            if strength == "forte":
                n_forte += 1
            else:
                n_fraco += 1
            geometric_results.append((entry, result))
        else:
            n_not_evaluable += 1

    geometric_results.sort(key=lambda pair: pair[1]["inlier_ratio"], reverse=True)
    for i, (entry, result) in enumerate(geometric_results):
        if i >= MAX_EVIDENCE_RENDERED:
            break
        array_a = cached_array("ham", entry["path_a_ham10000"])
        array_b = cached_array("other", entry["path_b"])
        evidence_bgr = gv.draw_match_evidence(array_a, array_b, result)
        out_path = PILOT_EVIDENCE_DIR / f"{i:05d}.png"
        Image.fromarray(evidence_bgr[:, :, ::-1]).save(out_path)  # cv2 is BGR, PIL expects RGB
        entry["evidence_image"] = str(out_path.relative_to(REPO_ROOT)).replace("\\", "/")
        evidence_rendered += 1

    print(
        f"avaliados geometricamente={n_evaluated} forte={n_forte} fraco={n_fraco} "
        f"não-avaliáveis={n_not_evaluable} evidências renderizadas={evidence_rendered}"
    )

    funnel = {
        "total_task7_candidates": len(candidates),
        "n_sampled": len(sampled),
        "buckets": bucket_info,
        "phash_threshold": phash_threshold,
        "n_phash_survivors": n_survivors,
        "phash_survivor_rate": n_survivors / len(sampled) if sampled else 0.0,
        "n_geometric_evaluated": n_evaluated,
        "n_forte": n_forte,
        "n_fraco": n_fraco,
        "n_not_evaluable": n_not_evaluable,
        "n_evidence_rendered": evidence_rendered,
        "max_evidence_rendered": MAX_EVIDENCE_RENDERED,
    }
    return results, funnel


def _build_recommendation_text(funnel: dict[str, Any], phash_validation: dict[str, Any]) -> str:
    """Code-generated from this run's real funnel numbers, not a static
    narrative -- a future rerun with a different seed/threshold/sample size
    would get a recommendation that actually reflects what happened, same
    "no stale manual prose" lesson as tasks 5/7.
    """
    lines = ["Recomendação técnica (para revisão do pesquisador, não decidida nem executada aqui):", ""]

    lines.append(
        "- **Manter o pipeline em 3 etapas** (dHash -> pHash -> ORB+RANSAC) para a execução "
        "completa futura, na mesma ordem: cada etapa é mais cara e mais discriminativa que a "
        "anterior, e aplicar a mais cara (geométrica) só aos sobreviventes das etapas baratas é "
        f"o que torna uma execução completa nos {_br(funnel['total_task7_candidates'])} "
        "candidatos (e não só numa amostra) viável em tempo razoável."
    )

    if funnel["n_phash_survivors"] == 0:
        lines.append(
            f"- **Achado real deste piloto que precisa ser resolvido antes da execução completa: "
            f"ZERO dos {_br(funnel['n_sampled'])} pares amostrados (incluindo os "
            f"{_br(next(b['n_pool'] for b in funnel['buckets'] if b['label'] == '0-2'))} pares "
            f"difíceis reais de distância dHash 0-2 completos) sobreviveram ao filtro pHash** "
            f"(limiar {funnel['phash_threshold']}; a menor distância pHash observada em toda a "
            "amostra foi bem acima do limiar, não um quase-empate). A etapa geométrica (item 3) "
            "não chegou a ser exercida em nenhum par real neste piloto como consequência direta "
            "disso -- só em dados sintéticos/teste de integração. Antes de rodar a execução "
            "completa com esta mesma configuração, **recomenda-se validar diretamente com "
            "ORB+RANSAC uma amostra de candidatos dHash de baixa distância SEM passar pelo "
            "filtro pHash**, para checar se o limiar pHash (que não recupera recorte, ver "
            "achado do item 2) está descartando positivos reais antes que a verificação "
            "geométrica tenha a chance de avaliá-los -- o resultado atual é consistente tanto "
            "com \"não há duplicatas reais na amostra\" quanto com \"o filtro pHash está "
            "rejeitando duplicatas reais baseadas em recorte antes da etapa geométrica\", e "
            "este piloto não consegue distinguir as duas."
        )
    else:
        lines.append(
            f"- {_br(funnel['n_phash_survivors'])} de {_br(funnel['n_sampled'])} pares amostrados "
            f"sobreviveram ao filtro pHash (limiar {funnel['phash_threshold']}) e "
            f"{_br(funnel['n_geometric_evaluated'])} puderam ser avaliados geometricamente "
            f"({_br(funnel['n_forte'])} com suporte forte, {_br(funnel['n_fraco'])} fraco, "
            f"{_br(funnel['n_not_evaluable'])} não avaliáveis) -- rodar o filtro pHash nos "
            f"{_br(funnel['total_task7_candidates'])} candidatos completos da Tarefa 7 usando "
            "este mesmo limiar é viável e deve produzir um funil proporcionalmente comparável."
        )

    lines.append(
        "- **Rodar o filtro pHash nos candidatos completos da Tarefa 7** (não recalcular dHash; "
        "reaproveitar o índice e os manifestos já existentes), usando o limiar derivado neste "
        "piloto a partir de dados reais -- reavaliar esse limiar nos números completos antes de "
        f"finalizar, já que {phash_validation['hard_pairs']['n_pairs']} pares difíceis e "
        f"{phash_validation['n_real_images']} imagens reais é uma base pequena para fixar um "
        "valor definitivo."
    )
    lines.append(
        "- **Documentar explicitamente, na execução completa, que recorte (mesmo pequeno) não é "
        "recuperado pelo pHash nesta implementação** (achado real deste piloto, item 2) -- se "
        "recorte for uma preocupação relevante do protocolo, a etapa pHash não deve ser tratada "
        "como suficiente sozinha para descartar candidatos de baixa distância dHash que o pHash "
        "classificar como não-candidatos; quem decidir o uso final dos dados precisa saber dessa "
        "lacuna."
    )
    lines.append(
        "- **Rodar a verificação geométrica em todos os sobreviventes do pHash na execução "
        "completa**, não apenas numa amostra -- o custo computacional por par (ORB + RANSAC) é "
        "pequeno comparado ao hashing em massa; o gargalo esperado é o número de sobreviventes "
        "do pHash, não o método em si."
    )
    lines.append(
        "- **Revisar os limiares de classificação \"forte\"/\"fraco\"** (`STRONG_MIN_INLIERS`, "
        "`STRONG_MIN_INLIER_RATIO` em `geometric_verification.py`) contra a distribuição real "
        "observada na execução completa -- os valores atuais são um ponto de partida razoável, "
        "não validados contra um conjunto rotulado."
    )
    lines.append(
        "- Para qualquer par classificado como suporte \"forte\" na execução completa: **isto "
        "continua sendo evidência para revisão humana, não confirmação automática de duplicata "
        "nem base para exclusão de dado**."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "dataset_inventory.yaml"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mid-bucket-sample-size", type=int, default=DEFAULT_MID_BUCKET_SAMPLE_SIZE)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ham_dir = Path(config["ham10000"]["images_dir"])
    isic_zip = Path(config["isic2018_task1"]["images_zip"])
    bcn_content = _load_json(MANIFEST_DIR / "bcn20000_content_manifest.json")
    bcn_train_zip = Path(bcn_content["train"]["materialization"]["dest_path"])
    bcn_test_zip = Path(bcn_content["test"]["materialization"]["dest_path"])

    ham_dhash = _load_json(MANIFEST_DIR / "ham10000_dhash_manifest.json")
    isic_dhash = _load_json(MANIFEST_DIR / "isic2018_task1_dhash_manifest.json")
    bcn_dhash = _load_json(MANIFEST_DIR / "bcn20000_dhash_manifest.json")
    index = _load_json(MANIFEST_DIR / "visual_candidates_index.json")
    candidates = index["candidates"]
    print(f"candidatos da Tarefa 7 carregados (somente leitura): {len(candidates)}")

    path_to_dhash_ham = {e["path"]: e["dhash"] for e in ham_dhash["manifest"]}
    path_to_dhash_other = {e["path"]: e["dhash"] for e in isic_dhash["manifest"]}
    path_to_dhash_other.update({e["path"]: e["dhash"] for e in bcn_dhash["manifest"]})

    reader = ImageSourceReader(ham_dir, isic_zip, bcn_train_zip, bcn_test_zip)
    try:
        preprocessing = run_preprocessing_audit(
            reader,
            ham_dhash["manifest"],
            isic_dhash["manifest"],
            bcn_dhash["manifest"],
            candidates,
            path_to_dhash_ham,
            path_to_dhash_other,
            args.seed,
        )

        hard_pairs = [c for c in candidates if c["distance"] <= 2]
        phash_validation = run_phash_validation(reader, ham_dhash["manifest"], hard_pairs, args.seed)

        pilot_results, funnel = run_pilot_sample(
            reader, candidates, args.seed, args.mid_bucket_sample_size, phash_validation["threshold"]
        )
    finally:
        reader.close()

    pilot_index = {
        "seed": args.seed,
        "mid_bucket_sample_size": args.mid_bucket_sample_size,
        "phash_threshold": phash_validation["threshold"],
        "funnel": funnel,
        "pairs": pilot_results,
    }
    _save_json(PILOT_INDEX_PATH, pilot_index)
    print(f"índice do piloto gravado: {PILOT_INDEX_PATH}")

    report_text = pilot_report.render_report(
        preprocessing=preprocessing,
        phash_validation=phash_validation,
        funnel=funnel,
        pilot_index_path=PILOT_INDEX_PATH.relative_to(REPO_ROOT),
        recommendation_text=_build_recommendation_text(funnel, phash_validation),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"relatório gravado: {REPORT_PATH}")


if __name__ == "__main__":
    main()
