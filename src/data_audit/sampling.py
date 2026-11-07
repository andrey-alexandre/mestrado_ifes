"""Task 12: experimental-sample construction for the HAM10000 lesion population.

Methodologically distinct from :mod:`src.data_audit.adapters`/``core``: this
module SAMPLES (selects a subset for the experiment), while ``adapters``/
``core`` only AUDIT structure (pairing, missing/conflicting fields, hashing).
No audit function here duplicates that logic; sanity checks reuse
``core.find_missing_or_conflicting`` directly.

Nothing in this module runs anything (no CLI, no I/O side effects beyond
reading the metadata CSV and an optional exclusions file). The executable
entry point is ``scripts/run_ham10000_sampling.py``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import core

ELIGIBLE_DX_CATEGORIES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def load_ham10000_metadata(metadata_csv: str | Path) -> list[dict[str, Any]]:
    """Read HAM10000_metadata rows, same parsing path as
    :func:`src.data_audit.adapters.audit_ham10000` (``core.parse_csv_text``)."""
    with open(metadata_csv, encoding="utf-8") as f:
        return core.parse_csv_text(f.read())


def check_lesion_population(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-sampling sanity check on the lesion population: no row missing
    ``lesion_id``, no ``lesion_id`` with conflicting ``dx`` across its images.

    Reuses :func:`core.find_missing_or_conflicting` verbatim (no re-implementation).
    Any unexpected ``dx`` value outside :data:`ELIGIBLE_DX_CATEGORIES` is
    reported but NOT excluded here -- the protocol admits all 7 HAM10000
    categories; an unknown value is a data-quality signal to surface, not a
    filtering decision for this function to make silently.
    """
    missing_conflicting = core.find_missing_or_conflicting(rows, "lesion_id", "dx")
    dx_values_seen = {row.get("dx") for row in rows if row.get("dx")}
    unexpected_dx_values = sorted(dx_values_seen - set(ELIGIBLE_DX_CATEGORIES))
    return {
        "ok": not missing_conflicting["missing_id"]
        and not missing_conflicting["conflicting"]
        and not unexpected_dx_values,
        "missing_lesion_id_rows": missing_conflicting["missing_id"],
        "conflicting_lesion_id_to_dx": missing_conflicting["conflicting"],
        "unexpected_dx_values": unexpected_dx_values,
    }


def build_lesion_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse image-level rows into one entry per distinct ``lesion_id``:
    its (single, already validated) ``dx`` and the list of its ``image_id``s.

    Call only after :func:`check_lesion_population` reports ``ok`` -- this
    function does not re-check for missing/conflicting fields.
    """
    lesion_table: dict[str, dict[str, Any]] = {}
    for row in rows:
        lesion_id = row["lesion_id"]
        entry = lesion_table.setdefault(lesion_id, {"dx": row["dx"], "image_ids": []})
        entry["image_ids"].append(row["image_id"])
    return lesion_table


def apply_exclusions(
    lesion_table: Mapping[str, dict[str, Any]], excluded_lesion_ids: Sequence[str]
) -> dict[str, Any]:
    """Remove ``excluded_lesion_ids`` from ``lesion_table`` before sampling.

    Returns the filtered table plus bookkeeping: how many of the given ids
    were actually present (vs. unknown ids, which are reported, not silently
    dropped). Today there is no approved exclusion list (Task 11 closed
    without confirming any duplicate) -- this exists so a future approved
    list can be plugged in without changing the sampling logic.
    """
    excluded_set = set(excluded_lesion_ids)
    present_in_table = excluded_set & lesion_table.keys()
    unknown_ids = sorted(excluded_set - lesion_table.keys())
    filtered = {lid: v for lid, v in lesion_table.items() if lid not in excluded_set}
    return {
        "filtered_lesion_table": filtered,
        "n_requested": len(excluded_set),
        "n_actually_excluded": len(present_in_table),
        "unknown_ids": unknown_ids,
    }


def compute_quotas_largest_remainder(
    counts_by_category: Mapping[str, int], target_n: int
) -> dict[str, Any]:
    """Hamilton's largest-remainder apportionment of ``target_n`` across
    categories, proportional to ``counts_by_category``.

    For each category: exact quota = count / total * target_n. Take the
    floor of each exact quota, then distribute the remaining
    ``target_n - sum(floors)`` units one at a time to the categories with
    the largest fractional remainder, in descending order, until exhausted
    (ties broken by category name, for determinism).

    Returns the per-category calculation table (exact quota, floor, remainder,
    whether it got an extra unit) plus the final integer quota per category.
    Raises ``ValueError`` if ``target_n`` exceeds the total population --
    callers must not silently cap the target.
    """
    total = sum(counts_by_category.values())
    if total == 0:
        raise ValueError(
            "população elegível total é 0 (nenhuma categoria tem lesões) -- "
            "não há base para calcular cotas proporcionais."
        )
    if target_n > total:
        raise ValueError(
            f"target_n ({target_n}) excede a população elegível total ({total}); "
            "o alvo não pode ser atingido sem reduzi-lo explicitamente ou ampliar a população."
        )

    categories = sorted(counts_by_category.keys())
    exact_quotas = {cat: counts_by_category[cat] / total * target_n for cat in categories}
    floors = {cat: int(exact_quotas[cat]) for cat in categories}
    remainders = {cat: exact_quotas[cat] - floors[cat] for cat in categories}

    n_extra_units = target_n - sum(floors.values())
    ranked = sorted(categories, key=lambda cat: (-remainders[cat], cat))
    extra_recipients = set(ranked[:n_extra_units])

    final_quotas = {
        cat: floors[cat] + (1 if cat in extra_recipients else 0) for cat in categories
    }

    calculation_table = [
        {
            "category": cat,
            "n_eligible": counts_by_category[cat],
            "exact_quota": exact_quotas[cat],
            "floor": floors[cat],
            "remainder": remainders[cat],
            "extra_unit": cat in extra_recipients,
            "final_quota": final_quotas[cat],
        }
        for cat in categories
    ]

    assert sum(final_quotas.values()) == target_n
    return {
        "target_n": target_n,
        "total_eligible": total,
        "n_extra_units_distributed": n_extra_units,
        "calculation_table": calculation_table,
        "final_quotas": final_quotas,
    }


def sample_lesions(
    lesion_table: Mapping[str, dict[str, Any]], quotas: Mapping[str, int], seed: int
) -> dict[str, Any]:
    """Within each category, draw ``quotas[category]`` distinct lesion_ids
    without replacement (``random.Random(seed).sample``), then draw exactly
    one image per sampled lesion (separate deterministic draw, same seed,
    so changing one draw's call order elsewhere does not perturb the other).

    Raises ``ValueError`` if a category's quota exceeds its eligible lesion
    count (should not happen if quotas came from
    :func:`compute_quotas_largest_remainder` on the same table, but this
    function does not assume that and checks explicitly).
    """
    lesions_by_category: dict[str, list[str]] = {}
    for lesion_id, entry in lesion_table.items():
        lesions_by_category.setdefault(entry["dx"], []).append(lesion_id)

    rng_lesions = random.Random(seed)
    rng_images = random.Random(seed)

    sampled_rows: list[dict[str, Any]] = []
    for category in sorted(quotas.keys()):
        quota = quotas[category]
        pool = sorted(lesions_by_category.get(category, []))
        if quota > len(pool):
            raise ValueError(
                f"cota de {quota} para categoria '{category}' excede as "
                f"{len(pool)} lesões elegíveis dessa categoria."
            )
        chosen_lesions = rng_lesions.sample(pool, quota)
        for lesion_id in sorted(chosen_lesions):
            image_ids = sorted(lesion_table[lesion_id]["image_ids"])
            chosen_image = rng_images.sample(image_ids, 1)[0]
            sampled_rows.append(
                {
                    "lesion_id": lesion_id,
                    "image_id": chosen_image,
                    "dx": category,
                    "binary_target": "melanoma" if category == "mel" else "nao_melanoma",
                }
            )

    return {"seed": seed, "sampled_rows": sampled_rows}


def read_sample_manifest(manifest_csv: str | Path) -> list[dict[str, Any]]:
    """Read a ``sample_manifest.csv`` produced by ``scripts/run_ham10000_sampling.py``
    (columns: ``lesion_id``, ``image_id``, ``dx``, ``binary_target``).

    Pure file read, same ``core.parse_csv_text`` path as the rest of this
    module -- no sampling logic here.
    """
    with open(manifest_csv, encoding="utf-8") as f:
        return core.parse_csv_text(f.read())


def verify_lesion_and_image_uniqueness(
    manifest_rows: Sequence[Mapping[str, Any]], expected_n: int
) -> dict[str, Any]:
    """Check criteria (a) and (b) from the researcher's 4-point verification:
    exactly ``expected_n`` distinct ``lesion_id`` values, and exactly one row
    (one image) per lesion -- i.e. no ``lesion_id`` or ``image_id`` repeated.

    Does not raise; reports the real counts found so a caller can decide
    pass/fail explicitly rather than trusting an assumption.
    """
    lesion_ids = [r["lesion_id"] for r in manifest_rows]
    image_ids = [r["image_id"] for r in manifest_rows]
    n_rows = len(manifest_rows)
    n_distinct_lesions = len(set(lesion_ids))
    n_distinct_images = len(set(image_ids))

    duplicate_lesion_ids = sorted(
        {lid for lid in lesion_ids if lesion_ids.count(lid) > 1}
    )
    duplicate_image_ids = sorted(
        {iid for iid in image_ids if image_ids.count(iid) > 1}
    )

    return {
        "n_rows": n_rows,
        "n_distinct_lesion_ids": n_distinct_lesions,
        "expected_n": expected_n,
        "lesion_count_matches_expected": n_distinct_lesions == expected_n,
        "one_image_per_lesion": n_rows == n_distinct_lesions,
        "n_distinct_image_ids": n_distinct_images,
        "no_duplicate_image_ids": n_distinct_images == n_rows,
        "duplicate_lesion_ids": duplicate_lesion_ids,
        "duplicate_image_ids": duplicate_image_ids,
        "ok": (
            n_distinct_lesions == expected_n
            and n_rows == n_distinct_lesions
            and n_distinct_images == n_rows
        ),
    }


def verify_quota_reproducibility(
    manifest_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    target_n: int,
    excluded_lesion_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Check criterion (c): recompute the expected per-category quotas from
    the real metadata population (same path the original sampling run used:
    ``check_lesion_population`` -> ``build_lesion_table`` -> (optional)
    ``apply_exclusions`` -> ``compute_quotas_largest_remainder``) and compare
    against the actual per-``dx`` counts found in the manifest.

    Reuses the exact same functions the sampling run itself used -- this is
    not a re-implementation, it is literally the same calculation replayed
    against the same inputs, so a mismatch means the manifest does not match
    what this population/target/exclusion/seed combination would produce.
    Does not attempt to recompute WHICH lesions/images were drawn (that
    depends on the seed, not checked here) -- only the category counts.
    """
    population_check = check_lesion_population(metadata_rows)
    lesion_table = build_lesion_table(metadata_rows)
    if excluded_lesion_ids:
        lesion_table = apply_exclusions(lesion_table, excluded_lesion_ids)["filtered_lesion_table"]

    counts_by_category = {
        cat: sum(1 for v in lesion_table.values() if v["dx"] == cat)
        for cat in ELIGIBLE_DX_CATEGORIES
    }
    expected_quotas = compute_quotas_largest_remainder(counts_by_category, target_n)

    actual_counts: dict[str, int] = {}
    for row in manifest_rows:
        actual_counts[row["dx"]] = actual_counts.get(row["dx"], 0) + 1

    mismatches = {
        cat: {"expected": expected_quotas["final_quotas"].get(cat, 0), "actual": actual_counts.get(cat, 0)}
        for cat in ELIGIBLE_DX_CATEGORIES
        if expected_quotas["final_quotas"].get(cat, 0) != actual_counts.get(cat, 0)
    }

    return {
        "population_check_ok": population_check["ok"],
        "expected_quotas": expected_quotas["final_quotas"],
        "actual_counts_by_category": actual_counts,
        "mismatches": mismatches,
        "ok": population_check["ok"] and not mismatches,
    }


def verify_mask_correspondence(
    manifest_rows: Sequence[Mapping[str, Any]], masks_dir: str | Path
) -> dict[str, Any]:
    """Check criterion (d): every ``image_id`` in the manifest has a
    corresponding ground-truth mask file at
    ``masks_dir / f"{image_id}_segmentation.png"`` -- required to render
    condition S1 (original image + reference-mask contour). Does not assume
    task 2's whole-population pairing result still holds for this specific
    sampled subset; checks each file directly.
    """
    masks_dir = Path(masks_dir)
    missing: list[str] = []
    present: list[str] = []
    for row in manifest_rows:
        image_id = row["image_id"]
        mask_path = masks_dir / f"{image_id}_segmentation.png"
        if mask_path.is_file():
            present.append(image_id)
        else:
            missing.append(image_id)

    return {
        "n_checked": len(manifest_rows),
        "n_with_mask": len(present),
        "n_missing_mask": len(missing),
        "missing_image_ids": sorted(missing),
        "ok": not missing,
    }


def build_sampling_summary_markdown(
    *,
    metadata_csv: str,
    target_n: int,
    seed: int,
    population_check: Mapping[str, Any],
    counts_by_category: Mapping[str, int],
    quotas: Mapping[str, Any],
    exclusions: Mapping[str, Any] | None,
    n_sampled_rows: int,
    n_eligible_total: int,
) -> str:
    """Render the Markdown summary written to ``sampling_summary.md``.

    Pure string formatting -- no file I/O, no sampling logic. Documents the
    Hamilton largest-remainder method inline (calculation table), same
    convention as :mod:`src.data_audit.report` for other audit summaries.
    """
    lines = [
        "# Amostragem HAM10000 — execução (NÃO congela a amostra definitiva)",
        "",
        "Execução de preparação/teste da infraestrutura de amostragem (Tarefa 12). "
        "**Esta execução não congela a amostra definitiva do protocolo** — o "
        "congelamento definitivo depende do fechamento de exclusões aprovadas, "
        "decisão fora do escopo desta tarefa.",
        "",
        f"- Metadados: `{metadata_csv}`",
        f"- Alvo (`--target-n`): {target_n}",
        f"- Semente (`--seed`): {seed}",
        "",
        "## 1. Checagem de integridade da população (lesion_id x dx)",
        "",
        f"- `missing_lesion_id_rows`: {len(population_check['missing_lesion_id_rows'])}",
        f"- `conflicting_lesion_id_to_dx`: {len(population_check['conflicting_lesion_id_to_dx'])}",
        f"- `unexpected_dx_values`: {population_check['unexpected_dx_values']}",
        f"- `ok`: {population_check['ok']}",
        "",
    ]

    if exclusions is not None:
        lines += [
            "## 2. Exclusões aplicadas antes da amostragem",
            "",
            f"- Solicitadas: {exclusions['n_requested']}",
            f"- Efetivamente excluídas (lesion_id presente na população): {exclusions['n_actually_excluded']}",
            f"- IDs desconhecidos (não encontrados na população, não excluídos por não existirem): "
            f"{exclusions['unknown_ids']}",
            "",
        ]
    else:
        lines += [
            "## 2. Exclusões aplicadas antes da amostragem",
            "",
            "Nenhum arquivo de exclusões foi informado (`--exclusions-file` não usado). "
            "Nenhuma exclusão aprovada existe hoje (Tarefa 11 encerrada sem confirmar "
            "nenhuma duplicata).",
            "",
        ]

    lines += [
        "## 3. População elegível por categoria (após exclusões, se houver)",
        "",
        f"Total elegível: **{n_eligible_total}**",
        "",
        "| Categoria | Lesões elegíveis |",
        "|---|---|",
    ]
    for category in sorted(counts_by_category.keys()):
        lines.append(f"| `{category}` | {counts_by_category[category]} |")

    lines += [
        "",
        "## 4. Cotas por categoria — método dos maiores restos (Hamilton)",
        "",
        "Cota exata = `lesões_da_categoria / total_elegível * alvo`. Parte inteira "
        "= piso da cota exata. As `n_extra_units_distribuidas` unidades restantes "
        "(alvo menos a soma das partes inteiras) vão, uma a uma, às categorias de "
        "maior resto (parte fracionária), em ordem decrescente (empate: ordem "
        "alfabética da categoria).",
        "",
        f"Unidades extras distribuídas: {quotas['n_extra_units_distributed']}",
        "",
        "| Categoria | Elegíveis | Cota exata | Parte inteira | Resto | Unidade extra | Cota final |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in quotas["calculation_table"]:
        lines.append(
            f"| `{row['category']}` | {row['n_eligible']} | {row['exact_quota']:.4f} | "
            f"{row['floor']} | {row['remainder']:.4f} | {'sim' if row['extra_unit'] else 'não'} | "
            f"{row['final_quota']} |"
        )

    lines += [
        "",
        f"Soma das cotas finais: {sum(r['final_quota'] for r in quotas['calculation_table'])} "
        f"(deve ser igual ao alvo {target_n}).",
        "",
        "## 5. Resultado da amostragem",
        "",
        f"- Lesões sorteadas (uma imagem cada): {n_sampled_rows}",
        f"- Alvo binário derivado (`binary_target`): `melanoma` se `dx == 'mel'`, "
        "`nao_melanoma` para qualquer outro `dx` -- não é sinônimo de "
        "benigno/maligno.",
        "",
        "## Aviso",
        "",
        "Esta execução é de preparação/teste da infraestrutura, não o congelamento "
        "da amostra definitiva do protocolo experimental.",
        "",
    ]
    return "\n".join(lines)


def build_verification_report_markdown(
    *,
    manifest_path: str,
    uniqueness: Mapping[str, Any],
    quotas: Mapping[str, Any],
    masks: Mapping[str, Any],
    overall_ok: bool,
) -> str:
    """Render the Markdown report for ``scripts/verify_sample_manifest.py``.

    Pure string formatting, same convention as
    :func:`build_sampling_summary_markdown` -- no verification logic here,
    no file I/O. This function only describes a TEMPLATE; its real content
    only exists once the researcher actually runs the verification script
    against a real manifest.
    """
    lines = [
        "# Verificação do Manifesto de Amostra — HAM10000",
        "",
        "**Isto é verificação, não congelamento.** Este relatório só confirma ou "
        "reprova os 4 critérios abaixo contra um `sample_manifest.csv` já gerado "
        "por `scripts/run_ham10000_sampling.py`. Não decide nem executa o "
        "congelamento da amostra definitiva do protocolo.",
        "",
        f"- Manifesto verificado: `{manifest_path}`",
        "",
        "## Resultado geral",
        "",
        f"**{'TODOS OS 4 CRITÉRIOS PASSARAM' if overall_ok else 'AO MENOS UM CRITÉRIO NÃO PASSOU — ver detalhes abaixo'}**",
        "",
        "## 1. Exatamente 3.000 `lesion_id` distintos",
        "",
        f"- Linhas no manifesto: {uniqueness['n_rows']}",
        f"- `lesion_id` distintos: {uniqueness['n_distinct_lesion_ids']} (esperado: {uniqueness['expected_n']})",
        f"- Critério atendido: {uniqueness['lesion_count_matches_expected']}",
        "",
        "## 2. Uma imagem por lesão",
        "",
        f"- Uma linha por `lesion_id` distinto: {uniqueness['one_image_per_lesion']}",
        f"- `image_id` distintos: {uniqueness['n_distinct_image_ids']} (sem duplicata: "
        f"{uniqueness['no_duplicate_image_ids']})",
        f"- `lesion_id` duplicados: {uniqueness['duplicate_lesion_ids']}",
        f"- `image_id` duplicados: {uniqueness['duplicate_image_ids']}",
        "",
        "## 3. Cotas por categoria reproduzíveis",
        "",
        "Cotas recomputadas da população real de metadados (mesma lógica/funções "
        "da amostragem original: `check_lesion_population` -> `build_lesion_table` "
        "-> (exclusões, se houver) -> `compute_quotas_largest_remainder`), "
        "comparadas contra a contagem real por `dx` no manifesto.",
        "",
        f"- Checagem de integridade da população (`lesion_id`/`dx`) ok: {quotas['population_check_ok']}",
        "",
        "| Categoria | Cota esperada | Contagem real no manifesto |",
        "|---|---|---|",
    ]
    for cat in sorted(quotas["expected_quotas"].keys()):
        lines.append(
            f"| `{cat}` | {quotas['expected_quotas'][cat]} | "
            f"{quotas['actual_counts_by_category'].get(cat, 0)} |"
        )
    lines += [
        "",
        f"- Divergências encontradas: {quotas['mismatches'] if quotas['mismatches'] else 'nenhuma'}",
        f"- Critério atendido: {quotas['ok']}",
        "",
        "## 4. Correspondência de máscara para a condição S1",
        "",
        f"- Imagens verificadas: {masks['n_checked']}",
        f"- Com máscara correspondente: {masks['n_with_mask']}",
        f"- Sem máscara (faltando): {masks['n_missing_mask']}",
        f"- `image_id` sem máscara: {masks['missing_image_ids']}",
        f"- Critério atendido: {masks['ok']}",
        "",
        "## Aviso",
        "",
        "Este script verifica um manifesto já gerado; não altera o manifesto, "
        "não decide exclusão, e não congela a amostra definitiva do protocolo.",
        "",
    ]
    return "\n".join(lines)
