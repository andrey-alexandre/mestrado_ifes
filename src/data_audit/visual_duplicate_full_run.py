"""Task 11: full-scale ORB+RANSAC + degeneracy-guard pass over all 421,196
dHash candidates produced by task 7 (``experiments/data_audit/visual_candidates_index.json``),
with NO pHash gate (the researcher's explicit instruction: any pHash filter
would discard pairs before the geometric stage, which is exactly what tasks
8/9 showed is unsafe to do without evidence), incremental checkpointing so
the ~7.6-hour estimated run survives interruption, and a dimensioned (not
exhaustive, not "problem solved") manual-review sample.

Why a separate module from ``geometric_verification``/``geometric_degeneracy``:
those two implement the detection method itself (task 8/9/10); this module
is the full-scale run's own plumbing -- checkpoint read/write/resume and the
review-queue selection/sampling rules (task 11's new scope) -- reusing the
method unchanged, per the researcher's explicit instruction not to alter
``verify_geometric_match``/``classify_geometric_support``/
``classify_with_degeneracy_guard`` or re-derive the degeneracy threshold.

Detection/audit only: no exclusion of any lesion/image is decided or
implemented here. Every one of the 421,196 candidates appears in the
checkpoint regardless of its classification -- the degeneracy guard changes
review PRIORITY, never membership in the full record.
"""
from __future__ import annotations

import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import UnidentifiedImageError

from src.data_audit import geometric_degeneracy as gd
from src.data_audit import geometric_verification as gv
from src.data_audit import visual_duplicate_pilot_v2 as v2

# Reused verbatim from task 10 (docs/reconstruction/07_geometric_validity_guard_report.md,
# section 3): derived entirely from a tune set disjoint from this run, never
# re-derived here per the researcher's explicit instruction.
DEGENERACY_DISPERSION_THRESHOLD = 0.061005

# Same three dHash-distance buckets used in task 9's cross-tabulation
# (src/data_audit/visual_duplicate_pilot_v2.py::DHASH_BUCKETS), spanning
# task 7's full candidate threshold (<=10) exactly -- reused, not redefined,
# so the "fraco" stratified sample lines up with prior tasks' bucket
# language.
DHASH_BUCKETS = v2.DHASH_BUCKETS

STATUS_FORTE = "forte"
STATUS_FORTE_DEGENERADA = "forte_degenerada"
STATUS_FRACO = "fraco"
STATUS_NOT_EVALUABLE = "not_evaluable"
STATUS_IMAGE_READ_ERROR = "image_read_error"

REVIEWABLE_STATUSES = (STATUS_FORTE, STATUS_FORTE_DEGENERADA, STATUS_FRACO, STATUS_NOT_EVALUABLE)

FORTE_REVIEW_CAP = 500
FORTE_DEGENERADA_REVIEW_CAP = 500
FRACO_REVIEW_TARGET = 400
NOT_EVALUABLE_REVIEW_TARGET = 100

# One fixed seed for every sampling decision in this task, documented here
# rather than per call site, so a rerun of the selection step (not the
# ORB/RANSAC pass itself, which is deterministic given the same images) is
# reproducible end to end.
SELECTION_SEED = 20261105

PRESUMED_DISTINCT_JUDGMENT = "presumidamente_distinto"
REVIEW_JUDGMENTS = ("mesma_lesao", "lesoes_diferentes", "indeterminado", PRESUMED_DISTINCT_JUDGMENT)


# ---------------------------------------------------------------------------
# Candidate identity + per-pair processing
# ---------------------------------------------------------------------------


def candidate_key(cand: dict[str, Any]) -> tuple[str, str, str]:
    """Identity of a candidate pair for checkpoint dedup/resume purposes.
    ``(pair, path_a_ham10000, path_b)`` is unique within the 421,196-row
    index (each row is a distinct HAM10000-vs-other edge within one of the
    two dataset pairs)."""
    return (cand["pair"], cand["path_a_ham10000"], cand["path_b"])


def process_candidate(
    reader: Any,
    cand: dict[str, Any],
    dispersion_threshold: float = DEGENERACY_DISPERSION_THRESHOLD,
) -> dict[str, Any]:
    """Run the unmodified task 8/9/10 method (ORB+RANSAC, then the
    degeneracy guard) on one candidate pair, with NO pHash gate -- the
    candidate reaches this function purely because it was in task 7's dHash
    candidate index, nothing upstream filters it further.

    A read/decode failure on either side is recorded as its own
    ``"image_read_error"`` status (own explicit label, distinct from
    ``"not_evaluable"``, which means "read fine but ORB had too little to
    work with") and does NOT raise -- the caller's batch loop must not abort
    on one bad file.
    """
    base_row = {
        "pair": cand["pair"],
        "path_a_ham10000": cand["path_a_ham10000"],
        "path_b": cand["path_b"],
        "distance": cand["distance"],
    }
    try:
        data_a = reader.read_ham10000(cand["path_a_ham10000"])
        data_b = reader.read_other(cand["path_b"])
        array_a = v2.decode_grayscale_array(data_a)
        array_b = v2.decode_grayscale_array(data_b)
    except (OSError, zipfile.BadZipFile, UnidentifiedImageError) as exc:
        return {
            **base_row,
            "status": STATUS_IMAGE_READ_ERROR,
            "n_keypoints_a": None,
            "n_keypoints_b": None,
            "n_matches": None,
            "n_inliers": None,
            "inlier_ratio": None,
            "dispersion_a": None,
            "dispersion_b": None,
            "base_strength": STATUS_IMAGE_READ_ERROR,
            "final_strength": STATUS_IMAGE_READ_ERROR,
            "error": str(exc),
        }

    result = gv.verify_geometric_match(array_a, array_b)
    guard = gd.classify_with_degeneracy_guard(result, array_a.shape, array_b.shape, dispersion_threshold)
    return {
        **base_row,
        "status": result["status"],
        "n_keypoints_a": result.get("n_keypoints_a"),
        "n_keypoints_b": result.get("n_keypoints_b"),
        "n_matches": result.get("n_matches"),
        "n_inliers": result.get("n_inliers"),
        "inlier_ratio": result.get("inlier_ratio"),
        "dispersion_a": guard["dispersion_a"],
        "dispersion_b": guard["dispersion_b"],
        "base_strength": guard["strength"],
        "final_strength": guard["final_strength"],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Checkpoint: JSON-Lines, append-only, resumable
# ---------------------------------------------------------------------------
#
# Format: one JSON object per line, in the exact shape ``process_candidate``
# returns. Chosen over one big JSON array because (a) it supports O(1)
# incremental append with no need to ever rewrite earlier rows, and (b) a
# process killed mid-write can corrupt at most the single last line, which
# ``load_checkpoint`` detects and skips rather than failing the whole load.
# To resume manually: just rerun the same orchestration script with the same
# ``--checkpoint`` path; it loads this file, skips every key already
# present, and keeps appending from where it left off. No separate "resume"
# flag or flag-day marker is needed -- presence of the file is the signal.


def load_checkpoint(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
    """Load all valid rows already written to ``path``, keyed by
    :func:`candidate_key`. Returns ``(rows_by_key, n_skipped_malformed)`` --
    a malformed line (JSON decode failure, e.g. a half-flushed last line from
    a killed process) is counted and skipped, never raised, so a resume
    never fails outright because of exactly the interruption it exists to
    tolerate.
    """
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    n_skipped = 0
    if not path.is_file():
        return rows, n_skipped
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            rows[(row["pair"], row["path_a_ham10000"], row["path_b"])] = row
    return rows, n_skipped


def pending_candidates(
    all_candidates: list[dict[str, Any]], done: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Candidates from the full 421,196-row index not yet present in the
    checkpoint -- the exact set a resumed run must still process."""
    return [cand for cand in all_candidates if candidate_key(cand) not in done]


def run_full_pass(
    candidates: list[dict[str, Any]],
    checkpoint_path: Path,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
    progress_every: int = 1000,
    fsync_every: int = 500,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Resumable batch driver: skip candidates already in the checkpoint,
    process every remaining one via ``process_fn``, appending+flushing each
    result immediately so an interruption loses at most the one row being
    computed when it happens, never any already-written row.

    ``process_fn`` takes just the candidate dict (the caller binds the image
    reader/threshold via closure) so this function stays testable with a
    trivial synthetic stub, independent of real image I/O.
    """
    import os
    import time

    done, n_skipped_malformed = load_checkpoint(checkpoint_path)
    if n_skipped_malformed:
        log(f"aviso: {n_skipped_malformed} linha(s) malformada(s) no checkpoint foram ignoradas")
    todo = pending_candidates(candidates, done)
    log(
        f"checkpoint: {len(done)} já processados, {len(todo)} pendentes de {len(candidates)} no total "
        f"({checkpoint_path})"
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    n_processed_this_run = 0
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for cand in todo:
            row = process_fn(cand)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_processed_this_run += 1
            if n_processed_this_run % fsync_every == 0:
                os.fsync(f.fileno())
            if n_processed_this_run % progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = n_processed_this_run / elapsed if elapsed > 0 else 0.0
                remaining = len(todo) - n_processed_this_run
                eta_s = remaining / rate if rate > 0 else float("inf")
                log(
                    f"progresso: {n_processed_this_run}/{len(todo)} nesta execução "
                    f"({len(done) + n_processed_this_run}/{len(candidates)} no total) "
                    f"taxa={rate:.2f}/s eta={eta_s / 3600:.2f}h"
                )
        os.fsync(f.fileno())

    elapsed = time.perf_counter() - start
    return {
        "n_total": len(candidates),
        "n_already_done_at_start": len(done),
        "n_processed_this_run": n_processed_this_run,
        "n_skipped_malformed_checkpoint_lines": n_skipped_malformed,
        "elapsed_seconds_this_run": elapsed,
        "complete": (len(done) + n_processed_this_run) == len(candidates),
    }


# ---------------------------------------------------------------------------
# Population summary (full 421,196, before any sampling)
# ---------------------------------------------------------------------------


def summarize_population(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Exact counts by ``final_strength`` across the complete checkpoint --
    the "composição completa da população" the closure report must state
    before any cap/sample is applied."""
    rows = list(rows)
    counts = Counter(r["final_strength"] for r in rows)
    by_pair: dict[str, Counter] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], Counter())[r["final_strength"]] += 1
    return {
        "n_total": len(rows),
        "counts_by_final_strength": dict(counts),
        "counts_by_pair_and_final_strength": {pair: dict(c) for pair, c in by_pair.items()},
    }


# ---------------------------------------------------------------------------
# Review-queue selection (dimensioned sampling rules, not fixed numbers)
# ---------------------------------------------------------------------------


def select_capped(
    rows: list[dict[str, Any]], cap: int, seed: int = SELECTION_SEED
) -> dict[str, Any]:
    """100% of ``rows`` up to ``cap``; a fixed-seed random sample of exactly
    ``cap`` rows if ``rows`` exceeds it. The real total is always reported
    alongside the (possibly capped) selection -- never silently lost."""
    n_total = len(rows)
    if n_total <= cap:
        return {"n_total": n_total, "capped": False, "seed": None, "selected": list(rows)}
    rng = random.Random(seed)
    selected = rng.sample(rows, cap)
    return {"n_total": n_total, "capped": True, "seed": seed, "selected": selected}


def _dhash_bucket_label(distance: int) -> str | None:
    for label, lo, hi in DHASH_BUCKETS:
        if lo <= distance <= hi:
            return label
    return None


def select_fraco_stratified(
    rows: list[dict[str, Any]], target_n: int = FRACO_REVIEW_TARGET, seed: int = SELECTION_SEED
) -> dict[str, Any]:
    """Stratified sample of ``"fraco"`` rows across the same dHash-distance
    buckets used in tasks 8/9, proportional to each bucket's share of the
    FULL "fraco" population (not of the sample), rounded to the nearest
    integer per bucket. If a bucket's population is smaller than its
    proportional share, every row in that bucket is taken (no fabricated
    rows) and the shortfall is reported explicitly rather than silently
    redistributed to other buckets -- the exact realized total may therefore
    land a little under ``target_n``; the real, documented number is what
    matters, not hitting the target exactly.
    """
    n_total = len(rows)
    by_bucket: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in DHASH_BUCKETS}
    for r in rows:
        label = _dhash_bucket_label(r["distance"])
        if label is not None:
            by_bucket[label].append(r)

    rng = random.Random(seed)
    bucket_results = []
    selected: list[dict[str, Any]] = []
    for label, _, _ in DHASH_BUCKETS:
        bucket_rows = by_bucket[label]
        bucket_n = len(bucket_rows)
        proportion = (bucket_n / n_total) if n_total else 0.0
        target_for_bucket = round(proportion * target_n)
        n_take = min(target_for_bucket, bucket_n)
        bucket_selected = rng.sample(bucket_rows, n_take) if n_take < bucket_n else list(bucket_rows)
        bucket_results.append(
            {
                "bucket": label,
                "n_total_in_population": bucket_n,
                "proportion_of_population": proportion,
                "target_for_bucket": target_for_bucket,
                "n_selected": len(bucket_selected),
            }
        )
        selected.extend(bucket_selected)

    return {
        "n_total": n_total,
        "target_n": target_n,
        "seed": seed,
        "buckets": bucket_results,
        "n_selected": len(selected),
        "selected": selected,
    }


def select_not_evaluable_sample(
    rows: list[dict[str, Any]], target_n: int = NOT_EVALUABLE_REVIEW_TARGET, seed: int = SELECTION_SEED
) -> dict[str, Any]:
    """Fixed-seed random sample of up to ``target_n`` ``"not_evaluable"``
    rows (all of them if fewer exist)."""
    n_total = len(rows)
    if n_total <= target_n:
        return {"n_total": n_total, "seed": None, "selected": list(rows)}
    rng = random.Random(seed)
    return {"n_total": n_total, "seed": seed, "selected": rng.sample(rows, target_n)}


def build_review_queue(rows: list[dict[str, Any]], seed: int = SELECTION_SEED) -> dict[str, Any]:
    """Apply every selection rule to the full checkpoint and return a report
    that (a) states the real population size of each of the four reviewable
    categories before any cap/sample, (b) the exact selection made per rule,
    and (c) every row from ``rows`` exactly once, tagged
    ``selected_for_review`` (bool) and, for rows NOT selected,
    ``review_judgment`` pre-filled to
    :data:`PRESUMED_DISTINCT_JUDGMENT` per the mandatory convention: a pair
    whose image was never opened is never called "confirmed distinct",
    only "presumed distinct". Rows outside the four reviewable statuses
    (currently only ``"image_read_error"``) are never selected and are
    tagged the same way -- there is no image to review for them either.
    """
    by_status: dict[str, list[dict[str, Any]]] = {status: [] for status in REVIEWABLE_STATUSES}
    other_rows: list[dict[str, Any]] = []
    for r in rows:
        if r["final_strength"] in by_status:
            by_status[r["final_strength"]].append(r)
        else:
            other_rows.append(r)

    forte_selection = select_capped(by_status[STATUS_FORTE], FORTE_REVIEW_CAP, seed)
    forte_degenerada_selection = select_capped(by_status[STATUS_FORTE_DEGENERADA], FORTE_DEGENERADA_REVIEW_CAP, seed)
    fraco_selection = select_fraco_stratified(by_status[STATUS_FRACO], FRACO_REVIEW_TARGET, seed)
    not_evaluable_selection = select_not_evaluable_sample(by_status[STATUS_NOT_EVALUABLE], NOT_EVALUABLE_REVIEW_TARGET, seed)

    selected_keys = {
        candidate_key(r)
        for r in (
            forte_selection["selected"]
            + forte_degenerada_selection["selected"]
            + fraco_selection["selected"]
            + not_evaluable_selection["selected"]
        )
    }

    full_output: list[dict[str, Any]] = []
    for r in rows:
        key = candidate_key(r)
        is_selected = key in selected_keys
        out_row = {
            **r,
            "selected_for_review": is_selected,
            "review_judgment": None if is_selected else PRESUMED_DISTINCT_JUDGMENT,
            "review_justification": None if is_selected else "fora da amostra de revisão dimensionada (Tarefa 11)",
        }
        full_output.append(out_row)

    return {
        "seed": seed,
        "population_by_status": {status: len(by_status[status]) for status in REVIEWABLE_STATUSES},
        "n_other_status_rows": len(other_rows),
        "forte": forte_selection,
        "forte_degenerada": forte_degenerada_selection,
        "fraco": fraco_selection,
        "not_evaluable": not_evaluable_selection,
        "n_selected_total": len(selected_keys),
        "n_total_rows": len(rows),
        "rows": full_output,
    }
