"""Stage 6 — stratified, PAIRED bootstrap over the frozen lesion sample.

"Stratified": each replica resamples WITH replacement, independently per
category (``dx``), exactly as many lesion_ids as that category has in the
original manifest -- so every replica preserves the original per-category
quotas exactly, not just approximately.

"Paired": a single replica (one concrete list of resampled lesion_ids, with
repetition) is generated ONCE and then reused, unchanged, to recompute the
indicators of EVERY system/condition being compared for that replica --
never resampled independently per system. This is what makes a bootstrap
comparison between two systems/conditions valid (same cases, same category
composition, every time) rather than a comparison between two independently
noisy resamples.

Pure functions, deterministic given a seed -- no I/O, no real data read
here. Verification with fabricated data lives in
``scripts/verify_stage6_evaluator.py``; unit tests in
``tests/evaluation/test_stage6_bootstrap.py``.
"""
from __future__ import annotations

import random
from collections import Counter


def generate_stratified_bootstrap_replica(
    lesion_ids_by_category: dict[str, list[str]], rng: random.Random
) -> list[str]:
    """One replica: for each category, draw WITH replacement exactly as
    many lesion_ids as that category has (quota preserved exactly), using
    the given, already-seeded ``rng``. Concatenates all categories in
    sorted-category order (deterministic given ``rng``'s state)."""
    replica: list[str] = []
    for category in sorted(lesion_ids_by_category):
        pool = lesion_ids_by_category[category]
        replica.extend(rng.choices(pool, k=len(pool)))
    return replica


def generate_bootstrap_replicas(
    lesion_ids_by_category: dict[str, list[str]], n_replicas: int, seed: int
) -> list[list[str]]:
    """``n_replicas`` independent replicas (independent from EACH OTHER;
    each one individually is reused across all systems/conditions by the
    caller -- see module docstring). Deterministic given ``seed``."""
    rng = random.Random(seed)
    return [
        generate_stratified_bootstrap_replica(lesion_ids_by_category, rng)
        for _ in range(n_replicas)
    ]


def replica_category_counts(
    replica: list[str], lesion_id_to_category: dict[str, str]
) -> Counter:
    """Counts how many resampled ids in ``replica`` belong to each category
    -- used to verify quota preservation (should equal the original
    manifest's per-category counts exactly, for every replica)."""
    return Counter(lesion_id_to_category[lesion_id] for lesion_id in replica)
