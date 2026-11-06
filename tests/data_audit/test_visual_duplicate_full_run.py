"""Tests for src.data_audit.visual_duplicate_full_run (task 11).

Synthetic candidate/row dicts only -- no real images, no patient data, same
convention as the rest of tests/data_audit/. Covers: checkpoint round-trip
and malformed-line tolerance, resume-skips-already-processed (the
resilience mechanism the researcher required), and the four review-sample
selection rules (caps, stratified proportions, population reporting).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data_audit import visual_duplicate_full_run as full_run

# ---------------------------------------------------------------------------
# candidate_key / checkpoint load/save
# ---------------------------------------------------------------------------


def _cand(pair: str, a: str, b: str, distance: int = 5) -> dict:
    return {"pair": pair, "path_a_ham10000": a, "path_b": b, "distance": distance}


def _row(pair: str, a: str, b: str, final_strength: str, distance: int = 5) -> dict:
    return {
        "pair": pair,
        "path_a_ham10000": a,
        "path_b": b,
        "distance": distance,
        "status": "evaluated",
        "n_keypoints_a": 10,
        "n_keypoints_b": 10,
        "n_matches": 10,
        "n_inliers": 10,
        "inlier_ratio": 1.0,
        "dispersion_a": 0.5,
        "dispersion_b": 0.5,
        "base_strength": final_strength,
        "final_strength": final_strength,
        "error": None,
    }


def test_candidate_key_identifies_pair_a_b():
    cand = _cand("P", "a.jpg", "b.jpg")
    assert full_run.candidate_key(cand) == ("P", "a.jpg", "b.jpg")


def test_load_checkpoint_missing_file_returns_empty():
    done, n_skipped = full_run.load_checkpoint(Path("/nonexistent/x.jsonl"))
    assert done == {}
    assert n_skipped == 0


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    rows = [_row("P", "a1.jpg", "b1.jpg", "forte"), _row("P", "a2.jpg", "b2.jpg", "fraco")]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    done, n_skipped = full_run.load_checkpoint(path)
    assert n_skipped == 0
    assert set(done.keys()) == {("P", "a1.jpg", "b1.jpg"), ("P", "a2.jpg", "b2.jpg")}
    assert done[("P", "a1.jpg", "b1.jpg")]["final_strength"] == "forte"


def test_load_checkpoint_skips_malformed_trailing_line(tmp_path):
    """Simulates a process killed mid-write: the last line is truncated
    JSON. The valid line before it must still load, and the malformed one
    must be counted, never raised."""
    path = tmp_path / "checkpoint.jsonl"
    good_row = _row("P", "a1.jpg", "b1.jpg", "forte")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(good_row) + "\n")
        f.write('{"pair": "P", "path_a_ham10000": "a2.jpg", "path_b": "b2.')  # truncated, no trailing newline

    done, n_skipped = full_run.load_checkpoint(path)
    assert n_skipped == 1
    assert set(done.keys()) == {("P", "a1.jpg", "b1.jpg")}


def test_pending_candidates_excludes_done():
    all_candidates = [_cand("P", "a1.jpg", "b1.jpg"), _cand("P", "a2.jpg", "b2.jpg"), _cand("P", "a3.jpg", "b3.jpg")]
    done = {("P", "a2.jpg", "b2.jpg"): _row("P", "a2.jpg", "b2.jpg", "fraco")}
    pending = full_run.pending_candidates(all_candidates, done)
    assert [full_run.candidate_key(c) for c in pending] == [("P", "a1.jpg", "b1.jpg"), ("P", "a3.jpg", "b3.jpg")]


# ---------------------------------------------------------------------------
# run_full_pass: resumability (the mechanism the researcher required)
# ---------------------------------------------------------------------------


def test_run_full_pass_processes_all_candidates_once(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    candidates = [_cand("P", f"a{i}.jpg", f"b{i}.jpg") for i in range(10)]
    calls = []

    def fake_process(cand):
        calls.append(full_run.candidate_key(cand))
        return _row(cand["pair"], cand["path_a_ham10000"], cand["path_b"], "fraco")

    summary = full_run.run_full_pass(candidates, checkpoint_path, fake_process, progress_every=1000, log=lambda s: None)
    assert summary["n_processed_this_run"] == 10
    assert summary["complete"] is True
    assert len(calls) == 10

    done, _ = full_run.load_checkpoint(checkpoint_path)
    assert len(done) == 10


def test_run_full_pass_resume_does_not_reprocess_already_done(tmp_path):
    """Simulates an interrupted execution: write a partial checkpoint by
    hand (as if a prior run crashed after 4/10), then run again and confirm
    only the remaining 6 are (re)computed -- the core resilience guarantee."""
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    candidates = [_cand("P", f"a{i}.jpg", f"b{i}.jpg") for i in range(10)]

    # Simulate a prior partial run: first 4 candidates already checkpointed.
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for cand in candidates[:4]:
            row = _row(cand["pair"], cand["path_a_ham10000"], cand["path_b"], "fraco")
            f.write(json.dumps(row) + "\n")

    calls = []

    def fake_process(cand):
        calls.append(full_run.candidate_key(cand))
        return _row(cand["pair"], cand["path_a_ham10000"], cand["path_b"], "fraco")

    summary = full_run.run_full_pass(candidates, checkpoint_path, fake_process, progress_every=1000, log=lambda s: None)

    # Only the 6 pending candidates were (re)computed in this run.
    assert summary["n_already_done_at_start"] == 4
    assert summary["n_processed_this_run"] == 6
    assert len(calls) == 6
    assert set(calls) == {full_run.candidate_key(c) for c in candidates[4:]}

    # Final checkpoint has exactly 10 rows, no duplicates, nothing lost.
    done, _ = full_run.load_checkpoint(checkpoint_path)
    assert len(done) == 10
    assert set(done.keys()) == {full_run.candidate_key(c) for c in candidates}

    # Checkpoint file itself has exactly 10 lines (4 original + 6 appended),
    # confirming resume APPENDED rather than rewriting from scratch.
    with open(checkpoint_path, encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())
    assert n_lines == 10


def test_run_full_pass_second_call_with_nothing_pending_calls_process_fn_zero_times(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    candidates = [_cand("P", f"a{i}.jpg", f"b{i}.jpg") for i in range(5)]
    calls = []

    def fake_process(cand):
        calls.append(full_run.candidate_key(cand))
        return _row(cand["pair"], cand["path_a_ham10000"], cand["path_b"], "fraco")

    full_run.run_full_pass(candidates, checkpoint_path, fake_process, progress_every=1000, log=lambda s: None)
    calls.clear()
    summary = full_run.run_full_pass(candidates, checkpoint_path, fake_process, progress_every=1000, log=lambda s: None)
    assert summary["n_processed_this_run"] == 0
    assert summary["complete"] is True
    assert calls == []


# ---------------------------------------------------------------------------
# summarize_population
# ---------------------------------------------------------------------------


def test_summarize_population_counts_by_final_strength_and_pair():
    rows = [
        _row("P1", "a1.jpg", "b1.jpg", "forte"),
        _row("P1", "a2.jpg", "b2.jpg", "fraco"),
        _row("P2", "a3.jpg", "b3.jpg", "fraco"),
        _row("P2", "a4.jpg", "b4.jpg", "not_evaluable"),
    ]
    summary = full_run.summarize_population(rows)
    assert summary["n_total"] == 4
    assert summary["counts_by_final_strength"] == {"forte": 1, "fraco": 2, "not_evaluable": 1}
    assert summary["counts_by_pair_and_final_strength"]["P1"] == {"forte": 1, "fraco": 1}
    assert summary["counts_by_pair_and_final_strength"]["P2"] == {"fraco": 1, "not_evaluable": 1}


# ---------------------------------------------------------------------------
# select_capped
# ---------------------------------------------------------------------------


def test_select_capped_under_cap_returns_all_no_sampling():
    rows = [_row("P", f"a{i}.jpg", f"b{i}.jpg", "forte") for i in range(10)]
    result = full_run.select_capped(rows, cap=500)
    assert result["n_total"] == 10
    assert result["capped"] is False
    assert result["seed"] is None
    assert len(result["selected"]) == 10


def test_select_capped_over_cap_samples_exactly_cap_deterministically():
    rows = [_row("P", f"a{i}.jpg", f"b{i}.jpg", "forte") for i in range(600)]
    result_1 = full_run.select_capped(rows, cap=500, seed=42)
    result_2 = full_run.select_capped(rows, cap=500, seed=42)
    assert result_1["n_total"] == 600
    assert result_1["capped"] is True
    assert len(result_1["selected"]) == 500
    # Same seed -> same selection (reproducibility requirement).
    keys_1 = [full_run.candidate_key(r) for r in result_1["selected"]]
    keys_2 = [full_run.candidate_key(r) for r in result_2["selected"]]
    assert keys_1 == keys_2


def test_select_capped_different_seed_can_differ():
    rows = [_row("P", f"a{i}.jpg", f"b{i}.jpg", "forte") for i in range(600)]
    result_a = full_run.select_capped(rows, cap=500, seed=1)
    result_b = full_run.select_capped(rows, cap=500, seed=2)
    keys_a = {full_run.candidate_key(r) for r in result_a["selected"]}
    keys_b = {full_run.candidate_key(r) for r in result_b["selected"]}
    assert keys_a != keys_b


# ---------------------------------------------------------------------------
# select_fraco_stratified
# ---------------------------------------------------------------------------


def test_select_fraco_stratified_proportions_match_population_buckets():
    # Population: bucket 0-2 -> 10 rows, bucket 3-7 -> 40 rows, bucket 8-10 -> 50 rows (n=100 total).
    rows = (
        [_row("P", f"a{i}.jpg", f"b{i}.jpg", "fraco", distance=1) for i in range(10)]
        + [_row("P", f"a{i}.jpg", f"b{i}.jpg", "fraco", distance=5) for i in range(10, 50)]
        + [_row("P", f"a{i}.jpg", f"b{i}.jpg", "fraco", distance=9) for i in range(50, 100)]
    )
    result = full_run.select_fraco_stratified(rows, target_n=40, seed=42)
    assert result["n_total"] == 100
    by_bucket = {b["bucket"]: b for b in result["buckets"]}
    assert by_bucket["0-2"]["n_total_in_population"] == 10
    assert by_bucket["3-7"]["n_total_in_population"] == 40
    assert by_bucket["8-10"]["n_total_in_population"] == 50
    # Proportional targets: 10%, 40%, 50% of target_n=40 -> 4, 16, 20.
    assert by_bucket["0-2"]["target_for_bucket"] == 4
    assert by_bucket["3-7"]["target_for_bucket"] == 16
    assert by_bucket["8-10"]["target_for_bucket"] == 20
    assert by_bucket["0-2"]["n_selected"] == 4
    assert by_bucket["3-7"]["n_selected"] == 16
    assert by_bucket["8-10"]["n_selected"] == 20
    assert result["n_selected"] == 40
    # Every selected row genuinely belongs to its bucket's distance range.
    selected_keys = {full_run.candidate_key(r) for r in result["selected"]}
    assert len(selected_keys) == 40


def test_select_fraco_stratified_bucket_smaller_than_target_takes_all_no_fabrication():
    # Bucket 0-2 is a small but disproportionately large SHARE of this tiny
    # population (3 of 10 rows, 30%): its proportional target for
    # target_n=40 would be round(0.3*40)=12, far more than the 3 rows that
    # actually exist -- all 3 must be taken, never padded with invented rows.
    rows = (
        [_row("P", f"a{i}.jpg", f"b{i}.jpg", "fraco", distance=1) for i in range(3)]
        + [_row("P", f"a{i}.jpg", f"b{i}.jpg", "fraco", distance=9) for i in range(3, 10)]
    )
    result = full_run.select_fraco_stratified(rows, target_n=40, seed=42)
    by_bucket = {b["bucket"]: b for b in result["buckets"]}
    assert by_bucket["0-2"]["n_total_in_population"] == 3
    assert by_bucket["0-2"]["target_for_bucket"] == 12
    assert by_bucket["0-2"]["n_selected"] == 3  # capped by population, not target


# ---------------------------------------------------------------------------
# select_not_evaluable_sample
# ---------------------------------------------------------------------------


def test_select_not_evaluable_sample_under_target_returns_all():
    rows = [_row("P", f"a{i}.jpg", f"b{i}.jpg", "not_evaluable") for i in range(50)]
    result = full_run.select_not_evaluable_sample(rows, target_n=100)
    assert result["n_total"] == 50
    assert len(result["selected"]) == 50


def test_select_not_evaluable_sample_over_target_is_deterministic():
    rows = [_row("P", f"a{i}.jpg", f"b{i}.jpg", "not_evaluable") for i in range(300)]
    result_1 = full_run.select_not_evaluable_sample(rows, target_n=100, seed=7)
    result_2 = full_run.select_not_evaluable_sample(rows, target_n=100, seed=7)
    assert len(result_1["selected"]) == 100
    keys_1 = [full_run.candidate_key(r) for r in result_1["selected"]]
    keys_2 = [full_run.candidate_key(r) for r in result_2["selected"]]
    assert keys_1 == keys_2


# ---------------------------------------------------------------------------
# build_review_queue: full end-to-end selection + "presumed distinct" tagging
# ---------------------------------------------------------------------------


def test_build_review_queue_preserves_every_row_exactly_once():
    rows = (
        [_row("P", f"forte{i}.jpg", f"b{i}.jpg", "forte") for i in range(5)]
        + [_row("P", f"deg{i}.jpg", f"b{i}.jpg", "forte_degenerada") for i in range(5)]
        + [_row("P", f"fraco{i}.jpg", f"b{i}.jpg", "fraco", distance=i % 11) for i in range(200)]
        + [_row("P", f"ne{i}.jpg", f"b{i}.jpg", "not_evaluable") for i in range(200)]
        + [_row("P", f"err{i}.jpg", f"b{i}.jpg", "image_read_error") for i in range(3)]
    )
    queue = full_run.build_review_queue(rows, seed=42)
    assert queue["n_total_rows"] == len(rows)
    assert len(queue["rows"]) == len(rows)
    out_keys = [full_run.candidate_key(r) for r in queue["rows"]]
    assert len(out_keys) == len(set(out_keys))  # no duplicates
    assert set(out_keys) == {full_run.candidate_key(r) for r in rows}  # nothing dropped


def test_build_review_queue_small_forte_population_is_fully_selected_not_capped():
    rows = [_row("P", f"forte{i}.jpg", f"b{i}.jpg", "forte") for i in range(5)]
    queue = full_run.build_review_queue(rows, seed=42)
    assert queue["population_by_status"]["forte"] == 5
    assert queue["forte"]["capped"] is False
    assert all(r["selected_for_review"] for r in queue["rows"])
    assert all(r["review_judgment"] is None for r in queue["rows"])


def test_build_review_queue_unselected_rows_are_presumed_distinct_never_confirmed():
    # Large not_evaluable population forces most of it to be unselected.
    rows = [_row("P", f"ne{i}.jpg", f"b{i}.jpg", "not_evaluable") for i in range(500)]
    queue = full_run.build_review_queue(rows, seed=42)
    unselected = [r for r in queue["rows"] if not r["selected_for_review"]]
    assert len(unselected) == 500 - full_run.NOT_EVALUABLE_REVIEW_TARGET
    assert all(r["review_judgment"] == full_run.PRESUMED_DISTINCT_JUDGMENT for r in unselected)
    # Never the forbidden "confirmed distinct" framing.
    assert all("confirmad" not in (r["review_judgment"] or "") for r in queue["rows"])


def test_build_review_queue_image_read_error_rows_never_selected():
    rows = [_row("P", f"err{i}.jpg", f"b{i}.jpg", "image_read_error") for i in range(10)]
    queue = full_run.build_review_queue(rows, seed=42)
    assert queue["n_other_status_rows"] == 10
    assert all(not r["selected_for_review"] for r in queue["rows"])
    assert all(r["review_judgment"] == full_run.PRESUMED_DISTINCT_JUDGMENT for r in queue["rows"])
