import pytest

from src.data_audit import sampling


def _fabricated_rows():
    """50 fictitious lesions across 4 categories, each with 1-2 images,
    designed so the proportional quota math is NOT exact for target_n=10
    (forces the largest-remainder path to actually distribute extra units).

    Counts: nv=28, mel=12, bcc=6, bkl=4 (total 50).
    """
    rows = []
    counts = {"nv": 28, "mel": 12, "bcc": 6, "bkl": 4}
    lesion_n = 0
    for dx, n in counts.items():
        for i in range(n):
            lesion_n += 1
            lesion_id = f"HAM_{lesion_n:04d}"
            rows.append({"lesion_id": lesion_id, "image_id": f"ISIC_{lesion_n:04d}_a", "dx": dx})
            if i % 3 == 0:  # some lesions get a second image
                rows.append({"lesion_id": lesion_id, "image_id": f"ISIC_{lesion_n:04d}_b", "dx": dx})
    return rows


def test_check_lesion_population_ok_on_clean_data():
    result = sampling.check_lesion_population(_fabricated_rows())
    assert result["ok"] is True
    assert result["missing_lesion_id_rows"] == []
    assert result["conflicting_lesion_id_to_dx"] == {}
    assert result["unexpected_dx_values"] == []


def test_check_lesion_population_flags_missing_lesion_id():
    rows = _fabricated_rows()
    rows.append({"lesion_id": "", "image_id": "ISIC_9999", "dx": "nv"})
    result = sampling.check_lesion_population(rows)
    assert result["ok"] is False
    assert len(result["missing_lesion_id_rows"]) == 1


def test_check_lesion_population_flags_conflicting_dx():
    rows = _fabricated_rows()
    rows.append({"lesion_id": "HAM_0001", "image_id": "ISIC_dup", "dx": "mel"})  # HAM_0001 is nv
    result = sampling.check_lesion_population(rows)
    assert result["ok"] is False
    assert "HAM_0001" in result["conflicting_lesion_id_to_dx"]


def test_check_lesion_population_flags_unexpected_dx_value():
    rows = _fabricated_rows()
    rows.append({"lesion_id": "HAM_9001", "image_id": "ISIC_9001", "dx": "not_a_real_category"})
    result = sampling.check_lesion_population(rows)
    assert result["ok"] is False
    assert result["unexpected_dx_values"] == ["not_a_real_category"]


def test_build_lesion_table_collapses_images_per_lesion():
    table = sampling.build_lesion_table(_fabricated_rows())
    assert len(table) == 50
    multi_image_lesions = [lid for lid, v in table.items() if len(v["image_ids"]) > 1]
    assert len(multi_image_lesions) > 0
    for lid in multi_image_lesions:
        assert len(set(table[lid]["image_ids"])) == len(table[lid]["image_ids"])


def test_apply_exclusions_removes_known_and_reports_unknown():
    table = sampling.build_lesion_table(_fabricated_rows())
    result = sampling.apply_exclusions(table, ["HAM_0001", "HAM_0002", "does_not_exist"])
    assert result["n_requested"] == 3
    assert result["n_actually_excluded"] == 2
    assert result["unknown_ids"] == ["does_not_exist"]
    assert "HAM_0001" not in result["filtered_lesion_table"]
    assert "HAM_0002" not in result["filtered_lesion_table"]
    assert len(result["filtered_lesion_table"]) == 48


def test_apply_exclusions_empty_list_is_a_no_op():
    table = sampling.build_lesion_table(_fabricated_rows())
    result = sampling.apply_exclusions(table, [])
    assert result["n_requested"] == 0
    assert result["n_actually_excluded"] == 0
    assert result["filtered_lesion_table"] == table


def test_compute_quotas_largest_remainder_sums_exactly_to_target_inexact_case():
    # 50 total, target 10: exact quotas nv=5.6, mel=2.4, bcc=1.2, bkl=0.8
    counts = {"nv": 28, "mel": 12, "bcc": 6, "bkl": 4}
    result = sampling.compute_quotas_largest_remainder(counts, target_n=10)
    assert sum(result["final_quotas"].values()) == 10
    # floors: nv=5, mel=2, bcc=1, bkl=0 -> sum=8, 2 extra units to largest remainders
    # remainders: nv=0.6, mel=0.4, bcc=0.2, bkl=0.8 -> top 2 are bkl(0.8), nv(0.6)
    assert result["final_quotas"]["bkl"] == 1
    assert result["final_quotas"]["nv"] == 6
    assert result["final_quotas"]["mel"] == 2
    assert result["final_quotas"]["bcc"] == 1
    assert result["n_extra_units_distributed"] == 2


def test_compute_quotas_largest_remainder_exact_division_needs_no_extra_units():
    # 100 total, target 10, evenly divisible by construction
    counts = {"a": 50, "b": 30, "c": 20}
    result = sampling.compute_quotas_largest_remainder(counts, target_n=10)
    assert result["final_quotas"] == {"a": 5, "b": 3, "c": 2}
    assert result["n_extra_units_distributed"] == 0
    assert sum(result["final_quotas"].values()) == 10


def test_compute_quotas_raises_when_target_exceeds_population():
    counts = {"nv": 28, "mel": 12, "bcc": 6, "bkl": 4}
    with pytest.raises(ValueError):
        sampling.compute_quotas_largest_remainder(counts, target_n=51)


def test_compute_quotas_calculation_table_has_one_row_per_category_with_expected_fields():
    counts = {"nv": 28, "mel": 12, "bcc": 6, "bkl": 4}
    result = sampling.compute_quotas_largest_remainder(counts, target_n=10)
    categories_in_table = {row["category"] for row in result["calculation_table"]}
    assert categories_in_table == set(counts.keys())
    for row in result["calculation_table"]:
        assert set(row.keys()) == {
            "category", "n_eligible", "exact_quota", "floor", "remainder", "extra_unit", "final_quota",
        }


def test_sample_lesions_respects_quotas_and_is_deterministic():
    table = sampling.build_lesion_table(_fabricated_rows())
    quotas = {"nv": 6, "mel": 2, "bcc": 1, "bkl": 1}

    result_a = sampling.sample_lesions(table, quotas, seed=123)
    result_b = sampling.sample_lesions(table, quotas, seed=123)

    assert result_a["sampled_rows"] == result_b["sampled_rows"]
    assert len(result_a["sampled_rows"]) == 10

    by_category = {}
    for row in result_a["sampled_rows"]:
        by_category.setdefault(row["dx"], 0)
        by_category[row["dx"]] += 1
    assert by_category == quotas

    lesion_ids_sampled = [row["lesion_id"] for row in result_a["sampled_rows"]]
    assert len(lesion_ids_sampled) == len(set(lesion_ids_sampled))  # no lesion sampled twice


def test_sample_lesions_different_seed_can_change_selection():
    table = sampling.build_lesion_table(_fabricated_rows())
    quotas = {"nv": 6, "mel": 2, "bcc": 1, "bkl": 1}

    result_a = sampling.sample_lesions(table, quotas, seed=1)
    result_b = sampling.sample_lesions(table, quotas, seed=2)

    assert result_a["sampled_rows"] != result_b["sampled_rows"]


def test_sample_lesions_each_sampled_lesion_gets_exactly_one_of_its_own_images():
    table = sampling.build_lesion_table(_fabricated_rows())
    quotas = {"nv": 6, "mel": 2, "bcc": 1, "bkl": 1}

    result = sampling.sample_lesions(table, quotas, seed=42)

    for row in result["sampled_rows"]:
        assert row["image_id"] in table[row["lesion_id"]]["image_ids"]


def test_sample_lesions_binary_target_derived_correctly():
    table = sampling.build_lesion_table(_fabricated_rows())
    quotas = {"nv": 6, "mel": 2, "bcc": 1, "bkl": 1}

    result = sampling.sample_lesions(table, quotas, seed=42)

    for row in result["sampled_rows"]:
        if row["dx"] == "mel":
            assert row["binary_target"] == "melanoma"
        else:
            assert row["binary_target"] == "nao_melanoma"


def test_sample_lesions_raises_when_quota_exceeds_category_pool():
    table = sampling.build_lesion_table(_fabricated_rows())
    quotas = {"bkl": 999}
    with pytest.raises(ValueError):
        sampling.sample_lesions(table, quotas, seed=1)


def test_compute_quotas_largest_remainder_ties_broken_alphabetically():
    # total=3, target=2: exact quota per category = 1/3*2 = 0.6667 (identical
    # for all three), floors all 0, 2 extra units to distribute among 3
    # tied categories -> alphabetical order must pick "a" and "b", not "c".
    counts = {"a": 1, "b": 1, "c": 1}
    result = sampling.compute_quotas_largest_remainder(counts, target_n=2)
    assert result["final_quotas"] == {"a": 1, "b": 1, "c": 0}
    assert sum(result["final_quotas"].values()) == 2
    extra_recipients = {row["category"] for row in result["calculation_table"] if row["extra_unit"]}
    assert extra_recipients == {"a", "b"}


def test_compute_quotas_handles_category_with_zero_eligible_lesions():
    # "b" has 0 eligible lesions -- must not crash, must get quota 0, and
    # must still appear in the calculation table.
    counts = {"a": 10, "b": 0, "c": 10}
    result = sampling.compute_quotas_largest_remainder(counts, target_n=5)
    assert sum(result["final_quotas"].values()) == 5
    assert result["final_quotas"]["b"] == 0
    categories_in_table = {row["category"] for row in result["calculation_table"]}
    assert categories_in_table == {"a", "b", "c"}
    b_row = next(row for row in result["calculation_table"] if row["category"] == "b")
    assert b_row["n_eligible"] == 0
    assert b_row["exact_quota"] == 0.0
    assert b_row["extra_unit"] is False


def test_compute_quotas_raises_clear_error_when_total_population_is_zero():
    counts = {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="população elegível total é 0"):
        sampling.compute_quotas_largest_remainder(counts, target_n=0)


def test_sample_lesions_category_with_zero_quota_and_zero_pool_is_a_no_op():
    # A category absent from the lesion table entirely (0 pool) with quota 0
    # must not raise and must contribute no rows -- this is the realistic
    # consequence of a category with 0 eligible lesions flowing from
    # compute_quotas_largest_remainder into sample_lesions.
    table = sampling.build_lesion_table(_fabricated_rows())  # has nv/mel/bcc/bkl only
    quotas = {"nv": 2, "akiec": 0}  # "akiec" has zero lesions in this table
    result = sampling.sample_lesions(table, quotas, seed=1)
    assert len(result["sampled_rows"]) == 2
    assert all(row["dx"] == "nv" for row in result["sampled_rows"])


def test_apply_exclusions_then_compute_quotas_raises_when_population_becomes_insufficient():
    # Exercises the exclusion mechanism combined with an insufficient
    # population AFTER exclusion, end to end through the pure functions
    # (the script's own "populacao_insuficiente" branch is a thin wrapper
    # around this same check and is not separately unit-tested -- see
    # report for this limitation).
    table = sampling.build_lesion_table(_fabricated_rows())  # 50 lesions total
    bkl_ids = [lid for lid, v in table.items() if v["dx"] == "bkl"]
    assert len(bkl_ids) == 4  # sanity on the fixture

    exclusion_result = sampling.apply_exclusions(table, bkl_ids)  # remove all 4 bkl lesions
    assert exclusion_result["n_actually_excluded"] == 4
    filtered_table = exclusion_result["filtered_lesion_table"]
    assert len(filtered_table) == 46

    counts_after_exclusion = {}
    for entry in filtered_table.values():
        counts_after_exclusion[entry["dx"]] = counts_after_exclusion.get(entry["dx"], 0) + 1
    assert "bkl" not in counts_after_exclusion  # category fully excluded, not just reduced

    with pytest.raises(ValueError):
        sampling.compute_quotas_largest_remainder(counts_after_exclusion, target_n=47)


def test_apply_exclusions_removing_entire_category_leaves_it_absent_not_zeroed():
    table = sampling.build_lesion_table(_fabricated_rows())
    bkl_ids = [lid for lid, v in table.items() if v["dx"] == "bkl"]
    result = sampling.apply_exclusions(table, bkl_ids)
    remaining_dx_values = {v["dx"] for v in result["filtered_lesion_table"].values()}
    assert "bkl" not in remaining_dx_values
