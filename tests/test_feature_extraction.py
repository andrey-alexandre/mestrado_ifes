import numpy as np
from PIL import Image

from src.agents.agents import _messages_for_condition, build_graph
from src.agents.schemas import GraphState
from src.evaluation.evaluate import compute_metrics
from src.features.asymmetry import block_has_required_lesion_coverage, color_symmetric_from_deltas
from src.features.blue_white_veil import compute_skin_reference, majority_filter
from src.features.border import extract_abcd_border
from src.features.color import classify_abcd_colors
from src.features.extractor import build_feature_package, unavailable
from src.features.pigment_network import extract_pigment_network_raw
from src.features.schemas import EvidenceStatus
from src.features.structureless_blotch import extract_structureless_blotch
from src.features.utils import binary_mask_rle, connected_components


def test_abcd_c_uses_published_five_percent_rule():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    rgb[:] = [255, 255, 255]
    rgb[:5, :5] = [150, 20, 20]

    result = classify_abcd_colors(rgb, mask)

    assert result.status == EvidenceStatus.DETERMINISTIC_CLINICAL
    assert result.evidence["present_per_color"]["white"] is True
    assert result.evidence["present_per_color"]["red"] is True
    assert result.value == 2
    assert "No PH2 calibration." in result.limitations


def test_abcd_c_boundaries_and_strict_presence_rule():
    rgb = np.full((10, 10, 3), [255, 255, 255], dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    rgb[:5, 0] = [150, 0, 0]

    exactly_five_percent = classify_abcd_colors(rgb, mask)
    assert exactly_five_percent.evidence["present_per_color"]["red"] is False

    rgb[5, 0] = [150, 0, 0]
    over_five_percent = classify_abcd_colors(rgb, mask)
    assert over_five_percent.evidence["present_per_color"]["red"] is True

    boundary_pixels = np.array(
        [
            [[204, 204, 204], [150, 50, 1], [62, 0, 1], [0, 100, 125], [61, 61, 61]],
        ],
        dtype=np.uint8,
    )
    boundary_mask = np.ones((1, 5), dtype=bool)
    boundary = classify_abcd_colors(boundary_pixels, boundary_mask)
    assert boundary.evidence["pixel_count_per_color"]["white"] == 1
    assert boundary.evidence["pixel_count_per_color"]["light_brown"] == 1
    assert boundary.evidence["pixel_count_per_color"]["black"] == 1


def test_stoecker_absolute_color_area_filter_and_connectivity():
    rgb = np.full((40, 40, 3), 180, dtype=np.uint8)
    mask = np.ones((40, 40), dtype=bool)
    rgb[:30, :30] = [100, 50, 90]

    result = extract_structureless_blotch(rgb, mask)

    assert result.status == EvidenceStatus.RAW_EVIDENCE
    assert result.evidence["candidate_pixel_count"] == 900
    assert result.evidence["retained_component_pixel_areas"] == [900]
    assert result.evidence["component_filter"] == "A_blotch > 800 pixels"
    assert result.evidence["connected_component_connectivity"] == 8


def test_stoecker_component_rule_is_strictly_greater_than_800():
    rgb = np.full((70, 70, 3), 180, dtype=np.uint8)
    mask = np.ones((70, 70), dtype=bool)
    rgb[0:20, 0:40] = [100, 50, 90]
    rgb[30:57, 0:30] = [100, 50, 90]

    result = extract_structureless_blotch(rgb, mask)

    assert sorted(result.evidence["component_pixel_areas"]) == [800, 810]
    assert result.evidence["retained_component_pixel_areas"] == [810]
    assert result.evidence["total_retained_blotch_area"] == 810


def test_connected_components_default_8_connectivity_links_diagonal_pixels():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)

    assert [len(component) for component in connected_components(mask, connectivity=8)] == [2]
    assert sorted(len(component) for component in connected_components(mask, connectivity=4)) == [1, 1]


def test_binary_mask_rle_preserves_shape_order_and_runs():
    mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=bool)

    rle = binary_mask_rle(mask)

    assert rle == {"shape": [2, 3], "order": "C", "starts": [1, 5], "lengths": [2, 1]}


def test_abcd_a_block_coverage_and_delta_e_majority_rule():
    block = np.zeros((20, 20), dtype=bool)
    block.flat[:296] = True
    assert block_has_required_lesion_coverage(block) is False
    block.flat[:300] = True
    assert block_has_required_lesion_coverage(block) is True
    block.flat[:304] = True
    assert block_has_required_lesion_coverage(block) is True

    assert color_symmetric_from_deltas([1.0, 6.0, 8.0]) is True
    assert color_symmetric_from_deltas([1.0, 8.0, 9.0]) is False
    assert color_symmetric_from_deltas([]) is False


def test_majority_filter_requires_strict_5x5_majority():
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    mask[1, 1] = False

    result = majority_filter(mask)

    assert result[3, 3] is True or bool(result[3, 3]) is True
    assert bool(result[0, 0]) is False


def test_unavailable_null_is_not_absence():
    result = unavailable("TEST", "No frozen method.")

    assert result.status == EvidenceStatus.UNAVAILABLE
    assert result.value is None
    assert "not clinical absence" in result.limitations[0]


def test_metrics_include_precision_and_f1():
    metrics = compute_metrics({
        "a": {"real_value": 1, "validation": {"diagnostico": "Suspeita de Melanoma"}},
        "b": {"real_value": 0, "validation": {"diagnostico": "Suspeita de Melanoma"}},
        "c": {"real_value": 1, "validation": {"diagnostico": "Lesão Benigna"}},
        "d": {"real_value": 0, "validation": {"diagnostico": "Lesão Benigna"}},
    })

    assert metrics["precision"] == 0.5
    assert metrics["f1"] == 0.5


def test_missing_mask_package_keeps_abcd_d_unresolved():
    result = build_feature_package("PH2_001.bmp", None, experiment_condition="image_plus_fe")

    assert result.abcd.dermoscopic_structures_d.status == EvidenceStatus.UNAVAILABLE
    assert result.abcd.dermoscopic_structures_d.value is None
    assert "STILL_UNRESOLVED" in result.abcd.dermoscopic_structures_d.failure_reason


def test_abcd_b_returns_limited_deterministic_score():
    mask = np.ones((20, 20), dtype=bool)

    result = extract_abcd_border(mask)

    assert result.status == EvidenceStatus.DETERMINISTIC_CLINICAL_WITH_LIMITATION
    assert result.value == 0
    assert "No PH2 calibration." in result.limitations


def test_anantha_pigment_network_unavailable_when_classifier_is_not_fully_specified():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)

    result = extract_pigment_network_raw(rgb, mask)

    assert result.status == EvidenceStatus.UNAVAILABLE
    assert result.value is None
    assert "SCIENTIFIC_SPECIFICATION_GAP" in " ".join(result.limitations)


def test_celebi_skin_reference_and_pixel_rule_intermediate_values():
    rgb = np.full((7, 7, 3), [180, 120, 100], dtype=np.uint8)
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    rgb[2:5, 2:5] = [100, 80, 150]

    skin = compute_skin_reference(rgb, mask)
    lesion_pixel = rgb[3, 3].astype(float)
    f3 = lesion_pixel[2] / lesion_pixel.sum()
    f10 = lesion_pixel[0] - skin[0]

    assert skin.tolist() == [180.0, 120.0, 100.0]
    assert round(float(f3), 6) == round(150 / 330, 6)
    assert f10 == -80
    assert f3 > 0.3 and -194 < f10 <= -51


def test_scientific_contract_on_synthetic_feature_package(tmp_path):
    rgb = np.full((80, 80, 3), [200, 130, 110], dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    rgb[20:60, 20:60] = [100, 80, 150]
    rgb[20:50, 20:50] = [100, 50, 90]
    image_path = tmp_path / "PH2_SYN.bmp"
    mask_path = tmp_path / "PH2_SYN_mask.bmp"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(mask).save(mask_path)

    result = build_feature_package(str(image_path), str(mask_path), experiment_condition="image_plus_fe")

    assert result.abcd.dermoscopic_structures_d.status == EvidenceStatus.UNAVAILABLE
    assert result.abcd.dermoscopic_structures_d.value is None
    assert result.seven_point.blue_whitish_veil.status == EvidenceStatus.RAW_EVIDENCE
    assert result.seven_point.blue_whitish_veil.value is None
    assert result.menzies.blue_white_veil.status == EvidenceStatus.RAW_EVIDENCE
    assert result.menzies.blue_white_veil.value is None
    assert result.shared.blue_white_veil.status in {EvidenceStatus.RAW_EVIDENCE, EvidenceStatus.UNAVAILABLE}
    assert result.shared.structureless_blotch.status == EvidenceStatus.RAW_EVIDENCE
    assert "relative-color path is blocked" in " ".join(result.shared.structureless_blotch.limitations)
    assert result.shared.pigment_network.status == EvidenceStatus.UNAVAILABLE
    assert result.shared.pigment_network.value is None


def test_experiment_mode_message_boundaries():
    fe = unavailable("TEST", "synthetic unavailable evidence").model_dump()
    state = GraphState(
        image_path="synthetic.bmp",
        lesion_size=0.0,
        seg_image_data="ZmFrZQ==",
        experiment_condition="image_only",
        feature_extraction={"experiment_condition": "image_only", "abcd": fe, "shared": fe, "metadata": {}},
    )

    image_only = _messages_for_condition(state, "prompt", "abcd")
    assert isinstance(image_only[0].content, list)
    assert "Feature Extraction" not in image_only[0].content[1]["text"]

    state.experiment_condition = "image_plus_fe"
    image_plus_fe = _messages_for_condition(state, "prompt", "abcd")
    assert isinstance(image_plus_fe[0].content, list)
    assert "Feature Extraction" in image_plus_fe[0].content[1]["text"]

    state.experiment_condition = "fe_only"
    fe_only = _messages_for_condition(state, "prompt", "abcd")
    assert isinstance(fe_only[0].content, str)
    assert "Feature Extraction" in fe_only[0].content


def test_graph_initializes_three_experiment_modes_without_api_calls():
    for condition in ["image_only", "image_plus_fe", "fe_only"]:
        graph = build_graph(
            retriever=None,
            model_name="llava:13b",
            segmentation_mode="precomputed",
            experiment_condition=condition,
            enable_critical_reviewer=False,
        )
        assert graph is not None
